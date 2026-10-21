"""The opt-in conversation burst for the single serial turn.

Off (the default), `Service.fifo` is a `threading.Lock` and only the checkpoint rule here is used.  On, `Service.fifo` is a
`TurnBroker`: a request joins with a `Lane` (its prompt and the pictures below the turn header), the broker
remembers the `Continuation` the last served conversation accepts, lets that conversation's next turn go
ahead of another conversation while the burst has room, and records a grace deadline when a turn ends.  The
next request to decide pays the grace, never the finishing request.

Lane identity is prompt continuation, as the engine's own `starts_with` rule: no session id, header or cookie.
A request continues the last served one when its prompt is longer and agrees up to the last `<|im_start|>`,
with the same pictures below that point.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import NewType, Sequence

TURN_ID = 248045                    # <|im_start|>, the engine's default --turn-token (its turn checkpoint sits at the last one)
BURST_MAX = 1000                    # requests
GRACE_MAX_MS = 5000
DEFAULT_GRACE_MS = 100

ImageKey = NewType("ImageKey", str)


def checkpoint_at(ids: Sequence[int]) -> int:
    """The index of the last TURN_ID in `ids`, or 0 when there is none."""
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == TURN_ID:
            return i
    return 0


@dataclass(frozen=True, slots=True)
class ImageMark:
    """One picture: the index of its first <|image_pad|>, and its 16-hex key."""
    at: int
    key: ImageKey


@dataclass(frozen=True, slots=True)
class Lane:
    """One request's conversation identity.  Built once, before the request waits for the turn."""
    prompt: tuple[int, ...]
    images: tuple[ImageMark, ...] | None = None   # None: keys unknown -> compare tokens only
    leaves_checkpoint: bool = True                # False: sampling["strata_checkpoint"] is False (#830)

    def continuation(self) -> "Continuation | None":
        """What the next request of this conversation must start with, or None."""
        if not self.leaves_checkpoint:
            return None
        k = checkpoint_at(self.prompt)
        if k <= 0:
            return None
        return Continuation(self.prompt[:k],
                            None if self.images is None else images_below(self.images, k))


@dataclass(frozen=True, slots=True)
class Continuation:
    """The engine's starts_with rule (#2518), derived once from the served Lane."""
    prefix: tuple[int, ...]
    images: tuple[ImageKey, ...] | None

    def matches(self, lane: Lane) -> bool:
        k = len(self.prefix)
        if len(lane.prompt) <= k or lane.prompt[:k] != self.prefix:
            return False
        if self.images is None or lane.images is None:
            return True                            # pictures unknown: a false positive only misses a reuse
        return images_below(lane.images, k) == self.images


def images_below(images: Sequence[ImageMark], k: int) -> tuple[ImageKey, ...]:
    return tuple(m.key for m in images if m.at < k)


def image_marks(ids: Sequence[int], keys: Sequence[str], pad_id: int | None,
                start_id: int | None) -> tuple[ImageMark, ...] | None:
    """The pictures in prompt order at their first pad.  None when the keys are absent or do not match."""
    if not keys:
        return None if pad_id is not None and pad_id in ids else ()
    if pad_id is None or start_id is None:
        return None
    marks, k = [], 0
    for j in range(1, len(ids)):
        if ids[j] == pad_id and ids[j - 1] == start_id:
            if k < len(keys):
                marks.append(ImageMark(j, ImageKey(keys[k])))
            k += 1
    return tuple(marks) if k == len(keys) else None


@dataclass(frozen=True, slots=True)
class BurstPolicy:
    requests: int        # 2..BURST_MAX
    grace_s: float       # 0..GRACE_MAX_MS / 1000

    def __post_init__(self) -> None:
        if not 2 <= self.requests <= BURST_MAX:
            raise ValueError(f"conversation_burst must be 2..{BURST_MAX}, not {self.requests}")
        if not 0.0 <= self.grace_s <= GRACE_MAX_MS / 1000.0:
            raise ValueError(f"conversation_grace_ms must be 0..{GRACE_MAX_MS}, not {self.grace_s * 1000:g}")


@dataclass(frozen=True, slots=True)
class BurstChoice:
    policy: BurstPolicy | None
    notes: tuple[str, ...]


def read_burst(burst: object, grace_ms: object) -> BurstChoice:
    """The two values from the config or the flags, validated and clamped.  Off for absent/0/1."""
    notes: list[str] = []
    n = 0
    if burst is not None:
        if isinstance(burst, bool) or not isinstance(burst, int):
            raise ValueError(f"conversation_burst must be a whole number of requests (0 turns it off), not {burst!r}")
        if burst < 0:
            raise ValueError("conversation_burst must be 0 or more (0 turns it off)")
        if burst > BURST_MAX:
            notes.append(f"conversation_burst {burst} is more than needed; using {BURST_MAX}.")
            burst = BURST_MAX
        if burst == 1:
            notes.append("conversation_burst 1 means no conversation affinity (the same as off).")
        n = burst
    grace = DEFAULT_GRACE_MS if grace_ms is None else grace_ms
    if isinstance(grace, bool) or not isinstance(grace, int):
        raise ValueError(f"conversation_grace_ms must be a whole number of milliseconds (0 = no wait), not {grace!r}")
    if grace < 0:
        raise ValueError("conversation_grace_ms must be 0 or more")
    if grace > GRACE_MAX_MS:
        notes.append(f"conversation_grace_ms {grace} is more than needed; using {GRACE_MAX_MS}.")
        grace = GRACE_MAX_MS
    if n < 2:
        if grace_ms is not None:
            notes.append("conversation_grace_ms has no effect without conversation_burst (2 or more).")
        return BurstChoice(None, tuple(notes))
    return BurstChoice(BurstPolicy(n, grace / 1000.0), tuple(notes))


class Ticket:
    """One waiter's identity across repeated wait() calls.  Identity equality only."""
    __slots__ = ("lane", "seq", "continuation", "continues")
    lane: Lane | None
    seq: int
    continuation: Continuation | None
    continues: bool

    def __init__(self, lane: Lane | None, seq: int, continuation: Continuation | None) -> None:
        self.lane, self.seq, self.continuation, self.continues = lane, seq, continuation, False


class TurnBroker:
    """The single turn when the burst is on: a threading.Lock for every existing caller, plus a ticket surface
    for inference requests.  All state is under one condition variable; `release` only records the hand-off and
    the next joining, waiting, leaving or acquiring thread decides who goes next."""

    def __init__(self, policy: BurstPolicy) -> None:
        self._policy = policy
        self._cond = threading.Condition()
        self._waiters: collections.deque[Ticket] = collections.deque()
        self._owner: Ticket | None = None
        self._next_seq = 0
        self._continuation: Continuation | None = None
        self._burst_served = 0
        self._cap_reached = False
        self._grace_until = 0.0

    # --- inference surface: Service._run only

    def join(self, lane: Lane) -> Ticket:
        """Register this request's lane and return its ticket.  May grant the turn at once."""
        with self._cond:
            t = Ticket(lane, self._next_seq, lane.continuation())
            self._next_seq += 1
            self._waiters.append(t)
            self._admit(time.monotonic())
            self._cond.notify_all()
            return t

    def wait(self, ticket: Ticket, timeout: float) -> bool:
        """Block until this ticket owns the turn (True) or `timeout` seconds pass (False).  A timeout keeps the
        ticket queued: the caller yields a keep-alive ping and calls wait() again."""
        end = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        with self._cond:
            while True:
                now = time.monotonic()
                self._admit(now)
                if self._owner is ticket:
                    return True
                if end is not None and now >= end:
                    return False
                wake = end
                if self._owner is None and self._grace_until > now and (wake is None or self._grace_until < wake):
                    wake = self._grace_until
                self._cond.wait(None if wake is None else max(0.0, wake - now))

    def leave(self, ticket: Ticket) -> None:
        """Give up: withdraw a waiting ticket, or abandon an owned turn without counting it as served."""
        with self._cond:
            if self._owner is ticket:
                self._owner = None               # abandoned: not a served turn, no burst/grace update
                self._admit(time.monotonic())
            elif ticket in self._waiters:        # identity, not equality
                self._waiters.remove(ticket)
                self._admit(time.monotonic())
            self._cond.notify_all()

    # --- threading.Lock surface

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking and timeout not in (-1, None):
            raise ValueError("can't specify a timeout for a non-blocking call")
        with self._cond:
            if not blocking:
                if self._owner is None:
                    self._owner = Ticket(None, self._next_seq, None)
                    self._next_seq += 1
                    return True
                return False
            t = Ticket(None, self._next_seq, None)
            self._next_seq += 1
            self._waiters.append(t)
            self._admit(time.monotonic())
            end = None if timeout is None or timeout < 0 else time.monotonic() + timeout
            while self._owner is not t:
                now = time.monotonic()
                self._admit(now)
                if self._owner is t:
                    break
                if end is not None and now >= end:
                    self._waiters.remove(t)
                    self._admit(now)
                    self._cond.notify_all()
                    return False
                self._cond.wait(None if end is None else max(0.0, end - now))
            return True

    def release(self) -> None:
        with self._cond:
            t = self._owner
            if t is None:
                raise RuntimeError("release unlocked turn")
            self._owner = None
            if t.lane is not None:
                self._burst_served = self._burst_served + 1 if t.continues else 1
                self._continuation = t.continuation
                self._cap_reached = self._burst_served >= self._policy.requests
                if self._cap_reached:
                    self._burst_served = 0
                    self._grace_until = 0.0
                else:
                    self._grace_until = (time.monotonic() + self._policy.grace_s
                                         if self._continuation is not None and self._policy.grace_s > 0 else 0.0)
            # a control release changes no lane state and does not close the grace
            self._cond.notify_all()

    def locked(self) -> bool:
        with self._cond:
            return self._owner is not None

    def __enter__(self) -> bool:
        self.acquire()
        return True

    def __exit__(self, *exc: object) -> None:
        self.release()

    # --- who goes next

    def _admit(self, now: float) -> None:
        if self._owner is not None or not self._waiters:
            return
        control = next((w for w in self._waiters if w.lane is None), None)
        if control is not None:                  # control first: never held by the grace or the cap
            self._grant(control)
            return
        cont = self._continuation
        if self._cap_reached:
            for w in self._waiters:
                if cont is None or not cont.matches(w.lane):
                    self._grant(w)
                    return
            self._grant(self._waiters[0])        # only continuations wait: the same lane may continue
            return
        if cont is not None:
            same = next((w for w in self._waiters if cont.matches(w.lane)), None)
            if same is not None:
                other = next((w for w in self._waiters if w is not same), None)
                if other is None or same.seq < other.seq or now <= self._grace_until:
                    self._grant(same)
                    return
        if now < self._grace_until:
            return
        self._grant(self._waiters[0])

    def _grant(self, ticket: Ticket) -> None:
        ticket.continues = (ticket.lane is not None and self._continuation is not None
                            and self._continuation.matches(ticket.lane))
        self._waiters.remove(ticket)
        self._owner = ticket
        if ticket.lane is not None:
            self._grace_until = 0.0              # the turn is held; nothing waits on the grace
        self._cond.notify_all()
