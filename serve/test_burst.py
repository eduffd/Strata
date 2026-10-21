"""The conversation burst without an engine: the lane predicate, the config boundary, and the turn broker.

    python -m pytest serve/test_burst.py -q
"""
import threading
import time
import unittest

from serve.burst import (BURST_MAX, DEFAULT_GRACE_MS, GRACE_MAX_MS, TURN_ID, BurstPolicy, ImageKey, ImageMark,
                         Lane, TurnBroker, checkpoint_at, image_marks, read_burst)


def lane(tokens, images=None, leaves=True):
    return Lane(tuple(tokens), images, leaves)


A1 = lane((1, 2, TURN_ID, 10, 11))
A2 = lane((1, 2, TURN_ID, 10, 11, 12))
A3 = lane((1, 2, TURN_ID, 10, 11, 12, 13))
B1 = lane((3, 4, TURN_ID, 20))


class Predicate(unittest.TestCase):
    def test_continuation_matches_a_later_turn(self):
        cont = A1.continuation()
        self.assertEqual(cont.prefix, (1, 2))
        self.assertTrue(cont.matches(A2))
        self.assertTrue(cont.matches(A3))
        self.assertTrue(cont.matches(A1))       # a retry is the same conversation: it starts with the checkpoint
        self.assertTrue(cont.matches(lane((1, 2, TURN_ID, 10, 99))))    # shares the checkpoint, another tail

    def test_shorter_or_rewritten_prompt_is_a_different_lane(self):
        cont = A1.continuation()
        self.assertFalse(cont.matches(lane((1,))))                        # shorter than the prefix
        self.assertFalse(cont.matches(lane((1, 2))))                      # not longer than the prefix
        self.assertFalse(cont.matches(lane((1, 9, TURN_ID, 10, 11, 12)))) # changed below it
        self.assertFalse(cont.matches(B1))

    def test_a_different_image_key_below_the_boundary_is_a_different_lane(self):
        prompt = (1, 2, TURN_ID, 10)
        cont = Lane(prompt, (ImageMark(1, ImageKey("aa")),)).continuation()
        self.assertEqual(cont.images, (ImageKey("aa"),))
        self.assertTrue(cont.matches(Lane(prompt + (11,), (ImageMark(1, ImageKey("aa")),))))
        self.assertFalse(cont.matches(Lane(prompt + (11,), (ImageMark(1, ImageKey("bb")),))))
        # a key above the boundary does not matter
        self.assertTrue(cont.matches(Lane(prompt + (11,), (ImageMark(1, ImageKey("aa")),
                                                              ImageMark(5, ImageKey("cc"))))))

    def test_unknown_image_keys_compare_tokens_only(self):
        cont = Lane((1, 2, TURN_ID, 10), None).continuation()
        self.assertTrue(cont.matches(Lane((1, 2, TURN_ID, 10, 11), (ImageMark(1, ImageKey("aa")),))))
        cont = Lane((1, 2, TURN_ID, 10), (ImageMark(1, ImageKey("aa")),)).continuation()
        self.assertTrue(cont.matches(Lane((1, 2, TURN_ID, 10, 11), None)))

    def test_one_shot_request_leaves_no_continuation(self):
        self.assertIsNone(lane((1, 2, TURN_ID, 10), leaves=False).continuation())
        self.assertIsNone(lane((1, 2, 3)).continuation())          # no turn header
        self.assertIsNone(lane((TURN_ID, 1, 2)).continuation())    # the checkpoint is at index 0

    def test_image_marks_follow_the_prompt(self):
        start, pad = 259, 260
        ids = [1, start, pad, pad, 5, start, pad, 7]
        marks = image_marks(ids, ["aa", "bb"], pad, start)
        self.assertEqual(marks, (ImageMark(2, ImageKey("aa")), ImageMark(6, ImageKey("bb"))))
        self.assertIsNone(image_marks(ids, ["aa"], pad, start))    # a count mismatch: compare tokens only
        self.assertIsNone(image_marks(ids, [], pad, start))        # pictures in the prompt, no keys
        self.assertEqual(image_marks([1, 2], [], pad, start), ())
        self.assertIsNone(image_marks(ids, ["aa", "bb"], None, start))
        self.assertEqual(checkpoint_at(ids), 0)


class Config(unittest.TestCase):
    def test_read_burst_off_by_default(self):
        off = read_burst(None, None)
        self.assertIsNone(off.policy)
        self.assertEqual(off.notes, ())
        self.assertIsNone(read_burst(0, None).policy)
        one = read_burst(1, None)
        self.assertIsNone(one.policy)
        self.assertEqual(len(one.notes), 1)
        self.assertIn("no conversation affinity", one.notes[0])
        on = read_burst(2, None)
        self.assertEqual((on.policy.requests, on.policy.grace_s), (2, DEFAULT_GRACE_MS / 1000.0))

    def test_read_burst_clamps_and_explains(self):
        big = read_burst(5000, None)
        self.assertEqual(big.policy.requests, BURST_MAX)
        self.assertEqual(len(big.notes), 1)
        slow = read_burst(4, 9000)
        self.assertAlmostEqual(slow.policy.grace_s, GRACE_MAX_MS / 1000.0)
        self.assertEqual(len(slow.notes), 1)
        lonely = read_burst(None, 100)
        self.assertIsNone(lonely.policy)
        self.assertEqual(len(lonely.notes), 1)
        self.assertIn("has no effect", lonely.notes[0])
        for bad in ("4", 4.0, True, -1):
            with self.assertRaises(ValueError):
                read_burst(bad, None)
        for bad in ("100", 1.5, True, -1):
            with self.assertRaises(ValueError):
                read_burst(None, bad)


class Broker(unittest.TestCase):
    def take(self, b, which, timeout=1.0):
        t = b.join(which)
        self.assertTrue(b.wait(t, timeout))
        return t

    def serve(self, b, which):
        self.take(b, which)
        b.release()

    def test_broker_serves_a_waiting_continuation_first(self):
        b = TurnBroker(BurstPolicy(3, 0.0))
        self.take(b, A1)
        t_a2 = b.join(A2)                  # the continuation was already waiting
        t_b = b.join(B1)                   # a different lane, later
        b.release()
        self.assertTrue(b.wait(t_a2, 0))
        b.release()
        self.assertTrue(b.wait(t_b, 0))
        b.release()
        self.assertFalse(b.locked())

    def test_broker_cap_sends_a_different_lane_first(self):
        b = TurnBroker(BurstPolicy(2, 0.0))
        self.take(b, A1)
        t_a2, t_a3, t_b = b.join(A2), b.join(A3), b.join(B1)
        b.release()
        self.assertTrue(b.wait(t_a2, 0))           # the burst's second turn
        b.release()
        self.assertTrue(b.wait(t_b, 0))            # the cap: another lane goes
        b.release()
        self.assertTrue(b.wait(t_a3, 0))
        b.release()

    def test_broker_grace_hit(self):
        b = TurnBroker(BurstPolicy(3, 0.5))
        self.serve(b, A1)                          # the grace opens
        t_b = b.join(B1)
        time.sleep(0.02)
        t_a = b.join(A2)                           # inside the grace
        self.assertTrue(b.wait(t_a, 0))
        b.release()
        self.assertTrue(b.wait(t_b, 1.0))
        b.release()

    def test_broker_grace_miss(self):
        b = TurnBroker(BurstPolicy(3, 0.05))
        self.serve(b, A1)
        t_b = b.join(B1)
        time.sleep(0.1)
        t_a = b.join(A2)                           # after the grace: the waiting lane goes first
        self.assertTrue(b.wait(t_b, 0))
        self.assertFalse(b.wait(t_a, 0))
        b.release()
        self.assertTrue(b.wait(t_a, 1.0))          # after B1's own grace
        b.release()

    def test_broker_a_late_continuation_does_not_jump(self):
        b = TurnBroker(BurstPolicy(3, 0.05))
        self.serve(b, A1)
        t_b = b.join(B1)
        time.sleep(0.08)
        t_a = b.join(A2)
        self.assertTrue(b.wait(t_b, 0))            # the waiting lane owns the turn
        self.assertFalse(b.wait(t_a, 0))
        b.release()

    def test_broker_nonblocking_control_takes_a_free_turn_during_grace(self):
        b = TurnBroker(BurstPolicy(3, 0.5))
        self.serve(b, A1)
        self.assertFalse(b.locked())               # the grace holds no owner
        self.assertTrue(b.acquire(blocking=False))
        self.assertTrue(b.locked())
        b.release()

    def test_broker_control_never_waits_for_the_grace(self):
        b = TurnBroker(BurstPolicy(3, 0.5))
        self.serve(b, A1)
        t_b = b.join(B1)
        started = time.monotonic()
        self.assertTrue(b.acquire(timeout=0.2))    # granted at once, before the grace closes
        self.assertLess(time.monotonic() - started, 0.2)
        b.release()
        self.assertTrue(b.wait(t_b, 1.0))
        b.release()

    def test_broker_control_queued_before_a_continuation_is_served_first(self):
        b = TurnBroker(BurstPolicy(3, 0.0))
        self.take(b, A1)
        got, release_me = threading.Event(), threading.Event()

        def control():
            if b.acquire(timeout=5):
                got.set()
                release_me.wait(5)
                b.release()

        th = threading.Thread(target=control, daemon=True)
        th.start()
        time.sleep(0.05)                           # the control is queued first
        t_a = b.join(A2)
        b.release()                                # A1's turn ends
        self.assertTrue(got.wait(2))               # the control goes before the continuation
        self.assertFalse(b.wait(t_a, 0))
        release_me.set()
        th.join(2)
        self.assertTrue(b.wait(t_a, 0))
        b.release()

    def test_broker_cancel_never_serves(self):
        b = TurnBroker(BurstPolicy(3, 0.0))
        self.take(b, A1)
        t_a2 = b.join(A2)
        t_b = b.join(B1)
        b.leave(t_a2)
        b.leave(t_a2)                              # already gone: harmless
        b.release()
        self.assertTrue(b.wait(t_b, 0))            # the other lane goes
        b.release()
        self.assertFalse(b.wait(t_a2, 0))          # the cancelled request never gets the turn
        self.assertFalse(b.locked())

        # an owned leave is not a served turn: the burst count and the continuation stay
        b = TurnBroker(BurstPolicy(2, 0.0))
        self.take(b, A1)
        t_a2 = b.join(A2)
        b.release()
        self.assertTrue(b.wait(t_a2, 0))
        b.leave(t_a2)                              # abandoned while owned
        self.assertTrue(b.acquire(timeout=1))      # hold the turn so both can queue
        t_a3 = b.join(A3)
        t_b = b.join(B1)
        b.release()
        self.assertTrue(b.wait(t_a3, 0))           # not counted: the cap was not reached, A3 waited first
        b.release()
        self.assertTrue(b.wait(t_b, 0))
        b.release()

    def test_broker_lock_surface(self):
        b = TurnBroker(BurstPolicy(2, 0.1))
        self.assertFalse(b.locked())
        self.assertTrue(b.acquire(blocking=False))
        self.assertTrue(b.locked())
        self.assertFalse(b.acquire(blocking=False))
        with self.assertRaises(ValueError):
            b.acquire(blocking=False, timeout=0)
        with self.assertRaises(RuntimeError):
            b.release()
            b.release()
        with b:
            self.assertTrue(b.locked())
            self.assertFalse(b.acquire(timeout=0.05))
        self.assertFalse(b.locked())

        self.assertTrue(b.acquire())               # released from another thread
        th = threading.Thread(target=b.release)
        th.start()
        th.join(2)
        self.assertFalse(b.locked())

        self.assertTrue(b.acquire())               # a timed acquire that the release wakes
        late = threading.Thread(target=lambda: (time.sleep(0.1), b.release()))
        late.start()
        self.assertTrue(b.acquire(timeout=1))
        late.join(2)
        b.release()


if __name__ == "__main__":
    unittest.main()
