"""The conversation burst through Service: order, grace, cap, cancel, deadline, control, keep-alive, parallel, off.

    python -m pytest serve/test_conversation_burst.py -q
"""
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import serve.server as server
from serve import burst
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, QueueWaitExpired, Service

ROOT = Path(__file__).resolve().parents[1]
T = burst.TURN_ID

A1 = [1, 2, T, 10, 11]
A2 = A1 + [12]
A3 = A2 + [13]
B1 = [3, 4, T, 20]


class RecordingEngine(MockEngine):
    """The mock engine, remembering the prompt of every generate() in the order the turn served them."""

    def __init__(self, tok, script="ok", delay_s=0.0):
        super().__init__(tok, script, delay_s=delay_s)
        self.seen = []
        self.seen_lock = threading.Lock()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with self.seen_lock:
            self.seen.append(tuple(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Burst(unittest.TestCase):
    def make(self, requests=None, grace_ms=None, delay_s=0.0, beat=None):
        tok = ByteTokenizer()
        self.engine = RecordingEngine(tok, delay_s=delay_s)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if requests is not None:
            self.svc.enable_burst(burst.read_burst(requests, grace_ms).policy)
        if beat is not None:
            p = mock.patch.object(server, "QUEUE_BEAT_S", beat)
            p.start()
            self.addCleanup(p.stop)
        return self.svc

    def start(self, ids, cancel=None):
        cancel = cancel or threading.Event()
        events = []

        def go():
            try:
                for ev in self.svc.run(list(ids), False, None, 8, {}, cancel):
                    events.append(ev)
            except Exception as e:                 # QueueWaitExpired reaches the caller of run()
                events.append(("error", e))

        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th, events, cancel

    def until(self, cond, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return
            time.sleep(0.005)
        self.fail("condition not met")

    def start_waiting(self, ids, cancel=None):
        """Start a request and wait for its keep-alive: it has joined the broker and is waiting for the turn."""
        th, events, cancel = self.start(ids, cancel)
        self.until(lambda: ("ping", None) in events)
        return th, events, cancel

    def test_off_is_the_plain_lock(self):
        svc = self.make()
        self.assertIsNone(svc.burst)
        self.assertIsInstance(svc.fifo, type(threading.Lock()))
        self.assertNotIsInstance(svc.fifo, burst.TurnBroker)
        th1, _, _ = self.start(A1)
        th2, _, _ = self.start(B1)
        th1.join(5)
        th2.join(5)
        self.assertEqual(sorted(self.engine.seen), sorted([tuple(A1), tuple(B1)]))

    def test_burst_order_with_four_lanes(self):
        svc = self.make(3, 0, beat=0.05)
        svc.fifo.acquire()                         # hold the turn with a control
        ths = [self.start_waiting(p) for p in (A1, A2, A3, B1)]
        svc.fifo.release()
        for th, _, _ in ths:
            th.join(5)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(A2), tuple(A3), tuple(B1)])

    def test_grace_hit_end_to_end(self):
        self.make(3, 1000, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(5)
        th_b, _, _ = self.start_waiting(B1)
        th_a, _, _ = self.start(A2)                # inside A1's grace
        th_a.join(5)
        th_b.join(5)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(A2), tuple(B1)])

    def test_grace_miss_end_to_end(self):
        self.make(3, 100, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(5)
        th_b, _, _ = self.start_waiting(B1)
        time.sleep(0.3)                            # past the grace
        th_a, _, _ = self.start(A2)
        th_b.join(5)
        th_a.join(5)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(B1), tuple(A2)])

    def test_a_reply_is_not_delayed_by_the_grace(self):
        self.make(3, 5000, delay_s=0.01)
        started = time.monotonic()
        th, _, _ = self.start(A1)
        th.join(10)
        self.assertLess(time.monotonic() - started, 2.0)   # the 5 s grace is paid by a waiter, never by the reply
        self.assertEqual(self.engine.seen, [tuple(A1)])

    def test_keepalive_during_the_grace(self):
        self.make(3, 1000, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(10)                               # the reply opens a 1 s grace
        th_b, ev_b, _ = self.start(B1)             # a different lane waits it out
        th_b.join(10)
        self.assertGreaterEqual(ev_b.count(("ping", None)), 1, ev_b)

    def test_cap_end_to_end(self):
        svc = self.make(2, 0, beat=0.05)
        svc.fifo.acquire()
        ths = [self.start_waiting(p) for p in (A1, A2, A3, B1)]
        svc.fifo.release()
        for th, _, _ in ths:
            th.join(5)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(A2), tuple(B1), tuple(A3)])

    def test_cancel_while_waiting(self):
        svc = self.make(3, 0, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(5)
        svc.fifo.acquire()
        cancel = threading.Event()
        th_a, ev_a, _ = self.start_waiting(A2, cancel)     # A1's next turn
        th_b, _, _ = self.start_waiting(B1)
        cancel.set()
        th_a.join(5)
        svc.fifo.release()
        th_b.join(5)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(B1)])
        self.assertEqual(ev_a[-1][1]["finish"], "cancel")
        self.assertEqual(svc.status["queued"], 0)

    def test_deadline_while_waiting(self):
        svc = self.make(3, 0, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(5)
        svc.queue_timeout_s = 0.3
        svc.fifo.acquire()
        th_a, ev_a, _ = self.start_waiting(A2)     # joined: its deadline is fixed
        svc.queue_timeout_s = None                 # the other lane is not on a deadline
        th_b, _, _ = self.start_waiting(B1)
        th_a.join(5)
        svc.fifo.release()
        th_b.join(5)
        errors = [value for kind, value in ev_a if kind == "error"]
        self.assertTrue(errors and isinstance(errors[0], QueueWaitExpired), errors)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(B1)])
        self.assertEqual(svc.status["queued"], 0)

    def test_control_while_bursting(self):
        svc = self.make(3, 0, delay_s=0.05)
        th1, _, _ = self.start(A1)
        th1.join(10)
        th_a, _, _ = self.start(A2)                # the continuation runs
        self.until(lambda: len(self.engine.seen) >= 2)
        got, release_me = threading.Event(), threading.Event()

        def control():
            if svc.fifo.acquire(timeout=5):
                got.set()
                release_me.wait(5)
                svc.fifo.release()

        th_c = threading.Thread(target=control, daemon=True)
        th_c.start()
        time.sleep(0.02)
        th_b, _, _ = self.start(A3)
        self.assertTrue(got.wait(5))               # the control completed during the burst
        self.assertEqual(len(self.engine.seen), 2) # the later continuation has not run
        release_me.set()
        th_c.join(5)
        th_a.join(10)
        th_b.join(10)
        self.assertEqual(self.engine.seen, [tuple(A1), tuple(A2), tuple(A3)])

    def test_keepalive_during_a_burst(self):
        self.make(3, 0, delay_s=0.2, beat=0.05)
        th1, _, _ = self.start(A1)
        th1.join(10)
        th_a, _, _ = self.start(A2)                # runs for about 0.6 s
        self.until(lambda: len(self.engine.seen) >= 2)
        th_b, ev_b, _ = self.start(B1)             # a different lane, waiting
        th_b.join(10)
        th_a.join(10)
        self.assertGreaterEqual(ev_b.count(("ping", None)), 1, ev_b)

    def test_parallel_note_and_no_broker(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.requested_slots = 4
        out = io.StringIO()
        with redirect_stdout(out):
            svc.enable_burst(burst.BurstPolicy(4, 0.1))
        self.assertIsNone(svc.burst)
        self.assertNotIsInstance(svc.fifo, burst.TurnBroker)
        self.assertIn("no effect with parallel requests", out.getvalue())


class ConfigBoundary(unittest.TestCase):
    """main()'s flags and config keys: refused at the boundary, then installed on the service before serving."""

    class Stop(Exception):
        """Stands in for serve(): main() has finished its setup when it arrives."""

    def main(self, argv, out=None, err=None):
        seen = {}

        def fake_serve(svc, host="127.0.0.1", port=8095):
            seen["svc"] = svc
            raise self.Stop()

        with mock.patch.object(server.lifeline, "install_stop_handlers"), \
                mock.patch.object(server, "serve", fake_serve), \
                redirect_stdout(out or io.StringIO()), redirect_stderr(err or io.StringIO()):
            with self.assertRaises(self.Stop):
                server.main(argv)
        return seen["svc"]

    def refuse(self, argv):
        err = io.StringIO()
        with mock.patch.object(server.lifeline, "install_stop_handlers"), redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                server.main(argv)
        self.assertEqual(cm.exception.code, 2)
        return err.getvalue()

    def config(self, values):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = Path(d.name) / "run.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        return str(path)

    def test_bad_burst_flag_is_refused_at_the_boundary(self):
        self.assertIn("conversation_burst must be 0 or more",
                      self.refuse(["--engine", "mock", "--conversation-burst", "-1"]))

    def test_bad_burst_config_key_is_refused_at_the_boundary(self):
        path = self.config({"conversation_burst": "4"})
        self.assertIn("conversation_burst must be a whole number",
                      self.refuse(["--engine", "mock", "--config", path]))

    def test_flags_install_the_broker_before_serving(self):
        out = io.StringIO()
        svc = self.main(["--engine", "mock", "--port", "0",
                         "--conversation-burst", "4", "--conversation-grace-ms", "50"], out=out)
        self.assertIsInstance(svc.fifo, burst.TurnBroker)
        self.assertEqual((svc.burst.requests, svc.burst.grace_s), (4, 0.05))
        self.assertIn("up to 4 turns of one conversation in a row, waiting 50 ms", out.getvalue())

    def test_config_keys_install_the_broker_before_serving(self):
        path = self.config({"conversation_burst": 2, "conversation_grace_ms": 0})
        svc = self.main(["--engine", "mock", "--port", "0", "--config", path])
        self.assertIsInstance(svc.fifo, burst.TurnBroker)
        self.assertEqual((svc.burst.requests, svc.burst.grace_s), (2, 0.0))

    def test_grace_without_burst_is_said_and_ignored(self):
        out = io.StringIO()
        svc = self.main(["--engine", "mock", "--port", "0", "--conversation-grace-ms", "50"], out=out)
        self.assertIsNone(svc.burst)
        self.assertNotIsInstance(svc.fifo, burst.TurnBroker)
        self.assertIn("has no effect without conversation_burst", out.getvalue())


if __name__ == "__main__":
    unittest.main()
