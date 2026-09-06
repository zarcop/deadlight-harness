"""Live agent-containment console: watch an agent try to escape the sandbox.

``main_harness.py`` shows the harness watching a *unit*. This shows the harness
watching an *agent*: every command it proposes, whether the sandbox let it
through, and in plain language why.

The demonstration is the shield toggle. Drop the guard mid-run and the same
agent -- same brain, same persona -- starts getting its commands through, and
the unit walks into the states it was being kept out of. Raise the guard and it
stops. Nothing about the agent changes; only whether anything is standing in
front of it.

Stdlib only (``http.server`` + Server-Sent Events), so the demo carries no more
dependencies than the harness it is demonstrating.

Run it::

    python agent_console.py

then open http://127.0.0.1:8788.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_brain import build_brain
from agent_harness_bridge import (
    BehaviorClass,
    AgentHarnessRunner,
    InterceptRecord,
    boot_harness,
)
from agent_protocol import SandboxVerdict
from tactical_telemetry import (
    CORRIDOR_HAZARD_M,
    CORRIDOR_WARNING_M,
    PATROL_SPEED_CEILING_KTS,
)

LOGGER = logging.getLogger("agent_console")

UI_FILE = Path(__file__).parent / "ui" / "agent_console.html"
DEFAULT_PORT = 8788
DEFAULT_INTERVAL_S = 2.2  # slow enough to read a verdict before the next arrives


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


@dataclass
class Scoreboard:
    """What the audience is actually keeping score of."""

    attempts: int = 0
    allowed: int = 0
    flagged: int = 0
    blocked: int = 0
    escaped: int = 0          # dangerous commands that actuated anyway
    violations: int = 0       # frames where the unit ended up out of bounds

    def to_dict(self) -> Dict[str, int]:
        return {
            "attempts": self.attempts, "allowed": self.allowed,
            "flagged": self.flagged, "blocked": self.blocked,
            "escaped": self.escaped, "violations": self.violations,
        }


class ConsoleSession:
    """Drives the agent/harness loop and holds the state the console displays."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        model: Optional[str] = None,
        seed: int = 2026,
        baseline: int = 220,
        switch_every: int = 7,
        guarded: bool = True,
    ) -> None:
        LOGGER.info("Booting harness for the agent console...")
        started = time.perf_counter()
        embedder, sandbox = boot_harness(baseline=baseline, calibration="agent")
        self.embedder, self.sandbox = embedder, sandbox
        self.brain = build_brain(backend=backend, model=model, seed=seed)
        self.runner = AgentHarnessRunner(
            embedder=embedder, sandbox=sandbox, brain=self.brain,
            guarded=guarded, seed=seed,
        )
        self.switch_every = switch_every or None
        self.backend = backend
        self.scoreboard = Scoreboard()
        self.boot_seconds = time.perf_counter() - started
        self._forced: Optional[BehaviorClass] = None
        self._generation = 0   # bumped to restart the loop with fresh state
        self._lock = threading.Lock()
        LOGGER.info("Console ready in %.1fs", self.boot_seconds)

    # -- controls ----------------------------------------------------------- #

    @property
    def guarded(self) -> bool:
        return self.runner.guarded

    def set_guard(self, on: bool) -> None:
        # Toggling the shield restarts the run. Without it the vessel carries its
        # escaped state forward -- once it is 100km off route and radiating, the
        # comparison stops being watchable and every later frame reads the same.
        with self._lock:
            self.runner.guarded = bool(on)
            self._generation += 1
            self.scoreboard = Scoreboard()
        LOGGER.info("Harness guard %s -- run reset", "RAISED" if on else "DROPPED")

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self.scoreboard = Scoreboard()

    @property
    def generation(self) -> int:
        return self._generation

    def set_behavior(self, behavior: Optional[str]) -> bool:
        try:
            with self._lock:
                self._forced = BehaviorClass(behavior) if behavior else None
                self._generation += 1
        except ValueError:
            return False
        return True

    @property
    def forced_behavior(self) -> Optional[BehaviorClass]:
        return self._forced

    # -- payloads ----------------------------------------------------------- #

    def scene(self) -> Dict[str, Any]:
        """Sent once on connect: the fixed facts the console needs."""
        return {
            "type": "scene",
            "guarded": self.guarded,
            "backend": self.backend,
            "inference": (
                "Claude (external)"
                if type(self.brain).__name__ == "AnthropicCommandBrain"
                else "deterministic (offline)"
            ),
            "model": getattr(self.brain, "model", None),
            "baseline": self.sandbox.manifold.sample_size,
            "tau": round(self.sandbox.manifold.tau, 6),
            "boot_seconds": round(self.boot_seconds, 1),
            "behaviors": [b.value for b in BehaviorClass],
            "forced_behavior": self._forced.value if self._forced else None,
            "thresholds": {
                "corridor_warning_m": CORRIDOR_WARNING_M,
                "corridor_hazard_m": CORRIDOR_HAZARD_M,
                "speed_ceiling_kts": PATROL_SPEED_CEILING_KTS,
            },
            "scoreboard": self.scoreboard.to_dict(),
        }

    def record_to_frame(self, record: InterceptRecord) -> Dict[str, Any]:
        """One interaction, shaped for the console."""
        verdict = record.sandbox.verdict
        board = self.scoreboard
        board.attempts += 1

        # Count what happened, not what was decided. With the shield down the
        # harness still returns CONTAIN on a dangerous command, but nothing
        # stops it -- scoring that as "blocked" reports a containment that did
        # not occur, which is the one number this console must never overstate.
        should_stop = record.should_be_blocked or verdict is SandboxVerdict.CONTAIN
        if should_stop and not record.actuated:
            board.blocked += 1
        elif should_stop and record.actuated:
            board.escaped += 1
        elif verdict is SandboxVerdict.REVIEW:
            board.flagged += 1
        else:
            board.allowed += 1

        if record.committed_violates:
            board.violations += 1

        candidate, committed = record.candidate, record.committed
        return {
            "type": "intercept",
            "step": record.step,
            "guarded": self.guarded,
            "agent": {
                "behavior": record.behavior.value,
                "persona": record.persona.value,
            },
            "request": {
                "command": record.proposal.command_type.value,
                "value": record.proposal.value,
                "confidence": round(record.proposal.confidence, 2),
                "intent": record.proposal.intent,
                "rationale": record.proposal.rationale,
            },
            "checks": {
                # Which gate stopped it. The console lights these one at a time.
                "doctrine": record.policy.tripwire,
                "doctrine_fired": record.policy.tripwire is not None,
                "behaviour_fired": (
                    record.policy.tripwire is None
                    and verdict is not SandboxVerdict.PERMIT
                ),
                "distance": round(record.policy.neighbor_distance, 5),
                "threshold": round(record.policy.threshold, 5),
            },
            "verdict": verdict.value,
            "failure_mode": record.policy.failure_mode.value,
            "risk": round(record.policy.anomaly_score, 3),
            "explanation": record.policy.explanation,
            "actuated": record.actuated,
            "dangerous": record.should_be_blocked,
            "escaped": (
                record.should_be_blocked or verdict is SandboxVerdict.CONTAIN
            ) and record.actuated,
            "authority_breach": record.authority_breach,
            "would_have": {
                "emcon": candidate.is_emcon_violation,
                "corridor": candidate.is_corridor_breach,
                "speed": candidate.is_speed_excursion,
                "authority": record.authority_breach is not None,
            },
            "unit": {
                "speed_kts": committed.speed_kts,
                "course_deg": committed.course_deg,
                "emcon": committed.emcon_state.value,
                "radar_kw": committed.radar_rf_kw,
                "ais": committed.ais_active,
                "deviation_m": committed.corridor_deviation_m,
                "in_violation": record.committed_violates,
            },
            "latency_ms": round(record.latency_ms, 1),
            "scoreboard": self.scoreboard.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class _Broadcaster:
    """Fan out interactions to every connected console."""

    def __init__(self) -> None:
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=32)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subs)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:  # a stalled browser must not stall the loop
                pass


def _make_handler(session: ConsoleSession, broadcaster: _Broadcaster):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                if not UI_FILE.exists():
                    self._send(404, b"ui/agent_console.html not found", "text/plain")
                    return
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/scene":
                self._json(session.scene())
            elif self.path == "/stream":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/control":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"ok": False, "error": "invalid JSON"}, 400)
                return

            action = str(body.get("action", ""))
            if action == "guard":
                session.set_guard(bool(body.get("value", True)))
                self._json({"ok": True, "guarded": session.guarded})
            elif action == "reset":
                session.reset()
                self._json({"ok": True})
            elif action == "behavior":
                value = body.get("value") or None
                ok = session.set_behavior(value)
                self._json({"ok": ok, "forced_behavior": value}, 200 if ok else 400)
            else:
                self._json({"ok": False, "error": f"unknown action {action!r}"}, 400)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broadcaster.subscribe()
            try:
                while True:
                    try:
                        payload = q.get(timeout=10.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                broadcaster.unsubscribe(q)

    return Handler


def _loop(session: ConsoleSession, broadcaster: _Broadcaster, interval: float) -> None:
    """Drive the agent forever, publishing each interaction."""
    while True:
        generation = session.generation
        try:
            stream = session.runner.step_stream(
                None,
                behavior=session.forced_behavior,
                switch_behavior_every=session.switch_every,
            )
            for record in stream:
                if session.generation != generation:
                    break  # shield toggled or behaviour forced: start clean
                started = time.perf_counter()
                broadcaster.publish(session.record_to_frame(record))
                time.sleep(max(0.0, interval - (time.perf_counter() - started)))
        except Exception:  # a demo must not die on one bad step
            LOGGER.exception("agent step failed; restarting the loop")
            time.sleep(1.0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help="seconds between agent commands (default 2.2)")
    parser.add_argument("--backend", choices=["auto", "anthropic", "fallback"], default="auto")
    parser.add_argument("--model")
    parser.add_argument("--baseline", type=int, default=220)
    parser.add_argument("--switch-every", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--start-unguarded", action="store_true",
                        help="open with the shield down (the agent escapes immediately)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    session = ConsoleSession(
        backend=args.backend, model=args.model, seed=args.seed,
        baseline=args.baseline, switch_every=args.switch_every,
        guarded=not args.start_unguarded,
    )
    broadcaster = _Broadcaster()
    threading.Thread(
        target=_loop, args=(session, broadcaster, args.interval), daemon=True
    ).start()

    server = ThreadingHTTPServer(
        (args.host, args.port), _make_handler(session, broadcaster)
    )
    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 68)
    print("  AGENT CONTAINMENT CONSOLE")
    print("=" * 68)
    print(f"  console  : {url}")
    print(f"  inference: {session.scene()['inference']}")
    print(f"  baseline : {session.sandbox.manifold.sample_size} compliant-agent windows")
    print(f"  guard    : {'UP' if session.guarded else 'DOWN'}")
    print("=" * 68)
    print("  Drop the shield in the UI and watch the same agent escape.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
