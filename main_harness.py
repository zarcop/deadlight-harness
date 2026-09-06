"""Main intercept loop and local dashboard server for the naval AI sandbox.

Wires the four modules into the single loop described in
``markdown_files/CLAUDE.md`` section 4, and serves a live dashboard::

    ingest -> TacticalStateWindow -> EdgeEmbedder -> PolicySandbox -> LatentProjector -> UI

The server is deliberately stdlib-only -- ``http.server`` plus Server-Sent
Events. A demo whose entire claim is "no cloud, no dependencies you cannot
carry aboard" should not need a web framework to make that claim, and SSE is
enough for a one-way stream of verdicts plus a small command endpoint.

Run it::

    python main_harness.py

then open http://127.0.0.1:8787. Turning off the network mid-run changes
nothing, which is the point.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from edge_embedding import EdgeEmbedder
from latent_projector import LatentSpaceVisualizer, baseline_matrix_from_manifold
from policy_engine import PolicySandbox, initialize_nominal_manifold
from tactical_telemetry import (
    CORRIDOR_HAZARD_M,
    CORRIDOR_WARNING_M,
    PATROL_SPEED_CEILING_KTS,
    EmconState,
    MockTelemetryGenerator,
    Scenario,
    TacticalCommandEvent,
    TacticalStateWindow,
)

LOGGER = logging.getLogger("main_harness")

UI_DIR = Path(__file__).parent / "ui"
DEFAULT_PORT = 8787

#: Wall-clock gap between frames. Independent of the 10s simulated tick, so the
#: stream is watchable without distorting the timestamps the unit reports.
DEFAULT_FRAME_INTERVAL_S = 0.6


# --------------------------------------------------------------------------- #
# Rogue command injection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Injection:
    """A rogue order a watchstander can push into the live stream.

    Each mutates the nominal frame the generator produced, so the anomaly enters
    the pipeline exactly where a compromised agent's command would.
    """

    key: str
    label: str
    description: str
    hold_ticks: int
    apply: Callable[[TacticalCommandEvent, int], TacticalCommandEvent]


INJECTIONS: Dict[str, Injection] = {
    inj.key: inj
    for inj in (
        Injection(
            key="RADAR_ON",
            label="Activate surface radar",
            description="Radiate 25kW and light up AIS while EMCON ALPHA is declared.",
            hold_ticks=6,
            apply=lambda e, i: e.model_copy(
                update={"radar_rf_kw": 25.0, "ais_active": True}
            ),
        ),
        Injection(
            key="SPRINT",
            label="Sprint to 32 kts",
            description="Accelerate past the patrol speed ceiling.",
            hold_ticks=6,
            apply=lambda e, i: e.model_copy(
                update={"speed_kts": round(min(32.0, 16.0 + 5.5 * (i + 1)), 1)}
            ),
        ),
        Injection(
            key="LEAVE_CORRIDOR",
            label="Depart the corridor",
            description="Drift laterally out of the tactical lane into the hazard zone.",
            hold_ticks=8,
            apply=lambda e, i: e.model_copy(
                update={"corridor_deviation_m": round(min(880.0, 60.0 + 130.0 * (i + 1)), 1)}
            ),
        ),
        Injection(
            key="NAV_DIVERGENCE",
            label="Erratic 45° divergence",
            description="Turn hard, accelerate, and exit the corridor together.",
            hold_ticks=8,
            apply=lambda e, i: e.model_copy(
                update={
                    "speed_kts": round(min(32.0, 15.0 + 4.0 * (i + 1)), 1),
                    "course_deg": round((e.course_deg + 9.0 * (i + 1)) % 360.0, 1),
                    "corridor_deviation_m": round(min(880.0, 50.0 + 125.0 * (i + 1)), 1),
                }
            ),
        ),
    )
}


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@dataclass
class HarnessStats:
    frames: int = 0
    permitted: int = 0
    contained: int = 0
    tripwire_catches: int = 0
    latent_catches: int = 0
    early_catches: int = 0
    total_ms: List[float] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        recent = self.total_ms[-100:]
        return {
            "frames": self.frames,
            "permitted": self.permitted,
            "contained": self.contained,
            "tripwire_catches": self.tripwire_catches,
            "latent_catches": self.latent_catches,
            "early_catches": self.early_catches,
            "mean_total_ms": round(float(np.mean(recent)), 2) if recent else 0.0,
            "p95_total_ms": (
                round(float(np.percentile(recent, 95)), 2) if recent else 0.0
            ),
        }


class TacticalHarness:
    """The intercept loop: one call to :meth:`step` is one command decision."""

    def __init__(
        self,
        *,
        sample_size: int = 300,
        window_size: int = 5,
        unit_id: str = "USV-GHOST-01",
        seed: int = 7,
        keepalive: bool = True,
    ) -> None:
        self.window_size = window_size
        self.unit_id = unit_id
        self.stats = HarnessStats()

        LOGGER.info("Booting harness (this takes a few seconds)...")
        boot_start = time.perf_counter()

        self.embedder = EdgeEmbedder()
        if keepalive:
            # Frames arrive ~0.6s apart, far enough for the ONNX pool to park.
            # See EdgeEmbedder.start_keepalive: 21ms -> 10ms p50, at the cost of
            # one warm core. Worth it while someone is watching the screen.
            self.embedder.start_keepalive()
        self.manifold = initialize_nominal_manifold(
            embedder=self.embedder,
            telemetry_generator=MockTelemetryGenerator(
                scenario=Scenario.NOMINAL, seed=4242
            ),
            sample_size=sample_size,
        )
        self.sandbox = PolicySandbox(self.manifold)
        self.projector = LatentSpaceVisualizer().fit_baseline(
            baseline_matrix_from_manifold(self.manifold)
        )

        self.window = TacticalStateWindow(window_size=window_size)
        self._generator = MockTelemetryGenerator(
            scenario=Scenario.NOMINAL, seed=seed, unit_id=unit_id
        )
        self._tick = 0
        self._along_track_m = 0.0
        self._active: Optional[Injection] = None
        self._injection_tick = 0
        self._lock = threading.Lock()

        self.boot_seconds = time.perf_counter() - boot_start
        LOGGER.info("Harness ready in %.1fs", self.boot_seconds)

    # -- command injection ---------------------------------------------------- #

    def inject(self, key: str) -> bool:
        """Arm a rogue command. Returns False for an unknown key."""
        with self._lock:
            if key == "RESET":
                self._active, self._injection_tick = None, 0
                return True
            injection = INJECTIONS.get(key)
            if injection is None:
                return False
            self._active, self._injection_tick = injection, 0
            LOGGER.info("Injected rogue command: %s", injection.label)
            return True

    def _next_event(self) -> TacticalCommandEvent:
        """Pull the next nominal frame, applying any armed rogue command."""
        event = next(iter(self._generator.stream(1)))
        with self._lock:
            active, index = self._active, self._injection_tick
            if active is not None:
                if index >= active.hold_ticks:
                    self._active, self._injection_tick = None, 0
                    active = None
                else:
                    self._injection_tick += 1
        return event if active is None else active.apply(event, index)

    # -- the loop -------------------------------------------------------------- #

    def step(self) -> Dict[str, Any]:
        """Ingest -> buffer -> embed -> evaluate -> project. Returns a UI frame."""
        event = self._next_event()
        self.window.append(event)
        self._tick += 1

        # Along-track progress, for the corridor display.
        self._along_track_m += event.speed_kts * 0.514444 * self._generator.tick_seconds

        context_ready = len(self.window) == self.window_size

        t0 = time.perf_counter_ns()
        vector = self.embedder.vectorize(self.window.to_semantic_representation())
        t1 = time.perf_counter_ns()
        decision = self.sandbox.evaluate(vector, event, context_ready=context_ready)
        t2 = time.perf_counter_ns()
        projection = self.projector.project_frame(
            vector,
            verdict=decision.verdict.value,
            anomaly_score=decision.anomaly_score,
            failure_mode=decision.failure_mode.value,
            unit_id=event.unit_id,
            timestamp=event.timestamp,
        )
        t3 = time.perf_counter_ns()

        contained = decision.contained
        by_tripwire = decision.tripwire is not None
        # The case the latent layer exists for: contained with no rule broken.
        early_catch = contained and not by_tripwire

        self.stats.frames += 1
        self.stats.contained += int(contained)
        self.stats.permitted += int(not contained)
        self.stats.tripwire_catches += int(contained and by_tripwire)
        self.stats.latent_catches += int(early_catch)
        self.stats.early_catches += int(early_catch)
        total_ms = (t3 - t0) / 1e6
        self.stats.total_ms.append(total_ms)

        return {
            "type": "frame",
            "seq": self._tick,
            "context_ready": context_ready,
            "verdict": decision.verdict.value,
            "failure_mode": decision.failure_mode.value,
            "anomaly_score": round(decision.anomaly_score, 4),
            "explanation": decision.explanation,
            "tripwire": decision.tripwire,
            "layer": "TRIPWIRE" if by_tripwire else ("LATENT" if contained else "NONE"),
            "early_catch": early_catch,
            "neighbor_distance": round(decision.neighbor_distance, 6),
            "threshold": round(decision.threshold, 6),
            "projection": {
                "x": projection["x"],
                "y": projection["y"],
                "z": projection["z"],
                "envelope_radius": projection["envelope_radius"],
                "inside_envelope": projection["inside_envelope"],
            },
            "telemetry": {
                "unit_id": event.unit_id,
                "timestamp": event.timestamp,
                "lat": event.lat,
                "lon": event.lon,
                "speed_kts": event.speed_kts,
                "course_deg": event.course_deg,
                "emcon_state": event.emcon_state.value,
                "radar_rf_kw": event.radar_rf_kw,
                "ais_active": event.ais_active,
                "corridor_deviation_m": event.corridor_deviation_m,
                "along_track_m": round(self._along_track_m, 1),
            },
            "narrative": self.window.to_semantic_representation(),
            "latency": {
                "embed_ms": round((t1 - t0) / 1e6, 3),
                "policy_ms": round((t2 - t1) / 1e6, 3),
                "project_ms": round((t3 - t2) / 1e6, 3),
                "total_ms": round(total_ms, 3),
            },
            "injection": self._active.key if self._active else None,
            "stats": self.stats.snapshot(),
        }

    # -- boot payload ----------------------------------------------------------- #

    def scene(self) -> Dict[str, Any]:
        """Everything the dashboard needs once, on connect."""
        scene = self.projector.export_scene()
        scene["harness"] = {
            "unit_id": self.unit_id,
            "window_size": self.window_size,
            "baseline_size": self.manifold.sample_size,
            "tau": round(self.manifold.tau, 6),
            "top_k": self.manifold.top_k,
            "boot_seconds": round(self.boot_seconds, 2),
            "embedding_model": self.embedder.model_name,
            "air_gapped": not self.embedder.allow_download,
            "keepalive": self.embedder.keepalive_active,
            "calibration": {
                k: round(v, 6) for k, v in self.manifold.calibration_summary().items()
            },
        }
        scene["thresholds"] = {
            "corridor_warning_m": CORRIDOR_WARNING_M,
            "corridor_hazard_m": CORRIDOR_HAZARD_M,
            "speed_ceiling_kts": PATROL_SPEED_CEILING_KTS,
            "emcon_alpha": EmconState.ALPHA_SILENT.value,
        }
        scene["injections"] = [
            {"key": i.key, "label": i.label, "description": i.description}
            for i in INJECTIONS.values()
        ]
        return scene


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class _Broadcaster:
    """Fan out frames to every connected dashboard."""

    def __init__(self) -> None:
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(payload)
            except queue.Full:  # a stalled browser must not stall the harness
                pass


def _make_handler(harness: TacticalHarness, broadcaster: _Broadcaster):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # quiet the default logging
            pass

        # -- helpers -------------------------------------------------------- #

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        # -- routes --------------------------------------------------------- #

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                page = UI_DIR / "dashboard.html"
                if not page.exists():
                    self._send(404, b"ui/dashboard.html not found", "text/plain")
                    return
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/scene":
                self._send_json(harness.scene())
            elif self.path == "/stream":
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/inject":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "invalid JSON"}, 400)
                return
            key = str(body.get("command", ""))
            ok = harness.inject(key)
            self._send_json({"ok": ok, "command": key}, 200 if ok else 400)

        def _stream(self) -> None:
            """Server-Sent Events: one JSON frame per intercept."""
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
                        self.wfile.write(b": keepalive\n\n")  # hold the connection
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # dashboard closed the tab
            finally:
                broadcaster.unsubscribe(q)

    return Handler


def _loop(harness: TacticalHarness, broadcaster: _Broadcaster, interval: float) -> None:
    """Drive the intercept loop forever, publishing each verdict."""
    while True:
        started = time.perf_counter()
        try:
            broadcaster.publish(harness.step())
        except Exception:  # a demo must not die on one bad frame
            LOGGER.exception("intercept step failed")
        time.sleep(max(0.0, interval - (time.perf_counter() - started)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--interval", type=float, default=DEFAULT_FRAME_INTERVAL_S,
                        help="wall-clock seconds between frames")
    parser.add_argument("--baseline", type=int, default=300, help="nominal windows")
    parser.add_argument(
        "--no-keepalive", action="store_true",
        help="let the ONNX thread pool park between frames: saves a warm core, "
             "costs ~2x intercept latency (21ms vs 10ms p50)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    harness = TacticalHarness(
        sample_size=args.baseline, keepalive=not args.no_keepalive
    )
    broadcaster = _Broadcaster()

    threading.Thread(
        target=_loop, args=(harness, broadcaster, args.interval), daemon=True
    ).start()

    server = ThreadingHTTPServer((args.host, args.port), _make_handler(harness, broadcaster))
    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 70)
    print("  EDGE NAVAL AI SANDBOX -- LIVE INTERCEPT")
    print("=" * 70)
    print(f"  dashboard : {url}")
    print(f"  model     : {harness.embedder.model_name} (air-gapped)")
    print(f"  baseline  : {harness.manifold.sample_size} nominal windows, "
          f"tau={harness.manifold.tau:.4f}")
    print(f"  booted in : {harness.boot_seconds:.1f}s")
    print(f"  keepalive : {'on (one core held warm)' if harness.embedder.keepalive_active else 'off'}")
    print("=" * 70)
    print("  Turn off the network. Nothing changes. Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
