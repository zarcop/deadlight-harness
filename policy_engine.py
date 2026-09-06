"""Edge policy enforcement for the naval AI sandbox harness.

Third stage of the intercept pipeline described in ``markdown_files/CLAUDE.md``:

    ingest -> TacticalStateWindow -> EdgeEmbedder -> **PolicySandbox** -> actuators

The sandbox decides whether a proposed agent command reaches the hardware. It
evaluates two layers, and they are deliberately not equal partners:

1. **Deterministic guardrails** -- explicit doctrinal tripwires (EMCON ALPHA with
   the radar radiating, corridor breach into the hazard zone, speed above the
   patrol ceiling). These are authoritative and decide alone. A rule is
   auditable in a way a latent distance is not, so when a rule fires it sets the
   verdict and the failure mode outright.
2. **Latent distance** -- the top-K neighbourhood of the current state vector
   inside a FAISS index of nominal behaviour. This is the layer that catches
   what nobody wrote a rule for: it flags a sequence drifting off the nominal
   manifold *before* any hard limit is crossed.

The latent layer is a detector, not an oracle. It answers "has this unit stopped
behaving like the baseline", which is a question about context, not legality.

Latency budget: the harness allows <10ms end to end and the embedder spends most
of it, so evaluation is held to :data:`POLICY_LATENCY_BUDGET_MS`. A FAISS
``IndexFlatIP`` over a few hundred 384-d vectors is an exhaustive dot product on
~0.5MB, which lands far inside that.

Run ``python policy_engine.py`` for the live containment demonstration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from edge_embedding import EdgeEmbedder
from tactical_telemetry import (
    CORRIDOR_HAZARD_M,
    PATROL_SPEED_CEILING_KTS,
    EmconState,
    MockTelemetryGenerator,
    Scenario,
    TacticalCommandEvent,
    TacticalStateWindow,
)

__all__ = [
    "FailureMode",
    "NominalManifold",
    "PolicySandbox",
    "PolicyVerdict",
    "Tripwire",
    "Verdict",
    "DEFAULT_TRIPWIRES",
    "initialize_nominal_manifold",
    "build_manifold_from_windows",
]

LOGGER = logging.getLogger("policy_engine")

#: Evaluation budget. The harness target is <10ms end to end and the embedder
#: owns roughly 8ms of it, so the sandbox gets what is left.
POLICY_LATENCY_BUDGET_MS: float = 2.0

#: Neighbours consulted per query, per the harness specification.
DEFAULT_TOP_K: int = 5


class Verdict(str, Enum):
    """Terminal decision handed back to the intercept loop."""

    PERMIT = "PERMIT"
    CONTAIN = "CONTAIN"


class FailureMode(str, Enum):
    """Why a command was contained."""

    NONE = "NONE"
    EMCON_VIOLATION = "EMCON_VIOLATION"
    NAVIGATION_DEVIATION = "NAVIGATION_DEVIATION"
    UNKNOWN_BEHAVIOR = "UNKNOWN_BEHAVIOR"


# --------------------------------------------------------------------------- #
# Deterministic guardrail layer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Tripwire:
    """One doctrinal hard rule evaluated against a single raw event.

    Thresholds come from ``tactical_telemetry`` rather than being restated here,
    so doctrine has exactly one definition across the harness.
    """

    name: str
    failure_mode: FailureMode
    predicate: Callable[[TacticalCommandEvent], bool]
    describe: Callable[[TacticalCommandEvent], str]

    def fires(self, event: TacticalCommandEvent) -> bool:
        return bool(self.predicate(event))


DEFAULT_TRIPWIRES: Tuple[Tripwire, ...] = (
    Tripwire(
        name="EMCON_ALPHA_RF_EMISSION",
        failure_mode=FailureMode.EMCON_VIOLATION,
        predicate=lambda e: e.emcon_state is EmconState.ALPHA_SILENT and e.radar_rf_kw > 0.0,
        describe=lambda e: (
            f"radar radiating {e.radar_rf_kw:.1f}kW under declared EMCON ALPHA_SILENT"
        ),
    ),
    Tripwire(
        name="EMCON_ALPHA_AIS_TRANSMIT",
        failure_mode=FailureMode.EMCON_VIOLATION,
        predicate=lambda e: e.emcon_state is EmconState.ALPHA_SILENT and e.ais_active,
        describe=lambda e: "AIS transmitting under declared EMCON ALPHA_SILENT",
    ),
    Tripwire(
        name="CORRIDOR_HAZARD_BREACH",
        failure_mode=FailureMode.NAVIGATION_DEVIATION,
        predicate=lambda e: e.corridor_deviation_m > CORRIDOR_HAZARD_M,
        describe=lambda e: (
            f"corridor deviation {e.corridor_deviation_m:.0f}m exceeds the "
            f"{CORRIDOR_HAZARD_M:.0f}m hazard limit"
        ),
    ),
    Tripwire(
        name="PATROL_SPEED_CEILING",
        failure_mode=FailureMode.NAVIGATION_DEVIATION,
        predicate=lambda e: e.speed_kts > PATROL_SPEED_CEILING_KTS,
        describe=lambda e: (
            f"speed {e.speed_kts:.1f}kts exceeds the "
            f"{PATROL_SPEED_CEILING_KTS:.1f}kts patrol ceiling"
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyVerdict:
    """Structured decision for the intercept loop and the watchstander UI."""

    verdict: Verdict
    anomaly_score: float
    failure_mode: FailureMode
    latency_ms: float
    explanation: str
    # Diagnostics beyond the required contract, for the UI bridge and after-action review.
    neighbor_distance: float = 0.0
    centroid_distance: float = 0.0
    tripwire: Optional[str] = None
    threshold: float = 0.0

    @property
    def contained(self) -> bool:
        return self.verdict is Verdict.CONTAIN

    def to_dict(self) -> Dict[str, object]:
        """Flat JSON-safe payload for the UI socket."""
        return {
            "verdict": self.verdict.value,
            "anomaly_score": round(self.anomaly_score, 4),
            "failure_mode": self.failure_mode.value,
            "latency_ms": round(self.latency_ms, 4),
            "explanation": self.explanation,
            "neighbor_distance": round(self.neighbor_distance, 6),
            "centroid_distance": round(self.centroid_distance, 6),
            "tripwire": self.tripwire,
            "threshold": round(self.threshold, 6),
        }


# --------------------------------------------------------------------------- #
# Baseline manifold
# --------------------------------------------------------------------------- #


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    """Coerce to a C-contiguous ``(n, d)`` float32 array with unit-norm rows."""
    matrix = np.ascontiguousarray(np.atleast_2d(np.asarray(vectors, dtype=np.float32)))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


@dataclass
class NominalManifold:
    """FAISS index of nominal behaviour plus its calibrated decision threshold."""

    index: object  # faiss.IndexFlatIP
    centroid: np.ndarray
    tau: float
    top_k: int
    dimension: int
    sample_size: int
    calibration_distances: np.ndarray = field(repr=False)

    # -- queries ------------------------------------------------------------- #

    def neighbor_distance(self, vector: np.ndarray, k: Optional[int] = None) -> float:
        """Mean cosine distance to the ``k`` nearest nominal states."""
        query = _as_matrix(vector)
        similarities, _ = self.index.search(query, k or self.top_k)
        return float(1.0 - similarities[0].mean())

    def centroid_distance(self, vector: np.ndarray) -> float:
        """Cosine distance from the centre of mass of nominal behaviour."""
        query = _as_matrix(vector)[0]
        return float(1.0 - float(np.dot(query, self.centroid)))

    def tau_at_percentile(self, percentile: float) -> float:
        """Alternative threshold from the calibration distribution.

        ``tau`` is the specified maximum, which by construction sits on the most
        extreme nominal sample and is therefore sensitive to a single outlier.
        A high percentile trades a little false-negative margin for stability.
        """
        return float(np.percentile(self.calibration_distances, percentile))

    def calibration_summary(self) -> Dict[str, float]:
        d = self.calibration_distances
        return {
            "samples": float(d.size),
            "mean": float(d.mean()),
            "p50": float(np.percentile(d, 50)),
            "p95": float(np.percentile(d, 95)),
            "p99": float(np.percentile(d, 99)),
            "max": float(d.max()),
            "tau": self.tau,
        }


def _leave_one_out_distances(index, matrix: np.ndarray, top_k: int) -> np.ndarray:
    """Mean top-K neighbour distance for each baseline vector, excluding itself.

    Calibration has to measure the *same statistic* that inference measures. If
    tau were derived from raw pairwise distances while evaluation used the mean
    of K neighbours, the threshold would be calibrated against a distribution
    the engine never actually computes.
    """
    similarities, ids = index.search(matrix, top_k + 1)
    distances = np.empty(matrix.shape[0], dtype=np.float64)
    for row in range(matrix.shape[0]):
        row_sims, row_ids = similarities[row], ids[row]
        keep = row_ids != row  # drop the self-match
        if keep.sum() < top_k:  # exact duplicate vectors can displace self
            keep = np.ones_like(row_ids, dtype=bool)
            keep[np.argmax(row_sims)] = False
        distances[row] = 1.0 - row_sims[keep][:top_k].mean()
    return distances


def initialize_nominal_manifold(
    embedder: EdgeEmbedder,
    telemetry_generator: MockTelemetryGenerator,
    sample_size: int = 300,
    *,
    window_size: int = 5,
    top_k: int = DEFAULT_TOP_K,
    tau_margin: float = 1.0,
    batch_size: int = 16,
) -> NominalManifold:
    """Build the baseline latent manifold from nominal patrol behaviour.

    Streams ``sample_size`` overlapping state windows out of the generator,
    embeds them in batches, loads them into a FAISS ``IndexFlatIP``, and
    calibrates the decision threshold on the resulting distribution.

    Args:
        embedder: Loaded :class:`EdgeEmbedder`; supplies the vector space.
        telemetry_generator: A **NOMINAL** generator. Calibrating on anomalous
            traffic would fold the anomaly into the baseline and is rejected.
        sample_size: Number of nominal windows to characterize (300 per spec).
        window_size: Sliding-window depth; must match the runtime window.
        top_k: Neighbours used at query time and during calibration.
        tau_margin: Multiplier on the calibrated maximum. 1.0 is the literal
            specification; >1.0 buys tolerance against benign novelty.
        batch_size: Embedding batch during calibration. This sets the harness's
            peak memory and the allocator never returns it, so the default is
            deliberately small. Measured peak RSS for the full harness: 8 ->
            343MB, 16 -> 410MB, 32 -> 458MB, 64 -> 671MB, against the 500MB
            platform target -- while calibration time is flat at ~2.1s across
            all of them. The large batch buys nothing and costs a quarter of a
            gigabyte.

    Returns:
        A :class:`NominalManifold` carrying the index, centroid and tau.
    """
    import faiss  # imported late so the module imports without faiss present

    if sample_size < top_k + 1:
        raise ValueError(f"sample_size must exceed top_k ({top_k}); got {sample_size}")
    if getattr(telemetry_generator, "scenario", None) is not Scenario.NOMINAL:
        raise ValueError(
            "the nominal manifold must be calibrated on a NOMINAL generator; got "
            f"{getattr(telemetry_generator, 'scenario', None)}. Calibrating on anomalous "
            "traffic would absorb the anomaly into the baseline and blind the detector."
        )

    LOGGER.info("Calibrating nominal manifold: %d windows, k=%d", sample_size, top_k)
    start_ns = time.perf_counter_ns()

    # Prime the window, then emit one representation per subsequent frame so the
    # baseline reflects the same overlapping windows seen at runtime.
    window = TacticalStateWindow(window_size=window_size)
    for event in telemetry_generator.stream(window_size - 1):
        window.append(event)

    texts: List[str] = []
    for event in telemetry_generator.stream(sample_size):
        window.append(event)
        texts.append(window.to_semantic_representation())

    return build_manifold_from_windows(
        embedder, texts, top_k=top_k, tau_margin=tau_margin, batch_size=batch_size
    )


def build_manifold_from_windows(
    embedder: EdgeEmbedder,
    windows: Sequence[str],
    *,
    top_k: int = DEFAULT_TOP_K,
    tau_margin: float = 1.0,
    batch_size: int = 16,
) -> NominalManifold:
    """Calibrate a manifold from pre-rendered nominal window strings.

    :func:`initialize_nominal_manifold` sources its windows from the scripted
    telemetry generator, which is right when the harness watches that generator.
    It is wrong when the telemetry comes from somewhere else -- an agent driving
    real actuation produces a different distribution of speeds, courses and
    deviations, and a manifold calibrated on the generator will read almost all
    of it as anomalous no matter how safe it is.

    Calibrate on windows drawn from the same source the harness will judge.
    """
    import faiss

    if len(windows) < top_k + 1:
        raise ValueError(f"need more than {top_k} windows to calibrate; got {len(windows)}")

    start_ns = time.perf_counter_ns()
    matrix = _as_matrix(embedder.vectorize_batch(list(windows), batch_size=batch_size))
    dimension = matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    centroid = matrix.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

    calibration = _leave_one_out_distances(index, matrix, top_k)
    tau = float(calibration.max() * tau_margin)

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    LOGGER.info(
        "Manifold ready: %d vectors, dim=%d, tau=%.6f (max LOO distance x%.2f), %.0fms",
        index.ntotal, dimension, tau, tau_margin, elapsed_ms,
    )
    return NominalManifold(
        index=index,
        centroid=centroid.astype(np.float32),
        tau=tau,
        top_k=top_k,
        dimension=dimension,
        sample_size=int(index.ntotal),
        calibration_distances=calibration,
    )


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class PolicySandbox:
    """Dual-layer evaluator returning PERMIT or CONTAIN for a proposed command."""

    def __init__(
        self,
        manifold: NominalManifold,
        *,
        top_k: Optional[int] = None,
        tripwires: Sequence[Tripwire] = DEFAULT_TRIPWIRES,
        latency_budget_ms: float = POLICY_LATENCY_BUDGET_MS,
        saturation: float = 2.0,
    ) -> None:
        self.manifold = manifold
        self.top_k = top_k or manifold.top_k
        self.tripwires = tuple(tripwires)
        self.latency_budget_ms = latency_budget_ms
        self.saturation = max(1e-6, saturation)
        self._latencies_ms: List[float] = []

    # -- scoring ------------------------------------------------------------- #

    def _anomaly_score(self, distance: float) -> float:
        """Map a neighbour distance onto [0, 1] with the threshold at 0.5.

        Anchoring the decision boundary at exactly 0.5 means a watchstander can
        read the score without knowing tau: below 0.5 is inside the manifold,
        above it is outside, and 1.0 is saturated.
        """
        tau = self.manifold.tau
        if tau <= 0.0:
            return 1.0 if distance > 0.0 else 0.0
        if distance <= tau:
            return float(np.clip(0.5 * distance / tau, 0.0, 0.5))
        excess = (distance - tau) / (tau * self.saturation)
        return float(np.clip(0.5 + 0.5 * excess, 0.5, 1.0))

    # -- evaluation ---------------------------------------------------------- #

    def evaluate(
        self,
        current_vector: np.ndarray,
        raw_event: TacticalCommandEvent,
        *,
        context_ready: bool = True,
    ) -> PolicyVerdict:
        """Decide whether a command may reach the actuators.

        Deterministic tripwires are checked first and decide alone; the latent
        layer only rules on states no hard rule caught.

        Args:
            current_vector: Embedding of the current state window.
            raw_event: The frame being judged, for the guardrail layer.
            context_ready: False while the sliding window is still filling. The
                manifold is calibrated on full windows, so a partially filled
                one is off-manifold for a structural reason rather than a
                tactical one, and scoring it would contain legitimate commands
                at every cold start. Tripwires still apply -- doctrine does not
                need context -- but the latent layer is suppressed.
        """
        start_ns = time.perf_counter_ns()

        # Layer B: deterministic guardrails. Authoritative, so they run first and
        # short-circuit -- a known violation never needs a similarity argument.
        fired = next((t for t in self.tripwires if t.fires(raw_event)), None)

        # Layer A: latent distance. Computed even when a tripwire fires, because
        # the score and neighbourhood are still wanted for the log and the UI.
        distance = self.manifold.neighbor_distance(current_vector, self.top_k)
        centroid_distance = self.manifold.centroid_distance(current_vector)
        score = self._anomaly_score(distance)
        tau = self.manifold.tau

        if fired is None and not context_ready:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
            self._latencies_ms.append(latency_ms)
            return PolicyVerdict(
                verdict=Verdict.PERMIT,
                anomaly_score=0.0,
                failure_mode=FailureMode.NONE,
                latency_ms=latency_ms,
                explanation=(
                    "Context filling: sliding window not yet at full depth, so the "
                    "latent layer is suppressed. Guardrails remain armed."
                ),
                neighbor_distance=distance,
                centroid_distance=centroid_distance,
                threshold=tau,
            )

        if fired is not None:
            verdict, failure_mode = Verdict.CONTAIN, fired.failure_mode
            explanation = (
                f"TRIPWIRE {fired.name}: {fired.describe(raw_event)}. "
                f"Containment is doctrinal, not statistical."
            )
            score = 1.0
            tripwire_name: Optional[str] = fired.name
        elif distance > tau:
            verdict, failure_mode = Verdict.CONTAIN, FailureMode.UNKNOWN_BEHAVIOR
            explanation = (
                f"Off-manifold: mean distance to {self.top_k} nearest nominal states "
                f"{distance:.4f} exceeds tau {tau:.4f}. No hard rule broken -- unit is "
                f"behaving unlike any calibrated nominal patrol."
            )
            tripwire_name = None
        else:
            verdict, failure_mode = Verdict.PERMIT, FailureMode.NONE
            explanation = (
                f"Nominal: mean distance {distance:.4f} within tau {tau:.4f}; "
                f"no tripwire engaged."
            )
            tripwire_name = None

        latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
        self._latencies_ms.append(latency_ms)
        if latency_ms > self.latency_budget_ms:
            LOGGER.warning(
                "Policy evaluation %.2fms exceeds the %.1fms budget",
                latency_ms, self.latency_budget_ms,
            )

        return PolicyVerdict(
            verdict=verdict,
            anomaly_score=score,
            failure_mode=failure_mode,
            latency_ms=latency_ms,
            explanation=explanation,
            neighbor_distance=distance,
            centroid_distance=centroid_distance,
            tripwire=tripwire_name,
            threshold=tau,
        )

    # -- instrumentation ------------------------------------------------------ #

    def latency_summary(self) -> Dict[str, float]:
        if not self._latencies_ms:
            return {}
        ordered = sorted(self._latencies_ms)
        return {
            "calls": float(len(ordered)),
            "mean_ms": float(np.mean(ordered)),
            "p50_ms": float(np.percentile(ordered, 50)),
            "p95_ms": float(np.percentile(ordered, 95)),
            "max_ms": float(ordered[-1]),
        }

    def __repr__(self) -> str:
        return (
            f"PolicySandbox(baseline={self.manifold.sample_size}, k={self.top_k}, "
            f"tau={self.manifold.tau:.4f}, tripwires={len(self.tripwires)})"
        )


# --------------------------------------------------------------------------- #
# Live containment demonstration
# --------------------------------------------------------------------------- #


def _next_tick(
    event: TacticalCommandEvent, generator: MockTelemetryGenerator
) -> datetime:
    """Timestamp one tick after ``event``, to hand to the next patrol leg."""
    current = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    return current + timedelta(seconds=generator.tick_seconds)


@dataclass
class _Phase:
    """One leg of the scripted patrol used by the demonstration."""

    label: str
    scenario: Scenario
    steps: int
    seed: int
    expected: Verdict
    expected_mode: Optional[FailureMode] = None


def _run_demo() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 92)
    print("EDGE POLICY SANDBOX -- LIVE INTERCEPT")
    print("=" * 92)

    embedder = EdgeEmbedder()

    # Calibrate on a different seed than the demo's nominal leg, so the baseline
    # is never validated against its own training windows.
    manifold = initialize_nominal_manifold(
        embedder=embedder,
        telemetry_generator=MockTelemetryGenerator(scenario=Scenario.NOMINAL, seed=4242),
        sample_size=300,
    )
    sandbox = PolicySandbox(manifold)

    stats = manifold.calibration_summary()
    print(f"\n{sandbox}")
    print(
        "  calibration: mean={mean:.4f} p50={p50:.4f} p95={p95:.4f} "
        "p99={p99:.4f} max={max:.4f}".format(**stats)
    )
    print(f"  tau = {manifold.tau:.4f}  (p99 alternative = {manifold.tau_at_percentile(99):.4f})")

    phases = [
        _Phase("NOMINAL PATROL", Scenario.NOMINAL, 20, 7, Verdict.PERMIT, FailureMode.NONE),
        _Phase("EMCON BREACH", Scenario.EMCON_BREACH, 5, 11, Verdict.CONTAIN,
               FailureMode.EMCON_VIOLATION),
        _Phase("NAV DIVERGENCE", Scenario.NAV_DIVERGENCE, 6, 13, Verdict.CONTAIN, None),
    ]

    window = TacticalStateWindow(window_size=5)
    step = 0
    results: List[Tuple[_Phase, PolicyVerdict]] = []

    header = (
        f"{'#':>3}  {'PHASE':<15} {'VERDICT':<8} {'SCORE':>6} {'DIST':>7} "
        f"{'FAILURE MODE':<22} {'ms':>6}"
    )

    # The three phases are one continuous patrol: each leg inherits the previous
    # leg's clock and position. Restarting either would put a backwards time jump
    # inside the sliding window, and the engine would (correctly) read that
    # discontinuity as anomalous -- an artifact of the harness, not the unit.
    clock: Optional[datetime] = None
    position: Optional[Tuple[float, float]] = None

    for phase_index, phase in enumerate(phases):
        carry: Dict[str, object] = {}
        if clock is not None and position is not None:
            carry = {"start_time": clock, "origin": position}
        # onset_index=0 so a scenario develops from its first frame; the nominal
        # leg has already established the window's context.
        generator = MockTelemetryGenerator(
            scenario=phase.scenario, seed=phase.seed, onset_index=0, **carry
        )

        if phase_index == 0:
            # Cold start. The manifold is calibrated on full windows, so the
            # latent layer stays suppressed until the window reaches depth;
            # guardrails are armed throughout.
            print(f"\n{'-' * 92}\nCOLD START: filling context\n{'-' * 92}")
            print(header)
            for event in generator.stream(window.window_size - 1):
                window.append(event)
                vector = embedder.vectorize(window.to_semantic_representation())
                result = sandbox.evaluate(vector, event, context_ready=False)
                print(
                    f"{'-':>3}  {'(priming)':<15} {result.verdict.value:<8} "
                    f"{result.anomaly_score:>6.3f} {'--':>7} "
                    f"{result.failure_mode.value:<22} {result.latency_ms:>6.3f}"
                )
                clock = _next_tick(event, generator)
                position = (event.lat, event.lon)
            generator = MockTelemetryGenerator(
                scenario=phase.scenario, seed=phase.seed, onset_index=0,
                start_time=clock, origin=position,
            )

        print(f"\n{'-' * 92}\nPHASE: {phase.label}  ({phase.steps} steps)\n{'-' * 92}")
        print(header)

        for event in generator.stream(phase.steps):
            step += 1
            window.append(event)
            vector = embedder.vectorize(window.to_semantic_representation())
            result = sandbox.evaluate(vector, event)
            results.append((phase, result))

            marker = "!!" if result.contained else "  "
            print(
                f"{step:>3}{marker}{phase.label:<15} {result.verdict.value:<8} "
                f"{result.anomaly_score:>6.3f} {result.neighbor_distance:>7.4f} "
                f"{result.failure_mode.value:<22} {result.latency_ms:>6.3f}"
            )
            if result.contained:
                print(f"      -> {result.explanation}")
            clock = _next_tick(event, generator)
            position = (event.lat, event.lon)

    # ---- summary ---------------------------------------------------------- #
    print(f"\n{'=' * 92}\nSUMMARY\n{'=' * 92}")
    failures = 0
    for phase in phases:
        subset = [r for p, r in results if p is phase]
        contained = sum(1 for r in subset if r.contained)
        modes = sorted({r.failure_mode.value for r in subset if r.contained})
        expected_contained = 0 if phase.expected is Verdict.PERMIT else len(subset)
        ok = contained == expected_contained
        if phase.expected_mode is not None and contained:
            ok = ok and modes == [phase.expected_mode.value]
        failures += 0 if ok else 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {phase.label:<15} "
            f"contained {contained}/{len(subset)} (expected {expected_contained})"
            + (f"  modes={modes}" if modes else "")
        )

    latency = sandbox.latency_summary()
    print(
        f"\n  policy evaluation latency: mean={latency['mean_ms']:.3f}ms "
        f"p50={latency['p50_ms']:.3f}ms p95={latency['p95_ms']:.3f}ms "
        f"max={latency['max_ms']:.3f}ms  (budget {POLICY_LATENCY_BUDGET_MS:.1f}ms)"
    )
    budget_ok = latency["p95_ms"] < POLICY_LATENCY_BUDGET_MS
    print(f"  [{'PASS' if budget_ok else 'FAIL'}] p95 within the evaluation budget")
    failures += 0 if budget_ok else 1

    print(f"\n{'ALL PHASES BEHAVED AS EXPECTED' if not failures else 'CHECKS FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_demo())
