"""Latent-space projection bridge for the naval AI sandbox dashboard.

Final stage of the intercept pipeline in ``markdown_files/CLAUDE.md``:

    ingest -> TacticalStateWindow -> EdgeEmbedder -> PolicySandbox -> **LatentProjector** -> UI

Reduces a 384-d state vector to ``(x, y, z)`` so a Three.js or Plotly dashboard
can show the unit moving relative to the green nominal manifold.

**The projection is a display, not a detector.** PCA fit on a tight nominal
cloud keeps only the directions along which *nominal* behaviour varies, and an
anomaly need not be large along any of them. Containment is decided by
:mod:`policy_engine` in the full 384-d space; what a watchstander sees here is a
faithful rendering of three coordinates, not of the decision. Screen distance and
anomaly score are different quantities and the module reports both so they are
never confused.

Stability is the other hard requirement. The basis is fitted exactly once, at
startup, on the baseline manifold. Refitting mid-stream would rotate the axes
under the operator and make a stationary unit appear to move, so
:meth:`LatentSpaceVisualizer.fit_baseline` refuses to run twice and
:meth:`project_point` only ever calls ``transform``. The fitted basis can be
serialized so the same coordinate frame survives a process restart.

Run ``python latent_projector.py`` for the end-to-end demonstration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BoundingEnvelope",
    "LatentSpaceVisualizer",
    "ProjectionError",
    "ProjectionFrozenError",
    "baseline_matrix_from_manifold",
]

LOGGER = logging.getLogger("latent_projector")

#: Bumped when the wire payload changes shape, so the frontend can refuse a
#: mismatched backend instead of rendering garbage.
SCHEMA_VERSION: str = "1.0"

#: Coordinates are rounded before they go on the wire.
COORD_PRECISION: int = 4


class ProjectionError(RuntimeError):
    """Raised when a projection is requested before the basis is fitted."""


class ProjectionFrozenError(RuntimeError):
    """Raised on any attempt to refit a basis that is already established."""


# --------------------------------------------------------------------------- #
# Bounding envelope
# --------------------------------------------------------------------------- #


@dataclass
class BoundingEnvelope:
    """The safe volume, in projected space, that the UI paints green.

    Carries two representations because they answer different questions:

    * an **ellipsoid** (centre + covariance) for a cheap, smooth inside/outside
      test via Mahalanobis radius, stable against the baseline's outliers;
    * a **convex hull** (vertices + triangles) because Three.js needs an actual
      mesh to draw, not a statistical description.
    """

    center: np.ndarray
    covariance: np.ndarray
    inverse_covariance: np.ndarray = field(repr=False)
    radius: float
    percentile: float
    axis_ranges: Dict[str, Tuple[float, float]]
    hull_vertices: List[List[float]] = field(default_factory=list, repr=False)
    hull_faces: List[List[int]] = field(default_factory=list, repr=False)

    def mahalanobis(self, point: np.ndarray) -> float:
        """Ellipsoidal radius of a projected point; 1.0 is the envelope skin."""
        delta = np.asarray(point, dtype=np.float64) - self.center
        distance = float(np.sqrt(max(0.0, delta @ self.inverse_covariance @ delta)))
        return distance / self.radius if self.radius > 0 else distance

    def contains(self, point: np.ndarray) -> bool:
        return self.mahalanobis(point) <= 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Renderable description of the safe volume."""
        return {
            # The ellipsoid is a containment region, not a hard bound: by
            # construction ~(100 - percentile)% of nominal points sit outside it.
            # The convex hull is the strict bound over the baseline.
            "kind": "containment_ellipsoid",
            "percentile": self.percentile,
            "center": {
                axis: round(float(value), COORD_PRECISION)
                for axis, value in zip("xyz", self.center)
            },
            "radius": round(self.radius, 6),
            "axis_ranges": {
                axis: [round(lo, COORD_PRECISION), round(hi, COORD_PRECISION)]
                for axis, (lo, hi) in self.axis_ranges.items()
            },
            "hull": {"vertices": self.hull_vertices, "faces": self.hull_faces},
        }


# --------------------------------------------------------------------------- #
# Projector
# --------------------------------------------------------------------------- #


def baseline_matrix_from_manifold(manifold: Any) -> np.ndarray:
    """Recover the baseline vectors from a :class:`NominalManifold`.

    ``IndexFlatIP`` stores its vectors verbatim, so the projector can reuse the
    exact cloud the policy engine calibrated on instead of embedding it twice.
    """
    index = manifold.index
    return np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)


class LatentSpaceVisualizer:
    """Fit-once PCA projection from embedding space to dashboard coordinates.

    Args:
        n_components: Output dimensionality; 3 for an ``(x, y, z)`` scene.
        normalize: Rescale projected coordinates by a factor captured at fit
            time so the baseline lands near unit scale. The factor is frozen
            with the basis, so this never introduces per-frame drift and the
            camera never needs to re-frame.
        envelope_percentile: Percentile of baseline Mahalanobis radii treated as
            the envelope skin. Below 100 the envelope ignores the most extreme
            baseline points, which keeps one outlier from inflating the green
            volume until everything looks safe.
        random_state: Seeds the SVD solver for a reproducible basis.
    """

    def __init__(
        self,
        n_components: int = 3,
        *,
        normalize: bool = True,
        envelope_percentile: float = 95.0,
        random_state: int = 0,
    ) -> None:
        if n_components != 3:
            LOGGER.warning(
                "n_components=%d: the dashboard schema assumes 3 (x, y, z)", n_components
            )
        self.n_components = n_components
        self.normalize = normalize
        self.envelope_percentile = envelope_percentile
        self.random_state = random_state

        self._pca: Optional[Any] = None
        self._scale: float = 1.0
        self._frozen: bool = False
        self._baseline_points: Optional[np.ndarray] = None
        self.envelope: Optional[BoundingEnvelope] = None
        self.input_dimension: Optional[int] = None

    # -- state --------------------------------------------------------------- #

    @property
    def is_fitted(self) -> bool:
        return self._pca is not None

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        return None if self._pca is None else self._pca.explained_variance_ratio_

    def _require_fitted(self) -> None:
        if self._pca is None:
            raise ProjectionError(
                "projection basis is not fitted; call fit_baseline() once at startup"
            )

    # -- fitting ------------------------------------------------------------- #

    def fit_baseline(
        self, baseline_vectors: np.ndarray, *, force: bool = False
    ) -> "LatentSpaceVisualizer":
        """Establish the coordinate frame from the nominal manifold. **Once.**

        Args:
            baseline_vectors: ``(n, d)`` nominal embeddings -- the same cloud the
                policy engine calibrated on.
            force: Deliberately re-establish the frame. This invalidates every
                coordinate previously sent to the UI, so the frontend must
                discard its scene and re-fetch. Never call this on a live stream.

        Raises:
            ProjectionFrozenError: if the basis is already fitted and ``force``
                is not set.
        """
        if self._frozen and not force:
            raise ProjectionFrozenError(
                "the projection basis is already fitted and is frozen for the life of "
                "the process. Refitting mid-stream rotates the axes underneath the "
                "watchstander, so a stationary unit appears to move and historical "
                "frames stop being comparable. Pass force=True only during a "
                "deliberate re-initialization, and re-send the scene to the UI."
            )

        from sklearn.decomposition import PCA  # imported late; heavy dependency

        matrix = np.ascontiguousarray(np.atleast_2d(baseline_vectors), dtype=np.float64)
        if matrix.shape[0] <= self.n_components:
            raise ValueError(
                f"need more than {self.n_components} baseline vectors to fit "
                f"{self.n_components} components; got {matrix.shape[0]}"
            )

        start_ns = time.perf_counter_ns()
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        projected = pca.fit_transform(matrix)

        # Canonical component signs. PCA is sign-indeterminate, so an otherwise
        # identical refit can mirror the scene. Forcing the largest-magnitude
        # loading of each component positive makes the frame reproducible across
        # sklearn versions and persisted models.
        flips = np.sign(pca.components_[np.arange(self.n_components),
                                        np.argmax(np.abs(pca.components_), axis=1)])
        flips[flips == 0] = 1.0
        pca.components_ *= flips[:, None]
        projected *= flips

        # Freeze the display scale with the basis so it can never drift per frame.
        self._scale = 1.0
        if self.normalize:
            spread = float(np.percentile(np.linalg.norm(projected, axis=1), 95))
            self._scale = 1.0 / spread if spread > 1e-12 else 1.0
            projected = projected * self._scale

        self._pca = pca
        self._frozen = True
        self.input_dimension = int(matrix.shape[1])
        self._baseline_points = projected
        self.envelope = self._build_envelope(projected)

        variance = float(pca.explained_variance_ratio_.sum())
        LOGGER.info(
            "Projection basis frozen: %d vectors, %d-d -> %d-d, "
            "explained variance %.1f%% (%s), %.0fms",
            matrix.shape[0], self.input_dimension, self.n_components, variance * 100.0,
            ", ".join(f"{v * 100:.1f}%" for v in pca.explained_variance_ratio_),
            (time.perf_counter_ns() - start_ns) / 1e6,
        )
        if variance < 0.5:
            LOGGER.warning(
                "The 3-d view retains only %.1f%% of baseline variance. It is a "
                "situational display, not a decision surface -- on-screen distance "
                "does not equal the anomaly score.",
                variance * 100.0,
            )
        return self

    def _build_envelope(self, projected: np.ndarray) -> BoundingEnvelope:
        """Ellipsoid plus convex hull describing the safe volume."""
        center = projected.mean(axis=0)
        centered = projected - center
        covariance = np.cov(centered, rowvar=False)
        # Ridge term keeps the inverse defined if a component is degenerate.
        covariance = covariance + np.eye(self.n_components) * 1e-9
        inverse = np.linalg.pinv(covariance)

        radii = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse, centered))
        radius = float(np.percentile(radii, self.envelope_percentile))
        if radius <= 0.0:
            radius = float(radii.max()) or 1.0

        axis_ranges = {
            axis: (float(projected[:, i].min()), float(projected[:, i].max()))
            for i, axis in enumerate("xyz"[: self.n_components])
        }

        vertices: List[List[float]] = []
        faces: List[List[int]] = []
        try:  # scipy ships with scikit-learn, but the hull is optional detail
            from scipy.spatial import ConvexHull

            hull = ConvexHull(projected)
            index_map = {old: new for new, old in enumerate(hull.vertices)}
            vertices = [
                [round(float(c), COORD_PRECISION) for c in projected[v]]
                for v in hull.vertices
            ]
            faces = [[index_map[i] for i in simplex] for simplex in hull.simplices]
        except Exception:
            LOGGER.debug("convex hull unavailable; ellipsoid only", exc_info=True)

        return BoundingEnvelope(
            center=center,
            covariance=covariance,
            inverse_covariance=inverse,
            radius=radius,
            percentile=self.envelope_percentile,
            axis_ranges=axis_ranges,
            hull_vertices=vertices,
            hull_faces=faces,
        )

    # -- projection ---------------------------------------------------------- #

    def _transform(self, vectors: np.ndarray) -> np.ndarray:
        """Transform only -- never fit. The single path all projection takes."""
        self._require_fitted()
        matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if matrix.shape[1] != self.input_dimension:
            raise ValueError(
                f"expected {self.input_dimension}-d vectors, got {matrix.shape[1]}-d"
            )
        return self._pca.transform(matrix) * self._scale

    def project_point(self, vector: np.ndarray) -> Dict[str, float]:
        """Project one state vector to dashboard coordinates."""
        point = self._transform(vector)[0]
        return {
            axis: round(float(value), COORD_PRECISION)
            for axis, value in zip("xyz", point)
        }

    def project_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Project many vectors at once; returns raw ``(n, 3)`` coordinates."""
        return self._transform(vectors)

    def project_frame(
        self,
        vector: np.ndarray,
        *,
        verdict: Optional[str] = None,
        anomaly_score: Optional[float] = None,
        failure_mode: Optional[str] = None,
        unit_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One live UI frame: coordinates plus the decision that produced them.

        This is the ``{x, y, z, verdict}`` payload of the harness flow. The
        envelope fields describe where the point sits *on screen*; they are
        diagnostics for the operator's eye and never override ``verdict``, which
        the policy engine decided in full dimensionality.
        """
        point = self._transform(vector)[0]
        coordinates = {
            axis: round(float(value), COORD_PRECISION) for axis, value in zip("xyz", point)
        }
        envelope_radius = self.envelope.mahalanobis(point) if self.envelope else None
        return {
            "schema": SCHEMA_VERSION,
            "type": "frame",
            **coordinates,
            "verdict": verdict,
            "anomaly_score": None if anomaly_score is None else round(float(anomaly_score), 4),
            "failure_mode": failure_mode,
            "unit_id": unit_id,
            "timestamp": timestamp,
            "envelope_radius": None if envelope_radius is None else round(envelope_radius, 4),
            "inside_envelope": None if envelope_radius is None else bool(envelope_radius <= 1.0),
        }

    # -- export -------------------------------------------------------------- #

    def export_baseline_cloud(self) -> List[Dict[str, float]]:
        """The nominal manifold as ``{x, y, z}`` points for the green cloud."""
        self._require_fitted()
        assert self._baseline_points is not None
        return [
            {axis: round(float(value), COORD_PRECISION) for axis, value in zip("xyz", point)}
            for point in self._baseline_points
        ]

    def export_scene(self) -> Dict[str, Any]:
        """Full boot payload for the dashboard: cloud, envelope and metadata.

        Send once on WebSocket connect, then stream :meth:`project_frame`.
        """
        self._require_fitted()
        assert self._pca is not None
        ratios = self._pca.explained_variance_ratio_
        return {
            "schema": SCHEMA_VERSION,
            "type": "scene",
            "baseline_cloud": self.export_baseline_cloud(),
            "envelope": self.envelope.to_dict() if self.envelope else None,
            "projection": {
                "input_dimension": self.input_dimension,
                "components": self.n_components,
                "explained_variance_ratio": [round(float(v), 6) for v in ratios],
                "explained_variance_total": round(float(ratios.sum()), 6),
                "scale": round(self._scale, 8),
                "frozen": self._frozen,
            },
            "advisory": (
                "Projected coordinates retain "
                f"{ratios.sum() * 100:.1f}% of baseline variance. Render distance is "
                "indicative only; containment is decided in full dimensionality by "
                "the policy engine."
            ),
        }

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize the frozen basis so a restart reuses the same frame.

        Freezing within a process is not enough on its own: a restart that refits
        would silently hand the UI a different coordinate system, and stored
        history would no longer line up with live frames.
        """
        self._require_fitted()
        assert self._pca is not None
        payload = {
            "schema": SCHEMA_VERSION,
            "n_components": self.n_components,
            "input_dimension": self.input_dimension,
            "scale": self._scale,
            "normalize": self.normalize,
            "envelope_percentile": self.envelope_percentile,
            "components": self._pca.components_.tolist(),
            "mean": self._pca.mean_.tolist(),
            "explained_variance": self._pca.explained_variance_.tolist(),
            "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
            "baseline_points": self._baseline_points.tolist(),  # type: ignore[union-attr]
        }
        text = json.dumps(payload)
        if path is not None:
            Path(path).write_text(text)
            LOGGER.info("Projection basis written to %s", path)
        return text

    @classmethod
    def from_json(cls, source: str | Path) -> "LatentSpaceVisualizer":
        """Restore a frozen basis produced by :meth:`to_json`."""
        from sklearn.decomposition import PCA

        if isinstance(source, Path):
            text = source.read_text()
        else:  # a str is either the JSON itself or a path to it
            candidate = str(source)
            text = (
                candidate
                if candidate.lstrip().startswith("{")
                else Path(candidate).read_text()
            )
        payload = json.loads(text)
        if payload.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"projection schema {payload.get('schema')!r} does not match "
                f"{SCHEMA_VERSION!r}; refusing to restore a mismatched frame"
            )

        projector = cls(
            n_components=payload["n_components"],
            normalize=payload["normalize"],
            envelope_percentile=payload["envelope_percentile"],
        )
        pca = PCA(n_components=payload["n_components"])
        pca.components_ = np.asarray(payload["components"], dtype=np.float64)
        pca.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        pca.explained_variance_ = np.asarray(payload["explained_variance"], dtype=np.float64)
        pca.explained_variance_ratio_ = np.asarray(
            payload["explained_variance_ratio"], dtype=np.float64
        )
        pca.n_components_ = payload["n_components"]
        pca.n_features_in_ = payload["input_dimension"]

        projector._pca = pca
        projector._scale = payload["scale"]
        projector._frozen = True
        projector.input_dimension = payload["input_dimension"]
        projector._baseline_points = np.asarray(payload["baseline_points"], dtype=np.float64)
        projector.envelope = projector._build_envelope(projector._baseline_points)
        return projector

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "LatentSpaceVisualizer(unfitted)"
        assert self._pca is not None
        return (
            f"LatentSpaceVisualizer({self.input_dimension}d->{self.n_components}d, "
            f"variance={self._pca.explained_variance_ratio_.sum() * 100:.1f}%, "
            f"baseline={0 if self._baseline_points is None else len(self._baseline_points)}, "
            f"frozen={self._frozen})"
        )


# --------------------------------------------------------------------------- #
# End-to-end demonstration
# --------------------------------------------------------------------------- #


def _run_demo() -> int:
    """Drive the whole harness and emit the payloads a dashboard would receive."""
    from datetime import datetime, timedelta

    from edge_embedding import EdgeEmbedder
    from policy_engine import PolicySandbox, Verdict, initialize_nominal_manifold
    from tactical_telemetry import (
        MockTelemetryGenerator,
        Scenario,
        TacticalStateWindow,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    print("=" * 94)
    print("LATENT PROJECTOR -- HARNESS INTEGRATION")
    print("=" * 94)

    # ---- startup: embed, calibrate, freeze the projection ------------------ #
    embedder = EdgeEmbedder()
    manifold = initialize_nominal_manifold(
        embedder=embedder,
        telemetry_generator=MockTelemetryGenerator(scenario=Scenario.NOMINAL, seed=4242),
        sample_size=300,
    )
    sandbox = PolicySandbox(manifold)

    baseline = baseline_matrix_from_manifold(manifold)
    projector = LatentSpaceVisualizer().fit_baseline(baseline)

    ratios = projector.explained_variance_ratio
    assert ratios is not None
    print(f"\n{projector}")
    print(
        "  components: "
        + ", ".join(f"PC{i + 1}={v * 100:.1f}%" for i, v in enumerate(ratios))
        + f"  (total {ratios.sum() * 100:.1f}%)"
    )
    envelope = projector.envelope
    assert envelope is not None
    print(
        f"  envelope: center=({envelope.center[0]:+.3f}, {envelope.center[1]:+.3f}, "
        f"{envelope.center[2]:+.3f}) radius={envelope.radius:.3f} "
        f"hull={len(envelope.hull_vertices)} vertices / {len(envelope.hull_faces)} faces"
    )

    cloud = projector.export_baseline_cloud()
    check("baseline cloud exports 300 points", len(cloud) == 300, f"n={len(cloud)}")
    check(
        "cloud points are {x, y, z}",
        all(set(p) == {"x", "y", "z"} for p in cloud),
    )
    check(
        "coordinates rounded to 4dp",
        all(
            round(v, COORD_PRECISION) == v
            for p in cloud[:50]
            for v in p.values()
        ),
    )

    # ---- stability: the basis must refuse to move -------------------------- #
    try:
        projector.fit_baseline(baseline)
        check("refit is refused", False, "second fit_baseline() succeeded")
    except ProjectionFrozenError:
        check("refit is refused while frozen", True)

    probe = baseline[0]
    first = projector.project_point(probe)
    for _ in range(200):
        projector.project_point(baseline[np.random.randint(len(baseline))])
    check(
        "coordinates are stable after 200 projections",
        projector.project_point(probe) == first,
        f"{first}",
    )

    unfitted = LatentSpaceVisualizer()
    try:
        unfitted.project_point(probe)
        check("unfitted projector refuses to project", False)
    except ProjectionError:
        check("unfitted projector refuses to project", True)

    # A restart must land in the identical coordinate frame.
    restored = LatentSpaceVisualizer.from_json(projector.to_json())
    check(
        "frame survives serialization round-trip",
        restored.project_point(probe) == first,
        f"{restored.project_point(probe)}",
    )

    # ---- live stream: one continuous patrol through all three scenarios ---- #
    def next_tick(event: Any, generator: Any) -> datetime:
        current = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        return current + timedelta(seconds=generator.tick_seconds)

    window = TacticalStateWindow(window_size=5)
    clock: Optional[datetime] = None
    position: Optional[Tuple[float, float]] = None
    frames: List[Dict[str, Any]] = []
    phases = [
        ("NOMINAL", Scenario.NOMINAL, 24, 7),
        ("EMCON", Scenario.EMCON_BREACH, 5, 11),
        ("NAV", Scenario.NAV_DIVERGENCE, 6, 13),
    ]

    print(f"\n{'-' * 94}")
    print(
        f"{'#':>3}  {'PHASE':<8} {'VERDICT':<8} {'x':>8} {'y':>8} {'z':>8} "
        f"{'env_r':>7} {'in':>4}  {'FAILURE MODE':<22}"
    )
    print("-" * 94)

    step = 0
    for label, scenario, count, seed in phases:
        carry = (
            {"start_time": clock, "origin": position}
            if clock is not None and position is not None
            else {}
        )
        generator = MockTelemetryGenerator(
            scenario=scenario, seed=seed, onset_index=0, **carry
        )
        for event in generator.stream(count):
            window.append(event)
            clock, position = next_tick(event, generator), (event.lat, event.lon)
            ready = len(window) == window.window_size
            vector = embedder.vectorize(window.to_semantic_representation())
            decision = sandbox.evaluate(vector, event, context_ready=ready)
            if not ready:
                continue  # still filling context; nothing to plot yet

            step += 1
            frame = projector.project_frame(
                vector,
                verdict=decision.verdict.value,
                anomaly_score=decision.anomaly_score,
                failure_mode=decision.failure_mode.value,
                unit_id=event.unit_id,
                timestamp=event.timestamp,
            )
            frames.append(frame)
            print(
                f"{step:>3}  {label:<8} {frame['verdict']:<8} "
                f"{frame['x']:>8.4f} {frame['y']:>8.4f} {frame['z']:>8.4f} "
                f"{frame['envelope_radius']:>7.2f} "
                f"{'yes' if frame['inside_envelope'] else 'NO':>4}  "
                f"{frame['failure_mode']:<22}"
            )

    # ---- how well does the picture match the decision? --------------------- #
    permitted = [f for f in frames if f["verdict"] == Verdict.PERMIT.value]
    contained = [f for f in frames if f["verdict"] == Verdict.CONTAIN.value]
    inside_permit = sum(1 for f in permitted if f["inside_envelope"])
    outside_contain = sum(1 for f in contained if not f["inside_envelope"])

    print(f"\n{'=' * 94}\nVISUAL / DECISION AGREEMENT\n{'=' * 94}")
    print(
        f"  PERMIT frames inside the green envelope : {inside_permit}/{len(permitted)}"
    )
    print(
        f"  CONTAIN frames outside the envelope     : {outside_contain}/{len(contained)}"
    )
    # The envelope is a p95 containment region, so ~5% of nominal points are
    # expected outside it. Asserting "all inside" would contradict the design.
    expected_rate = projector.envelope_percentile / 100.0
    tolerance = 0.10  # sampling slack at n=20
    observed_rate = inside_permit / max(1, len(permitted))
    check(
        f"permitted traffic inside the p{projector.envelope_percentile:.0f} envelope "
        f"at the expected rate",
        observed_rate >= expected_rate - tolerance,
        f"{inside_permit}/{len(permitted)} = {observed_rate:.0%}, "
        f"expected ~{expected_rate:.0%}",
    )
    if outside_contain < len(contained):
        print(
            f"  NOTE: {len(contained) - outside_contain} contained frame(s) still plot "
            "inside the envelope -- the 3-d view cannot show every dimension the\n"
            "        policy engine used. Read the verdict, not the picture."
        )

    # ---- wire payloads ------------------------------------------------------ #
    scene = projector.export_scene()
    scene_json = json.dumps(scene)
    frame_json = json.dumps(frames[-1])
    check("scene payload is JSON-serializable", bool(scene_json))
    check("frame payload is JSON-serializable", bool(frame_json))
    check(
        "frame carries {x, y, z, verdict} for the UI socket",
        all(k in frames[-1] for k in ("x", "y", "z", "verdict")),
    )

    print(f"\n{'=' * 94}\nWIRE PAYLOADS\n{'=' * 94}")
    print(f"  scene  : {len(scene_json):,} bytes (sent once on connect)")
    print(f"           keys={list(scene)}")
    print(f"  frame  : {len(frame_json):,} bytes (streamed per tick)")
    print(f"           {frame_json[:180]}...")
    print(f"\n  advisory: {scene['advisory']}")

    print(f"\n{'=' * 94}\nCHECKS\n{'=' * 94}")
    failed = 0
    for name, ok, detail in checks:
        failed += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_demo())
