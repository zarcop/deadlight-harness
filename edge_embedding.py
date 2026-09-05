"""Air-gapped text embedding engine for edge tactical nodes.

Converts tactical state strings (see ``tactical_telemetry.py``) into L2-normalized
float32 vectors on CPU, with no cloud dependency and no PyTorch in the image.

Design constraints for a disconnected deployment:

* **No network at runtime.** The model must already be staged in a local cache
  directory; the engine verifies this before it will construct a session, and
  fails with an actionable error rather than silently reaching for the network.
* **Bounded latency.** ONNX Runtime graph initialization is paid once at
  construction (warmup), so steady-state single-inference latency stays inside
  the budget. Every call is timed with :func:`time.perf_counter_ns` and a
  breach of ``latency_warn_ms`` is logged as a warning.
* **Small footprint.** ``fastembed`` runs ONNX Runtime directly; no torch,
  no transformers.

Staging a model onto a node that still has connectivity, before it deploys::

    python edge_embedding.py --stage

Then run the self-tests (offline)::

    python edge_embedding.py
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "EdgeEmbedder",
    "LatencyReport",
    "ModelCacheError",
    "cosine_similarity",
    "resolve_cache_dir",
    "stage_model",
]

LOGGER = logging.getLogger("edge_embedding")

# --------------------------------------------------------------------------- #
# Model registry and cache layout
# --------------------------------------------------------------------------- #

#: Embedding dimensionality of every model this engine is qualified for.
SUPPORTED_MODELS: Dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
}

#: MiniLM-L6 is the default because it is the only qualified model that meets
#: the latency budget on ARM edge hardware. fastembed serves bge-small as an
#: int8-quantized graph (``qdrant/bge-small-en-v1.5-onnx-q``), and on ARM64 those
#: quantized kernels are a pessimization: measured on an Apple M-series host,
#: bge-small runs ~39ms per inference against MiniLM-L6's ~7.8ms, and neither
#: CoreML nor thread tuning closes the gap. Prefer bge-small only on x86 nodes,
#: or where its retrieval quality is worth ~5x the latency.
DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Latency budget. Exceeding this on a single inference is logged as a warning.
DEFAULT_LATENCY_WARN_MS: float = 15.0

#: Environment variable an operator can set to point at the staged model cache.
CACHE_ENV_VAR: str = "EDGE_EMBED_CACHE"

#: Weight filenames to look for when the registry cannot tell us the exact one.
_FALLBACK_ONNX_NAMES = ("model.onnx", "model_optimized.onnx", "model_quantized.onnx")
_REQUIRED_TOKENIZER_NAMES = ("tokenizer.json",)


def _default_threads() -> Optional[int]:
    """Thread count for ONNX Runtime, pinned to the machine's performance cores.

    On big.LITTLE CPUs (Apple silicon and most ARM SoCs) letting ORT spread work
    onto efficiency cores costs more in synchronization than it recovers in
    parallelism. Measured on a 10-core M-series host (6 performance cores),
    median single-inference latency by thread count was: 1 -> 35.6ms,
    2 -> 19.6ms, 3 -> 13.3ms, ORT default -> 9.4ms, 6 -> 7.8ms.
    """
    if sys.platform == "darwin":
        try:
            probe = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if probe.returncode == 0 and probe.stdout.strip().isdigit():
                return max(1, int(probe.stdout.strip()))
        except Exception:  # non-Apple-silicon darwin, or sysctl unavailable
            LOGGER.debug("performance-core probe failed", exc_info=True)
    count = os.cpu_count()
    return max(1, count) if count else None


class ModelCacheError(RuntimeError):
    """Raised when the model is not fully staged in the local cache."""


def resolve_cache_dir(cache_dir: Optional[os.PathLike[str] | str] = None) -> Path:
    """Resolve the model cache directory.

    Precedence: explicit argument, then ``$EDGE_EMBED_CACHE``, then
    ``~/.cache/edge_embedding``.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    from_env = os.environ.get(CACHE_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return (Path.home() / ".cache" / "edge_embedding").resolve()


def _registry_entry(model_name: str) -> Dict[str, object]:
    """fastembed's own metadata for a model: source repo, weight file, dimension."""
    try:
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry.get("model") == model_name:
                return dict(entry)
    except Exception:  # registry shape and import path vary across versions
        LOGGER.debug("could not read the fastembed model registry", exc_info=True)
    return {}


def _source_repos(model_name: str) -> List[str]:
    """Hub repositories this model may have been staged from.

    fastembed serves its own ONNX conversions, so ``BAAI/bge-small-en-v1.5``
    lands on disk as ``models--qdrant--bge-small-en-v1.5-onnx-q``. Ask the
    registry rather than assuming the cache is named after the logical model.
    """
    repos: List[str] = []
    sources = _registry_entry(model_name).get("sources")
    if isinstance(sources, dict) and sources.get("hf"):
        repos.append(str(sources["hf"]))
    repos.append(model_name)
    return repos


def _required_onnx_names(model_name: str) -> Sequence[str]:
    """The weight file this model actually needs, per the registry."""
    model_file = _registry_entry(model_name).get("model_file")
    if isinstance(model_file, str) and model_file:
        return (Path(model_file).name,)
    return _FALLBACK_ONNX_NAMES


def _candidate_model_dirs(cache_dir: Path, model_name: str) -> List[Path]:
    """Directories where a staged copy of ``model_name`` could live.

    Covers the HuggingFace hub layout
    (``models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/<sha>/``) and the
    older flat archive layout (``fast-bge-small-en-v1.5/``).
    """
    if not cache_dir.is_dir():
        return []

    candidates: List[Path] = []
    seen: set[Path] = set()

    def offer(path: Path) -> None:
        if path.is_dir() and path not in seen:
            seen.add(path)
            candidates.append(path)

    # Most specific first, so a precise hit wins when several models are staged.
    for repo in _source_repos(model_name):
        hub_style = cache_dir / ("models--" + repo.replace("/", "--"))
        for root in (hub_style / "snapshots", hub_style):
            if root.is_dir():
                offer(root)
                for child in root.iterdir():
                    offer(child)

    # Flat archive layout: only siblings carrying this model's slug, otherwise a
    # different staged model could satisfy the check.
    slug = model_name.split("/")[-1].lower()
    for child in cache_dir.iterdir():
        if child.is_dir() and slug in child.name.lower():
            offer(child)
    return candidates


def _has_required_files(directory: Path, model_name: str) -> bool:
    """True when a directory holds this model's ONNX graph and a tokenizer."""
    files = {p.name for p in directory.rglob("*") if p.is_file()}
    return any(n in files for n in _required_onnx_names(model_name)) and any(
        n in files for n in _REQUIRED_TOKENIZER_NAMES
    )


def locate_staged_model(cache_dir: Path, model_name: str) -> Optional[Path]:
    """Return the directory holding a usable staged model, or ``None``."""
    for candidate in _candidate_model_dirs(cache_dir, model_name):
        if _has_required_files(candidate, model_name):
            return candidate
    return None


def enforce_offline() -> None:
    """Hard-disable outbound model downloads for this process.

    Call before importing anything that touches ``huggingface_hub`` if you want
    a guaranteed air-gap; :class:`EdgeEmbedder` also sets these defensively.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def stage_model(
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[os.PathLike[str] | str] = None,
) -> Path:
    """Download a model into the local cache. **Requires network access.**

    Run this once on a connected host before the node deploys. Returns the
    resolved cache directory.
    """
    from fastembed import TextEmbedding  # imported late: staging is optional

    target = resolve_cache_dir(cache_dir)
    target.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Staging %s into %s (network required)", model_name, target)

    model = TextEmbedding(model_name=model_name, cache_dir=str(target))
    list(model.embed(["staging warmup"]))  # force a real forward pass

    staged = locate_staged_model(target, model_name)
    if staged is None:
        raise ModelCacheError(
            f"{model_name} downloaded into {target} but no usable model directory "
            "was found afterwards; inspect the cache layout."
        )
    LOGGER.info("Staged %s at %s", model_name, staged)
    return target


# --------------------------------------------------------------------------- #
# Latency accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LatencyReport:
    """Summary of a timed run, all values in milliseconds."""

    samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_nanoseconds(cls, timings_ns: Sequence[int]) -> "LatencyReport":
        if not timings_ns:
            raise ValueError("no timing samples collected")
        ms = sorted(t / 1e6 for t in timings_ns)
        p95_index = max(0, min(len(ms) - 1, int(round(0.95 * (len(ms) - 1)))))
        return cls(
            samples=len(ms),
            mean_ms=statistics.fmean(ms),
            median_ms=statistics.median(ms),
            p95_ms=ms[p95_index],
            min_ms=ms[0],
            max_ms=ms[-1],
        )

    def __str__(self) -> str:
        return (
            f"n={self.samples} mean={self.mean_ms:.2f}ms median={self.median_ms:.2f}ms "
            f"p95={self.p95_ms:.2f}ms min={self.min_ms:.2f}ms max={self.max_ms:.2f}ms"
        )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector (or each row of a matrix) to unit L2 norm."""
    if vector.ndim == 1:
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else (vector / norm).astype(np.float32, copy=False)
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (vector / norms).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (a dot product once normalized)."""
    return float(np.dot(_l2_normalize(a), _l2_normalize(b)))


class EdgeEmbedder:
    """CPU-only, air-gapped sentence embedder with a latency budget.

    Args:
        model_name: One of :data:`SUPPORTED_MODELS`.
        cache_dir: Where the model is staged. Defaults to ``$EDGE_EMBED_CACHE``
            or ``~/.cache/edge_embedding``.
        allow_download: When ``False`` (the default, and the correct setting
            afloat), a missing cache raises :class:`ModelCacheError` instead of
            reaching for the network.
        latency_warn_ms: Single-inference budget; breaches are logged.
        threads: ONNX Runtime intra-op threads. ``None`` auto-detects the
            performance-core count (see :func:`_default_threads`), which is what
            actually meets the budget on ARM; pass ``0`` to defer to ONNX
            Runtime's own heuristic, or an explicit count to pin it.
        warmup: Run a throwaway inference at construction so the first real
            call does not pay graph-initialization cost.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Optional[os.PathLike[str] | str] = None,
        allow_download: bool = False,
        latency_warn_ms: float = DEFAULT_LATENCY_WARN_MS,
        threads: Optional[int] = None,
        warmup: bool = True,
        history: int = 512,
    ) -> None:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"{model_name!r} is not qualified for this engine; "
                f"choose one of {sorted(SUPPORTED_MODELS)}"
            )

        self.model_name = model_name
        self.dimension = SUPPORTED_MODELS[model_name]
        registry_dim = _registry_entry(model_name).get("dim")
        if isinstance(registry_dim, int) and registry_dim != self.dimension:
            LOGGER.warning(
                "fastembed reports %s as %d-d, overriding the expected %d-d",
                model_name, registry_dim, self.dimension,
            )
            self.dimension = registry_dim
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.allow_download = allow_download
        self.latency_warn_ms = latency_warn_ms
        self._timings_ns: Deque[int] = deque(maxlen=history)
        self._breaches = 0

        self.model_path = self._preflight()
        self._model = self._build_model(threads)

        if warmup:
            self.warmup()

    # -- construction helpers ----------------------------------------------- #

    def _preflight(self) -> Optional[Path]:
        """Verify the model is staged locally; enforce the air-gap if required."""
        staged = locate_staged_model(self.cache_dir, self.model_name)
        if staged is not None:
            LOGGER.info("Model %s resolved from local cache: %s", self.model_name, staged)
            enforce_offline()
            return staged

        if not self.allow_download:
            raise ModelCacheError(
                f"{self.model_name} is not staged under {self.cache_dir}. "
                f"This node is configured air-gapped (allow_download=False). "
                f"Stage it on a connected host with:\n"
                f"    python edge_embedding.py --stage --model {self.model_name}\n"
                f"then ship the directory and set {CACHE_ENV_VAR}={self.cache_dir}."
            )

        LOGGER.warning(
            "Model %s not found in %s; downloading (allow_download=True). "
            "This must not happen on a deployed node.",
            self.model_name,
            self.cache_dir,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return None

    def _build_model(self, threads: Optional[int]):
        from fastembed import TextEmbedding

        kwargs: Dict[str, object] = {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
        }
        resolved_threads = _default_threads() if threads is None else threads
        if resolved_threads:  # 0 means "defer to ONNX Runtime"
            kwargs["threads"] = resolved_threads
            self.threads = resolved_threads
        else:
            self.threads = None

        if not self.allow_download:
            # Newer fastembed forwards this to huggingface_hub; older versions
            # reject it, in which case the preflight above is our guarantee.
            try:
                return TextEmbedding(local_files_only=True, **kwargs)
            except TypeError:
                LOGGER.debug("fastembed does not accept local_files_only; relying on preflight")
        return TextEmbedding(**kwargs)

    def warmup(self, rounds: int = 2) -> None:
        """Pay ONNX graph-init and allocator cost before the first real call."""
        for _ in range(max(1, rounds)):
            list(self._model.embed(["warmup"]))
        self._timings_ns.clear()
        self._breaches = 0

    # -- inference ----------------------------------------------------------- #

    def vectorize(self, text: str) -> np.ndarray:
        """Embed one string into a 1D L2-normalized ``float32`` vector.

        The call is timed with :func:`time.perf_counter_ns`; exceeding
        ``latency_warn_ms`` emits a warning but still returns the vector.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("vectorize() requires a non-empty string")

        start_ns = time.perf_counter_ns()
        raw = next(iter(self._model.embed([text])))
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._record(elapsed_ns, 1)

        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise RuntimeError(
                f"expected {self.dimension}-d output, got {vector.shape[0]}-d "
                f"from {self.model_name}"
            )
        return _l2_normalize(vector)

    def vectorize_batch(self, texts: List[str], *, batch_size: int = 32) -> np.ndarray:
        """Embed many strings into an ``(n, dim)`` matrix of normalized rows.

        Used to pre-compute the latent-space baseline of reference tactical
        states; amortizes tokenization and session overhead across the batch.
        """
        items = list(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if any(not isinstance(t, str) or not t.strip() for t in items):
            raise ValueError("vectorize_batch() requires non-empty strings")

        start_ns = time.perf_counter_ns()
        raw = list(self._model.embed(items, batch_size=batch_size))
        elapsed_ns = time.perf_counter_ns() - start_ns
        self._record(elapsed_ns, len(items))

        matrix = np.asarray(raw, dtype=np.float32).reshape(len(items), -1)
        if matrix.shape[1] != self.dimension:
            raise RuntimeError(
                f"expected {self.dimension}-d output, got {matrix.shape[1]}-d "
                f"from {self.model_name}"
            )
        return _l2_normalize(matrix)

    # -- instrumentation ----------------------------------------------------- #

    def _record(self, elapsed_ns: int, items: int) -> None:
        """Log timing and warn on a per-item budget breach."""
        per_item_ns = elapsed_ns / max(1, items)
        self._timings_ns.append(int(per_item_ns))
        if per_item_ns / 1e6 > self.latency_warn_ms:
            self._breaches += 1
            LOGGER.warning(
                "Embedding latency %.2fms exceeds the %.1fms budget "
                "(%d item(s), %d breach(es) this session)",
                per_item_ns / 1e6,
                self.latency_warn_ms,
                items,
                self._breaches,
            )

    @property
    def breaches(self) -> int:
        """Number of latency-budget breaches observed since construction."""
        return self._breaches

    def latency_stats(self) -> Optional[LatencyReport]:
        """Report over recent calls, or ``None`` if nothing has been timed."""
        if not self._timings_ns:
            return None
        return LatencyReport.from_nanoseconds(list(self._timings_ns))

    def benchmark(self, text: str, *, runs: int = 100, warmup_runs: int = 5) -> LatencyReport:
        """Time ``runs`` single inferences, excluding warmup, without warning spam."""
        for _ in range(max(0, warmup_runs)):
            self.vectorize(text)

        timings_ns: List[int] = []
        for _ in range(max(1, runs)):
            start_ns = time.perf_counter_ns()
            next(iter(self._model.embed([text])))
            timings_ns.append(time.perf_counter_ns() - start_ns)
        return LatencyReport.from_nanoseconds(timings_ns)

    def __repr__(self) -> str:
        return (
            f"EdgeEmbedder(model={self.model_name!r}, dim={self.dimension}, "
            f"cache={str(self.cache_dir)!r}, threads={self.threads}, "
            f"air_gapped={not self.allow_download})"
        )


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #

#: Reference tactical states, shaped like TacticalStateWindow output.
REFERENCE_STATES: List[str] = [
    "Unit USV-GHOST-01: 5 telemetry frames. EMCON state remained ALPHA_SILENT. "
    "Radar power steady near 0.0kW. AIS transponder remained silent. Speed steady "
    "near 14.0kts. Course held near 45deg. Route deviation steady near 20m. "
    "ASSESSMENT: nominal.",
    "Unit USV-GHOST-02: 5 telemetry frames. EMCON state remained ALPHA_SILENT. "
    "Radar power steady near 0.0kW. AIS transponder remained silent. Speed steady "
    "near 13.2kts. Course held near 50deg. Route deviation steady near 31m. "
    "ASSESSMENT: nominal.",
    "Unit USV-GHOST-01: 5 telemetry frames. EMCON state remained ALPHA_SILENT. "
    "Radar power spiked from 0.0kW to 25.0kW. AIS transponder activated "
    "(silent -> transmitting). ASSESSMENT: anomalous. Detected EMCON ALPHA breach.",
    "Unit USV-GHOST-01: 5 telemetry frames. Speed spiked from 13.7kts to 31.6kts. "
    "Course executed 45deg turn to starboard. Route deviation spiked from 34m to "
    "847m. ASSESSMENT: anomalous. Detected corridor breach into hazard zone.",
]


class _TestRunner:
    """Minimal assertion harness so the module stays dependency-free."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failed += 1
            print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))

    def summary(self) -> int:
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed.")
        return 1 if self.failed else 0


def _run_self_tests(model_name: str, cache_dir: Optional[str], allow_download: bool) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runner = _TestRunner()
    budget_ms = 10.0

    print("=" * 78)
    print("EDGE EMBEDDING ENGINE -- SELF TEST")
    print("=" * 78)

    # --- air-gapped load ---------------------------------------------------- #
    resolved = resolve_cache_dir(cache_dir)
    staged = locate_staged_model(resolved, model_name)
    print(f"\nCache directory : {resolved}")
    print(f"Staged model    : {staged if staged else 'NOT FOUND'}")
    if staged is None and not allow_download:
        print(
            f"\nModel is not staged. Run once on a connected host:\n"
            f"    python {Path(__file__).name} --stage --model {model_name}\n"
        )
        return 1

    embedder = EdgeEmbedder(
        model_name=model_name, cache_dir=cache_dir, allow_download=allow_download
    )
    print(f"Engine          : {embedder}\n")
    runner.check(
        "loads from local cache without network",
        embedder.model_path is not None,
        f"{embedder.model_path}",
    )

    # --- shape and dtype ---------------------------------------------------- #
    vector = embedder.vectorize(REFERENCE_STATES[0])
    runner.check("vectorize() returns np.ndarray", isinstance(vector, np.ndarray))
    runner.check("output is 1D", vector.ndim == 1, f"ndim={vector.ndim}")
    runner.check("output is 384-dimensional", vector.shape == (384,), f"shape={vector.shape}")
    runner.check("output dtype is float32", vector.dtype == np.float32, str(vector.dtype))

    # --- normalization ------------------------------------------------------ #
    norm = float(np.linalg.norm(vector))
    runner.check("vector is L2-normalized", abs(norm - 1.0) < 1e-5, f"||v||={norm:.8f}")

    # --- batch -------------------------------------------------------------- #
    matrix = embedder.vectorize_batch(REFERENCE_STATES)
    runner.check(
        "vectorize_batch() shape is (n, 384)",
        matrix.shape == (len(REFERENCE_STATES), 384),
        f"shape={matrix.shape}",
    )
    runner.check("batch dtype is float32", matrix.dtype == np.float32, str(matrix.dtype))
    row_norms = np.linalg.norm(matrix, axis=1)
    runner.check(
        "every batch row is L2-normalized",
        bool(np.allclose(row_norms, 1.0, atol=1e-5)),
        f"min={row_norms.min():.8f} max={row_norms.max():.8f}",
    )
    runner.check(
        "batch agrees with single inference",
        bool(np.allclose(matrix[0], vector, atol=1e-5)),
        f"max|delta|={float(np.abs(matrix[0] - vector).max()):.2e}",
    )
    runner.check(
        "empty batch returns (0, 384)",
        embedder.vectorize_batch([]).shape == (0, 384),
    )

    # --- determinism -------------------------------------------------------- #
    runner.check(
        "identical input yields identical vector",
        bool(np.array_equal(vector, embedder.vectorize(REFERENCE_STATES[0]))),
    )

    # --- latency ------------------------------------------------------------ #
    report = embedder.benchmark(REFERENCE_STATES[0], runs=100)
    print(f"\n  Single-inference latency: {report}")
    runner.check(
        f"median single inference under {budget_ms:.0f}ms on CPU",
        report.median_ms < budget_ms,
        f"median={report.median_ms:.2f}ms",
    )
    # The tail is scheduler jitter, not model cost, so it is held to the
    # declared warning budget rather than the steady-state target.
    runner.check(
        f"p95 single inference within the {embedder.latency_warn_ms:.0f}ms warn budget",
        report.p95_ms < embedder.latency_warn_ms,
        f"p95={report.p95_ms:.2f}ms",
    )

    batch_start = time.perf_counter_ns()
    embedder.vectorize_batch(REFERENCE_STATES * 8)
    batch_per_item_ms = (time.perf_counter_ns() - batch_start) / 1e6 / (len(REFERENCE_STATES) * 8)
    print(f"  Batch amortized per item: {batch_per_item_ms:.2f}ms")
    runner.check(
        "batch is at least as fast per item as single inference",
        batch_per_item_ms <= report.median_ms,
        f"{batch_per_item_ms:.2f}ms vs {report.median_ms:.2f}ms",
    )

    # --- budget warning path ------------------------------------------------ #
    strict = EdgeEmbedder(
        model_name=model_name,
        cache_dir=cache_dir,
        allow_download=allow_download,
        latency_warn_ms=0.0,  # force a breach to prove the warning path fires
    )
    strict.vectorize("forced latency budget breach")
    runner.check("latency breach is detected and logged", strict.breaches == 1,
                 f"breaches={strict.breaches}")
    runner.check("latency_stats() reports samples", strict.latency_stats() is not None)

    # --- input validation ---------------------------------------------------- #
    for bad in ("", "   "):
        try:
            embedder.vectorize(bad)
            runner.check(f"rejects empty input {bad!r}", False)
        except ValueError:
            runner.check(f"rejects empty input {bad!r}", True)

    # --- semantic sanity ----------------------------------------------------- #
    nominal_a, nominal_b, emcon_breach, nav_divergence = matrix
    same = cosine_similarity(nominal_a, nominal_b)
    different = cosine_similarity(nominal_a, emcon_breach)
    print(f"\n  cos(nominal, nominal')      = {same:.4f}")
    print(f"  cos(nominal, emcon_breach)  = {different:.4f}")
    print(f"  cos(nominal, nav_divergence)= {cosine_similarity(nominal_a, nav_divergence):.4f}")
    runner.check(
        "two nominal states are closer than nominal vs. breach",
        same > different,
        f"{same:.4f} > {different:.4f}",
    )
    runner.check("self-similarity is 1.0", abs(cosine_similarity(vector, vector) - 1.0) < 1e-5)

    return runner.summary()


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Download the model into the local cache (requires network).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--cache-dir", default=None, help=f"Overrides ${CACHE_ENV_VAR}.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit an on-demand download during the self-test (not for deployed nodes).",
    )
    args = parser.parse_args(argv)

    if args.stage:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        path = stage_model(args.model, args.cache_dir)
        print(f"Staged {args.model} into {path}")
        print(f"Deploy that directory and set {CACHE_ENV_VAR} to its path on the node.")
        return 0

    return _run_self_tests(args.model, args.cache_dir, args.allow_download)


if __name__ == "__main__":
    sys.exit(_main())
