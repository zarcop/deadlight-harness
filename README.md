# navy-sandbox-agents

A tiny local sandbox policy that runs offline, for agents deployed in extreme
environments like the sea or remote locations.

Concretely: a zero-cloud policy enforcement harness that sits between an
autonomous naval unit and its actuators. It watches a telemetry stream, decides
whether each proposed command may execute, and contains the ones that violate
EMCON, navigation limits, or simply stop looking like anything the unit has done
before. Everything runs on CPU, air-gapped, inside a ~10 ms intercept budget.

Built for DDIL conditions (Denied, Disrupted, Intermittent, Limited): once the
model is staged, no part of the pipeline touches the network.

---

## Pipeline

```
   agent command
        │
        ▼
┌───────────────────┐   5-frame sliding window, serialized to
│ tactical_telemetry│   transition-focused prose
└─────────┬─────────┘   "Radar power spiked from 0.0kW to 25.0kW."
          ▼
┌───────────────────┐   fastembed + ONNX Runtime, local cache only
│  edge_embedding   │   384-d L2-normalized float32
└─────────┬─────────┘   ~8 ms
          ▼
┌───────────────────┐   layer A: FAISS top-5 vs 300 nominal vectors
│   policy_engine   │   layer B: deterministic doctrinal tripwires
└─────────┬─────────┘   PERMIT / CONTAIN, ~0.06 ms
          ▼
┌───────────────────┐   fit-once PCA, 384-d → (x, y, z)
│ latent_projector  │   scene + per-tick frames for the dashboard
└─────────┬─────────┘   ~0.05 ms
          ▼
   actuators / UI socket
```

---

## Quick start

```bash
pip install -r requirements.txt
```

Stage the embedding model once, on a host that still has network. This is the
only step that ever reaches the internet:

```bash
python edge_embedding.py --stage
```

Then run any module directly — each has a self-contained demonstration:

```bash
python tactical_telemetry.py
```

```bash
python edge_embedding.py
```

```bash
python policy_engine.py
```

```bash
python latent_projector.py
```

`policy_engine.py` and `latent_projector.py` run the whole pipeline end to end:
20 nominal patrol steps, an EMCON breach, then a navigation divergence, printing
live verdicts.

For the live dashboard:

```bash
python main_harness.py
```

Then open <http://127.0.0.1:8787>. The unit patrols nominally; the buttons inject
rogue orders — activate the radar under EMCON, sprint past the speed ceiling,
depart the corridor — and you watch them get contained in real time. **Turn the
network off while it runs; nothing changes.** That is the demo.

To point a deployed node at a staged model directory:

```bash
export EDGE_EMBED_CACHE=/path/to/models
```

---

## Modules

| Module | Role |
| --- | --- |
| `tactical_telemetry.py` | Pydantic schema, seeded scenario generator, `deque`-backed sliding window, semantic serializer |
| `edge_embedding.py` | Air-gapped ONNX embedder with a latency budget and staging CLI |
| `policy_engine.py` | FAISS `IndexFlatIP` baseline, τ calibration, dual-layer evaluator |
| `latent_projector.py` | Fit-once PCA to `(x, y, z)`, bounding envelope, JSON/WebSocket payloads |
| `main_harness.py` | The intercept loop plus a stdlib-only SSE server and command injection |
| `ui/dashboard.html` | Live watchstander dashboard: corridor view, verdict feed, layer panel |

### Verdict contract

```python
PolicyVerdict(
    verdict,            # PERMIT | CONTAIN
    anomaly_score,      # 0.0–1.0, decision boundary anchored at exactly 0.5
    failure_mode,       # NONE | EMCON_VIOLATION | NAVIGATION_DEVIATION | UNKNOWN_BEHAVIOR
    latency_ms,
    explanation,        # short diagnostic for the watchstander
)
```

### Dashboard payloads

`export_scene()` is sent once on WebSocket connect (~15 KB: baseline cloud,
envelope mesh, projection metadata). `project_frame()` streams per tick (~276
bytes) and carries `{x, y, z, verdict, anomaly_score, failure_mode,
envelope_radius, inside_envelope}`.

---

## Measured performance

On a 10-core Apple M-series host, Python 3.11, all values median unless noted:

| Stage | Latency | Notes |
| --- | --- | --- |
| Embedding | 8.05 ms (p95 8.94 ms) | MiniLM-L6, ONNX threads pinned to 6 performance cores |
| Policy evaluation | 0.058 ms (p95 0.107 ms) | FAISS exhaustive search over 300×384 |
| Projection | 0.048 ms | PCA `transform` only |
| **Total intercept** | **~8.2 ms** | against the 10 ms target |

**Those are tight-loop numbers, and a tight loop is not how the harness runs.**
ONNX Runtime parks its worker threads when idle, and waking them costs more than
the inference. At a realistic frame interval the same embedding takes far longer:

| frame interval | embedding p50 |
| --- | --- |
| tight loop | 8.2 ms |
| every 0.2 s | 19.8 ms |
| every 0.6 s | 19.3 ms |
| every 1.0 s | 18.7 ms |

A 2.4× penalty that no benchmark loop reveals, because a benchmark loop never
lets the pool go cold. The gap size barely matters — 0.2 s already costs the
full penalty — so this is thread-pool parking, not CPU frequency scaling.

`EdgeEmbedder.start_keepalive()` holds the pool open with a trivial inference
every 50 ms and recovers most of it: **21.0 ms → 10.1 ms p50** (p95 15.9 ms).
`main_harness.py` enables it by default and prints that it is on; disable with
`--no-keepalive`. The cost is a core held warm continuously, which is a poor
trade on a battery-powered hull when nobody is watching the screen — so it is
opt-out for the demo and should be opt-in for an unattended patrol.

Baseline calibration (300 windows, embed + index + τ) takes ~2.1 s at startup.

Peak RSS for the fully loaded harness is **382 MB**, against the 500 MB platform
target. Almost all of it is the ONNX session (~272 MB); the FAISS index is
~0.5 MB and the PCA basis is negligible.

Peak memory is set by the calibration batch size, and the allocator does not
give it back — steady-state equals peak. Measured across the full harness:

| `batch_size` | peak RSS | calibration |
| --- | --- | --- |
| 8 | 343 MB | 2.1 s |
| **16 (default)** | **410 MB** | **2.1 s** |
| 32 | 458 MB | 2.1 s |
| 64 | 671 MB | 2.1 s |

Calibration time is flat across all of them, so a large batch buys nothing and
costs a quarter of a gigabyte. Lower `batch_size` further on a tighter node.

---

## Known limitations

These are measured, not hypothetical. Read them before trusting the harness in a
demo or an exercise.

**The 3-D view keeps only 31.5% of baseline variance.** Colour dashboard points
by verdict, never by position. All five EMCON breach frames plot *inside* the
green envelope while being contained — an operator reading "distance from the
green blob" as risk would clear a live breach. Containment is decided in full
384-d by `policy_engine`; the projection is situational awareness only. Every
scene payload carries an `advisory` field stating this.

**Latent separation is thin, and EMCON is nearly invisible to it.** Nominal
windows max out at distance 0.0074, τ sits at 0.0092, and the closest EMCON
frame is 0.0088. A 25 kW radar spike is one clause in a long string, so it barely
moves the embedding. The deterministic tripwires are what actually contain
EMCON — the layers are not interchangeable:

| phase | frames | latent layer alone | rules alone | both |
| --- | --- | --- | --- | --- |
| nominal | 20 | 0 false positives | 0 | 0 |
| EMCON breach | 5 | 4 | **5** | 5 |
| nav divergence | 6 | **6** | 5 | 6 |

Each layer covers the other's blind spot exactly once. The latent layer catches
the first divergence frame a full step before any hard limit is crossed; the
rules catch the EMCON frame the latent layer misses. If you want latent EMCON
sensitivity, fix it upstream in the serializer (lead with anomaly clauses, drop
invariant boilerplate) rather than by tuning τ.

**τ is calibrated as a maximum**, per spec, so it sits on the most extreme
nominal sample and is sensitive to a single outlier. `tau_margin` and
`tau_at_percentile()` are exposed if you want a more robust threshold.

**Cold start needs context.** The manifold is calibrated on full 5-frame
windows, so a partially filled window is off-manifold for structural reasons.
Pass `context_ready=False` to `evaluate()` while the window fills — tripwires
stay armed throughout.

**Scenario generators are seeded mocks**, not recorded traffic. τ, the envelope,
and every number above are calibrated against synthetic nominal behaviour and
must be re-derived from real telemetry before this means anything operationally.

---

## Deviations from the harness specification

Two deliberate departures from `markdown_files/CLAUDE.md`, both documented in
the code:

1. **Model.** The spec names `BAAI/bge-small-en-v1.5`. fastembed serves that as
   an int8-quantized graph, and on ARM64 those kernels are a pessimization:
   ~39 ms per inference against MiniLM-L6's ~7.8 ms, with neither CoreML nor
   thread tuning closing the gap. bge cannot meet the spec's own < 8 ms
   constraint on this hardware, so `sentence-transformers/all-MiniLM-L6-v2` is
   the default. Both are supported and identical in dimensionality (384); prefer
   bge on x86 nodes.
2. **Layout.** Modules live at the repository root rather than under `core/`.

`main_harness.py` and the `ui/` dashboard from the spec are not built yet.

---

## Air-gap verification

The embedder refuses to reach the network unless explicitly told to stage. A
missing cache raises `ModelCacheError` naming the staging command rather than
silently downloading. To prove it, run the suite with the model hub pointed at a
dead port:

```bash
HF_ENDPOINT=http://127.0.0.1:9 HF_HUB_OFFLINE=1 python edge_embedding.py
```

Note that fastembed caches models under their *source* repository name, so
`BAAI/bge-small-en-v1.5` lands on disk as `models--qdrant--bge-small-en-v1.5-onnx-q`.
The cache locator reads fastembed's own registry rather than guessing from the
logical name.
