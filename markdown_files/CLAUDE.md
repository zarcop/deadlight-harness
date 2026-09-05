# Edge Naval AI Sandbox Policy Harness

## 1. System Overview
The Edge Naval AI Sandbox is a zero-cloud, sub-10ms latency policy enforcement engine designed for Denied, Disrupted, Intermittent, and Limited (DDIL) environments. It acts as an authoritative, offline guardrail for autonomous agents, ensuring they do not violate Rules of Engagement (ROE), EMCON (Emissions Control), or navigational boundaries when severed from the primary command cloud.

By utilizing a lightweight latent-space evaluation pipeline (embeddings + FAISS), the harness evaluates the *context* of a sequence of commands rather than relying solely on brittle, static rule-sets.

---

## 2. Directory & Module Structure

```text
naval_edge_harness/
│
├── core/
│   ├── __init__.py
│   ├── tactical_telemetry.py  # Data schema, mock agent, and sliding window buffer
│   ├── edge_embedding.py      # ONNX/fastembed engine for zero-cloud vectorization
│   ├── policy_engine.py       # FAISS vector store and anomaly threshold logic
│   └── latent_projector.py    # PCA projection for 3D UI visualization
│
├── ui/
│   └── dashboard_mockup/      # (Optional) Three.js or Plotly UI assets
│
├── models/
│   └── bge-small-en-v1.5/     # Locally cached ONNX model weights (Air-gapped)
│
├── main_harness.py            # Main execution loop integrating all modules
└── requirements.txt           # pydantic, fastembed, faiss-cpu, scikit-learn, numpy
```

---

## 3. Component Breakdown & Specifications

### 3.1. `tactical_telemetry.py` (State Management)
**Role:** Ingests raw JSON tactical commands and maintains the contextual state of the agent.
* **Pydantic Schema:** Validates incoming payloads (`unit_id`, `lat/lon`, `speed_kts`, `emcon_state`, `radar_rf_kw`, `corridor_deviation_m`).
* **Sliding Window Buffer:** Uses `collections.deque(maxlen=5)` to hold the last $N$ states.
* **Semantic Serializer:** Flattens the window into a dense string focusing on state transitions (e.g., *"EMCON state remained ALPHA. Radar power spiked from 0.0kW to 25.0kW."*).

### 3.2. `edge_embedding.py` (Latent Space Generator)
**Role:** Converts the semantic string into a high-dimensional vector representation.
* **Engine:** `fastembed` leveraging ONNX runtime for CPU execution.
* **Model:** `BAAI/bge-small-en-v1.5` (384 dimensions).
* **Constraints:** Must execute in $< 8.0$ ms. Must load strictly from the local `models/` cache to ensure air-gapped capability.

### 3.3. `policy_engine.py` (The Sandbox Evaluator)
**Role:** The decisive brain of the harness that permits or blocks commands.
* **Store:** `faiss-cpu` utilizing `IndexFlatIP` for Cosine Similarity.
* **Baseline:** Pre-loaded with ~300 vectors representing safe, nominal operations.
* **Logic:** 
  1. Queries the top $K=5$ nearest neighbors.
  2. Calculates the mean distance.
  3. If $\Delta > 	au$ (threshold), flags the state sequence as anomalous.
  4. Also enforces rigid, deterministic tripwires (e.g., `IF EMCON == ALPHA AND radar > 0.0 -> CONTAIN`).

### 3.4. `latent_projector.py` (Visualization Bridge)
**Role:** Reduces the 384-dimensional vector down to $(x, y, z)$ coordinates for the UI.
* **Engine:** `sklearn.decomposition.PCA`.
* **Constraint:** The PCA model is fit *only once* at startup on the baseline manifold. Live agent telemetry uses `.transform()` only, preventing the 3D visual space from shifting dynamically.

---

## 4. Main Execution Flow (`main_harness.py`)

The overarching loop operates on a strict millisecond latency budget intercepting the agent's socket:

1. **Ingest:** Agent proposes Command $C_t$.
2. **Buffer:** Harness pushes $C_t$ to `TacticalStateWindow` $
ightarrow$ outputs string $S_t$.
3. **Embed:** $S_t$ is passed to `EdgeEmbedder` $
ightarrow$ outputs vector $V_t$.
4. **Evaluate:** $V_t$ is queried in `PolicySandbox` against the FAISS index.
5. **Decide:**
   * If `VERDICT == PERMIT`: Forward packet to hardware actuators.
   * If `VERDICT == CONTAIN`: Drop packet, log tripwire event, alert UI.
6. **Visualize:** $V_t$ is passed to `LatentProjector` $
ightarrow$ pushes $\{x, y, z, 	ext{verdict}\}$ to the UI socket.

---

## 5. Performance Targets

| Metric | Target | Verification Method |
| :--- | :--- | :--- |
| **Total Intercept Latency** | $< 10.0$ ms | `time.perf_counter()` wrappers around main loop |
| **Cloud Dependency** | ZERO | Disable Wi-Fi during Hackathon demo |
| **Memory Footprint** | $< 500$ MB | Docker stats / local process monitor |


