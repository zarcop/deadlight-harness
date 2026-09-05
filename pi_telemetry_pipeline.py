"""Raspberry Pi integration: local telemetry window -> local embedding advisory.

This imports the existing simulator and air-gapped ONNX embedder.  It makes no
network calls: the embedding model must be staged on the Pi before deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from edge_embedding import EdgeEmbedder, REFERENCE_STATES, cosine_similarity
from tactical_telemetry import MockTelemetryGenerator, Scenario, TacticalStateWindow

@dataclass(frozen=True)
class Advisory:
    scenario: str
    semantic_state: str
    closest_reference: str
    similarity: float
    anomalous: bool

class PiTelemetryEmbeddingPipeline:
    """Keeps five local frames, then compares their meaning to local references."""
    def __init__(self, embedder: EdgeEmbedder, reference_states: Sequence[str] = REFERENCE_STATES, *, anomaly_threshold: float = 0.80) -> None:
        if not 0.0 <= anomaly_threshold <= 1.0:
            raise ValueError("anomaly_threshold must be between zero and one")
        self.embedder = embedder
        self.reference_states = list(reference_states)
        self.reference_vectors = embedder.vectorize_batch(self.reference_states)
        self.anomaly_threshold = anomaly_threshold

    def assess(self, scenario: Scenario | str, *, frames: int = 5, seed: int = 1337) -> Advisory:
        generator = MockTelemetryGenerator(scenario=Scenario(scenario), seed=seed)
        window = TacticalStateWindow(window_size=5)
        window.extend(generator.stream(frames))
        semantic_state = window.to_semantic_representation()
        vector = self.embedder.vectorize(semantic_state)
        similarities = [cosine_similarity(vector, reference) for reference in self.reference_vectors]
        index = max(range(len(similarities)), key=similarities.__getitem__)
        score = similarities[index]
        return Advisory(
            scenario=Scenario(scenario).value,
            semantic_state=semantic_state,
            closest_reference=self.reference_states[index],
            similarity=score,
            anomalous=score < self.anomaly_threshold,
        )

if __name__ == "__main__":
    pipeline = PiTelemetryEmbeddingPipeline(EdgeEmbedder(allow_download=False))
    for scenario in Scenario:
        result = pipeline.assess(scenario, frames=7)
        print(f"{result.scenario}: similarity={result.similarity:.3f} anomalous={result.anomalous}")
