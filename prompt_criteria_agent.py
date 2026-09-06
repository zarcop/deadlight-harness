"""Prompt-to-policy criteria agent for the edge sandbox.

This module sits one layer above the telemetry harness. Given an operator prompt
or mission instruction, it derives explicit acceptance and rejection criteria
that a local safety harness can enforce against tactical telemetry windows.

The implementation is deliberately deterministic and offline-first for DDIL
settings. An LLM can be added later as a prompt normalizer, but the emitted
criteria should remain typed, inspectable and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from tactical_telemetry import (
    CORRIDOR_HAZARD_M,
    CORRIDOR_WARNING_M,
    PATROL_SPEED_CEILING_KTS,
    RADAR_BREACH_KW,
    EmconState,
    MockTelemetryGenerator,
    Scenario,
    TacticalStateWindow,
)


class CriterionKind(str, Enum):
    """Whether a criterion describes admissible or inadmissible behavior."""

    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"


class PromptVerdict(str, Enum):
    """Prompt-level admissibility before live telemetry is considered."""

    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Criterion(BaseModel):
    """One enforceable criterion generated from a mission prompt."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: CriterionKind
    description: str
    telemetry_fields: List[str] = Field(default_factory=list)
    severity: RiskLevel = RiskLevel.MEDIUM


class PromptCriteriaReport(BaseModel):
    """Decision artifact produced by the prompt criteria agent."""

    model_config = ConfigDict(frozen=True)

    prompt_id: str
    prompt: str
    verdict: PromptVerdict
    risk_level: RiskLevel
    mission_objective: str
    detected_intents: List[str]
    acceptance_criteria: List[Criterion]
    rejection_criteria: List[Criterion]
    required_observations: List[str]
    rationale: List[str]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class PromptCriteriaAgent:
    """Offline prompt analyzer that emits typed safety criteria."""

    INTENT_KEYWORDS: Dict[str, Sequence[str]] = {
        "emcon_control": ("emcon", "silent", "emissions", "radar", "ais", "transponder"),
        "navigation_corridor": ("route", "corridor", "waypoint", "patrol", "navigate", "course", "heading"),
        "speed_governance": ("speed", "sprint", "throttle", "intercept", "pursue", "accelerate"),
        "ddil_resilience": ("ddil", "offline", "disconnected", "uplink", "denied", "degraded"),
        "human_review": ("review", "operator", "human", "approval", "authorize"),
    }

    HIGH_RISK_TERMS: Sequence[str] = (
        "ignore",
        "bypass",
        "override",
        "disable safety",
        "no review",
        "unrestricted",
        "maximum throttle",
        "evade",
    )

    def analyze(self, prompt: str) -> PromptCriteriaReport:
        normalized = self._normalize(prompt)
        prompt_id = "prompt_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        intents = self._detect_intents(normalized)
        acceptance = self._acceptance_criteria(intents)
        rejection = self._rejection_criteria(intents)
        risky_terms = [term for term in self.HIGH_RISK_TERMS if term in normalized]

        if risky_terms and "human_review" not in intents:
            verdict = PromptVerdict.REJECT
            risk = RiskLevel.HIGH
            rationale = [
                "Prompt contains autonomy-escalating language without explicit review constraints.",
                f"Matched high-risk terms: {', '.join(risky_terms)}.",
            ]
        elif "ddil_resilience" in intents or risky_terms:
            verdict = PromptVerdict.NEEDS_REVIEW
            risk = RiskLevel.MEDIUM if not risky_terms else RiskLevel.HIGH
            rationale = [
                "Prompt is operationally valid but requires stricter local thresholds under degraded connectivity.",
            ]
        else:
            verdict = PromptVerdict.ACCEPT
            risk = RiskLevel.LOW if len(intents) <= 2 else RiskLevel.MEDIUM
            rationale = [
                "Prompt can be bounded by available telemetry fields and deterministic local criteria.",
            ]

        if not intents:
            intents = ["general_tactical_safety"]
            acceptance.extend(self._general_acceptance())
            rejection.extend(self._general_rejection())
            verdict = PromptVerdict.NEEDS_REVIEW
            risk = RiskLevel.MEDIUM
            rationale.append("No specific operational intent was detected, so generic safety criteria were applied.")

        return PromptCriteriaReport(
            prompt_id=prompt_id,
            prompt=prompt.strip(),
            verdict=verdict,
            risk_level=risk,
            mission_objective=self._mission_objective(prompt, intents),
            detected_intents=sorted(intents),
            acceptance_criteria=self._dedupe(acceptance),
            rejection_criteria=self._dedupe(rejection),
            required_observations=self._required_observations(intents),
            rationale=rationale,
        )

    def evaluate_window(
        self,
        report: PromptCriteriaReport,
        window: TacticalStateWindow,
    ) -> Dict[str, object]:
        """Evaluate a telemetry window against generated rejection criteria."""
        latest = window.latest
        if latest is None:
            return {
                "prompt_id": report.prompt_id,
                "window_verdict": "needs_review",
                "matched_rejection_criteria": [],
                "summary": "No telemetry frames available for evaluation.",
            }

        matches: List[str] = []
        if latest.is_emcon_violation:
            matches.append("REJ-EMCON-ALPHA-EMIT")
        if latest.is_corridor_breach:
            matches.append("REJ-CORRIDOR-HAZARD")
        if latest.is_speed_excursion:
            matches.append("REJ-PATROL-SPEED")

        if matches:
            verdict = "reject"
        elif any(frame.corridor_deviation_m > CORRIDOR_WARNING_M for frame in window):
            verdict = "needs_review"
        else:
            verdict = "accept"

        return {
            "prompt_id": report.prompt_id,
            "window_verdict": verdict,
            "matched_rejection_criteria": matches,
            "semantic_window": window.to_semantic_representation(),
        }

    @classmethod
    def _normalize(cls, prompt: str) -> str:
        return re.sub(r"\s+", " ", prompt.strip().lower())

    def _detect_intents(self, normalized: str) -> Set[str]:
        return {
            intent
            for intent, keywords in self.INTENT_KEYWORDS.items()
            if any(keyword in normalized for keyword in keywords)
        }

    def _acceptance_criteria(self, intents: Iterable[str]) -> List[Criterion]:
        criteria: List[Criterion] = []
        intents = set(intents)

        if "emcon_control" in intents:
            criteria.append(Criterion(
                id="ACC-EMCON-ALPHA-SILENT",
                kind=CriterionKind.ACCEPTANCE,
                description=(
                    f"When EMCON is {EmconState.ALPHA_SILENT.value}, radar power remains 0.0kW "
                    "and AIS remains inactive."
                ),
                telemetry_fields=["emcon_state", "radar_rf_kw", "ais_active"],
                severity=RiskLevel.HIGH,
            ))

        if "navigation_corridor" in intents:
            criteria.append(Criterion(
                id="ACC-CORRIDOR-NOMINAL",
                kind=CriterionKind.ACCEPTANCE,
                description=(
                    f"Route deviation stays at or below {CORRIDOR_WARNING_M:.0f}m during nominal patrol."
                ),
                telemetry_fields=["corridor_deviation_m", "course_deg", "lat", "lon"],
                severity=RiskLevel.MEDIUM,
            ))

        if "speed_governance" in intents:
            criteria.append(Criterion(
                id="ACC-PATROL-SPEED",
                kind=CriterionKind.ACCEPTANCE,
                description=(
                    f"Speed remains at or below the {PATROL_SPEED_CEILING_KTS:.1f}kt patrol ceiling "
                    "unless a human-review criterion is satisfied."
                ),
                telemetry_fields=["speed_kts"],
                severity=RiskLevel.MEDIUM,
            ))

        if "ddil_resilience" in intents:
            criteria.append(Criterion(
                id="ACC-DDIL-LOCAL-ONLY",
                kind=CriterionKind.ACCEPTANCE,
                description=(
                    "All accept/reject decisions are made from locally buffered telemetry without cloud dependency."
                ),
                telemetry_fields=["timestamp", "unit_id"],
                severity=RiskLevel.HIGH,
            ))

        if "human_review" in intents:
            criteria.append(Criterion(
                id="ACC-REVIEW-ESCALATION",
                kind=CriterionKind.ACCEPTANCE,
                description="Ambiguous or high-severity states produce needs_review instead of autonomous acceptance.",
                telemetry_fields=["corridor_deviation_m", "speed_kts", "radar_rf_kw", "ais_active"],
                severity=RiskLevel.MEDIUM,
            ))

        return criteria

    def _rejection_criteria(self, intents: Iterable[str]) -> List[Criterion]:
        criteria: List[Criterion] = []
        intents = set(intents)

        if "emcon_control" in intents:
            criteria.append(Criterion(
                id="REJ-EMCON-ALPHA-EMIT",
                kind=CriterionKind.REJECTION,
                description=(
                    f"Reject if EMCON is {EmconState.ALPHA_SILENT.value} and radar power is above 0.0kW "
                    "or AIS is active."
                ),
                telemetry_fields=["emcon_state", "radar_rf_kw", "ais_active"],
                severity=RiskLevel.HIGH,
            ))

        if "navigation_corridor" in intents:
            criteria.append(Criterion(
                id="REJ-CORRIDOR-HAZARD",
                kind=CriterionKind.REJECTION,
                description=(
                    f"Reject if route deviation exceeds {CORRIDOR_HAZARD_M:.0f}m from the approved corridor."
                ),
                telemetry_fields=["corridor_deviation_m"],
                severity=RiskLevel.HIGH,
            ))

        if "speed_governance" in intents:
            criteria.append(Criterion(
                id="REJ-PATROL-SPEED",
                kind=CriterionKind.REJECTION,
                description=(
                    f"Reject if speed exceeds {PATROL_SPEED_CEILING_KTS:.1f}kt without review authority."
                ),
                telemetry_fields=["speed_kts"],
                severity=RiskLevel.HIGH,
            ))

        if "ddil_resilience" in intents:
            criteria.append(Criterion(
                id="REJ-CLOUD-DEPENDENCY",
                kind=CriterionKind.REJECTION,
                description="Reject criteria sets that require cloud calls, remote APIs, or nonlocal state.",
                telemetry_fields=["timestamp", "unit_id"],
                severity=RiskLevel.HIGH,
            ))

        return criteria

    @staticmethod
    def _general_acceptance() -> List[Criterion]:
        return [
            Criterion(
                id="ACC-VALID-TELEMETRY",
                kind=CriterionKind.ACCEPTANCE,
                description="Telemetry frames validate against the local TacticalCommandEvent schema.",
                telemetry_fields=["timestamp", "unit_id", "lat", "lon"],
                severity=RiskLevel.MEDIUM,
            )
        ]

    @staticmethod
    def _general_rejection() -> List[Criterion]:
        return [
            Criterion(
                id="REJ-INVALID-TELEMETRY",
                kind=CriterionKind.REJECTION,
                description="Reject malformed, missing, or physically impossible telemetry frames.",
                telemetry_fields=["timestamp", "unit_id", "lat", "lon", "speed_kts", "course_deg"],
                severity=RiskLevel.MEDIUM,
            )
        ]

    @staticmethod
    def _mission_objective(prompt: str, intents: Iterable[str]) -> str:
        clean_prompt = re.sub(r"\s+", " ", prompt.strip())
        if clean_prompt:
            return clean_prompt[:220]
        return "No prompt supplied; derive generic tactical safety criteria."

    @staticmethod
    def _required_observations(intents: Iterable[str]) -> List[str]:
        fields = {"timestamp", "unit_id"}
        for intent in intents:
            if intent == "emcon_control":
                fields.update({"emcon_state", "radar_rf_kw", "ais_active"})
            elif intent == "navigation_corridor":
                fields.update({"lat", "lon", "course_deg", "corridor_deviation_m"})
            elif intent == "speed_governance":
                fields.update({"speed_kts"})
            elif intent == "ddil_resilience":
                fields.update({"local_window_size", "cloud_dependency_status"})
            elif intent == "human_review":
                fields.update({"operator_review_available"})
        return sorted(fields)

    @staticmethod
    def _dedupe(criteria: Iterable[Criterion]) -> List[Criterion]:
        by_id = {criterion.id: criterion for criterion in criteria}
        return [by_id[key] for key in sorted(by_id)]


def _demo_window(scenario: Scenario) -> TacticalStateWindow:
    generator = MockTelemetryGenerator(scenario=scenario)
    window = TacticalStateWindow(window_size=5)
    window.extend(generator.stream(7))
    return window


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate prompt acceptance/rejection criteria.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "Maintain EMCON silent patrol inside the approved route corridor while operating offline "
            "and escalate unsafe states for human review."
        ),
        help="Mission or operator prompt to analyze.",
    )
    parser.add_argument(
        "--evaluate-scenario",
        choices=[scenario.value for scenario in Scenario],
        help="Optionally evaluate generated criteria against a mock telemetry scenario.",
    )
    args = parser.parse_args(argv)

    agent = PromptCriteriaAgent()
    report = agent.analyze(args.prompt)
    print(report.to_json())

    if args.evaluate_scenario:
        scenario = Scenario(args.evaluate_scenario)
        evaluation = agent.evaluate_window(report, _demo_window(scenario))
        print(json.dumps(evaluation, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
