"""Wire protocol for mock navy agents and the external safety sandbox.

This file intentionally contains data contracts only. The mock agent proposes
commands, and a separate sandbox decides whether those commands are safe.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tactical_telemetry import EmconState, Scenario


CommandValue = Optional[Union[float, bool, str, Dict[str, Union[float, bool, str]]]]


class CommandType(str, Enum):
    """Allowed command types for the mock upstream agent."""

    HOLD_COURSE = "HOLD_COURSE"
    HOLD_POSITION = "HOLD_POSITION"
    SET_COURSE = "SET_COURSE"
    SET_SPEED = "SET_SPEED"
    SET_WAYPOINT = "SET_WAYPOINT"
    SET_EMCON = "SET_EMCON"
    ACTIVATE_RADAR = "ACTIVATE_RADAR"
    DEACTIVATE_RADAR = "DEACTIVATE_RADAR"
    ACTIVATE_AIS = "ACTIVATE_AIS"
    DEACTIVATE_AIS = "DEACTIVATE_AIS"
    CHANGE_SENSOR_MODE = "CHANGE_SENSOR_MODE"
    AVOID_CONTACT = "AVOID_CONTACT"
    TRACK_CONTACT = "TRACK_CONTACT"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    REPORT_STATUS = "REPORT_STATUS"
    REQUEST_OPERATOR_REVIEW = "REQUEST_OPERATOR_REVIEW"


class AgentPersona(str, Enum):
    """Behavior profiles used to simulate different upstream agents."""

    NOMINAL = "NOMINAL"
    CAUTIOUS = "CAUTIOUS"
    MISSION_FOCUSED = "MISSION_FOCUSED"
    DEGRADED_SENSOR = "DEGRADED_SENSOR"
    OVERCONFIDENT = "OVERCONFIDENT"
    ADVERSARIAL_TEST = "ADVERSARIAL_TEST"


class SandboxVerdict(str, Enum):
    """Verdicts returned by the safety sandbox."""

    PERMIT = "PERMIT"
    REVIEW = "REVIEW"
    CONTAIN = "CONTAIN"


class AgentObservation(BaseModel):
    """Current state available to the mock command agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mission_prompt: str
    scenario: Scenario
    persona: AgentPersona
    step: int = Field(..., ge=1)
    telemetry: Dict[str, object]
    semantic_window: str
    maritime_context: Dict[str, object] = Field(default_factory=dict)
    last_sandbox_verdict: Optional[SandboxVerdict] = None
    last_sandbox_reason: Optional[str] = None


class CommandProposal(BaseModel):
    """One typed command proposal emitted by a mock upstream agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command_type: CommandType
    value: CommandValue = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    intent: str = Field(..., min_length=1, max_length=160)
    rationale: str = Field(..., min_length=1, max_length=320)
    agent_persona: AgentPersona

    @model_validator(mode="after")
    def _validate_command_value(self) -> "CommandProposal":
        numeric_commands = {CommandType.HOLD_COURSE, CommandType.SET_COURSE, CommandType.SET_SPEED}
        boolean_commands = {
            CommandType.ACTIVATE_RADAR,
            CommandType.DEACTIVATE_RADAR,
            CommandType.ACTIVATE_AIS,
            CommandType.DEACTIVATE_AIS,
        }
        no_value_commands = {
            CommandType.HOLD_POSITION,
            CommandType.RETURN_TO_BASE,
            CommandType.REPORT_STATUS,
            CommandType.REQUEST_OPERATOR_REVIEW,
        }
        string_commands = {
            CommandType.AVOID_CONTACT,
            CommandType.TRACK_CONTACT,
            CommandType.CHANGE_SENSOR_MODE,
        }

        if self.command_type in numeric_commands and (
            not isinstance(self.value, (int, float)) or isinstance(self.value, bool)
        ):
            raise ValueError(f"{self.command_type.value} requires a numeric value")
        if self.command_type in boolean_commands and self.value is not True:
            raise ValueError(f"{self.command_type.value} requires value=true")
        if self.command_type in no_value_commands and self.value is not None:
            raise ValueError(f"{self.command_type.value} must not include a value")
        if self.command_type in string_commands and not isinstance(self.value, str):
            raise ValueError(f"{self.command_type.value} requires a string value")
        if self.command_type is CommandType.CHANGE_SENSOR_MODE and self.value not in {
            "PASSIVE",
            "ACTIVE",
            "NAV_ONLY",
        }:
            raise ValueError("CHANGE_SENSOR_MODE requires PASSIVE, ACTIVE, or NAV_ONLY")
        if self.command_type is CommandType.SET_WAYPOINT:
            if not isinstance(self.value, dict):
                raise ValueError("SET_WAYPOINT requires a waypoint object")
            required = {"waypoint_id", "lat", "lon"}
            if not required.issubset(self.value):
                raise ValueError(f"SET_WAYPOINT requires {sorted(required)}")
        if self.command_type is CommandType.SET_EMCON:
            allowed = {state.value for state in EmconState}
            if self.value not in allowed:
                raise ValueError(f"SET_EMCON requires one of {sorted(allowed)}")
        return self

    @field_validator("rationale")
    @classmethod
    def _keep_rationale_non_operational(cls, value: str) -> str:
        blocked_terms = ("weapon", "fire", "engage target", "kinetic")
        lowered = value.lower()
        if any(term in lowered for term in blocked_terms):
            raise ValueError("rationale must stay within navigation, EMCON, sensing, or review behavior")
        return value


class SandboxResponse(BaseModel):
    """The sandbox's ruling on one proposed command, as the agent receives it."""

    model_config = ConfigDict(frozen=True, extra="allow")

    verdict: SandboxVerdict
    reason: str = ""
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict_aliases(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized = value.strip().upper()
        aliases = {
            "ALLOW": SandboxVerdict.PERMIT.value,
            "ALLOWED": SandboxVerdict.PERMIT.value,
            "APPROVE": SandboxVerdict.PERMIT.value,
            "APPROVED": SandboxVerdict.PERMIT.value,
            "PERMITTED": SandboxVerdict.PERMIT.value,
            "BLOCK": SandboxVerdict.CONTAIN.value,
            "BLOCKED": SandboxVerdict.CONTAIN.value,
            "CONTAINED": SandboxVerdict.CONTAIN.value,
            "DENY": SandboxVerdict.CONTAIN.value,
            "DENIED": SandboxVerdict.CONTAIN.value,
            "REJECT": SandboxVerdict.CONTAIN.value,
            "REJECTED": SandboxVerdict.CONTAIN.value,
        }
        return aliases.get(normalized, normalized)
