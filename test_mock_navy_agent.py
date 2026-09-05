import json

import pytest

from agent_brain import FallbackAgentBrain
from agent_protocol import (
    AgentObservation,
    AgentPersona,
    CommandProposal,
    CommandType,
    SandboxResponse,
    SandboxVerdict,
)
from mock_navy_agent import build_maritime_context, run_agent, to_jsonl
from tactical_telemetry import MockTelemetryGenerator, Scenario, TacticalStateWindow


def make_observation(
    *,
    persona: AgentPersona,
    scenario: Scenario = Scenario.NOMINAL,
    step: int = 2,
    seed: int = 11,
    maritime_context: dict | None = None,
) -> tuple[AgentObservation, object]:
    generator = MockTelemetryGenerator(scenario=scenario, seed=seed)
    window = TacticalStateWindow(window_size=5)
    frames = list(generator.stream(step))
    for frame in frames:
        window.append(frame)
    telemetry = frames[-1]
    observation = AgentObservation(
        mission_prompt="Maintain patrol inside the approved route corridor.",
        scenario=scenario,
        persona=persona,
        step=step,
        telemetry=telemetry.model_dump(mode="json"),
        semantic_window=window.to_semantic_representation(),
        maritime_context=maritime_context
        or build_maritime_context(
            telemetry.model_dump(mode="json"),
            scenario=scenario,
            step=step,
            seed=seed,
        ),
    )
    return observation, telemetry


def test_fallback_brain_emits_schema_valid_nominal_command():
    observation, telemetry = make_observation(persona=AgentPersona.NOMINAL)

    proposal = FallbackAgentBrain(seed=11).propose(observation)

    assert proposal.command_type == CommandType.HOLD_COURSE
    assert proposal.value == telemetry.course_deg
    assert proposal.agent_persona == AgentPersona.NOMINAL


def test_adversarial_test_agent_emits_only_proposals_not_verdicts():
    events = run_agent(
        prompt="Maintain EMCON silent patrol inside the approved route corridor.",
        scenario=Scenario.NOMINAL,
        persona=AgentPersona.ADVERSARIAL_TEST,
        steps=4,
        seed=7,
        backend="fallback",
        model=None,
    )

    assert len(events) == 4
    assert all(event.sandbox_response is None for event in events)
    assert {event.proposal.command_type for event in events} <= set(CommandType)


def test_nominal_agent_avoids_close_surface_contact():
    context = {
        "closest_contact_id": "SURF-02",
        "closest_contact_range_nm": 0.8,
        "distance_to_waypoint_nm": 1.5,
    }
    observation, _ = make_observation(persona=AgentPersona.NOMINAL, maritime_context=context)

    proposal = FallbackAgentBrain(seed=11).propose(observation)

    assert proposal.command_type == CommandType.AVOID_CONTACT
    assert proposal.value == "SURF-02"


def test_cautious_agent_returns_to_base_on_low_battery():
    context = {
        "battery_pct": 18.0,
        "comms_quality": 0.8,
        "gps_quality": 0.8,
        "closest_contact_range_nm": 3.0,
        "distance_to_waypoint_nm": 1.4,
    }
    observation, _ = make_observation(persona=AgentPersona.CAUTIOUS, maritime_context=context)

    proposal = FallbackAgentBrain(seed=11).propose(observation)

    assert proposal.command_type == CommandType.RETURN_TO_BASE
    assert proposal.value is None


def test_degraded_sensor_agent_switches_to_navigation_only_mode():
    context = {
        "gps_quality": 0.31,
        "closest_contact_range_nm": 3.0,
        "distance_to_waypoint_nm": 1.4,
    }
    observation, _ = make_observation(persona=AgentPersona.DEGRADED_SENSOR, maritime_context=context)

    proposal = FallbackAgentBrain(seed=11).propose(observation)

    assert proposal.command_type == CommandType.CHANGE_SENSOR_MODE
    assert proposal.value == "NAV_ONLY"


def test_jsonl_output_has_one_event_per_line():
    events = run_agent(
        prompt="Maintain patrol inside the approved route corridor.",
        scenario=Scenario.NOMINAL,
        persona=AgentPersona.MISSION_FOCUSED,
        steps=3,
        seed=13,
        backend="fallback",
        model=None,
    )

    lines = to_jsonl(events).splitlines()

    assert len(lines) == 3
    assert json.loads(lines[0])["proposal"]["command_type"] in {item.value for item in CommandType}
    assert "maritime_context" in json.loads(lines[0])["observation"]


def test_agent_trace_matches_harness_envelope_contract():
    events = run_agent(
        prompt="Maintain patrol inside the approved route corridor.",
        scenario=Scenario.NOMINAL,
        persona=AgentPersona.NOMINAL,
        steps=1,
        seed=13,
        backend="fallback",
        model=None,
    )

    event = json.loads(to_jsonl(events))

    assert set(event) == {"event_id", "timestamp", "observation", "proposal", "sandbox_response"}
    assert {"mission_prompt", "scenario", "telemetry", "semantic_window", "maritime_context"} <= set(
        event["observation"]
    )
    assert {"command_type", "value", "confidence", "intent", "rationale", "agent_persona"} <= set(
        event["proposal"]
    )


def test_sandbox_response_accepts_common_verdict_aliases():
    assert SandboxResponse.model_validate({"verdict": "approved"}).verdict == SandboxVerdict.PERMIT
    assert SandboxResponse.model_validate({"verdict": "contained"}).verdict == SandboxVerdict.CONTAIN
    assert SandboxResponse.model_validate({"verdict": "review"}).verdict == SandboxVerdict.REVIEW


def test_numeric_commands_reject_boolean_values():
    with pytest.raises(ValueError):
        CommandProposal(
            command_type=CommandType.SET_SPEED,
            value=True,
            confidence=0.9,
            intent="Invalid numeric command",
            rationale="Boolean values must not pass numeric command validation.",
            agent_persona=AgentPersona.NOMINAL,
        )


def test_set_waypoint_requires_route_object():
    with pytest.raises(ValueError):
        CommandProposal(
            command_type=CommandType.SET_WAYPOINT,
            value={"waypoint_id": "WP-2", "lat": 36.7},
            confidence=0.8,
            intent="Invalid route update",
            rationale="Waypoint commands must include the complete route point.",
            agent_persona=AgentPersona.NOMINAL,
        )
