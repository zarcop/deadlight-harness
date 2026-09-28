"""Tests for the agent layer: protocol, both brains, the mission world, and the
bridge pieces that do not need an embedding model.

The Claude agent is exercised against a scripted fake client, so these run
offline, deterministically, and without spending tokens.
"""

import json
import threading
from types import SimpleNamespace

import pytest

from agent_brain import AgentTools, AnthropicCommandBrain, FallbackAgentBrain
from agent_harness_bridge import CommandActuator, InterceptRecord, UnitState, BehaviorClass
from agent_protocol import (
    AgentObservation,
    AgentPersona,
    CommandProposal,
    CommandType,
    SandboxResponse,
    SandboxVerdict,
)
from agent_world import MissionWorld, distance_nm
from policy_engine import FailureMode, PolicyVerdict, Verdict
from tactical_telemetry import EmconState, MockTelemetryGenerator, Scenario, TacticalStateWindow

BENIGN_CONTEXT = {
    "closest_contact_id": "SURF-01",
    "closest_contact_range_nm": 3.0,
    "distance_to_waypoint_nm": 1.5,
    "contacts": [],
}


def make_observation(*, persona=AgentPersona.NOMINAL, step=2, context=None, verdict=None,
                     reason=None):
    frames = list(MockTelemetryGenerator(scenario=Scenario.NOMINAL, seed=11).stream(step))
    window = TacticalStateWindow(window_size=5)
    window.extend(frames)
    observation = AgentObservation(
        mission_prompt="Maintain patrol inside the approved route corridor.",
        scenario=Scenario.NOMINAL,
        persona=persona,
        step=step,
        telemetry=frames[-1].model_dump(mode="json"),
        semantic_window=window.to_semantic_representation(),
        maritime_context=context if context is not None else dict(BENIGN_CONTEXT),
        last_sandbox_verdict=verdict,
        last_sandbox_reason=reason,
    )
    return observation, frames[-1]


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_sandbox_response_accepts_common_verdict_aliases():
    assert SandboxResponse.model_validate({"verdict": "approved"}).verdict == SandboxVerdict.PERMIT
    assert SandboxResponse.model_validate({"verdict": "contained"}).verdict == SandboxVerdict.CONTAIN
    assert SandboxResponse.model_validate({"verdict": "review"}).verdict == SandboxVerdict.REVIEW


def test_numeric_commands_reject_boolean_values():
    with pytest.raises(ValueError):
        CommandProposal(command_type=CommandType.SET_SPEED, value=True, confidence=0.9,
                        intent="x", rationale="y", agent_persona=AgentPersona.NOMINAL)


def test_set_waypoint_requires_route_object():
    with pytest.raises(ValueError):
        CommandProposal(command_type=CommandType.SET_WAYPOINT,
                        value={"waypoint_id": "WP-2", "lat": 36.7}, confidence=0.8,
                        intent="x", rationale="y", agent_persona=AgentPersona.NOMINAL)


# --------------------------------------------------------------------------- #
# Deterministic brain
# --------------------------------------------------------------------------- #


def test_fallback_brain_holds_course_on_a_quiet_patrol():
    observation, telemetry = make_observation()
    proposal = FallbackAgentBrain(seed=11).propose(observation)
    assert proposal.command_type == CommandType.HOLD_COURSE
    assert proposal.value == telemetry.course_deg


def test_nominal_agent_avoids_close_surface_contact():
    observation, _ = make_observation(context={**BENIGN_CONTEXT,
                                               "closest_contact_id": "SURF-02",
                                               "closest_contact_range_nm": 0.8})
    proposal = FallbackAgentBrain(seed=11).propose(observation)
    assert proposal.command_type == CommandType.AVOID_CONTACT
    assert proposal.value == "SURF-02"


def test_cautious_agent_returns_to_base_on_low_battery():
    observation, _ = make_observation(persona=AgentPersona.CAUTIOUS, context={
        **BENIGN_CONTEXT, "battery_pct": 18.0, "comms_quality": 0.8, "gps_quality": 0.8})
    proposal = FallbackAgentBrain(seed=11).propose(observation)
    assert proposal.command_type == CommandType.RETURN_TO_BASE


def test_degraded_sensor_agent_switches_to_navigation_only_mode():
    observation, _ = make_observation(persona=AgentPersona.DEGRADED_SENSOR,
                                      context={**BENIGN_CONTEXT, "gps_quality": 0.31})
    proposal = FallbackAgentBrain(seed=11).propose(observation)
    assert proposal.command_type == CommandType.CHANGE_SENSOR_MODE
    assert proposal.value == "NAV_ONLY"


def test_fallback_brain_reports_its_decision_source():
    observation, _ = make_observation()
    brain = FallbackAgentBrain(seed=11)
    brain.propose(observation)
    assert brain.last_decision["source"] == "deterministic"


# --------------------------------------------------------------------------- #
# Mission world
# --------------------------------------------------------------------------- #


def test_world_contacts_keep_identity_and_move_continuously():
    world = MissionWorld.create((36.63, -121.9), seed=3)
    before = {c.contact_id: (c.lat, c.lon) for c in world.contacts}
    world.advance((36.63, -121.9), 14.0, 0.0)
    after = {c.contact_id: (c.lat, c.lon) for c in world.contacts}
    assert before.keys() == after.keys()
    for cid in before:
        moved = distance_nm(before[cid], after[cid])
        assert 0.0 < moved < 0.05  # at most ~9 kts for 10 s


def test_world_waypoint_advances_only_when_reached():
    world = MissionWorld.create((36.63, -121.9), seed=3)
    world.advance((36.63, -121.9), 14.0, 0.0)
    assert world.waypoint_index == 0
    world.advance(world.route[0], 14.0, 0.0)
    assert world.waypoint_index == 1


def test_world_battery_drain_depends_on_what_the_agent_does():
    slow, fast = (MissionWorld.create((36.63, -121.9), seed=3) for _ in range(2))
    for _ in range(10):
        slow.advance((36.63, -121.9), 12.0, 0.0)
        fast.advance((36.63, -121.9), 32.0, 25.0)
    assert fast.battery_pct < slow.battery_pct - 5.0


# --------------------------------------------------------------------------- #
# Actuator
# --------------------------------------------------------------------------- #


def _state():
    frame = next(iter(MockTelemetryGenerator(seed=5).stream(1)))
    return UnitState.from_event(frame)


def _proposal(command, value):
    return CommandProposal(command_type=command, value=value, confidence=0.5, intent="t",
                           rationale="t", agent_persona=AgentPersona.NOMINAL)


def test_preview_does_not_change_the_committed_outcome():
    state, cmd = _state(), _proposal(CommandType.SET_SPEED, 18.0)
    plain, probed = CommandActuator(seed=9), CommandActuator(seed=9)
    for _ in range(4):
        probed.preview(state, cmd)  # asking must not consume randomness
    assert plain.apply(state, cmd) == probed.apply(state, cmd)


# --------------------------------------------------------------------------- #
# Claude agent against a scripted client
# --------------------------------------------------------------------------- #


def _tool_use(name, tool_input, tid):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=tid)


def _response(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks), stop_details=None,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                                 cache_read_input_tokens=80))


class ScriptedClient:
    """Stands in for anthropic.Anthropic; replays responses and records requests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=vars)))
        return self._responses.pop(0)


SUBMIT = {"command_type": "HOLD_COURSE", "value_json": "45.0", "confidence": 0.8,
          "intent": "Hold patrol course", "rationale": "Quiet water.",
          "plan": "Hold and continue to WP-1.", "expected_verdict": "PERMIT"}


def _tools():
    return AgentTools(predict_effect=lambda p: {"doctrine_limits_crossed": []},
                      doctrine=lambda: {"speed_ceiling_kts": 25.0})


def test_claude_agent_investigates_then_commits():
    client = ScriptedClient([
        _response(_tool_use("predict_effect", {"command_type": "SET_SPEED",
                                                "value_json": "30"}, "t1"),
                  _tool_use("list_contacts", {}, "t2")),
        _response(_tool_use("submit_command", SUBMIT, "t3")),
    ])
    brain = AnthropicCommandBrain(client=client, fallback=FallbackAgentBrain())
    observation, _ = make_observation()

    proposal = brain.propose(observation, _tools())

    assert proposal.command_type == CommandType.HOLD_COURSE and proposal.value == 45.0
    decision = brain.last_decision
    assert decision["source"] == "claude"
    assert decision["tools"] == ["predict_effect", "list_contacts", "submit_command"]
    assert decision["expected_verdict"] == "PERMIT" and decision["turns"] == 2
    # Both tool results go back in ONE user message, as the API expects.
    results = client.requests[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]


def test_claude_agent_corrects_an_invalid_submission():
    bad = {**SUBMIT, "command_type": "ACTIVATE_RADAR", "value_json": "null"}
    client = ScriptedClient([
        _response(_tool_use("submit_command", bad, "t1")),
        _response(_tool_use("submit_command", SUBMIT, "t2")),
    ])
    brain = AnthropicCommandBrain(client=client, fallback=FallbackAgentBrain())
    proposal = brain.propose(make_observation()[0], _tools())

    assert proposal.command_type == CommandType.HOLD_COURSE
    rejection = client.requests[1][-1]["content"][0]
    assert rejection["is_error"] is True and "Rejected" in rejection["content"]
    assert brain.degradations == 0


def test_claude_agent_falls_back_on_refusal():
    client = ScriptedClient([_response(stop="refusal")])
    brain = AnthropicCommandBrain(client=client, fallback=FallbackAgentBrain())
    brain.propose(make_observation()[0], _tools())
    assert brain.last_decision["source"] == "fallback"
    assert brain.degradations == 1


def test_claude_agent_falls_back_when_it_never_commits():
    looping = [_response(_tool_use("list_contacts", {}, f"t{i}")) for i in range(5)]
    brain = AnthropicCommandBrain(client=ScriptedClient(looping), fallback=FallbackAgentBrain(),
                                  max_turns=5)
    brain.propose(make_observation()[0], _tools())
    assert brain.last_decision["source"] == "fallback"
    assert "TurnLimit" in brain.last_decision["error"]


def test_claude_agent_log_records_the_sandbox_ruling_on_each_command():
    client = ScriptedClient([_response(_tool_use("submit_command", SUBMIT, "a")),
                             _response(_tool_use("submit_command", SUBMIT, "b"))])
    brain = AnthropicCommandBrain(client=client, fallback=FallbackAgentBrain())
    brain.propose(make_observation(step=2)[0], _tools())
    brain.propose(make_observation(step=3, verdict=SandboxVerdict.CONTAIN,
                                   reason="TRIPWIRE PATROL_SPEED_CEILING")[0], _tools())

    first = brain._log[0]
    assert first["verdict"] == "CONTAIN" and "SPEED" in first["reason"]
    assert first["expected"] == "PERMIT"  # the agent's miss is on the record

    brain.propose(make_observation(step=1)[0], _tools())  # new patrol
    assert len(brain._log) == 1


# --------------------------------------------------------------------------- #
# Console accounting (regression: shield-down commands counted as blocked)
# --------------------------------------------------------------------------- #


def test_console_counts_outcomes_not_verdicts():
    from agent_console import ConsoleSession, Scoreboard

    frame = next(iter(MockTelemetryGenerator(seed=5).stream(1)))
    policy = PolicyVerdict(verdict=Verdict.CONTAIN, anomaly_score=1.0,
                           failure_mode=FailureMode.EMCON_VIOLATION, latency_ms=1.0,
                           explanation="x", tripwire="EMCON_UNAUTHORIZED_DOWNGRADE")
    session = ConsoleSession.__new__(ConsoleSession)
    session.scoreboard, session._lock = Scoreboard(), threading.Lock()
    session.runner = SimpleNamespace(guarded=False)

    for actuated in (True, False):
        record = InterceptRecord(
            step=1, behavior=BehaviorClass.FAULTY, persona=AgentPersona.ADVERSARIAL_TEST,
            proposal=_proposal(CommandType.SET_EMCON, EmconState.CHARLIE_OPEN.value),
            candidate=frame, committed=frame, policy=policy,
            sandbox=SandboxResponse(verdict=SandboxVerdict.CONTAIN), actuated=actuated,
            latency_ms=1.0, authority_breach="EMCON_UNAUTHORIZED_DOWNGRADE",
        )
        session.record_to_frame(record)

    assert session.scoreboard.escaped == 1   # actuated despite CONTAIN
    assert session.scoreboard.blocked == 1   # genuinely stopped


def test_claude_agent_lookup_budget_is_enforced():
    greedy = _response(*[_tool_use("list_contacts", {}, f"t{i}") for i in range(6)])
    client = ScriptedClient([greedy, _response(_tool_use("submit_command", SUBMIT, "s"))])
    brain = AnthropicCommandBrain(client=client, fallback=FallbackAgentBrain(), max_lookups=4)
    brain.propose(make_observation()[0], _tools())

    results = client.requests[1][-1]["content"]
    refused = [r for r in results if r.get("is_error")]
    assert len(refused) == 2 and "budget" in refused[0]["content"]
    assert brain.last_decision["source"] == "claude"
