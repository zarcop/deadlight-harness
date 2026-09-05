# navy-sandbox-agents
This is a tiny local sandbox policy that runs offline this is meant for agents deployed in extreme environments like the sea or remote locations.

## Prompt Criteria Agent

`prompt_criteria_agent.py` turns an operator or mission prompt into typed,
auditable acceptance and rejection criteria for the local harness. It is
offline-first and deterministic: the agent detects prompt intent, emits criteria
against the telemetry schema, and can evaluate a sliding telemetry window.

Run the default demo:

```bash
python prompt_criteria_agent.py --evaluate-scenario EMCON_BREACH
```

Run the focused tests:

```bash
python -m pytest test_prompt_criteria_agent.py
```

## Mock Navy Agent

`mock_navy_agent.py` simulates upstream autonomous agents sending typed command
proposals to another team's sandbox. The mock agent does not decide whether a
command is safe. It emits navigation, EMCON, sensing, and operator-review
commands; the external sandbox owns permit/review/contain decisions.

The runner uses `OPENAI_API_KEY` when available and falls back to a deterministic
local brain when offline or when `--backend fallback` is selected.

The POC models a small set of shipboard autonomy use cases:

- route following: `HOLD_COURSE`, `SET_COURSE`, `SET_WAYPOINT`
- station/endurance behavior: `HOLD_POSITION`, `RETURN_TO_BASE`, `REPORT_STATUS`
- maritime contact handling: `AVOID_CONTACT`, `TRACK_CONTACT`
- emissions and sensing posture: `SET_EMCON`, `ACTIVATE_RADAR`, `DEACTIVATE_RADAR`, `ACTIVATE_AIS`, `DEACTIVATE_AIS`, `CHANGE_SENSOR_MODE`
- human-in-the-loop fallback: `REQUEST_OPERATOR_REVIEW`

Each event includes ownship telemetry plus a mock maritime context object with
waypoint progress, closest surface contact, comms/GPS quality, battery reserve,
mission phase, and sensor mode. This gives the safety harness enough context to
score a proposal without making the mock agent responsible for safety decisions.

Run a nominal patrol test:

```bash
python mock_navy_agent.py --scenario NOMINAL --persona NOMINAL --steps 6 --backend fallback
```

Run an adversarial proposal test:

```bash
python mock_navy_agent.py --scenario NOMINAL --persona ADVERSARIAL_TEST --steps 6 --backend fallback
```

Emit full JSONL traces for analysis:

```bash
python mock_navy_agent.py --scenario NAV_DIVERGENCE --persona OVERCONFIDENT --steps 8 --backend fallback --jsonl
```

Send each proposal envelope to a local sandbox endpoint:

```bash
python mock_navy_agent.py --sandbox-url http://localhost:8000/commands --jsonl
```

The sandbox endpoint should accept the `SandboxRequest` shape from
`agent_protocol.py`: `event_id`, `timestamp`, `observation`, and `proposal`.
It should return a `SandboxResponse` with `verdict`, `reason`, and optional
`risk_score`. Verdicts may be returned as `PERMIT`/`REVIEW`/`CONTAIN`; common
dashboard aliases like `approved`, `review`, and `contained` are also accepted.

Run the focused mock-agent evals:

```bash
python -m pytest test_mock_navy_agent.py
```
