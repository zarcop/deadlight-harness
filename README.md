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
