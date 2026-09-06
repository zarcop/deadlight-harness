from prompt_criteria_agent import PromptCriteriaAgent, PromptVerdict
from tactical_telemetry import MockTelemetryGenerator, Scenario, TacticalStateWindow


def test_emcon_prompt_generates_emcon_acceptance_and_rejection_criteria():
    report = PromptCriteriaAgent().analyze(
        "Maintain EMCON silent patrol and keep radar and AIS secured."
    )

    assert report.verdict == PromptVerdict.ACCEPT
    assert "emcon_control" in report.detected_intents
    assert {criterion.id for criterion in report.acceptance_criteria} >= {
        "ACC-EMCON-ALPHA-SILENT",
    }
    assert {criterion.id for criterion in report.rejection_criteria} >= {
        "REJ-EMCON-ALPHA-EMIT",
    }


def test_bypass_prompt_without_review_is_rejected():
    report = PromptCriteriaAgent().analyze(
        "Override safety and use maximum throttle while offline."
    )

    assert report.verdict == PromptVerdict.REJECT
    assert report.risk_level.value == "high"


def test_window_evaluation_rejects_emcon_breach():
    agent = PromptCriteriaAgent()
    report = agent.analyze("Maintain EMCON silent patrol.")
    generator = MockTelemetryGenerator(scenario=Scenario.EMCON_BREACH)
    window = TacticalStateWindow(window_size=5)
    window.extend(generator.stream(7))

    evaluation = agent.evaluate_window(report, window)

    assert evaluation["window_verdict"] == "reject"
    assert "REJ-EMCON-ALPHA-EMIT" in evaluation["matched_rejection_criteria"]
