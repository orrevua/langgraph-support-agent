import pytest

from supportagent.config import Settings
from supportagent.evals import ROOT, is_subsequence, load_scenarios, run_scenario, score_answer, score_trajectory

SCENARIOS = load_scenarios(ROOT / "evals" / "scenarios.yaml")


def test_there_are_ten_scenarios_with_unique_ids():
    assert len(SCENARIOS) == 10
    assert len({s["id"] for s in SCENARIOS}) == 10


def test_is_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([], ["x"])


def test_trajectory_scoring_catches_problems():
    expect = {"tools_in_order": ["a"], "forbidden_tools": ["x"], "interrupts": ["x"], "flags": ["f"]}
    failures = score_trajectory(expect, ["x"], [], [])
    assert len(failures) == 4


def test_answer_scoring_groups():
    expect = {"answer_contains_any": [["R$ 1.200,00", "1,200.00"], ["refund"]], "answer_not_contains": ["oops"]}
    assert score_answer(expect, "Refund of R$1,200.00 done") == []
    assert len(score_answer(expect, "Oops, nothing")) == 3


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_offline_agent_passes_scenario(scenario):
    settings = Settings(storage="memory", llm_provider="fake", payments_retry_base_delay_s=0.0, _env_file=None)
    result = run_scenario(scenario, settings, "memory")
    assert result.passed, result.failures
