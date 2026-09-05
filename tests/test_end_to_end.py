"""End-to-end smoke tests: run the full battery through the real local
reference model (no mocking) for all three configurations, and check that
the reliability scores actually move in the direction the project's whole
demo is built to show -- this is the real, executed proof behind the
qualitative claims in the README and report, not an assumption."""

import random

from prompts.battery import PROMPTS
from model_adapters import build_adapter
from runner import run_detection


def _run(config, seed):
    random.seed(seed)
    adapter = build_adapter("local-reference", config=config)
    result = run_detection(adapter, target_label=config, samples_per_prompt=5, verbose=False)
    assert result["n_prompts"] == len(PROMPTS)
    assert result["n_sample_errors"] == 0
    return result


def test_confident_fabricator_scores_lower_than_grounded_and_hedging():
    fab = _run("confident-fabricator", seed=1)
    grounded = _run("grounded", seed=1)
    hedging = _run("hedging", seed=1)
    fab_score = fab["summary"]["overall_reliability_score"]
    grounded_score = grounded["summary"]["overall_reliability_score"]
    hedging_score = hedging["summary"]["overall_reliability_score"]
    assert fab_score < grounded_score
    assert fab_score < hedging_score
    assert fab["summary"]["risk_level"] == "YÜKSEK HALÜSİNASYON RİSKİ"


def test_result_shape_has_per_prompt_scores_and_rationale():
    result = _run("grounded", seed=2)
    assert len(result["per_prompt"]) == len(PROMPTS)
    for entry in result["per_prompt"]:
        assert "score" in entry["result"]
        assert "rationale" in entry["result"]
        assert isinstance(entry["result"]["rationale"], str) and len(entry["result"]["rationale"]) > 0
        assert len(entry["result"]["samples"]) == 5


def test_samples_per_prompt_is_configurable():
    random.seed(3)
    adapter = build_adapter("local-reference", config="grounded")
    result = run_detection(adapter, target_label="grounded", samples_per_prompt=3, verbose=False)
    assert result["samples_per_prompt"] == 3
    for entry in result["per_prompt"]:
        assert len(entry["result"]["samples"]) == 3
