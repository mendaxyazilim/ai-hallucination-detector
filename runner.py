"""
runner.py
---------
Orchestrates a full detection run: for every prompt in the battery, draw N
independent stochastic samples from the configured model adapter, score
their self-consistency, and collect results.
"""

from __future__ import annotations

import json
import time
import dataclasses
from typing import Dict, List, Optional

from prompts.battery import PROMPTS
from consistency_checker import score_consistency, aggregate
from model_adapters import ModelAdapter


def _result_to_dict(result) -> Dict:
    d = dataclasses.asdict(result)
    return d


def run_detection(adapter: ModelAdapter, target_label: str, samples_per_prompt: int = 5,
                   system_prompt: Optional[str] = None, temperature: float = 0.9,
                   prompts: Optional[List[Dict]] = None, verbose: bool = True) -> Dict:
    """Runs the full (or a filtered) prompt battery against `adapter`,
    drawing `samples_per_prompt` independent samples per prompt, and returns
    a JSON-serializable results dict: per-prompt samples + consistency score,
    plus the aggregated summary."""
    prompts = prompts if prompts is not None else PROMPTS
    per_prompt: List[Dict] = []
    n_sample_errors = 0

    for i, entry in enumerate(prompts, 1):
        responses = adapter.sample_n(entry["prompt"], n=samples_per_prompt, system=system_prompt,
                                      temperature=temperature)
        samples = []
        for r in responses:
            if not r.ok:
                n_sample_errors += 1
                samples.append(f"[ADAPTER ERROR: {r.error}]")
            else:
                samples.append(r.text)

        result = score_consistency(samples)
        result_dict = _result_to_dict(result)
        per_prompt.append({
            "id": entry["id"],
            "category": entry["category"],
            "prompt": entry["prompt"],
            "result": result_dict,
        })
        if verbose:
            print(f"[{i:2}/{len(prompts)}] {entry['id']:10} güvenilirlik={result.score:5.1f}  "
                  f"({entry['category']})")

    summary = aggregate(per_prompt)
    result = {
        "target": target_label,
        "provider": adapter.name,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples_per_prompt": samples_per_prompt,
        "n_prompts": len(prompts),
        "n_sample_errors": n_sample_errors,
        "summary": summary,
        "per_prompt": per_prompt,
    }
    return result


def save_results(result: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
