#!/usr/bin/env python3
"""
cli.py
------
Command-line entry point for the AI Hallucination Detector.

Examples
--------
Run against the local reference demo system (no API key / no network needed):
    python3 cli.py --provider local-reference --config grounded --samples-per-prompt 5 --out results/grounded.json

Run against a real hosted model (once network access to the provider and an
API key are available):
    export OPENAI_API_KEY=sk-...
    python3 cli.py --provider openai --model gpt-4o-mini --samples-per-prompt 5 --out results/gpt4o-mini.json

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 cli.py --provider anthropic --model claude-3-5-haiku-20241022 --out results/claude-haiku.json

    export GEMINI_API_KEY=...
    python3 cli.py --provider gemini --model gemini-1.5-flash --out results/gemini-flash.json

    # any OpenAI-compatible endpoint (Ollama, vLLM, Groq, OpenRouter, ...)
    export OPENAI_API_KEY=...
    python3 cli.py --provider openai-compatible --model llama-3.1-8b \\
        --base-url https://api.groq.com/openai/v1 --out results/groq-llama.json
"""

import argparse
import sys

from model_adapters import build_adapter
from runner import run_detection, save_results


def main():
    ap = argparse.ArgumentParser(description="AI Hallucination Detector (self-consistency based)")
    ap.add_argument("--provider", required=True,
                     choices=["openai", "openai-compatible", "anthropic", "gemini", "local-reference"])
    ap.add_argument("--model", default=None, help="Model name/ID (ignored for local-reference)")
    ap.add_argument("--base-url", default=None, help="Override API base URL (openai / openai-compatible)")
    ap.add_argument("--api-key-env", default=None, help="Env var name holding the API key")
    ap.add_argument("--config", default="grounded",
                     choices=["confident-fabricator", "hedging", "grounded"],
                     help="Only used for --provider local-reference")
    ap.add_argument("--samples-per-prompt", type=int, default=5,
                     help="How many independent stochastic samples to draw per prompt (N in self-consistency)")
    ap.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (must be > 0)")
    ap.add_argument("--system-prompt", default=None, help="Optional system prompt to send with every test")
    ap.add_argument("--out", required=True, help="Path to write the JSON results file")
    ap.add_argument("--label", default=None, help="Human-readable label for this target in the report/dashboard")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.temperature <= 0:
        print("WARNING: self-consistency sampling needs temperature > 0 to produce independent samples; "
              "a temperature of 0 will make every sample (near-)identical and the reliability score "
              "meaningless.", file=sys.stderr)

    kwargs = {}
    if args.provider == "local-reference":
        kwargs["config"] = args.config
        label = args.label or f"local-reference ({args.config})"
    else:
        if args.model:
            kwargs["model"] = args.model
        if args.base_url:
            kwargs["base_url"] = args.base_url
        if args.api_key_env:
            kwargs["api_key_env"] = args.api_key_env
        label = args.label or f"{args.provider}:{args.model or 'default'}"

    adapter = build_adapter(args.provider, **kwargs)
    result = run_detection(adapter, target_label=label, samples_per_prompt=args.samples_per_prompt,
                            system_prompt=args.system_prompt, temperature=args.temperature,
                            verbose=not args.quiet)
    save_results(result, args.out)

    s = result["summary"]
    print(f"\n=== {label} ===")
    print(f"Genel güvenilirlik skoru: {s['overall_reliability_score']} / 100  ->  {s['risk_level']}")
    print(f"{result['n_prompts']} istem × {result['samples_per_prompt']} örneklem")
    print(f"Sonuçlar yazıldı: {args.out}")

    if result["n_sample_errors"]:
        print(f"WARNING: {result['n_sample_errors']} örnekleme adaptör/ağ seviyesinde başarısız oldu "
              f"(bkz. '{args.out}' içindeki '[ADAPTER ERROR: ...]' işaretli örnekler).", file=sys.stderr)


if __name__ == "__main__":
    main()
