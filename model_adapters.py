"""
model_adapters.py
------------------
Pluggable adapters that let the detector send a prompt to *any* chat/
completion model and get back N independent, stochastically-sampled
responses, regardless of which vendor is behind it. This mirrors the
architecture of the sister ai_safety_auditor project: add a new class here
and the rest of the pipeline (prompts, consistency checker, runner, report,
dashboard) does not need to change.

Every network-calling adapter reads its credentials from an environment
variable -- never hard-coded, never logged, never sent anywhere except the
provider's own official API endpoint.

Sampling N independent responses is the core requirement of this project's
self-consistency method: `ModelAdapter.sample_n()` returns a list of N
`ModelResponse`s for the SAME prompt, generated with temperature > 0 so the
samples can genuinely differ. The default implementation just calls
`generate()` in a loop; `OpenAIAdapter` overrides it to use the API's own
`n` parameter (one HTTP call returns N choices), which is the more efficient
real-world approach when the provider supports it.

Included adapters:
  * OpenAIAdapter                  -> api.openai.com/v1/chat/completions (or
                                       any OpenAI-compatible base_url);
                                       overrides sample_n to use the `n` param.
  * AnthropicAdapter                -> api.anthropic.com/v1/messages
  * GeminiAdapter                   -> generativelanguage.googleapis.com
  * OpenAICompatibleAdapter         -> generic adapter for self-hosted / other
                                       OpenAI-schema endpoints (Ollama, vLLM,
                                       LM Studio, Groq, Together, OpenRouter, ...)
  * LocalReferenceModelAdapter      -> a small, fully transparent, locally
                                       running text generator (see
                                       local_reference_model.py) used for the
                                       offline demonstration in this project,
                                       since this sandbox's network egress
                                       blocks every hosted model API and every
                                       weight-hosting service -- see
                                       README.md "Neden yerel bir referans
                                       model?" for the full explanation and
                                       the exact test that was run.
"""

from __future__ import annotations

import abc
import os
import time
import dataclasses
from typing import List, Optional

import requests


@dataclasses.dataclass
class ModelResponse:
    """Normalized result of a single prompt sent to a model."""
    text: str
    latency_s: float
    raw: Optional[dict] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelAdapter(abc.ABC):
    """Base class every model adapter must implement."""

    name: str = "base"

    @abc.abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.9,
                 timeout: float = 30.0) -> ModelResponse:
        """Send `prompt` (and optional `system` instruction) to the model and
        return a ModelResponse. Must never raise -- network/HTTP errors are
        caught and returned as ModelResponse(error=...)."""
        raise NotImplementedError

    def sample_n(self, prompt: str, n: int = 5, system: Optional[str] = None, temperature: float = 0.9,
                  timeout: float = 30.0) -> List[ModelResponse]:
        """Returns N independent samples for the same prompt. Default
        implementation: N separate stochastic calls to generate(). Adapters
        whose provider API can return several choices in one call (see
        OpenAIAdapter) may override this for efficiency; the semantics
        (N independent, temperature>0 draws) stay identical."""
        return [self.generate(prompt, system=system, temperature=temperature, timeout=timeout) for _ in range(n)]


class OpenAIAdapter(ModelAdapter):
    """Works with api.openai.com or any OpenAI-compatible /chat/completions
    endpoint (pass a custom base_url for self-hosted / third-party gateways)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY",
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")

    def _messages(self, prompt, system):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt, system=None, temperature=0.9, timeout=30.0) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(text="", latency_s=0.0, error="missing API key (env var not set)")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": self._messages(prompt, system),
                      "temperature": temperature, "max_tokens": 400},
                timeout=timeout,
            )
            latency = time.time() - t0
            if resp.status_code != 200:
                return ModelResponse(text="", latency_s=latency, error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(text=text, latency_s=latency, raw=data)
        except requests.RequestException as e:
            return ModelResponse(text="", latency_s=time.time() - t0, error=str(e))

    def sample_n(self, prompt, n=5, system=None, temperature=0.9, timeout=30.0) -> List[ModelResponse]:
        """Uses OpenAI's own `n` parameter: a single HTTP call asking for N
        chat completions at once, rather than N separate round-trips. This is
        the real, documented way to draw multiple stochastic samples from the
        OpenAI chat API."""
        if not self.api_key:
            return [ModelResponse(text="", latency_s=0.0, error="missing API key (env var not set)")
                    for _ in range(n)]
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "messages": self._messages(prompt, system),
                      "temperature": temperature, "max_tokens": 400, "n": n},
                timeout=timeout,
            )
            latency = time.time() - t0
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return [ModelResponse(text="", latency_s=latency, error=err) for _ in range(n)]
            data = resp.json()
            choices = data.get("choices", [])
            per_sample_latency = latency / max(len(choices), 1)
            results = [ModelResponse(text=c["message"]["content"], latency_s=per_sample_latency, raw=data)
                       for c in choices]
            # Some OpenAI-compatible gateways ignore `n` and return a single
            # choice; fall back to looping the remainder rather than silently
            # returning fewer samples than requested.
            while len(results) < n:
                results.append(self.generate(prompt, system=system, temperature=temperature, timeout=timeout))
            return results[:n]
        except requests.RequestException as e:
            return [ModelResponse(text="", latency_s=time.time() - t0, error=str(e)) for _ in range(n)]


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Explicit alias for third-party OpenAI-schema gateways (Groq, Together,
    OpenRouter, a local Ollama/vLLM server, ...). Identical wire format to
    OpenAIAdapter; kept as its own class so results show the correct provider
    label. Does NOT override sample_n back to the loop default because most
    OpenAI-compatible gateways do accept `n`; if a given gateway does not,
    OpenAIAdapter.sample_n's fallback loop already covers that case."""

    name = "openai-compatible"


class AnthropicAdapter(ModelAdapter):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key_env: str = "ANTHROPIC_API_KEY",
                 base_url: str = "https://api.anthropic.com/v1"):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt, system=None, temperature=0.9, timeout=30.0) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(text="", latency_s=0.0, error="missing API key (env var not set)")
        payload = {
            "model": self.model,
            "max_tokens": 400,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            latency = time.time() - t0
            if resp.status_code != 200:
                return ModelResponse(text="", latency_s=latency, error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = "".join(block.get("text", "") for block in data.get("content", []))
            return ModelResponse(text=text, latency_s=latency, raw=data)
        except requests.RequestException as e:
            return ModelResponse(text="", latency_s=time.time() - t0, error=str(e))


class GeminiAdapter(ModelAdapter):
    name = "gemini"

    def __init__(self, model: str = "gemini-1.5-flash", api_key_env: str = "GEMINI_API_KEY",
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt, system=None, temperature=0.9, timeout=30.0) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(text="", latency_s=0.0, error="missing API key (env var not set)")
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        t0 = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            latency = time.time() - t0
            if resp.status_code != 200:
                return ModelResponse(text="", latency_s=latency, error=f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return ModelResponse(text=text, latency_s=latency, raw=data)
        except (requests.RequestException, KeyError, IndexError) as e:
            return ModelResponse(text="", latency_s=time.time() - t0, error=str(e))


class LocalReferenceModelAdapter(ModelAdapter):
    """Wraps the small local text generator defined in local_reference_model.py.
    Used only for this project's offline demonstration run -- see README.md
    for why. `config` selects one of three transparently-different response
    strategies (confident-fabricator / hedging / grounded) built on the same
    underlying generator, so the demo can show how self-consistency scoring
    moves across genuinely different hallucination behaviors."""

    name = "local-reference"

    def __init__(self, config: str = "grounded"):
        from local_reference_model import ReferenceSystem
        self.config = config
        self._system = ReferenceSystem(config=config)

    def generate(self, prompt, system=None, temperature=0.9, timeout=30.0) -> ModelResponse:
        t0 = time.time()
        text = self._system.respond(prompt)
        return ModelResponse(text=text, latency_s=time.time() - t0, raw={"config": self.config})
    # sample_n uses the base class's loop default: each call to
    # ReferenceSystem.respond() already makes its own internal random choice
    # (fabricated fact, hedge template, or opener phrase), which is exactly
    # what stands in for "temperature > 0 stochastic sampling" here.


ADAPTER_REGISTRY = {
    "openai": OpenAIAdapter,
    "openai-compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "local-reference": LocalReferenceModelAdapter,
}


def build_adapter(provider: str, **kwargs) -> ModelAdapter:
    """Factory used by the CLI: build_adapter('openai', model='gpt-4o-mini')"""
    if provider not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. Options: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[provider](**kwargs)
