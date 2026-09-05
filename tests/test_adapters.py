"""
Adapter tests use requests_mock to fake HTTP responses from each provider's
real endpoint shape -- this verifies the request/response parsing logic
(including the N-samples-per-call path) is correct WITHOUT needing network
access or a real API key (neither is available in this sandbox -- see
README.md). This is separate from the project's demo results, which are
produced by actually running the local-reference adapter for real (see
tests/test_end_to_end.py and results/*.json).
"""
import pytest
import requests_mock

from model_adapters import OpenAIAdapter, AnthropicAdapter, GeminiAdapter, build_adapter


def test_openai_adapter_missing_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = OpenAIAdapter()
    r = a.generate("türkiye'nin başkenti neresidir?")
    assert not r.ok
    assert "API key" in r.error


def test_openai_adapter_parses_single_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    a = OpenAIAdapter(model="gpt-4o-mini")
    with requests_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json={
            "choices": [{"message": {"role": "assistant", "content": "Ankara'dır."}}]
        })
        r = a.generate("türkiye'nin başkenti neresidir?")
    assert r.ok
    assert "Ankara" in r.text


def test_openai_adapter_handles_http_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    a = OpenAIAdapter()
    with requests_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", status_code=401, text="unauthorized")
        r = a.generate("selam")
    assert not r.ok
    assert "401" in r.error


def test_openai_adapter_sample_n_uses_n_parameter(monkeypatch):
    """sample_n should make ONE request carrying n=5 and split the returned
    `choices` list into 5 separate ModelResponses -- the real, documented way
    to draw multiple stochastic samples from the OpenAI chat API in a single
    call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    a = OpenAIAdapter(model="gpt-4o-mini")
    with requests_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json={
            "choices": [{"message": {"content": f"cevap {i}"}} for i in range(5)]
        })
        results = a.sample_n("bir soru", n=5, temperature=0.9)
        assert len(m.request_history) == 1
        sent_body = m.request_history[0].json()
        assert sent_body["n"] == 5
        assert sent_body["temperature"] == 0.9
    assert len(results) == 5
    assert all(r.ok for r in results)
    assert {r.text for r in results} == {f"cevap {i}" for i in range(5)}


def test_openai_adapter_sample_n_pads_when_gateway_ignores_n(monkeypatch):
    """Some OpenAI-compatible gateways silently ignore `n` and return a
    single choice; sample_n must still return exactly N samples by falling
    back to looped calls for the remainder."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    a = OpenAIAdapter(model="gpt-4o-mini")
    with requests_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json={
            "choices": [{"message": {"content": "tek cevap"}}]
        })
        results = a.sample_n("bir soru", n=3)
    assert len(results) == 3
    assert all(r.ok for r in results)


def test_openai_adapter_sample_n_missing_key_returns_n_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = OpenAIAdapter()
    results = a.sample_n("soru", n=4)
    assert len(results) == 4
    assert all(not r.ok for r in results)


def test_anthropic_adapter_parses_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    a = AnthropicAdapter()
    with requests_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json={
            "content": [{"type": "text", "text": "1923 yılında."}]
        })
        r = a.generate("türkiye cumhuriyeti hangi yıl ilan edildi?")
    assert r.ok
    assert "1923" in r.text


def test_anthropic_adapter_sample_n_defaults_to_loop(monkeypatch):
    """Anthropic's API has no `n` parameter, so sample_n should fall back to
    the base class's loop of N independent calls."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    a = AnthropicAdapter()
    with requests_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json={
            "content": [{"type": "text", "text": "1923 yılında."}]
        })
        results = a.sample_n("soru", n=5)
        assert len(m.request_history) == 5
    assert len(results) == 5


def test_gemini_adapter_parses_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    a = GeminiAdapter(model="gemini-1.5-flash")
    with requests_mock.Mocker() as m:
        m.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            json={"candidates": [{"content": {"parts": [{"text": "Ankara"}]}}]},
        )
        r = a.generate("soru", temperature=0.8)
    assert r.ok
    assert r.text == "Ankara"


def test_build_adapter_factory_local_reference():
    a = build_adapter("local-reference", config="grounded")
    r = a.generate("türkiye'nin başkenti neresidir?")
    assert r.ok
    assert isinstance(r.text, str) and len(r.text) > 0


def test_build_adapter_factory_openai_compatible_label(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    a = build_adapter("openai-compatible", model="llama-3.1-8b", base_url="https://api.groq.com/openai/v1")
    assert a.name == "openai-compatible"


def test_build_adapter_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_adapter("not-a-real-provider")
