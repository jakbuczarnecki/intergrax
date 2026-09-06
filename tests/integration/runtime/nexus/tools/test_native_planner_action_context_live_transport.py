# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — live typed planner annotation transport qualification (gated)."""

from __future__ import annotations

import os

import pytest

from intergrax.llm_adapters.registry.profile import LLMProfile
from testing_support.native_planner_action_context_transport import (
    live_transport_enabled,
    qualify_planner_transport,
)

pytestmark = [
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.qualification,
    pytest.mark.external_provider,
]


def _skip_unless_live() -> None:
    if not live_transport_enabled():
        pytest.skip("Set INTERGRAX_DS_E2E_12_LIVE=1 to run DS-E2E-12 live transport gate")


def _ollama_profile() -> LLMProfile | None:
    model = os.environ.get("INTERGRAX_DS_E2E_12_OLLAMA_MODEL", "qwen2.5:32b").strip()
    base_url = os.environ.get("INTERGRAX_DS_E2E_12_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    if not model:
        return None
    return LLMProfile(provider="ollama", model=model, options={"base_url": base_url})


def _openai_profile() -> LLMProfile | None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        return None
    model = os.environ.get("INTERGRAX_DS_E2E_12_OPENAI_MODEL", "gpt-4.1").strip()
    return LLMProfile(provider="openai", model=model)


def test_qwen32_atomic_planner_transport_gate() -> None:
    _skip_unless_live()
    profile = _ollama_profile()
    if profile is None:
        pytest.skip("Ollama model not configured for DS-E2E-12")
    adapter = profile.create_adapter()
    if not adapter.supports_tools():
        pytest.skip("Ollama adapter does not support native tools in this environment")
    result = qualify_planner_transport(adapter, provider="ollama", required_attempts=3)
    for index, capture in enumerate(result.captures, start=1):
        assert capture.annotation_count >= 0
        assert capture.business_call_count >= 0
        _ = index
    assert result.gate_passed, (
        f"Qwen transport gate failed: {result.successful_attempts}/"
        f"{result.required_attempts} successes"
    )


def test_openai_control_planner_transport_diagnostic() -> None:
    _skip_unless_live()
    profile = _openai_profile()
    if profile is None:
        pytest.skip("BLOCKED_CREDENTIAL: OPENAI_API_KEY not set")
    adapter = profile.create_adapter()
    if not adapter.supports_tools():
        pytest.skip("OpenAI adapter does not support native tools in this environment")
    result = qualify_planner_transport(adapter, provider="openai", required_attempts=1)
    assert result.successful_attempts >= 1, "OpenAI control transport diagnostic failed"
