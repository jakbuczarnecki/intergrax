# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for Tier-2 agent unit tests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared import llm_resolver
from testing_support.builder import MeteringFakeLLMAdapter


@pytest.fixture(autouse=True)
def _deterministic_host_llm_for_token_budget_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-context ACP runs must not call live Ollama from ``LLMProfile.lab()``."""
    nodeid = request.node.nodeid.replace("\\", "/")
    if "test_acp_token_" not in nodeid:
        return
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> MeteringFakeLLMAdapter:
        del env
        if agent_override is not None:
            return agent_override  # type: ignore[return-value]
        return adapter

    monkeypatch.setattr(llm_resolver, "resolve_llm_adapter", _resolve)
    monkeypatch.setattr(
        "intergrax.runtime.nexus.context.compile_service.compile_prompt_text",
        lambda prompt, config, **_: prompt,
    )
