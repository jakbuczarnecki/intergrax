# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for Tier-2 agent unit tests."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.agents.authoring.acp_session_host import ACPSessionHostContext
from intergrax.applications._shared.runtime_boundary_adapters import (
    agent_binding_to_run_binding,
    application_profile_to_runtime_profile,
)
from intergrax.runtime.wiring import llm_resolver
from testing_support.builder import MeteringFakeLLMAdapter


def make_acp_host_context(
    app_profile: Any,
    *,
    binding: Any = None,
    **kwargs: Any,
) -> ACPSessionHostContext:
    """Build host context using application → runtime boundary adapters."""
    return ACPSessionHostContext(
        runtime_profile=application_profile_to_runtime_profile(app_profile),
        binding=agent_binding_to_run_binding(binding),
        **kwargs,
    )


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
