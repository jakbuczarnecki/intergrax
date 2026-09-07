# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing.context_bridge import tokens_used_from_usage


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot


class _StubCustomAdapter(LLMAdapter):
    provider = "custom_gateway"
    model = "custom-model"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.model = str(kwargs.get("model", self.model))

    def generate_messages(self, messages, *, temperature=None, max_tokens=None, run_id=None):
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response

        return build_adapter_response(content="ok")


@pytest.mark.unit
@pytest.mark.gate
def test_tokens_used_from_acp_usage_prefers_agent_then_environment() -> None:
    usage = AcpInvocationUsageView(
        agent=AcpTokenUsage(tokens_total=0),
        environment=AcpTokenUsage(tokens_total=420),
    )
    assert tokens_used_from_usage(usage) == 420

    usage_agent = AcpInvocationUsageView(agent=AcpTokenUsage(tokens_total=150))
    assert tokens_used_from_usage(usage_agent) == 150


@pytest.mark.unit
@pytest.mark.gate
def test_llm_profile_accepts_custom_registered_provider_slug(_restore_registry_state) -> None:
    LLMAdapterRegistry.register(
        "custom_gateway",
        lambda **kwargs: _StubCustomAdapter(**kwargs),
        override=True,
    )
    profile = LLMProfile(provider="custom_gateway", model="custom-model")
    adapter = profile.create_adapter()
    assert isinstance(adapter, LLMAdapter)
    assert adapter.provider == "custom_gateway"


@pytest.mark.unit
@pytest.mark.gate
def test_llm_profile_rejects_unregistered_custom_slug(_restore_registry_state) -> None:
    with pytest.raises(ValueError, match="unknown LLM provider slug"):
        LLMProfile(provider="not_registered_slug", model="x")
