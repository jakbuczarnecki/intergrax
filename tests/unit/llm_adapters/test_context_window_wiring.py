# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.model_catalog import reset_model_catalog_cache
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight


def test_resolve_context_budget_policy_uses_adapter_when_unset() -> None:
    from intergrax.applications._shared.context_wiring import resolve_context_budget_policy
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    adapter = MagicMock()
    adapter.context_window_tokens = 200_000
    env = ApplicationEnvironmentProfile.lab_defaults()
    policy = resolve_context_budget_policy(env, llm_adapter=adapter)
    assert policy.max_tokens_estimate > 4000


def test_llm_profile_context_window_override_propagates_to_claude() -> None:
    reset_model_catalog_cache()
    profile = LLMProfile(
        provider=LLMProvider.CLAUDE,
        model="claude-3-5-sonnet-latest",
        options={"context_window_tokens": 512_000},
    )
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}, clear=False):
        adapter = profile.create_adapter(client=MagicMock())
    assert adapter.context_window_tokens == 512_000


def test_context_budget_policy_from_adapter_uses_window() -> None:
    adapter = MagicMock()
    adapter.context_window_tokens = 200_000
    policy = ContextBudgetPolicy.from_adapter(adapter)
    assert policy.max_tokens_estimate > 4000
    assert policy.max_chars >= policy.max_tokens_estimate


def test_preflight_uses_adapter_token_counter() -> None:
    adapter = MagicMock()
    adapter.context_window_tokens = 200_000
    adapter.count_messages_tokens.return_value = 100
    messages = [ChatMessage(role="user", content="hello")]
    result = verify_context_preflight(messages, adapter)
    assert result.ok is True
    assert result.assembled_tokens == 100
    adapter.count_messages_tokens.assert_called_once_with(messages)
