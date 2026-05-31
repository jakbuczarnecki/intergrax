# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import (
    default_secret_path_for_provider,
    load_api_key_from_secrets_store,
)

pytestmark = pytest.mark.unit


def test_default_secret_path() -> None:
    assert default_secret_path_for_provider(LLMProvider.GROQ) == "llm/groq/api_key"


def test_load_api_key_from_secrets_store() -> None:
    store = MagicMock()
    store.get_secret.return_value = "vault-key"
    assert load_api_key_from_secrets_store(store, LLMProvider.GROQ) == "vault-key"
    store.get_secret.assert_called_once_with("llm/groq/api_key")


def test_profile_create_adapter_from_secrets_store() -> None:
    store = MagicMock()
    store.get_secret.return_value = "k"
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m")
    with patch(
        "intergrax.llm_adapters.llm_provider_registry.LLMAdapterRegistry.create",
        return_value=MagicMock(),
    ) as create:
        profile.create_adapter_from_secrets_store(store, client=MagicMock())
        assert create.call_args.kwargs.get("api_key") == "k"
