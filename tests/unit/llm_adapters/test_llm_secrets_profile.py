# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import resolve_api_key

pytestmark = pytest.mark.unit


def test_resolve_api_key_from_secrets_map() -> None:
    key = resolve_api_key(LLMProvider.GROQ, {"api_key": "secret-key"})
    assert key == "secret-key"


def test_profile_create_adapter_passes_ephemeral_api_key() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m")
    with patch(
        "intergrax.llm_adapters.llm_provider_registry.LLMAdapterRegistry.create"
    ) as create:
        create.return_value = MagicMock()
        profile.create_adapter(secrets={"api_key": "k"}, client=MagicMock())
        assert create.call_args.kwargs.get("api_key") == "k"
