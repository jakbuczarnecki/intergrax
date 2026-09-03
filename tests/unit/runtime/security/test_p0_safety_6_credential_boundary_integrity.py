# © Artur Czarnecki. All rights reserved.

"""P0-SAFETY-6 — credential boundary integrity conformance."""

from __future__ import annotations

import json
from typing import Mapping
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.secrets import (
    default_secret_path_for_provider,
    load_api_key_from_secrets_store,
    merge_secrets_into_options,
)
from intergrax.tools.providers.platform.contracts import PlatformGetSecretInput
from intergrax.tools.providers.platform.service import platform_get_secret
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SENTINEL = "SUPER_SECRET_SENTINEL_9f3a2b1c0d4e5f678901234567890abcd"


class _FakeSecretsStore:
    """Provider-neutral in-memory SecretsStore for boundary tests."""

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.lookup_paths: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.lookup_paths.append(path.strip())
        return self.values[path.strip()]

    def put_secret(self, path: str, value: str) -> None:
        self.values[path.strip()] = value

    def delete_secret(self, path: str) -> None:
        self.values.pop(path.strip(), None)


def _assert_sentinel_absent(payload: object) -> None:
    text = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    assert _SENTINEL not in text


def test_sentinel_absent_from_profile_serialization_with_secrets() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m").with_secrets({"api_key": _SENTINEL})
    _assert_sentinel_absent(profile.model_dump())
    _assert_sentinel_absent(profile.model_dump_json())
    _assert_sentinel_absent(repr(profile))


def test_sentinel_relocated_from_options_before_serialization() -> None:
    profile = LLMProfile(
        provider=LLMProvider.GROQ,
        model="m",
        options={"api_key": _SENTINEL, "max_retries": 1},
    )
    _assert_sentinel_absent(profile.model_dump())
    assert "api_key" not in profile.options


def test_sentinel_absent_from_validate_runtime_diagnostics() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m").with_secrets({"api_key": _SENTINEL})
    warnings = profile.validate_runtime()
    _assert_sentinel_absent(warnings)


def test_sentinel_absent_from_secrets_store_loader_errors() -> None:
    store = _FakeSecretsStore({"llm/groq/api_key": ""})
    with pytest.raises(RuntimeError, match="Empty secret"):
        load_api_key_from_secrets_store(store, LLMProvider.GROQ)
    _assert_sentinel_absent(str(store.lookup_paths))


def test_provider_receives_resolved_secret_at_adapter_boundary() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m").with_secrets({"api_key": _SENTINEL})
    with patch(
        "intergrax.llm_adapters.llm_provider_registry.LLMAdapterRegistry.create",
        return_value=MagicMock(),
    ) as create:
        profile.create_adapter(client=MagicMock())
        assert create.call_args.kwargs.get("api_key") == _SENTINEL


def test_create_adapter_from_secrets_store_passes_secret_without_durable_profile_state() -> None:
    store = _FakeSecretsStore({default_secret_path_for_provider(LLMProvider.GROQ): _SENTINEL})
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m")
    with patch(
        "intergrax.llm_adapters.llm_provider_registry.LLMAdapterRegistry.create",
        return_value=MagicMock(),
    ) as create:
        profile.create_adapter_from_secrets_store(store, client=MagicMock())
        assert create.call_args.kwargs.get("api_key") == _SENTINEL
    _assert_sentinel_absent(profile.model_dump())


def test_missing_secret_fails_closed_before_provider_operation() -> None:
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m")
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            profile.create_adapter()


def test_provider_neutral_secrets_store_without_vault_behavior() -> None:
    store: SecretsStore = _FakeSecretsStore({"llm/groq/api_key": "neutral-key"})
    assert load_api_key_from_secrets_store(store, LLMProvider.GROQ) == "neutral-key"


def test_default_secret_path_is_platform_defined() -> None:
    store = _FakeSecretsStore({default_secret_path_for_provider(LLMProvider.GROQ): "k"})
    profile = LLMProfile(provider=LLMProvider.GROQ, model="m")
    with patch(
        "intergrax.llm_adapters.llm_provider_registry.LLMAdapterRegistry.create",
        return_value=MagicMock(),
    ):
        profile.create_adapter_from_secrets_store(store, client=MagicMock())
    assert store.lookup_paths == [default_secret_path_for_provider(LLMProvider.GROQ)]


def test_merge_secrets_into_options_keeps_secret_in_constructor_kwargs_only() -> None:
    merged = merge_secrets_into_options(LLMProvider.GROQ, {"max_retries": 1}, {"api_key": _SENTINEL})
    assert merged["api_key"] == _SENTINEL
    assert merged["max_retries"] == 1


def test_non_llm_secrets_store_consumer_uses_tenant_scoped_path() -> None:
    store = _FakeSecretsStore({"tenant/demo/token": _SENTINEL})
    ctx = ToolWiringContext(secrets_store=store)
    out = platform_get_secret(ctx, PlatformGetSecretInput(path="tenant/demo/token"))
    assert out.value == _SENTINEL
    assert store.lookup_paths == ["tenant/demo/token"]
