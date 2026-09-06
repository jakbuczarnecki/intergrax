# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import (
    LLMAdapterDependencyError,
    LLMAdapterRegistry,
)
from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.registry.catalog_capabilities import (
    unwrap_catalog_capability_adapter,
)
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationSpec,
    OptionalDependencyRequirement,
)


pytestmark = pytest.mark.unit


_Factory = Callable[..., LLMAdapter]


class _TestAdapter(LLMAdapter):
    provider = "unit-test"
    model = "unit-test"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = dict(kwargs)

    @property
    def context_window_tokens(self) -> int:
        return 1

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="ok")


@pytest.fixture()
def _restore_registry_state() -> Iterator[Dict[str, _Factory]]:
    snapshot: Dict[str, _Factory] = dict(LLMAdapterRegistry._factories)
    installed = LLMAdapterRegistry._builtin_registrations_installed
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot
        LLMAdapterRegistry._builtin_registrations_installed = installed


def test_normalize_provider_accepts_enum_values(_restore_registry_state: Dict[str, Any]) -> None:
    key = LLMAdapterRegistry._normalize_provider(LLMProvider.OPENAI)
    assert key == LLMProvider.OPENAI.value


def test_normalize_provider_strips_and_lowercases(_restore_registry_state: Dict[str, Any]) -> None:
    key = LLMAdapterRegistry._normalize_provider("  OpEnAI  ")
    assert key == "openai"


def test_normalize_provider_rejects_empty(_restore_registry_state: Dict[str, Any]) -> None:
    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry._normalize_provider("   ")

    assert "provider must not be empty" in str(exc.value)


def test_register_overwrites_existing_factory(_restore_registry_state: Dict[str, Any]) -> None:
    provider = "unit-test-provider"

    def factory_v1(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(version="v1")

    def factory_v2(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(version="v2")

    LLMAdapterRegistry.register(provider, factory_v1)
    out1 = LLMAdapterRegistry.create(provider)
    assert isinstance(out1, _TestAdapter)
    assert out1.kwargs["version"] == "v1"

    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry.register(provider, factory_v2)

    assert "already registered" in str(exc.value)

    LLMAdapterRegistry.register(provider, factory_v2, override=True)
    out2 = LLMAdapterRegistry.create(provider)
    assert isinstance(out2, _TestAdapter)
    assert out2.kwargs["version"] == "v2"


def test_create_raises_for_unregistered_provider(_restore_registry_state: Dict[str, Any]) -> None:
    LLMAdapterRegistry._factories = {}
    LLMAdapterRegistry._builtin_registrations_installed = True
    with pytest.raises(ValueError) as exc:
        LLMAdapterRegistry.create("missing-provider")

    msg = str(exc.value)
    assert "LLM adapter not registered" in msg
    assert "missing-provider" in msg


def test_create_forwards_kwargs_to_factory(_restore_registry_state: Dict[str, Any]) -> None:
    provider = "unit-test-kwargs"

    def factory(**kwargs: Any) -> LLMAdapter:
        return _TestAdapter(**kwargs)

    LLMAdapterRegistry.register(provider, factory)

    out = LLMAdapterRegistry.create(provider, x=1, y="a")
    assert isinstance(out, _TestAdapter)
    assert out.kwargs == {"x": 1, "y": "a"}


def test_create_rejects_non_adapter_factory(_restore_registry_state: Dict[str, Any]) -> None:
    provider = "invalid-return-type"

    def bad_factory(**kwargs: object) -> object:
        return object()

    LLMAdapterRegistry.register(provider, bad_factory, override=True)
    with pytest.raises(TypeError, match="expected LLMAdapter"):
        LLMAdapterRegistry.create(provider)


def test_ollama_default_resolves_to_native_adapter(
    _restore_registry_state: Dict[str, Any],
) -> None:
    adapter = LLMAdapterRegistry.create(
        LLMProvider.OLLAMA,
        client=object(),
        model="qwen2.5:14b",
    )
    concrete = unwrap_catalog_capability_adapter(adapter)

    assert isinstance(concrete, NativeOllamaAdapter)
    assert not isinstance(concrete, LangChainOllamaAdapter)


def test_normalize_provider_rejects_non_string_and_non_enum(_restore_registry_state: Dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(None)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(42)  # type: ignore[arg-type]

    with pytest.raises((TypeError, ValueError)):
        LLMAdapterRegistry._normalize_provider(object())  # type: ignore[arg-type]


def test_missing_optional_llm_dependency_is_controlled(
    _restore_registry_state: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(self: OptionalDependencyRequirement, *, provider_id: str) -> None:
        raise LLMAdapterDependencyError(
            "LLM provider 'claude' requires dependency 'anthropic'. "
            "Install it with 'Intergrax-ai[llm-anthropic]' before selecting this provider."
        )

    monkeypatch.setattr(OptionalDependencyRequirement, "ensure_available", missing_dependency)

    with pytest.raises(
        LLMAdapterDependencyError,
        match=r"Intergrax-ai\[llm-anthropic\]",
    ):
        LLMAdapterRegistry.create(LLMProvider.CLAUDE, model="claude-3")


def test_register_from_spec_delegates_to_same_storage(
    _restore_registry_state: Dict[str, Any],
) -> None:
    provider = "typed-spec-provider"

    def factory(**kwargs: object) -> LLMAdapter:
        return _TestAdapter(**kwargs)

    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(provider_id=provider, factory=factory)
    )
    out = LLMAdapterRegistry.create(provider)
    assert isinstance(out, _TestAdapter)
