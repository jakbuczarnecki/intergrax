# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence

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
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationSpec,
    OptionalDependencyRequirement,
)


pytestmark = pytest.mark.unit

_REGISTRY_SOURCE = (
    Path(__file__).resolve().parents[3] / "intergrax" / "llm_adapters" / "llm_provider_registry.py"
)

_SDK_ROOTS = (
    "openai",
    "anthropic",
    "ollama",
    "mistralai",
    "boto3",
    "cohere",
    "google.genai",
)

_REPRESENTATIVE_PROVIDERS = (
    (LLMProvider.OPENAI, "openai", "llm-openai"),
    (LLMProvider.GEMINI, "google-genai", "llm-gemini"),
    (LLMProvider.CLAUDE, "anthropic", "llm-anthropic"),
    (LLMProvider.OLLAMA, "ollama", "llm-ollama"),
    (LLMProvider.AWS_BEDROCK, "boto3", "llm-bedrock"),
    (LLMProvider.GROQ, "openai", "llm-groq"),
)


class _FakeEnterpriseGatewayAdapter(LLMAdapter):
    provider = "fake_enterprise_gateway"
    model = "fake-model"

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.model = str(kwargs.get("model", self.model))
        self.validated = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def validate(self) -> None:
        super().validate()
        self.validated = True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="gateway-ok")


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    installed = LLMAdapterRegistry._builtin_registrations_installed
    try:
        yield
    finally:
        LLMAdapterRegistry._factories = snapshot
        LLMAdapterRegistry._builtin_registrations_installed = installed


def _clear_sdk_modules() -> dict[str, object]:
    removed: dict[str, object] = {}
    for name in list(sys.modules):
        if name.split(".", 1)[0] in {root.split(".", 1)[0] for root in _SDK_ROOTS}:
            removed[name] = sys.modules.pop(name)
    return removed


def test_registry_source_has_no_central_builtin_vendor_map() -> None:
    source = _REGISTRY_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "_BUILTIN_ADAPTERS",
        "_BUILTIN_OPTIONAL_DEPENDENCIES",
        "_ensure_builtin",
        "importlib.import_module",
        "attribute_access.optional",
    )
    for token in forbidden:
        assert token not in source


def test_bootstrap_registration_does_not_import_vendor_sdks(
    _restore_registry_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed = _clear_sdk_modules()
    LLMAdapterRegistry._factories = {}
    LLMAdapterRegistry._builtin_registrations_installed = False

    real_import_module = importlib.import_module

    def _fail_sdk_import(name: str, package: str | None = None) -> object:
        root = name.split(".", 1)[0]
        if root in {item.split(".", 1)[0] for item in _SDK_ROOTS}:
            raise ModuleNotFoundError(f"blocked import {name}", name=name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fail_sdk_import)
    try:
        providers = LLMAdapterRegistry.registered_providers()
        assert len(providers) == len(LLMProvider)
        for root in _SDK_ROOTS:
            assert root not in sys.modules
    finally:
        sys.modules.update(removed)


@pytest.mark.parametrize(("provider", "distribution", "extra"), _REPRESENTATIVE_PROVIDERS)
def test_provider_creation_surfaces_dependency_error_without_sdk(
    provider: LLMProvider,
    distribution: str,
    extra: str,
    _restore_registry_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(self: OptionalDependencyRequirement, *, provider_id: str) -> None:
        raise LLMAdapterDependencyError(
            f"LLM provider '{provider_id}' requires dependency '{distribution}'. "
            f"Install it with 'Intergrax-ai[{extra}]' before selecting this provider."
        )

    monkeypatch.setattr(OptionalDependencyRequirement, "ensure_available", missing_dependency)

    with pytest.raises(LLMAdapterDependencyError, match=extra):
        LLMAdapterRegistry.create(provider, model="test-model")


def test_fake_enterprise_gateway_pluginability(_restore_registry_state) -> None:
    provider = "fake_enterprise_gateway"

    def factory(**kwargs: object) -> LLMAdapter:
        return _FakeEnterpriseGatewayAdapter(**kwargs)

    LLMAdapterRegistry.register(
        LLMAdapterRegistrationSpec(provider_id=provider, factory=factory),
        override=True,
    )
    assert provider in LLMAdapterRegistry.registered_providers()

    adapter = LLMAdapterRegistry.create(provider, model="enterprise-model")
    assert isinstance(adapter, _FakeEnterpriseGatewayAdapter)
    assert adapter.model == "enterprise-model"
    assert adapter.validated is True


def test_register_accepts_registration_spec(_restore_registry_state) -> None:
    provider = "fake_enterprise_gateway"

    def factory(**kwargs: object) -> LLMAdapter:
        return _FakeEnterpriseGatewayAdapter(**kwargs)

    LLMAdapterRegistry.register(
        LLMAdapterRegistrationSpec(provider_id=provider, factory=factory),
        override=True,
    )
    adapter = LLMAdapterRegistry.create(provider)
    assert isinstance(adapter, _FakeEnterpriseGatewayAdapter)
