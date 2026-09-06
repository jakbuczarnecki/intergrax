# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.llm_provider_registry import (
    LLMAdapterRegistrationError,
    LLMAdapterRegistry,
)


pytestmark = pytest.mark.unit

_REGISTRY_SOURCE = (
    Path(__file__).resolve().parents[3] / "intergrax" / "llm_adapters" / "llm_provider_registry.py"
)


class _ModelContractAdapter(LLMAdapter):
    provider = "model-contract-test"
    model = "default-model"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if "model" in kwargs:
            self.model = str(kwargs["model"])

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="ok")


class _InvalidModelAdapter(LLMAdapter):
    provider = "invalid-model-contract"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def context_window_tokens(self) -> int:
        return 4096

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
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    installed = LLMAdapterRegistry._builtin_registrations_installed
    try:
        yield
    finally:
        LLMAdapterRegistry._factories = snapshot
        LLMAdapterRegistry._builtin_registrations_installed = installed


def test_valid_adapter_public_model_is_used(_restore_registry_state) -> None:
    provider = "model-contract-test"

    def factory(**kwargs: object) -> LLMAdapter:
        return _ModelContractAdapter(**kwargs)

    LLMAdapterRegistry.register(provider, factory, override=True)
    adapter = LLMAdapterRegistry.create(provider)
    assert adapter.model == "default-model"


def test_kwargs_model_takes_precedence(_restore_registry_state) -> None:
    provider = "model-contract-test"

    def factory(**kwargs: object) -> LLMAdapter:
        return _ModelContractAdapter(**kwargs)

    LLMAdapterRegistry.register(provider, factory, override=True)
    adapter = LLMAdapterRegistry.create(provider, model="kw-model")
    assert adapter.model == "kw-model"


def test_invalid_adapter_missing_model_fails_clearly(_restore_registry_state) -> None:
    provider = "invalid-model-contract"

    def factory(**kwargs: object) -> LLMAdapter:
        return _InvalidModelAdapter(**kwargs)

    LLMAdapterRegistry.register(provider, factory, override=True)
    with pytest.raises(LLMAdapterRegistrationError, match="required public 'model'"):
        LLMAdapterRegistry.create(provider)


def test_registry_model_resolution_has_no_dict_introspection() -> None:
    source = _REGISTRY_SOURCE.read_text(encoding="utf-8")
    assert "adapter.__dict__" not in source
    assert "type(adapter).__dict__" not in source
    assert "getattr(" not in source
    assert "hasattr(" not in source
