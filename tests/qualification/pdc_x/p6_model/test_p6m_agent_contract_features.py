# © Artur Czarnecki. All rights reserved.

"""B-P6-M finding 10: an agent's required model features are checked when the adapter is bound."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile
from pydantic import ValidationError
from tests.qualification.pdc_x.xd._support import REPO_ROOT


class _PlainAdapter(BaseLLMAdapter):
    """Buffers only: declares no streaming and no tool streaming."""

    def __init__(self, *, provider: str, model: str | None = None, **_: object) -> None:
        super().__init__()
        self.provider = provider
        self.model = model or f"{provider}-default"
        self.calls: list[str] = []

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def generate_messages(self, messages: Sequence[ChatMessage], **_: object) -> LLMAdapterResponse:
        self.calls.append("generate_messages")
        return LLMAdapterResponse(content="plain", provider=self.provider, model=self.model)

    def generate_with_tools(self, messages, tools, **_: object) -> LLMAdapterResponse:  # type: ignore[no-untyped-def]
        self.calls.append("generate_with_tools")
        return LLMAdapterResponse(content="tools", provider=self.provider, model=self.model)


class _RichAdapter(_PlainAdapter):
    """Honest full declaration: streaming, tools and tool streaming."""

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_streaming(self) -> bool:
        return True

    def stream_messages(self, messages: Sequence[ChatMessage], **_: object) -> Iterable[LLMStreamEvent]:
        yield LLMStreamEvent(
            kind=LLMStreamEventKind.FINAL,
            response=LLMAdapterResponse(content="s", provider=self.provider, model=self.model),
        )

    def stream_with_tools(self, messages, tools, **_: object) -> Iterable[LLMStreamEvent]:  # type: ignore[no-untyped-def]
        yield LLMStreamEvent(
            kind=LLMStreamEventKind.FINAL,
            response=LLMAdapterResponse(content="t", provider=self.provider, model=self.model),
        )


def _register() -> None:
    def rich(**kw: Any) -> _RichAdapter:
        return _RichAdapter(provider="p6m_rich", **kw)

    def plain(**kw: Any) -> _PlainAdapter:
        return _PlainAdapter(provider="p6m_plain", **kw)

    LLMAdapterRegistry.register("p6m_rich", rich, override=True)
    LLMAdapterRegistry.register("p6m_plain", plain, override=True)


def _env(provider: str) -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.model_validate({"llm_profile": LLMProfile(provider=provider, model="m")})


def _contract(**extra: Any) -> AgentContract:
    return AgentContract(id="p6m-agent", name="P6M", description="model binding", **extra)


def test_agent_contract_defaults_to_no_required_llm_features() -> None:
    contract = _contract()
    assert contract.required_llm_features == []
    # The default must not be shared between instances.
    contract.required_llm_features.append(LLMFeature.TOOLS)
    assert _contract().required_llm_features == []


def test_agent_contract_carries_declared_required_llm_features() -> None:
    contract = _contract(required_llm_features=[LLMFeature.TOOL_STREAMING, "vision"])
    assert contract.required_llm_features == [LLMFeature.TOOL_STREAMING, LLMFeature.VISION]


def test_agent_contract_rejects_an_unknown_required_llm_feature() -> None:
    with pytest.raises(ValidationError):
        _contract(required_llm_features=["telepathy"])


def test_contracts_package_imports_first_in_a_fresh_interpreter() -> None:
    """The contract must not import ``intergrax.llm_adapters`` at module load (circular import at package init)."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import intergrax.contracts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_agent_contract_without_the_field_validates_from_legacy_payload() -> None:
    payload = _contract().model_dump(exclude={"required_llm_features"})
    assert "required_llm_features" not in payload
    assert AgentContract.model_validate(payload).required_llm_features == []


def test_resolver_without_required_features_is_unchanged() -> None:
    _register()
    adapter = resolve_llm_adapter(_env("p6m_plain"))
    assert adapter.provider == "p6m_plain"  # type: ignore[attr-defined]
    assert resolve_llm_adapter(_env("p6m_plain"), required_llm_features=()).provider == "p6m_plain"  # type: ignore[attr-defined]


def test_resolver_returns_the_adapter_when_required_features_are_declared() -> None:
    _register()
    contract = _contract(required_llm_features=[LLMFeature.STREAMING, LLMFeature.TOOL_STREAMING])
    adapter = resolve_llm_adapter(_env("p6m_rich"), required_llm_features=contract.required_llm_features)
    assert adapter.provider == "p6m_rich"  # type: ignore[attr-defined]


def test_resolver_fails_closed_at_composition_on_a_capability_mismatch() -> None:
    _register()
    contract = _contract(required_llm_features=[LLMFeature.TOOL_STREAMING])
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        resolve_llm_adapter(_env("p6m_plain"), required_llm_features=contract.required_llm_features)
    assert caught.value.features == (LLMFeature.TOOL_STREAMING,)


def test_resolver_checks_an_agent_override_adapter_too() -> None:
    _register()
    override = _PlainAdapter(provider="override")
    with pytest.raises(LLMCapabilityMismatchError):
        resolve_llm_adapter(None, override, required_llm_features=(LLMFeature.STREAMING,))
    assert override.calls == [], "no provider method was invoked"
    assert resolve_llm_adapter(None, override) is override
    assert resolve_llm_adapter(None, override, required_llm_features=(LLMFeature.TOOLS,)) is override
