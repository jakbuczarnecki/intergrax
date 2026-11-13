# © Artur Czarnecki. All rights reserved.

"""PDC-D3-R011 — cancel outcome is typed per provider (terminated / transport_closed / not_supported)."""

from __future__ import annotations

import asyncio

import pytest
from intergrax.contracts.dependency_concurrency_admission import DependencyConcurrencyKind
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)
from intergrax.llm_adapters.contracts.capabilities import (
    CancelCapability,
    cancel_capability_from_external_operation,
    cancel_outcome_of,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from tests.qualification.pdc_x.d3.fakes import StreamingToolsAdapter

pytestmark = pytest.mark.unit

#: Providers whose registration declares a physical cancellation seam.
_WITH_SEAM = {
    LLMProvider.OPENAI: CancelCapability.TERMINATED,
    LLMProvider.AZURE_OPENAI: CancelCapability.TERMINATED,
    LLMProvider.GROQ: CancelCapability.TERMINATED,
    LLMProvider.CLAUDE: CancelCapability.TERMINATED,
    LLMProvider.MISTRAL: CancelCapability.TERMINATED,
    LLMProvider.GEMINI: CancelCapability.TERMINATED,
    LLMProvider.VERTEX_GEMINI: CancelCapability.TERMINATED,
    LLMProvider.AWS_BEDROCK: CancelCapability.TRANSPORT_CLOSED,
    LLMProvider.OLLAMA: CancelCapability.TRANSPORT_CLOSED,
}


def _identity(provider: str, operation_id: str) -> ExternalOperationIdentity:
    return ExternalOperationIdentity(
        execution_id=mint_execution_id(),
        attempt_id=mint_attempt_id(),
        dependency_kind=DependencyConcurrencyKind.LLM_PROVIDER,
        dependency_identity=provider,
        operation_id=operation_id,
        tenant_id="tenant-d3",
    )


def _terminate_with_open_stream(provider: LLMProvider) -> tuple[TerminationResult, list[str]]:
    seam = LLMAdapterRegistry.resolve_external_operation_seam(provider)
    closed: list[str] = []
    seam.stream_registry.register("op-1", lambda: closed.append("closed"))
    result = asyncio.run(seam.termination.terminate(_identity(provider.value, "op-1")))
    return result, closed


@pytest.mark.parametrize("provider", sorted(_WITH_SEAM, key=lambda p: p.value), ids=lambda p: p.value)
def test_declared_cancel_capability_equals_observed_outcome(provider: LLMProvider) -> None:
    declared = LLMAdapterRegistry.cancel_capability_for(provider)
    assert declared is _WITH_SEAM[provider]

    result, closed = _terminate_with_open_stream(provider)
    assert closed == ["closed"], "the in-flight stream transport must really be closed"
    assert cancel_outcome_of(result) is declared


@pytest.mark.parametrize("provider", sorted(_WITH_SEAM, key=lambda p: p.value), ids=lambda p: p.value)
def test_cancel_without_an_open_stream_is_typed_not_supported(provider: LLMProvider) -> None:
    seam = LLMAdapterRegistry.resolve_external_operation_seam(provider)
    result = asyncio.run(seam.termination.terminate(_identity(provider.value, "unknown-op")))
    assert cancel_outcome_of(result) is CancelCapability.NOT_SUPPORTED


def test_provider_without_a_registered_seam_declares_not_supported() -> None:
    assert LLMAdapterRegistry.cancel_capability_for("no-such-provider") is CancelCapability.NOT_SUPPORTED
    seam = LLMAdapterRegistry.resolve_external_operation_seam("no-such-provider")
    result = asyncio.run(seam.termination.terminate(_identity("no-such-provider", "op-x")))
    assert cancel_outcome_of(result) is CancelCapability.NOT_SUPPORTED


def test_every_registered_provider_has_a_typed_cancel_declaration() -> None:
    for slug in LLMAdapterRegistry.registered_providers():
        capability = LLMAdapterRegistry.cancel_capability_for(slug)
        assert isinstance(capability, CancelCapability)
    # Providers that registered a seam must not be silently typed as not_supported.
    for provider in _WITH_SEAM:
        assert LLMAdapterRegistry.cancel_capability_for(provider) is not CancelCapability.NOT_SUPPORTED


def test_cancel_capability_is_read_through_the_registry_for_an_adapter() -> None:
    declared = LLMAdapterRegistry.declared_capabilities_for(StreamingToolsAdapter(provider=LLMProvider.OLLAMA.value))
    assert declared.cancel is CancelCapability.TRANSPORT_CLOSED
    unknown = LLMAdapterRegistry.declared_capabilities_for(StreamingToolsAdapter(provider="acme"))
    assert unknown.cancel is CancelCapability.NOT_SUPPORTED


def test_capability_mapping_rule() -> None:
    def caps(native: bool, abort: bool, remote: bool) -> ExternalOperationCapabilities:
        return ExternalOperationCapabilities(
            supports_native_cancel=native, supports_stream_abort=abort, supports_remote_termination=remote
        )

    assert cancel_capability_from_external_operation(caps(True, True, True)) is CancelCapability.TERMINATED
    assert cancel_capability_from_external_operation(caps(False, True, False)) is CancelCapability.TRANSPORT_CLOSED
    assert cancel_capability_from_external_operation(caps(False, False, False)) is CancelCapability.NOT_SUPPORTED


def test_signal_failure_is_not_a_cancel_outcome() -> None:
    with pytest.raises(ValueError):
        cancel_outcome_of(TerminationResult.signal_failed())
