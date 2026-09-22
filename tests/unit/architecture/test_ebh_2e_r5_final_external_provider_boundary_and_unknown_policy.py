# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R5 — external provider pluginability & unknown-provider policy gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.external_operation_identity import ExternalOperationIdentity
from intergrax.contracts.external_operation_termination import (
    ExternalOperationCapabilities,
    TerminationResult,
)
from intergrax.llm_adapters._shared.default_external_operation_seam import (
    NoOpExternalOperationCancellationPort,
    NoOpExternalOperationTerminationPort,
    default_external_operation_seam,
    unregistered_external_operation_capabilities,
)
from intergrax.llm_adapters._shared.provider_stream_transport_registry import (
    ProviderStreamTransportRegistry,
)
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterRegistrationSpec,
    ProviderExternalOperationSeam,
)
from intergrax.runtime.external_operations.provider_cancellation import (
    resolve_llm_provider_external_operation_seam,
)
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
)
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_PATH = _REPO_ROOT / "intergrax/llm_adapters/llm_provider_registry.py"
_UNKNOWN_SLUG = "totally-unknown-provider-for-r5"
_EXTERNAL_PROVIDER_ID = "test-external-provider"


class _TestExternalProviderAdapter(ExternalStructuralAdapter):
    provider = _EXTERNAL_PROVIDER_ID


def _external_factory(**_kwargs: object) -> _TestExternalProviderAdapter:
    return _TestExternalProviderAdapter()


class _ManagedCancelPort:
    async def request_cancel(self, operation_id: str) -> None:
        if type(operation_id) is not str or not operation_id:
            raise ValueError("operation_id must be a non-empty str")


class _ManagedTermPort:
    async def terminate(self, identity: ExternalOperationIdentity) -> TerminationResult:
        _ = identity
        return TerminationResult.not_supported()


def _managed_seam_factory() -> ProviderExternalOperationSeam:
    return ProviderExternalOperationSeam(
        cancellation=_ManagedCancelPort(),
        termination=_ManagedTermPort(),
        stream_registry=ProviderStreamTransportRegistry(),
    )


def _managed_caps() -> ExternalOperationCapabilities:
    return ExternalOperationCapabilities(
        supports_native_cancel=True,
        supports_stream_abort=False,
        supports_remote_termination=True,
    )


def test_ebh_2e_r5_external_provider_not_in_llm_provider_enum() -> None:
    builtin_values = {member.value for member in LLMProvider}
    assert _EXTERNAL_PROVIDER_ID not in builtin_values


def test_ebh_2e_r5_external_execution_only_provider_e2e() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=_EXTERNAL_PROVIDER_ID,
            factory=_external_factory,
        )
    )
    adapter = LLMAdapterRegistry.create(_EXTERNAL_PROVIDER_ID, model="proof-model")
    assert isinstance(adapter, LLMAdapter)
    assert isinstance(adapter, _TestExternalProviderAdapter)
    response = adapter.generate_messages([])
    assert response.content == "external"
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r5_external_lifecycle_capable_provider_e2e() -> None:
    LLMAdapterRegistry.reset_for_testing()
    managed_id = "test-external-provider-managed"
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=managed_id,
            factory=_external_factory,
            external_operation_seam_factory=_managed_seam_factory,
            external_operation_capabilities=_managed_caps(),
        )
    )
    cancel, termination, registry = resolve_llm_provider_external_operation_seam(managed_id)
    assert type(cancel) is _ManagedCancelPort
    assert type(termination) is _ManagedTermPort
    assert registry is not None
    assert LLMAdapterRegistry.external_operation_capabilities_for(managed_id) == _managed_caps()
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r5_unknown_create_fails_closed() -> None:
    LLMAdapterRegistry.reset_for_testing()
    with pytest.raises(ValueError, match="not registered"):
        LLMAdapterRegistry.create(_UNKNOWN_SLUG)
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r5_unknown_seam_is_neutral_no_op() -> None:
    LLMAdapterRegistry.reset_for_testing()
    cancel, termination, _registry = resolve_llm_provider_external_operation_seam(_UNKNOWN_SLUG)
    reference = default_external_operation_seam()
    assert type(cancel) is type(reference.cancellation)
    assert type(termination) is type(reference.termination)
    assert isinstance(cancel, NoOpExternalOperationCancellationPort)
    assert isinstance(termination, NoOpExternalOperationTerminationPort)


def test_ebh_2e_r5_unknown_capabilities_all_false() -> None:
    LLMAdapterRegistry.reset_for_testing()
    caps = LLMAdapterRegistry.external_operation_capabilities_for(_UNKNOWN_SLUG)
    assert caps == unregistered_external_operation_capabilities()
    assert caps.supports_native_cancel is False
    assert caps.supports_stream_abort is False
    assert caps.supports_remote_termination is False


def test_ebh_2e_r5_unknown_lookups_do_not_register_unknown_slug() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=_EXTERNAL_PROVIDER_ID,
            factory=_external_factory,
        )
    )
    custom_factory = LLMAdapterRegistry._factories[_EXTERNAL_PROVIDER_ID]
    resolve_llm_provider_external_operation_seam(_UNKNOWN_SLUG)
    LLMAdapterRegistry.external_operation_capabilities_for(_UNKNOWN_SLUG)
    LLMAdapterRegistry.resolve_external_operation_seam(_UNKNOWN_SLUG)
    assert _UNKNOWN_SLUG not in LLMAdapterRegistry._factories
    assert _UNKNOWN_SLUG not in LLMAdapterRegistry._external_operation_seam_factories
    assert _UNKNOWN_SLUG not in LLMAdapterRegistry._external_operation_capabilities
    assert LLMAdapterRegistry._factories.get(_EXTERNAL_PROVIDER_ID) is custom_factory


def test_ebh_2e_r5_custom_registration_survives_builtin_bootstrap() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.register_from_spec(
        LLMAdapterRegistrationSpec(
            provider_id=_EXTERNAL_PROVIDER_ID,
            factory=_external_factory,
        )
    )
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    assert _EXTERNAL_PROVIDER_ID in LLMAdapterRegistry.registered_providers()
    adapter = LLMAdapterRegistry.create(_EXTERNAL_PROVIDER_ID, model="retained")
    assert isinstance(adapter, LLMAdapter)
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r5_builtin_collision_requires_explicit_override() -> None:
    LLMAdapterRegistry.reset_for_testing()
    LLMAdapterRegistry.ensure_builtin_registrations_installed()
    with pytest.raises(ValueError, match="already registered"):
        LLMAdapterRegistry.register("openai", _external_factory)
    LLMAdapterRegistry.reset_for_testing()


def test_ebh_2e_r5_create_has_no_openai_compat_unknown_fallback() -> None:
    source = _REGISTRY_PATH.read_text(encoding="utf-8-sig")
    create_body = _extract_method_source(source, "create")
    lowered = create_body.lower()
    assert "openai_compat" not in lowered
    assert "create_openai_compat" not in lowered
    assert "openai" not in lowered.replace("llmprovider", "")


def test_ebh_2e_r5_registry_create_fail_closed_on_missing_factory() -> None:
    source = _REGISTRY_PATH.read_text(encoding="utf-8-sig")
    create_body = _extract_method_source(source, "create")
    assert "not registered" in create_body
    assert "key not in cls._factories" in create_body


def _extract_method_source(module_source: str, method_name: str) -> str:
    tree = ast.parse(module_source)
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMAdapterRegistry"
    )
    method = next(
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    start = method.lineno - 1
    end = method.end_lineno or start + 1
    lines = module_source.splitlines()
    return "\n".join(lines[start:end])


def test_ebh_2e_r5_structural_adapter_does_not_inherit_base_llm_adapter() -> None:
    assert not issubclass(ExternalStructuralAdapter, BaseLLMAdapter)
