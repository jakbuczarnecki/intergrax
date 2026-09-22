# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.contracts.external_operation_termination import ExternalOperationCapabilities
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterFactory,
    LLMAdapterRegistrationSpec,
    LLMAdapterRegistrationTarget,
    OptionalDependencyRequirement,
    ProviderExternalOperationSeamFactory,
)

_AdapterCls = TypeVar("_AdapterCls", bound=LLMAdapter)


def lazy_adapter_registration_spec(
    *,
    provider_id: str,
    dependency: OptionalDependencyRequirement | None,
    load_adapter_cls: Callable[[], type[_AdapterCls]],
    external_operation_seam_factory: ProviderExternalOperationSeamFactory | None = None,
    external_operation_capabilities: ExternalOperationCapabilities | None = None,
) -> LLMAdapterRegistrationSpec:
    def factory(**kwargs: object) -> LLMAdapter:
        try:
            adapter_cls = load_adapter_cls()
        except ModuleNotFoundError as exc:
            if dependency is not None and dependency.matches_missing_module(exc):
                raise dependency.to_dependency_error(
                    provider_id=provider_id,
                    exc=exc,
                ) from exc
            raise
        return adapter_cls(**kwargs)

    return LLMAdapterRegistrationSpec(
        provider_id=provider_id,
        factory=factory,
        external_operation_seam_factory=external_operation_seam_factory,
        external_operation_capabilities=external_operation_capabilities,
    )


def register_lazy_adapter(
    registry: LLMAdapterRegistrationTarget,
    *,
    provider_id: str,
    dependency: OptionalDependencyRequirement | None,
    load_adapter_cls: Callable[[], type[_AdapterCls]],
    external_operation_seam_factory: ProviderExternalOperationSeamFactory | None = None,
    external_operation_capabilities: ExternalOperationCapabilities | None = None,
    override: bool = False,
) -> None:
    spec = lazy_adapter_registration_spec(
        provider_id=provider_id,
        dependency=dependency,
        load_adapter_cls=load_adapter_cls,
        external_operation_seam_factory=external_operation_seam_factory,
        external_operation_capabilities=external_operation_capabilities,
    )
    registry.register_from_spec(spec, override=override)
