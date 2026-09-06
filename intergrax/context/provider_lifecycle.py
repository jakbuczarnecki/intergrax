# © Artur Czarnecki. All rights reserved.

"""Context provider lifecycle, snapshots, and execution pinning (P1.9)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.context.contracts import (
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderCollectionOutcome,
    ContextProviderCollectionStatus,
    ContextProviderContext,
    ContextProviderDescriptor,
    ContextProviderProvenance,
    ContextProviderSetSnapshot,
)
from intergrax.context.errors import (
    ContextProviderContractViolationError,
    ContextProviderRegistrationError,
    RequiredContextSourceUnavailableError,
)
from intergrax.context.provider_descriptor import (
    compute_provider_set_fingerprint,
    resolve_provider_descriptor,
)
from intergrax.context.protocols import ContextSourceProvider
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.execution_identity import ExecutionId, peek_active_execution_id

CONTEXT_PROVIDER_BINDING_HANDLE = "context_provider_binding"
CONTEXT_PROVIDER_PINNING_STORE_HANDLE = "context_provider_pinning_store"


def snapshot_context_provider_set(
    registry: ContextPluginRegistry,
    *,
    engine_id: str,
) -> ContextProviderSetSnapshot:
    descriptors = registry.list_provider_descriptors()
    fingerprint = compute_provider_set_fingerprint(descriptors)
    return ContextProviderSetSnapshot(
        engine_id=engine_id,
        providers=descriptors,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class BoundContextProvider:
    descriptor: ContextProviderDescriptor
    provider: ContextSourceProvider


@dataclass(frozen=True, slots=True)
class BoundContextProviderSet:
    snapshot: ContextProviderSetSnapshot
    providers: tuple[BoundContextProvider, ...]


@dataclass(frozen=True, slots=True)
class ContextProviderExecutionBinding:
    tenant_id: str
    execution_id: ExecutionId
    bound_set: BoundContextProviderSet


@runtime_checkable
class ContextProviderExecutionPinningStore(Protocol):
    """Execution-scoped provider semantic binding — descriptors durable, objects in-memory."""

    def pin(self, binding: ContextProviderExecutionBinding) -> None: ...

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ContextProviderExecutionBinding | None: ...


class InMemoryContextProviderExecutionPinningStore:
    """Process-local execution provider pinning for tests and in-memory hosts."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str], ContextProviderExecutionBinding] = {}
        self._lock = threading.Lock()

    def pin(self, binding: ContextProviderExecutionBinding) -> None:
        key = (binding.tenant_id, str(binding.execution_id))
        with self._lock:
            if key in self._bindings and self._bindings[key] != binding:
                raise ContextProviderRegistrationError(
                    f"execution already pinned with different provider set: {binding.execution_id}",
                )
            self._bindings[key] = binding

    def get(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ContextProviderExecutionBinding | None:
        with self._lock:
            return self._bindings.get((tenant_id, str(execution_id)))


def bind_context_provider_set_from_registry(
    registry: ContextPluginRegistry,
    *,
    engine_id: str,
) -> BoundContextProviderSet:
    bound: list[BoundContextProvider] = []
    for provider in registry.list_providers():
        descriptor = resolve_provider_descriptor(provider)
        bound.append(BoundContextProvider(descriptor=descriptor, provider=provider))
    snapshot = snapshot_context_provider_set(registry, engine_id=engine_id)
    return BoundContextProviderSet(snapshot=snapshot, providers=tuple(bound))


def pin_context_provider_set_for_execution(
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    bound_set: BoundContextProviderSet,
    pinning_store: ContextProviderExecutionPinningStore,
) -> ContextProviderExecutionBinding:
    if not tenant_id or tenant_id != tenant_id.strip():
        raise ContextProviderRegistrationError("tenant_id must be non-empty")
    binding = ContextProviderExecutionBinding(
        tenant_id=tenant_id,
        execution_id=execution_id,
        bound_set=bound_set,
    )
    pinning_store.pin(binding)
    return binding


def _pinning_store_from_context(ctx: ContextProviderContext) -> ContextProviderExecutionPinningStore | None:
    store = ctx.handles.get(CONTEXT_PROVIDER_PINNING_STORE_HANDLE)
    if store is None:
        return None
    if not isinstance(store, ContextProviderExecutionPinningStore):
        raise ContextProviderRegistrationError(
            f"{CONTEXT_PROVIDER_PINNING_STORE_HANDLE} must implement ContextProviderExecutionPinningStore",
        )
    return store


def resolve_bound_context_provider_set(
    *,
    registry: ContextPluginRegistry,
    engine_id: str,
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> BoundContextProviderSet:
    explicit = ctx.handles.get(CONTEXT_PROVIDER_BINDING_HANDLE)
    if explicit is not None:
        if not isinstance(explicit, BoundContextProviderSet):
            raise ContextProviderRegistrationError(
                f"{CONTEXT_PROVIDER_BINDING_HANDLE} must be BoundContextProviderSet",
            )
        return explicit

    execution_id = peek_active_execution_id()
    pinning_store = _pinning_store_from_context(ctx)
    if execution_id is not None and pinning_store is not None:
        binding = pinning_store.get(tenant_id=request.tenant_id, execution_id=execution_id)
        if binding is not None:
            return binding.bound_set
        bound_set = bind_context_provider_set_from_registry(registry, engine_id=engine_id)
        pin_context_provider_set_for_execution(
            tenant_id=request.tenant_id,
            execution_id=execution_id,
            bound_set=bound_set,
            pinning_store=pinning_store,
        )
        return bound_set

    return bind_context_provider_set_from_registry(registry, engine_id=engine_id)


def is_provider_eligible(
    descriptor: ContextProviderDescriptor,
    request: ContextAssemblyRequest,
) -> bool:
    supported = descriptor.supported_sources
    if not supported:
        return False
    if supported.issubset(request.excluded_sources):
        return False
    if request.required_sources:
        relevant = supported & request.required_sources
    else:
        relevant = supported - request.excluded_sources
    return bool(relevant)


def provider_covers_required_source(
    descriptor: ContextProviderDescriptor,
    request: ContextAssemblyRequest,
) -> bool:
    if not request.required_sources:
        return False
    return bool(descriptor.supported_sources & request.required_sources)


def validate_required_sources_have_eligible_providers(
    bound_set: BoundContextProviderSet,
    request: ContextAssemblyRequest,
) -> None:
    if not request.required_sources:
        return
    eligible_sources: set[ContextFragmentSource] = set()
    for bound in bound_set.providers:
        if is_provider_eligible(bound.descriptor, request):
            eligible_sources |= bound.descriptor.supported_sources
    for required in request.required_sources:
        if required in request.excluded_sources:
            raise RequiredContextSourceUnavailableError(
                source=required,
                reason_code="required_source.excluded",
            )
        if required not in eligible_sources:
            raise RequiredContextSourceUnavailableError(
                source=required,
                reason_code="required_source.no_provider",
            )


def safe_provider_failure_reason(exc: BaseException) -> str:
    return exc.__class__.__name__


def canonicalize_fragment(
    fragment: ContextFragment,
    *,
    descriptor: ContextProviderDescriptor,
) -> ContextFragment:
    provenance = ContextProviderProvenance.from_descriptor(descriptor)
    if fragment.provider_provenance is not None:
        forged = fragment.provider_provenance
        if (
            forged.provider_id != provenance.provider_id
            or forged.provider_version != provenance.provider_version
        ):
            raise ContextProviderContractViolationError(
                descriptor=descriptor,
                reason_code="provider.contract_violation",
                detail="forged provider provenance",
            )
    if fragment.source not in descriptor.supported_sources:
        raise ContextProviderContractViolationError(
            descriptor=descriptor,
            reason_code="provider.contract_violation",
            detail=f"unsupported source {fragment.source.value}",
        )
    if not fragment.fragment_id.strip():
        raise ContextProviderContractViolationError(
            descriptor=descriptor,
            reason_code="provider.contract_violation",
            detail="fragment_id must be non-empty",
        )
    return ContextFragment(
        fragment_id=fragment.fragment_id,
        source=fragment.source,
        source_id=fragment.source_id,
        content=fragment.content,
        token_estimate=fragment.token_estimate,
        relevance_score=fragment.relevance_score,
        freshness_score=fragment.freshness_score,
        confidence_score=fragment.confidence_score,
        mandatory=fragment.mandatory,
        metadata=fragment.metadata,
        content_hash=fragment.content_hash,
        provider_provenance=provenance,
    )


def assembly_provenance_for_fragment(fragment: ContextFragment) -> ContextAssemblyProvenance:
    provider = fragment.provider_provenance
    return ContextAssemblyProvenance(
        source_type=fragment.source.value,
        source_id=fragment.source_id,
        fragment_id=fragment.fragment_id,
        provider_id=provider.provider_id if provider is not None else "",
        provider_version=provider.provider_version if provider is not None else "",
        provider_origin=provider.origin if provider is not None else "",
        content_hash=fragment.content_hash,
    )


def collection_outcome(
    *,
    descriptor: ContextProviderDescriptor,
    status: ContextProviderCollectionStatus,
    fragment_count: int = 0,
    failure_reason: str = "",
    reason_code: str = "",
) -> ContextProviderCollectionOutcome:
    return ContextProviderCollectionOutcome(
        descriptor=descriptor,
        status=status,
        fragment_count=fragment_count,
        failure_reason=failure_reason,
        reason_code=reason_code,
    )
