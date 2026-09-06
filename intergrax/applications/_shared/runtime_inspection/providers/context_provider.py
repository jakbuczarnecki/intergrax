# © Artur Czarnecki. All rights reserved.

"""Context provider lifecycle inspection provider (P1.9)."""

from __future__ import annotations

from intergrax.applications.contracts.runtime_inspection.explanation import (
    InspectionExplanation,
    InspectionProvenanceRef,
)
from intergrax.applications.contracts.runtime_inspection.provider import (
    InspectionExtensionEvidence,
    InspectionProviderContribution,
    RuntimeInspectionProvider,
)
from intergrax.applications.contracts.runtime_inspection.scope import InspectionScope
from intergrax.context.contracts import ContextProviderDescriptor
from intergrax.context.provider_lifecycle import ContextProviderExecutionPinningStore
from intergrax.contracts.execution_identity import ExecutionId


_PROVIDER_ID = "context_provider_lifecycle"


def _descriptor_facts(descriptor: ContextProviderDescriptor) -> tuple[str, ...]:
    sources = ",".join(sorted(source.value for source in descriptor.supported_sources))
    return (
        f"provider_id={descriptor.provider_id}",
        f"provider_version={descriptor.provider_version}",
        f"supported_sources={sources}",
        f"origin={descriptor.origin}",
    )


class ContextProviderInspectionProvider:
    """Read-only bound provider-set view for in-flight executions."""

    def __init__(
        self,
        *,
        pinning_store: ContextProviderExecutionPinningStore | None = None,
    ) -> None:
        self._pinning_store = pinning_store

    @property
    def provider_id(self) -> str:
        return _PROVIDER_ID

    def contribute_profile(
        self,
        *,
        resolution: object,
        configured_profile_ref: str | None,
    ) -> InspectionProviderContribution:
        del resolution, configured_profile_ref
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision(
        self,
        *,
        revision_id: object,
        scope: object,
        revision: object | None,
    ) -> InspectionProviderContribution:
        del revision_id, scope, revision
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        scope_application_id: str,
        scope_tenant_id: str | None,
        binding: object | None,
        pinned_revision: object | None,
    ) -> InspectionProviderContribution:
        del binding, pinned_revision, scope_application_id, scope_tenant_id
        if self._pinning_store is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        context_binding = self._pinning_store.get(tenant_id=tenant_id, execution_id=execution_id)
        if context_binding is None:
            return InspectionProviderContribution(provider_id=self.provider_id)
        snapshot = context_binding.bound_set.snapshot
        explanations: list[InspectionExplanation] = []
        extensions: list[InspectionExtensionEvidence] = []
        for descriptor in snapshot.providers:
            explanations.append(
                InspectionExplanation(
                    subject=descriptor.provider_id,
                    facts=_descriptor_facts(descriptor),
                    reasons=("bound context provider",),
                    provenance_refs=(
                        InspectionProvenanceRef(
                            kind="context_provider_descriptor",
                            ref=descriptor.provider_id,
                        ),
                    ),
                ),
            )
        extensions.append(
            InspectionExtensionEvidence(
                provider_id=self.provider_id,
                scope=InspectionScope.EXECUTION,
                subject="context_provider_set",
                payload={
                    "engine_id": snapshot.engine_id,
                    "fingerprint": snapshot.fingerprint,
                    "provider_count": str(len(snapshot.providers)),
                },
            ),
        )
        return InspectionProviderContribution(
            provider_id=self.provider_id,
            explanations=tuple(explanations),
            extension_evidence=tuple(extensions),
        )

    def contribute_capability(
        self,
        *,
        capability_key: str,
        validation: object,
    ) -> InspectionProviderContribution:
        del capability_key, validation
        return InspectionProviderContribution(provider_id=self.provider_id)

    def contribute_revision_compare(
        self,
        *,
        from_revision: object,
        to_revision: object,
    ) -> InspectionProviderContribution:
        del from_revision, to_revision
        return InspectionProviderContribution(provider_id=self.provider_id)


def context_provider_inspection_provider(
    *,
    pinning_store: ContextProviderExecutionPinningStore | None = None,
) -> RuntimeInspectionProvider:
    return ContextProviderInspectionProvider(pinning_store=pinning_store)
