# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-bound configured adoption resolution and provenance pin (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_integration_configuration_provenance import (
    ConfiguredIntegrationProvenanceSlice,
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    IntegrationConfigurationSubject,
)
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.contracts.execution_integration_configuration import (
    EffectiveIntegrationIdentity,
    ExecutionIntegrationConfigurationAdoption,
    ExecutionIntegrationConfigurationAdoptionError,
    ExecutionIntegrationConfigurationAdoptionFailureReason,
    IntegrationMaterializationKind,
    validate_configured_adoption_match,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
    ExecutionIntegrationConfigurationPinningStore,
    validate_pin_subject_against_provenance,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.contract_spec import validate_contract_spec_identity
from intergrax.integrations.registry.factory import resolve, resolve_from_profile
from intergrax.runtime.integrations.contract_metadata import CategoryIntegrationInstance
from intergrax.runtime.integrations.contracts import PlatformIntegrationContract


@runtime_checkable
class ExecutionBoundIntegrationMaterializationPort(Protocol):
    """Indirection over sanctioned Integrations resolve entrypoints for tests."""

    def resolve_catalog(
        self,
        category: IntegrationCategory,
        *,
        slug: str,
        profile: IntegrationProfile | None = None,
    ) -> CategoryIntegrationInstance:
        ...

    def resolve_from_profile(
        self,
        profile: IntegrationProfile,
        category: IntegrationCategory,
    ) -> CategoryIntegrationInstance:
        ...


class _DefaultExecutionBoundIntegrationMaterialization(
    ExecutionBoundIntegrationMaterializationPort,
):
    def resolve_catalog(
        self,
        category: IntegrationCategory,
        *,
        slug: str,
        profile: IntegrationProfile | None = None,
    ) -> CategoryIntegrationInstance:
        return resolve(category, slug=slug, profile=profile, config=None)

    def resolve_from_profile(
        self,
        profile: IntegrationProfile,
        category: IntegrationCategory,
    ) -> CategoryIntegrationInstance:
        instance = resolve_from_profile(profile, category, config=None)
        if not isinstance(instance, PlatformIntegrationContract):
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                detail="profile materialization is not PlatformIntegrationContract",
            )
        return instance


@dataclass(frozen=True, slots=True)
class ExecutionBoundIntegrationResolutionRequest:
    tenant_id: str
    execution_id: ExecutionId
    adoption: ExecutionIntegrationConfigurationAdoption
    integration_profile: IntegrationProfile | None = None
    catalog_slug: str | None = None
    resource_scope: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionBoundIntegrationResolutionResult:
    effective: EffectiveIntegrationIdentity
    provenance: ExecutionIntegrationConfigurationProvenance
    subject: IntegrationConfigurationSubject


@dataclass(frozen=True, slots=True)
class ExecutionBoundIntegrationMaterializedResult(ExecutionBoundIntegrationResolutionResult):
    materialized: CategoryIntegrationInstance


class ExecutionBoundIntegrationResolution:
    """Pattern A lifecycle — materialize, validate, pin; retain materialized instance."""

    def __init__(
        self,
        *,
        pinning_store: ExecutionIntegrationConfigurationPinningStore,
        materialization: ExecutionBoundIntegrationMaterializationPort | None = None,
    ) -> None:
        self._pinning_store = pinning_store
        self._materialization = materialization or _DefaultExecutionBoundIntegrationMaterialization()

    def resolve_and_pin(
        self,
        request: ExecutionBoundIntegrationResolutionRequest,
    ) -> ExecutionBoundIntegrationResolutionResult:
        materialized = self.materialize_validate_and_pin(request)
        return ExecutionBoundIntegrationResolutionResult(
            effective=materialized.effective,
            provenance=materialized.provenance,
            subject=materialized.subject,
        )

    def materialize_validate_and_pin(
        self,
        request: ExecutionBoundIntegrationResolutionRequest,
    ) -> ExecutionBoundIntegrationMaterializedResult:
        adoption = request.adoption
        binding = adoption.configured_binding
        category = adoption.integration_category
        if request.tenant_id != binding.tenant_id:
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_TENANT_MISMATCH,
            )
        materialized, profile_used = self._materialize_instance(
            category=category,
            configured_provider_id=binding.provider_id,
            profile=request.integration_profile,
            catalog_slug=request.catalog_slug,
        )
        effective = _effective_identity_from_materialized(
            category=category,
            materialized=materialized,
            profile_used=profile_used,
        )
        scope = request.resource_scope if request.resource_scope is not None else adoption.resource_scope
        validate_configured_adoption_match(
            adoption=adoption,
            effective=effective,
            expected_tenant_id=request.tenant_id,
        )
        if scope != adoption.resource_scope:
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_RESOURCE_SCOPE_MISMATCH,
            )
        configured_slice = ConfiguredIntegrationProvenanceSlice(
            tenant_id=binding.tenant_id,
            integration_category=binding.integration_category,
            provider_id=binding.provider_id,
            resource_scope=binding.resource_scope,
            configuration_type=binding.configuration_type,
            configuration_version=binding.configuration_version,
            configuration_fingerprint=binding.configuration_fingerprint,
            realization_evidence_refs=binding.realization_evidence_refs,
        )
        provenance = ExecutionIntegrationConfigurationProvenance(
            tenant_id=request.tenant_id,
            execution_id=request.execution_id,
            mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED,
            effective=effective,
            configured=configured_slice,
        )
        subject = IntegrationConfigurationSubject(
            integration_category=binding.integration_category,
            provider_id=effective.provider_id,
            resource_scope=binding.resource_scope,
            configuration_type=binding.configuration_type,
        )
        validate_pin_subject_against_provenance(
            subject=subject,
            provenance=provenance,
        )
        try:
            self._pinning_store.pin(subject=subject, provenance=provenance)
        except ExecutionIntegrationConfigurationPinningError:
            raise
        return ExecutionBoundIntegrationMaterializedResult(
            effective=effective,
            provenance=provenance,
            subject=subject,
            materialized=materialized,
        )

    def _materialize_instance(
        self,
        *,
        category: IntegrationCategory,
        configured_provider_id: str,
        profile: IntegrationProfile | None,
        catalog_slug: str | None,
    ) -> tuple[CategoryIntegrationInstance, bool]:
        if category is IntegrationCategory.EXTERNAL_WORK:
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                detail="external_work configured adoption unsupported",
            )
        if profile is not None:
            instance = self._materialization.resolve_from_profile(profile, category)
            return instance, True
        slug = catalog_slug if catalog_slug is not None else configured_provider_id
        materialized = self._materialization.resolve_catalog(
            category,
            slug=slug,
            profile=profile,
        )
        try:
            entry = get_entry(slug)
        except UnknownIntegrationError:
            if materialized.provider_id != configured_provider_id:
                raise ExecutionIntegrationConfigurationAdoptionError(
                    ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                    detail="catalog slug unavailable for configured provider validation",
                ) from None
            return materialized, False
        for spec in entry.contract_specs:
            if spec.category == category:
                validate_contract_spec_identity(
                    slug=slug,
                    spec=spec,
                    observed_provider_id=materialized.provider_id,
                )
                break
        return materialized, False


def _effective_identity_from_materialized(
    *,
    category: IntegrationCategory,
    materialized: PlatformIntegrationContract,
    profile_used: bool,
) -> EffectiveIntegrationIdentity:
    return EffectiveIntegrationIdentity(
        integration_category=category,
        provider_id=materialized.provider_id,
        materialization_kind=(
            IntegrationMaterializationKind.PROFILE_PREBUILT
            if profile_used
            else IntegrationMaterializationKind.CATALOG_FACTORY
        ),
    )


__all__ = [
    "ExecutionBoundIntegrationMaterializationPort",
    "ExecutionBoundIntegrationMaterializedResult",
    "ExecutionBoundIntegrationResolution",
    "ExecutionBoundIntegrationResolutionRequest",
    "ExecutionBoundIntegrationResolutionResult",
]
