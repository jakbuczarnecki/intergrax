# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lazy Pattern A execution-bound relational port (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    ExecutionIntegrationConfigurationProvenanceRequirementCommitPort,
    ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)
from intergrax.integrations.execution_integration_configuration_requirement_fact import (
    build_requirement_fact_from_pin_record,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.configured_relational_store_execution import (
    ConfiguredRelationalStoreExecutionPort,
    RelationalExecuteRequest,
    RelationalExecuteResult,
    RelationalQueryRequest,
    RelationalQueryResult,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
    ExecutionIntegrationConfigurationAdoptionError,
    ExecutionIntegrationConfigurationAdoptionFailureReason,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.execution_bound_integration_resolution import (
    ExecutionBoundIntegrationResolution,
    ExecutionBoundIntegrationResolutionRequest,
)
from intergrax.integrations.relational_store_execution_adapter import (
    RelationalStoreExecutionAdapter,
)
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract


class ExecutionBoundConfiguredRelationalStorePort(ConfiguredRelationalStoreExecutionPort):
    """One logical invocation — at most one materialized provider."""

    __slots__ = (
        "_adoption",
        "_catalog_slug",
        "_execution_id",
        "_init_failed",
        "_init_lock",
        "_integration_profile",
        "_requirement_commit_port",
        "_requirement_recovery_staging",
        "_resolution",
        "_tenant_id",
        "_typed_adapter",
    )

    def __init__(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
        resolution: ExecutionBoundIntegrationResolution,
        integration_profile: IntegrationProfile | None = None,
        catalog_slug: str | None = None,
        requirement_recovery_staging: (
            ExecutionIntegrationConfigurationRequirementRecoveryStaging | None
        ) = None,
        requirement_commit_port: (
            ExecutionIntegrationConfigurationProvenanceRequirementCommitPort | None
        ) = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._adoption = adoption
        self._resolution = resolution
        self._integration_profile = integration_profile
        self._catalog_slug = catalog_slug
        self._requirement_recovery_staging = requirement_recovery_staging
        self._requirement_commit_port = requirement_commit_port
        self._typed_adapter: RelationalStoreExecutionAdapter | None = None
        self._init_failed = False
        self._init_lock = threading.Lock()

    def query(self, request: RelationalQueryRequest) -> RelationalQueryResult:
        return self._adapter().query(request)

    def execute(self, request: RelationalExecuteRequest) -> RelationalExecuteResult:
        return self._adapter().execute(request)

    def _adapter(self) -> RelationalStoreExecutionAdapter:
        cached = self._typed_adapter
        if cached is not None:
            return cached
        with self._init_lock:
            if self._init_failed:
                raise ExecutionIntegrationConfigurationAdoptionError(
                    ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                    detail="configured relational port initialization failed",
                )
            if self._typed_adapter is not None:
                return self._typed_adapter
            try:
                self._typed_adapter = self._initialize_adapter()
            except Exception:
                self._init_failed = True
                raise
            return self._typed_adapter

    def _initialize_adapter(self) -> RelationalStoreExecutionAdapter:
        adoption = self._adoption
        if adoption.integration_category is not IntegrationCategory.RELATIONAL_STORE:
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.CONFIGURED_ADOPTION_CATEGORY_MISMATCH,
            )
        materialized_result = self._resolution.materialize_validate_and_pin(
            ExecutionBoundIntegrationResolutionRequest(
                tenant_id=self._tenant_id,
                execution_id=self._execution_id,
                adoption=adoption,
                integration_profile=self._integration_profile,
                catalog_slug=self._catalog_slug,
                requirement_recovery_staging=self._requirement_recovery_staging,
            ),
        )
        if self._requirement_commit_port is not None:
            pin_record = self._resolution.read_pin_record_for_subject(
                tenant_id=self._tenant_id,
                execution_id=self._execution_id,
                subject=materialized_result.subject,
            )
            if pin_record is None:
                raise ExecutionIntegrationConfigurationAdoptionError(
                    ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                    detail="configured pin missing after materialize",
                )
            fact = build_requirement_fact_from_pin_record(pin_record)
            commit_result = self._requirement_commit_port.commit_configured_adopted_requirement(
                fact,
            )
            if (
                commit_result.status
                is not ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus.COMMITTED
            ):
                raise ExecutionIntegrationConfigurationAdoptionError(
                    ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                    detail="requirement spine commit unavailable",
                )
        materialized = materialized_result.materialized
        if not isinstance(materialized, RelationalStoreIntegrationContract):
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                detail="materialized provider is not RelationalStoreIntegrationContract",
            )
        if not isinstance(materialized, RelationalStore):
            raise ExecutionIntegrationConfigurationAdoptionError(
                ExecutionIntegrationConfigurationAdoptionFailureReason.EFFECTIVE_PROVIDER_IDENTITY_UNAVAILABLE,
                detail="materialized provider does not satisfy RelationalStore",
            )
        return RelationalStoreExecutionAdapter(materialized)


__all__ = ["ExecutionBoundConfiguredRelationalStorePort"]
