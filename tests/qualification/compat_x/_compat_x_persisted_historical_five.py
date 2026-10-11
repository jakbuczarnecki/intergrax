# © Artur Czarnecki. All rights reserved.

"""Reconciliation of historical P0 `PERSISTED_SCHEMA_WITHOUT_VERSION` five findings."""

from __future__ import annotations

from functools import lru_cache

from tests.qualification.compat_x._compat_x_persisted_types import (
    HistoricalFindingAssessment,
    HistoricalPersistedFindingReconciliation,
)

_HISTORICAL_FIVE: tuple[HistoricalPersistedFindingReconciliation, ...] = (
    HistoricalPersistedFindingReconciliation(
        finding_class="AgentPackageContractAuthorityService",
        module_path="intergrax/agent_distribution/agent_contract_authority.py",
        assessment=HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR,
        actual_persisted_payload_type="PackageAgentContractAuthorityRecord",
        actual_current_version="package_agent_contract_authority.v1",
        evidence_paths=(
            "intergrax/agent_distribution/agent_contract_authority.py",
            "tests/qualification/compat_x/_compat_x_persisted_inventory.py",
        ),
    ),
    HistoricalPersistedFindingReconciliation(
        finding_class="PackageAgentContractAuthorityError",
        module_path="intergrax/agent_distribution/agent_contract_authority.py",
        assessment=HistoricalFindingAssessment.FALSE_POSITIVE_EXCEPTION,
        actual_persisted_payload_type=None,
        actual_current_version=None,
        evidence_paths=("intergrax/agent_distribution/agent_contract_authority.py",),
    ),
    HistoricalPersistedFindingReconciliation(
        finding_class="RuntimeRevisionService",
        module_path="intergrax/agent_distribution/runtime_revision_service.py",
        assessment=HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR,
        actual_persisted_payload_type="RuntimeRevision",
        actual_current_version="runtime_revision.v1",
        evidence_paths=(
            "intergrax/agent_distribution/runtime_revision_service.py",
            "intergrax/agent_distribution/sqlite_stores.py",
        ),
    ),
    HistoricalPersistedFindingReconciliation(
        finding_class="DocumentStoreSuspendedExecutionOperationStore",
        module_path=(
            "intergrax/runtime/execution/suspended_operation/"
            "document_store_suspended_operation_store.py"
        ),
        assessment=HistoricalFindingAssessment.REAL_DURABLE_BOUNDARY_STORE,
        actual_persisted_payload_type="suspended_operation_backing.v1",
        actual_current_version="suspended_operation_backing.v1",
        evidence_paths=(
            "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
        ),
    ),
    HistoricalPersistedFindingReconciliation(
        finding_class="RuntimeEventPublisher",
        module_path="intergrax/runtime/nexus/orchestration/long_running_bridge.py",
        assessment=HistoricalFindingAssessment.FALSE_POSITIVE_PUBLISHER_PROTOCOL,
        actual_persisted_payload_type="RuntimeEvent",
        actual_current_version="runtime_event.v2",
        evidence_paths=(
            "intergrax/runtime/nexus/orchestration/long_running_bridge.py",
            "intergrax/contracts/runtime_event.py",
        ),
    ),
)


@lru_cache(maxsize=1)
def historical_persisted_five_reconciliation() -> tuple[HistoricalPersistedFindingReconciliation, ...]:
    return _HISTORICAL_FIVE


def historical_five_contradiction_count() -> int:
    return sum(
        1
        for row in historical_persisted_five_reconciliation()
        if row.assessment == HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )
