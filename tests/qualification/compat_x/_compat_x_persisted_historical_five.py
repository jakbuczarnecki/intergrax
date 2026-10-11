# © Artur Czarnecki. All rights reserved.

"""Evidence-based reconciliation of historical P0 `PERSISTED_SCHEMA_WITHOUT_VERSION` five."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from tests.qualification.compat_x._compat_x_ast_signals import parse_module
from tests.qualification.compat_x._compat_x_persisted_types import (
    HistoricalFindingAssessment,
    HistoricalPersistedFindingReconciliation,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _module_defines_or_calls_method(module_rel: str, method_name: str) -> bool:
    source = _read(module_rel)
    if f"def {method_name}" in source:
        return True
    tree = parse_module(module_rel, source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == method_name:
                return True
    return False


def _module_serializes_exception(module_rel: str, exception_name: str) -> bool:
    source = _read(module_rel)
    if f"{exception_name}(" not in source:
        return False
    return "model_dump_json" in source and exception_name in source


@lru_cache(maxsize=1)
def historical_persisted_five_reconciliation() -> tuple[HistoricalPersistedFindingReconciliation, ...]:
    authority_mod = "intergrax/agent_distribution/agent_contract_authority.py"
    service_persists = _module_defines_or_calls_method(authority_mod, "persist_authority_record")
    store_persist = "persist_package_contract_authority" in _read(authority_mod)
    authority_assessment = (
        HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR
        if service_persists and store_persist
        else HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )

    error_assessment = (
        HistoricalFindingAssessment.FALSE_POSITIVE_EXCEPTION
        if not _module_serializes_exception(authority_mod, "PackageAgentContractAuthorityError")
        else HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )

    revision_mod = "intergrax/agent_distribution/runtime_revision_service.py"
    revision_assessment = (
        HistoricalFindingAssessment.FALSE_POSITIVE_SERVICE_OR_ORCHESTRATOR
        if _module_defines_or_calls_method(revision_mod, "persist_candidate_revision")
        else HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )

    suspended_mod = (
        "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py"
    )
    suspended_source = _read(suspended_mod)
    suspended_assessment = (
        HistoricalFindingAssessment.REAL_DURABLE_BOUNDARY_STORE
        if "DocumentRecord" in suspended_source and "suspended_operation_backing" in suspended_source
        else HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )

    publisher_mod = "intergrax/runtime/nexus/orchestration/long_running_bridge.py"
    publisher_source = _read(publisher_mod)
    publisher_assessment = (
        HistoricalFindingAssessment.FALSE_POSITIVE_PUBLISHER_PROTOCOL
        if "RuntimeEventPublisher" in publisher_source and "RuntimeEvent" in publisher_source
        else HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )

    return (
        HistoricalPersistedFindingReconciliation(
            finding_class="AgentPackageContractAuthorityService",
            module_path=authority_mod,
            assessment=authority_assessment,
            actual_persisted_payload_type="PackageAgentContractAuthorityRecord",
            actual_current_version="package_agent_contract_authority.v1",
            evidence_paths=(authority_mod, "intergrax/agent_distribution/stores.py"),
        ),
        HistoricalPersistedFindingReconciliation(
            finding_class="PackageAgentContractAuthorityError",
            module_path=authority_mod,
            assessment=error_assessment,
            actual_persisted_payload_type=None,
            actual_current_version=None,
            evidence_paths=(authority_mod,),
        ),
        HistoricalPersistedFindingReconciliation(
            finding_class="RuntimeRevisionService",
            module_path=revision_mod,
            assessment=revision_assessment,
            actual_persisted_payload_type="RuntimeRevision",
            actual_current_version="runtime_revision.v1",
            evidence_paths=(revision_mod, "intergrax/agent_distribution/sqlite_stores.py"),
        ),
        HistoricalPersistedFindingReconciliation(
            finding_class="DocumentStoreSuspendedExecutionOperationStore",
            module_path=suspended_mod,
            assessment=suspended_assessment,
            actual_persisted_payload_type="suspended_operation_backing.v1",
            actual_current_version="suspended_operation_backing.v1",
            evidence_paths=(suspended_mod,),
        ),
        HistoricalPersistedFindingReconciliation(
            finding_class="RuntimeEventPublisher",
            module_path=publisher_mod,
            assessment=publisher_assessment,
            actual_persisted_payload_type="RuntimeEvent",
            actual_current_version="runtime_event.v2",
            evidence_paths=(publisher_mod, "intergrax/contracts/runtime_event.py"),
        ),
    )


def historical_five_contradiction_count() -> int:
    return sum(
        1
        for row in historical_persisted_five_reconciliation()
        if row.assessment == HistoricalFindingAssessment.PRE_AUDIT_CONTRADICTION
    )
