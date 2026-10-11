# © Artur Czarnecki. All rights reserved.

"""Classification layer: raw durable candidates → family member or evidence-backed exclusion."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from tests.qualification.compat_x._compat_x_persisted_mechanical_scan import (
    discover_agent_distribution_payload_tables_from_sql,
)
from tests.qualification.compat_x._compat_x_persisted_types import (
    DurableBoundaryCandidate,
    DurableCandidateClassification,
    DurableExclusionReason,
    DurableStorageSignal,
    PersistedCandidateDisposition,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Classification metadata (not discovery universe): module anchors → semantic family.
_MODULE_FAMILY_ANCHOR: dict[str, str] = {
    "intergrax/agent_distribution/sqlite_stores.py": "agent_distribution.sqlite_container",
    "intergrax/agent_distribution/in_memory_stores.py": "agent_distribution.sqlite_container",
    "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py": (
        "runtime.suspended_operation_backing_envelope"
    ),
    "intergrax/runtime/events/stores/sqlite_runtime_event_store.py": "runtime.runtime_event",
    "intergrax/runtime/long_running/store.py": "runtime.scheduled_resume",
    "intergrax/runtime/long_running/checkpoint_resume_validation.py": "runtime.task_checkpoint",
    "intergrax/runtime/long_running/runtime_checkpoint.py": "runtime.runtime_checkpoint",
    "intergrax/runtime/long_running/models.py": "runtime.task_checkpoint",
    "intergrax/runtime/events/stores/document_backed_runtime_event_store.py": "runtime.runtime_event",
    "intergrax/applications/_shared/registry_snapshot_store.py": "applications.registry_snapshot",
    "intergrax/applications/_shared/registry_snapshot_persistence.py": "applications.registry_snapshot",
    "intergrax/memory/conversational_memory_persistence.py": "memory.conversational_turn",
    "intergrax/runtime/task_memory/stores/sqlite_task_memory_store.py": "runtime.task_memory_record",
    "intergrax/memory/stores/sqlite_user_profile_store.py": "memory.user_profile",
    "intergrax/memory/stores/sqlite_conversational_memory_store.py": "memory.conversational_turn",
    "intergrax/memory/stores/document_store_user_profile_store.py": "memory.user_profile",
    "intergrax/distributed/providers/sqlite_kv_store.py": "distributed.kv_entry",
    "intergrax/distributed/providers/redis_kv_store.py": "distributed.kv_entry",
    "intergrax/runtime/tools/sqlite_idempotency_store.py": "runtime.tool_idempotency_ledger",
    "intergrax/distributed/providers/redis_idempotency_store.py": "runtime.tool_idempotency_ledger",
    "intergrax/runtime/execution/continuation/durable_state_file.py": (
        "runtime.execution_continuation_durable_state"
    ),
    "intergrax/runtime/execution/continuation/persistence.py": (
        "runtime.execution_continuation_durable_state"
    ),
    "synthetic/qualification/compat_x_r2_redis_classified.py": "distributed.kv_entry",
    "synthetic/qualification/compat_x_r2_cache_false_exclude.py": "runtime.tool_idempotency_ledger",
    "intergrax/runtime/execution/sqlite_decision_event_append_persistence.py": "runtime.decision_event_append",
    "intergrax/autonomous_work/postgresql_repository.py": "autonomous_work.postgresql_container",
    "intergrax/collaborative_work/sqlite_repository.py": "collaborative_work.sqlite_container",
}

# SQLite payload table → semantic family (classification after mechanical table discovery).
_AD_TABLE_FAMILY: dict[str, str] = {
    "installations": "agent_distribution.agent_installation_record",
    "bindings": "agent_distribution.application_agent_binding",
    "revisions": "agent_distribution.runtime_revision",
    "artifact_metadata": "agent_distribution.agent_artifact_metadata",
    "runtime_locks": "agent_distribution.materialized_runtime_lock",
    "materializations": "agent_distribution.runtime_materialization_record",
    "effective_roster_snapshots": "agent_distribution.effective_roster",
    "deployment_instances": "agent_distribution.deployment_instance_record",
    "serving_records": "agent_distribution.application_environment_serving_record",
    "projection_descriptors": "agent_distribution.runtime_registry_projection_descriptor",
    "package_contract_authority": "agent_distribution.package_agent_contract_authority_record",
    "conversational_memory": "memory.conversational_turn",
}

_SQLITE_PAYLOAD_TABLE_FAMILY: dict[str, str] = {
    "registry_snapshots": "applications.registry_snapshot",
}

_SQLITE_SCHEMA_TABLE_FAMILY: dict[str, str] = {
    "task_checkpoints": "runtime.task_checkpoint",
    "scheduled_resumes": "runtime.scheduled_resume",
    "scheduler_ledger": "runtime.scheduled_resume",
    "task_memory": "runtime.task_memory_record",
    "distributed_kv_entries": "distributed.kv_entry",
    "user_profiles": "memory.user_profile",
    "runtime_events": "runtime.runtime_event",
    "runtime_event_run_sequences": "runtime.runtime_event",
    "task_execution_terminal": "runtime.task_checkpoint",
    "idempotency_ledger": "runtime.tool_idempotency_ledger",
}

# Model symbol on read/decode → family (sqlite_stores decode paths).
_MODEL_SYMBOL_FAMILY: dict[str, str] = {
    "SuspendedExecutionOperationDescriptor": "runtime.suspended_execution_operation_descriptor",
    "RuntimeEvent": "runtime.runtime_event",
    "TaskCheckpoint": "runtime.task_checkpoint",
    "RuntimeCheckpoint": "runtime.runtime_checkpoint",
    "ScheduledResume": "runtime.scheduled_resume",
    "TaskMemoryRecord": "runtime.task_memory_record",
    "UserProfile": "memory.user_profile",
    "ConversationalMemoryRecord": "memory.conversational_turn",
    "ConversationalMemoryPersistedPayload": "memory.conversational_turn",
    "RegistrySnapshotPersistedPayload": "applications.registry_snapshot",
    "RegistrySnapshot": "applications.registry_snapshot",
    "AgentInstallationRecord": "agent_distribution.agent_installation_record",
    "ApplicationAgentBinding": "agent_distribution.application_agent_binding",
    "RuntimeRevision": "agent_distribution.runtime_revision",
    "AgentArtifactMetadata": "agent_distribution.agent_artifact_metadata",
    "MaterializedRuntimeLock": "agent_distribution.materialized_runtime_lock",
    "RuntimeMaterializationRecord": "agent_distribution.runtime_materialization_record",
    "EffectiveRoster": "agent_distribution.effective_roster",
    "DeploymentInstanceRecord": "agent_distribution.deployment_instance_record",
    "ApplicationEnvironmentServingRecord": "agent_distribution.application_environment_serving_record",
    "RuntimeRegistryProjectionDescriptor": "agent_distribution.runtime_registry_projection_descriptor",
    "PackageAgentContractAuthorityRecord": "agent_distribution.package_agent_contract_authority_record",
}

_SERVICE_MODULE_MARKERS: tuple[str, ...] = (
    "_service.py",
    "runtime_revision_service.py",
    "agent_contract_authority.py",
    "admin_service.py",
)

_ORCHESTRATOR_CLASS_SUFFIXES: tuple[str, ...] = ("Service", "Publisher", "Orchestrator")

_PERSISTENCE_CRITICAL_PREFIXES: tuple[str, ...] = (
    "intergrax/agent_distribution/",
    "intergrax/runtime/events/",
    "intergrax/runtime/execution/",
    "intergrax/runtime/long_running/",
    "intergrax/runtime/task_memory/",
    "intergrax/runtime/tools/",
    "intergrax/memory/stores/",
    "intergrax/distributed/providers/",
    "intergrax/applications/_shared/",
    "intergrax/autonomous_work/",
    "intergrax/collaborative_work/",
)

_INTEGRATION_PREFIXES: tuple[str, ...] = (
    "intergrax/integrations/",
    "applications/",
)

_METADATA_TABLES: frozenset[str] = frozenset(
    {
        "agent_distribution_meta",
        "active_installation_slot",
        "active_revision_scope",
    }
)

_DECODE_RE = re.compile(
    r"_decode\(\s*(\w+)\s*,\s*row\[\"payload_json\"\]",
)


@lru_cache(maxsize=1)
def mechanical_agent_distribution_payload_tables() -> frozenset[str]:
    path = _REPO_ROOT / "intergrax/agent_distribution/sqlite_stores.py"
    source = path.read_text(encoding="utf-8")
    return discover_agent_distribution_payload_tables_from_sql(source)


@lru_cache(maxsize=1)
def mechanical_agent_distribution_table_family_map() -> dict[str, str]:
    tables = mechanical_agent_distribution_payload_tables()
    mapping: dict[str, str] = {}
    for table in tables:
        if table in _METADATA_TABLES:
            continue
        family = _AD_TABLE_FAMILY.get(table)
        if family is None:
            continue
        mapping[table] = family
    return mapping


def agent_distribution_table_parity_holds() -> bool:
    discovered = mechanical_agent_distribution_payload_tables() - _METADATA_TABLES
    classified = frozenset(mechanical_agent_distribution_table_family_map())
    return discovered == classified


def _family_for_candidate(candidate: DurableBoundaryCandidate) -> str | None:
    module = candidate.module_path.replace("\\", "/")
    if (
        "sqlite.table.schema" in candidate.candidate_id
        and module.startswith("intergrax/collaborative_work/")
    ):
        return "collaborative_work.sqlite_container"
    if candidate.storage_signal == DurableStorageSignal.SQLITE_PAYLOAD_TABLE:
        table = candidate.payload_symbol
        if table and table in _METADATA_TABLES:
            return None
        if table:
            return (
                _AD_TABLE_FAMILY.get(table)
                or _SQLITE_PAYLOAD_TABLE_FAMILY.get(table)
                or _SQLITE_SCHEMA_TABLE_FAMILY.get(table)
            )
    evidence0 = candidate.evidence[0] if candidate.evidence else ""
    if candidate.storage_signal == DurableStorageSignal.EXPLICIT_PERSIST_API:
        if (
            "get_package_contract_authority" in evidence0
            or "persist_package_contract_authority" in evidence0
            or "def persist_package_contract_authority" in evidence0
        ):
            if module == "intergrax/agent_distribution/sqlite_stores.py":
                return "agent_distribution.package_agent_contract_authority_record"
            if "get_package_contract_authority" in evidence0:
                return None  # protocol surface on in-memory / contract module
            return "agent_distribution.package_agent_contract_authority_record"
        if "persist_candidate_revision" in evidence0:
            return "agent_distribution.runtime_revision"
        if "save_checkpoint" in evidence0:
            if "runtime_checkpoint" in module:
                return "runtime.runtime_checkpoint"
            return "runtime.task_checkpoint"
        if "export_durable_continuation_state" in evidence0:
            return "runtime.execution_continuation_durable_state"
        if "execution_continuation_state_store_from_durable_export" in evidence0:
            return "runtime.execution_continuation_durable_state"
        if "restore_durable_continuation_backing" in evidence0:
            return "runtime.execution_continuation_durable_state"
    if module.startswith("intergrax/autonomous_work/"):
        return "autonomous_work.postgresql_container"
    if module.startswith("intergrax/collaborative_work/"):
        return "collaborative_work.sqlite_container"
    if (
        candidate.storage_signal == DurableStorageSignal.SQLITE_PAYLOAD_TABLE
        and module.startswith("intergrax/collaborative_work/")
    ):
        return "collaborative_work.sqlite_container"
    if module == "intergrax/runtime/long_running/store.py":
        if candidate.payload_symbol in _MODEL_SYMBOL_FAMILY:
            return _MODEL_SYMBOL_FAMILY[candidate.payload_symbol]
        if "ScheduledResume" in evidence0:
            return "runtime.scheduled_resume"
        if "RuntimeCheckpoint" in evidence0:
            return "runtime.runtime_checkpoint"
        if "scheduled_resumes" in evidence0 or "ScheduledResume" in evidence0:
            return "runtime.scheduled_resume"
        return "runtime.task_checkpoint"
    if module == (
        "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py"
    ):
        if candidate.payload_symbol == "SuspendedExecutionOperationDescriptor":
            return "runtime.suspended_execution_operation_descriptor"
        return "runtime.suspended_operation_backing_envelope"
    if module in _MODULE_FAMILY_ANCHOR:
        anchor = _MODULE_FAMILY_ANCHOR[module]
        if candidate.storage_signal == DurableStorageSignal.SQLITE_PAYLOAD_TABLE:
            return _AD_TABLE_FAMILY.get(candidate.payload_symbol or "", anchor)
        if candidate.storage_signal in (
            DurableStorageSignal.REDIS_KV_WRITE,
            DurableStorageSignal.REDIS_HASH_WRITE,
            DurableStorageSignal.REDIS_SCRIPTED_STATE,
            DurableStorageSignal.REDIS_READ,
            DurableStorageSignal.PICKLE_DURABLE_WIRE,
            DurableStorageSignal.FILE_WRITE,
        ):
            return anchor
        return anchor
    if module.endswith("sqlite_stores.py") and candidate.payload_symbol:
        return _AD_TABLE_FAMILY.get(candidate.payload_symbol)
    if candidate.payload_symbol and candidate.payload_symbol in _MODEL_SYMBOL_FAMILY:
        return _MODEL_SYMBOL_FAMILY[candidate.payload_symbol]
    if "long_running" in module and "checkpoint" in module.lower():
        if "runtime_checkpoint" in module:
            return "runtime.runtime_checkpoint"
        if "scheduled_resume" in module:
            return "runtime.scheduled_resume"
        return "runtime.task_checkpoint"
    return None


def _exclusion_for_candidate(
    candidate: DurableBoundaryCandidate,
) -> tuple[DurableExclusionReason, tuple[str, ...]] | None:
    module = candidate.module_path.replace("\\", "/")
    if candidate.storage_signal == DurableStorageSignal.SQLITE_PAYLOAD_TABLE:
        if candidate.payload_symbol in _METADATA_TABLES:
            return (
                DurableExclusionReason.SQL_METADATA_NOT_SEMANTIC_PAYLOAD,
                (f"table:{candidate.payload_symbol}",),
            )
    if module.endswith("in_memory_stores.py") and "contract_authority" in (
        candidate.function_name or ""
    ).lower():
        return (
            DurableExclusionReason.EPHEMERAL_IN_MEMORY_ONLY,
            ("in_memory contract authority slot",),
        )
    for marker in _SERVICE_MODULE_MARKERS:
        if module.endswith(marker):
            return (
                DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD,
                (f"service_module:{marker}",),
            )
    if candidate.class_name and candidate.class_name.endswith("Error"):
        return (
            DurableExclusionReason.EXCEPTION_NOT_PAYLOAD,
            (f"exception_class:{candidate.class_name}",),
        )
    if candidate.class_name and any(
        candidate.class_name.endswith(s) for s in _ORCHESTRATOR_CLASS_SUFFIXES
    ):
        return (
            DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD,
            (f"class_suffix:{candidate.class_name}",),
        )
    if candidate.storage_signal == DurableStorageSignal.WIRE_TO_DURABLE_STORE:
        if module in _MODULE_FAMILY_ANCHOR:
            return None
        if _family_for_candidate(candidate) is None:
            return (
                DurableExclusionReason.SERIALIZATION_WITHOUT_DURABLE_IO,
                ("wire_signal_without_family_anchor",),
            )
    if module == "intergrax/agent_distribution/stores.py" and "get_package_contract_authority" in (
        candidate.evidence[0] if candidate.evidence else ""
    ):
        return (
            DurableExclusionReason.PROTOCOL_OR_INTERFACE_ONLY,
            ("AgentArtifactMetadataStore protocol method",),
        )
    if module.endswith("stores.py") and "Protocol" in (candidate.evidence[0] if candidate.evidence else ""):
        return (
            DurableExclusionReason.PROTOCOL_OR_INTERFACE_ONLY,
            ("protocol_surface",),
        )
    if "/agents/" in module and _family_for_candidate(candidate) is None:
        if candidate.storage_signal in (
            DurableStorageSignal.WIRE_TO_DURABLE_STORE,
            DurableStorageSignal.EXPLICIT_PERSIST_API,
        ):
            return (
                DurableExclusionReason.SERIALIZATION_WITHOUT_DURABLE_IO,
                ("tier2_agent_no_durable_anchor",),
            )
    if "intergrax/agents/persistence" in module and _family_for_candidate(candidate) is None:
        return (
            DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD,
            ("agent_harness_persistence_wiring",),
        )
    if any(module.startswith(prefix) for prefix in _INTEGRATION_PREFIXES):
        return (
            DurableExclusionReason.TRANSPORT_ONLY_ENCODING,
            ("integration_or_application_adapter_boundary",),
        )
    for frag, reason in (
        ("intergrax/runtime/execution/deadline_authority/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/budget/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/delegated_execution/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/attempt_lifecycle/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/lineage/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/execution_terminal/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/execution/document_store_execution_terminal", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/agent_distribution/runtime_context_staging.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/agent_distribution/materialization_adapters.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/applications/_shared/package_wiring.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/applications/_shared/registry_ops_wiring.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/runtime/execution/sqlite_decision_", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/applications/_shared/application_build_context.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/applications/_shared/sqlite_async_task_index.py", DurableExclusionReason.SQL_METADATA_NOT_SEMANTIC_PAYLOAD),
        ("intergrax/applications/_shared/docker_templates.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/applications/_shared/integrations/persistence.py", DurableExclusionReason.TRANSPORT_ONLY_ENCODING),
        ("intergrax/applications/_shared/profile_resolution/persistence.py", DurableExclusionReason.TRANSPORT_ONLY_ENCODING),
        ("intergrax/distributed/providers/redis_execution_semaphore.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/distributed/providers/redis_rate_limiter.py", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/queueing/", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/runtime/diagnostics/", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/runtime/adaptive/", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/runtime/human/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
        ("intergrax/runtime/vendor_knowledge/", DurableExclusionReason.CACHE_NON_DURABLE),
        ("intergrax/core/qualification/", DurableExclusionReason.TEST_OR_LAB_ONLY),
        ("agents/", DurableExclusionReason.ORCHESTRATOR_SERVICE_NOT_PAYLOAD),
    ):
        if frag in module:
            if (
                reason == DurableExclusionReason.CACHE_NON_DURABLE
                and module.replace("\\", "/") in _MODULE_FAMILY_ANCHOR
            ):
                continue
            return (reason, (f"path:{frag}",))
    if module.startswith("synthetic/qualification/"):
        return None
    if not any(module.startswith(prefix) for prefix in _PERSISTENCE_CRITICAL_PREFIXES):
        return (
            DurableExclusionReason.SERIALIZATION_WITHOUT_DURABLE_IO,
            ("non_critical_persistence_prefix",),
        )
    return None


def classify_durable_candidate(
    candidate: DurableBoundaryCandidate,
) -> DurableCandidateClassification:
    family_id = _family_for_candidate(candidate)
    if family_id is not None:
        return DurableCandidateClassification(
            candidate=candidate,
            disposition=PersistedCandidateDisposition.PERSISTED_FAMILY_MEMBER,
            family_id=family_id,
            exclusion_reason=None,
            classification_evidence=(f"family_anchor:{family_id}",),
        )
    exclusion = _exclusion_for_candidate(candidate)
    if exclusion is not None:
        reason, evidence = exclusion
        return DurableCandidateClassification(
            candidate=candidate,
            disposition=PersistedCandidateDisposition.EVIDENCE_BACKED_EXCLUSION,
            family_id=None,
            exclusion_reason=reason,
            classification_evidence=evidence,
        )
    return DurableCandidateClassification(
        candidate=candidate,
        disposition=PersistedCandidateDisposition.EVIDENCE_BACKED_EXCLUSION,
        family_id=None,
        exclusion_reason=None,
        classification_evidence=(),
    )


def classify_is_unclassified(classification: DurableCandidateClassification) -> bool:
    if classification.disposition == PersistedCandidateDisposition.PERSISTED_FAMILY_MEMBER:
        return False
    if classification.exclusion_reason is not None:
        return False
    return True


def classify_durable_candidates(
    candidates: tuple[DurableBoundaryCandidate, ...],
) -> tuple[DurableCandidateClassification, ...]:
    return tuple(classify_durable_candidate(c) for c in candidates)


def semantic_family_ids_from_classifications(
    classifications: tuple[DurableCandidateClassification, ...],
) -> frozenset[str]:
    return frozenset(
        c.family_id
        for c in classifications
        if c.disposition == PersistedCandidateDisposition.PERSISTED_FAMILY_MEMBER
        and c.family_id is not None
    )
