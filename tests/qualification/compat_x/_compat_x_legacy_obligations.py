# © Artur Czarnecki. All rights reserved.

"""R2-R2 closed-world legacy compatibility obligations (qualification SSOT)."""

from __future__ import annotations

from functools import lru_cache

from intergrax.agent_distribution.deployment import SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1
from intergrax.agent_distribution.runtime_materialization_record import (
    SCHEMA_RUNTIME_MATERIALIZATION_RECORD_V1,
)
from intergrax.agent_distribution.agent_contract_authority import (
    SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1,
)
from intergrax.agent_distribution.stores import (
    SCHEMA_AGENT_ARTIFACT_METADATA_V1,
    SCHEMA_APPLICATION_ENVIRONMENT_SERVING_RECORD_V1,
)
from intergrax.applications._shared.registry_snapshot_persistence import SCHEMA_REGISTRY_SNAPSHOT_V1
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1,
)
from intergrax.contracts.idempotency_store import SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1
from intergrax.memory.conversational_memory_persistence import SCHEMA_CONVERSATIONAL_MEMORY_V1
from intergrax.memory.user_profile_serialization import SCHEMA_USER_PROFILE_V1
from intergrax.runtime.task_memory.models import SCHEMA_TASK_MEMORY_RECORD_V1

from tests.qualification.compat_x._compat_x_persisted_inventory import PERSISTED_SCHEMA_FAMILIES
from tests.qualification.compat_x._compat_x_persisted_types import (
    LegacyCompatibilityDisposition,
    LegacyPersistedCompatibilityObligation,
)

_PRE_R2_R1_SHA = "e5ae6a3e358470607c2acd2eeb5a18334f36db63"
_AD_SQLITE = "intergrax/agent_distribution/sqlite_stores.py"
_AD_LEGACY = "intergrax/agent_distribution/sqlite_payload_legacy.py"


def _obligation(
    family_id: str,
    legacy_shape_id: str,
    current_version: str,
    disposition: LegacyCompatibilityDisposition,
    semantic_owner: str,
    migration_owner: str | None,
    reader_path: str,
    evidence: tuple[str, ...],
    *,
    sanctioned_reader_symbols: tuple[str, ...] = (),
) -> LegacyPersistedCompatibilityObligation:
    return LegacyPersistedCompatibilityObligation(
        family_id=family_id,
        legacy_shape_id=legacy_shape_id,
        current_version=current_version,
        disposition=disposition,
        semantic_owner_path=semantic_owner,
        migration_owner_path=migration_owner,
        reader_path=reader_path,
        evidence_paths=evidence,
        sanctioned_reader_symbols=sanctioned_reader_symbols,
    )


@lru_cache(maxsize=1)
def legacy_persisted_compatibility_obligations() -> tuple[LegacyPersistedCompatibilityObligation, ...]:
    return (
        _obligation(
            "agent_distribution.agent_artifact_metadata",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_AGENT_ARTIFACT_METADATA_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/agent_distribution/stores.py",
            _AD_LEGACY,
            _AD_SQLITE,
            (_AD_LEGACY, _AD_SQLITE, f"git:{_PRE_R2_R1_SHA}:intergrax/agent_distribution/stores.py"),
            sanctioned_reader_symbols=(
                "decode_agent_artifact_metadata_payload",
                "_decode_versioned_or_legacy",
            ),
        ),
        _obligation(
            "agent_distribution.runtime_materialization_record",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_RUNTIME_MATERIALIZATION_RECORD_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/agent_distribution/runtime_materialization_record.py",
            _AD_LEGACY,
            _AD_SQLITE,
            (_AD_LEGACY, f"git:{_PRE_R2_R1_SHA}:intergrax/agent_distribution/runtime_materialization_record.py"),
            sanctioned_reader_symbols=(
                "decode_runtime_materialization_record_payload",
                "_decode_versioned_or_legacy",
            ),
        ),
        _obligation(
            "agent_distribution.deployment_instance_record",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/agent_distribution/deployment.py",
            _AD_LEGACY,
            _AD_SQLITE,
            (_AD_LEGACY, f"git:{_PRE_R2_R1_SHA}:intergrax/agent_distribution/deployment.py"),
            sanctioned_reader_symbols=(
                "decode_deployment_instance_record_payload",
                "_decode_versioned_or_legacy",
            ),
        ),
        _obligation(
            "agent_distribution.application_environment_serving_record",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_APPLICATION_ENVIRONMENT_SERVING_RECORD_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/agent_distribution/stores.py",
            _AD_LEGACY,
            _AD_SQLITE,
            (_AD_LEGACY, f"git:{_PRE_R2_R1_SHA}:intergrax/agent_distribution/stores.py"),
            sanctioned_reader_symbols=(
                "decode_application_environment_serving_record_payload",
                "_decode_versioned_or_legacy",
            ),
        ),
        _obligation(
            "runtime.task_memory_record",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_TASK_MEMORY_RECORD_V1,
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            "intergrax/runtime/task_memory/models.py",
            "intergrax/runtime/task_memory/sqlite_row_legacy.py",
            "intergrax/runtime/task_memory/stores/sqlite_task_memory_store.py",
            (
                "intergrax/runtime/task_memory/sqlite_row_legacy.py",
                "intergrax/runtime/task_memory/stores/sqlite_task_memory_store.py",
            ),
            sanctioned_reader_symbols=("task_memory_record_from_sqlite_row",),
        ),
        _obligation(
            "memory.user_profile",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_USER_PROFILE_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/memory/user_profile_serialization.py",
            "intergrax/memory/user_profile_serialization.py",
            "intergrax/memory/stores/sqlite_user_profile_store.py",
            (
                "intergrax/memory/user_profile_serialization.py",
                f"git:{_PRE_R2_R1_SHA}:intergrax/memory/user_profile_serialization.py",
            ),
            sanctioned_reader_symbols=("deserialize_user_profile_payload",),
        ),
        _obligation(
            "memory.conversational_turn",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_CONVERSATIONAL_MEMORY_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/memory/conversational_memory_persistence.py",
            "intergrax/memory/conversational_memory_persistence.py",
            "intergrax/memory/stores/sqlite_conversational_memory_store.py",
            ("intergrax/memory/conversational_memory_persistence.py",),
            sanctioned_reader_symbols=("parse_conversational_memory_persisted",),
        ),
        _obligation(
            "applications.registry_snapshot",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_REGISTRY_SNAPSHOT_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/applications/_shared/registry_snapshot_persistence.py",
            "intergrax/applications/_shared/registry_snapshot_persistence.py",
            "intergrax/applications/_shared/registry_snapshot_store.py",
            ("intergrax/applications/_shared/registry_snapshot_persistence.py",),
            sanctioned_reader_symbols=("parse_registry_snapshot_persisted",),
        ),
        _obligation(
            "runtime.tool_idempotency_ledger",
            "LEGACY_UNVERSIONED_PRE_V1",
            SCHEMA_TOOL_IDEMPOTENCY_LEDGER_V1,
            LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED,
            "intergrax/contracts/idempotency_store.py",
            "intergrax/contracts/idempotency_store.py",
            "intergrax/runtime/tools/sqlite_idempotency_store.py",
            (
                "intergrax/contracts/idempotency_store.py",
                "intergrax/runtime/tools/sqlite_idempotency_store.py",
                "intergrax/distributed/providers/redis_idempotency_store.py",
            ),
            sanctioned_reader_symbols=("assert_tool_idempotency_ledger_schema_version",),
        ),
        _obligation(
            "runtime.suspended_execution_operation_descriptor",
            "LEGACY_DESCRIPTOR_WITHOUT_AUTHORITY_SCOPE",
            SCHEMA_SUSPENDED_EXECUTION_OPERATION_DESCRIPTOR_V1,
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/contracts/execution/suspended_operation/descriptor.py",
            "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
            "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
            (
                "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
                "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
            ),
            sanctioned_reader_symbols=(
                "parse_suspended_execution_operation_descriptor",
                "_migrate_legacy_descriptor",
            ),
        ),
        _obligation(
            "runtime.suspended_operation_backing_envelope",
            "suspended_operation_backing.v1",
            "suspended_operation_backing.v1",
            LegacyCompatibilityDisposition.EXPLICIT_LEGACY_READER,
            "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
            "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
            "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
            (
                "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
                "intergrax/contracts/execution/suspended_operation/descriptor_persisted.py",
            ),
            sanctioned_reader_symbols=(
                "parse_suspended_execution_operation_descriptor",
                "_migrate_legacy_descriptor",
            ),
        ),
        _obligation(
            "runtime.runtime_event",
            "LEGACY_UNVERSIONED_PRE_V1",
            "runtime_event.v2",
            LegacyCompatibilityDisposition.CURRENT_ONLY_FAIL_CLOSED,
            "intergrax/contracts/runtime_event.py",
            "intergrax/contracts/runtime_event.py",
            "intergrax/contracts/runtime_event.py",
            ("intergrax/contracts/runtime_event.py",),
        ),
        _obligation(
            "runtime.runtime_event",
            "runtime_event.v1",
            "runtime_event.v2",
            LegacyCompatibilityDisposition.DEFERRED_TO_EVENT_EVOLUTION,
            "intergrax/contracts/runtime_event.py",
            None,
            "intergrax/contracts/runtime_event.py",
            ("intergrax/runtime/schema/registry.py",),
        ),
        _obligation(
            "agent_distribution.package_agent_contract_authority_record",
            "NO_PRIOR_DURABLE_REPRESENTATION",
            SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1,
            LegacyCompatibilityDisposition.NO_PRIOR_DURABLE_REPRESENTATION,
            "intergrax/agent_distribution/agent_contract_authority.py",
            None,
            _AD_SQLITE,
            ("intergrax/agent_distribution/agent_contract_authority.py",),
        ),
        _obligation(
            "autonomous_work.postgresql_container",
            "EXPLICIT_DOMAIN_MIGRATION",
            "autonomous_work.postgresql",
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            "intergrax/autonomous_work/postgresql_repository.py",
            "intergrax/autonomous_work/postgresql_repository.py",
            "intergrax/autonomous_work/postgresql_repository.py",
            ("intergrax/autonomous_work/postgresql_repository.py",),
            sanctioned_reader_symbols=("_require_revision_argument",),
        ),
        _obligation(
            "collaborative_work.sqlite_container",
            "EXPLICIT_DOMAIN_MIGRATION",
            "collaborative_work.sqlite",
            LegacyCompatibilityDisposition.EXPLICIT_MIGRATION,
            "intergrax/collaborative_work/sqlite_repository.py",
            "intergrax/collaborative_work/serialization.py",
            "intergrax/collaborative_work/sqlite_repository.py",
            (
                "intergrax/collaborative_work/sqlite_repository.py",
                "intergrax/collaborative_work/serialization.py",
            ),
            sanctioned_reader_symbols=("workspace_membership_from_json",),
        ),
    )


def frz_cmp_06_family_ids() -> frozenset[str]:
    return frozenset(
        f.family_id
        for f in PERSISTED_SCHEMA_FAMILIES
        if "FRZ-CMP-06" in f.frz_cmp_tags
    )


def legacy_r2_r2_remediation_family_ids() -> tuple[str, ...]:
    return tuple(
        sorted({o.family_id for o in legacy_persisted_compatibility_obligations()})
    )


LEGACY_COMPATIBILITY_OBLIGATIONS = legacy_persisted_compatibility_obligations()
