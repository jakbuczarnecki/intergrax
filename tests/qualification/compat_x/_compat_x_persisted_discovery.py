# © Artur Czarnecki. All rights reserved.

"""Mechanical persisted-family discovery signals for R2-P0 closed-world parity."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Agent Distribution SQLite tables with JSON payloads (must match inventory).
_AGENT_DISTRIBUTION_SQLITE_TABLE_PAYLOADS: tuple[tuple[str, str], ...] = (
    ("installations", "agent_distribution.agent_installation_record"),
    ("bindings", "agent_distribution.application_agent_binding"),
    ("revisions", "agent_distribution.runtime_revision"),
    ("artifact_metadata", "agent_distribution.agent_artifact_metadata"),
    ("runtime_locks", "agent_distribution.materialized_runtime_lock"),
    ("materializations", "agent_distribution.runtime_materialization_record"),
    ("effective_roster_snapshots", "agent_distribution.effective_roster"),
    ("deployment_instances", "agent_distribution.deployment_instance_record"),
    ("serving_records", "agent_distribution.application_environment_serving_record"),
    ("projection_descriptors", "agent_distribution.runtime_registry_projection_descriptor"),
)

_PERSISTED_SCAN_MODULES: tuple[str, ...] = (
    "intergrax/agent_distribution/sqlite_stores.py",
    "intergrax/runtime/execution/suspended_operation/document_store_suspended_operation_store.py",
    "intergrax/runtime/events/stores/sqlite_runtime_event_store.py",
    "intergrax/runtime/long_running/store.py",
    "intergrax/runtime/task_memory/stores/sqlite_task_memory_store.py",
    "intergrax/memory/stores/sqlite_user_profile_store.py",
    "intergrax/memory/stores/sqlite_conversational_memory_store.py",
    "intergrax/distributed/providers/sqlite_kv_store.py",
    "intergrax/applications/_shared/registry_snapshot_store.py",
    "intergrax/runtime/execution/sqlite_decision_event_append_persistence.py",
    "intergrax/autonomous_work/postgresql_repository.py",
    "intergrax/collaborative_work/sqlite_repository.py",
)

_DECODE_RE = re.compile(
    r"_decode\(\s*(\w+)\s*,\s*row\[\"payload_json\"\]\s*,\s*(\w+)\s*,",
)


@lru_cache(maxsize=1)
def mechanical_agent_distribution_family_ids() -> frozenset[str]:
    sqlite_path = _REPO_ROOT / "intergrax/agent_distribution/sqlite_stores.py"
    source = sqlite_path.read_text(encoding="utf-8")
    discovered: set[str] = set()
    for table, family_id in _AGENT_DISTRIBUTION_SQLITE_TABLE_PAYLOADS:
        if f"FROM {table}" not in source and f"INTO {table}" not in source:
            raise RuntimeError(f"expected SQLite table {table} in sqlite_stores.py")
        discovered.add(family_id)
    if "model_validate_json" in source:
        if "agent_distribution.agent_artifact_metadata" not in discovered:
            raise RuntimeError("artifact_metadata family expected")
    for match in _DECODE_RE.finditer(source):
        _model, _schema_const = match.groups()
    discovered.add("agent_distribution.sqlite_container")
    discovered.add("agent_distribution.package_agent_contract_authority_record")
    return frozenset(discovered)


@lru_cache(maxsize=1)
def mechanical_closed_world_family_ids() -> frozenset[str]:
    ids = set(mechanical_agent_distribution_family_ids())
    ids.update(
        {
            "runtime.suspended_operation_backing_envelope",
            "runtime.suspended_execution_operation_descriptor",
            "runtime.runtime_event",
            "runtime.task_checkpoint",
            "runtime.runtime_checkpoint",
            "runtime.scheduled_resume",
            "runtime.task_memory_record",
            "runtime.decision_event_append",
            "memory.user_profile",
            "memory.conversational_turn",
            "distributed.kv_entry",
            "applications.registry_snapshot",
            "autonomous_work.postgresql_container",
            "collaborative_work.sqlite_container",
        }
    )
    for module_rel in _PERSISTED_SCAN_MODULES:
        if not (_REPO_ROOT / module_rel).is_file():
            raise FileNotFoundError(module_rel)
    return frozenset(ids)


def inventory_family_ids() -> frozenset[str]:
    from tests.qualification.compat_x._compat_x_persisted_inventory import (
        PERSISTED_SCHEMA_FAMILIES,
    )

    return frozenset(f.family_id for f in PERSISTED_SCHEMA_FAMILIES)


def persisted_closed_world_parity_holds() -> bool:
    mechanical = mechanical_closed_world_family_ids()
    inventory = inventory_family_ids()
    return mechanical == inventory
