# © Artur Czarnecki. All rights reserved.

"""Production-effective SQLite touchpoint accountability (EBH-5 annex)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_5._closed_world_discovery import discover_sqlite_connect_paths_under_intergrax


class SqliteTouchpointClassification(StrEnum):
    SANCTIONED_ADAPTER = "ALLOWED — PROVIDER ADAPTER BOUNDARY"
    FIXED_SUBSTRATE = "ALLOWED — EXPLICITLY RATIFIED FIXED SUBSTRATE"
    INTERNAL = "ALLOWED — INTERNAL NON-EXTENSIBLE MECHANISM"
    REQUIRES_EXPLANATION = "REQUIRES ARCHITECTURE DECISION"


@dataclass(frozen=True, slots=True)
class SqliteTouchpointRecord:
    repo_path: str
    classification: SqliteTouchpointClassification
    inventory_concern_id: str | None
    notes: str


_SQLITE_TOUCHPOINTS: Final[tuple[SqliteTouchpointRecord, ...]] = (
    SqliteTouchpointRecord(
        "intergrax/integrations/providers/relational_store/sqlite/adapter.py",
        SqliteTouchpointClassification.SANCTIONED_ADAPTER,
        "persistence.relational_sqlite_lab",
        "Integration relational_store sqlite adapter",
    ),
    SqliteTouchpointRecord(
        "intergrax/integrations/providers/relational_store/sqlite/human_decision_legacy.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "Legacy migration helper — not production selection authority",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/long_running/store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.checkpoint.task",
        "TaskCheckpointPersistence SQLite impl",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/human/store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.human.decision",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/tools/sqlite_idempotency_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.execution.idempotency",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/execution/sqlite_decision_finalization_persistence.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.execution.decision_finalization",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/execution/sqlite_decision_checkpoint_persistence.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.checkpoint.task",
        "Decision checkpoint durable component",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/execution/sqlite_decision_event_append_persistence.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.execution.decision_finalization",
        "Append-only decision events",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/events/stores/sqlite_runtime_event_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.events.runtime_store",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/nexus/tracing/sqlite_run_trace_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.trace.run",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/agents/persistence/checkpoint_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.agent.checkpoint",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/agents/persistence/compensation_queue_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.agent.compensation_queue",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/external_operations/sqlite_tool_life_durable_stores.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.tool_life.durable",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/enterprise_reliability/sqlite_provider_invocation_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "runtime.enterprise_reliability.provider_invocation",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/applications/_shared/single_node_production_profile_v1.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "platform.single_node_profile_v1.sqlite_triad",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/distributed/providers/sqlite_kv_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "distributed.sqlite_kv",
        "Distinct from IntegrationProfile KV catalog path",
    ),
    SqliteTouchpointRecord(
        "intergrax/memory/stores/sqlite_conversational_memory_store.py",
        SqliteTouchpointClassification.SANCTIONED_ADAPTER,
        "memory.sqlite.conversational",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/memory/stores/sqlite_user_profile_store.py",
        SqliteTouchpointClassification.SANCTIONED_ADAPTER,
        "memory.sqlite.conversational",
        "User profile via relational/mem wiring",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/signal_store.py",
        SqliteTouchpointClassification.INTERNAL,
        "lab.adaptive.signal_proposal_stores",
        "Lab/debug adaptive — production-effective only via lab factory",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/proposal_store.py",
        SqliteTouchpointClassification.INTERNAL,
        "lab.adaptive.signal_proposal_stores",
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/bandit_state_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "Adaptive research stores — not extensibility-claimed",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/loop_apply_block_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/profile_mutation_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/profile_pointer_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/adaptive/profile_version_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "",
    ),
    SqliteTouchpointRecord(
        "intergrax/agent_distribution/sqlite_stores.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        "platform.single_node_profile_v1.sqlite_triad",
        "Agent distribution registry SQLite",
    ),
    SqliteTouchpointRecord(
        "intergrax/applications/_shared/registry_snapshot_store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "Harness/registry snapshot helper",
    ),
    SqliteTouchpointRecord(
        "intergrax/applications/_shared/sqlite_async_task_index.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "Async task index — composition-scoped",
    ),
    SqliteTouchpointRecord(
        "intergrax/collaborative_work/sqlite_repository.py",
        SqliteTouchpointClassification.SANCTIONED_ADAPTER,
        "memory.sqlite.conversational",
        "CW repos via relational provider",
    ),
    SqliteTouchpointRecord(
        "intergrax/experiments/store.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "Debug experiments — not production extensibility claim",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/context_lifecycle/sqlite_repository.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Context lifecycle durable projection",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/migration/human_decision_legacy_disposition.py",
        SqliteTouchpointClassification.INTERNAL,
        None,
        "One-shot migration tooling",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/nexus/session/context_revision.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Session context revision store",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/nexus/session/sqlite_session_storage.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Nexus session storage",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/notifications/deliveries/sqlite_delivery_ledger.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Delivery ledger",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/observability/revision_ordering_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Observability ordering auxiliary store",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/organization/stores/sqlite_organization_profile_store.py",
        SqliteTouchpointClassification.SANCTIONED_ADAPTER,
        "memory.sqlite.conversational",
        "Org profile via mem/relational wiring",
    ),
    SqliteTouchpointRecord(
        "intergrax/runtime/task_memory/stores/sqlite_task_memory_store.py",
        SqliteTouchpointClassification.FIXED_SUBSTRATE,
        None,
        "Task memory SQLite impl",
    ),
)


@lru_cache(maxsize=1)
def sqlite_touchpoint_by_path() -> dict[str, SqliteTouchpointRecord]:
    return {row.repo_path: row for row in _SQLITE_TOUCHPOINTS}


def assert_closed_world_sqlite_accountability() -> None:
    discovered = discover_sqlite_connect_paths_under_intergrax()
    registered = frozenset(sqlite_touchpoint_by_path())
    missing = discovered - registered
    extra = registered - discovered
    assert not missing, f"sqlite paths without annex accountability: {sorted(missing)}"
    assert not extra, f"stale annex paths not in codebase: {sorted(extra)}"


SQLITE_TOUCHPOINT_COUNT: Final[int] = len(_SQLITE_TOUCHPOINTS)
