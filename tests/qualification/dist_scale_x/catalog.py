# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X canonical qualification catalog — dimensions A–N, WL-1..7, exit 1–16, backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tests.qualification.dist_scale_x._constants import (
    DIMENSION_IDS,
    EXIT_CRITERION_IDS,
    MANDATORY_BACKEND_IDS,
    WORKLOAD_IDS,
)


class BackendQualificationClass(StrEnum):
    MANDATORY_SCALE = "M"
    REPRESENTATIVE = "R"
    FUNCTIONAL_ONLY = "F"
    EXTERNAL_SANDBOX = "E"
    UNSUPPORTED_DISTRIBUTED = "U"


class ThresholdLockState(StrEnum):
    NOT_LOCKED = "NOT_LOCKED"
    LOCKED = "LOCKED"


def _l1(test_name: str) -> str:
    return f"tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py::{test_name}"


def _l2(test_name: str) -> str:
    return f"tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py::{test_name}"


@dataclass(frozen=True, slots=True)
class DistScaleDimensionEntry:
    dimension_id: str
    title: str
    workload_ids: tuple[str, ...]
    exit_criterion_ids: tuple[int, ...]
    implementation_batch: str
    p1_structural_gate_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DistScaleWorkloadEntry:
    workload_id: str
    title: str
    dimension_ids: tuple[str, ...]
    exit_criterion_ids: tuple[int, ...]
    mandatory_backend_ids: tuple[str, ...]
    implementation_batch: str
    future_qual_owner_glob: str


@dataclass(frozen=True, slots=True)
class DistScaleExitCriterionEntry:
    exit_id: int
    title: str
    dimension_ids: tuple[str, ...]
    workload_ids: tuple[str, ...]
    evidence_layer: str


@dataclass(frozen=True, slots=True)
class DistScaleBackendEntry:
    backend_id: str
    name: str
    qualification_class: BackendQualificationClass
    code_anchor: str
    qualification_owner: str
    implementation_batch: str
    operator_decision_required: bool = False


@dataclass(frozen=True, slots=True)
class DistScaleHistoricalEvidenceEntry:
    stage_id: str
    accepted_sha: str
    reconciliation: str
    scope_note: str


DIST_SCALE_X_DIMENSION_CATALOG: tuple[DistScaleDimensionEntry, ...] = (
    DistScaleDimensionEntry(
        "A",
        "Throughput and concurrency",
        ("WL-1", "WL-2", "WL-4", "WL-5"),
        (1, 3, 4, 13),
        "P2+",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "B",
        "Horizontal scaling",
        ("WL-1",),
        (3, 13),
        "P2",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "C",
        "Backpressure and overload",
        ("WL-1", "WL-6"),
        (4, 5, 13),
        "P2-P3",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "D",
        "Hot partitions / hot keys",
        ("WL-2", "WL-3"),
        (6, 13),
        "P3",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "E",
        "Retry storms",
        ("WL-4", "WL-6"),
        (5, 13),
        "P3",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "F",
        "Partial failure",
        ("WL-4", "WL-5"),
        (8, 13),
        "P4",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "G",
        "Crash consistency",
        ("WL-5",),
        (7, 13),
        "P4",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "H",
        "Durable restart",
        ("WL-5",),
        (7, 13),
        "P4",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "I",
        "Network partitions / unavailable nodes",
        (),
        (8, 13),
        "P5",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "J",
        "Ordering and concurrency anomalies",
        ("WL-3",),
        (6, 13),
        "P5",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "K",
        "Tenant isolation under stress",
        ("WL-2",),
        (9, 13),
        "P3",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "L",
        "Resource exhaustion",
        ("WL-4", "WL-6"),
        (10, 13),
        "P5",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "M",
        "Soak testing",
        ("WL-7",),
        (11, 13),
        "P6",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
    DistScaleDimensionEntry(
        "N",
        "Chaos / fault injection",
        ("WL-7",),
        (12, 13),
        "P6",
        (_l1("test_l1_dimension_catalog_covers_a_through_n"),),
    ),
)

DIST_SCALE_X_WORKLOAD_CATALOG: tuple[DistScaleWorkloadEntry, ...] = (
    DistScaleWorkloadEntry(
        "WL-1",
        "Many independent executions",
        ("A", "B", "C"),
        (1, 3, 4, 13),
        ("MB-1", "MB-2", "MB-3"),
        "P2",
        "tests/qualification/dist_scale_x/wl_01_*",
    ),
    DistScaleWorkloadEntry(
        "WL-2",
        "Hot tenant",
        ("A", "D", "K"),
        (1, 6, 9, 13),
        ("MB-1", "MB-2", "MB-3"),
        "P2-P3",
        "tests/qualification/dist_scale_x/wl_02_*",
    ),
    DistScaleWorkloadEntry(
        "WL-3",
        "Hot key / execution family",
        ("D", "J"),
        (1, 6, 13),
        ("MB-1", "MB-2"),
        "P3",
        "tests/qualification/dist_scale_x/wl_03_*",
    ),
    DistScaleWorkloadEntry(
        "WL-4",
        "Provider-heavy (HTTP loopback physical)",
        ("A", "E", "F", "L"),
        (2, 5, 8, 10, 13),
        ("MB-5",),
        "P3",
        "tests/qualification/dist_scale_x/wl_04_http_physical_*",
    ),
    DistScaleWorkloadEntry(
        "WL-5",
        "Persistence-heavy",
        ("A", "F", "G", "H"),
        (2, 7, 8, 13),
        ("MB-1",),
        "P4",
        "tests/qualification/dist_scale_x/wl_05_*",
    ),
    DistScaleWorkloadEntry(
        "WL-6",
        "Retry storm",
        ("C", "E", "L"),
        (4, 5, 10, 13),
        ("MB-1", "MB-2", "MB-3", "MB-4"),
        "P3",
        "tests/qualification/dist_scale_x/wl_06_*",
    ),
    DistScaleWorkloadEntry(
        "WL-7",
        "Long-running soak",
        ("M", "N"),
        (11, 12, 13),
        ("MB-1", "MB-2", "MB-3", "MB-4", "MB-5"),
        "P6",
        "tests/qualification/dist_scale_x/wl_07_*",
    ),
)

DIST_SCALE_X_EXIT_CRITERION_CATALOG: tuple[DistScaleExitCriterionEntry, ...] = (
    DistScaleExitCriterionEntry(1, "Representative workloads defined", DIMENSION_IDS, WORKLOAD_IDS, "L2"),
    DistScaleExitCriterionEntry(
        2,
        "Production adapters exercised",
        ("A", "F", "G", "H"),
        ("WL-4", "WL-5"),
        "L3/L4",
    ),
    DistScaleExitCriterionEntry(3, "Multi-worker scale tested", ("B",), ("WL-1",), "L4"),
    DistScaleExitCriterionEntry(4, "Overload/backpressure tested", ("C",), ("WL-1", "WL-6"), "L3"),
    DistScaleExitCriterionEntry(5, "Retry storms tested", ("E",), ("WL-6",), "L3"),
    DistScaleExitCriterionEntry(6, "Hot partition/key scenarios tested", ("D", "J"), ("WL-2", "WL-3"), "L3"),
    DistScaleExitCriterionEntry(7, "Crash/restart tested", ("G", "H"), ("WL-5",), "L4"),
    DistScaleExitCriterionEntry(8, "Partial failures tested", ("F", "I"), ("WL-4", "WL-5"), "L3/L4"),
    DistScaleExitCriterionEntry(9, "Tenant stress isolation tested", ("K",), ("WL-2",), "L3"),
    DistScaleExitCriterionEntry(10, "Resource exhaustion tested", ("L",), ("WL-4", "WL-6"), "L3"),
    DistScaleExitCriterionEntry(11, "Soak completed", ("M",), ("WL-7",), "L4+"),
    DistScaleExitCriterionEntry(12, "Chaos matrix completed", ("N",), ("WL-7",), "L4+"),
    DistScaleExitCriterionEntry(13, "No correctness invariant violated", DIMENSION_IDS, WORKLOAD_IDS, "L3+"),
    DistScaleExitCriterionEntry(14, "Certified capacity envelope documented", DIMENSION_IDS, (), "L2 P7"),
    DistScaleExitCriterionEntry(15, "Certified failure/recovery envelope documented", DIMENSION_IDS, (), "L2 P7"),
    DistScaleExitCriterionEntry(
        16,
        "Unresolved scale blockers = 0 or explicit release-blocking",
        (),
        (),
        "L2",
    ),
)

DIST_SCALE_X_BACKEND_CATALOG: tuple[DistScaleBackendEntry, ...] = (
    DistScaleBackendEntry(
        "MB-1",
        "PostgreSQL (relational + CW)",
        BackendQualificationClass.MANDATORY_SCALE,
        "PostgreSQLIntegrationConfig / open_postgresql_collaborative_work_repositories",
        "tests/qualification/dist_scale_x/wl_05_*",
        "P2-P4",
    ),
    DistScaleBackendEntry(
        "MB-2",
        "DocumentStore TaskQueue",
        BackendQualificationClass.MANDATORY_SCALE,
        "intergrax/queueing/providers/document_store/document_store_task_queue.py",
        "tests/qualification/dist_scale_x/wl_01_*",
        "P2",
    ),
    DistScaleBackendEntry(
        "MB-3",
        "Multi-process workers (representative host topology)",
        BackendQualificationClass.MANDATORY_SCALE,
        "applications/local_workspace_application/host/background_worker_*",
        "tests/qualification/dist_scale_x/wl_01_*",
        "P2",
        operator_decision_required=True,
    ),
    DistScaleBackendEntry(
        "MB-4",
        "Message broker (RabbitMQ or Kafka — operator M)",
        BackendQualificationClass.MANDATORY_SCALE,
        "intergrax/queueing/providers/rabbitmq_task_queue.py | kafka_task_queue.py",
        "tests/qualification/dist_scale_x/wl_06_*",
        "P2/P5",
        operator_decision_required=True,
    ),
    DistScaleBackendEntry(
        "MB-5",
        "HTTP loopback (Class B tools, WL-4)",
        BackendQualificationClass.MANDATORY_SCALE,
        "TOOL-LIFE-X strict-host loopback pattern",
        "tests/qualification/dist_scale_x/wl_04_http_physical_*",
        "P3",
    ),
    DistScaleBackendEntry(
        "BK-RABBIT",
        "RabbitMQ TaskQueue",
        BackendQualificationClass.MANDATORY_SCALE,
        "rabbitmq_task_queue.py",
        "tests/qualification/dist_scale_x/wl_06_*",
        "P2",
    ),
    DistScaleBackendEntry(
        "BK-KAFKA",
        "Kafka TaskQueue",
        BackendQualificationClass.REPRESENTATIVE,
        "kafka_task_queue.py",
        "tests/qualification/dist_scale_x/wl_06_*",
        "P2 or P5",
        operator_decision_required=True,
    ),
    DistScaleBackendEntry(
        "BK-CELERY",
        "Celery TaskQueue",
        BackendQualificationClass.REPRESENTATIVE,
        "celery_task_queue.py",
        "tests/qualification/dist_scale_x/wl_01_*",
        "P2 optional",
    ),
    DistScaleBackendEntry(
        "BK-SQLITE",
        "SQLite runtime bundle",
        BackendQualificationClass.REPRESENTATIVE,
        "SQLiteRuntimePersistenceBundle",
        "tests/qualification/dist_scale_x/wl_05_*",
        "P4 smoke",
    ),
    DistScaleBackendEntry(
        "BK-INMEM",
        "In-memory / fakes",
        BackendQualificationClass.FUNCTIONAL_ONLY,
        "test doubles",
        "tests/qualification/dist_scale_x/",
        "—",
    ),
    DistScaleBackendEntry(
        "BK-REDIS",
        "Redis (distributed KV / RL)",
        BackendQualificationClass.REPRESENTATIVE,
        "KeyValueCache / distributed rate limit",
        "tests/qualification/dist_scale_x/wl_06_*",
        "P3/P5",
        operator_decision_required=True,
    ),
    DistScaleBackendEntry(
        "BK-CLOUD-HTTP",
        "Cloud LLM / SaaS HTTP",
        BackendQualificationClass.EXTERNAL_SANDBOX,
        "integrations registry",
        "PROD-Q Dimension B",
        "—",
    ),
    DistScaleBackendEntry(
        "BK-CLOUD-Q",
        "CloudTaskQueue",
        BackendQualificationClass.EXTERNAL_SANDBOX,
        "integrations/_shared/cloud_task_queue.py",
        "operator sandbox",
        "—",
    ),
    DistScaleBackendEntry(
        "BK-VECTOR",
        "Vector / graph / search stores",
        BackendQualificationClass.FUNCTIONAL_ONLY,
        "integration providers",
        "N/A WITH EVIDENCE unless host declares",
        "—",
    ),
)

DIST_SCALE_X_HISTORICAL_EVIDENCE_CATALOG: tuple[DistScaleHistoricalEvidenceEntry, ...] = (
    DistScaleHistoricalEvidenceEntry(
        "TOOL-LIFE-X",
        "1548640708c850b3fbfca3f2db0e5ecfee4274dd",
        "RE-VALIDATE",
        "Lifecycle semantics — not substitute for DIST physical scale",
    ),
    DistScaleHistoricalEvidenceEntry(
        "HARNESS-W4",
        "f38e455ccc391dad9a6a64d7867a33cfc5c9b21d",
        "REUSE",
        "Harness boundedness/cancellation — NOT SUFFICIENT for whole-platform multi-worker",
    ),
    DistScaleHistoricalEvidenceEntry(
        "STATE-X",
        "bd54941d933069b8bfb2819bb819c1cbdbe71576",
        "REUSE",
        "Semantic state/recovery — RE-VALIDATE under WL-5 load",
    ),
    DistScaleHistoricalEvidenceEntry(
        "TENANT-X",
        "5ccb72fccd4e3820ebdc9c18008afe6f7d59f558",
        "RE-VALIDATE",
        "Isolation rules — WL-2/K",
    ),
    DistScaleHistoricalEvidenceEntry(
        "BG-01",
        "tests/qualification/bg_01/catalog.py @ START_HEAD",
        "RE-VALIDATE",
        "Worker port/idempotency — replay on broker M",
    ),
)

DIST_SCALE_X_THRESHOLD_LOCK_STATE: ThresholdLockState = ThresholdLockState.NOT_LOCKED

DIST_SCALE_X_P1_L1_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/qualification/dist_scale_x/test_dist_scale_x_l1_catalog_integrity.py",
)

DIST_SCALE_X_P1_L2_PYTEST_TARGETS: tuple[str, ...] = (
    "tests/qualification/dist_scale_x/test_dist_scale_x_l2_evidence_contract_gates.py",
)

DIST_SCALE_X_P1_SELECTION_BATCH: tuple[str, ...] = (
    *DIST_SCALE_X_P1_L1_PYTEST_TARGETS,
    *DIST_SCALE_X_P1_L2_PYTEST_TARGETS,
)

# Manifest `qualification_phase` values — frozen plan §7 batches P0–P7 (not implementation_batch labels).
DIST_SCALE_X_MANIFEST_QUALIFICATION_PHASES: tuple[str, ...] = (
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)


def dist_scale_x_all_p1_gate_node_ids() -> tuple[str, ...]:
    nodes: list[str] = []
    for entry in DIST_SCALE_X_DIMENSION_CATALOG:
        nodes.extend(entry.p1_structural_gate_node_ids)
    return tuple(dict.fromkeys(nodes))


def dist_scale_x_catalog_index() -> dict[str, object]:
    return {
        "dimensions": {e.dimension_id: e for e in DIST_SCALE_X_DIMENSION_CATALOG},
        "workloads": {e.workload_id: e for e in DIST_SCALE_X_WORKLOAD_CATALOG},
        "exit_criteria": {e.exit_id: e for e in DIST_SCALE_X_EXIT_CRITERION_CATALOG},
        "backends": {e.backend_id: e for e in DIST_SCALE_X_BACKEND_CATALOG},
        "mandatory_backends": {
            e.backend_id: e for e in DIST_SCALE_X_BACKEND_CATALOG if e.backend_id in MANDATORY_BACKEND_IDS
        },
    }


def validate_dist_scale_x_catalog_mappings() -> tuple[str, ...]:
    """Return human-readable violations; empty tuple means PASS."""
    violations: list[str] = []
    dim_ids = {e.dimension_id for e in DIST_SCALE_X_DIMENSION_CATALOG}
    wl_ids = {e.workload_id for e in DIST_SCALE_X_WORKLOAD_CATALOG}
    exit_ids = {e.exit_id for e in DIST_SCALE_X_EXIT_CRITERION_CATALOG}

    if dim_ids != set(DIMENSION_IDS):
        violations.append(f"dimension id set mismatch: {sorted(dim_ids)} vs {DIMENSION_IDS}")
    if wl_ids != set(WORKLOAD_IDS):
        violations.append(f"workload id set mismatch: {sorted(wl_ids)} vs {WORKLOAD_IDS}")
    if exit_ids != set(EXIT_CRITERION_IDS):
        violations.append(f"exit criterion id set mismatch: {sorted(exit_ids)} vs {EXIT_CRITERION_IDS}")

    for entry in DIST_SCALE_X_DIMENSION_CATALOG:
        for wl in entry.workload_ids:
            if wl not in wl_ids:
                violations.append(f"dimension {entry.dimension_id} references unknown workload {wl}")
        for ex in entry.exit_criterion_ids:
            if ex not in exit_ids:
                violations.append(f"dimension {entry.dimension_id} references unknown exit {ex}")

    for entry in DIST_SCALE_X_WORKLOAD_CATALOG:
        for dim in entry.dimension_ids:
            if dim not in dim_ids:
                violations.append(f"workload {entry.workload_id} references unknown dimension {dim}")
        for ex in entry.exit_criterion_ids:
            if ex not in exit_ids:
                violations.append(f"workload {entry.workload_id} references unknown exit {ex}")
        for mb in entry.mandatory_backend_ids:
            if mb not in MANDATORY_BACKEND_IDS:
                violations.append(f"workload {entry.workload_id} references non-mandatory backend id {mb}")

    for entry in DIST_SCALE_X_EXIT_CRITERION_CATALOG:
        for dim in entry.dimension_ids:
            if dim and dim not in dim_ids:
                violations.append(f"exit {entry.exit_id} references unknown dimension {dim}")
        for wl in entry.workload_ids:
            if wl not in wl_ids:
                violations.append(f"exit {entry.exit_id} references unknown workload {wl}")

    for mb_id in MANDATORY_BACKEND_IDS:
        match = [e for e in DIST_SCALE_X_BACKEND_CATALOG if e.backend_id == mb_id]
        if not match:
            violations.append(f"mandatory backend {mb_id} missing from backend catalog")
            continue
        owner = match[0].qualification_owner.strip()
        if not owner:
            violations.append(f"mandatory backend {mb_id} has empty qualification_owner")

    for entry in DIST_SCALE_X_WORKLOAD_CATALOG:
        for dim in entry.dimension_ids:
            dim_entry = next(d for d in DIST_SCALE_X_DIMENSION_CATALOG if d.dimension_id == dim)
            if entry.workload_id not in dim_entry.workload_ids:
                violations.append(
                    f"workload {entry.workload_id} lists dimension {dim} but dimension omits workload",
                )
    for entry in DIST_SCALE_X_DIMENSION_CATALOG:
        for wl in entry.workload_ids:
            wl_entry = next(w for w in DIST_SCALE_X_WORKLOAD_CATALOG if w.workload_id == wl)
            if entry.dimension_id not in wl_entry.dimension_ids:
                violations.append(
                    f"dimension {entry.dimension_id} lists workload {wl} but workload omits dimension",
                )

    hist_classes = {"REUSE", "RE-VALIDATE", "NOT SUFFICIENT"}
    for hist in DIST_SCALE_X_HISTORICAL_EVIDENCE_CATALOG:
        if hist.reconciliation not in hist_classes:
            violations.append(f"historical {hist.stage_id} has invalid reconciliation {hist.reconciliation}")

    if DIST_SCALE_X_THRESHOLD_LOCK_STATE is not ThresholdLockState.NOT_LOCKED:
        violations.append("P1 requires thresholds remain NOT_LOCKED")

    return tuple(violations)
