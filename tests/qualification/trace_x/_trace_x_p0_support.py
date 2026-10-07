# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P0 typed traceability inventory SSOT (qualification-local; no production contracts)."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

TRACE_X_P0_START_HEAD: Final[str] = "a2eb3d6e1e748e99a0f0430b32f6a44f25345696"
TRACE_X_P0_BASELINE_COMMIT: Final[str] = "6be91ed91e3132dddd77d1fb14312bf23870e4e1"
TRACE_X_P0_AUDITED_HEAD: Final[str] = TRACE_X_P0_BASELINE_COMMIT

MANDATORY_FRZ_TRC_IDS: Final[tuple[str, ...]] = tuple(
    f"FRZ-TRC-{i:02d}" for i in range(1, 13)
)

SUPPORTING_FRZ_OBS_IDS: Final[tuple[str, ...]] = tuple(
    f"FRZ-OBS-{i:02d}" for i in range(1, 8)
)


class TraceabilityDomain(StrEnum):
    EXECUTION_IDENTITY = "EXECUTION_IDENTITY"
    PARENT_CHILD_CAUSALITY = "PARENT_CHILD_CAUSALITY"
    TRANSPORT_CAUSALITY = "TRANSPORT_CAUSALITY"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    PROVIDER_INVOCATION = "PROVIDER_INVOCATION"
    MODEL_CALL = "MODEL_CALL"
    CONTEXT_DECISION = "CONTEXT_DECISION"
    GOVERNANCE_DECISION = "GOVERNANCE_DECISION"
    SIDE_EFFECT_AUTHORIZATION = "SIDE_EFFECT_AUTHORIZATION"
    POLICY_REVISION = "POLICY_REVISION"
    PROFILE_REVISION = "PROFILE_REVISION"
    CONFIGURATION_PROVENANCE = "CONFIGURATION_PROVENANCE"
    RESTART_RESUME_CONTINUITY = "RESTART_RESUME_CONTINUITY"
    TERMINAL_OUTCOME = "TERMINAL_OUTCOME"
    DIAGNOSTIC_PROVENANCE = "DIAGNOSTIC_PROVENANCE"


class EvidencePlaneClassification(StrEnum):
    CANONICAL_EXECUTION_EVIDENCE = "CANONICAL_EXECUTION_EVIDENCE"
    CANONICAL_RELATION_EVIDENCE = "CANONICAL_RELATION_EVIDENCE"
    CANONICAL_LINEAGE_TRUTH = "CANONICAL_LINEAGE_TRUTH"
    DERIVED_FACTUAL_RECONSTRUCTION = "DERIVED_FACTUAL_RECONSTRUCTION"
    DIAGNOSTIC_READ_MODEL = "DIAGNOSTIC_READ_MODEL"
    OBSERVABILITY_PROJECTION = "OBSERVABILITY_PROJECTION"
    GOVERNANCE_EVIDENCE = "GOVERNANCE_EVIDENCE"
    CONFIGURATION_PROVENANCE = "CONFIGURATION_PROVENANCE"
    SUPPORTING_ONLY = "SUPPORTING_ONLY"
    OUTSIDE_TRACE_X = "OUTSIDE_TRACE_X"


class AuthorityRole(StrEnum):
    MINT_EXECUTION_IDENTITY = "MINT_EXECUTION_IDENTITY"
    RECORD_EXECUTION_EVIDENCE = "RECORD_EXECUTION_EVIDENCE"
    RECORD_RELATION_EVIDENCE = "RECORD_RELATION_EVIDENCE"
    RECORD_LINEAGE_TRUTH = "RECORD_LINEAGE_TRUTH"
    DERIVE_READ_MODEL = "DERIVE_READ_MODEL"
    PROJECT_DIAGNOSTIC = "PROJECT_DIAGNOSTIC"
    PROJECT_OBSERVABILITY = "PROJECT_OBSERVABILITY"
    GOVERNANCE_EVIDENCE_ONLY = "GOVERNANCE_EVIDENCE_ONLY"
    NOT_AUTHORITY_BEARING = "NOT_AUTHORITY_BEARING"


class TraceCoverageStatus(StrEnum):
    SUPPORTED_CURRENT_HEAD = "SUPPORTED_CURRENT_HEAD"
    PARTIAL_CURRENT_HEAD = "PARTIAL_CURRENT_HEAD"
    GAP_REQUIRES_CHILD = "GAP_REQUIRES_CHILD"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"


class FrzTrcP0Status(StrEnum):
    SUPPORTED_CURRENT_HEAD = "SUPPORTED_CURRENT_HEAD"
    PARTIAL_CURRENT_HEAD = "PARTIAL_CURRENT_HEAD"
    GAP_REQUIRES_CHILD = "GAP_REQUIRES_CHILD"
    NA_WITH_EVIDENCE = "N/A_WITH_EVIDENCE"


class ReverseReconstructionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TenantIsolationP0Result(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class EnterpriseAuditResult(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class BlockerClassification(StrEnum):
    IN_SCOPE_BLOCKER = "IN-SCOPE BLOCKER"
    TRACKED_FREEZE_DEBT = "TRACKED FREEZE DEBT"
    ENVIRONMENT_TEST_ISSUE = "ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED"


class TraceXChildId(StrEnum):
    P1 = "TRACE-X-P1"
    P2 = "TRACE-X-P2"
    P3 = "TRACE-X-P3"
    P4 = "TRACE-X-P4"
    P5 = "TRACE-X-P5"
    P6 = "TRACE-X-P6"
    CERT = "TRACE-X-CERT"


class SensitiveClassificationError(AssertionError):
    """Closed-world sensitive mechanism classification failure."""


FRZ_TO_CHILD: Final[dict[str, TraceXChildId]] = {
    "FRZ-TRC-01": TraceXChildId.P2,
    "FRZ-TRC-02": TraceXChildId.P1,
    "FRZ-TRC-03": TraceXChildId.P3,
    "FRZ-TRC-04": TraceXChildId.P3,
    "FRZ-TRC-05": TraceXChildId.P4,
    "FRZ-TRC-06": TraceXChildId.P3,
    "FRZ-TRC-07": TraceXChildId.P5,
    "FRZ-TRC-08": TraceXChildId.P5,
    "FRZ-TRC-09": TraceXChildId.P6,
    "FRZ-TRC-10": TraceXChildId.P6,
    "FRZ-TRC-11": TraceXChildId.P5,
    "FRZ-TRC-12": TraceXChildId.P1,
}

REVERSE_RECONSTRUCTION_CHILD_BY_SUBJECT: Final[dict[str, TraceXChildId]] = {
    "external effect": TraceXChildId.P3,
    "provider invocation": TraceXChildId.P3,
    "tool invocation": TraceXChildId.P3,
    "model call": TraceXChildId.P4,
    "failure": TraceXChildId.P6,
    "terminal outcome": TraceXChildId.P6,
}


@dataclass(frozen=True, slots=True)
class TraceabilitySurface:
    surface_id: str
    domain: TraceabilityDomain
    semantic_owner: str
    semantic_contract: str
    producer_paths: tuple[str, ...]
    persistence_owner: str | None
    reconstruction_owner: str | None
    consumer_paths: tuple[str, ...]
    evidence_plane: EvidencePlaneClassification
    authority_role: AuthorityRole
    tenant_semantics: str
    continuity_semantics: str
    frz_criteria: tuple[str, ...]
    current_status: TraceCoverageStatus
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticOwnerMatrixRow:
    concern: str
    canonical_semantic_owner: str
    canonical_contract: str
    projection_read_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArchitectureLockEntry:
    lock_id: str
    statement: str
    contract_paths: tuple[str, ...]
    contradiction_policy: str


@dataclass(frozen=True, slots=True)
class ForwardChainTransition:
    transition_id: str
    source_semantic_owner: str
    target_semantic_owner: str
    joining_identity: str
    evidence_contract: str
    producer_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReverseReconstructionRow:
    subject: str
    execution_id: ReverseReconstructionStatus
    attempt_id: ReverseReconstructionStatus
    run_id: ReverseReconstructionStatus
    task_id: ReverseReconstructionStatus
    tenant: ReverseReconstructionStatus
    parent_execution: ReverseReconstructionStatus
    governance_evidence: ReverseReconstructionStatus
    policy_profile_revision: ReverseReconstructionStatus
    provider_contract: ReverseReconstructionStatus
    future_child_owner: TraceXChildId | None
    evidence_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrzTrcP0MatrixRow:
    criterion: str
    p0_status: FrzTrcP0Status
    contracts: tuple[str, ...]
    code_evidence: tuple[str, ...]
    test_evidence: tuple[str, ...]
    gap: str
    future_child_owner: str


@dataclass(frozen=True, slots=True)
class TraceXChildProposal:
    child_id: str
    scope: str
    frz_criteria: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class TraceXKnownBlocker:
    blocker_id: str
    title: str
    owner_child: str
    classification: BlockerClassification
    evidence: str


@dataclass(frozen=True, slots=True)
class HistoricalEvidenceReconciliation:
    stage_id: str
    proves: str
    supports_frz_trc: tuple[str, ...]
    does_not_prove: str


@dataclass(frozen=True, slots=True)
class EnterpriseAuditMatrixRow:
    area: str
    result: EnterpriseAuditResult


CLOSED_WORLD_ROOTS: Final[tuple[str, ...]] = (
    "intergrax/contracts",
    "intergrax/runtime/events",
    "intergrax/runtime/observability",
    "intergrax/runtime/diagnostics",
    "intergrax/runtime/execution",
    "intergrax/runtime/governance",
    "intergrax/runtime/nexus",
    "intergrax/runtime/external_operations",
    "intergrax/runtime/replay",
    "intergrax/tools",
    "intergrax/llm_adapters",
    "intergrax/integrations",
    "agents",
    "applications",
    "intergrax/applications",
)

ARCHITECTURE_LOCK: Final[tuple[ArchitectureLockEntry, ...]] = (
    ArchitectureLockEntry(
        "TX-LOCK-01",
        "RuntimeEvent = canonical execution event evidence (NOT Governance permission).",
        ("intergrax/contracts/runtime_event.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-02",
        "ExecutionLineage = canonical execution topology/lineage truth.",
        ("intergrax/contracts/execution_lineage.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-03",
        "PlatformCausalEvidence = canonical cross-boundary transport→execution relation.",
        ("intergrax/contracts/platform_causal_evidence.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-04",
        "ExecutionReconstructor = exactly-one factual reconstruction owner.",
        (
            "intergrax/contracts/execution_reconstruction.py",
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
        ),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-05",
        "TraceEvent / RunTraceStore = Plane B diagnostic/read-model telemetry.",
        (
            "intergrax/contracts/tracing/events.py",
            "intergrax/contracts/run_trace_store.py",
        ),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-06",
        "Diagnostics = interpretation only; consumes ExecutionReconstructionReader.",
        ("intergrax/runtime/diagnostics/diagnostic_orchestrator.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-07",
        "Observability/export = projection/delivery only; no execution truth minting.",
        ("intergrax/runtime/observability/emitter.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-08",
        "ExecutionReconstruction = derived read model; NOT persisted; NOT source of truth.",
        ("intergrax/contracts/execution_reconstruction_models.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
    ArchitectureLockEntry(
        "TX-LOCK-09",
        "ExecutionBoundaryEvent = governed external/provider evidence path (no competing spine).",
        ("intergrax/contracts/execution_evidence/boundary_event.py",),
        "STOP — ARCHITECTURE DECISION REQUIRED",
    ),
)

SEMANTIC_OWNER_MATRIX: Final[tuple[SemanticOwnerMatrixRow, ...]] = (
    SemanticOwnerMatrixRow(
        "execution event truth",
        "Evidence Plane / RuntimeEvent persistence",
        "RuntimeEvent @ intergrax/contracts/runtime_event.py",
        ("RuntimeEventExportSink", "Run journal projections"),
    ),
    SemanticOwnerMatrixRow(
        "cross-transport causal relation",
        "Platform causal evidence subsystem",
        "PlatformCausalEvidence @ intergrax/contracts/platform_causal_evidence.py",
        ("ExecutionReconstruction causal join",),
    ),
    SemanticOwnerMatrixRow(
        "execution lineage",
        "Execution Lineage subsystem",
        "ExecutionLineagePersistence @ intergrax/contracts/execution_lineage.py",
        ("ExecutionReconstruction lineage join",),
    ),
    SemanticOwnerMatrixRow(
        "factual reconstruction",
        "Evidence Plane / ExecutionReconstructor",
        "ExecutionReconstructionReader @ intergrax/contracts/execution_reconstruction.py",
        ("ExecutionReconstruction derived model",),
    ),
    SemanticOwnerMatrixRow(
        "diagnostic interpretation",
        "Diagnostics orchestrator",
        "ExecutionReconstructionReader (consumer contract)",
        ("DiagnosticFinding projections",),
    ),
    SemanticOwnerMatrixRow(
        "trace read model",
        "Nexus tracing / RunTrace plane",
        "TraceEvent @ intergrax/contracts/tracing/events.py",
        ("trace_bridge RuntimeEvent projection",),
    ),
    SemanticOwnerMatrixRow(
        "governance evidence",
        "Governance evidence persistence",
        "GovernanceEvidenceSection @ intergrax/contracts/execution_evidence/boundary_event.py",
        ("RuntimeEvent governance payloads",),
    ),
    SemanticOwnerMatrixRow(
        "provider-effect evidence",
        "Execution boundary evidence + external operations",
        "ExecutionBoundaryEvent @ intergrax/contracts/execution_evidence/boundary_event.py",
        ("ProviderInvocationSection", "external operation recorders"),
    ),
)

_TRACEABILITY_SURFACES: tuple[TraceabilitySurface, ...] = (
    TraceabilitySurface(
        "TX-S01",
        TraceabilityDomain.EXECUTION_IDENTITY,
        "Evidence Plane / RuntimeEvent",
        "RuntimeEvent @ intergrax/contracts/runtime_event.py",
        (
            "intergrax/runtime/events/event_bus.py",
            "intergrax/runtime/events/trace_bridge.py",
        ),
        "EvidencePersistencePort / RuntimeEventPersistence",
        "ExecutionReconstructor",
        (
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
            "intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
        ),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "tenant_id on RuntimeEvent; tenant-scoped persistence queries",
        "event_id/parent_event_id/correlation_id/traceparent continuity",
        ("FRZ-TRC-01", "FRZ-TRC-09", "FRZ-TRC-10", "FRZ-TRC-12"),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        (
            "STATE-X SX-F07",
            "tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",
        ),
    ),
    TraceabilitySurface(
        "TX-S02",
        TraceabilityDomain.EXECUTION_IDENTITY,
        "RuntimeEventBus + identity minting policy",
        "RuntimeEventIdentityKwargs @ intergrax/runtime/events/runtime_event_identity.py",
        ("intergrax/runtime/events/event_bus.py",),
        "RuntimeEventPersistence",
        None,
        ("intergrax/runtime/events/evidence_persistence_adapter.py",),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.MINT_EXECUTION_IDENTITY,
        "tenant_id required on publish path",
        "attempt_id/execution_id bound at sanctioned publish sites only",
        ("FRZ-TRC-01",),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("intergrax/runtime/events/runtime_event_identity.py",),
    ),
    TraceabilitySurface(
        "TX-S03",
        TraceabilityDomain.DIAGNOSTIC_PROVENANCE,
        "Plane B TraceEvent",
        "TraceEvent @ intergrax/contracts/tracing/events.py",
        (
            "intergrax/runtime/task/task_trace.py",
            "intergrax/runtime/observability/emitter.py",
        ),
        "RunTraceStore (run-scoped)",
        None,
        (
            "intergrax/runtime/events/trace_bridge.py",
            "intergrax/runtime/replay/trace_replay_bridge.py",
        ),
        EvidencePlaneClassification.DIAGNOSTIC_READ_MODEL,
        AuthorityRole.PROJECT_DIAGNOSTIC,
        "run_id scope only; no tenant_id on TraceEvent contract",
        "run-scoped chronology; not restart truth owner",
        ("FRZ-TRC-01",),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        (
            "tests/unit/runtime/observability/test_obs_trace_1_qualification.py",
            "OBS_TRACE_1_VERDICT=NOT_REQUIRED",
        ),
    ),
    TraceabilitySurface(
        "TX-S04",
        TraceabilityDomain.TRANSPORT_CAUSALITY,
        "Platform causal evidence",
        "PlatformCausalEvidence @ intergrax/contracts/platform_causal_evidence.py",
        (
            "intergrax/runtime/observability/causal_evidence_persistence.py",
            "intergrax/runtime/observability/causal_evidence_enrichment.py",
            "intergrax/runtime/background_execution/required_audit_evidence.py",
        ),
        "CausalEvidencePersistence",
        "ExecutionReconstructor",
        ("intergrax/runtime/observability/reconstruction/execution_reconstruction.py",),
        EvidencePlaneClassification.CANONICAL_RELATION_EVIDENCE,
        AuthorityRole.RECORD_RELATION_EVIDENCE,
        "tenant on MessageBusTaskRef and RuntimeExecutionRef",
        "TRANSPORT_TASK_TRIGGERED_EXECUTION relation kind",
        ("FRZ-TRC-01", "FRZ-TRC-12"),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("STATE-X F16 identity persistence", "GOV-X2 identity nodes"),
    ),
    TraceabilitySurface(
        "TX-S05",
        TraceabilityDomain.PARENT_CHILD_CAUSALITY,
        "Execution Lineage",
        "ExecutionLineagePersistence @ intergrax/contracts/execution_lineage.py",
        ("intergrax/runtime/execution/lineage/",),
        "ExecutionLineagePersistence port implementations",
        "ExecutionReconstructor",
        ("intergrax/runtime/observability/reconstruction/execution_reconstruction.py",),
        EvidencePlaneClassification.CANONICAL_LINEAGE_TRUTH,
        AuthorityRole.RECORD_LINEAGE_TRUTH,
        "tenant-scoped lineage records",
        "parent execution / child execution topology",
        ("FRZ-TRC-02", "FRZ-TRC-09"),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",),
    ),
    TraceabilitySurface(
        "TX-S06",
        TraceabilityDomain.EXECUTION_IDENTITY,
        "ExecutionReconstructor (derived factual view)",
        "ExecutionReconstructionReader @ intergrax/contracts/execution_reconstruction.py",
        (
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
        ),
        None,
        "ExecutionReconstructor",
        ("intergrax/runtime/diagnostics/diagnostic_orchestrator.py",),
        EvidencePlaneClassification.DERIVED_FACTUAL_RECONSTRUCTION,
        AuthorityRole.DERIVE_READ_MODEL,
        "reconstruct_execution(tenant_id, task_id, run_id)",
        "NOT persisted; joins evidence + causal + optional lineage",
        ("FRZ-TRC-01", "FRZ-TRC-10"),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",),
    ),
    TraceabilitySurface(
        "TX-S07",
        TraceabilityDomain.TOOL_INVOCATION,
        "Execution-bound tool invokers",
        "ExecutionBoundDeclarativeToolInvoker @ intergrax/contracts/execution_bound_declarative_tool_invocation.py",
        (
            "intergrax/runtime/nexus/tools/invoker.py",
            "intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py",
            "intergrax/runtime/tools/",
        ),
        "RuntimeEvent persistence",
        "ExecutionReconstructor",
        (
            "intergrax/runtime/nexus/tools/invoker.py",
            "tests/unit/runtime/tools/test_fresh_side_effect_authorization.py",
        ),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "execution identity bound at invoker boundary",
        "invocation_id / tool call RuntimeEvents",
        ("FRZ-TRC-03", "FRZ-TRC-06"),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X2 tool/effect matrix", "tests/qualification/governance/gov_x2/"),
    ),
    TraceabilitySurface(
        "TX-S08",
        TraceabilityDomain.PROVIDER_INVOCATION,
        "Provider / external operation evidence",
        "ProviderInvocationSection @ intergrax/contracts/execution_evidence/boundary_event.py",
        (
            "intergrax/runtime/external_operations/",
            "intergrax/llm_adapters/",
        ),
        "RuntimeEvent + boundary evidence paths",
        "ExecutionReconstructor",
        ("intergrax/runtime/governance/",),
        EvidencePlaneClassification.GOVERNANCE_EVIDENCE,
        AuthorityRole.GOVERNANCE_EVIDENCE_ONLY,
        "tenant + provider scope on boundary sections",
        "provider invocation correlated via boundary + RuntimeEvent",
        ("FRZ-TRC-04", "FRZ-TRC-06"),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X2 provider nodes",),
    ),
    TraceabilitySurface(
        "TX-S09",
        TraceabilityDomain.MODEL_CALL,
        "LLM adapter invocation evidence",
        "RuntimeEventType + LLM adapter emitters",
        ("intergrax/llm_adapters/", "intergrax/runtime/nexus/"),
        "RuntimeEvent persistence",
        "ExecutionReconstructor",
        ("intergrax/runtime/observability/",),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "tenant via execution binding",
        "model/provider call events on execution timeline",
        ("FRZ-TRC-05",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        (
            "HARNESS-W5/W6",
            "CE-01 tests/qualification/ce_01/",
            "HARNESS_RESIDUAL_CE_01 scoped evidence",
        ),
    ),
    TraceabilitySurface(
        "TX-S10",
        TraceabilityDomain.CONTEXT_DECISION,
        "Context assembly (CE-01 path)",
        "Context assembly contracts + RuntimeEvent payloads",
        ("agents/", "intergrax/runtime/nexus/"),
        "RuntimeEvent persistence (scoped)",
        "ExecutionReconstructor",
        ("tests/qualification/ce_01/",),
        EvidencePlaneClassification.SUPPORTING_ONLY,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "CE-01 qualified path only",
        "context decision attribution partial globally",
        ("FRZ-TRC-05",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("CE-01 @ a8d47a6396c3e5e73a979be74429ed59a93e1551",),
    ),
    TraceabilitySurface(
        "TX-S11",
        TraceabilityDomain.GOVERNANCE_DECISION,
        "Governance decision evidence",
        "PolicyDecisionSection @ intergrax/contracts/execution_evidence/boundary_event.py",
        (
            "intergrax/runtime/governance/",
            "intergrax/runtime/policy/",
        ),
        "GovernanceEvidencePersistencePort",
        "ExecutionReconstructor (read join)",
        ("tests/qualification/governance/gov_x2/",),
        EvidencePlaneClassification.GOVERNANCE_EVIDENCE,
        AuthorityRole.GOVERNANCE_EVIDENCE_ONLY,
        "tenant-scoped governance records",
        "decision_ref / bundle identity fields",
        ("FRZ-TRC-06", "FRZ-TRC-07"),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X1", "GOV-X2"),
    ),
    TraceabilitySurface(
        "TX-S12",
        TraceabilityDomain.POLICY_REVISION,
        "Policy bundle / revision identity",
        "bundle_id/bundle_version/bundle_digest + decision_ref (multi-field)",
        (
            "intergrax/runtime/governance/",
            "intergrax/runtime/policy/",
        ),
        "Governance + RuntimeEvent payloads",
        None,
        ("tests/qualification/governance/",),
        EvidencePlaneClassification.GOVERNANCE_EVIDENCE,
        AuthorityRole.GOVERNANCE_EVIDENCE_ONLY,
        "tenant-scoped",
        "multiple representations; canonical owner = Governance evidence plane",
        ("FRZ-TRC-07",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X2 stale/revision negatives",),
    ),
    TraceabilitySurface(
        "TX-S13",
        TraceabilityDomain.PROFILE_REVISION,
        "Effective profile / sandbox revision",
        "Profile revision fields (distributed; no global SSOT yet)",
        (
            "intergrax/runtime/execution/",
            "intergrax/runtime/nexus/",
        ),
        None,
        None,
        (),
        EvidencePlaneClassification.SUPPORTING_ONLY,
        AuthorityRole.NOT_AUTHORITY_BEARING,
        "partial propagation via admission/sandbox projections",
        "no global effective-profile revision trace SSOT",
        ("FRZ-TRC-08",),
        TraceCoverageStatus.GAP_REQUIRES_CHILD,
        ("P0 inventory: no meaningful global checklist evidence",),
    ),
    TraceabilitySurface(
        "TX-S14",
        TraceabilityDomain.CONFIGURATION_PROVENANCE,
        "Configuration fingerprint / revision (INT-CONFIG scope)",
        "INT-CONFIG realization evidence",
        ("intergrax/integrations/",),
        "configuration realization stores (scoped)",
        None,
        ("tests/qualification/existing_capability_configuration/",),
        EvidencePlaneClassification.CONFIGURATION_PROVENANCE,
        AuthorityRole.NOT_AUTHORITY_BEARING,
        "INT-CONFIG qualified tenant continuity",
        "configured≠effective within INT-CONFIG only",
        ("FRZ-TRC-11",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("INT-CONFIG-REAL-X-CERT @ a59744517b92847f55def1db22826d17d89ee155",),
    ),
    TraceabilitySurface(
        "TX-S15",
        TraceabilityDomain.RESTART_RESUME_CONTINUITY,
        "Checkpoint / continuation / causal re-bind",
        "TaskCheckpoint + ExecutionContinuation + PlatformCausalEvidence",
        (
            "intergrax/runtime/long_running/",
            "intergrax/runtime/execution/continuation/",
        ),
        "STATE-X durable families",
        "ExecutionReconstructor",
        ("STATE-X qualification",),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "tenant preserved across resume (STATE-X evidence)",
        "attempt/execution lineage after restart",
        ("FRZ-TRC-09",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("STATE-X @ bd54941d933069b8bfb2819bb819c1cbdbe71576",),
    ),
    TraceabilitySurface(
        "TX-S16",
        TraceabilityDomain.TERMINAL_OUTCOME,
        "Terminal execution outcome evidence",
        "ExecutionTerminalStore @ intergrax/contracts/execution_terminal.py",
        (
            "intergrax/runtime/execution/execution_terminal/",
            "intergrax/runtime/events/",
        ),
        "ExecutionTerminalPersistence",
        "ExecutionReconstructor",
        ("intergrax/runtime/diagnostics/",),
        EvidencePlaneClassification.CANONICAL_EXECUTION_EVIDENCE,
        AuthorityRole.RECORD_EXECUTION_EVIDENCE,
        "tenant-scoped terminal records",
        "terminal linked via reconstruction not Diagnostics invention",
        ("FRZ-TRC-10",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X2 terminal nodes", "STATE-X SX-F05"),
    ),
    TraceabilitySurface(
        "TX-S17",
        TraceabilityDomain.DIAGNOSTIC_PROVENANCE,
        "Diagnostics orchestrator",
        "ExecutionReconstructionReader consumer",
        ("intergrax/runtime/diagnostics/diagnostic_orchestrator.py",),
        None,
        None,
        ("intergrax/runtime/diagnostics/functional_evidence_reconstruction.py",),
        EvidencePlaneClassification.DIAGNOSTIC_READ_MODEL,
        AuthorityRole.PROJECT_DIAGNOSTIC,
        "tenant via reconstruction inputs",
        "interpretation only",
        ("FRZ-TRC-10",),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py",),
    ),
    TraceabilitySurface(
        "TX-S18",
        TraceabilityDomain.DIAGNOSTIC_PROVENANCE,
        "Observability emitter / export",
        "ObservabilityEmitter @ intergrax/runtime/observability/emitter.py",
        ("intergrax/runtime/observability/emitter.py",),
        None,
        None,
        ("intergrax/runtime/observability/event_delivery/",),
        EvidencePlaneClassification.OBSERVABILITY_PROJECTION,
        AuthorityRole.PROJECT_OBSERVABILITY,
        "projection preserves tenant from source facts",
        "delivery only",
        ("FRZ-TRC-01",),
        TraceCoverageStatus.SUPPORTED_CURRENT_HEAD,
        ("OBS-TRACE-1",),
    ),
    TraceabilitySurface(
        "TX-S19",
        TraceabilityDomain.SIDE_EFFECT_AUTHORIZATION,
        "Fresh side-effect authorization coupling",
        "ExecutionBoundaryEvent sections + tool invoker gates",
        (
            "intergrax/runtime/tools/",
            "intergrax/runtime/nexus/tools/",
        ),
        "RuntimeEvent + governance evidence",
        "ExecutionReconstructor",
        ("tests/unit/runtime/tools/test_fresh_side_effect_authorization.py",),
        EvidencePlaneClassification.GOVERNANCE_EVIDENCE,
        AuthorityRole.GOVERNANCE_EVIDENCE_ONLY,
        "tenant-scoped authorization",
        "side effect → provider → governance decision chain composable on qualified paths",
        ("FRZ-TRC-06",),
        TraceCoverageStatus.PARTIAL_CURRENT_HEAD,
        ("GOV-X2", "INT-CONFIG authorization provenance (scoped)"),
    ),
)

TRACEABILITY_SURFACES: Final[tuple[TraceabilitySurface, ...]] = _TRACEABILITY_SURFACES

FORWARD_CHAIN: Final[tuple[ForwardChainTransition, ...]] = (
    ForwardChainTransition(
        "TX-FWD-01",
        "transport provider/task",
        "PlatformCausalEvidence",
        "MessageBusTaskRef → RuntimeExecutionRef",
        "PlatformCausalEvidence",
        ("intergrax/runtime/background_execution/required_audit_evidence.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-02",
        "PlatformCausalEvidence",
        "RuntimeEvent execution identity",
        "TaskId/RunId/AttemptId/ExecutionId + tenant_id",
        "RuntimeEvent",
        ("intergrax/runtime/events/event_bus.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-03",
        "RuntimeEvent timeline",
        "ExecutionReconstruction",
        "tenant_id/task_id/run_id",
        "ExecutionReconstructionReader",
        (
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
        ),
    ),
    ForwardChainTransition(
        "TX-FWD-04",
        "ExecutionLineage",
        "ExecutionReconstruction parent topology",
        "ExecutionLineageReader join",
        "ExecutionLineagePersistence",
        ("intergrax/runtime/execution/lineage/",),
    ),
    ForwardChainTransition(
        "TX-FWD-05",
        "TraceEvent",
        "RuntimeEvent (sanctioned bridge)",
        "current execution identity at bridge",
        "trace_bridge",
        ("intergrax/runtime/events/trace_bridge.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-06",
        "tool invoker",
        "RuntimeEvent tool invocation evidence",
        "execution-bound invocation identity",
        "ExecutionBound*ToolInvoker contracts",
        ("intergrax/runtime/nexus/tools/invoker.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-07",
        "governance decision",
        "provider invocation / boundary evidence",
        "decision_ref + ProviderInvocationSection",
        "ExecutionBoundaryEvent",
        ("intergrax/contracts/execution_evidence/boundary_event.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-08",
        "ExecutionReconstruction",
        "Diagnostics findings",
        "ExecutionReconstructionReader",
        "ExecutionReconstructionReader",
        ("intergrax/runtime/diagnostics/diagnostic_orchestrator.py",),
    ),
    ForwardChainTransition(
        "TX-FWD-09",
        "terminal evidence",
        "ExecutionReconstruction terminal linkage",
        "ExecutionTerminalStore + RuntimeEvents",
        "ExecutionTerminalStore",
        ("intergrax/runtime/execution/execution_terminal/",),
    ),
)

REVERSE_RECONSTRUCTION_MATRIX: Final[tuple[ReverseReconstructionRow, ...]] = (
    ReverseReconstructionRow(
        "external effect",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        TraceXChildId.P3,
        (
            "ExecutionBoundaryEvent + GOV-X2 qualified paths",
            "global reverse closure → TRACE-X-P3",
        ),
    ),
    ReverseReconstructionRow(
        "provider invocation",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        TraceXChildId.P3,
        ("ProviderInvocationSection inventory",),
    ),
    ReverseReconstructionRow(
        "tool invocation",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        TraceXChildId.P3,
        ("Tool invoker surfaces TX-S07",),
    ),
    ReverseReconstructionRow(
        "model call",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.PARTIAL,
        TraceXChildId.P4,
        ("CE-01 scoped; global → TRACE-X-P4",),
    ),
    ReverseReconstructionRow(
        "failure",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        TraceXChildId.P6,
        ("RuntimeEvent failure recorders",),
    ),
    ReverseReconstructionRow(
        "diagnostic finding",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_APPLICABLE,
        None,
        ("Diagnostics consumes reconstruction only",),
    ),
    ReverseReconstructionRow(
        "terminal outcome",
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.PARTIAL,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        ReverseReconstructionStatus.NOT_AVAILABLE,
        TraceXChildId.P6,
        ("TX-S16 + GOV-X2 partial",),
    ),
)

FRZ_TRC_P0_MATRIX: Final[tuple[FrzTrcP0MatrixRow, ...]] = (
    FrzTrcP0MatrixRow(
        "FRZ-TRC-01",
        FrzTrcP0Status.SUPPORTED_CURRENT_HEAD,
        (
            "intergrax/contracts/runtime_event.py",
            "intergrax/contracts/execution_reconstruction.py",
            "intergrax/contracts/platform_causal_evidence.py",
        ),
        (
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
            "intergrax/runtime/events/event_bus.py",
        ),
        (
            "tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",
            "tests/qualification/trace_x/test_trace_x_p0_baseline.py",
        ),
        "End-to-end adversarial certification → TRACE-X-P2",
        "TRACE-X-P2",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-02",
        FrzTrcP0Status.SUPPORTED_CURRENT_HEAD,
        ("intergrax/contracts/execution_lineage.py",),
        ("intergrax/runtime/execution/lineage/",),
        ("tests/qualification/trace_x/test_trace_x_p0_baseline.py",),
        "Adversarial parent-child matrix → TRACE-X-P1",
        "TRACE-X-P1",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-03",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        (
            "intergrax/contracts/execution_bound_declarative_tool_invocation.py",
            "intergrax/contracts/execution_bound_catalog_tool_invocation.py",
        ),
        (
            "intergrax/runtime/nexus/tools/invoker.py",
            "intergrax/runtime/nexus/tools/nexus_execution_bound_catalog_tool_invoker.py",
        ),
        (
            "tests/unit/runtime/tools/test_fresh_side_effect_authorization.py",
            "tests/qualification/governance/gov_x2/test_gov_x2_qualification_batch.py",
        ),
        "Global tool attribution closure",
        "TRACE-X-P3",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-04",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        ("intergrax/contracts/execution_evidence/boundary_event.py",),
        (
            "intergrax/runtime/external_operations/",
            "intergrax/llm_adapters/",
        ),
        ("tests/qualification/governance/gov_x2/",),
        "Global provider attribution closure",
        "TRACE-X-P3",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-05",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        ("RuntimeEvent LLM payloads", "CE-01 context assembly"),
        ("intergrax/llm_adapters/", "tests/qualification/ce_01/"),
        ("HARNESS_RESIDUAL_CE_01 qualification",),
        "Execution-wide model/context join",
        "TRACE-X-P4",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-06",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        ("ExecutionBoundaryEvent sections",),
        (
            "intergrax/runtime/governance/",
            "intergrax/runtime/tools/",
        ),
        ("tests/qualification/governance/gov_x2/",),
        "Global side-effect→authorization chain",
        "TRACE-X-P3",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-07",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        ("PolicyDecisionSection", "bundle_id/bundle_version/decision_ref"),
        ("intergrax/runtime/governance/", "intergrax/runtime/policy/"),
        ("GOV-X1", "GOV-X2"),
        "Canonical policy revision SSOT for trace",
        "TRACE-X-P5",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-08",
        FrzTrcP0Status.GAP_REQUIRES_CHILD,
        ("distributed profile fields",),
        ("intergrax/runtime/execution/",),
        (),
        "No global effective profile revision trace SSOT",
        "TRACE-X-P5",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-09",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        (
            "TaskCheckpoint",
            "ExecutionContinuation",
            "PlatformCausalEvidence",
        ),
        ("intergrax/runtime/long_running/", "STATE-X qualification"),
        ("tests/qualification/state_x/",),
        "Trace continuity adversarial proof",
        "TRACE-X-P6",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-10",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        (
            "ExecutionTerminalStore",
            "RuntimeEvent",
            "ExecutionReconstruction",
        ),
        (
            "intergrax/runtime/execution/terminal/",
            "intergrax/runtime/diagnostics/",
        ),
        ("GOV-X2", "tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py"),
        "Terminal↔causal adversarial linkage",
        "TRACE-X-P6",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-11",
        FrzTrcP0Status.PARTIAL_CURRENT_HEAD,
        ("INT-CONFIG fingerprint/revision",),
        ("tests/qualification/existing_capability_configuration/",),
        ("INT-CONFIG-REAL-X-CERT",),
        "Global runtime configured→effective→invocation",
        "TRACE-X-P5",
    ),
    FrzTrcP0MatrixRow(
        "FRZ-TRC-12",
        FrzTrcP0Status.SUPPORTED_CURRENT_HEAD,
        ("intergrax/contracts/platform_causal_evidence.py",),
        ("intergrax/runtime/observability/causal_evidence_persistence.py",),
        ("STATE-X F16", "tests/qualification/trace_x/test_trace_x_p0_baseline.py"),
        "Adversarial transport↔runtime mapping proof",
        "TRACE-X-P1",
    ),
)

TRACE_X_CHILD_DECOMPOSITION: Final[tuple[TraceXChildProposal, ...]] = (
    TraceXChildProposal(
        "TRACE-X-P1",
        "Identity, transport mapping, parent-child causality adversarial closure",
        ("FRZ-TRC-02", "FRZ-TRC-12"),
        "Derived from SUPPORTED/PARTIAL baseline; needs certification not inventory",
    ),
    TraceXChildProposal(
        "TRACE-X-P2",
        "Execution causal chain end-to-end adversarial reconstruction",
        ("FRZ-TRC-01",),
        "ExecutionReconstructionReader proof beyond architecture gates",
    ),
    TraceXChildProposal(
        "TRACE-X-P3",
        "Tool, provider, side-effect authorization attribution closure",
        ("FRZ-TRC-03", "FRZ-TRC-04", "FRZ-TRC-06"),
        "GOV-X2 supporting only; global TRACE-X ownership required",
    ),
    TraceXChildProposal(
        "TRACE-X-P4",
        "Model call and context decision provenance for one execution",
        ("FRZ-TRC-05",),
        "CE-01 scoped; extend to platform-wide reconstruction",
    ),
    TraceXChildProposal(
        "TRACE-X-P5",
        "Policy, profile, and configured/effective provenance",
        ("FRZ-TRC-07", "FRZ-TRC-08", "FRZ-TRC-11"),
        "FRZ-TRC-08 GAP; CONFIG-X remains future debt",
    ),
    TraceXChildProposal(
        "TRACE-X-P6",
        "Restart/resume continuity, terminal outcome, reverse reconstruction",
        ("FRZ-TRC-09", "FRZ-TRC-10"),
        "Reverse matrix PARTIAL rows",
    ),
    TraceXChildProposal(
        "TRACE-X-CERT",
        "Final adversarial TRACE-X certification and FRZ-TRC PASS promotion",
        MANDATORY_FRZ_TRC_IDS,
        "After child gaps closed; independent audit only",
    ),
)

TRACE_X_KNOWN_BLOCKERS: Final[tuple[TraceXKnownBlocker, ...]] = (
    TraceXKnownBlocker(
        "TX-B01",
        "Global effective profile revision trace SSOT missing (FRZ-TRC-08)",
        "TRACE-X-P5",
        BlockerClassification.TRACKED_FREEZE_DEBT,
        "P0 inventory TX-S13 GAP_REQUIRES_CHILD; not CONFIG-X implementation",
    ),
)

HISTORICAL_EVIDENCE: Final[tuple[HistoricalEvidenceReconciliation, ...]] = (
    HistoricalEvidenceReconciliation(
        "HARNESS-W5",
        "Harness execution/tool reachability on qualified paths",
        ("FRZ-TRC-03",),
        "Not global TRACE-X PASS",
    ),
    HistoricalEvidenceReconciliation(
        "HARNESS-W6",
        "Harness continuation/residual surfaces",
        ("FRZ-TRC-09",),
        "Not restart trace continuity PASS",
    ),
    HistoricalEvidenceReconciliation(
        "CE-01",
        "Context assembly attribution on CE qualified path",
        ("FRZ-TRC-05",),
        "Not execution-wide model/context PASS",
    ),
    HistoricalEvidenceReconciliation(
        "GOV-X1",
        "Governance authority boundary",
        ("FRZ-TRC-06", "FRZ-TRC-07", "FRZ-TRC-10"),
        "Partial scoped contribution only",
    ),
    HistoricalEvidenceReconciliation(
        "GOV-X2",
        "Governance+Execution E2E on certified scope",
        ("FRZ-TRC-03", "FRZ-TRC-04", "FRZ-TRC-06", "FRZ-TRC-07", "FRZ-TRC-09", "FRZ-TRC-10", "FRZ-TRC-12"),
        "FRZ-TRC remain OPEN; supporting evidence only",
    ),
    HistoricalEvidenceReconciliation(
        "INT-CONFIG-REAL-X",
        "Configuration realization adversarial scope",
        ("FRZ-TRC-06", "FRZ-TRC-11"),
        "Not CONFIG-X or global configured→effective PASS",
    ),
    HistoricalEvidenceReconciliation(
        "STATE-X",
        "Durable state, recovery, identity persistence F16",
        ("FRZ-TRC-09", "FRZ-TRC-12"),
        "Not TRACE-X causal certification",
    ),
    HistoricalEvidenceReconciliation(
        "OBS-TRACE-1",
        "TraceEvent Plane B; no execution identity on TraceEvent",
        ("FRZ-TRC-01",),
        "NOT_REQUIRED to add attempt/execution to TraceEvent",
    ),
    HistoricalEvidenceReconciliation(
        "OBS-RECONSTRUCTION-1",
        "Single ExecutionReconstructor; diagnostics consumes reader",
        ("FRZ-TRC-01", "FRZ-TRC-10"),
        "Not full causal chain PASS",
    ),
    HistoricalEvidenceReconciliation(
        "OBS-DIAG-CONFORMANCE",
        "Diagnostics boundary conformance",
        ("FRZ-TRC-10",),
        "Not terminal outcome global PASS",
    ),
)

TENANT_ISOLATION_AUDIT: Final[dict[str, str]] = {
    "tenant_scope_applicable": "YES",
    "canonical_tenant_identity": "tenant_id on RuntimeEvent, causal evidence, lineage, reconstruction inputs",
    "tenant_owner": "Execution identity + evidence plane owners (not TraceEvent)",
    "propagation_path": "transport evidence → runtime execution evidence → persistence → reconstruction → diagnostics/observability",
    "state_isolation": "STATE-X supporting evidence only",
    "provider_config_isolation": "TRACKED FREEZE DEBT → CONFIG-X / TENANT-X",
    "evidence_trace_isolation": "P0 inventory surfaces TX-S01..S19; cross-tenant must fail closed (qualified paths)",
    "async_recovery_continuity": "PARTIAL — STATE-X + causal re-bind; TRACE-X-P6",
    "cross_tenant_path": "must fail closed on qualified governance/config paths",
    "fail_closed_behavior": "GOV-X2 + INT-CONFIG adversarial tests (supporting)",
    "adversarial_evidence": "GOV-X2, INT-CONFIG-REAL-X-CERT, STATE-X tenant audits (supporting)",
    "result": TenantIsolationP0Result.PARTIAL.value,
}

ENTERPRISE_AUDIT_MATRIX: Final[tuple[EnterpriseAuditMatrixRow, ...]] = (
    EnterpriseAuditMatrixRow("execution event truth owner", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("causal relation owner", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("lineage owner", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("reconstruction owner", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("TraceEvent non-authoritative", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("Diagnostics non-authoritative", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("Observability non-authoritative", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("transport/runtime identity separation", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("parent-child ownership", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("tool attribution inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("provider attribution inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("model/context inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("authorization/effect inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("policy provenance inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("profile provenance inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("configuration provenance inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("restart/resume continuity inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("terminal evidence inventory", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("reverse reconstruction coverage", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("contracts-over-implementations", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("strong typing", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("pluginability/replaceability", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("bypass resistance", EnterpriseAuditResult.PASS),
    EnterpriseAuditMatrixRow("tenant isolation P0 audit", EnterpriseAuditResult.PARTIAL),
    EnterpriseAuditMatrixRow("FRZ decomposition complete", EnterpriseAuditResult.PASS),
)

# Closed-world class registry: sensitive reconstruction/causal/trace authority classes.
TRACE_MECHANISM_CLASS_REGISTRY: Final[dict[str, str]] = {
    "RuntimeEvent": "TX-S01",
    "RuntimeEventBus": "TX-S01",
    "TraceEvent": "TX-S03",
    "TraceEventDTO": "TX-S03",
    "ExecutionReconstructor": "TX-S06",
    "FunctionalEvidenceReconstructor": "TX-S17",
    "PlatformCausalEvidence": "TX-S04",
    "LegacyPlatformCausalEvidence": "TX-S04",
    "DecodedPlatformCausalEvidence": "TX-S04",
    "ExecutionReconstruction": "TX-S06",
    "ExecutionReconstructionReader": "TX-S06",
    "RunTraceStore": "TX-S03",
    "RunTraceReader": "TX-S03",
    "ExecutionLineageReader": "TX-S05",
    "ExecutionLineagePersistence": "TX-S05",
    "DocumentStoreExecutionLineagePersistence": "TX-S05",
    "InMemoryExecutionLineagePersistence": "TX-S05",
}

# Registry symbols intentionally retained though not discovered by closed-world AST scan.
TRACE_MECHANISM_REGISTRY_COMPATIBILITY_ALIASES: Final[frozenset[str]] = frozenset()

_SENSITIVE_CLASS_SUFFIXES: Final[tuple[str, ...]] = (
    "Reconstructor",
    "CausalEvidence",
    "LineageReader",
    "LineageWriter",
    "LineagePersistence",
)

_FORBIDDEN_GLOBAL_TRACE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "GlobalTraceRecord",
        "TraceEnvelopeV2",
        "UnifiedEvidenceEvent",
        "UniversalCausalRecord",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CLOSED_WORLD_SKIP_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        "tests",
        "docker",
        "runtime-context",
    }
)


def repo_root() -> Path:
    return _REPO_ROOT


def _python_files_under(rel_root: str) -> list[Path]:
    root = _REPO_ROOT / rel_root
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in _CLOSED_WORLD_SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def _class_names_in_file(path: Path) -> frozenset[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
    return frozenset(names)


def discover_sensitive_classes() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for rel in CLOSED_WORLD_ROOTS:
        for path in _python_files_under(rel):
            rel_path = path.relative_to(_REPO_ROOT).as_posix()
            for name in _class_names_in_file(path):
                if name in TRACE_MECHANISM_CLASS_REGISTRY:
                    found.setdefault(name, []).append(rel_path)
                    continue
                for suffix in _SENSITIVE_CLASS_SUFFIXES:
                    if name.endswith(suffix):
                        found.setdefault(name, []).append(rel_path)
    return found


def assert_sensitive_classes_explicitly_classified(
    discovered: Mapping[str, Sequence[str]],
) -> None:
    unclassified = {
        name: list(paths)
        for name, paths in discovered.items()
        if name not in TRACE_MECHANISM_CLASS_REGISTRY
    }
    if unclassified:
        detail = ", ".join(sorted(unclassified))
        raise SensitiveClassificationError(
            f"unclassified sensitive trace mechanisms: {detail}"
        )


def assert_registry_references_valid_surfaces() -> None:
    surface_ids = {surface.surface_id for surface in TRACEABILITY_SURFACES}
    invalid = {
        symbol: surface_id
        for symbol, surface_id in TRACE_MECHANISM_CLASS_REGISTRY.items()
        if surface_id not in surface_ids
    }
    assert invalid == {}, f"registry references unknown TX surfaces: {invalid}"


def assert_registry_no_unexplained_orphans(
    discovered: Mapping[str, Sequence[str]],
) -> None:
    discovered_names = frozenset(discovered)
    orphans = {
        symbol
        for symbol in TRACE_MECHANISM_CLASS_REGISTRY
        if symbol not in discovered_names
        and symbol not in TRACE_MECHANISM_REGISTRY_COMPATIBILITY_ALIASES
    }
    assert orphans == frozenset(), f"unexplained registry orphans: {sorted(orphans)}"


def assert_frz_to_child_mapping_consistent() -> None:
    assert set(FRZ_TO_CHILD) == set(MANDATORY_FRZ_TRC_IDS)
    for criterion, child in FRZ_TO_CHILD.items():
        row = frz_row_by_id()[criterion]
        assert row.future_child_owner == child.value, (
            f"{criterion}: matrix={row.future_child_owner} canonical={child.value}"
        )
    child_frz: dict[TraceXChildId, set[str]] = {}
    for child in TRACE_X_CHILD_DECOMPOSITION:
        if child.child_id == TraceXChildId.CERT.value:
            continue
        child_frz[TraceXChildId(child.child_id)] = set(child.frz_criteria)
    for child_id in (
        TraceXChildId.P1,
        TraceXChildId.P2,
        TraceXChildId.P3,
        TraceXChildId.P4,
        TraceXChildId.P5,
        TraceXChildId.P6,
    ):
        assert child_frz.get(child_id), f"{child_id} has no FRZ criteria"
    cert = next(c for c in TRACE_X_CHILD_DECOMPOSITION if c.child_id == TraceXChildId.CERT.value)
    assert set(cert.frz_criteria) == set(MANDATORY_FRZ_TRC_IDS)


def assert_reverse_reconstruction_child_mapping_consistent() -> None:
    for row in REVERSE_RECONSTRUCTION_MATRIX:
        expected = REVERSE_RECONSTRUCTION_CHILD_BY_SUBJECT.get(row.subject)
        if expected is None:
            assert row.future_child_owner is None, row.subject
        else:
            assert row.future_child_owner == expected, row.subject


def _contract_path_from_semantic_contract(semantic_contract: str) -> str | None:
    if "@" not in semantic_contract:
        return None
    path = semantic_contract.split("@", 1)[1].strip()
    if path.endswith(".py"):
        return path
    return None


def classified_paths_union() -> frozenset[str]:
    paths: set[str] = set()
    for surface in TRACEABILITY_SURFACES:
        paths.update(surface.producer_paths)
        paths.update(surface.consumer_paths)
        contract_path = _contract_path_from_semantic_contract(surface.semantic_contract)
        if contract_path:
            paths.add(contract_path)
    for lock in ARCHITECTURE_LOCK:
        paths.update(lock.contract_paths)
    for row in FRZ_TRC_P0_MATRIX:
        paths.update(row.code_evidence)
        paths.update(p for p in row.contracts if p.endswith(".py"))
    return frozenset(paths)


def surfaces_by_id() -> dict[str, TraceabilitySurface]:
    return {surface.surface_id: surface for surface in TRACEABILITY_SURFACES}


def frz_row_by_id() -> dict[str, FrzTrcP0MatrixRow]:
    return {row.criterion: row for row in FRZ_TRC_P0_MATRIX}
