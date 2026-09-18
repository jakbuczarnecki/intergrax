# © Artur Czarnecki. All rights reserved.

"""GR-10 strategy coverage catalog — production paths and pytest evidence pointers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Gr10CoverageStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    WIRED_NOT_QUALIFIED = "WIRED_NOT_QUALIFIED"
    PARTIAL = "PARTIAL"
    GAP = "GAP"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Gr10Applicability(StrEnum):
    """Whether a governance capability belongs to a strategy scope (not implementation)."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class Gr10InferenceCapabilitySemantics:
    capability: str
    applicability: Gr10Applicability
    coverage: Gr10CoverageStatus | None
    reason: str


@dataclass(frozen=True, slots=True)
class Gr10ProductionEntry:
    strategy: str
    root_entry: str
    inner_path: str
    mse_path: str
    hitl_path: str
    reliability_path: str


@dataclass(frozen=True, slots=True)
class Gr10CapabilityCell:
    capability: str
    inference: Gr10CoverageStatus
    agentic: Gr10CoverageStatus
    orchestration: Gr10CoverageStatus
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Gr10ScenarioEvidence:
    scenario_id: str
    strategy: str
    title: str
    pytest_node_ids: tuple[str, ...]
    expected_status: Gr10CoverageStatus


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_GR2_LAUNCHER = "tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py"
_Q_AUTH = "tests/qualification/governance/test_governance_e2e_authorization.py"
_GR3 = "tests/unit/runtime/governance/test_gr3_canonical_inner_enforcement.py"
_G3B = "tests/unit/runtime/governance/test_g3b_governance_coverage.py"
_GR8 = "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py"
_GR2_GATES = "tests/unit/runtime/architecture/test_gr2_r3_model_c1_architecture_gates.py"
_INFERENCE_EXEC = "tests/unit/runtime/execution/test_inference_executor.py"
_FACADE = "tests/unit/runtime/execution/test_execution_facade.py"
_MP4R7 = "tests/unit/mp4r7/test_enterprise_integration_qualification.py"
_GR5_ORCH = "tests/unit/runtime/nexus/orchestration/test_gr5_r4_internal_hitl_orchestration.py"
_ORCH_TOPO = "tests/unit/runtime/execution/test_orchestration_topology_e2e_proof.py"
_GR7_A3 = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a3_durable_provider_invocation.py"
)
_GR10_GATES = "tests/qualification/governance/strategy/test_gr10_gates.py"

# SSOT for INFERENCE applicability vs coverage (GR-10-R1). Matrix/inventory must not contradict.
GR10_INFERENCE_CAPABILITY_SEMANTICS: tuple[Gr10InferenceCapabilitySemantics, ...] = (
    Gr10InferenceCapabilitySemantics(
        "Root admission",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "GR-10-R4: no independent production root for INFERENCE — HostTaskExecution resolves "
        "AGENT/ORCHESTRATION only; InferenceExecutor is an internal StrategyExecutionRouter "
        "delegate wired in composition stacks (Execution facade is not a legal production root; "
        "MODEL C1 gates). ROOT_INFERENCE operation exists for strategy-neutral launcher/policy "
        "mapping only; Tier-3 hosts never dispatch INFERENCE strategy at root.",
    ),
    Gr10InferenceCapabilitySemantics(
        "Inner Governance",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "GR-10-R5: matrix row is GR-3 CanonicalInnerExecutionGuardPort + meaningful-side-effect "
        "spine — distinct from Policy evaluation (inner GEPs). INFERENCE has no applicable "
        "protected inner action on that spine (MSE/tool/agent-decision N/A); sole permission "
        "boundary for model invocation is PRE_MODEL (Policy evaluation QUALIFIED). "
        "require_active_execution_identity on InferenceExecutor is execution lifecycle context, "
        "not a second governance authority; UAEP parity is not required.",
    ),
    Gr10InferenceCapabilitySemantics(
        "Policy evaluation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "InferenceExecutor enforces PRE_MODEL (evaluate_pre_llm) before "
        "LLMAdapter.generate_structured; qualification proofs on executor seam.",
    ),
    Gr10InferenceCapabilitySemantics(
        "MSE",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "Structured inference seam is not a meaningful external side effect (GOVERNED_EXECUTION §9).",
    ),
    Gr10InferenceCapabilitySemantics(
        "Decision-bound effect",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "No DecisionRequirementPolicy / consequential MSE on canonical inference-only path.",
    ),
    Gr10InferenceCapabilitySemantics(
        "HITL",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "No INFERENCE evaluation point emits REQUIRE_HUMAN (AGENT_DECISION/INTERRUPT/MSE N/A; "
        "PRE_MODEL engine returns ALLOW/DENY only). Root admission REQUIRE_HUMAN is root "
        "admission, not GR-5 continuation.",
    ),
    Gr10InferenceCapabilitySemantics(
        "Continuation",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "ExecutionContinuationPort applies when governance pauses mid-strategy; INFERENCE has "
        "no applicable HITL trigger on the strategy path.",
    ),
    Gr10InferenceCapabilitySemantics(
        "Reliability",
        Gr10Applicability.NOT_APPLICABLE,
        None,
        "GR-7 ProviderInvocation boundary applies to governed external effects after MSE "
        "authorization, not structured LLM read/inference adapter calls.",
    ),
    Gr10InferenceCapabilitySemantics(
        "Governance Evidence",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "ROOT_EXECUTION_ADMISSION wired on launcher path; PRE_MODEL evidence emitted from "
        "InferenceExecutor on structured inference path (full enterprise matrix still open).",
    ),
)


def gr10_matrix_inference_status(capability: str) -> Gr10CoverageStatus:
    """Map semantics to matrix cell (NOT_APPLICABLE when capability is out of scope)."""
    for row in GR10_INFERENCE_CAPABILITY_SEMANTICS:
        if row.capability == capability:
            if row.applicability is Gr10Applicability.NOT_APPLICABLE:
                return Gr10CoverageStatus.NOT_APPLICABLE
            assert row.coverage is not None
            return row.coverage
    raise KeyError(f"unknown GR-10 capability for INFERENCE semantics: {capability!r}")


GR10_PRODUCTION_INVENTORY: tuple[Gr10ProductionEntry, ...] = (
    Gr10ProductionEntry(
        "INFERENCE",
        "No Tier-3/public production root — internal only: composition-wired "
        "ExecutionRuntime → StrategyExecutionRouter → InferenceExecutor (MODEL C1 non-root)",
        "NOT_APPLICABLE — no GR-3 inner-guard/MSE spine on canonical structured inference path; "
        "PRE_MODEL permission is Policy evaluation row (QUALIFIED)",
        "NOT_APPLICABLE — structured inference is not classified MSE spine",
        "NOT_APPLICABLE — no REQUIRE_HUMAN on inference-only ExecutionRequest",
        "NOT_APPLICABLE — no provider mutation on InferenceExecutor structured path",
    ),
    Gr10ProductionEntry(
        "AGENTIC",
        "HostTaskExecution.execute → DefaultRootExecutionLauncher → ExecutionRuntime",
        "TaskBoundAgenticDelegate → AgentEnginePort (UAEP / inner guard on tool paths)",
        "MeaningfulSideEffectAuthorizationBoundary on external-work hosts",
        "GovernedContinuationRequest → ExecutionContinuationPort (MP-4R7 host)",
        "ProviderInvocation after Governance ALLOW (governed contractor GR-7)",
    ),
    Gr10ProductionEntry(
        "ORCHESTRATION",
        "HostTaskExecution.execute (orchestration capability) → same root launcher",
        "OrchestrationExecutor / Nexus graph runners under active identity",
        "RuntimeToolInvoker + MSE boundary where consequential",
        "GR-5 orchestration HITL + continuation port",
        "External Work compositions — same GR-7 boundary as agentic",
    ),
)


GR10_FINAL_CAPABILITY_MATRIX: tuple[Gr10CapabilityCell, ...] = (
    Gr10CapabilityCell(
        "Root admission",
        gr10_matrix_inference_status("Root admission"),
        Gr10CoverageStatus.QUALIFIED,
        Gr10CoverageStatus.QUALIFIED,
        "INFERENCE: root admission N/A (GR-10-R4); Tier-3 host resolves AGENT/ORCH only.",
    ),
    Gr10CapabilityCell(
        "Inner Governance",
        gr10_matrix_inference_status("Inner Governance"),
        Gr10CoverageStatus.PARTIAL,
        Gr10CoverageStatus.PARTIAL,
        "INFERENCE: GR-10-R5 N/A (no GR-3/MSE inner spine). AGENTIC/ORCH: residual inner GEP + "
        "MSE/tool paths PARTIAL (not UAEP parity for inference).",
    ),
    Gr10CapabilityCell(
        "Policy evaluation",
        gr10_matrix_inference_status("Policy evaluation"),
        Gr10CoverageStatus.PARTIAL,
        Gr10CoverageStatus.QUALIFIED,
        "INFERENCE: InferenceExecutor PRE_MODEL qualified. AGENTIC: GR-3/ACP PRE_MODEL and root "
        "paths qualified; kernel policy_pre DENY → final GovernanceResolution DENY (GR-10-R3 closed). "
        "Residual AGENTIC PARTIAL: not all strategy GEP/production policy paths enterprise-qualified "
        "(e.g. optional POST_RUN wiring; GR-8 per-GEP adoption — not the historical R3 defect).",
    ),
    Gr10CapabilityCell(
        "MSE",
        gr10_matrix_inference_status("MSE"),
        Gr10CoverageStatus.PARTIAL,
        Gr10CoverageStatus.PARTIAL,
        "INFERENCE: no meaningful external side effect on canonical inference seam.",
    ),
    Gr10CapabilityCell(
        "Decision-bound effect",
        gr10_matrix_inference_status("Decision-bound effect"),
        Gr10CoverageStatus.QUALIFIED,
        Gr10CoverageStatus.PARTIAL,
        "MP-4R7 agentic; orchestration via External Work host slices.",
    ),
    Gr10CapabilityCell(
        "HITL",
        gr10_matrix_inference_status("HITL"),
        Gr10CoverageStatus.QUALIFIED,
        Gr10CoverageStatus.PARTIAL,
        "INFERENCE: no strategy-path REQUIRE_HUMAN (N/A). Agentic/orch GR-5 proofs.",
    ),
    Gr10CapabilityCell(
        "Continuation",
        gr10_matrix_inference_status("Continuation"),
        Gr10CoverageStatus.QUALIFIED,
        Gr10CoverageStatus.PARTIAL,
        "GR-5 port; orchestration transitional Task projection in places.",
    ),
    Gr10CapabilityCell(
        "Reliability",
        gr10_matrix_inference_status("Reliability"),
        Gr10CoverageStatus.QUALIFIED,
        Gr10CoverageStatus.PARTIAL,
        "INFERENCE: outside GR-7 external-effect boundary. GR-7 on governed contractor host.",
    ),
    Gr10CapabilityCell(
        "Governance Evidence",
        gr10_matrix_inference_status("Governance Evidence"),
        Gr10CoverageStatus.PARTIAL,
        Gr10CoverageStatus.PARTIAL,
        "GR-8 spine on root admission + MSE; not all GEPs per strategy.",
    ),
)


GR10_SCENARIO_CATALOG: tuple[Gr10ScenarioEvidence, ...] = (
    Gr10ScenarioEvidence(
        "INF-A",
        "INFERENCE",
        "internal delegate executes under active identity (not a production root)",
        (
            _nid(_FACADE, "test_facade_mints_platform_execution_id_not_supplied_by_caller"),
            _nid(_INFERENCE_EXEC, "test_direct_structured_request_executes_full_path"),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r4_inference_root_admission_qualification.py",
                "test_gr10_r4_execution_facade_is_not_legal_production_root",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-B",
        "INFERENCE",
        "launcher ROOT_INFERENCE op DENY → zero intake (platform contract; not INFERENCE strategy root)",
        (
            _nid(_GR2_LAUNCHER, "test_launcher_deny_skips_intake[root.execution.inference]"),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r4_inference_root_admission_qualification.py",
                "test_gr10_r4_root_inference_launcher_deny_zero_intake",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-C",
        "INFERENCE",
        "policy failure → zero inference",
        (
            _nid(_Q_AUTH, "test_scenario_c_root_policy_plugin_fail_closed_via_composition"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-PREMODEL-ALLOW",
        "INFERENCE",
        "PRE_MODEL ALLOW on InferenceExecutor invokes provider once",
        (
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_allow_invokes_provider_once"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-PREMODEL-DENY",
        "INFERENCE",
        "PRE_MODEL DENY on InferenceExecutor blocks provider",
        (
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_deny_blocks_provider"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-R5-INNER",
        "INFERENCE",
        "Inner Governance (GR-3/MSE spine) not applicable — PRE_MODEL is Policy evaluation row",
        (
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r5_inference_inner_governance_qualification.py",
                "test_gr10_r5_inference_inner_governance_semantics_not_applicable",
            ),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r5_inference_inner_governance_qualification.py",
                "test_gr10_r5_inference_executor_provider_only_after_pre_model_ast",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-D",
        "INFERENCE",
        "PRE_MODEL reference proof on agentic LLM router (semantics reference)",
        (
            _nid(_G3B, "test_pre_model_policy_blocks_provider_before_complete"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "INF-E",
        "INFERENCE",
        "Governance evidence on wired GEP",
        (
            _nid(_GR8, "test_root_allow_emits_exactly_one_governance_fact"),
        ),
        Gr10CoverageStatus.WIRED_NOT_QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "AGT-ROOT",
        "AGENTIC",
        "root admission ALLOW/DENY",
        (
            _nid(_GR2_LAUNCHER, "test_launcher_allow_runs_admission_and_intake_once[root.execution.agent]"),
            _nid(_GR2_LAUNCHER, "test_launcher_deny_skips_intake[root.execution.agent]"),
            _nid(_Q_AUTH, "test_scenario_a_root_admission_allow_single_intake"),
            _nid(_Q_AUTH, "test_scenario_b_root_admission_deny_zero_intake"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "AGT-INNER",
        "AGENTIC",
        "inner ALLOW/DENY fail-closed",
        (_nid(_GR3, "test_exact_four_id_match_allow_executes_once"),),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "AGT-MSE-HITL",
        "AGENTIC",
        "MSE + Decision + HITL + fresh governance",
        (
            _nid(_MP4R7, "test_mp4r7_success_e2e"),
            _nid(_MP4R7, "test_mp4r7_human_approve_governance_deny_prevents_continuation_and_operation"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "AGT-REL",
        "AGENTIC",
        "Reliability handoff",
        (_nid(_GR7_A3, "test_success_persists_intent_before_outcome_and_ger"),),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "ORCH-ROOT",
        "ORCHESTRATION",
        "root admission",
        (
            _nid(
                _GR2_LAUNCHER,
                "test_launcher_allow_runs_admission_and_intake_once[root.execution.orchestration]",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "ORCH-INNER",
        "ORCHESTRATION",
        "nested / graph execution under identity",
        (_nid(_ORCH_TOPO, "test_canonical_orchestration_topology_submission_proof"),),
        Gr10CoverageStatus.PARTIAL,
    ),
    Gr10ScenarioEvidence(
        "ORCH-HITL",
        "ORCHESTRATION",
        "HITL / continuation",
        (_nid(_GR5_ORCH, "test_resume_authorized_blocks_planning"),),
        Gr10CoverageStatus.PARTIAL,
    ),
    Gr10ScenarioEvidence(
        "ARCH-C1",
        "ALL",
        "MODEL C1 root entry gates",
        (
            _nid(_GR2_GATES, "test_production_has_no_unauthorized_root_construction"),
            _nid(_GR10_GATES, "test_gr10_host_task_wires_mandatory_root_launcher"),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
)


def gr10_catalog_pytest_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in GR10_SCENARIO_CATALOG:
        for node_id in entry.pytest_node_ids:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return tuple(ordered)
