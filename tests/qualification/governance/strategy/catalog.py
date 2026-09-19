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
        Gr10CoverageStatus.QUALIFIED,
        "PRE_MODEL emits typed GovernanceDecisionEvidenceFact (source PolicyDecision only) through "
        "mandatory GovernanceEvidencePersistencePort on build_governed_inference_executor; "
        "unsupported PRE_MODEL actions fail closed without GR-8 fact; authority unchanged on "
        "persistence failure. Canonical MODEL C1 / decision-e2e explicitly wires in-memory port.",
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


@dataclass(frozen=True, slots=True)
class Gr10ResidualStrategyCapabilitySemantics:
    capability: str
    applicability: Gr10Applicability
    coverage: Gr10CoverageStatus | None
    reason: str


def _residual_matrix_status(
    semantics: tuple[Gr10ResidualStrategyCapabilitySemantics, ...],
    capability: str,
    strategy: str,
) -> Gr10CoverageStatus:
    for row in semantics:
        if row.capability == capability:
            if row.applicability is Gr10Applicability.NOT_APPLICABLE:
                return Gr10CoverageStatus.NOT_APPLICABLE
            assert row.coverage is not None
            return row.coverage
    raise KeyError(f"unknown GR-10 capability for {strategy} semantics: {capability!r}")


# SSOT for AGENTIC applicability vs coverage (GR-10-R7 requalification).
GR10_AGENTIC_CAPABILITY_SEMANTICS: tuple[Gr10ResidualStrategyCapabilitySemantics, ...] = (
    Gr10ResidualStrategyCapabilitySemantics(
        "Root admission",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "HostTaskExecution → DefaultRootExecutionLauncher → RuntimeExecutionPolicyAdmissionPort; "
        "GR-2 launcher + e2e authorization proofs (AGT-ROOT).",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Inner Governance",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "Primary HostTaskExecution→TaskBoundAgenticDelegate→UAEP path: GR-3 four-id inner guard proofs. "
        "Residual: not every legal agent delegate/tool phase wires CanonicalInnerExecutionGuardPort on "
        "all inner GEPs (TOOL_PLAN_OR_ACCESS, TOOL_INVOCATION_POLICY, PRE_OUTPUT, INTERRUPT) with "
        "enterprise adoption — UAEP coverage ≠ strategy-wide inner spine.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Policy evaluation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "GR-10-R3 closed: kernel policy_pre DENY → GovernanceResolution.DENY on UAEP path (not the "
        "historical R3 defect). Residual: optional POST_RUN when governance_service unset; PRE_OUTPUT "
        "and other per-GEP policy rows lack enterprise qualification on all production delegates.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "MSE",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "MeaningfulSideEffectAuthorizationBoundary qualified on governed contractor / MP-4R7 host; "
        "not all agent tool/host compositions route consequential effects through the canonical boundary.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Decision-bound effect",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "DecisionRequirementPolicy + DecisionGovernedSideEffectCoordinator on MP-4R7 / governed contractor "
        "production proofs.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "HITL",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GovernedContinuationRequest → ExecutionContinuationPort; fresh post-human governance reevaluation "
        "proven (MP-4R7 human-approve + governance-deny).",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Continuation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "ExecutionContinuationPort authority on MP-4R7 host; not Task registry / Nexus state.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Reliability",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "ProviderInvocation boundary on governed contractor GR-7 host (AGT-REL).",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Governance Evidence",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "GR-8 spine + root/MSE emission on qualified hosts; mandatory typed facts not enterprise-adopted "
        "for all applicable GEPs (TOOL_*, INTERRUPT, POST_RUN, PRE_OUTPUT, fresh post-human) on every "
        "production agent path.",
    ),
)


# SSOT for ORCHESTRATION applicability vs coverage (GR-10-R7 requalification).
GR10_ORCHESTRATION_CAPABILITY_SEMANTICS: tuple[Gr10ResidualStrategyCapabilitySemantics, ...] = (
    Gr10ResidualStrategyCapabilitySemantics(
        "Root admission",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "HostTaskExecution orchestration capability → same DefaultRootExecutionLauncher / GR-2 proofs "
        "(ORCH-ROOT).",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Inner Governance",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R8 / R8-R1: production host exposes CanonicalInnerExecutionGuardPort via RuntimeConfig and "
        "declarative composition; build_production_runtime_tool_invoker wires guard before tool authorization; "
        "inner request projects ActiveExecutionGovernanceIdentity.principal_id (GR-3 execution-binding spine, "
        "not full GEP matrix closure); TOOL_PLAN_OR_ACCESS remains ToolAccessPolicy.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Policy evaluation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "NEXUS_PLANNING / PlanningRunner PRE_MODEL with active governance identity; orchestration root "
        "policy admission same as host GR-2 path.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "MSE",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "GR-10-R9-ADR1: RuntimeToolInvoker pre-effect gate is a qualified slice; authority port "
        "must move to intergrax/contracts with typed MeaningfulSideEffectAuthorizationResult; "
        "production must fail closed (no synthetic membership/grant, no default ALLOW, no InMemory "
        "authority fallback). Graph/non-tool orchestration consequential seams open — GR-10-R9-R1.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Decision-bound effect",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "External Work host slices prove GR-6 wiring; not all orchestration consequential paths bind "
        "DecisionRequirementPolicy before effect.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "HITL",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "GR-5 orchestration slices (ORCH-HITL); HumanPauseCoordinator Task projection remains transitional — "
        "human judgment evidence ≠ Governance ALLOW; fresh governance + continuation port not enterprise-closed "
        "on all orchestration pause/resume paths.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Continuation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "ExecutionContinuationPort wired in GR-5-R4 proofs; residual Task-shaped projection in "
        "internal_continuation_orchestration — not continuation authority.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Reliability",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "GR-7 on External Work compositions; not all orchestration provider/external-effect paths adopt "
        "ProviderInvocation enterprise boundary.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Governance Evidence",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.PARTIAL,
        "Some orchestration GEP emission via Nexus/governance bridges; mandatory GR-8 typed facts not "
        "qualified per GEP on production orchestration paths (POST_RUN optional service; planning evidence "
        "partial).",
    ),
)


def gr10_matrix_agentic_status(capability: str) -> Gr10CoverageStatus:
    return _residual_matrix_status(GR10_AGENTIC_CAPABILITY_SEMANTICS, capability, "AGENTIC")


def gr10_matrix_orchestration_status(capability: str) -> Gr10CoverageStatus:
    return _residual_matrix_status(GR10_ORCHESTRATION_CAPABILITY_SEMANTICS, capability, "ORCHESTRATION")


@dataclass(frozen=True, slots=True)
class Gr10GepCoverageRow:
    gep: str
    agentic_applicable: bool
    agentic_coverage: Gr10CoverageStatus
    agentic_gap: str
    orchestration_applicable: bool
    orchestration_coverage: Gr10CoverageStatus
    orchestration_gap: str


GR10_GEP_COVERAGE_INVENTORY: tuple[Gr10GepCoverageRow, ...] = (
    Gr10GepCoverageRow(
        "ROOT_EXECUTION_ADMISSION",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "",
    ),
    Gr10GepCoverageRow(
        "PRE_MODEL",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "Agentic LLM router / ACP proofs; not every delegate.",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "",
    ),
    Gr10GepCoverageRow(
        "AGENT_DECISION",
        True,
        Gr10CoverageStatus.PARTIAL,
        "UAEP kernel path qualified (GR-10-R3); other agent delegates partial.",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R8: orchestration graph routing is internal; permission GEP owned by Policy evaluation — "
        "not duplicate CanonicalInnerExecutionGuardPort boundary.",
    ),
    Gr10GepCoverageRow(
        "INTERRUPT",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Not all agent interrupt paths emit enterprise inner guard + evidence.",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Orchestration interrupt bridges partial.",
    ),
    Gr10GepCoverageRow(
        "TOOL_PLAN_OR_ACCESS",
        True,
        Gr10CoverageStatus.PARTIAL,
        "UAEP tool plan paths primary; harness/kernel gaps remain.",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "ToolAccessPolicy + scope policy canonical access gate on production planner/tool exposure (GR-10-R8).",
    ),
    Gr10GepCoverageRow(
        "TOOL_INVOCATION_AUTHORIZATION",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Tool invoke on UAEP qualified slices; not universal.",
        True,
        Gr10CoverageStatus.QUALIFIED,
        "RuntimeToolInvoker requires CanonicalInnerExecutionGuardPort before physical invoke (GR-10-R8).",
    ),
    Gr10GepCoverageRow(
        "TOOL_INVOCATION_POLICY",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Policy engine on tool paths partial per host composition.",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Declarative enforcer coverage incomplete vs enterprise matrix.",
    ),
    Gr10GepCoverageRow(
        "MEANINGFUL_SIDE_EFFECT",
        True,
        Gr10CoverageStatus.PARTIAL,
        "MP-4R7 / contractor host qualified.",
        True,
        Gr10CoverageStatus.PARTIAL,
        "GR-10-R9-ADR1: tool invoke pre-effect gate wired; orchestration authority composition "
        "not enterprise-closed until GR-10-R9-R1 (ADR-GR-10-002).",
    ),
    Gr10GepCoverageRow(
        "PRE_OUTPUT",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Optional wiring on some agent outputs.",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Orchestration output gates partial.",
    ),
    Gr10GepCoverageRow(
        "POST_RUN",
        True,
        Gr10CoverageStatus.PARTIAL,
        "invoke_post_run_governance optional when service None.",
        True,
        Gr10CoverageStatus.PARTIAL,
        "Nexus finish_task POST_RUN wired; qualification harness drift; optional service.",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10R7NextRemediation:
    task_name: str
    strategy: str
    capability: str
    exact_blocker: str
    why_highest: str


GR10_R7_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R8 — ORCHESTRATION Inner Governance Production GEP Coverage Remediation",
    strategy="ORCHESTRATION",
    capability="Inner Governance",
    exact_blocker=(
        "Production Nexus orchestration (graph runners, RuntimeToolInvoker tool loop) does not "
        "enterprise-qualify CanonicalInnerExecutionGuardPort on every applicable inner GEP — "
        "declarative/MSE substitutes cover tool invoke only; AGENT_DECISION and TOOL_PLAN_OR_ACCESS "
        "lack uniform fail-closed inner spine proofs on all consequential orchestration paths."
    ),
    why_highest=(
        "P1 applicable production gap on external side-effect orchestration surface (above continuation/"
        "evidence polish); no P0 authority inversion or cross-layer contract change required for bounded fix."
    ),
)


GR10_R8_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R9 — ORCHESTRATION MSE production GEP coverage",
    strategy="ORCHESTRATION",
    capability="MSE",
    exact_blocker=(
        "require_meaningful_side_effect_authorization and MeaningfulSideEffectAuthorizationBoundary "
        "not enterprise-qualified on every consequential orchestration external-work / graph side-effect seam."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION capability row still PARTIAL after GR-10-R8 inner guard closure; "
        "distinct from Inner Governance spine (GR-3 identity binding vs consequential authorization)."
    ),
)


GR10_R9_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R9-R1 — Canonical MSE Contract Migration & Fail-Closed Production Composition",
    strategy="ORCHESTRATION",
    capability="MSE",
    exact_blocker=(
        "Nexus-local MeaningfulSideEffectAuthorizationPort returns object; production orchestration "
        "composition synthesizes membership/authority and default ALLOW via InMemory repositories "
        "(ADR-GR-10-002 rejected design)."
    ),
    why_highest=(
        "P0 authority architecture blocker before ORCHESTRATION MSE can re-qualify or GR-10-R10 "
        "decision-bound remediation proceeds on honest MSE foundation."
    ),
)


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
        "RuntimeToolInvoker (build_production_runtime_tool_invoker + inner guard) + MSE boundary where consequential",
        "GR-5 orchestration HITL + continuation port",
        "External Work compositions — same GR-7 boundary as agentic",
    ),
)


GR10_FINAL_CAPABILITY_MATRIX: tuple[Gr10CapabilityCell, ...] = (
    Gr10CapabilityCell(
        "Root admission",
        gr10_matrix_inference_status("Root admission"),
        gr10_matrix_agentic_status("Root admission"),
        gr10_matrix_orchestration_status("Root admission"),
        "INFERENCE: GR-10-R4 N/A. AGENTIC/ORCH: GR-10-R7 SSOT semantics.",
    ),
    Gr10CapabilityCell(
        "Inner Governance",
        gr10_matrix_inference_status("Inner Governance"),
        gr10_matrix_agentic_status("Inner Governance"),
        gr10_matrix_orchestration_status("Inner Governance"),
        "GR-10-R8: ORCHESTRATION Inner Governance QUALIFIED; AGENTIC residual per SSOT.",
    ),
    Gr10CapabilityCell(
        "Policy evaluation",
        gr10_matrix_inference_status("Policy evaluation"),
        gr10_matrix_agentic_status("Policy evaluation"),
        gr10_matrix_orchestration_status("Policy evaluation"),
        "GR-10-R3 closed on UAEP kernel DENY; AGENTIC residual per-GEP (not R3 defect).",
    ),
    Gr10CapabilityCell(
        "MSE",
        gr10_matrix_inference_status("MSE"),
        gr10_matrix_agentic_status("MSE"),
        gr10_matrix_orchestration_status("MSE"),
        "INFERENCE N/A; residual host coverage per GR-10-R7.",
    ),
    Gr10CapabilityCell(
        "Decision-bound effect",
        gr10_matrix_inference_status("Decision-bound effect"),
        gr10_matrix_agentic_status("Decision-bound effect"),
        gr10_matrix_orchestration_status("Decision-bound effect"),
        "MP-4R7 agentic qualified; orchestration External Work partial.",
    ),
    Gr10CapabilityCell(
        "HITL",
        gr10_matrix_inference_status("HITL"),
        gr10_matrix_agentic_status("HITL"),
        gr10_matrix_orchestration_status("HITL"),
        "Agentic MP-4R7 qualified; orchestration GR-5 slices partial.",
    ),
    Gr10CapabilityCell(
        "Continuation",
        gr10_matrix_inference_status("Continuation"),
        gr10_matrix_agentic_status("Continuation"),
        gr10_matrix_orchestration_status("Continuation"),
        "ExecutionContinuationPort; orch Task projection transitional.",
    ),
    Gr10CapabilityCell(
        "Reliability",
        gr10_matrix_inference_status("Reliability"),
        gr10_matrix_agentic_status("Reliability"),
        gr10_matrix_orchestration_status("Reliability"),
        "INFERENCE N/A; GR-7 host-qualified paths per strategy.",
    ),
    Gr10CapabilityCell(
        "Governance Evidence",
        gr10_matrix_inference_status("Governance Evidence"),
        gr10_matrix_agentic_status("Governance Evidence"),
        gr10_matrix_orchestration_status("Governance Evidence"),
        "GR-8 per-GEP adoption partial for AGENTIC/ORCH (INFERENCE PRE_MODEL qualified R6).",
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
                "test_gr10_r5_inference_executor_same_local_adapter_ast_gate",
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
        "PRE_MODEL governance evidence — mandatory composition, typed fact, verdict semantics",
        (
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_allow_invokes_provider_once"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_deny_blocks_provider"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_custom_persistence_port_records_typed_fact"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_deny_evidence_failure_still_denies_zero_provider"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_evidence_failure_still_allows_provider"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_require_human_emits_fact_before_fail_closed"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_escalate_no_fact_fail_closed"),
            _nid(_INFERENCE_EXEC, "test_inference_pre_model_modify_no_fact_fail_closed"),
            _nid(
                "tests/unit/runtime/execution/test_inference_composition.py",
                "test_build_governed_inference_executor_requires_persistence_port_signature",
            ),
            _nid(
                "tests/unit/runtime/execution/test_inference_composition.py",
                "test_governed_inference_executor_wires_custom_port",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
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
        "nested / graph execution under identity + canonical inner guard on tool invoke",
        (
            _nid(_ORCH_TOPO, "test_canonical_orchestration_topology_submission_proof"),
            _nid(
                "tests/unit/runtime/nexus/tools/test_gr10_r8_orchestration_inner_guard.py",
                "test_gr10_r8_custom_guard_deny_zero_physical_invocation",
            ),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r8_orchestration_inner_governance_qualification.py",
                "test_gr10_r8_orchestration_inner_governance_qualified",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
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
