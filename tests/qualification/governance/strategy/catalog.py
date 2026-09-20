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
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R9-R4: production topology composition requires OrchestrationTopologySlotMsePolicy; "
        "lab builder explicit for policy-less harness; authority delegation via "
        "orchestration_slot_effect_authority_owner contract surface; fan-out PhysicalDelegation "
        "unchanged; fail-closed production.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Decision-bound effect",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R10: production orchestration MSE composition binds explicit DecisionRequirementPolicy "
        "before physical effect; canonical boundary enforces REQUIRED/UNDETERMINED material; External "
        "Work host retains decision-governed coordinator; Physical Delegation owns separate boundary.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "HITL",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R11-R4: post-HITL MATCHED_HITL_PROPOSAL requires exact symmetric "
        "GovernedContinuationCorrelation (execution + operation + resource + side-effect "
        "scope/digest); missing correlation fields never wildcard-match concrete proposal "
        "values (CORRELATION_INSUFFICIENT). GR-10-R11-R3: human_request_id alone never "
        "marks the current effect as post-HITL; same-execution unrelated human continuation "
        "is UNRELATED_HUMAN_CONTINUATION / CORRELATION_INSUFFICIENT → ordinary ALLOW. "
        "GR-10-R11-R2: ordinary ALLOW without grant; post-HITL ALLOW requires RESUMED + "
        "matching GovernedContinuationApprovalGrant evidence (never permission); fresh "
        "REQUIRE_HUMAN never PROCEED; continue_slot / RuntimeToolInvoker reauthorize; "
        "External Work authorize_and_execute; Physical Delegation HITL delegated.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Continuation",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R12-R1: ExecutionContinuationPort is sole pause/wait/resolution/resume authority; "
        "current-episode SSOT; Task/HumanPauseCoordinator/continuable_slots/checkpoint are "
        "projection or eligibility only; production requires store.is_durable=True "
        "(explicit non-durable rejected); strict factory never invents in-memory; "
        "HostTaskExecution shares NexusLoop continuation store; durable restart via "
        "export/reconstruct contract (GR-5-R5); named vendor adapter N/A.",
    ),
    Gr10ResidualStrategyCapabilitySemantics(
        "Reliability",
        Gr10Applicability.APPLICABLE,
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R13-R1: post-admission OrchestrationConsequentialEffectReliabilityPort fail-safe "
        "ProviderInvocationStatus (typed definitive failure vs post-dispatch UNKNOWN); canonical "
        "tenant/slot invocation identity; intent-without-outcome blocks blind replay; GR-7 "
        "reconciliation/repeat eligibility delegated; RuntimeToolInvoker + External Work unchanged owners.",
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
        Gr10CoverageStatus.QUALIFIED,
        "GR-10-R9-R2: tool + graph/non-tool orchestration consequential seams closed via canonical "
        "MeaningfulSideEffectAuthorizationPort (ADR-GR-10-002).",
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


GR10_R9_R1_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R9-R2 — ORCHESTRATION graph/non-tool consequential MSE seam closure",
    strategy="ORCHESTRATION",
    capability="MSE",
    exact_blocker=(
        "Production orchestration graph runners and non-tool consequential mutations lack canonical "
        "MSE boundary coverage beyond RuntimeToolInvoker side_effects=True tool slice."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION MSE PARTIAL row after GR-10-R9-R1 authority migration; "
        "must close or honestly delegate before ORCHESTRATION MSE re-qualification."
    ),
)


GR10_R9_R3_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R10 — ORCHESTRATION decision-bound consequential effect coverage",
    strategy="ORCHESTRATION",
    capability="Decision-bound effect",
    exact_blocker=(
        "Not all production orchestration consequential paths bind DecisionRequirementPolicy before "
        "effect despite canonical topology MSE enforcement (GR-10-R9-R3)."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R9-R3 topology slot MSE "
        "enforcement closure."
    ),
)


GR10_R9_R2_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R10 — ORCHESTRATION decision-bound consequential effect coverage",
    strategy="ORCHESTRATION",
    capability="Decision-bound effect",
    exact_blocker=(
        "Not all production orchestration consequential paths bind DecisionRequirementPolicy before "
        "effect despite MSE spine qualification."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R9-R2 MSE qualification; "
        "Decision-bound remediation requires honest MSE foundation."
    ),
)


GR10_R10_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R11 — ORCHESTRATION HITL enterprise closure",
    strategy="ORCHESTRATION",
    capability="HITL",
    exact_blocker=(
        "GR-5 orchestration slices partial — human judgment evidence ≠ Governance ALLOW; fresh governance "
        "+ continuation port not enterprise-closed on all orchestration pause/resume paths."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R10 decision-bound "
        "effect qualification."
    ),
)


GR10_R11_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R12 — ORCHESTRATION Continuation enterprise closure",
    strategy="ORCHESTRATION",
    capability="Continuation",
    exact_blocker=(
        "ExecutionContinuationPort wired in GR-5-R4 proofs; residual Task-shaped projection in "
        "internal_continuation_orchestration — not full ORCHESTRATION Continuation enterprise qualification."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R11-R4 HITL "
        "exact proposal correlation (fail-closed optional fields); distinct from HITL "
        "human-judgment vs permission boundary."
    ),
)


GR10_R12_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R13 — ORCHESTRATION Reliability enterprise closure",
    strategy="ORCHESTRATION",
    capability="Reliability",
    exact_blocker=(
        "GR-7 on External Work compositions; not all orchestration provider/external-effect "
        "paths adopt ProviderInvocation enterprise boundary — post-admission uncertainty remains."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R12 Continuation "
        "enterprise qualification; distinct from continuation pause/resume authority."
    ),
)


GR10_R13_NEXT_REMEDIATION: Gr10R7NextRemediation = Gr10R7NextRemediation(
    task_name="GR-10-R14 — ORCHESTRATION Governance Evidence & ORCHESTRATION Closure",
    strategy="ORCHESTRATION",
    capability="Governance Evidence",
    exact_blocker=(
        "Mandatory GR-8 typed governance evidence facts not enterprise-qualified on all production "
        "orchestration paths."
    ),
    why_highest=(
        "Highest remaining ORCHESTRATION applicable capability row after GR-10-R13 Reliability "
        "enterprise qualification."
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationReliabilityInventoryRow:
    path: str
    production: bool
    consequential: bool
    governance_admitted: bool
    reliability_boundary: str
    idempotency: str
    reconciliation: str
    outcome_classification: str
    coverage: str


GR10_ORCHESTRATION_RELIABILITY_INVENTORY: tuple[
    Gr10OrchestrationReliabilityInventoryRow,
    ...,
] = (
    Gr10OrchestrationReliabilityInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=True)",
        True,
        True,
        True,
        "IdempotencyPreEffectCoordinator + ToolExternalOperationAttempt + dependency attempt boundary",
        "idempotency key + ledger",
        "external operation termination port when configured",
        "ToolEffectCertainty + ProviderInvocationStatus N/A (tool contract)",
        "QUALIFIED",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=False)",
        True,
        False,
        False,
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "External Work / governed contractor host",
        True,
        True,
        True,
        "ProviderInvocationDispatchGate + GR-7 recovery/reconciliation",
        "provider idempotency_key",
        "provider_invocation reconciliation contract",
        "ProviderInvocationStatus",
        "QUALIFIED — delegated GR-7",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "Canonical production topology composition (host → durable store → Reliability)",
        True,
        True,
        True,
        "build_orchestration_reliability_composition + build_strict_production_orchestration_topology_slot_mse_policy",
        "operation_id idempotency_key",
        "GR-7 reconciliation via shared ProviderInvocationStore",
        "ProviderInvocationStatus (adapter typed failure vs post-dispatch UNKNOWN)",
        "QUALIFIED — intergrax/runtime/execution/orchestration_topology_production_composition.py; harness + governed contractor host factories",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "GovernedOrchestrationSlotExecutor.execute_slot",
        True,
        True,
        True,
        "OrchestrationConsequentialEffectReliabilityPort (production composition)",
        "operation_id idempotency_key",
        "GR-7 reconciliation where host wires recovery",
        "ProviderInvocationStatus (adapter typed failure vs post-dispatch UNKNOWN)",
        "QUALIFIED",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "GovernedOrchestrationSlotContinuationExecutor.continue_slot",
        True,
        True,
        True,
        "OrchestrationConsequentialEffectReliabilityPort (production composition)",
        "operation_id idempotency_key",
        "GR-7 reconciliation where host wires recovery",
        "ProviderInvocationStatus (adapter typed failure vs post-dispatch UNKNOWN)",
        "QUALIFIED",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "Physical delegation / fan-out coordination slot",
        True,
        True,
        True,
        "DelegatedExecutionProvider + delegated invocation correlation",
        "delegated idempotency_key",
        "delegated execution reconciliation",
        "DelegatedExecutionOutcome",
        "delegated to another canonical owner",
    ),
    Gr10OrchestrationReliabilityInventoryRow(
        "Orchestration event bus publish",
        True,
        False,
        False,
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationStrictHostDecisionPolicyInventoryRow:
    host: str
    strict_capable: bool
    production: bool
    orchestration_mse_applicable: bool
    policy_source: str
    explicit_policy: bool
    coverage: str


GR10_ORCHESTRATION_STRICT_HOST_DECISION_POLICY_INVENTORY: tuple[
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow,
    ...,
] = (
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "governed_contractor_application",
        True,
        True,
        True,
        "host/orchestration_decision_requirement_policy.default_governed_contractor_harness_orchestration_decision_requirement_policy",
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "research_application",
        True,
        True,
        True,
        "host/orchestration_decision_requirement_policy.resolve_research_harness_orchestration_decision_requirement_policy",
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "legal_application",
        True,
        True,
        True,
        "host/orchestration_decision_requirement_policy.resolve_legal_harness_orchestration_decision_requirement_policy",
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "dispute_sim_application",
        True,
        True,
        True,
        "host/orchestration_decision_requirement_policy.resolve_dispute_sim_harness_orchestration_decision_requirement_policy",
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "local_workspace_application",
        True,
        True,
        True,
        "host/host_runtime_composition + orchestration_decision_requirement_policy.resolve_local_workspace_harness_orchestration_decision_requirement_policy",
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "lab_application",
        False,
        False,
        False,
        "N/A — lab_defaults execution_mode balanced",
        False,
        "N/A",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "attestation_demo",
        False,
        False,
        False,
        "N/A — partner PoC lab profile",
        False,
        "N/A",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "poc_template_application",
        False,
        False,
        False,
        "N/A — lab scaffold template",
        False,
        "N/A",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "intergrax_assistant_application",
        False,
        False,
        False,
        "N/A — harness chat lab",
        False,
        "N/A",
    ),
    Gr10OrchestrationStrictHostDecisionPolicyInventoryRow(
        "HarnessApplication (intergrax.harness.app)",
        True,
        False,
        True,
        "caller-supplied orchestration_decision_requirement_policy on build_runtime when execution_mode strict",
        True,
        "N/A",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationDecisionBoundInventoryRow:
    path: str
    production: bool
    consequential: bool
    decision_requirement_applicable: bool
    policy: str
    decision_binding: str
    effect_boundary: str
    coverage: str


GR10_ORCHESTRATION_DECISION_BOUND_EFFECT_INVENTORY: tuple[
    Gr10OrchestrationDecisionBoundInventoryRow,
    ...
] = (
    Gr10OrchestrationDecisionBoundInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=True)",
        True,
        True,
        True,
        "explicit host/domain DecisionRequirementPolicy → production orchestration MSE port",
        "MeaningfulSideEffectAuthorizationBoundary._enforce_decision_requirement",
        "MeaningfulSideEffectAuthorizationPort.authorize before ToolExecutor",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=False)",
        True,
        False,
        False,
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "External Work / governed contractor host",
        True,
        True,
        True,
        "host default_external_work_decision_requirement_policy (injectable)",
        "authorize_and_execute_decision_bound_side_effect + boundary",
        "MeaningfulSideEffectAuthorizationBoundary",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.submit",
        True,
        True,
        True,
        "explicit host/domain DecisionRequirementPolicy → production MSE port",
        "boundary authorize before slot execute",
        "GovernedOrchestrationSlotExecutor",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.recover_failed_slot",
        True,
        True,
        True,
        "explicit host/domain DecisionRequirementPolicy → production MSE port (recovery)",
        "boundary authorize before recovery physical effect",
        "GovernedOrchestrationSlotExecutor",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.continue_slot",
        True,
        True,
        True,
        "explicit host/domain DecisionRequirementPolicy → production MSE port (continuation)",
        "boundary authorize before continuation physical effect",
        "GovernedOrchestrationSlotContinuationExecutor",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "Decision-governed side effect helper (GR-6)",
        True,
        True,
        True,
        "caller-supplied DecisionRequirementPolicy",
        "decision material attach + boundary",
        "authorize_and_execute_decision_bound_side_effect",
        "QUALIFIED",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "Physical delegation / fan-out coordination slot",
        True,
        True,
        False,
        "N/A — PhysicalDelegationGovernancePort",
        "delegated to Physical Delegation canonical boundary",
        "PhysicalDelegationGovernanceBoundary",
        "delegated to another canonical owner",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "OrchestrationExecutor graph routing / checkpoints",
        True,
        False,
        False,
        "N/A",
        "N/A",
        "internal orchestration control",
        "N/A",
    ),
    Gr10OrchestrationDecisionBoundInventoryRow(
        "build_lab_orchestration_topology_submission_port",
        False,
        False,
        False,
        "N/A",
        "N/A",
        "lab qualification only",
        "N/A",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationHitlInventoryRow:
    path: str
    production: bool
    hitl_applicable: bool
    human_evidence_owner: str
    continuation_authority: str
    reauthorization_boundary: str
    coverage: str


GR10_ORCHESTRATION_HITL_INVENTORY: tuple[Gr10OrchestrationHitlInventoryRow, ...] = (
    Gr10OrchestrationHitlInventoryRow(
        "MeaningfulSideEffectAuthorizationBoundary.authorize_and_execute",
        True,
        True,
        "Human Review / GovernedContinuationGrantCoordinator (evidence + grant only)",
        "ExecutionContinuationPort via apply_governed_continuation_pause",
        "fresh authorize_and_execute (GR-5: grant match on REQUIRE_HUMAN or ALLOW) — External Work path",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=True)",
        True,
        True,
        "Human Review evidence; grant on task.runtime.governance (correlation only)",
        "ExecutionContinuationPort (canonical RESUMED + human-governed correlation post-HITL)",
        "evaluate_mse_hitl_effect_gate: ordinary ALLOW no grant; post-HITL ALLOW + RESUMED + scoped grant",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=False)",
        True,
        False,
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "External Work / governed contractor host",
        True,
        True,
        "Human Review via GovernedContinuationRequest surface",
        "ExecutionContinuationPort + grant coordinator",
        "authorize_and_execute / decision-bound coordinator",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "GovernedOrchestrationSlotExecutor.execute_slot",
        True,
        True,
        "Human Review (continuation request on REQUIRE_HUMAN/ESCALATE)",
        "host register_governed_continuation_slots + ExecutionContinuationPort on pause paths",
        "evaluate_mse_hitl_effect_gate: ordinary ALLOW without grant; post-HITL requires grant+RESUMED",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "GovernedOrchestrationSlotContinuationExecutor.continue_slot",
        True,
        True,
        "Human Review grant evidence (correlation only; never permission)",
        "ExecutionContinuationPort RESUMED (continuable_slots is eligibility only)",
        "fresh authorize + evaluate_mse_hitl_effect_gate (ALLOW + RESUMED + scoped grant; missing grant blocks)",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.recover_failed_slot",
        True,
        True,
        "N/A — recovery refuses continuable HITL slots (require continue_slot)",
        "continable_slots gate blocks recovery reuse of HITL path",
        "fresh MSE on non-HITL recovery only",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "governed_continuation_bridge / grant coordinator",
        True,
        True,
        "Human Review evidence owner",
        "ExecutionContinuationPort (canonical)",
        "compose from Governance result; grant ≠ ALLOW",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "HumanPauseCoordinator / task pause projection",
        True,
        True,
        "projection only — not permission",
        "must not be used as continuation authority (GR-5-R4)",
        "N/A — projection",
        "delegated to canonical owner",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "Physical delegation / fan-out coordination slot",
        True,
        True,
        "Physical Delegation human evidence / continuation grant",
        "PhysicalDelegationGovernedContinuation + grant coordinator",
        "PhysicalDelegationGovernancePort (MSE delegated)",
        "delegated to another canonical owner",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "Declarative policy REQUIRE_HITL tool path",
        True,
        True,
        "DeclarativeHitlPendingApproval / DeclarativeHitlApprovalGrant",
        "ExecutionContinuationPort via declarative HITL bridge",
        "declarative grant scope match then fresh tool MSE authorize; not MSE post-HITL "
        "classification via human_request_id alone (GR-10-R11-R3)",
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlInventoryRow(
        "mse_hitl_effect_gate.classify_human_governed_proposal_relation",
        True,
        True,
        "N/A — classification only (Human Review evidence elsewhere)",
        "GovernedContinuationCorrelation exact symmetric proposal match only",
        "CORRELATION_INSUFFICIENT / UNRELATED_HUMAN_CONTINUATION → ordinary ALLOW; "
        "missing scope/resource/digest never wildcard; human_request_id alone never "
        "POST_HITL_* (GR-10-R11-R3/R4)",
        "QUALIFIED",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationHitlHumanContinuationProducerRow:
    path: str
    production: bool
    human_continuation: bool
    governed_correlation_present: bool
    proposal_scope_recoverable: bool
    status: str


GR10_ORCHESTRATION_HITL_HUMAN_CONTINUATION_PRODUCER_INVENTORY: tuple[
    Gr10OrchestrationHitlHumanContinuationProducerRow,
    ...,
] = (
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "governed_continuation_bridge.apply_governed_continuation_pause",
        True,
        True,
        True,
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "establish_canonical_hitl_pause (graph / internal HITL)",
        True,
        True,
        True,
        False,
        "QUALIFIED — generic internal_hitl_* correlation when caller omits governed_correlation; "
        "not MSE proposal-scoped unless real GovernedContinuationCorrelation supplied",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "graph_runner HITL pause → establish_canonical_hitl_pause",
        True,
        True,
        True,
        False,
        "QUALIFIED — proposal-scoped only when human_request.governed_continuation present; "
        "otherwise generic internal HITL (not MSE post-HITL authority)",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "declarative_policy_hitl_bridge (DeclarativeHitlPendingApproval)",
        True,
        True,
        False,
        False,
        "N/A — non-MSE declarative HITL; not mse_hitl_effect_gate post-HITL authority",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "HumanPauseCoordinator (projection only)",
        True,
        False,
        False,
        False,
        "N/A — projection; does not request_pause",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "Physical Delegation continuation grant",
        True,
        True,
        True,
        True,
        "QUALIFIED — delegated owner; no duplicate MSE gate",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "ExecutionContinuationPort.request_pause direct (tests/scaffolds)",
        False,
        True,
        False,
        False,
        "LEGACY_GAP bounded — CORRELATION_INSUFFICIENT; not post-HITL for current effect",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "External Work / MSE authorize_and_execute HITL",
        True,
        True,
        True,
        True,
        "QUALIFIED",
    ),
    Gr10OrchestrationHitlHumanContinuationProducerRow(
        "RuntimeToolInvoker MSE HITL (via governed continuation bridge)",
        True,
        True,
        True,
        True,
        "QUALIFIED",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationContinuationInventoryRow:
    path: str
    production: bool
    creates_pause: bool
    resolves: bool
    resumes: bool
    uses_canonical_port: bool
    projection_only: bool
    current_episode_safe: bool
    restart_safe: bool
    coverage: str


GR10_ORCHESTRATION_CONTINUATION_INVENTORY: tuple[
    Gr10OrchestrationContinuationInventoryRow,
    ...,
] = (
    Gr10OrchestrationContinuationInventoryRow(
        "ExecutionContinuationPort / ExecutionContinuationService",
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        "QUALIFIED — sole pause/wait/resolve/resume authority",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "ExecutionContinuationLifecycleDriver",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — PAUSE_REQUESTED→PAUSED→WAITING_FOR_HUMAN only",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "establish_canonical_hitl_pause",
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — canonical-first then Task projection",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "canonical_execution_is_resumed / progress gates",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — current episode RESUMED only",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "HumanPauseCoordinator / Task projection",
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        "projection only — not resume authority",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "InternalOrchestrationContinuation",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — capability bundle; not parallel engine",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "graph_runner._handle_needs_input",
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — NEEDS_INPUT projects wait; port owns pause",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "intake_runner long-running re-entry",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — Task CREATED only after canonical RESUMED",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "GovernedOrchestrationSlotContinuationExecutor.continue_slot",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — MSE+HITL gate requires port RESUMED",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.continuable_slots",
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        "projection only — eligibility; not resume authority",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.recover_failed_slot",
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        "QUALIFIED — refuses continuable HITL slots",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "mse_hitl_effect_gate / resolve_continuation_port_for_mse_hitl_gate",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — active store / injected port only",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "governed_continuation_bridge.apply_governed_continuation_pause",
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — fail-closed without active store",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "HumanPauseCoordinator.apply_human_response → apply_resolution",
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — resolution only; RESUME_AUTHORIZED ≠ RESUMED",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "checkpoint / LongRunningCoordinator",
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        "projection only — terminal authority may block; never resume permission",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "NexusLoop + HostTaskExecution shared store composition",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — production requires durable store; shared canonical store",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "validate_execution_continuation_for_composition (is_durable)",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — production requires store.is_durable; no type whitelist",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "ExecutionContinuationStateStore (pluginable persistence)",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — contract ABC; custom durable provider via composition",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "ReconstructedDurableExecutionContinuationStateStore / export",
        True,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
        "QUALIFIED — reference restart durability; named vendor adapter N/A",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "InMemoryExecutionContinuationStateStore",
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        "N/A — lab/test only; production rejects even when explicit",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "BackingExecutionContinuationStateStore (live reconnect)",
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        "N/A — reconnect only; is_durable=False; not production",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "NexusWorkerRuntime.from_registry (standalone worker)",
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        "QUALIFIED — explicit production_mode; durable store enforced in production",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "create_nexus_celery_worker_app / build_nexus_task_execution_registry",
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        "QUALIFIED — mode propagated; host_execution path reuses qualified host",
    ),
    Gr10OrchestrationContinuationInventoryRow(
        "wire_optional_queue_execution (prebuilt host_execution)",
        True,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        "QUALIFIED — no second NexusLoop; host continuation authority reused",
    ),
)


@dataclass(frozen=True, slots=True)
class Gr10OrchestrationMseNonToolInventoryRow:
    path: str
    production: bool
    consequential: bool
    boundary: str
    classification: str


GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY: tuple[Gr10OrchestrationMseNonToolInventoryRow, ...] = (
    Gr10OrchestrationMseNonToolInventoryRow(
        "RuntimeToolInvoker.invoke (contract.side_effects=True)",
        True,
        True,
        "MeaningfulSideEffectAuthorizationPort",
        "A — consequential + canonical MSE",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "RuntimeToolInvoker.invoke (side_effects=False)",
        True,
        False,
        "N/A",
        "C — N/A",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "External Work / governed contractor host boundary",
        True,
        True,
        "MeaningfulSideEffectAuthorizationBoundary (host-composed)",
        "A — consequential + already canonical MSE",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "OrchestrationExecutor / Nexus graph routing & state transitions",
        True,
        False,
        "CanonicalInnerExecutionGuardPort (identity) + Policy evaluation",
        "C — N/A (non-consequential orchestration control)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "PlanningRunner PRE_MODEL structured planning",
        True,
        False,
        "Runtime policy PRE_MODEL",
        "D — owned by Policy evaluation GEP",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "Physical delegation / agent distribution mutations",
        True,
        True,
        "PhysicalDelegationGovernanceBoundary",
        "D — owned by another canonical effect boundary",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "Decision-governed side effect helper (GR-6)",
        True,
        True,
        "MeaningfulSideEffectAuthorizationBoundary",
        "A — consequential + canonical MSE (non-tool caller)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "GraphExecutor nested child agent execution (ChildExecutionRunner → StrategyExecutionRouter)",
        True,
        True,
        "RuntimeToolInvoker / CatalogDeclarativeToolInvoker (ACP metadata)",
        "A — consequential effects only via governed tool contract",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "GraphExecutor orchestration topology slot (production fan-out coordination)",
        True,
        True,
        "PhysicalDelegationGovernanceBoundary via MultiAgentCoordinationService",
        "D — owned by Physical Delegation canonical boundary",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "GraphExecutor handoff / runtime events / execution-tree checkpoint",
        True,
        False,
        "Internal orchestration bookkeeping",
        "C — N/A (non-consequential platform control)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "ContextEngine provider.collect (graph context assembly)",
        True,
        False,
        "Read-only provider collection",
        "C — N/A (non-consequential read path)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.submit (custom slot)",
        True,
        True,
        "OrchestrationTopologySlotMsePolicy + MeaningfulSideEffectAuthorizationPort",
        "A — consequential + canonical MSE (mandatory composition enforcement)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.recover_failed_slot",
        True,
        True,
        "OrchestrationTopologySlotMsePolicy + fresh MeaningfulSideEffectAuthorizationPort",
        "A — consequential + fresh canonical MSE on recovery",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "CanonicalOrchestrationTopologySubmissionPort.continue_slot (custom continuation)",
        True,
        True,
        "OrchestrationTopologySlotMsePolicy + fresh MeaningfulSideEffectAuthorizationPort",
        "A — consequential + fresh canonical MSE on continuation physical effect",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "build_lab_orchestration_topology_submission_port (explicit non-production)",
        False,
        False,
        "N/A — lab/qualification only",
        "E — non-production (build_lab_orchestration_topology_submission_port)",
    ),
    Gr10OrchestrationMseNonToolInventoryRow(
        "Uncertified raw OrchestrationSlotExecutor bypassing canonical submission port",
        False,
        True,
        "N/A — not canonical production topology path",
        "E — non-production (bypasses CanonicalOrchestrationTopologySubmissionPort)",
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
        "INFERENCE N/A; ORCHESTRATION MSE QUALIFIED (GR-10-R9-R2); AGENTIC residual per SSOT.",
    ),
    Gr10CapabilityCell(
        "Decision-bound effect",
        gr10_matrix_inference_status("Decision-bound effect"),
        gr10_matrix_agentic_status("Decision-bound effect"),
        gr10_matrix_orchestration_status("Decision-bound effect"),
        "MP-4R7 agentic qualified; orchestration QUALIFIED (GR-10-R10 explicit policy binding).",
    ),
    Gr10CapabilityCell(
        "HITL",
        gr10_matrix_inference_status("HITL"),
        gr10_matrix_agentic_status("HITL"),
        gr10_matrix_orchestration_status("HITL"),
        "Agentic MP-4R7 qualified; orchestration GR-10-R11 QUALIFIED.",
    ),
    Gr10CapabilityCell(
        "Continuation",
        gr10_matrix_inference_status("Continuation"),
        gr10_matrix_agentic_status("Continuation"),
        gr10_matrix_orchestration_status("Continuation"),
        "ExecutionContinuationPort sole authority; orch GR-10-R12-R1 durable production QUALIFIED.",
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
        "ORCH-DECISION-BOUND",
        "ORCHESTRATION",
        "decision requirement before orchestration consequential effect",
        (
            _nid(
                "tests/unit/runtime/architecture/"
                "test_gr10_r10_orchestration_decision_bound_e2e.py",
                "test_mse_allow_without_required_decision_zero_effect",
            ),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r10_orchestration_decision_bound_qualification.py",
                "test_gr10_r10_orchestration_decision_bound_qualified",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
    ),
    Gr10ScenarioEvidence(
        "ORCH-HITL",
        "ORCHESTRATION",
        "HITL / continuation",
        (
            _nid(_GR5_ORCH, "test_resume_authorized_blocks_planning"),
            _nid(
                "tests/qualification/governance/strategy/"
                "test_gr10_r11_orchestration_hitl_qualification.py",
                "test_gr10_r11_orchestration_hitl_qualified",
            ),
            _nid(
                "tests/unit/runtime/architecture/"
                "test_gr10_r11_orchestration_hitl_e2e.py",
                "test_scenario_b_approval_plus_allow_executes_once",
            ),
            _nid(
                "tests/unit/runtime/architecture/"
                "test_gr10_r11_orchestration_hitl_e2e.py",
                "test_scenario_c_approval_plus_deny_zero_effect",
            ),
            _nid(
                "tests/unit/runtime/architecture/"
                "test_gr10_r11_r2_post_hitl_approval_evidence.py",
                "test_scenario_a_ordinary_allow_no_continuation_no_grant_proceeds",
            ),
            _nid(
                "tests/unit/runtime/architecture/"
                "test_gr10_r11_r2_post_hitl_approval_evidence.py",
                "test_scenario_c_post_hitl_resumed_missing_grant_blocks",
            ),
            _nid(
                "tests/unit/runtime/nexus/tools/"
                "test_gr10_r11_r2_runtime_tool_invoker_post_hitl.py",
                "test_invoker_ordinary_allow_without_grant_provider_once",
            ),
            _nid(
                "tests/unit/runtime/nexus/tools/"
                "test_gr10_r11_r2_runtime_tool_invoker_post_hitl.py",
                "test_invoker_post_hitl_resumed_missing_grant_zero_provider_calls",
            ),
        ),
        Gr10CoverageStatus.QUALIFIED,
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
