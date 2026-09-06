# DIAG-8A - AI Incident Investigation Ownership Audit

> **Historical audit (2026-08-26).** Superseded for current architecture by [`docs/project/architecture/DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) and [`DIAGNOSTIC_HARDENING_CLOSEOUT.md`](../qualification/DIAGNOSTIC_HARDENING_CLOSEOUT.md). Retained as decision record - not canonical current architecture.

**Audit ID:** DIAG-8A  
**Date:** 2026-08-26  
**Branch:** `development`  
**Scope:** `platform_proofs/scenarios/ai_incident_investigation/` vs central Diagnostic Engine  
**Mode:** documentation / architecture audit only - no implementation

---

## 1. Executive verdict

**One-spine question (§19):** Does `ai_incident_investigation` constitute a second generic Diagnostic Engine?

**Answer: NO** - domain investigator only, but terminology and authority boundaries need cleanup.

**Evidence:**

| Central platform responsibility | Present in scenario? |
|---|---|
| Canonical `RuntimeEvent` execution truth | **No** - zero imports/references |
| Causal evidence reconstruction | **No** |
| `ExecutionReconstructor` / lifecycle anomaly analysis | **No** |
| `DiagnosticAssessment` / `Problem` / `ProblemId` | **No** |
| `ProblemGroupingEngine` / cross-run stable grouping | **No** |
| Generic Problem persistence | **No** |
| Tenant-isolated diagnostic read surface | **No** (uses fixed `scenario-tenant`) |

The scenario **does** own legitimate domain investigation intelligence (hypotheses H1/H2/H3, tool-gathered manufacturing evidence, model proposal, scenario-local L0 critic predicates, falsification loop). It **reuses** platform execution spine (`GraphExecutor`, `Critic` hooks, `EvidenceBackedClaim` contracts) without reimplementing generic diagnostics.

**Residual risks (not second-engine duplication):**

- Terminology conflates **scenario investigation conclusion** (`RESOLVED`, `incident.root_cause_diagnosis`, `ClaimResolution.SUPPORTED`) with **platform root-cause authority**.
- Parallel domain disposition derivations exist in `domain_reasoning.py` (evaluator oracle) and `validation.py` (runtime critic authority) - same predicates, two call sites; maintenance hazard, not platform duplication.
- Scenario does not consume central diagnostics today - **integration gap**, not ownership violation.

---

## 2. Terminology definitions (§14)

| Term | Recommended meaning | Current scenario usage | Risk |
|---|---|---|---|
| **Problem** | Stable platform-tracked operational diagnostic pattern (`ProblemId`, recurrence, tenant scope) | Used only in README narrative (“production problem”) | Low - no code persistence |
| **Incident Investigation** | Domain workflow investigating one scope with competing hypotheses | Correct - primary scenario identity | None |
| **Hypothesis** | Provisional explanatory possibility (H1/H2/H3) | `HypothesisProposal`, `HypothesisDisposition` (model-owned, non-authoritative) | Low |
| **EvidenceBackedClaim** | Auditable structured conclusion with evidence refs | Reused from `intergrax.contracts.evidence_claims` | Good reuse |
| **Root Cause** | Causal claim with platform proof requirements (`PROVEN` only from canonical causal evidence under DIAG-8+) | Claim kind `incident.root_cause_diagnosis`; `RESOLVED` outcome | **High semantic risk** |
| **Operational resolution** | Remediation / closure of underlying Problem | Not modeled | N/A |
| **InvestigationConclusion** (target) | Derived scenario outcome separate from Problem lifecycle | Conflated with `OUTCOME_RESOLVED` / `OUTCOME_UNRESOLVED` | **Medium** |

---

## 3. Current architecture

```text
GraphExecutor (platform)
    └── IncidentInvestigatorAgent (scenario)
            ├── gather_incident_evidence (bounded tool loop, domain tools)
            ├── propose_incident_reasoning (LLM → IncidentReasoningProposal)
            ├── convert_proposal_to_pending_claims (→ EvidenceClaimSet, all PENDING)
            └── domain_payload (claim_set, evidence_nodes, completion_mode, …)

Post-execution (scenario.py):
    ├── apply_critic_claim_resolutions (deterministic domain predicates)
    ├── Critic hooks + IncidentInvestigationValidationEngine (L0)
    ├── derive_terminal_outcome → RESOLVED | UNRESOLVED
    └── ScenarioEvaluationResult → proof evidence / evaluator

Parallel oracle path (post-run only):
    └── evaluator.py + derive_hypothesis_dispositions (fixture truth checks)
```

**Central Diagnostic Engine:** not wired. Scenario is self-contained synthetic fixture world.

---

## 4. Ownership matrix (§4)

| Responsibility | Current owner / file | Input | Output | Authority | Canonical vs derived | Target owner | Migration? | Rationale |
|---|---|---|---|---|---|---|---|---|
| RuntimeEvent execution truth | Platform (`intergrax/runtime/diagnostics`) | Run/attempt events | Reconstructed execution | Platform canonical | Canonical | Platform | No (scenario should consume) | Scenario does not read RuntimeEvents |
| Causal platform evidence | Platform diagnostic orchestrator | Runtime events, signals | Causal evidence summaries | Platform | Canonical | Platform | No | Not duplicated |
| Execution reconstruction | `ExecutionReconstructor` | RuntimeEvent stream | `ExecutionReconstruction` | Platform | Canonical | Platform | No | Absent in scenario |
| Lifecycle anomaly analysis | `LifecycleAnomalyAnalyzer` | Lifecycle transitions | `LifecycleAnalysis` | Platform | Canonical | Platform | No | Absent |
| DiagnosticAssessment / Problem | `DiagnosticOrchestrator`, persistence | Orchestration request | `Problem`, assessments | Platform | Canonical | Platform | **Yes - consume** | Scenario has no Problem identity |
| Problem grouping / recurrence | `ProblemGroupingEngine`, lifecycle | Feature vectors, keys | Stable `ProblemId` | Platform | Canonical | Platform | **Yes - consume** | No cross-run grouping in scenario |
| Tenant isolation (diagnostics) | Platform read service | `tenant_id` | Scoped Problem views | Platform | Canonical | Platform | Later | Scenario uses fixed tenant string |
| Generic diagnostic subject identity | `DiagnosticSubjectRef` | App/execution refs | Subject tokens | Platform | Canonical | Platform | Later | Not used |
| Graph execution / agent step | `scenario.py`, `GraphExecutor` | Task, agent registry | `AgentExecutionResult` | Platform runtime | Canonical execution envelope | Platform | No | Correct reuse |
| Platform Critic falsification loop | `build_critic_graph_hooks`, evaluator loop | Agent output | `CriticVerdict` | Platform mechanism | Platform process | Platform | No | Scenario configures, does not replace |
| Evidence claim contracts | `intergrax.contracts.evidence_claims` | Structured claims | `EvidenceClaimSet` | Shared contract | Contract (not truth) | Platform contract | No | Good reuse |
| Domain hypotheses H1/H2/H3 | `incident_reasoning.py`, fixtures | Model + evidence | `HypothesisProposal` | Model provisional | Derived | Scenario | No | Legitimate domain intelligence |
| Model reasoning proposal | `incident_reasoning.py` | Evidence nodes, prior state | `IncidentReasoningProposal` | Model (non-authoritative) | Derived | Scenario | No | Valid |
| Evidence gathering strategy | `evidence_gathering.py` | Scope, planner LLM | `evidence_nodes`, tool traces | Model + bounded loop | Derived | Scenario | No | Domain tool orchestration |
| Domain evidence (workload, staffing, telemetry) | `tools.py`, fixtures | Synthetic fixture | Tool payloads | Fixture / tool | Domain evidence | Scenario / fixture | No | Not platform analytics |
| Domain predicate library | `domain_reasoning.py` | Typed observations | Predicates, `IncidentAssessment` | Deterministic domain | Derived | Scenario | No | Manufacturing-specific |
| Claim resolution (diagnosis claims) | `validation.apply_critic_claim_resolutions` | PENDING claims + observations | `ClaimResolution` per claim | **Scenario L0 critic** | **Scenario-authoritative for proof** | Scenario (renamed scope) | **Rename/separate** | Uses platform enum for domain conclusions |
| Claim content validation | `validation.validate_claim_set_against_observations` | Resolved claim set | `ValidationResult` | Scenario L0 critic | Derived gate | Scenario | Later align naming | Workflow quality gate |
| Critic → EvidenceChallenge mapping | `critic_adapter.py` | `CriticVerdict` | `EvidenceChallenge` | Adapter | Derived trace | Scenario | No | Thin projection, no verdict synthesis |
| Terminal investigation outcome | `scenario.derive_terminal_outcome` | Critic pass, supported claim, completion_mode | `RESOLVED` / `UNRESOLVED` | Scenario workflow | **Derived investigation conclusion** | Scenario (`InvestigationConclusion`) | **Rename** | Not Problem closure |
| Proof evaluation / oracle | `evaluator.py` | Execution result + fixture private truth | `ScenarioEvaluationResult` | Proof harness | Derived test oracle | Proof layer | No | Post-run only, not runtime truth |
| Proof evidence projection | `evidence_builder.py`, `evaluator_evidence.py` | Evaluation + result | `PlatformProofEvidence` | Proof artifact | Derived | Proof layer | No | Publication surface |
| Investigation observability | `investigation_observability.py` | Planner/reasoning events | `DiagnosticPayload` traces | Scenario telemetry | Derived trace | Scenario | No | Must not be mistaken for RuntimeEvent truth |
| Scenario evidence store | `tools.ScenarioEvidenceStore` | Tool reads | In-memory nodes | Session-local | Fixture/session | Scenario | No | Allowed local store |
| Cross-run Problem persistence | - | - | - | - | - | Platform | N/A | **Absent - good** |

---

## 5. Incident reasoning audit (§5)

| Concept | Classification | Notes |
|---|---|---|
| `HypothesisDisposition` | **Legitimate provisional investigation** | Documented non-authoritative; values mirror claim vocabulary but live only on `HypothesisProposal` |
| `HypothesisProposal` | **Legitimate scenario-domain** | Model-owned provisional semantics; no evidence-ID fields |
| `ClaimProposal` | **Model-owned semantics only** | `hypothesis_id`, `statement`, `claim_kind`, `rationale`, `replaces_prior_claim` — **no** evidence-ID fields |
| `ClaimEvidenceAttribution` | **Domain-owned evidence binding** | Pure deterministic policy in `claim_evidence_attribution.py` |
| `IncidentReasoningProposal` | **Legitimate scenario-domain** | Structured model output; drives gathering intent, not final authority |
| `CompletionIntent` | **Legitimate provisional** | Maps to `completion_mode`; gated by critic + claim resolutions |
| `ClaimHypothesisBinding` | **Legitimate scenario-local semantic** | Independent claim identity ↔ hypothesis; not a platform concept |
| `convert_proposal_to_pending_claims` | **Correct platform contract reuse** | Mints claim IDs; applies domain attribution; forbids model resolution via `PENDING` only |

**DS-E2E-12 ownership freeze:** Model owns semantic claim proposal. Domain attribution owns evidence relation. Critic owns resolution. Decision verification owns structural evidence verification. Lifecycle owns authority. Silent model-output repair is forbidden.

**Downstream misuse check:** No code path treats `HypothesisDisposition.SUPPORTED` as `ClaimResolution.SUPPORTED`. Disposition is observability/prompting only. Authority path is `ClaimProposal` → domain attribution → `PENDING` → `apply_critic_claim_resolutions` → critic validation.

`FORBIDDEN_MODEL_RESOLUTIONS` is declared in `incident_reasoning.py` but enforcement is effectively in `apply_critic_claim_resolutions` (rejects non-`PENDING`).

---

## 6. Claim authority (§6)

### Authority transition chain

```text
Model ClaimProposal (semantic only)
    → attribute_claim_evidence (domain deterministic policy)
    → convert_proposal_to_pending_claims (resolution = PENDING only)
    → apply_critic_claim_resolutions (validation.py - deterministic domain predicates)
    → IncidentInvestigationValidationEngine.validate (L0 critic via NexusValidationEngine)
    → CriticVerdict.passed + validate_claim_set_against_observations
    → derive_terminal_outcome
```

### Who may change `ClaimResolution`?

| Actor | May set resolution? | Scope |
|---|---|---|
| LLM / `IncidentReasoningProposal` | **No** (only `PENDING` at conversion) | - |
| `apply_critic_claim_resolutions` | **Yes** | Scenario-domain diagnosis claims (`incident.root_cause_diagnosis`) |
| `IncidentInvestigationValidationEngine` | **Validates** (does not mutate resolutions) | Scenario workflow gate |
| Platform Critic (generic) | **Indirect** - fails validation, triggers revision loop | Workflow quality, not Problem lifecycle |
| Central Diagnostic Engine | **No** (not connected) | Future root-cause promotion only |

**Verdict:** Claim resolution is **scenario-domain claim resolution** for the proof workflow. It is **not** platform generic root-cause resolution. A supported H3 claim means “bounded equipment degradation explanation best supported by gathered domain evidence in this investigation,” not `DiagnosticAssessment.certainty = PROVEN`.

---

## 7. `derive_terminal_outcome` semantics (§7)

**Definition (code):** `RESOLVED` iff `critic_verdict_passed ∧ has_supported_diagnosis ∧ completion_mode == supported_diagnosis`.

| Possible meaning | Applies? |
|---|---|
| A. Investigation workflow completed with supported **domain** conclusion | **Yes - actual meaning** |
| B. Underlying operational **Problem** is resolved | **No** |
| C. Platform **root cause PROVEN** | **No** |
| D. Incident **remediation** completed | **No** |

`UNRESOLVED` means epistemic refusal: critic passed, no `SUPPORTED` diagnosis claim, `completion_mode == unresolved`.

**Semantic risk:** README and SCENARIO_SPEC describe `RESOLVED` as “bounded operational **root-cause diagnosis**,” which readers may equate with platform root-cause proof. HTML reports label “Incident Outcome: RESOLVED” alongside proof PASS - correct for proof, ambiguous for Problem lifecycle.

**Non-accepted states:** Critic failure → `RuntimeError(incident_terminal_state_not_accepted)` - neither RESOLVED nor UNRESOLVED.

---

## 8. “Supported diagnosis” trace (§8)

| Symbol / field | Locations | Meaning |
|---|---|---|
| `COMPLETION_SUPPORTED_DIAGNOSIS` (`supported_diagnosis`) | `scenario_contract.py`, `scenario.py`, `incident_reasoning.py` | Model **intent** to complete with a supported diagnosis path |
| `DIAGNOSIS_KIND` / `incident.root_cause_diagnosis` | `scenario_contract.py`, claims | Scenario-domain material claim kind |
| `ClaimResolution.SUPPORTED` | After `apply_critic_claim_resolutions` | Evidence-backed **scenario** claim accepted by L0 predicates |
| `has_supported_diagnosis` | `scenario.py` | Any claim in resolved set with `SUPPORTED` |

**Ambiguity:** `incident.root_cause_diagnosis` + `SUPPORTED` reads like platform root-cause proof. Actual authority: deterministic scenario validation over domain tool evidence.

**Files requiring later rename/separation:**

- `scenario_contract.py` - `DIAGNOSIS_CLAIM_KIND`, `COMPLETION_SUPPORTED_DIAGNOSIS`
- `validation.py` - `DIAGNOSIS_CLAIM_KIND`, error tokens mentioning `supported_diagnosis`
- `evaluator_evidence.py` - user-facing “root-cause diagnosis” labels
- `README.md` / `SCENARIO_SPEC.md` - outcome tables

---

## 9. Central diagnostics consumption (§9)

### Current state

Scenario does **not** consume: `DiagnosticAssessment`, `Problem`, occurrences, grouping provenance, runtime history completeness, platform typed findings, or causal evidence summaries.

### Minimal useful bounded input (recommended)

```text
IncidentInvestigationInput
├── tenant_id: TenantId
├── problem_ids: tuple[ProblemId, ...]        # stable platform identity
├── subject_refs: tuple[DiagnosticSubjectRef, ...]
├── assessment_summary: DiagnosticAssessment    # findings + certainty + limitations (bounded)
├── occurrence_views: tuple[DiagnosticProblemOccurrenceView, ...]  # optional, capped
├── grouping_provenance: DiagnosticGroupingProvenance | None
├── evidence_refs: tuple[CanonicalEvidenceRef, ...]  # pointers, not RuntimeEvent payloads
└── orchestration_limitations: tuple[DiagnosticLimitation, ...]
```

**Not required for this proof:** full execution reconstruction in scenario, Problem persistence writes, manufacturing analytics from central engine.

**Rationale:** Central engine supplies trustworthy **starting scope and stable Problem identity**; scenario retains active investigation, tool use, and falsification.

---

## 10. Generic execution reconstruction audit (§10)

| Check | Result |
|---|---|
| Reads `RuntimeEvent` directly | **No** |
| Reconstructs generic execution history | **No** |
| Infers lifecycle anomalies | **No** |
| Creates cross-run problem grouping | **No** |
| Persists stable diagnostic Problems | **No** |

**Explicit statement:** No generic diagnostic engine duplication in execution-reconstruction or Problem persistence layers.

---

## 11. Domain vs platform evidence (§11)

| Class | Examples in scenario | Owner |
|---|---|---|
| **Platform execution evidence** | Would be RuntimeEvent, causal edges, Problem/DiagnosticAssessment | Platform (not used today) |
| **Domain evidence** | workload, throughput, staffing schedule/attendance, comparison, telemetry tool results | Scenario (`tools.py`, `evidence_gathering.py`) |
| **Derived analysis evidence** | `evidence.analysis.*` nodes from evaluate tools | Scenario |
| **Proof artifact evidence** | `PlatformProofEvidence`, claim graph in JSON/HTML | Proof layer |

Central diagnostics must **not** become a manufacturing analytics engine. Domain tool plane remains scenario-owned.

---

## 12. Critic ownership (§12)

| Role | How used | Valid? |
|---|---|---|
| A. Generic platform governance / falsification | `build_critic_graph_hooks`, `EvaluatorLoopSpec`, `CriticTraceEmitter` | **Yes** |
| B. Scenario-local root-cause authority | `IncidentInvestigationValidationEngine` + `apply_critic_claim_resolutions` | **Yes, if scoped as domain L0** - not platform PROVEN |
| C. Workflow quality gate | `derive_terminal_outcome`, revision loop, challenge lifecycle | **Yes** |

**Target alignment:** Central diagnostics → bounded facts/Problems → investigator proposes → critic challenges → scenario `EvidenceBackedClaim` → `InvestigationConclusion`. Current implementation matches this **except** missing central input slice and naming cleanup.

---

## 13. Evidence claim contracts (§13)

**Reuse:** `EvidenceBackedClaim`, `EvidenceClaimSet`, `EvidenceChallenge`, `ClaimResolution` from `intergrax.contracts.evidence_claims`.

**Assessment:** Good reuse - scenario does **not** introduce parallel generic claim machinery. Scenario adds:

- `ClaimHypothesisBinding` (scenario semantic)
- `incident.root_cause_diagnosis` claim kind (domain-specific kind string - valid extension)
- `critic_adapter` challenge projection (GAP-1A alignment)

**DIAG-8 recommendation:** Platform root-cause adjudication should **extend** these contracts (e.g. promotion record linking domain claim → platform PROVEN), not duplicate new claim types.

---

## 14. Cross-run behavior (§15)

**Finding:** No cross-run stable identity or grouping. Each proof run is isolated (`mint_run_id`, in-memory `ScenarioEvidenceStore`, synthetic fixture).

**Good:** Scenario should consume central `ProblemId` when integrated; must not introduce its own grouping.

---

## 15. Persistence (§16)

| Store | Classification |
|---|---|
| `ScenarioEvidenceStore` (`tools.py`) | Temporary investigation / session state |
| `fixtures.py` synthetic data | Fixture / evidence store |
| `output/*.json`, HTML reports | Proof artifacts (not canonical platform) |
| Platform Problem persistence | **Not used** |

No scenario-local generic Problem persistence - compliant.

---

## 16. Observability (§17)

`investigation_observability.py` emits `DiagnosticPayload` subclasses (`incident.planner_decision.v1`, `incident.reasoning_update.v1`, etc.) via `RuntimeState.trace_event`.

| Emission | Nature |
|---|---|
| Planner decisions / stop / scope rejection | Scenario workflow telemetry |
| Reasoning update / evidence gap / claim proposed | Model reasoning trace |
| Completion intent | Provisional model intent trace |

These are **scenario investigation traces**, not canonical RuntimeEvent or Problem truth. Safe if consumers treat `TraceComponent.PLANNER` incident steps as derived observability.

---

## 17. Model output boundary (§18)

| Model field | Post-model validation | Can establish canonical Problem / platform root cause? |
|---|---|---|
| `preferred_hypothesis_id` | Used in payload / prompts; terminal outcome uses resolved claims | **No** |
| `HypothesisDisposition.*` | Non-authoritative; not copied to `ClaimResolution` | **No** |
| `ClaimProposal` | → `PENDING` only | **No** |
| `completion_intent` | Must align with `validate_claim_set_against_observations` + critic | **No** alone |
| `ClaimResolution` on model path | Blocked (`MODEL_SELF_APPROVED_ERROR`) | **No** |

**P0 paths:** None found where raw model output directly creates canonical Problem, platform root cause, execution truth, or durable resolution. Deterministic gates (`apply_critic_claim_resolutions`, `IncidentInvestigationValidationEngine`) sit between model and terminal outcome.

---

## 18. Legal scenario-local intelligence (§2, §23)

**Retain without migration to central engine:**

- Competing hypotheses H1/H2/H3 and correlation-trap narrative
- Autonomous bounded tool loop and planner decisions
- Manufacturing domain tools and analysis predicates
- Model hypothesis/claim **proposal** and revision under critic feedback
- Scenario L0 falsification and `EvidenceChallenge` lifecycle
- Epistemic `UNRESOLVED` path
- Proof evaluator oracle and publication evidence
- Synthetic fixture world

**Do not reduce to:** “display `DiagnosticAssessment` only.”

---

## 19. One-spine violations

| Violation type | Severity | Detail |
|---|---|---|
| Second generic Diagnostic Engine | **None** | No RuntimeEvent/Problem orchestration duplication |
| Terminology / authority conflation | **Medium** | `root_cause_diagnosis`, `RESOLVED`, evaluator copy |
| Dual disposition derivations | **Low** | `derive_hypothesis_dispositions` vs `apply_critic_claim_resolutions` |
| Central diagnostics not consumed | **Gap (future)** | Expected for DIAG-8B+ |
| `ClaimResolution` enum reused for domain hypothesis assessment in `domain_reasoning.HypothesisAssessment` | **Ambiguous** | Same enum, evaluator-oracle context only |

---

## 20. Target architecture (§20)

```text
Central Diagnostic Engine
    │  ProblemId, DiagnosticAssessment, limitations, evidence refs
    ▼
IncidentInvestigationInput (typed bounded contract)
    ▼
IncidentInvestigationContext (scenario session)
    ▼
AI Incident Investigator Agent
    ├── domain evidence gathering (tools)
    ├── model hypothesis / claim proposals (PENDING only)
    └── scenario observability traces
    ▼
Platform Critic loop (workflow falsification)
    ▼
Scenario L0 claim adjudication (domain predicates)
    ▼
EvidenceBackedClaim set (scenario authority)
    ▼
InvestigationConclusion { RESOLVED | UNRESOLVED | REJECTED }
    │
    └── optional: RootCausePromotionRequest → Platform adjudicator
              → PROVEN | NOT_PROVEN | INSUFFICIENT_EVIDENCE
              (canonical causal evidence required for PROVEN)
```

---

## 21. Target boundary contract (§21)

**DIAG-8B implemented** in `intergrax/runtime/diagnostics/investigation_contracts.py`.

```python
@dataclass(frozen=True)
class IncidentInvestigationProblemContext:
    problem: DiagnosticProblemSummary
    occurrences: tuple[DiagnosticProblemOccurrenceView, ...]

@dataclass(frozen=True)
class IncidentInvestigationInput:
    tenant_id: str
    problem_contexts: tuple[IncidentInvestigationProblemContext, ...]

@dataclass(frozen=True)
class InvestigationConclusion:
    status: InvestigationConclusionStatus  # SUPPORTED | UNRESOLVED | NOT_ACCEPTED
    investigated_problem_ids: tuple[ProblemId, ...]
    claim_set: EvidenceClaimSet | None = None
    summary: str | None = None
```

**Canonical reuse:** `ProblemId`, `DiagnosticProblemSummary`, `DiagnosticProblemOccurrenceView`, `DiagnosticFinding`, `DiagnosticLimitation`, `DiagnosticGroupingProvenance`, `EvidenceBackedClaim`, `EvidenceClaimSet`.

**Occurrence choice:** reuse `DiagnosticProblemOccurrenceView` directly - already bounded, carries typed subject refs, read status, optional `DiagnosticAssessment`, and `NON_EXECUTION_SUBJECT` unavailable reason without coupling investigators to read-service pagination internals.

**Evidence refs:** no standalone `tuple[str, ...]` handoff in v1; typed refs remain inside `DiagnosticFinding` / `DiagnosticLimitation` (`EventId`, etc.). Richer evidence navigation is future work.

**Mapping helper:** `incident_investigation_input_from_problem_details(tenant_id, details)` projects `DiagnosticProblemDetail` → `IncidentInvestigationInput`. **DIAG-8C** wires `ai_incident_investigation` through `scenario_composition.py` → `DiagnosticReadService.get_problem(...)`.

**Forbidden in contract:** raw `RuntimeEvent` dumps, unbounded `dict[str, Any]` bags, scenario-side Problem minting, top-level mandatory `TaskId`/`RunId`, platform `ProblemStatus.RESOLVED` / `DiagnosticCertainty.PROVEN` as investigation conclusion status.

---

## 22. Root cause promotion model (§22 - design only)

```text
Scenario produces:
    RootCauseCandidate / EvidenceBackedClaim (kind: incident.root_cause_diagnosis, SUPPORTED in scenario scope)

Platform RootCauseAdjudicator (future):
    Input: RootCauseCandidate + ProblemId + canonical causal evidence bundle
    Output: RootCauseVerdict { PROVEN | NOT_PROVEN | INSUFFICIENT_EVIDENCE }
    Rule: PROVEN only if canonical causal criteria satisfied (DIAG-8+)
    LLM agreement or scenario critic pass alone → insufficient
```

---

## 23. Migration slices (§26)

| Slice | Goal | Status |
|---|---|---|
| **DIAG-8B** | Define `IncidentInvestigationInput` / `InvestigationConclusion` typed contracts; document mapping from `DiagnosticReadService` | **Complete** - `investigation_contracts.py`, unit + architecture gate tests |
| **DIAG-8C** | Scenario entry accepts optional `ProblemId`(s); seed investigation scope from `DiagnosticProblemSummary` + limitations (read-only) | **Complete** - `scenario_composition.py`, `platform_diagnostic_context.py`, integration + architecture gate tests |
| **DIAG-8D** | Rename scenario symbols: `InvestigationConclusion`, `incident.domain_diagnosis` claim kind; decouple `RESOLVED` from “root cause” in docs/UI |
| **DIAG-8E** | Platform `RootCauseAdjudication` contract; promotion gate from scenario `SUPPORTED` domain claim |
| **DIAG-8F** | Consolidate `derive_hypothesis_dispositions` with critic resolution helpers or mark evaluator-only; migrate tests |
| **DIAG-8G** | Proof evaluator labels and report copy - distinguish proof PASS vs investigation conclusion vs platform Problem state |
| **DIAG-8H** | Optional platform plugin seam: register scenario investigator as extension consuming central diagnostics |

---

## 24. Test ownership map (§24)

| Test file | Classification | Notes |
|---|---|---|
| `test_strict_outcome_boundary.py` | **KEEP-BUT-RENAME** | When `InvestigationConclusion` replaces outcome strings |
| `test_incident_model_reasoning.py` | **KEEP-AS-IS** | Model boundary / PENDING-only authority |
| `test_evidence_driven_reasoning.py` | **KEEP-AS-IS** | Domain predicates + validation |
| `test_autonomous_evidence_gathering.py` | **KEEP-AS-IS** | Domain tool loop |
| `test_analysis_tools.py` | **KEEP-AS-IS** | Domain tools |
| `test_runtime_composition.py` | **KEEP-AS-IS** | Runtime wiring |
| `test_skeleton_integration.py` | **KEEP-AS-IS** | End-to-end scenario |
| `test_full_resolved_evidence_world.py` | **KEEP-AS-IS** | Proof path |
| `test_full_unresolved_evidence_world.py` | **KEEP-AS-IS** | Proof path |
| `test_dual_canonical_evidence.py` | **KEEP-AS-IS** | Proof artifacts |
| `test_canonical_reproduction.py` | **KEEP-AS-IS** | Reproduction |
| `test_single_execution_provenance.py` | **KEEP-AS-IS** | Provenance |
| `test_incident_report_quality.py` | **KEEP-BUT-RENAME** | Report copy after terminology split |
| `conftest.py`, `planner_doubles.py` | **KEEP-AS-IS** | Test infra |
| - | **ADD-LATER** | Root-cause promotion gate tests (DIAG-8E) |
| `test_investigation_contracts.py` | **ADDED (DIAG-8B)** | Boundary validation, execution + application-instance occurrences, tenant integrity |
| `test_investigation_contracts_no_proof_import_gate.py` | **ADDED (DIAG-8B)** | Platform contract must not import proof scenario |
| - | **ADD-LATER** | Scenario wiring tests consuming `DiagnosticReadService` output (DIAG-8C) |
| `test_diagnostic_platform_integration.py` | **ADDED (DIAG-8C)** | Real central Problem → read service → scenario composition → execution |
| `test_diagnostic_architecture_gate.py` | **ADDED (DIAG-8C)** | Composition-only read surface; no persistence ownership in scenario runtime |
| - | **MIGRATE-TO-CENTRAL-DIAGNOSTIC-CONTRACT** | Any future test asserting scenario owns ProblemId |
| - | **DELETE-AS-DUPLICATE** | None today |

---

## 25. Risks

1. **Semantic:** Operators interpret proof `RESOLVED` as platform Problem closure or PROVEN root cause.
2. **Integration:** Future wiring may tempt scenario to reimplement Problem grouping when central input is delayed.
3. **Maintenance:** Duplicated disposition logic between `validation.py` and `domain_reasoning.py` may drift.
4. **Enum overload:** `ClaimResolution` used for both platform-neutral claims and domain-only evaluator assessments.
5. **Plugin boundary:** `runtime_composition.py` already bridges application environment - clear extension point exists but is not formalized.

---

## 26. P0 findings

**None.** No path from raw model output to canonical Problem, platform execution truth, or durable platform resolution without deterministic scenario gates.

---

## 27. Remaining uncertainties

1. Whether `incident.root_cause_diagnosis` should remain a `ClaimKind` or split into `incident.domain_diagnosis` before platform promotion naming freeze.
2. Which `DiagnosticFindingKind` subset is sufficient seed for manufacturing incident investigations.
3. Whether proof scenarios should bind to a real `ProblemId` from a preceding diagnostic orchestration run or continue standalone with optional attachment.
4. Plugin packaging: `platform_proofs` vs `applications/` host for production incident investigator.

---

## 29. DIAG-8C - platform-attached investigation (complete)

**Modes:**

| Mode | Entry | `investigation_input` | Task tenant |
|---|---|---|---|
| **Standalone synthetic proof** | `build_runtime_bundle()` | `None` | `scenario-tenant` |
| **Platform-attached** | `build_runtime_bundle_from_diagnostic_problem(...)` | `IncidentInvestigationInput` | `input.tenant_id` |

**Data path (attached mode only):**

```text
ProblemPersistence (central - not imported by scenario runtime)
    ↓
DiagnosticReadService.get_problem(tenant_id, problem_id)
    ↓
DiagnosticProblemDetail
    ↓
incident_investigation_input_from_problem_details(...)
    ↓
IncidentInvestigationInput
    ↓
IncidentInvestigatorAgent (constructor-owned immutable context)
    ↓
build_reasoning_messages / domain evidence tools
    ↓
InvestigationConclusion on ScenarioExecutionResult (attached mode)
```

**Ownership:** scenario composition (`scenario_composition.py`) is the only module importing `DiagnosticReadService`. Reasoning modules consume materialized `IncidentInvestigationInput`; no persistence, reconstruction, grouping, lifecycle, or orchestrator imports in scenario runtime.

**Identity:** investigation run mints new `TaskId`/`RunId`; investigated occurrence subject retains its own execution or application-instance identity - never reused as investigation run identity.

---

## 28. Audit metadata

- **Start HEAD:** `24506c3c14e30984d78b7b22c5cd4c42e711d125`
- **DIAG-8B HEAD:** see git final SHA in DIAG-8B report
- **Files inspected:** all modules listed in DIAG-8A §3 plus 15 unit test files
- **Central baseline:** `intergrax/runtime/diagnostics/__init__.py` exports (DiagnosticAssessment, Problem lifecycle, orchestrator, read service, reconstruction, investigation contracts)
- **Implementation:** DIAG-8A audit only; **DIAG-8B boundary contracts** in `intergrax/runtime/diagnostics/investigation_contracts.py` (no scenario or central-engine behavior changes)
