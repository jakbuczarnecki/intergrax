# Adaptive Harness Evolution — Implementation Roadmap

**Revision:** 1.3
**Status:** ARCHITECT-DEFINED / OWNER_BOUNDARIES_ACCEPTED_UNDER_OD-AHE-02 / W0_CLOSED
**Architecture reference:** [ADAPTIVE_HARNESS_EVOLUTION.md](../../../architecture/satellites/ADAPTIVE_HARNESS_EVOLUTION.md)

Execution register and implementation work breakdown remain in this document.

---

## Program status

| Item | Status |
|---|---|
| Closed waves | 1 |
| Formally closed packages | 3 |
| W1 executable authorization | NOT_AUTHORIZED |

**Cross-program coordination:** Explicit. AHE roadmap dependencies do not themselves authorize changes to other programs.

## Operator decision register

| ID | Date | Status | Record |
|---|---|---|---|
| OD-AHE-01 | 2026-11-27 | APPROVED | I approve the full scope of the presented operator decision, including the limited historical waiver **F-AHE-01**. Approved scope: (1) AHE architecture revision **1.3**; (2) AHE roadmap revision **1.3**; (3) independent qualification **AHE-W0-IQ-01** within documentation-only scope; (4) historical waiver of **F-AHE-01** for exactly three specified commits; (5) preservation of Git history without rewrite; (6) ADAS remains the canonical runtime implementation owner; (7) existing repository governance and publication rules remain applicable; (8) authorization to prepare **W1-B01** after settlement of remaining G0 requirements. Does not authorize product-code implementation, production authority, external program-owner acceptance replacement, or G6/IIAC certification. |
| OD-AHE-02 | 2026-11-27 | APPROVED | As operator and authorized owner of the stated domains, I approve the full presented scope of responsibility boundaries, preparation of exact future canonical ADAS register updates, and formal W0 settlement. Authorizes preparation of exact future ADAS register updates only; does not approve a product implementation ticket. |
| OD-AHE-02 ownership boundaries | 2026-11-27 | RECORDED | See architecture §5.18 and **OD-AHE-02 ownership boundary table** below. Acceptance authority: Operator within declared ownership authority only. |



## OD-AHE-02 ownership boundary table

| Domain | Acceptance |
|---|---|
| AHI / ADAS | AHE delivers existing canonical ADAS obligations without competing implementation |
| Integrax Intelligence | Retains consumer wiring, roadmap and program-level acceptance |
| Scenario Factory | Consumes accepted capabilities through existing platform interfaces |
| ARCH-FREEZE | No scope expansion and no additional freeze gates |
| Execution | Retains canonical execution authority |
| Evaluation | Retains canonical evaluation authority |
| Memory | Retains trusted experience authority |
| Governance | Retains admission, approval and promotion authority |

## Independent documentation qualification — AHE-W0-IQ-01

| Item | Record |
|---|---|
| Audited SHA | `1d0181051fa034b5f45101755002e48f5dc35715` |
| Audited architecture revision | 1.3 |
| Audited roadmap revision | 1.3 |
| Scope | Architecture and roadmap documentation only |
| Independent auditor final verdict (preserved) | `CONDITIONAL_ARCHITECTURE_QUALIFIED / PROGRAM_G0_NOT_CLOSABLE` |
| Operator disposition (separate) | `INDEPENDENT_DOCUMENTATION_QUALIFICATION_ACCEPTED_SCOPED_UNDER_OD-AHE-01` |

| Dimension | Verdict |
|---|---|
| Architecture qualification | PASS_SCOPED |
| Roadmap qualification | PASS_SCOPED |
| Canonical ADAS compatibility | PASS_ARCHITECTURE_ONLY |
| Reuse qualification | PASS |
| Cross-program boundaries | PASS_ARCHITECT_DEFINED |
| W1 implementation boundary | PASS_DEFINITION |
| Enterprise safety | PASS_INVARIANTS_ON_PAPER |
| Historical publication governance | FAIL_WITH_OPERATOR_WAIVER |
| Executable behavioral qualification | NOT_EXECUTED |
| G6 certification | NOT_CLAIMED |
| IIAC certification | NOT_CLAIMED |

## Historical publication waiver — F-AHE-01

| Item | Record |
|---|---|
| Finding | F-AHE-01 |
| Disposition | HISTORICAL_NONCOMPLIANCE_ACCEPTED_BY_OPERATOR |
| Waiver authority | OD-AHE-01 |
| Historical publication compliance (waived commits) | FAIL (not retroactively relabeled compliant) |
| Corrective publication evidence SHA | `1d0181051fa034b5f45101755002e48f5dc35715` |
| Corrective commit metadata | PASS |
| Waiver exhausted | Yes — exactly three commits; no future-commit exception |

| Historical commit | Historical publication compliance |
|---|---|
| `a4ff0130dbaf054bfdaeb8522644c2f8b39d28ac` | FAIL |
| `ae93859a5e9574b8402f681cb322c1d86be8f772` | FAIL |
| `8a4f962e25bd72ab0c183d0d25437f93a8c001d6` | FAIL |

No history rewrite authorized. Future commits, PR text, AI attribution, mandatory gates and publication procedures remain fully applicable.

## Non-blocking independent audit observations

| Observation | Disposition |
|---|---|
| Missing AHE reference from the AHI hub | Deferred — outside authorized write set for AHE-W0-SETTLEMENT-01 |
| Reuse baseline versus current documentation SHA | Exact-SHA evidence discipline required |
| Canonical ADAS publication reconciliation | Pending authorization (**F-AHE-05**) |

## ADAS traceability matrix

| AHE scope | ADAS source scope |
|---|---|
| AHE-W0 | Existing AHI-ADAS-00 documentation foundation |
| AHE-W1-01 | AHI-ADAS-10 |
| AHE-W1-02 | Selected AHI-ADAS-40 strategy and objective obligations |
| AHE-W1-03 | Selected AHI-ADAS-20 and AHI-ADAS-30 execution obligations |
| AHE-W1-04 | Selected AHI-ADAS-30 and AHI-ADAS-40 selection obligations |
| AHE-W1-05 | Integrated W1 behavioral qualification |
| AHE-W2 | AHI-ADAS-20 and AHI-ADAS-40 extensions |
| AHE-W3 | AHI-ADAS-30 extensions |
| AHE-W4 | AHI-ADAS-10, 30, 40, 50 and 90 extensions |
| AHE-W5 | AHI-ADAS-30 and AHI-ADAS-40 resource extensions |
| AHE-W6 | AHI-ADAS-50, 70 and 90 |
| AHE-W7 | Cross-phase behavioral qualification |
| AHE-W8 | Consumer integration and final acceptance |

Clarifications:

- Mapping is obligation-level traceability.
- Mapping does not automatically complete a referenced ADAS phase.
- AHI-ADAS-60 optional Tier-2 agents are not mandatory AHE scope.
- AHI-ADAS-80 optional Tier-3 application is not mandatory AHE scope.
- Unimplemented obligations remain explicitly tracked.
- The existing ADAS plan remains authoritative for its original obligations.

## W1 execution boundary

The first executable wave shall implement a minimal configuration-only search loop:

1. Resolve accepted objective and baseline.
2. Admit a bounded search.
3. Execute baseline through canonical runtime.
4. Generate one configuration candidate.
5. Validate its identity, scope, configuration delta and authority.
6. Execute candidate through canonical runtime.
7. Compare actual results using existing Evaluation mechanisms.
8. Select candidate, retain baseline or abstain.
9. Archive candidate lineage and evidence references.

Constraints:

- W1 must use an existing admitted execution path.
- W1 shall initially use a deterministic candidate strategy to qualify orchestration and evidence behavior.
- Model-driven proposal generation and multiple candidate composition shall be introduced under the relevant subsequent authorized scope.
- The deterministic W1 qualification does not prove general intelligent adaptation.
- No production promotion.
- No trusted cross-task experience publication.
- No external business effects.
- No new independent execution or governance authority.

## Dependency and gate register

| Gate | Requirement | Status |
|---|---|---|
| AHE-G0 | Architecture and roadmap acceptance; operator settlement; independent documentation qualification; historical publication waiver | ACCEPTED_WITH_HISTORICAL_WAIVER |
| AHE-G1 | Exact W1 implementation contract, approved ticket and scope authorization | PENDING |
| AHE-G2 | Executable bounded LAB adaptive loop | PLANNED |
| AHE-G3 | Propose–Probe–Compose and selection qualification | PLANNED |
| AHE-G4 | Trusted experience and resource optimization qualification | PLANNED |
| AHE-G5 | Enterprise governed evolution qualification | PLANNED |
| AHE-G6 | Independent behavioral qualification | PLANNED |
| AHE-G7 | Consumer adoption and final independent acceptance | PLANNED |

These are AHE program milestone names, not substitutes for DevOS G0–G6, IIAC, FRZ or existing admission controls.

### AHE-G0 subconditions

| Subcondition | Status |
|---|---|
| Architecture acceptance | PASS |
| Roadmap acceptance | PASS |
| Independent documentation qualification | ACCEPTED_SCOPED |
| Operator acceptance | APPROVED |
| Historical publication finding F-AHE-01 | WAIVED |
| External owner acceptance | PENDING |
| Settlement publication | VERIFIED (AHE-W0-SETTLEMENT-01 @ `550b258bf8d524d907b7720da41ecac64a6bb1a5`) |

### AHE-G1 outstanding requirements

- Architect-issued exact W1-B01 executable contract.
- Explicit canonical schema compatibility decisions.
- Authorized ADAS publication procedure.
- Exact canonical plan update scope.
- Valid DevOS ticket.
- Required Operator approval.
- Applicable repository governance checks.

W1 execution requires AHE-G0, AHE-G1 and all applicable existing governance authorizations.

No roadmap row or gate may be declared accepted solely by the implementer.

## Execution register

| Package | Implementation | Review / acceptance | Notes |
|---|---|---|---|
| AHE-W0-01 | CLOSED | Architecture acceptance: ACCEPTED_SCOPED; Independent documentation qualification: ACCEPTED_SCOPED; Historical publication finding F-AHE-01: WAIVED_UNDER_OD-AHE-01 | Commit: `a4ff0130dbaf054bfdaeb8522644c2f8b39d28ac` (historical publication compliance FAIL; not retroactively relabeled). |
| AHE-W0-02 | CLOSED | Architecture acceptance: ACCEPTED_SCOPED; Independent documentation qualification: ACCEPTED_SCOPED; Historical publication finding F-AHE-01: WAIVED_UNDER_OD-AHE-01; Cross-program boundaries: OWNER_BOUNDARIES_ACCEPTED_UNDER_OD-AHE-02 | Commit: `ae93859a5e9574b8402f681cb322c1d86be8f772` (historical publication compliance FAIL; not retroactively relabeled). |
| AHE-W0-03 | CLOSED | Architecture acceptance: ACCEPTED_SCOPED; Corrective compatibility status: ACCEPTED_SCOPED; Independent documentation qualification: ACCEPTED_SCOPED; Historical publication finding F-AHE-01: WAIVED_UNDER_OD-AHE-01 | Commit: `8a4f962e25bd72ab0c183d0d25437f93a8c001d6`. Corrective operation **AHE-W0-03-C01** (not a new roadmap package): corrective publication SHA `1d0181051fa034b5f45101755002e48f5dc35715`; corrective commit metadata PASS. |
| AHE-W1-01 (W1-B01) | PREPARATION / PENDING_TICKET_APPROVAL | Canonical ticket: `governance/tickets/T-IGX-093-A.md`; canonical plan: `governance/tickets/plans/PLAN_T-IGX-093-A.md`; lifecycle: `plan_pending` | Roadmap package **AHE-W1-01**; executable authorization **NOT_AUTHORIZED** until ticket approval and external implementation authorization. |
| W1–W8 (other) | PLANNED_SCOPE / EXECUTION_NOT_AUTHORIZED | W1 executable authorization: NOT_AUTHORIZED | — |
| AHE-G0 | ACCEPTED_WITH_HISTORICAL_WAIVER | W0 closed under OD-AHE-02 | — |
| AHE-G1 | PENDING | Approved ticket and scope authorization required | — |

**Closed waves:** 1 (W0). **Formally closed packages:** 3 (AHE-W0-01, AHE-W0-02, AHE-W0-03). **W0:** CLOSED.

## Process findings register

| ID | Finding | Disposition |
|---|---|---|
| F-AHE-01 | Repeated prohibited AI-tool attribution and violation of publication stop condition. | HISTORICAL_NONCOMPLIANCE_ACCEPTED_BY_OPERATOR (OD-AHE-01; three commits only; historical publication FAIL preserved) |
| F-AHE-02 | W0-03 execution register reconciliation. | RESOLVED |
| F-AHE-03 | Canonical ADAS contract compatibility. | RESOLVED_ARCHITECTURE_ONLY |
| F-AHE-04 | Candidate lifecycle vocabulary compatibility. | RESOLVED |
| F-AHE-05 | Canonical delivery and acceptance reconciliation. | PENDING_AUTHORIZATION |
| F-AHE-06 | Configuration-only versus scaffold materialization boundary. | RESOLVED_ARCHITECTURE_ONLY |
| F-AHE-07 | Restore historical W0-01 process finding. | RESOLVED |


---

## W0 — Architecture and canonical baseline

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W0-01 | Architecture and roadmap | Both architect-defined documents published; independent review pending |
| AHE-W0-02 | Canonical ownership and baseline reconciliation | Exact-SHA reuse mapping and cross-program boundary acceptance |
| AHE-W0-03 | W1 implementation contract | Concrete authorized write set, dependencies, tests and acceptance criteria |

**W0 settlement (AHE-W0-SETTLEMENT-01):** independent publication verification recorded; architect verdict **AHE-G0 ACCEPTED_WITH_HISTORICAL_WAIVER**; **W0: CLOSED**; packages **AHE-W0-01**, **AHE-W0-02**, **AHE-W0-03**: **CLOSED**.

**W0 exit:** accepted architecture, roadmap and implementation boundaries (pending final G0 settlement closure).

---

## W1 — Minimum Adaptive Loop

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W1-01 | Existing ADAS candidate contracts and archive foundation | Candidate identity, lineage, scope and immutable configuration references |
| AHE-W1-02 | Baseline and candidate proposal | One bounded candidate produced against an immutable baseline |
| AHE-W1-03 | Experimentation and Execution integration | Baseline and candidate executed through canonical authority |
| AHE-W1-04 | Comparative evaluation and selection | Evidence-backed candidate or baseline selection |
| AHE-W1-05 | End-to-end adaptive loop qualification | Real executable LAB task with positive and negative cases |

**W1 exit:** a complete executable baseline–propose–probe–select loop without production effects.

## W1 implementation contract

These batches define **PLANNED_SCOPE / EXECUTION_NOT_AUTHORIZED** for W1. They are not present execution authority until AHE-G0, AHE-G1 and applicable governance requirements are satisfied.

### W1-B01 — Candidate contracts and archive

**Implementation status:** PREPARATION / PENDING_TICKET_APPROVAL

**Canonical DevOS ticket:** [`T-IGX-093-A`](../../../../governance/tickets/T-IGX-093-A.md)

**Canonical implementation plan:** [`PLAN_T-IGX-093-A`](../../../../governance/tickets/plans/PLAN_T-IGX-093-A.md) (`plan_pending`; operator approval pending)

**Roadmap package:** AHE-W1-01

**Canonical obligation coverage:** AHI-ADAS-10.1 through AHI-ADAS-10.14, subject to exact obligation-level acceptance and explicit residual tracking.

**W1-B01 canonical obligation inventory (planned coverage; acceptance not claimed):**

| Canonical obligation | W1-B01 responsibility | Status |
|---|---|---|
| AHI-ADAS-10.1 | Importable canonical package skeleton, copyright headers and no import side effects | NOT_STARTED |
| AHI-ADAS-10.2 | Compatible operational envelope and provenance | NOT_STARTED |
| AHI-ADAS-10.3 | Compatible objective contract, scope, budgets and approval semantics | NOT_STARTED |
| AHI-ADAS-10.4 | Compatible search-run lifecycle, accounting and stop reasons | NOT_STARTED |
| AHI-ADAS-10.5 | Compatible candidate draft, lineage and configuration-only restriction | NOT_STARTED |
| AHI-ADAS-10.6 | Immutable candidate records and canonical lifecycle status | NOT_STARTED |
| AHI-ADAS-10.7 | Compatible evaluation result and canonical metric references | NOT_STARTED |
| AHI-ADAS-10.8 | Sealed evidence bundle and deterministic integrity | NOT_STARTED |
| AHI-ADAS-10.9 | Scoped archive protocol and query semantics | NOT_STARTED |
| AHI-ADAS-10.10 | In-memory archive with append-only and isolation semantics | NOT_STARTED |
| AHI-ADAS-10.11 | SQLite archive and restart durability | NOT_STARTED |
| AHI-ADAS-10.12 | Candidate lineage and historical queries | NOT_STARTED |
| AHI-ADAS-10.13 | Canonical package exports | NOT_STARTED |
| AHI-ADAS-10.14 | Contract, archive, isolation and evidence integrity tests | NOT_STARTED |

Explicit residual tracking is required for canonical capabilities outside W1, including:

- Executable agent scaffold artifacts.
- Promotion-related evidence.
- Later lifecycle states.
- Archive query behavior associated with later lifecycle states.
- Enterprise retention and production integration.

Do not mark any of these obligations complete through documentation alone.

The future W1-B01 executable contract must pin exact schema fields, validation rules, archive operations, test cases and canonical obligation-level acceptance.


**Proposed runtime write set:**

- `intergrax/runtime/adaptive/agent_design_search/__init__.py`
- `intergrax/runtime/adaptive/agent_design_search/contracts.py`
- `intergrax/runtime/adaptive/agent_design_search/archive.py`
- `intergrax/runtime/adaptive/agent_design_search/sqlite_archive.py`

**Proposed test write set:**

- `tests/unit/runtime/adaptive/agent_design_search/test_contracts.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_archive.py`
- `tests/unit/runtime/adaptive/agent_design_search/test_sqlite_archive.py`

The existing AHE documentation pair may be updated for evidence and status only when explicitly authorized by the executable batch.

**Required acceptance:**

- Typed contract validation.
- Immutable configuration identity.
- Scope binding.
- Deterministic evidence digests.
- Candidate lineage.
- Valid lifecycle transitions.
- Duplicate identity handling.
- In-memory and SQLite semantic parity.
- Cross-tenant and cross-workspace rejection.
- Restart durability.
- No execution or promotion side effects on package import.

**Mandatory planned tests:**

`uv run --no-sync pytest -q tests/unit/runtime/adaptive/agent_design_search/`

`uv run --no-sync pytest -q tests/unit/runtime/adaptive/`

Run the focused tests first. The adjacent invocation may exclude already executed focused tests to avoid unnecessary duplication.

Run compilation, applicable lint, typing and mandatory governance gates.

No modifications to existing runtime owners in this batch.

### W1-B02 — Integrated adaptive search

**Roadmap packages:**

- AHE-W1-02.
- AHE-W1-03.
- AHE-W1-04.

**Proposed runtime write set:**

- `intergrax/runtime/adaptive/agent_design_search/static_gate.py`
- `intergrax/runtime/adaptive/agent_design_search/search_policy.py`
- `intergrax/runtime/adaptive/agent_design_search/strategies.py`
- `intergrax/runtime/adaptive/agent_design_search/candidate_evaluator.py`
- `intergrax/runtime/adaptive/agent_design_search/experiment_bridge.py`
- `intergrax/runtime/adaptive/agent_design_search/search_controller.py`

Updates to W1-B01 contracts and exports require explicitly authorized additive scope.

**Proposed focused tests:**

- `test_static_gate.py`
- `test_search_policy.py`
- `test_strategies.py`
- `test_candidate_evaluator.py`
- `test_experiment_bridge.py`
- `test_search_controller.py`

All located under:

`tests/unit/runtime/adaptive/agent_design_search/`

**Required adjacent regression families:**

- Existing adaptive runtime tests.
- Existing ExperimentSession and experiment persistence tests.
- Existing Evaluation unit and integration tests.
- Existing execution budget enforcement tests.

The exact executable batch contract will pin selected commands and applicable dependencies after W1-B01 acceptance.

### W1-B03 — Behavioral qualification

**Roadmap package:** AHE-W1-05

**Proposed behavioral witness:**

`tests/integration/runtime/adaptive/agent_design_search/test_minimum_adaptive_loop.py`

**Required behavior:**

All positive and negative witnesses defined in Architecture §5.30.

Tests must execute actual canonical LAB behavior rather than compare fabricated results.

**Required evidence:**

- Execution SHA.
- Exact effective baseline and candidate configurations.
- Objective and dataset digests.
- Canonical root execution identities.
- Evaluation results.
- Resource and isolation evidence.
- Selected outcome.
- Restart persistence result.
- Focused and adjacent regression results.

W1 exit requires independently accepted behavioral evidence.

### Canonical ADAS publication boundary

The existing canonical ADAS plan prescribes one phase or narrowly scoped sub-phase per PR and corresponding canonical plan updates.

AHE implementation batches represent engineering coordination units. They do not automatically override canonical ADAS publication units.

The planned W1-B02 engineering batch spans obligations associated with AHI-ADAS-20, AHI-ADAS-30 and AHI-ADAS-40.

Its executable authorization must explicitly reconcile the engineering batch with applicable canonical publication and acceptance boundaries.

Final publication targets development through the repository-authorized procedure.

Direct-push authorization must not be inferred from this roadmap.

The exact executable write set must identify any authorized canonical ADAS register updates.

Protected cross-program documents must not be modified without explicit authorization.

`ADAS_DELIVERY_RECONCILIATION: PENDING_AUTHORIZATION`



### W1 authorization boundary

These planned write sets are not present execution authority.

W1-B01, W1-B02 and W1-B03 each require an architect-issued executable contract and applicable repository approval.

No product-path batch may begin without a valid ticket and required approval.

Do not reserve a competing ticket ID or fabricate an Approved line.

W1 remains unauthorized until AHE-G0, AHE-G1 and applicable existing governance requirements are satisfied.


Reuse applicable AHI-ADAS-10 through AHI-ADAS-40 obligations rather than duplicate their ownership (see [ADAPTIVE_HARNESS_INTELLIGENCE plan](../ADAPTIVE_HARNESS_INTELLIGENCE.md)).

---

## W2 — Propose–Probe–Compose

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W2-01 | Invariant/domain refinement | Separate change scopes and leakage prevention |
| AHE-W2-02 | Multiple bounded candidates | Distinct validated configuration alternatives |
| AHE-W2-03 | Sequential and parallel search | Comparable isolated trials within aggregate budgets |
| AHE-W2-04 | Candidate composition | Conflict rejection and immutable composite lineage |
| AHE-W2-05 | Composite verification | Re-evaluation of combined configuration before selection |

**W2 exit:** independently verified Propose–Probe–Compose behavior.

---

## W3 — Evaluation and candidate selection

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W3-01 | Multi-criteria evaluation | Deterministic, safety, business, quality and cost criteria |
| AHE-W3-02 | Blinded comparison | Candidate identity mapping and origin-blind assessment |
| AHE-W3-03 | Open-ended task assessment | Rubric-based advisory evaluation with independent evidence |
| AHE-W3-04 | Selection robustness | Order-bias, insufficient-evidence and fallback tests |

**W3 exit:** selection cannot override mandatory failures and supports explicit abstention.

---

## W4 — Experience and continual improvement

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W4-01 | Experience identity and evidence references | Scoped provenance, lineage and outcome binding |
| AHE-W4-02 | Existing persistence integration | Restart-safe retention and retrieval in the qualified scope |
| AHE-W4-03 | Experience applicability and ranking | Relevant retrieval with freshness and scope controls |
| AHE-W4-04 | Continual proposal improvement | Prior verified experience influences new proposals |
| AHE-W4-05 | Knowledge quality controls | Poisoning, stale knowledge, counterexamples and deletion tests |

**W4 exit:** measurable and safe experience reuse across distinct tasks.

---

## W5 — Adaptive resource optimization

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W5-01 | Experiment budget integration | Hard aggregate execution, token, cost and time limits |
| AHE-W5-02 | Bounded adaptive experimentation | Additional experiments justified by expected value |
| AHE-W5-03 | Early stopping | Deterministic termination at defined limits |
| AHE-W5-04 | Model and tool optimization | Evidence-backed recommendations within admitted capabilities |
| AHE-W5-05 | Economic evaluation | Cost per verified successful task measured |

**W5 exit:** bounded resource-aware adaptation, with no independent budget authority.

---

## W6 — Enterprise governed evolution

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W6-01 | Promotion authority integration | Exact candidate, version, scope and approval binding |
| AHE-W6-02 | Mandatory evidence completeness | Missing mandatory evidence fails closed |
| AHE-W6-03 | Shadow/canary integration | Qualified controlled evaluation without duplicate effects |
| AHE-W6-04 | Activation consistency | Existing CAS, fencing, idempotency and recovery semantics |
| AHE-W6-05 | Rollback and restart qualification | Recovery preserves authority and configuration lineage |
| AHE-W6-06 | Enterprise isolation and trust qualification | Tenant, permission, poisoning and adversarial negative tests |

**W6 exit:** independent acceptance of the qualified enterprise adaptation scope.

Do not interpret W6 as permission to begin security implementation late. Relevant controls are mandatory throughout W1–W5.

---

## W7 — Behavioral qualification

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W7-01 | Frozen benchmark protocol | Tasks, datasets, model settings, budgets and criteria pinned |
| AHE-W7-02 | Baseline and budget-matched control | Comparable baseline and repeated-baseline evidence |
| AHE-W7-03 | Adaptive benchmark | Measured adaptive task performance |
| AHE-W7-04 | Adaptive plus experience benchmark | Measured incremental experience benefit |
| AHE-W7-05 | Independent holdout and safety validation | No evaluation contamination; negative safety evidence |
| AHE-W7-06 | Business value and qualification dossier | Full quality, cost, latency, success and intervention analysis |

**W7 exit:** evidence-backed assessment of actual harness improvement, including negative findings and limitations.

Use an initial pilot proposal of **60 tasks across six task classes**. Final sample size and numerical acceptance thresholds require an architect-approved frozen benchmark protocol before execution.

---

## W8 — Consumer and product adoption

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W8-01 | Consumer contract qualification | Stable contracts within canonical ownership |
| AHE-W8-02 | Intelligence integration | Authorized consumer wiring and traceable W7 evidence mapping |
| AHE-W8-03 | Scenario Factory integration | Authorized scenario design consumption without duplicate harness |
| AHE-W8-04 | Reference business scenario | End-to-end measured adaptive benefit |
| AHE-W8-05 | Final independent audit | Exact-SHA evidence-backed final acceptance |

**W8 exit:** qualified consumer adoption within existing production admission boundaries.

---

## Roadmap execution rules

The following rules govern all AHE implementation batches:

- Sequential architect-controlled implementation batches.
- Vertical-slice-first delivery.
- Batch-first and risk-based validation.
- Existing ADAS phase IDs and canonical ownership retained.
- No parallel duplicate implementation of mapped Intelligence packages.
- No subsequent wave execution without its explicit authorization.
- No ARCH-FREEZE scope expansion.
- No protected cross-program changes without owner authorization.
- Tenant isolation audit for each executable batch.
- Exact-SHA implementation and qualification evidence.
- Independent acceptance before closure.
- Commit and publication according to current repository governance.

---

## Definition of done

Every executable package must eventually provide:

- Functional implementation evidence.
- Existing owner reuse evidence.
- Exact changed-file inventory.
- Positive and relevant negative tests.
- Adjacent regression results.
- Tenant isolation verdict.
- Governance compliance evidence.
- Commit SHA.
- Remote publication evidence.
- Independent acceptance status.

No package becomes **CLOSED** merely because the implementation agent reports PASS.
