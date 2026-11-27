# Adaptive Harness Evolution — Implementation Roadmap

**Revision:** 1.1
**Status:** ARCHITECT-DEFINED / PENDING INDEPENDENT ACCEPTANCE
**Architecture reference:** [ADAPTIVE_HARNESS_EVOLUTION.md](../../../architecture/satellites/ADAPTIVE_HARNESS_EVOLUTION.md)

Execution register and implementation work breakdown remain in this document.

---

## Program status

| Item | Status |
|---|---|
| Closed waves | 0 |
| Formally closed packages | 0 |

**Cross-program coordination:** Explicit. AHE roadmap dependencies do not themselves authorize changes to other programs.

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

| Gate | Requirement | Initial status |
|---|---|---|
| AHE-G0 | Architecture and roadmap acceptance | PENDING |
| AHE-G1 | Exact W1 implementation contract, approved ticket and scope authorization | PENDING |
| AHE-G2 | Executable bounded LAB adaptive loop | PLANNED |
| AHE-G3 | Propose–Probe–Compose and selection qualification | PLANNED |
| AHE-G4 | Trusted experience and resource optimization qualification | PLANNED |
| AHE-G5 | Enterprise governed evolution qualification | PLANNED |
| AHE-G6 | Independent behavioral qualification | PLANNED |
| AHE-G7 | Consumer adoption and final independent acceptance | PLANNED |

These are AHE program milestone names, not substitutes for DevOS G0–G6, IIAC, FRZ or existing admission controls.

W1 execution requires AHE-G0, AHE-G1 and all applicable existing governance authorizations.

No roadmap row or gate may be declared accepted solely by the implementer.

## Execution register

| Package | Implementation | Review / acceptance | Notes |
|---|---|---|---|
| AHE-W0-01 | PUBLISHED | Architect review: ACCEPTED_SCOPED; Independent acceptance: PENDING | Commit: `a4ff0130dbaf054bfdaeb8522644c2f8b39d28ac`. Process finding: AI-tool attribution in commit metadata. Historical commit shall remain unchanged. |
| AHE-W0-02 | IN_PROGRESS | Architect-defined ownership and reuse mapping: supplied; Independent acceptance: PENDING; External owner acceptance: PENDING | — |
| AHE-W0-03 | PLANNED | Await architect-issued implementation contract | — |
| W1–W8 | PLANNED | No executable wave authorization inferred | — |

---

## W0 — Architecture and canonical baseline

| ID | Deliverable | Acceptance |
|---|---|---|
| AHE-W0-01 | Architecture and roadmap | Both architect-defined documents published; independent review pending |
| AHE-W0-02 | Canonical ownership and baseline reconciliation | Exact-SHA reuse mapping and cross-program boundary acceptance |
| AHE-W0-03 | W1 implementation contract | Concrete authorized write set, dependencies, tests and acceptance criteria |

**W0 exit:** accepted architecture, roadmap and implementation boundaries.

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
