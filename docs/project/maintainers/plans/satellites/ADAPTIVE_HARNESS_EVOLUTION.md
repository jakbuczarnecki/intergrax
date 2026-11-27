# Adaptive Harness Evolution — Implementation Roadmap

**Revision:** 1.0
**Status:** ARCHITECT-DEFINED / PENDING INDEPENDENT ACCEPTANCE
**Architecture reference:** [ADAPTIVE_HARNESS_EVOLUTION.md](../../../architecture/satellites/ADAPTIVE_HARNESS_EVOLUTION.md)

Execution register and implementation work breakdown remain in this document.

---

## Program status

| Item | Status |
|---|---|
| **AHE-W0-01** | IN_PROGRESS |
| All remaining packages | PLANNED |
| Closed waves | 0 |

**Cross-program coordination:** Explicit. AHE roadmap dependencies do not themselves authorize changes to other programs.

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
