<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# ERL-QUAL-004 — Platform Capability Gap Analysis

**Qualification:** ERL-QUAL-004 (`enterprise_payment_uncertainty_recovery`)  
**Role:** Architecture analysis only — no capability implementation in this task  
**Scope:** `intergrax/` contracts and runtime, plus architecture hubs cited below  
**Scenario observation source:** [`ERL_QUAL_004_EXTERNAL_PAYMENT_BOUNDARY.md`](../../../platform_proofs/scenarios/enterprise_payment_uncertainty_recovery/docs/ERL_QUAL_004_EXTERNAL_PAYMENT_BOUNDARY.md)

**Related hubs:**

| Hub | Role |
| --- | --- |
| [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md) | ERL purpose and boundaries |
| [`UNCERTAINTY_MANAGEMENT.md`](UNCERTAINTY_MANAGEMENT.md) | UNKNOWN lifecycle (target + partial runtime) |
| [`RECONCILIATION.md`](RECONCILIATION.md) | External truth verification |
| [`EXTERNAL_EFFECT_CONTRACTS.md`](EXTERNAL_EFFECT_CONTRACTS.md) | Declared effect safety |
| [`ERL_QUAL_004_SCENARIO_QUALITY_GATE.md`](ERL_QUAL_004_SCENARIO_QUALITY_GATE.md) | Scenario definition gate |

---

## 1. Business problem

Enterprise applications invoke **external systems of record** (payments, inventory, identity, carriers). Communication can end without a definitive answer: timeouts, partial responses, or async acceptance while truth lives elsewhere.

**Business impact:** the organization cannot confidently know whether an external side effect occurred. Treating ambiguity as **error**, **retry**, or **silent failure** causes double effects, orphaned state, and un-auditable decisions.

ERL-QUAL-004 external payment boundary models this honestly: integration channel can return **UNKNOWN** while SoR truth may already be terminal. The scenario therefore surfaces two enterprise needs:

1. **Explicit uncertain external outcome** — a managed business/platform state, not an exception class.
2. **End-to-end explainability** — reconstruct intent → effect → evidence → decision.

This document maps those needs to **what Intergrax already provides** versus **what remains integration or maturity work**.

---

## 2. Current platform capabilities (evidence-based)

### 2.1 Enterprise Reliability Layer (ERL) — contracts and orchestration runtime

| Area | Location | Maturity |
| --- | --- | --- |
| Tri-state external effect outcome | `intergrax/contracts/enterprise_reliability/outcome.py` — `ExternalEffectOutcome` (`success` / `failure` / `unknown`) | **Implemented (contracts)** |
| UNKNOWN lifecycle | `intergrax/contracts/enterprise_reliability/lifecycle.py`, `intergrax/runtime/enterprise_reliability/uncertainty_lifecycle.py` | **Implemented (orchestration)** |
| Reliability case lifecycle | `intergrax/contracts/enterprise_reliability/case_lifecycle.py`, `intergrax/runtime/enterprise_reliability/case_lifecycle_coordination.py` | **Implemented (state machine)** |
| Effect safety declarations | `intergrax/contracts/enterprise_reliability/effect_contract.py` | **Implemented (contracts)** |
| Evidence verdict → outcome | `intergrax/contracts/enterprise_reliability/evidence.py` | **Implemented (contracts)** |
| Reconciliation planning and probe execution | `intergrax/contracts/enterprise_reliability/reconciliation*.py`, `intergrax/runtime/enterprise_reliability/reconciliation_*.py` | **Foundation + unit-tested orchestration** |
| Plugin SPI (domain probes, resolution, compensation, governance) | `intergrax/contracts/enterprise_reliability/plugin_spi.py`, `intergrax/runtime/enterprise_reliability/plugin_gateway.py` | **Implemented (ports)** |
| UNKNOWN vs Reliability failure | `intergrax/contracts/enterprise_reliability/reliability_boundary.py` — `project_external_effect_to_reliability` | **Implemented (contracts)** |
| Recovery handoff to execution runtime | `intergrax/contracts/enterprise_reliability/execution_lifecycle_port.py` | **Port defined; UER adapter not in `intergrax/runtime/nexus/`** |
| Observability semantic facts | `intergrax/contracts/enterprise_reliability/observability.py` | **Contracts only** (no runtime emitter found under `intergrax/`) |

Unit coverage exists under `tests/unit/contracts/test_enterprise_reliability_*` and `tests/unit/runtime/enterprise_reliability/`.

Architecture hubs explicitly state ERL is **target architecture with partial runtime maturity** — consistent with code: rich contracts and orchestration modules, limited automatic wiring into Unified Execution Runtime (UER).

### 2.2 Execution reliability (adjacent, not ERL-complete)

| Mechanism | Location | Relation to UNKNOWN |
| --- | --- | --- |
| `ExecutionFailureKind.UNKNOWN` | `intergrax/contracts/execution_retry.py`, `intergrax/runtime/execution/retry/policy.py` | Retry policy respects `has_unknown_side_effect`; does **not** replace ERL admission or reconciliation |
| Partial recovery | `intergrax/contracts/partial_recovery.py` | Fan-out / recovery semantics; orthogonal to external-effect tri-state |

### 2.3 External work correlation (Tier-2 / integration shape)

`intergrax/contracts/external_work.py` defines `external_task_correlation.v1` (`run_id`, `correlation_id`, `idempotency_key`). This supports **agent/external-task** correlation, not a full enterprise payment chain by itself.

### 2.4 Scenario state (outside platform core)

ERL-QUAL-004 implements payment SoR, `correlation_id`, and `payment_intent_id` in scenario PostgreSQL and `external_payment/` — **not** in `intergrax/`. The boundary doc states Integrax runtime / ERL are **not** owners of the acquirer simulator.

---

## 3. Analysis questions

### 3.1 UNKNOWN state model

**Question:** Does Intergrax have a universal concept for unknown external effect, uncertain execution, unresolved external outcome, pending verification?

| Concept | Platform artifact | Verdict |
| --- | --- | --- |
| Unknown external effect | `ExternalEffectOutcome.UNKNOWN` | **Yes** |
| Managed uncertainty (not raw error) | `UncertaintyLifecyclePhase`, `admit_external_effect_unknown` | **Yes** |
| Unresolved until evidence | `ExternalEffectEvidenceVerdict.INSUFFICIENT` → UNKNOWN via `classify_external_effect_outcome` | **Yes** |
| Separation from Reliability “failure” | `project_external_effect_to_reliability` → `UNCERTAINTY_FAIL_CLOSED` / `ExecutionFailureKind.UNKNOWN` | **Yes** |
| Reliability case entry | `ReliabilityCaseLifecycleState.UNKNOWN_DETECTED` | **Yes** |

**Mapping to scenario vocabulary:**

| Scenario term | Platform term |
| --- | --- |
| UNKNOWN | `ExternalEffectOutcome.UNKNOWN` + uncertainty / case lifecycle |
| KNOWN_SUCCESS | `ExternalEffectOutcome.SUCCESS` (terminal) |
| KNOWN_FAILURE | `ExternalEffectOutcome.FAILURE` (terminal) |

**Existing capability:** **YES** at platform contract and ERL runtime orchestration level.

**Gaps (integration / maturity, not missing domain model):**

| Gap | Nature |
| --- | --- |
| UER / Nexus does not automatically admit UNKNOWN on external calls | **Platform integration** — `ExecutionLifecyclePort` exists; no `enterprise_reliability` imports under `intergrax/runtime/nexus/` |
| No durable platform store for `UncertaintyStateRecord` / `ReliabilityCaseLifecycleRecord` in `intergrax/` | **Platform persistence** — orchestration is in-process; scenario uses its own DB |
| Applications can still map timeout → exception unless they call ERL admission APIs | **Adoption** — capability exists but is not ambient |
| Observability facts (`UncertaintyAdmissionFact`, etc.) are not emitted from runtime | **Platform observability wiring** |

**If the business problem were unserved:** the missing piece would be **(B) platform capability** — and that capability is **already modeled**; ERL-QUAL-004 exposes **wiring and persistence** work, not a new tri-state invention.

**Scenario-specific:** payment capture lifecycle labels (`REQUESTED`, `PROCESSING`, `UNKNOWN` on the integration channel) remain in `platform_proofs/.../external_payment/`.

---

### 3.2 Correlation model

**Question:** Can Intergrax correlate business action → execution → external effect → evidence → decision?

| Link | Mechanism | Verdict |
| --- | --- | --- |
| External effect episode | `correlation_id` + `contract_id` on ERL contracts (uncertainty, reconciliation, evidence, governance) | **Yes (convention)** |
| Reliability case | `case_id`, `correlation_id`, `ReliabilityCaseLifecycleRefs` (`uncertainty_state_ref`, `evidence_ref`, `execution_ref`, …) | **Yes (explicit refs)** |
| Evidence binding | `ExternalEffectEvidenceOperationLink` (`correlation_id`, `contract_id`, `probe_ref`, `attempt_index`) | **Yes** |
| Decision audit | Resolution / governance / recovery contracts carry `correlation_id` | **Yes** |
| Business request → payment intent | Scenario DB + application context keys | **Scenario-owned** |
| Automatic graph across tiers | No single `intergrax/` registry indexing business entities | **No** |

**Existing capability:** **PARTIAL — YES for ERL-internal chain; NO for universal cross-domain identity graph.**

ERL correlation is **intentionally** a stable `correlation_id` plus opaque refs (`evidence_ref`, `execution_ref`). The platform defines **how** artifacts link in a reliability case; the **application or proof harness** must assign the same `correlation_id` to business intent, payment intent, and execution.

Scenario gap note (boundary doc): durable correlation between lab execution references, payment intents, and external effect ids is needed for proof orchestration — classified as **enterprise proof harness / lab runtime** plus **scenario application** propagation, not acquirer simulator logic.

**Classification:** Universal **correlation contract** exists (ERL + refs). Universal **business-entity correlation registry** does **not** exist in `intergrax/` today.

---

### 3.3 Reconciliation foundation

**Ownership:** Platform (`intergrax/contracts/enterprise_reliability/`, `intergrax/runtime/enterprise_reliability/`).

**Lifecycle:** UNKNOWN admission → reconciliation plan → probe execution → evidence materialization → resolution / compensation / recovery → governance → handoff (`case_lifecycle.py` states align with reconciliation orchestration modules).

**Contracts:** `ReconciliationPlan`, `ReconciliationProbeRequest` / `Result`, effect contract `reconciliation_probe_refs`.

**Extensibility:** `EnterpriseReliabilityPluginRegistry` + gateway; payment-specific reads belong in a **plugin** reading scenario SoR or real PSP APIs.

**Can ERL-QUAL-004 use it naturally?** **Yes, via integration path:**

1. Declare an `ExternalEffectContract` for payment capture (financial category, reconciliation supported).
2. Admit UNKNOWN when integration returns uncertain channel outcome.
3. Register a reconciliation plugin that probes `external_sor` truth (scenario adapter).
4. Drive case lifecycle transitions with returned `evidence_ref`.

**Not yet done:** scenario application and proof runner do not call ERL orchestration; external boundary deliberately stops at SoR persistence.

---

### 3.4 Evidence model

**Contracts:** `ExternalEffectEvidence`, verdict types, `materialize_external_effect_evidence_from_probe` (runtime export in `intergrax/runtime/enterprise_reliability/__init__.py`).

**Provenance:** Provider payloads stay behind `evidence_ref`; core models normalized `check_result` / `confidence` / `verdict` (`reconciliation_evidence.py`).

**Auditability:** Case lifecycle requires `evidence_ref` before `RESOLUTION_PENDING`; observability fact schemas exist for reconciliation attempts and resolution decisions.

**Can external payment truth be represented?** **Yes** — as reconciliation probe evidence with domain payload referenced opaquely. Payment semantics (ledger rows, PSP codes) stay in scenario SoR or plugin storage; platform stores **whether** truth is definitive.

**Gap:** emission of observability facts and long-term evidence retention are not fully wired in runtime (contracts ready).

---

## 4. Missing capabilities vs scenario-specific work

| Item | Universal platform? | Status |
| --- | --- | --- |
| Tri-state external effect + UNKNOWN lifecycle | Yes | **Present** |
| Reconciliation orchestration + plugin SPI | Yes | **Foundation present** |
| Reliability case state machine + ref chain | Yes | **Present** |
| Effect contract admission + gating | Yes | **Contracts + runtime helpers** |
| UER pause/resume on UNKNOWN | Yes | **Port only** — adapter gap |
| Durable ERL case / uncertainty store | Yes | **Gap** — no `intergrax/` repository module found |
| Observability spine emission for ERL facts | Yes | **Gap** — contracts only |
| End-to-end auto-correlation of business entities | Yes (lightweight) | **Gap** — correlation_id convention + refs; no central index |
| Payment capture simulator + SoR schema | No | **Scenario** (`external_payment/`, PostgreSQL lab) |
| Commerce order / intent tables | No | **Scenario application** |
| PSP-specific probe implementation | No | **Plugin** (scenario or integration package) |

---

## 5. Ownership decision

| Responsibility | Owner |
| --- | --- |
| Payment domain, orders, intents, commerce workflow | Scenario application (`platform_proofs/.../application/`) |
| Acquirer channel, SoR persistence, dataset variants | Scenario `external_payment/` + PostgreSQL lab |
| UNKNOWN as external-effect truth (not transport error) | **Platform ERL** — admit via `ExternalEffectOutcome` / uncertainty lifecycle |
| Reconciliation orchestration and bounds | **Platform ERL** — plugin executes domain read |
| Evidence normalization and verdict mapping | **Platform ERL** |
| Resolution, compensation, recovery, governance sequencing | **Platform ERL** |
| Execution pause/resume after recovery decision | **UER** via `ExecutionLifecyclePort` adapter |
| Assigning one `correlation_id` across intent, effect, case, run | **Scenario application + proof harness** (must align with ERL contracts) |
| Declaring payment capture effect safety | **Scenario or product integration** (`ExternalEffectContract` instance) |

**Anti-patterns (out of scope):** encoding UNKNOWN only in HTTP status codes; duplicating reconciliation state machines in the scenario app; coupling `intergrax/` to scenario PostgreSQL schemas.

---

## 6. Recommended future roadmap (integration — not new mechanism design)

Ordered for ERL-QUAL-004 without inventing parallel capabilities:

1. **Contract declaration** — publish scenario payment capture as `ExternalEffectContract` with reconciliation probe ref(s) pointing to the scenario plugin id.
2. **Application admission** — when `external_payment` returns uncertain integration outcome, call `admit_external_effect_unknown_with_contract` / open `initial_reliability_case_lifecycle` with shared `correlation_id` (already in scenario persistence).
3. **Reconciliation plugin** — implement `EnterpriseReliabilityPluginRegistry` reconciliation strategy reading scenario SoR; return `ExternalEffectEvidence` + `evidence_ref`.
4. **Case progression** — use existing runtime orchestration (`plan_external_effect_reconciliation`, `execute_external_effect_reconciliation_probe`, resolution / governance modules) in proof or application composition root.
5. **Execution handoff** — wire `ExecutionLifecyclePort` to UER when recovery decisions require pause/resume (after architect defines adapter ownership).
6. **Observability** — emit `UncertaintyAdmissionFact` and reconciliation attempt facts per existing schemas (closes “why did the system decide?” for operators).
7. **Persistence (if required for enterprise durability)** — add platform or shared lab store for case records; until then, scenario DB may hold business rows while ERL refs remain opaque pointers.

Each step **reuses** modules listed in §2.1; roadmap items are **wiring and ownership**, not replacement of ERL.

---

## 7. Validation statement

| Check | Result |
| --- | --- |
| Conclusions tied to `intergrax/contracts/enterprise_reliability/` and `intergrax/runtime/enterprise_reliability/` | **Done** |
| Scenario code referenced only for boundary observations | **Done** |
| No code, contract, or runtime changes in this task | **Done** |
| No speculative new platform mechanisms proposed | **Done** — roadmap references existing ERL surface |

---

## 8. Summary answers (executive)

| # | Topic | Existing capability | Notes |
| --- | --- | --- | --- |
| 1 | UNKNOWN state model | **YES** (platform) | Integration into UER and apps incomplete |
| 2 | Correlation chain | **PARTIAL** | ERL chain yes; business graph is by convention |
| 3 | Reconciliation foundation | **YES** (foundation) | ERL-QUAL-004 should use plugin + contracts |
| 4 | Evidence model | **YES** (foundation) | Payment truth via probe + `evidence_ref` |
| 5 | Ownership | — | See §5 |

**GitHub Audit Required:** The introduced changes must be audited against the actual GitHub repository state before starting the next implementation task.
