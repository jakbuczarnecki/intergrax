---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: postgresql_data_model_architecture
lifecycle: IMPLEMENTATION_PREPARED
status: ARCHITECTURE_DOCUMENTED
---

# ERL-QUAL-004 — PostgreSQL Data Model Architecture

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md) · [Data Provisioning Architecture](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md) · [PostgreSQL Infrastructure](ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md)

---

## 1. Purpose

The future PostgreSQL data model describes **operational state** for an enterprise payment workflow in the ERL-QUAL-004 lab: a buyer organization places a high-value order, the commerce application initiates capture with an external payment system, integration uncertainty leaves the application without a definitive outcome, and reconciliation must close the gap between what the business believes and what actually happened externally.

The model is architected around a non-negotiable separation:

| Plane | Meaning |
| --- | --- |
| **External Reality** | Authoritative outcome in the payment system of record (SoR) and related external artifacts—what **did** happen, independent of whether the commerce application ever received a confirmation. |
| **Application Knowledge** | What the commerce application (and its persisted domain state) **currently holds** as truth—including explicit **UNKNOWN** when terminal payment outcome is not yet knowable from integration signals alone. |

PostgreSQL in this scenario is where those planes are **materialized for the lab** so provisioning, the commerce application, the external SoR simulator, and evaluator alignment can exercise realistic reads and writes. The model does **not** collapse “no confirmation yet” into “failed” or “succeeded.”

Logical business truth remains defined in the vendor-neutral [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md). This document defines **how that truth maps to a future relational shape** in the scenario PostgreSQL instance—not Integrax platform storage.

This document is **architecture only**. It does not create SQL, migrations, ORM models, or seed scripts.

---

## 2. Architecture Model

PostgreSQL serves as a **system-of-record laboratory environment** for the standalone E2E proof: transactional commerce state, external payment SoR truth (simulator-backed), and optionally mirrored or referenced reliability-case facts that the application persists for audit—not as a substitute for the production Integrax observability spine.

```text
Vendor-neutral dataset (qualification truth)
        ↓
Provisioning (future PostgreSQL adapter)
        ↓
PostgreSQL lab instance (infrastructure/)
        ├── Commerce application zone (orders, intents, application knowledge)
        ├── External SoR zone (external effects, external reality)
        └── Reconciliation / investigation zone (cases, attempts, evidence refs)
        ↓
Application contracts (ports) — no raw dataset access at runtime
        ↓
Scenario runtime + ERL reliability processing (observability spine)
```

| Role | PostgreSQL in ERL-QUAL-004 |
| --- | --- |
| **Is** | Lab SoR for commerce and simulated payment ledger; durable anchor for variant A/B/C materialization; enterprise-realistic relational patterns (orders, monetary amounts, lifecycle timestamps, correlation IDs). |
| **Is not** | Integrax platform database, checkpoint store, agent memory, vector index, or global Intergrax tenant data. |
| **Is not** | Production merchant catalog, PCI cardholder vault, or a specific PSP’s proprietary schema. |

Changing PostgreSQL to another lab technology must not change variant **meaning**—only **materialization** ([Scenario Data Architecture § 2](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#2-data-architecture-overview)).

---

## 3. Core Business Entities

Entities below are **conceptual**. Future implementers may group them into schemas, tables, or views; this architecture prescribes **semantics and ownership**, not DDL.

### 3.1 Organization / Customer

Represents the **enterprise buyer** obligated on the purchase—not a synthetic demo account.

| Concept | Description |
| --- | --- |
| Organization identity | Stable business key (e.g. legal entity reference used in B2B commerce). |
| Display / legal name | Human-recognizable company name for operator narrative and audit. |
| Business context | Segment, payment terms class, or account tier relevant to high-value hold policy (non-PCI). |
| Account relationship | Link to commercial account or contract under which orders are placed (buyer–seller relationship). |

**Relationships:** One organization may have many orders over time; the scenario focuses on **one** qualifying order per run.

### 3.2 Order

Represents the **commerce transaction** under test—the business commitment whose fulfillment must not proceed on guessed payment truth.

| Concept | Description |
| --- | --- |
| Order identity | Business document number (purchase order / sales order style). |
| Customer reference | Surrogate link to organization / account (no cardholder data). |
| Amount | Line-total or order-total monetary value (decimal semantics). |
| Currency | ISO 4217 currency code. |
| Business status | Commerce lifecycle (e.g. `AWAITING_PAYMENT_CONFIRMATION`, `PAYMENT_CONFIRMED`, `CANCELLED`)—**application-facing**, may lag External Reality. |
| Lifecycle timestamps | `created_at`, status transition times, optional `fulfillment_eligible_at` after governed resolution. |

**Relationships:** Exactly one primary order per scenario run; one or more payment intents may reference the order; inventory reservation correlates by order identity ([Scenario Data Architecture § 4](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#4-business-data-model)).

### 3.3 Payment Intent

Represents the **requested payment operation** initiated by the application—intent to capture funds, distinct from SoR ledger truth.

| Concept | Description |
| --- | --- |
| Payment intent identity | Business reference for the capture attempt (traceable in support and reconciliation). |
| Related order | Binding to order identity. |
| Amount / currency | Must align with order capture scope for the scenario (full order capture). |
| Initiation time | When the application submitted the operation to the external boundary. |
| Correlation identifiers | Idempotency key, client request ID, or end-to-end trace ID shared with external effect and reconciliation queries. |
| Attempt ordinal | Supports narrative of single canonical capture (qualification invariant: no duplicate capture). |

**Relationships:** One external payment effect record per intent on the canonical path; application knowledge rows reference the same correlation set.

### 3.4 External Payment Effect

Represents the **side effect outside full application control**—the operation as observed at the integration boundary immediately after invoke.

| Concept | Description |
| --- | --- |
| External effect identity | Stable ID for the capture operation in external/integration terms. |
| Correlation identifier | Same family as payment intent (reconciliation lookup key). |
| Requested state | What the application asked for (e.g. capture for order total). |
| Observed state (integration)** | Immediate wire-level outcome: on canonical entry, **indeterminate / unknown**—not a substitute for SoR truth. |

**Relationships:** Links to payment intent and order; pairs with communication-event narrative (timeout, partial response) in logical dataset—not duplicated as vendor webhook payloads in this model.

### 3.5 External Reality

Represents **what actually happened** in the payment SoR (simulator in the lab)—authoritative for evaluator and reconciliation adapter, **not** writable by the commerce application on the canonical qualification path.

| Concept | Description |
| --- | --- |
| Reality record identity | Surrogate for SoR truth row or ledger entry. |
| Correlation to effect | Same correlation family as payment intent / external effect. |
| Terminal outcome | `PAYMENT_COMPLETED`, `PAYMENT_FAILED`, or `TRUTH_INDETERMINATE` (Variant C). |
| Funds captured flag | Business boolean aligned with outcome (e.g. true only when completed). |
| Observed at | When SoR truth became fixed (may be before application knew). |

**Examples (semantic, not application state):**

- Payment completed — funds captured in SoR.
- Payment failed — authorization/capture did not succeed in SoR.
- Truth unavailable — SoR cannot return authoritative answer within policy (Variant C).

### 3.6 Application Knowledge

Represents **what the application persists** as its current belief about payment and order alignment.

| Concept | Description |
| --- | --- |
| Knowledge snapshot / state row | Versioned or current row per order–intent pair. |
| Payment outcome knowledge | `CONFIRMED`, `FAILED`, or **`UNKNOWN`**. |
| Order payment substate | e.g. pending payment truth, confirmed paid, failed payment path. |
| Integration facts | Confirmation received or not; last envelope observed (abstract). |
| Inventory alignment knowledge | Reservation held pending truth vs released—domain dependency on payment knowledge. |

**Examples (application belief):**

- `UNKNOWN` — terminal payment outcome not knowable from integration yet.
- `CONFIRMED` — application accepted paid truth (typically after reconciliation).
- `FAILED` — application accepted failed truth (typically after reconciliation).

Application Knowledge may **intentionally lag** External Reality until reconciliation or governed escalation ([Scenario Data Architecture § 5](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#5-external-reality-vs-application-knowledge)).

### 3.7 Supporting entities (commerce context)

Aligned with the logical dataset, the relational materialization also carries:

| Entity | Role |
| --- | --- |
| **Inventory reservation** | Hold tied to order; fulfillment eligibility expectations per variant. |
| **Communication event** | Integration-layer reason uncertainty arose (not SoR outcome). |

These support realistic enterprise coupling: high-value orders stay on fulfillment hold while payment knowledge is `UNKNOWN`.

---

## 4. UNKNOWN Representation

**UNKNOWN is not failure.** In this architecture, `UNKNOWN` means: *the system cannot determine external truth yet*—not that the payment failed, not that it succeeded, and not that reconciliation has finished.

| Mistake | Correct model stance |
| --- | --- |
| Map timeout to `FAILED` in application knowledge | Preserve `UNKNOWN` until reconciliation or policy-bound escalation. |
| Map silence to `CONFIRMED` | Separate Application Knowledge from External Reality. |
| Store SoR outcome only in application tables | External Reality lives in SoR zone; application row shows `UNKNOWN` while lagging. |

**Preservation rules:**

1. **External Reality** holds the variant-authoritative terminal truth (`completed`, `failed`, or indeterminate/unavailable for C)—owned by provisioning + SoR simulator, read through reconciliation contracts.
2. **Application Knowledge** holds `UNKNOWN` at scenario entry when `immediate_integration_outcome` is unknown, even if External Reality is already `PAYMENT_COMPLETED` (Variant A) or `PAYMENT_FAILED` (Variant B).
3. **Transitions** from `UNKNOWN` to `CONFIRMED` or `FAILED` are reconciliation outcomes recorded in application knowledge (and reliability processing on the spine)—not silent overwrites of External Reality.
4. **Variant C** may leave Application Knowledge at `UNKNOWN` and External Reality at `TRUTH_INDETERMINATE` with reconciliation **unavailable**—terminal containment, not a forced guess.

```text
  External Reality (SoR)          Application Knowledge
  ─────────────────────          ─────────────────────
  may be COMPLETED                 UNKNOWN  ← entry
  may be FAILED                    UNKNOWN  ← entry
  may be INDETERMINATE             UNKNOWN  ← may persist
         │                                  │
         └──── reconciliation ────────────┘
                    (when available)
```

Integrax **reliability processing** classifies and journals UNKNOWN on the observability spine; PostgreSQL materializes **domain** application state the commerce app owns—not a replacement for ERL journal semantics ([Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md)).

---

## 5. Reconciliation Model

Reconciliation closes the gap between External Reality and Application Knowledge. The data model includes **persistent artifacts** for audit and operator narrative; it does **not** define workflow engines, retry policies, or ERL capability code.

| Entity | Purpose |
| --- | --- |
| **Reconciliation case** | Opened when payment knowledge is `UNKNOWN` or drift is suspected; links order, payment intent, correlation IDs, variant context; carries **resolution state** (`OPEN`, `RESOLVED`, `ESCALATED`, `CLOSED_UNRESOLVED`). |
| **Investigation attempt** | Each query or poll against SoR/reconciliation port: attempt number, requested at, outcome (`SUCCESS`, `FAILED`, `UNAVAILABLE`), summary code (abstract). |
| **Evidence reference** | Pointer to auditable artifacts (journal event IDs on spine, hash of SoR response envelope, operator attachment ID)—not raw PAN or PSP secrets. |
| **Resolution record** | Declared alignment action: e.g. `KNOWLEDGE_ALIGNED_TO_SUCCEEDED`, `KNOWLEDGE_ALIGNED_TO_FAILED`, `ESCALATED_HUMAN`, `CONTAINED_NO_TRUTH`. |

**Relationships:**

- One reconciliation case per qualifying uncertainty episode per order (canonical path).
- Many investigation attempts per case (Variants A/B: eventual success; C: failures or unavailability).
- Resolution state drives whether application knowledge may transition from `UNKNOWN`—workflow logic remains outside this document.

Reconciliation **availability** is a variant parameter ([Scenario Data Architecture § 6](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#6-scenario-variant-data-slices)): available for A/B; degraded or exhausted for C.

---

## 6. Temporal Consistency

Enterprise payment flows depend on **ordered, meaningful timestamps**. The model uses UTC-style instants with clear semantics (implementation may use `timestamptz` later).

| Timestamp | Typical anchor |
| --- | --- |
| `created_at` | Entity row first persisted (order, intent, case). |
| `requested_at` | Payment intent submitted to external boundary. |
| `processed_at` | SoR claims processing completed (External Reality)—may precede application awareness. |
| `observed_at` | When integration or reconciliation **observed** a fact (wire response, SoR query result). |
| `resolved_at` | Reconciliation case or application knowledge reached terminal alignment or escalation. |

**Ordering narrative (Variant A example):**

1. Order `created_at` — buyer submits PO.
2. Payment intent `requested_at` — capture submitted.
3. External Reality `processed_at` — SoR captures funds (truth fixed externally).
4. Communication failure — no confirmation; application knowledge remains `UNKNOWN` with `observed_at` on integration attempt only.
5. Investigation attempt `observed_at` — reconciliation returns succeeded.
6. Application knowledge update + case `resolved_at` — knowledge becomes `CONFIRMED`.

**Invariant:** `processed_at` in External Reality may be **earlier** than application `CONFIRMED` state—proving lag is a first-class scenario condition. Timestamps must not be used to infer that UNKNOWN equals failure.

---

## 7. Realistic Data Examples

Examples illustrate **production-like** lab materialization. Provisioning maps from logical dataset IDs (e.g. `erl-qual-004-ord-0001`) to these business-facing values; qualification meaning is unchanged.

### Organization

| Field | Example value |
| --- | --- |
| Organization key | `ORG-NIC-PL-004872` |
| Legal name | Nordic Industrial Components Sp. z o.o. |
| Account | `ACCT-EU-B2B-88421` (annual net-30, high-value approval tier) |

### Order

| Field | Example value |
| --- | --- |
| Order number | `PO-2026-004872` |
| Amount | `12 500,00` (stored as decimal `12500.00`) |
| Currency | `EUR` |
| Business status (at entry) | `AWAITING_PAYMENT_CONFIRMATION` |
| `created_at` | `2026-09-12T08:14:22Z` |
| Context | Spare-parts replenishment for Göteborg plant line 3; fulfillment hold until payment truth |

### Payment intent

| Field | Example value |
| --- | --- |
| Intent reference | `PAY-20260912-8F31A` |
| Idempotency key | `idem-capture-po-2026-004872-v1` |
| Amount / currency | `12500.00` / `EUR` |
| `requested_at` | `2026-09-12T08:15:03Z` |

### External payment effect (integration immediate)

| Field | Example value |
| --- | --- |
| Effect reference | `EXT-CAP-20260912-8F31A` |
| Observed integration state | `INDETERMINATE` (unknown on wire) |
| `observed_at` | `2026-09-12T08:15:41Z` (timeout / incomplete response) |

### External Reality (SoR — Variant A)

| Field | Example value |
| --- | --- |
| SoR transaction ref | `SOR-TXN-7C2E91B4F0` |
| Terminal outcome | `PAYMENT_COMPLETED` |
| Funds captured | `true` |
| `processed_at` | `2026-09-12T08:15:18Z` (before integration timeout) |

### Application Knowledge (at UNKNOWN entry)

| Field | Example value |
| --- | --- |
| Payment outcome knowledge | `UNKNOWN` |
| Order payment substate | `PENDING_PAYMENT_TRUTH` |
| Inventory reservation | `HELD` (SKU bundle `NIC-VALVE-4402`, qty 50) |
| Confirmation received | `false` |

### Reconciliation case (after UNKNOWN admission)

| Field | Example value |
| --- | --- |
| Case reference | `RECON-20260912-004872` |
| Resolution state (mid-flight) | `OPEN` |
| Investigation attempt 2 | `observed_at` `2026-09-12T08:22:09Z`, outcome `SUCCESS`, SoR matches `PAYMENT_COMPLETED` |

These examples avoid toy names (`customer_1`, `order_123`, `test_payment`) and mirror how operations teams discuss incidents: PO numbers, capture references, SoR transaction IDs, and reconciliation case numbers.

---

## 8. Scenario Variants Mapping

All variants share the **same structural entities**: organization, order, payment intent, external effect, inventory reservation, communication precondition, and application knowledge starting at **`UNKNOWN`**. Variants differ in **External Reality** and **reconciliation availability** slices ([Scenario Data Architecture § 6](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#6-scenario-variant-data-slices)).

| Variant | External Reality (SoR zone) | Reconciliation | Application Knowledge terminus (expected) | Entities most affected |
| --- | --- | --- | --- | --- |
| **A** — payment completed after unknown | `PAYMENT_COMPLETED`, funds captured | Available; discoverable `payment_succeeded` | `CONFIRMED` after resolution | External Reality row; investigation attempts with success; resolution → aligned to succeeded; order status → paid path |
| **B** — payment failed after unknown | `PAYMENT_FAILED`, funds not captured | Available; discoverable `payment_failed` | `FAILED` after resolution | External Reality row; resolution → aligned to failed; order → failed/cancel path; inventory release knowledge |
| **C** — truth unavailable | `TRUTH_INDETERMINATE` or SoR unreachable | Unavailable or exhausted within policy | `UNKNOWN` persists; case `CLOSED_UNRESOLVED` or `ESCALATED` | Investigation attempts with `UNAVAILABLE`; no forced alignment row; governance escalation evidence refs |

**Unchanged across variants:** Order identity, amounts, currency, payment intent correlation, external effect immediate indeterminate outcome, entry application knowledge `UNKNOWN`.

**Changed per variant:** External Reality terminal fields, reconciliation case attempt outcomes, final resolution state, terminal order/inventory application knowledge (except C, where risky alignment rows must not appear).

Logical variant files: `dataset/variants/payment_completed_after_unknown`, `payment_failed_after_unknown`, `payment_truth_unavailable`.

---

## 9. Ownership Model

| Owner | PostgreSQL data responsibility |
| --- | --- |
| **Business application (commerce)** | Orders, payment intents, application knowledge, inventory reservation state, commerce status transitions driven by workflow ports. **Does not** authoritatively write External Reality on the canonical path. |
| **External reality simulator (SoR zone)** | External payment effects’ SoR truth, `processed_at`, terminal outcomes—seeded and updated only via provisioning / simulator rules for the selected variant. |
| **Provisioning layer** | Maps [vendor-neutral dataset](../dataset/manifest.json) → relational rows; selects variant slice; teardown between runs. |
| **Integrax platform** | Reliability processing facts on the **observability spine** (UNKNOWN classification, reconciliation journaling, governance). PostgreSQL may hold **application-audit mirrors** referenced by evidence; it is not the platform SoR. |
| **Proof harness** | Variant selection, evaluator comparison of materialized External Reality vs projected evidence—the harness does not replace commerce or SoR ownership. |

Consistent with [Scenario Data Architecture § 8](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#8-ownership-model) and [Data Provisioning Architecture § 3](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md#3-responsibilities).

---

## 10. Non-Goals

This document explicitly does **not** define:

| Excluded | Reason |
| --- | --- |
| SQL schema, tables, columns, constraints | Future implementation task after architecture acceptance. |
| Indexes, partitioning, replication | Production database design out of scope for qualification lab. |
| Migrations or ORM models | Code artifacts forbidden in this task. |
| Deployment, backup, HA | Covered only at infrastructure boundary ([PostgreSQL Infrastructure](ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md)). |
| Payment provider integration | Adapter-specific mapping behind contracts. |
| Reconciliation / ERL workflow logic | Platform capabilities; only data shapes for cases and attempts are described here. |
| Integrax platform storage | Tier-0/1 persistence unrelated to scenario lab PostgreSQL. |

---

## Architectural alignment

| Principle | How this design complies |
| --- | --- |
| Enterprise realism | B2B organization, PO-style IDs, EUR amounts, SoR and reconciliation references (§ 7). |
| Modularity | Commerce, SoR, and reconciliation zones separable; provisioning maps logical dataset (§ 2). |
| Future provisioning | Entity set matches provisioning phases in [Data Provisioning Architecture](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md). |
| Vendor neutrality | No PSP-specific columns or webhook clones; correlation IDs are abstract (§ 3). |
| Explicit ownership | Application vs simulator vs Integrax vs harness (§ 9). |
| Truth vs knowledge | UNKNOWN semantics and dual planes (§ 4). |

---

## References

| Document | Role |
| --- | --- |
| [ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md) | Logical entities, variant slices, reality vs knowledge. |
| [ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md) | Materialization lifecycle and adapter boundary. |
| [ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md](ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md) | Docker PostgreSQL lab instance (empty until provisioned). |
| [ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md](ERL_QUAL_004_VENDOR_INFRASTRUCTURE_ARCHITECTURE.md) | Why PostgreSQL in E2E lab. |
| [ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) | Proof flow, evidence, ownership. |
| [Scenario Specification](../SCENARIO_SPEC.md) | Normative variants A/B/C and PASS/FAIL. |

---

## Document Validation (quality gate)

| Check | Result |
| --- | --- |
| Separates External Reality and Application Knowledge; UNKNOWN not failure | **Pass** |
| No toy/synthetic generator-style examples (§ 7) | **Pass** |
| No SQL, migrations, ORM, code, or fixtures | **Pass** |
| Consistent with scenario dataset and variant JSON semantics | **Pass** |
| PostgreSQL positioned as lab SoR, not Integrax platform storage | **Pass** |
| Variants A/B/C mapped to differing entities (§ 8) | **Pass** |
| Scope limited to scenario `docs/` architecture | **Pass** |

---

**Architecture status:** PostgreSQL data model architecture documented for future DDL and provisioning adapter work. Schema implementation remains **not started**.
