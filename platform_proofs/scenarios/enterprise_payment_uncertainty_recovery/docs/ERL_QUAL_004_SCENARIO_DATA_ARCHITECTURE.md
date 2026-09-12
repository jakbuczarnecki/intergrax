---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: scenario_data_architecture
lifecycle: IMPLEMENTATION_INITIALIZED
status: ARCHITECTURE_DOCUMENTED
---

# ERL-QUAL-004 — Scenario Data Architecture

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Scenario Specification](../SCENARIO_SPEC.md) · [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md)

---

## 1. Purpose

Platform Proof scenarios must separate **what the business world is** (controlled qualification inputs) from **how an application and platform observe that world** (knowledge, traces, reliability cases). Payment uncertainty is inherently a **data and truth-boundary** problem: the same checkout can be “paid” in an external ledger while the integration layer has no confirmation, or reconciliation can be temporarily unreachable while funds may already have moved.

An intermediate **scenario dataset architecture** is required because:

- Variants A, B, and C differ only in **authoritative external truth** and **reconciliation availability**, not in commerce workflow shape—those truths must be expressed once, vendor-neutrally, and provisioned consistently into lab integrations.
- The proof evaluator must compare **application and platform observability** against **fixture-controlled reality** without leaking hidden truth into model-visible prompts on the canonical path.
- Future implementations may load the same logical dataset into relational stores, document stores, event backbones, or in-memory simulators—the qualification contract must not fork per storage technology.

**Key statement:** The scenario dataset represents **business reality and uncertainty conditions**, not a specific vendor or storage technology.

This document defines **data architecture only**. It does not create datasets, loaders, adapters, schemas, or runtime code.

---

## 2. Data Architecture Overview

The scenario data plane is a stable source of qualification truth that flows through provisioning into whatever technology backs the lab application and external simulators. Runtime and ERL consume **application knowledge** (requests, responses, reconciliation envelopes, domain state); they do not own authoritative payment ledger truth.

```text
Scenario Dataset
        ↓
Data Provisioning Layer
        ↓
Target Data Sources
        ↓
Scenario Runtime
```

| Layer | Role |
| --- | --- |
| **Scenario Dataset** | Vendor-neutral description of orders, external payment effects, communication uncertainty, inventory context, and variant-specific external reality. **Stable source of scenario truth** for qualification. |
| **Data Provisioning Layer** | Future mechanism that maps logical dataset entities into concrete stores or simulator seeds (not implemented in this task). |
| **Target Data Sources** | Lab-facing persistence or simulators (e.g. commerce DB, provider SoR simulator, inventory backend) that expose truth only through **application contracts**. |
| **Scenario Runtime** | Integrax execution + ERL processing application workflow; records UNKNOWN, reconciliation, evidence, resolution, governance on the observability spine. |

The dataset sits **above** any single database or API mock: changing PostgreSQL to an in-memory fixture must not change the **meaning** of Variant A/B/C, only how truth is **materialized**.

---

## 3. Vendor Neutrality Principles

### The dataset MUST NOT contain

- Database-specific fields (primary keys as storage artifacts, ORM types, index hints, migration versions).
- Vendor API structures (proprietary PSP JSON shapes, webhook payload clones, SDK error enums).
- Provider-specific objects (named gateway products, merchant account schemas tied to one PSP).
- Infrastructure assumptions (cloud region, queue names, connection strings, cache TTL as qualification requirements).

### The dataset MUST contain

- **Business entities** — order, customer context, monetary amount and currency, inventory reservation correlation.
- **External effect facts** — payment capture attempt identity, operation type, correlation to order, immediate integration outcome (unknown on the wire).
- **Communication events** — why confirmation was lost, delayed, or ambiguous (timeout, partial response, partition) without naming a vendor protocol.
- **External reality** — authoritative outcome in the system of record (succeeded, failed, or indeterminate for Variant C) used by evaluator and simulator, not by unconstrained model prompts.
- **Expected outcomes** — terminal business expectations and evidence themes per variant (aligned with [Proof Architecture Design § 4](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#4-scenario-variants) and [Scenario Specification § B](SCENARIO_SPEC.md#variant-a--successful-reconciliation-payment-confirmed)).

Neutrality preserves **contract-first** qualification: the application depends on capability ports; provisioning supplies data that any conforming adapter could have produced.

---

## 4. Business Data Model

Conceptual entities only—**no database schemas, no serialization format mandated here**.

### Order

Represents the commerce commitment under test.

| Concept | Description |
| --- | --- |
| Order identity | Stable business identifier for the checkout instance in the scenario. |
| Customer context | Non-PCI surrogate (e.g. customer reference) for narrative and correlation—not cardholder data. |
| Amount | Monetary value of the high-value purchase. |
| Currency | ISO-style currency code as business fact. |
| Lifecycle intent | Expected path (fulfillment hold until payment truth) as scenario input, not platform state. |

### Payment Effect

Represents the **external business operation** initiated by the application, distinct from ledger truth.

| Concept | Description |
| --- | --- |
| External effect identity | Correlates capture attempt with order and reconciliation queries. |
| Business operation type | Capture / settlement-class operation (abstract, not PSP method name). |
| Relation to order | Binding between order identity and payment effect identity. |
| Immediate integration outcome | After invoke: **unknown** on the canonical qualification entry (no definitive success/failure signal to the application). |

### Inventory Context

Represents domain dependency tied to payment truth (per [Scenario Specification](SCENARIO_SPEC.md) actors).

| Concept | Description |
| --- | --- |
| Reservation identity | Links inventory hold to order. |
| Reservation state (input) | Initial hold consistent with “pending payment truth.” |
| Fulfillment eligibility (expected) | Variant-dependent expectation after resolution—not a platform instruction. |

### Communication Event

Represents **integration-layer** conditions that create uncertainty—not the payment outcome itself.

| Concept | Description |
| --- | --- |
| Uncertainty cause | e.g. timeout after request accepted, truncated response, gateway partition. |
| Lost response | Definitive provider confirmation not delivered to Integrax. |
| Delayed confirmation | Optional narrative dimension; reconciliation may still discover truth later (Variants A/B). |

### External Reality

Authoritative **system-of-record** truth for qualification (fixture-controlled, evaluator-visible).

| Concept | Description |
| --- | --- |
| Actual provider outcome | Whether funds were captured, attempt failed, or truth cannot be established in policy bounds. |
| Final business truth | Single coherent outcome used to judge PASS/FAIL invariants (no duplicate capture, no ship-without-capture, Variant C containment). |
| Reconciliation availability | Whether SoR query can return truth (Variant C: unavailable or exhausted within policy). |

### Application Knowledge (conceptual snapshot)

Not a duplicate of External Reality—records what the **application and platform legitimately know** at material decision points.

| Concept | Description |
| --- | --- |
| Observed invoke envelope | Request sent; response missing or ambiguous. |
| Classified platform state | UNKNOWN admission (future runtime artifact, not dataset vendor field). |
| Reconciliation observations | Results or failures as returned through contract envelopes. |
| Domain flags | Order “pending payment,” inventory held—**may lag** External Reality until reconciliation. |

### Expected Qualification Outcomes

Per variant, logical expectations for evaluator cross-check (see § 6).

| Concept | Description |
| --- | --- |
| Terminal business result | RESOLVED (A/B) or UNRESOLVED (C). |
| Evidence themes | UNKNOWN, reconciliation, resolution/compensation, governance as in [Proof Architecture § 8](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#8-evidence-model). |
| Invariant checklist | No duplicate capture; no fulfillment without capture; Variant C escalation. |

---

## 5. Reality vs Knowledge Model

This scenario exists to demonstrate the **gap** between external authority and application/platform knowledge.

### External Reality

Facts that would be true in the payment provider’s system of record and inventory backend if an operator could inspect them directly.

**Example:** Payment completed—funds captured; ledger shows success even though the integration never delivered confirmation.

External Reality is **owned by the scenario dataset** (for qualification) and **materialized** behind application-facing contracts. It is the evaluator’s independent ground truth (see [Scenario Specification § A — Independence](SCENARIO_SPEC.md#conditional-authoring-prompts-_complete-when-relevant_)).

### Application Knowledge

Facts available to the commerce application and Integrax through normal execution: tool calls, external-effect responses, reconciliation plugin results, domain state updates, ERL journal events.

**Example:** No confirmation received—payment effect classified UNKNOWN; order remains pending payment; inventory reservation still held under policy.

Application Knowledge is **owned by the application workflow** during the run and **recorded** on the production observability spine. It may be **incomplete or stale** relative to External Reality until reconciliation succeeds—or indefinitely in Variant C.

### Purpose of the gap

| Dimension | External Reality | Application Knowledge |
| --- | --- | --- |
| Authority | Provider SoR / controlled simulator | Application + platform observations |
| At UNKNOWN entry | May already be “paid” or “failed” | Explicitly **does not know** terminal outcome |
| Risk | Wrong downstream action if knowledge is treated as truth | Blind retry or optimistic ship |
| Scenario proof | Evaluator compares both | Platform must **not** collapse UNKNOWN without path |

The qualification narrative is: Integrax **contains** the gap (UNKNOWN, reconciliation, governance) rather than **pretending** the gap does not exist.

```text
  External Reality          Application Knowledge
  (dataset / SoR)         (runtime + domain state)
        │                           │
        │    ← reconciliation →     │
        │         (when available)  │
        └───────── gap ─────────────┘
              UNKNOWN state
```

---

## 6. Scenario Variants Data Model

All variants share the same **structural** dataset: one order, one payment effect, inventory context, and a communication event that yields **immediate unknown** integration outcome. Variants differ in **External Reality** and **reconciliation availability** slices.

### Variant A — Payment completed after unknown

| Aspect | Data content |
| --- | --- |
| **Input conditions** | Communication event: lost/delayed confirmation. External Reality: **capture succeeded** in SoR. Reconciliation: **available**; returns confirmed success. |
| **Expected platform behavior** | UNKNOWN → reconciliation → resolution to continue → governance as needed → lifecycle handoff; **no** ungoverned second capture. |
| **Expected evidence** | UNKNOWN admitted; reconciliation completed with success; single capture confirmed; continuation without duplicate charge; order/inventory alignment with paid truth. |

### Variant B — Payment failed after unknown

| Aspect | Data content |
| --- | --- |
| **Input conditions** | Same unknown entry. External Reality: **capture failed** or not present in SoR. Reconciliation: **available**; returns failure. |
| **Expected platform behavior** | UNKNOWN → reconciliation → resolution to compensate / controlled recovery → consistent terminal domain state. |
| **Expected evidence** | Failed payment resolution; compensation or equivalent recovery recorded; no paid/shipped inconsistency; reservations released per policy. |

### Variant C — Truth unavailable

| Aspect | Data content |
| --- | --- |
| **Input conditions** | Same unknown entry. External Reality may be succeeded, failed, or in-flight—**not authoritatively discoverable** within policy. Reconciliation: **unavailable** or exhausted (endpoint down, policy window exceeded). |
| **Expected platform behavior** | UNKNOWN → reconciliation failure/unavailability → governance → human escalation; risky automation contained. |
| **Expected evidence** | Reconciliation exhaustion or unavailability; governance deny/escalate; operator-visible rationale; **no** ungoverned capture retry or fulfillment; terminal **UNRESOLVED** or governed containment path. |

Variant outcome summary (aligned with [Scenario Specification § B](SCENARIO_SPEC.md#outcomes)):

| Outcome | Variants |
| --- | --- |
| **RESOLVED** | A, B |
| **UNRESOLVED** | C (authoritative truth not established in bounds) |

---

## 7. Data Provisioning Architecture

Provisioning is the **future** bridge from the logical Scenario Dataset to lab **Target Data Sources**. This section is conceptual only—**no loaders or implementations** in this task.

```text
┌─────────────────────┐
│  Scenario Dataset   │  logical entities (§ 4)
│  (vendor-neutral)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Provisioning   │  validation, mapping, idempotent seed rules
│ Layer (future)      │
└──────────┬──────────┘
           │
     ┌─────┴─────┬─────────────┬──────────────┐
     ▼           ▼             ▼              ▼
 Relational   Document     Event bus    External adapter
  storage      storage      (facts)      seed / simulator
     │           │             │              │
     └───────────┴─────────────┴──────────────┘
                           │
                           ▼
              Application contracts only
                           │
                           ▼
                   Scenario Runtime
```

| Target style | Conceptual use |
| --- | --- |
| **Relational storage** | Order and inventory rows for commerce application reads/writes. |
| **Document storage** | Flexible scenario bundles or operator narrative attachments. |
| **Event systems** | Stream of communication events or simulated provider notifications (still behind contracts). |
| **External adapters** | Payment/inventory simulators implementing the same ports as production adapters; seeded from dataset External Reality. |

Rules:

- Provisioning **must not** embed vendor API shapes in the canonical dataset—only in adapter-specific mapping layers (future).
- The same variant identifier (A/B/C) must produce the same **logical** truth regardless of target technology.
- Proof configuration selects variant; provisioning materializes truth for simulators **without** exposing hidden truth to model prompts outside controlled application interfaces ([Scenario Specification § A — Hidden truth](SCENARIO_SPEC.md#conditional-authoring-prompts-_complete-when-relevant_)).

---

## 8. Ownership Model

| Owner | Data responsibility |
| --- | --- |
| **Scenario Dataset** | Proof input; **controlled business reality** (External Reality, communication preconditions, variant parameters); expected qualification outcomes for evaluator alignment. |
| **Application** | **Business workflow** data during execution: order lifecycle decisions, domain state transitions, correlation IDs, policy choices (hold, continue, compensate, escalate). Does not fabricate SoR truth on the canonical path. |
| **Integrax Platform** | **Reliability processing** data: UNKNOWN classification, reconciliation attempts, evidence/journal facts, resolution and governance decisions, recovery lifecycle transitions—on the observability spine. |
| **Proof harness** | Variant selection, provisioning orchestration metadata, evaluator assertions against dataset truth, evidence **projection** from canonical artifacts—not replacement observability. |

Consistent with [Proof Architecture Design § 6](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#6-ownership-model) and [Scenario Specification § B — APPLICATION vs PROOF](SCENARIO_SPEC.md#application-vs-proof-harness).

---

## 9. Non-Goals

This document does **not** define:

| Excluded | Reason |
| --- | --- |
| Payment provider integration | Adapters are future application/scenario work behind contracts ([Proof Architecture § 7](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#7-contract-and-plugin-boundaries)). |
| Database implementation | No tables, indexes, or ORM models. |
| Production commerce data model | Qualification minimalism only; not merchant catalog or PCI systems. |
| ETL pipelines | No batch extract/load design; provisioning may be seed/load scripts later, not specified here. |
| Runtime UNKNOWN/reconciliation logic | Platform capabilities documented elsewhere; not data-layer implementation. |

---

## 10. Future Implementation Preparation

Future work ( **not part of this architecture task** ) should implement the following against this design:

| Need | Description |
| --- | --- |
| **Dataset package** | Versioned, vendor-neutral logical content under scenario `dataset/` (format TBD by implementer; must respect § 3). |
| **Provisioning mechanism** | Maps dataset → target sources; supports variant A/B/C selection; enforces redaction (no PAN/cardholder data in operator views). |
| **Validation rules** | Schema or invariant checks on logical entities before run; cross-check order ↔ payment effect ↔ inventory correlation. |
| **Evidence generation inputs** | Stable identifiers in dataset for correlating evaluator expectations with projected `PlatformProofEvidence` v3 steps. |
| **Truth boundary tests** | Automated checks that fixture truth does not leak into disallowed prompt surfaces. |

Implementation **must not** (per gap decision and proof architecture):

- Modify ERL runtime core or shared platform contracts for scenario shortcuts.
- Place proof-only fake executors on the canonical payment path.
- Couple qualification PASS to a specific database vendor.

Planned scenario locations (scaffold convention): `dataset/`, application tools with fixture-backed contracts, `proof/evaluator.py` independent truth checks—see [Proof Architecture § 10](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#10-implementation-preparation).

---

## Architectural alignment

| Principle | How this design complies |
| --- | --- |
| Contract-first thinking | Data expresses business and reality; contracts express access ([Proof Architecture § 7](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#7-contract-and-plugin-boundaries)). |
| Modularity | Dataset ↔ provisioning ↔ targets ↔ runtime are separable layers (§ 2, § 7). |
| Vendor neutrality | § 3; no PSP or storage coupling in canonical model. |
| Separation of concerns | Reality vs knowledge (§ 5); application vs proof vs platform (§ 8). |
| Reusable scenario patterns | Variant parameterization on shared entity graph (§ 6). |

---

## References

| Document | Role |
| --- | --- |
| [Scenario Specification](../SCENARIO_SPEC.md) | Normative variants, observability contract, APPLICATION vs PROOF. |
| [ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) | Proof components, evidence model, implementation preparation. |
| [README](../README.md) | Public scenario summary. |
| [SCENARIO_STRUCTURE.md](../../docs/SCENARIO_STRUCTURE.md) | Scenario package layout (`dataset/`, `docs/`). |

---

## Document Validation (quality gate)

| Check | Result |
| --- | --- |
| Aligns with ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md (variants, ownership, evidence) | **Pass** |
| No unsupported platform capability claims beyond spec § C foundation | **Pass** |
| No vendor or database coupling in canonical model | **Pass** |
| No implementation artifacts (schemas, fixtures, code) | **Pass** |
| Scope limited to `docs/` data architecture documentation | **Pass** |

---

**Architecture status:** Scenario data architecture documented for implementation guidance. Dataset and provisioning remain **not implemented**.
