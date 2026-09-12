---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: proof_architecture_design
lifecycle: IMPLEMENTATION_INITIALIZED
status: ARCHITECTURE_DOCUMENTED
---

# ERL-QUAL-004 — Proof Architecture Design

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Scenario Specification](../SCENARIO_SPEC.md)

---

## 1. Executive Summary

### Enterprise problem demonstrated

After a high-value customer checkout, an external payment capture may have succeeded, failed, or still be in flight—but the integration returns **no definitive confirmation**. Operations cannot safely retry capture (duplicate charge), ship (fulfillment without payment), or cancel (customer charged but order abandoned) without establishing external truth or a governed escalation path.

This Platform Proof scenario demonstrates that Integrax treats that ambiguity as a **managed reliability case**, not as a silent transport error or an excuse for blind retries.

### Business statement

> Distributed systems cannot always know whether an external business operation succeeded. Integrax converts uncertainty into a controlled recovery process.

### What qualification proves

The proof shows that Integrax:

- Admits **UNKNOWN** as a first-class platform state when payment outcome is ambiguous.
- **Pauses risky work** (duplicate capture, unconditional fulfillment) until reconciliation, resolution, or governance allows proceed.
- **Discovers truth** through reconciliation when the external system of record is reachable.
- **Records auditable evidence** linking classification, reconciliation, resolution, and governance decisions.
- **Separates execution from material continuation decisions** under Enterprise Reliability Layer (ERL) boundaries.
- Reaches **safe terminal business outcomes** across three controlled variants (paid, failed, truth unavailable).

This document defines **proof architecture only**. It does not implement runtime logic, adapters, or scenario execution code.

---

## 2. Business Scenario Architecture

### Actors and responsibilities

| Actor | Responsibility |
| --- | --- |
| **Customer** | Places order; expects a single correct charge and accurate fulfillment status. Does not participate in platform recovery mechanics. |
| **Commerce Application** | Owns order lifecycle, inventory reservations, and business decisions (hold, continue, compensate, escalate) under policy. Invokes payment and inventory through declared external-effect contracts—not direct PSP SDK coupling in proof design. |
| **Integrax Runtime** | Executes workflow steps and external effects under Unified Execution Runtime rules. Surfaces UNKNOWN, reconciliation hooks, and lifecycle handoff without substituting for merchant business policy. |
| **Payment Provider** | External system of record for capture/settlement status. Authoritative truth lives outside Integrax; accessed only through abstract reconciliation/capture capability contracts in this proof. |
| **Inventory System** | Holds reservations and fulfillment commitments tied to payment truth. Must stay consistent with order and payment resolution outcomes. |
| **Human Operator** | Receives governed escalation when automated reconciliation cannot establish truth within policy bounds (Variant C). Acts on evidence bundle, not proof-only artifacts. |

### Architectural placement

```text
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Customer   │────▶│ Commerce Application │────▶│ Integrax Runtime │
└─────────────┘     │  (order / inventory) │     │  + ERL capabilities │
                    └──────────┬───────────┘     └────────┬─────────┘
                               │                          │
                               │    external-effect       │
                               │    + reconciliation      │
                               ▼                          ▼
                    ┌──────────────────────┐     ┌──────────────────┐
                    │ Inventory System     │     │ Payment Provider │
                    │ (domain dependency)  │     │ (SoR for capture)│
                    └──────────────────────┘     └──────────────────┘
                               ▲
                               │ escalation (Variant C)
                    ┌──────────┴───────────┐
                    │   Human Operator     │
                    └──────────────────────┘
```

No vendor-specific payment product names or proprietary APIs are required for this architecture. Provider behavior is represented by a **controlled simulator** behind the same contract shape the application would use in production.

---

## 3. End-to-End Proof Flow

### Happy recovery flow (primary narrative)

```text
Order
  ↓
Payment Request
  ↓
External Outcome Unknown
  ↓
UNKNOWN
  ↓
Reconciliation
  ↓
Evidence
  ↓
Resolution
  ↓
Governance
  ↓
Recovery Lifecycle
  ↓
Continue
```

### Stage-by-stage proof value

| Stage | What happens (business) | What the stage **proves** |
| --- | --- | --- |
| **Order** | Customer commits to purchase; order and inventory correlation identifiers exist. | Application owns domain workflow; platform does not invent orders. |
| **Payment Request** | Capture initiated via **External Effect Contract** with safety metadata. | Contract-first external effects; no hardcoded PSP in platform core. |
| **External Outcome Unknown** | Integration returns no definitive success/failure after capture attempt. | Real-world ambiguity is modeled—not collapsed into generic HTTP error. |
| **UNKNOWN** | Platform classifies effect as managed uncertainty; risky steps paused. | **UNKNOWN state** is explicit; not silently mapped to SUCCESS or FAILURE. |
| **Reconciliation** | Query payment system of record when reachable. | **Reconciliation** discovers external truth without blind retry. |
| **Evidence** | Classification and reconciliation facts persisted on observability / journal spine. | **Evidence** is auditable; proof projects canonical events only. |
| **Resolution** | Map truth to continue, compensate, or escalate path. | **Resolution** separates decision from execution retry. |
| **Governance** | Policy evaluates risky continuation (especially Variant C). | **Governance** gates material continuation; does not execute payments. |
| **Recovery Lifecycle** | Reliability case transitions (e.g. UNKNOWN detected → reconciled → closed). | **Recovery Lifecycle** coordinates case without duplicate orchestration engines. |
| **Continue** | Lifecycle handoff resumes or stops execution under runtime rules. | **Lifecycle Handoff** preserves execution isolation; safe business continuation or containment. |

Variant-specific terminal paths branch after **Reconciliation** (see § 4). The primary flow proves the **shared uncertainty containment spine**; variants prove **outcome-specific invariants**.

---

## 4. Scenario Variants

All variants share the same entry: order → payment request → ambiguous immediate outcome → **UNKNOWN**. They differ in **fixture-controlled external truth** and **expected terminal business result**.

### Variant A — Payment succeeded

```text
UNKNOWN → Reconciliation → Payment confirmed → Safe continuation
```

| Aspect | Expectation |
| --- | --- |
| **External truth** | Provider system of record confirms capture succeeded (confirmation was missing on the wire, not in the ledger). |
| **Expected behavior** | Order continues toward fulfillment **without duplicate payment**; order and inventory align with captured funds. |
| **Proof value** | **No duplicate charge**—demonstrates reconciliation before retry and idempotency at the business layer, not only HTTP keys. |
| **PASS invariant** | Single confirmed capture; no second capture attempt without governed policy; no ship-without-capture. |

### Variant B — Payment failed

```text
UNKNOWN → Reconciliation → Payment failed → Controlled recovery
```

| Aspect | Expectation |
| --- | --- |
| **External truth** | Reconciliation reports failure or non-capture. |
| **Expected behavior** | **Compensation** / controlled recovery releases reservations; order reflects failed payment consistently. |
| **Proof value** | **No inconsistent business state**—no silent “paid” flag, no shipped goods, no orphaned inventory lock. |
| **PASS invariant** | Terminal order/inventory state matches failed payment; compensation path recorded in evidence. |

### Variant C — Truth unavailable

```text
UNKNOWN → Reconciliation unavailable → Governance → Human escalation
```

| Aspect | Expectation |
| --- | --- |
| **External truth** | Reconciliation cannot establish authoritative outcome within policy bounds (endpoint unavailable, stale/disputed signals per spec). |
| **Expected behavior** | **Governance escalation** to human operator; risky automation contained. |
| **Proof value** | **Safe failure handling**—no blind retry, no optimistic fulfillment, no unbounded automation pretending certainty. |
| **PASS invariant** | Escalation with evidence bundle; UNRESOLVED or governed terminal path; no ungoverned continuation. |

### Scenario outcomes

| Outcome | Meaning |
| --- | --- |
| **RESOLVED** | External truth established (Variants A or B); business state aligned; safe continuation or controlled compensation completed with evidence. |
| **UNRESOLVED** | Authoritative truth not determined in bounded time (Variant C); operator path with containment. |

---

## 5. Proof Architecture Components

Conceptual components only—**no implementation** in this task.

| Component | Responsibility | Must not |
| --- | --- | --- |
| **Scenario Application** | Simulate commerce context: order, payment correlation, inventory reservations, business decisions under policy. | Own platform UNKNOWN semantics or replace ERL governance. |
| **External System Simulator** | Represent uncertain payment provider and inventory backends behind **application-facing contracts**; inject Variant A/B/C truth via controlled fixtures. | Leak fixture truth into model-visible prompts outside controlled interfaces; act as proof-only fake on canonical execution path. |
| **ERL Capability Adapter** | Wire existing platform capabilities (external effects, UNKNOWN, reconciliation, resolution, compensation, governance, recovery lifecycle, lifecycle handoff) through lab **runtime composition**—not new orchestration. | Introduce scenario-specific framework or duplicate ERL engines. |
| **Evidence Generator** | Project qualification evidence from canonical runtime and ERL observability (e.g. `PlatformProofEvidence` v3)—steps, graph, stable identifiers. | Invent payment outcomes absent from runtime artifacts. |
| **Proof Evaluator** | Verify PASS/FAIL invariants independently of application optimism (no duplicate capture, no ship-without-capture, governance on Variant C). | Fabricate rationale or reconstruct intent not present in traces. |
| **Proof Runner** | Thin scenario entry (`run_proof.py` scaffold) orchestrating variant execution and report hooks per platform proof protocol. | Become hidden execution authority overriding application or runtime. |

### Data and control separation

```text
  PROOF layer          APPLICATION + PLATFORM (canonical path)
  ───────────          ─────────────────────────────────────
  variant config  ──▶   application workflow
  evaluator       ◀──   traces, ERL journal, diagnostics
  evidence proj   ◀──   observability spine (read-only projection)
  report          ◀──   aggregated PASS/FAIL + metadata
```

---

## 6. Ownership Model

### APPLICATION owns

- Business workflow (order → payment capture → fulfillment decisions).
- Domain data (order identifiers, inventory reservations, payment correlation IDs).
- Business decisions: when to hold, continue, compensate, or request escalation under declared policy.
- Consumption of external systems through **normal application tools/contracts**.

### PROOF owns

- Scenario execution orchestration and variant selection (A/B/C fixture configuration).
- Controlled simulation configuration (adversarial inputs, reconciliation availability).
- Evaluator falsification assertions and invariant verification against fixture truth.
- Evidence projection and reproduction metadata for qualification reports.
- Demonstration artifacts that do **not** replace application observability.

### INTEGRAX PLATFORM owns

- Reliability mechanisms: UNKNOWN classification, reconciliation gateway, resolution, compensation hooks.
- Lifecycle decisions on reliability cases and **governance boundaries** (evaluate, do not silently execute risky continuation).
- Execution runtime isolation and lifecycle handoff semantics.
- Canonical observability spine (`TraceEvent`, tool traces, ERL journal events).

**Non-negotiable:** Proof **projects** platform and application facts; it does **not** own business workflow, domain state, or payment truth fabrication on the canonical path.

---

## 7. Contract and Plugin Boundaries

### Contract-first external systems

External dependencies are **abstract capabilities**, not named products:

| Capability | Contract role (conceptual) |
| --- | --- |
| Payment capture | External effect with safety metadata; outcome may be unknown after invoke. |
| Payment reconciliation | Plugin gateway to system-of-record status query. |
| Inventory reservation | Domain external effect or tool contract tied to order state. |

**Do not hardcode** in architecture or future implementation:

- Specific payment provider brands or proprietary REST paths as platform requirements.
- Vendor SDKs inside `intergrax/` or ERL core.
- Database schemas of third-party ledgers except as exposed through reconciliation envelopes.

### Plugin architecture application

- **Reconciliation** and related ERL evaluators register through existing Enterprise Reliability plugin SPI—scenario lab composition wires plugins; scenario does not fork platform contracts.
- **External Effect Contracts** declare payment capture semantics; simulator implements the same contract surface as a production adapter would.
- **Dependency inversion:** Application depends on contract ports; fixtures and future real adapters implement those ports.

### Forbidden architectural patterns

- Scenario-specific orchestration engine parallel to Unified Execution Runtime.
- Proof-layer executor that intercepts canonical payment calls without application tools.
- Direct coupling from evaluator to provider internals bypassing runtime observability.

---

## 8. Evidence Model

### Qualification must demonstrate (evidence themes)

Evidence is derived from **production-path observability**, projected by proof packaging—not logged only inside the evaluator.

| Evidence theme | Demonstrates |
| --- | --- |
| **UNKNOWN state created** | Ambiguous payment outcome admitted; not classified as terminal SUCCESS/FAILURE without path. |
| **Reconciliation executed** | Reconciliation started/completed/failed with bounded policy semantics. |
| **Evidence received** | Classification and reconciliation facts persisted with stable identifiers for audit. |
| **Decision created** | Resolution and/or governance decision recorded with explainable bounded rationale. |
| **Lifecycle transitioned** | Recovery lifecycle case moves through documented states to handoff or escalation. |
| **Final business state consistent** | Order, payment correlation, and inventory align with established truth (A/B) or containment (C). |

### Variant-specific evidence expectations

| Variant | Additional evidence emphasis |
| --- | --- |
| **A** | Confirmed single capture; continuation without second capture; fulfillment authorized after resolution/governance. |
| **B** | Failed payment resolution; compensation/recovery actions; no paid/shipped inconsistency. |
| **C** | Reconciliation exhaustion or unavailability; governance deny/escalate; operator-visible escalation reason. |

### Machine-readable projection

Proof build targets platform proof framework shapes (e.g. **PlatformProofEvidence v3** graph/steps) referencing canonical events. Evaluator cross-checks invariants against **fixture truth independently** of application narrative.

### FAIL conditions (evidence gaps)

- Missing records for UNKNOWN admission, reconciliation attempt, or governance on Variant C.
- Proof or report asserts payment outcomes not present in runtime artifacts.
- Duplicate capture or ship-without-capture detected by evaluator.

---

## 9. Non-Goals

This scenario is **explicitly not**:

| Non-goal | Rationale |
| --- | --- |
| **Payment product** | Integrax is not a PSP, gateway, or PCI-certified payment processor. |
| **Production commerce system** | Minimal lab application demonstrates reliability patterns, not merchant feature completeness. |
| **Replacement for customer application** | Real merchants keep their checkout; this package qualifies platform behavior around a representative app component. |
| **Performance benchmark** | Qualification is correctness, containment, and auditability under uncertainty—not throughput or latency SLAs. |
| **Universal PSP guarantee** | No claim that any third-party API is always available or behaves identically to fixtures. |
| **Fraud, chargeback, or PCI certification** | Adjacent enterprise concerns are out of scope (see Scenario Specification § B Excluded claims). |

---

## 10. Implementation Preparation

Future implementation work ( **not part of this architecture task** ):

### Required components

| Component | Location (scaffold convention) | Notes |
| --- | --- | --- |
| Scenario application workflow | `application/` | Order, payment invoke, inventory; Application Survival **YES**. |
| Runtime composition | `application/runtime_composition.py` | Wire ERL plugins and external-effect metadata—**USE_EXISTING_CAPABILITY** per gap decision. |
| Controlled provider + inventory simulator | Application tools / fixture-backed integration | Variant A/B/C truth; canonical path only. |
| Proof evaluator | `proof/evaluator.py` | Invariants: no duplicate capture, no ship-without-capture, Variant C escalation. |
| Evidence builder | `proof/evidence_builder.py` | Project v3 evidence from traces/journal. |
| Scenario contracts (if needed) | `contracts/` | Shared types across application and proof—scenario-owned only. |

### Required adapters

- **Application-facing** payment capture and reconciliation implementations matching external-effect and reconciliation contract shapes.
- **ERL lab wiring** for governance, resolution, compensation, recovery lifecycle, lifecycle handoff—no new platform orchestration.

### Required fixtures

- Fixture-controlled authoritative payment truth for Variants A, B, C.
- Optional reconciliation unavailability injection for Variant C.
- Redaction-safe diagnostic payloads (no PAN/cardholder data in operator views).

### Required validation

- Application Observability Test: UNKNOWN, reconciliation, governance visible without proof-only logger.
- Canonical path free of prohibited TEST-ONLY substitutes on execution path.
- PASS/FAIL matrix in Scenario Specification § B satisfied per variant run.
- Independent evaluator checks against fixture truth.

### Documentation follow-ups (architect-owned)

- Hero illustration (light/dark SVG under `assets/`).
- Technical Mermaid diagrams for primary flow, UNKNOWN state machine, variants (post–quality gate enrichment).

### Explicit implementation scope limit

Implementation **must not**:

- Modify ERL runtime core, shared platform contracts, or scenario framework for scenario-specific shortcuts.
- Add hidden execution authority in proof layer.
- Introduce duplicate workflow engines.

---

## Architectural Rules (Compliance Checklist)

| Rule | Design compliance |
| --- | --- |
| Contract-first architecture | External systems via capability contracts only (§ 7). |
| Dependency inversion | Application → ports; simulators/adapters implement ports (§ 5, § 7). |
| Pluginability | ERL reconciliation/governance via existing SPI wiring (§ 7). |
| Modular boundaries | Application / proof / platform ownership separated (§ 6). |
| Application/proof separation | Proof projects; application owns workflow (§ 6, § 8). |
| No scenario-specific framework | Lab composition only (§ 5, § 10). |
| No duplicate orchestration | Unified Execution Runtime + ERL lifecycle (§ 3, § 7). |
| No hidden execution authority | Runner thin; runtime executes (§ 5). |
| No direct provider coupling | Generic provider simulator (§ 2, § 7). |

---

## References

| Document | Role |
| --- | --- |
| [Scenario Specification § A–E](../SCENARIO_SPEC.md) | Normative scenario contract, variants, fit, gap decision. |
| [README](../README.md) | Public scenario summary and PASS/FAIL table. |
| [ENTERPRISE_RELIABILITY_LAYER.md](../../../../docs/project/architecture/ENTERPRISE_RELIABILITY_LAYER.md) | ERL capability baseline. |
| [UNCERTAINTY_MANAGEMENT.md](../../../../docs/project/architecture/UNCERTAINTY_MANAGEMENT.md) | UNKNOWN and resolution semantics. |
| [RECONCILIATION.md](../../../../docs/project/architecture/RECONCILIATION.md) | Reconciliation gateway pattern. |
| [RECOVERY_AND_COMPENSATION.md](../../../../docs/project/architecture/RECOVERY_AND_COMPENSATION.md) | Compensation for Variant B. |
| [UNIFIED_EXECUTION_RUNTIME.md](../../../../docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md) | Lifecycle handoff. |
| [ERL_QUAL_004_SCENARIO_QUALITY_GATE.md](../../../../docs/project/architecture/ERL_QUAL_004_SCENARIO_QUALITY_GATE.md) | Quality gate acceptance basis. |
| [SCENARIO_STRUCTURE.md](../../docs/SCENARIO_STRUCTURE.md) | Scenario package layout standard. |

---

## Document Validation (quality gate)

| Check | Result |
| --- | --- |
| Matches accepted ERL-QUAL-004 scenario (spec § A–B, variants A/B/C) | **Pass** |
| Capabilities limited to documented ERL foundation (spec § C matrix) | **Pass** |
| No contradiction with APPLICATION vs PROOF ownership (spec § B) | **Pass** |
| Ownership boundaries explicit (§ 6) | **Pass** |
| Implementation scope limited to preparation list (§ 10); no runtime in this task | **Pass** |
| No forbidden artifacts (Python/tests/adapters/ERL changes) in this commit | **Pass** |

---

**Architecture status:** Document complete for implementation guidance. Executable proof remains **NOT STARTED** until post-scaffold build (Scenario Specification § E).
