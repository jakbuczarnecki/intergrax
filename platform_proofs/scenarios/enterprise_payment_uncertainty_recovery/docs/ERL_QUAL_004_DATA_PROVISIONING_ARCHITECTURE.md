---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: data_provisioning_architecture
lifecycle: IMPLEMENTATION_INITIALIZED
status: ARCHITECTURE_DOCUMENTED
---

# ERL-QUAL-004 — Data Provisioning Architecture

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Scenario Specification](../SCENARIO_SPEC.md) · [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) · [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md)

---

## 1. Purpose

Platform Proof scenarios require a **provisioning layer** between the audited **intermediate scenario dataset** and the **scenario execution environment**. Without that layer, qualification would either couple proof runs to one storage technology or duplicate business truth inside runtime-specific fixtures—breaking reuse and vendor neutrality.

The provisioning layer exists to **materialize scenario business reality** into execution environments (commerce application state, external simulators, lab backends) while preserving **vendor neutrality**. It translates logical dataset content into whatever concrete sources the lab uses, without changing what the scenario **means**.

**Key statement:** The dataset defines **what is true**. Provisioning defines **how that truth becomes available**.

This document defines **provisioning architecture only**. It does not create contracts, loaders, adapters, registries, or runtime code. The logical dataset under `dataset/` is documented in [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md); proof orchestration and evidence are in [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md).

---

## 2. Architecture Overview

Provisioning sits on a **boundary** between stable qualification inputs and replaceable infrastructure. The scenario definition (specification + dataset + variant selection) remains unchanged when the lab swaps relational storage for an in-memory simulator or an event-backed external adapter.

```text
Scenario Dataset
        ↓
Provisioning Boundary
        ↓
Provisioning Implementation
        ↓
Target Data Source
        ↓
Scenario Runtime
```

| Layer | Owner | Role |
| --- | --- | --- |
| **Scenario Dataset** | Proof / scenario authors | Vendor-neutral business truth, shared entities, variant slices, expected outcomes for evaluator alignment. **Source of qualification truth**—not execution state. |
| **Provisioning Boundary** | Proof harness (conceptual contract) | Stable capability surface between dataset and implementations: validate inputs, orchestrate prepare/provision/cleanup, isolate scenario definition from technology choices. |
| **Provisioning Implementation** | Future lab/scenario implementers | Concrete mapping from logical entities to seeds, rows, events, or simulator configuration—**one of several possible backends** behind the same boundary. |
| **Target Data Source** | Lab infrastructure / simulators | Storage, retrieval, and operational behavior of order DB, inventory backend, payment SoR simulator, etc. Truth is reached only through **application contracts**, not raw dataset access by the runtime. |
| **Scenario Runtime** | Integrax execution + scenario application | Workflow execution, ERL capability usage, observability spine. Consumes **application knowledge** produced by reads/writes against targets—not the canonical dataset file graph. |

Dependency direction follows **inversion**: runtime and application depend on ports; provisioning depends on dataset semantics and target capabilities; the dataset does not depend on any provisioner or database.

---

## 3. Responsibilities

Clear ownership prevents business rules from leaking into seed scripts and prevents storage details from becoming qualification requirements.

### Scenario Dataset owns

- **Business truth** — order, payment external effect, communication uncertainty, inventory context, application knowledge at entry (shared graph).
- **Scenario variants** — parameterization of external reality and reconciliation availability (`payment_completed_after_unknown`, `payment_failed_after_unknown`, `payment_truth_unavailable`).
- **Expected outcomes** — terminal business expectations and evidence themes per variant for independent evaluator alignment.

### Provisioning Layer owns

- **Data delivery** — translating logical dataset entities into materialized state in target sources.
- **Environment preparation** — ensuring lab sources are initialized, idempotent where required, and consistent with selected variant before execution starts.
- **Source initialization** — wiring variant selection to the correct external reality and simulator behavior without redefining variant semantics.

### Target Data Source owns

- **Storage** — persistence model chosen by the lab (not prescribed here).
- **Retrieval** — query/API behavior exposed through application-facing contracts.
- **Operational behavior** — availability, latency, and failure modes of lab backends (e.g. reconciliation endpoint down in Variant C).

### Scenario Runtime owns

- **Execution flow** — order → payment → UNKNOWN → reconciliation → resolution/compensation/governance per [Proof Architecture Design § 3](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#3-end-to-end-proof-flow).
- **Platform capability usage** — ERL processing on the observability spine; not authoritative SoR truth fabrication on the canonical path.

The **proof harness** orchestrates variant selection, provisioning lifecycle, evaluator truth checks, and evidence projection—consistent with [Scenario Data Architecture § 8](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#8-ownership-model) and [Proof Architecture Design § 6](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#6-ownership-model).

---

## 4. Vendor Neutrality Model

Provisioning must allow the **same** scenario dataset and specification to be exercised through different materialization strategies. The architect does not select an implementation in this document; the model only requires that implementations remain **interchangeable** at the provisioning boundary.

Future provisioning implementations may target, among others:

| Style | Illustrative role (non-exclusive) |
| --- | --- |
| **Relational storage** | Seed order and inventory rows for commerce application tools. |
| **Document storage** | Load bundled logical documents where the lab prefers document-native APIs. |
| **Event systems** | Emit communication or provider-side facts as streams, still behind contract-shaped consumers. |
| **External system adapters** | Initialize payment/inventory simulators with SoR truth from variant slices. |

Neutrality rules:

- Canonical dataset content remains free of vendor API shapes, ORM artifacts, and infrastructure identifiers ([Scenario Data Architecture § 3](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#3-vendor-neutrality-principles)).
- A change of target technology changes **how** truth is loaded, not **what** Variant A/B/C assert about external reality.
- Application and platform continue to depend on **contracts** ([Proof Architecture Design § 7](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#7-contract-and-plugin-boundaries)), not on dataset file layout.

---

## 5. Provisioning Contract Concept

At the provisioning **boundary**, consumers (proof runner, harness) should interact with **stable capabilities**, not with storage drivers or vendor SDKs. This section describes concepts only—**no code contracts, no Python interfaces, no schema IDs**.

| Capability (conceptual) | Intent |
| --- | --- |
| **Load scenario context** | Resolve qualification id, scenario slug, dataset manifest, and selected variant; validate logical invariants before materialization. |
| **Prepare required data** | Ensure shared entities and variant slice are consistent (correlation across order, external effect, inventory, communication). |
| **Provision environment** | Materialize logical truth into configured target data sources and simulator seeds for the active variant. |
| **Expose scenario state** | Offer harness-visible metadata (e.g. provisioning succeeded, target handles, redaction applied)—not hidden external reality to model prompts. |
| **Cleanup environment** | Tear down or reset lab state after evidence collection so repeated runs remain isolated and idempotent per lab policy. |

The boundary is the **plugin surface** for future implementations: multiple backends can satisfy the same capability set while reading the same `dataset/` package.

---

## 6. Lifecycle

Provisioning participates in the proof run as a **phased** responsibility, with explicit handoffs to runtime execution and evaluation.

```text
Prepare
   ↓
Provision
   ↓
Execute Scenario
   ↓
Collect Evidence
   ↓
Cleanup
```

| Phase | Primary owner | Provisioning role |
| --- | --- | --- |
| **Prepare** | Proof harness | Select variant; load and validate dataset; configure which targets participate; fail fast on invalid or incomplete logical inputs. |
| **Provision** | Provisioning implementation | Materialize shared + variant truth into targets; confirm readiness for application entry state. |
| **Execute Scenario** | Scenario runtime + application | Run canonical workflow; ERL capabilities engage on observability spine—provisioning does not drive business decisions. |
| **Collect Evidence** | Proof harness + evidence generator | Project traces and journal facts; evaluator compares observability to **dataset expected outcomes**, not to provisioner internals. |
| **Cleanup** | Provisioning implementation (invoked by harness) | Reset or dispose lab materialization; ownership of long-lived lab infra remains outside scenario definition. |

Execution and business PASS/FAIL are **runtime/evaluator** concerns. Provisioning success means the environment truthfully reflects the selected variant inputs—not that the merchant workflow succeeded.

---

## 7. Scenario Variant Handling

Variants share the same commerce entry and differ in **fixture-controlled external truth** and **reconciliation availability** ([Proof Architecture Design § 4](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#4-scenario-variants)). Provisioning **selects and materializes** the variant slice; it does **not** decide business outcomes.

| Variant | Dataset identity | What provisioning materializes (logical) |
| --- | --- | --- |
| **A — Payment completed after unknown** | `payment_completed_after_unknown` | External SoR truth: capture succeeded; reconciliation reachable; communication explains missing wire confirmation. |
| **B — Payment failed after unknown** | `payment_failed_after_unknown` | External SoR truth: capture failed or not present; reconciliation reports failure; inventory recovery preconditions consistent with failed payment. |
| **C — Truth unavailable** | `payment_truth_unavailable` | Reconciliation unavailable or exhausted within policy; external reality may exist but is **not authoritatively discoverable** in bounds—materialize unavailability, not a fake “success” for the application. |

Rules:

- Variant selection is **configuration** (harness/proof config), not logic embedded in provisioners that infers A vs B from runtime behavior.
- Provisioning **must not** contain business decisions (e.g. choosing compensation vs continuation—that remains application + ERL under policy).
- The same variant identifier must yield the same **logical** truth regardless of provisioning implementation ([Scenario Data Architecture § 7](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#7-data-provisioning-architecture)).

---

## 8. Failure Model

Provisioning failures are **environment and qualification-setup** failures. They are distinct from **business scenario failures** (e.g. governed UNRESOLVED in Variant C, which may still be a successful proof when invariants hold).

| Failure class | Examples | Typical handling |
| --- | --- | --- |
| **Dataset invalid** | Broken manifest, missing variant file, inconsistent correlation IDs, invariant violation in shared graph | Fail before execution; report validation errors; no partial materialization that could mislead evaluator. |
| **Source unavailable** | Target DB down, simulator not reachable, adapter misconfigured | Fail provision phase; do not start scenario runtime on a lie. |
| **Incomplete provisioning** | Only subset of entities seeded; variant slice not applied to SoR simulator | Treat as provision failure; harness should not mark run as valid qualification attempt. |
| **Inconsistent environment** | Order row present but inventory reservation missing; duplicate seeds from non-idempotent re-run | Detect at provision or prepare; fail or cleanup per lab policy before trusting evaluator results. |

Business-path outcomes (UNKNOWN, reconciliation failure, escalation) are **expected behaviors** in Variant C and are validated by the evaluator against scenario specification—not classified as provisioning errors when the environment correctly reflects “reconciliation unavailable.”

---

## 9. Plugin / Adapter Direction

Extensibility follows the same pattern as application external effects: **one dataset**, **one conceptual provisioning contract**, **many implementations**.

```text
Dataset
   ↓
Provisioning Contract (boundary capabilities)
   ↓
┌────────────────┬────────────────┬────────────────┐
│ Implementation │ Implementation │ Implementation │
│       A        │       B        │       C        │
└────────────────┴────────────────┴────────────────┘
```

Example illustrations only (not chosen, not registered):

- **Implementation A** — relational seed scripts for commerce + SQL-backed SoR simulator.
- **Implementation B** — in-memory document store for rapid local proof runs.
- **Implementation C** — event-initialized provider simulator with adapter matching production port shape.

This document does **not** define a registry, discovery mechanism, or plugin SPI. Future work selects how the harness binds a implementation to a lab profile.

---

## 10. Non-Goals

This document does **not** define:

| Excluded | Reason |
| --- | --- |
| **Database schema** | Target storage is implementer-chosen; dataset stays logical. |
| **Deployment architecture** | No Kubernetes, regions, or network topology for qualification labs. |
| **Production ingestion** | Merchant ETL, PCI systems, and live PSP feeds are out of scope. |
| **Infrastructure automation** | Terraform, CI secrets, and environment provisioning pipelines are not specified here. |
| **Provider integrations** | PSP SDKs and vendor APIs belong behind application adapters ([Proof Architecture Design § 9](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#9-non-goals)). |

---

## 11. Future Implementation Preparation

Implementation tasks (not part of this architecture document) should prepare:

| Need | Description |
| --- | --- |
| **Provisioning contract** | Formalize boundary capabilities (§ 5) in the scenario/proof layer without coupling to a single storage vendor. |
| **Implementation adapters** | One or more provisioners that read `dataset/` manifest + shared + variant paths and materialize targets. |
| **Lifecycle integration** | Wire Prepare → Provision → Execute → Evidence → Cleanup into proof runner orchestration with clear failure taxonomy (§ 8). |
| **Validation** | Pre-run checks on dataset invariants; post-provision consistency checks; truth-boundary tests that hidden SoR truth does not leak to disallowed prompt surfaces ([Scenario Data Architecture § 10](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#10-future-implementation-preparation)). |

Implementation must preserve:

- **Contract-first design** — runtime reads through application ports.
- **Dependency inversion** — dataset and scenario spec do not depend on provisioner types.
- **Modular boundaries** — dataset ↔ provisioning ↔ targets ↔ runtime separable.
- **Vendor neutrality** — interchangeable provisioners for the same variant identifiers.
- **Reusable scenario patterns** — shared entity graph + variant slices reusable across ERL qualification scenarios.

---

## Architectural alignment

| Principle | How this design complies |
| --- | --- |
| Contract-first | Provisioning materializes truth for **contract-backed** targets; does not replace ports ([Proof Architecture § 7](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#7-contract-and-plugin-boundaries)). |
| Dependency inversion | Scenario definition depends on logical dataset only; implementations plug in at boundary (§ 2, § 9). |
| Modular boundaries | Lifecycle phases and ownership tables (§ 3, § 6) separate harness, provisioner, targets, runtime. |
| Vendor neutrality | § 4; no selected storage or PSP. |
| Reusable patterns | Variant materialization without business logic in provisioner (§ 7). |

---

## References

| Document | Role |
| --- | --- |
| [Scenario Specification](../SCENARIO_SPEC.md) | Normative variants, observability, APPLICATION vs PROOF. |
| [ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) | Proof flow, components, evidence, ownership. |
| [ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md) | Logical data model, dataset ownership, high-level provisioning pointer (§ 7). |
| [dataset/README.md](../dataset/README.md) | Dataset package layout and manifest convention. |
| [README](../README.md) | Public scenario summary. |

---

## Document Validation (quality gate)

| Check | Result |
| --- | --- |
| Aligns with ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md and ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md | **Pass** |
| No implementation decisions (DB, vendor, deployment) | **Pass** |
| No vendor coupling in canonical provisioning model | **Pass** |
| Ownership boundaries explicit (§ 2, § 3, § 6) | **Pass** |
| Scope limited to documentation under scenario `docs/` | **Pass** |
| No code, contracts, adapters, or runtime changes | **Pass** |

---

**Architecture status:** Data provisioning architecture documented for future implementation. Provisioning implementations remain **not implemented**.
