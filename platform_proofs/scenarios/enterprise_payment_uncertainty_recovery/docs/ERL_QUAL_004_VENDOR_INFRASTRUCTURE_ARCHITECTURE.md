---
qualification_id: ERL-QUAL-004
scenario_slug: enterprise_payment_uncertainty_recovery
document_type: vendor_infrastructure_architecture
lifecycle: IMPLEMENTATION_INITIALIZED
status: ARCHITECTURE_DOCUMENTED
---

# ERL-QUAL-004 — Vendor Infrastructure Architecture Decision

**Enterprise Payment Uncertainty Recovery**

[← Public scenario page](../README.md) · [Scenario Specification](../SCENARIO_SPEC.md) · [Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) · [Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md) · [Data Provisioning Architecture](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md)

---

## 1. Purpose

This document records **concrete infrastructure choices** for the executable end-to-end (E2E) proof of ERL-QUAL-004. It answers which lab technologies host materialized scenario state and how those hosts relate to the existing vendor-neutral dataset and provisioning contract—**without** implementing containers, schemas, adapters, or runtime wiring.

**Important distinction:**

| Concept | Meaning |
| --- | --- |
| **Scenario Dataset** | **Business truth** — vendor-neutral logical entities, variants, and expected outcomes under `dataset/`. Unchanged when lab infrastructure changes. |
| **Vendor Infrastructure** | **Execution environment** — replaceable hosts (database, simulators, containers) that materialize truth for the scenario application and proof harness. |

The architect has selected direction for v1 E2E realism; this document **explains why** those choices support qualification goals while preserving dependency inversion, modularity, and plugin readiness defined in prior architecture artifacts.

This document does **not** implement infrastructure. The current **reference** in-memory provisioner remains a contract validator, not the final E2E stack.

---

## 2. Architecture Overview

The E2E proof stack is a **downward dependency chain**: qualification inputs stay at the top; concrete vendors plug in only at the provisioning implementation and Docker layers.

```text
Vendor-Neutral Dataset
        ↓
Provisioning Contract
        ↓
Vendor Implementation
        ↓
Docker Infrastructure
        ↓
Scenario Application
        ↓
Integrax Runtime
```

| Layer | Role in E2E proof |
| --- | --- |
| **Vendor-neutral dataset** | Canonical business truth and variant slices ([Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md)). |
| **Provisioning contract** | Stable boundary for prepare/provision/cleanup and materialization semantics ([Data Provisioning Architecture](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md)). |
| **Vendor implementation** | Concrete provisioner (e.g. PostgreSQL-backed adapter) mapping logical entities to lab state—**swappable** behind the contract. |
| **Docker infrastructure** | Reproducible composition of database, scenario app, external boundary, and runtime dependencies for local and CI-style runs. |
| **Scenario application** | Commerce workflow, contracts, and tools exercising uncertainty → reconciliation → recovery ([Proof Architecture Design](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md)). |
| **Integrax runtime** | Platform capability execution and observability spine; not authoritative SoR fabrication on the canonical path. |

Dependency direction: application and runtime depend on **ports**; provisioning depends on dataset semantics and target capabilities; the dataset does not depend on PostgreSQL, Docker, or any specific vendor.

---

## 3. Database Decision

### Selected technology

**PostgreSQL** (relational database engine in the lab E2E environment).

### Rationale

| Factor | Why it matters for ERL-QUAL-004 |
| --- | --- |
| **Relational business model** | Order, payment state, inventory context, and communication events map naturally to transactional records and constraints—matching how enterprise commerce systems represent operational truth. |
| **Transactional consistency** | Reconciliation and recovery paths depend on coherent read/write semantics when uncertainty resolves; ACID transactions support realistic race and consistency scenarios without inventing a custom store. |
| **Enterprise adoption** | PostgreSQL is a common SoR choice in enterprise labs and CI; qualification gains realism when the proof exercises patterns teams already operate. |
| **Docker support** | Official images and compose-friendly operation make **reproducible** developer and automation environments practical for a standalone E2E proof. |
| **Reproducible environments** | Fresh volumes and scripted provisioning align with variant materialization and idempotent lab setup described in provisioning architecture. |

### Explicit non-scope

This decision does **not** define tables, columns, indexes, migrations, or ORM mappings. Schema design belongs to a later implementation task behind application contracts.

---

## 4. Vector Database Decision

### Decision

**Not required in v1** of the E2E proof.

### Rationale

The scenario qualifies **uncertainty handling, reconciliation, evidence, and recovery** under ERL capabilities—not retrieval-augmented reasoning over unstructured knowledge.

The proof does **not** require:

- semantic search over documents,
- embedding pipelines,
- RAG workflows,
- vector-backed knowledge retrieval.

Introducing a vector store would add operational surface and coupling without increasing proof value for the documented variants (`payment_completed_after_unknown`, `payment_failed_after_unknown`, `payment_truth_unavailable`). If future scenarios extend into semantic operations, vector storage may be reconsidered as a **separate** architecture decision.

---

## 5. Event Streaming Decision

### Decision

**Not required in v1** of the E2E proof.

### Rationale

ERL-QUAL-004 focuses on **operational uncertainty and reconciliation** against commerce and payment truth boundaries—not on event-sourced platform architecture, stream processing, or multi-subscriber event backbones.

Event streaming (e.g. Kafka-style buses) is **out of scope** for this qualification slice. A future extension that models async payment notifications at scale could introduce streaming **independently**, without changing the vendor-neutral dataset or provisioning contract semantics.

---

## 6. External System Boundary

The **payment system boundary** represents **external reality** relative to the commerce application: authoritative payment outcomes, reconciliation availability, and simulator behavior for variant slices.

### Architectural requirements

| Requirement | Intent |
| --- | --- |
| **Replaceable** | Any lab implementation (HTTP simulator, stub service, recorded fixtures) may sit behind the application port as long as variant truth matches the dataset. |
| **Isolated** | External truth is not merged into the canonical dataset files; provisioning materializes **runtime** state separately ([Scenario Data Architecture](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md)). |
| **Independent from the dataset** | Dataset defines **what** external effect means for a variant; the boundary service defines **how** the application observes it at runtime. |

### Explicit exclusions

- **No real payment vendor** is selected (no PSP, acquirer, or card network integration in this document).
- **No provider implementation** is specified here—only the boundary concept for Docker composition and future adapter work.

---

## 7. Docker Runtime Model

The final standalone E2E proof should run through **reproducible containers** so operators and automation share the same topology.

### Conceptual compose stack

```text
docker compose
        ↓
database (PostgreSQL)
        ↓
scenario application
        ↓
external boundary (payment / reconciliation simulator)
        ↓
Integrax runtime (and proof harness orchestration)
```

| Component | Responsibility |
| --- | --- |
| **Compose orchestration** | Single entry point to start/stop lab dependencies, wire networks, and inject configuration—without baking vendor choices into the dataset. |
| **Database service** | Host for materialized commerce state produced by the PostgreSQL provisioner. |
| **Scenario application** | Business workflow and contract-backed tools under test. |
| **External boundary** | Isolated stand-in for payment SoR / reconciliation endpoints per variant. |
| **Integrax runtime** | Executes platform capabilities and observability spine for the proof. |

### Explicit non-scope

This document does **not** author `Dockerfile`s, `docker-compose.yml`, health-check scripts, or image pins. Those are implementation tasks that must follow this model.

---

## 8. Vendor Neutrality Preservation

Choosing **PostgreSQL** for the v1 lab does **not** break vendor neutrality of the qualification inputs.

Neutrality is preserved by **separation of concerns**:

```text
Dataset (business truth)
        ↓
Provisioning Contract (stable capability surface)
        ↓
PostgreSQL Provisioner (one replaceable implementation)
```

| Principle | How PostgreSQL fits |
| --- | --- |
| **Dataset unchanged** | JSON manifest and shared/variant slices remain free of PostgreSQL-specific identifiers ([Scenario Data Architecture § 3](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#3-vendor-neutrality-principles)). |
| **Contract stability** | Application and proof harness depend on provisioning ports, not on SQL dialect ([Data Provisioning Architecture § 4](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md#4-vendor-neutrality-model)). |
| **Implementation swap** | In-memory reference, another RDBMS, or hybrid simulators can implement the same contract for different labs without revising scenario meaning. |

PostgreSQL is an **execution-environment decision**, not a qualification requirement encoded in business truth.

---

## 9. Ownership Model

Consistent with [Proof Architecture Design § 6](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#6-ownership-model), [Scenario Data Architecture § 8](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md#8-ownership-model), and [Data Provisioning Architecture § 3](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md#3-responsibilities):

| Owner | Owns |
| --- | --- |
| **Dataset** | Business truth, variants, expected outcomes for evaluator alignment. |
| **Provisioning** | Materialization lifecycle—how logical truth becomes available in target sources. |
| **Infrastructure** | Runtime environment: containers, PostgreSQL service, network topology, lab secrets/configuration (non-production). |
| **Application** | Business workflow, ports, tools, and application knowledge surfaces—not canonical dataset file graph. |
| **Platform** | Integrax capability execution, ERL processing on the observability spine, harness integration boundaries. |

Infrastructure ownership **does not** own business rules or variant semantics; it hosts state produced under provisioning and consumed through contracts.

---

## 10. Non-Goals

This document explicitly does **not** define:

| Excluded topic | Reason |
| --- | --- |
| **Production deployment** | Qualification lab only; no SLA, scaling, or multi-region design. |
| **Cloud architecture** | No AWS/GCP/Azure topology, managed services selection, or IAM models. |
| **Database schema** | Deferred to implementation behind contracts (§ 3). |
| **Payment provider integration** | External boundary is conceptual; no PSP SDK or live payments (§ 6). |
| **Observability stack** | Metrics/tracing/log backends for production are out of scope; proof uses scenario/platform observability spine per proof architecture. |
| **Kubernetes** | v1 targets compose-based reproducibility; orchestration platforms are not required for qualification value. |

---

## 11. Future Implementation Preparation

Ordered preparation steps for implementers (documentation pointers only):

1. **Docker compose setup** — Implement compose stack per § 7; pin images for reproducibility; separate config from dataset.
2. **PostgreSQL provisioning adapter** — Implement `ScenarioProvisioningPort` (or successor contract) to seed and tear down variant materialization in PostgreSQL; keep reference in-memory provisioner for fast contract tests.
3. **Scenario application** — Wire application tools to contract-backed reads/writes against materialized state and external boundary endpoints.
4. **External boundary** — Deploy replaceable payment/reconciliation simulator service; map variant slices to simulator behavior without altering dataset JSON.
5. **Runtime integration** — Connect proof harness lifecycle (variant selection → provision → execute → evaluate → cleanup) to Integrax runtime and evidence projection per [Proof Architecture Design § 10](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md#10-implementation-preparation).

Each step must preserve **dependency inversion**, **modular boundaries**, and **plugin readiness** from prior architecture documents.

---

## Architecture requirements (summary)

| Requirement | Satisfied by |
| --- | --- |
| Dependency inversion | Dataset → contract → vendor implementation → app/runtime (§ 2, § 8). |
| Modularity | Distinct Docker services and ownership table (§ 7, § 9). |
| Plugin readiness | PostgreSQL provisioner as one plugin; reference provisioner retained (§ 8, § 11). |
| Vendor neutrality | Business truth isolated from lab engine choice (§ 1, § 8). |
| Enterprise boundaries | External payment reality isolated and replaceable (§ 6). |

---

## References

| Document | Role |
| --- | --- |
| [Scenario Specification](../SCENARIO_SPEC.md) | Normative variants, APPLICATION vs PROOF. |
| [ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md](ERL_QUAL_004_PROOF_ARCHITECTURE_DESIGN.md) | E2E flow, components, evidence, implementation preparation. |
| [ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md](ERL_QUAL_004_SCENARIO_DATA_ARCHITECTURE.md) | Logical data model, vendor neutrality, dataset ownership. |
| [ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md](ERL_QUAL_004_DATA_PROVISIONING_ARCHITECTURE.md) | Provisioning boundary, lifecycle, interchangeable implementations. |
| [dataset/README.md](../dataset/README.md) | Dataset package layout. |
| [README](../README.md) | Public scenario summary. |

---

## Document Validation (quality gate)

| Check | Result |
| --- | --- |
| Aligns with Proof, Scenario Data, and Data Provisioning architecture documents | **Pass** |
| Decisions justified (PostgreSQL yes; vector and streaming deferred) | **Pass** |
| No unnecessary technology added for v1 | **Pass** |
| No vendor lock-in on business truth (dataset + contract separation) | **Pass** |
| No implementation leakage (no Docker files, schema, migrations, adapters) | **Pass** |
| Scope limited to documentation under scenario `docs/` (+ README link) | **Pass** |

---

**Architecture status:** Vendor infrastructure decisions documented for E2E proof implementation. Containers, PostgreSQL adapter, and compose files remain **not implemented**.
