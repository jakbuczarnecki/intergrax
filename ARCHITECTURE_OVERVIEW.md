<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Architecture Overview

A concise view of how Intergrax separates product workflows, orchestration, agent decisions, governed execution, and evidence.

> [!NOTE]
> This document explains **responsibility boundaries** and **system flow**. It is **not** the complete architecture canon or a production-readiness claim. Deep technical readers should use the [Technical Documentation Map](docs/DOCUMENTATION_MAP.md).

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="docs/assets/public/intergrax-hero-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="docs/assets/public/intergrax-hero-light.svg"
  >
  <img
    alt="Intergrax connects specialized agent applications with reusable policy, knowledge, evidence, integration and execution foundations."
    src="docs/assets/public/intergrax-hero-light.svg"
  >
</picture>

---

## At a glance

| Part | Owns | Does not own |
| ---- | ---- | ------------ |
| **Specialized application** | User workflow, product context, product permissions, and acceptance criteria | Governed execution internals, provider wiring, or evidence substrate |
| **Orchestration** | Coordinating work across agents and steps | Domain decisions or product UX |
| **Agent** | Domain decisions inside a bounded session | Policy enforcement, tool I/O substrate, or cross-product hosting |
| **Governed execution harness** | Execution cycle, policy, budgets, trace, and evidence collection | Product-specific business rules outside shared boundaries |
| **Knowledge, tools and model systems** | Retrieval, integrations, and model access behind shared boundaries | The product workflow or end-user experience |
| **Trace and evidence** | Receipts, observability, and reviewable execution records | Product certification or commercial validation |

---

## The system flow

```mermaid
flowchart LR
    U[User or system request] --> APP[Specialized application]
    APP --> O[Orchestration]
    O --> A[Agent decision]
    A --> H[Governed execution]
    H --> K[Knowledge and memory]
    H --> T[Tools and integrations]
    H --> M[Models]
    H --> E[Trace and evidence]
    E --> APP
```

A specialized application hosts the concrete workflow. Orchestration coordinates work; agents make domain decisions; the harness governs execution, policy, and evidence around knowledge, tools, and models. Trace and evidence return to the application so reviewers can inspect what happened without inferring control flow from ad hoc logs.

This flow describes responsibility boundaries, not every deployment topology or internal component name.

---

## Responsibility boundaries

The architecture uses five clear responsibility boundaries:

- **Applications** own the concrete product environment.
- **Orchestration** coordinates work.
- **Agents** make domain decisions.
- **The harness** controls execution, policy, and evidence.
- **External systems** provide models, tools, and organizational knowledge.

The separation matters because product-specific rules stay in the product; agents do not become hidden operating systems; policy and evidence are execution concerns, not optional decorations; and integrations remain replaceable behind shared boundaries.

---

## Shared foundation versus product ownership

| Intergrax reusable foundation | Product application |
| ----------------------------- | ------------------- |
| Governed execution | User workflow |
| Policy and HITL | Identity and tenant context |
| Trace and evidence | Product permissions |
| Knowledge, RAG, and memory | Product configuration |
| Tool and integration boundaries | User experience |
| Model portability | Product acceptance criteria |
| Runtime verification | Product-specific business rules |

Products compose platform capabilities; they do not reimplement the harness for each new workflow.

---

## How LKW uses the architecture

Local Knowledge Workspace (LKW) is the **Primary product proof** at **Backend Product Alpha / MVP** with **PARTIAL** public proof status.

LKW supplies the workspace workflow. Intergrax supplies reusable ingest, knowledge, execution, policy, evidence, and hosting foundations.

```mermaid
flowchart LR
    S[Local files or approved Web URLs]
    S --> I[Background ingest]
    I --> K[Persistent knowledge index]
    K --> Q[Ask over indexed knowledge]
    Q --> R[Grounded result and evidence]
    R --> T[Trace and ProofReceipt]
```

LKW does not demonstrate finished SaaS, complete Hybrid Ask, or completed commercial validation. Bounded proof details: [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md).

---

## How Token Optimization fits

Token Optimization is a **Featured platform-capability proof** used within governed execution — a shared platform mechanism for deterministic prompt and context optimization under policy, not a generic prompt-shortening utility. Public status remains **PARTIAL**.

It operates inside the harness execution path alongside policy, protected-region validation, receipts, and observability. Detailed phase status and claim guardrails live in the owning guides — not duplicated here.

**Go deeper:** [Token Optimization guide](docs/features/token_optimization/README.md) · [Claim guardrails](docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md)

---

## Go deeper

| Topic | Document |
| ----- | -------- |
| Harness AI narrative | [INTERGRAX_HARNESS_NARRATIVE.md](docs/guides/INTERGRAX_HARNESS_NARRATIVE.md) |
| Technical documentation map | [DOCUMENTATION_MAP.md](docs/DOCUMENTATION_MAP.md) |
| Public proof dashboard | [PROOFS.md](PROOFS.md) |
| LKW product proof | [LKW_PLATFORM_PROOF.md](docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| Token Optimization | [Token Optimization guide](docs/features/token_optimization/README.md) |

Next: choose an evaluation or building path in [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md).
