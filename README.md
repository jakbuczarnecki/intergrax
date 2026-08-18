# Intergrax

Intergrax helps teams build specialized AI applications that use controlled
knowledge and tools while keeping access, actions, and evidence reviewable.

It is a reusable governed foundation—an application operating layer around
execution boundaries—so product teams do not rebuild policy, approvals,
integrations, recovery, and evidence mechanisms for every workflow.

**Local Knowledge Workspace (LKW)** is a private, governed AI knowledge workspace
with **Slack as its primary daily-use conversational interface** for LKW 1.0.
Add approved sources, ask over indexed knowledge from Slack or other supported
clients, and inspect grounded answers with source references and persisted
evidence. Reusable LKW HTTP and application APIs remain the backend boundary —
Slack is the familiar work-surface direction, not the only client. A bounded
Slack DM Ask path is live-verified today; the broader Slack-first daily-use
experience remains under productization.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Source-available](https://img.shields.io/badge/source--available-evaluation-6c5ce7.svg)](LICENSE)
[![Active R&D](https://img.shields.io/badge/active-R%26D-0969da.svg)](#license-and-collaboration)
[![Documented proof paths](https://img.shields.io/badge/documented-proof%20paths-2ea44e.svg)](docs/project/proofs/PROOFS.md)

**[See LKW](applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md)** · **[Run LKW locally](applications/local_workspace_application/docs/product/QUICKSTART.md)** · **[Why Intergrax](docs/project/overview/WHY_INTERGRAX.md)**

> Intergrax is **source-available** and under **active R&D**. LKW is a
> **Backend Product Alpha / MVP**. **Real-user validation** and **commercial
> validation** are incomplete.

---

<a id="start-here"></a>
## Choose your path

| If you are… | Start here | What this gives you |
| --- | --- | --- |
| AI Engineer / Builder | [Builder Quick Start](docs/project/builders/BUILDER_QUICKSTART.md) | Build a runnable agent + application stack |
| Architect / Principal Engineer | [Architecture Overview](docs/project/architecture/ARCHITECTURE_OVERVIEW.md) | Understand responsibilities, boundaries, and system design |
| CTO / Engineering Leader | [Use Cases](docs/project/overview/USE_CASES.md) | Decide whether Intergrax fits a concrete workflow |
| Technical Reviewer | [PROOFS](docs/project/proofs/PROOFS.md) | Inspect what is actually implemented / boundedly proven |
| Investor / Strategic Evaluator | [Why Intergrax](docs/project/overview/WHY_INTERGRAX.md) | Understand the platform thesis, LKW wedge, and open validation gates |
| Design Partner / Integrator | [Partners](docs/project/community/PARTNERS.md) | Explore a bounded evaluation or pilot around a concrete workflow |

Looking for configuration, evaluation guidance, roadmap, collaboration,
permissions, capability-specific material, or deeper technical documentation?
Explore the [Public Documentation Map](docs/project/community/PUBLIC_DOCUMENTATION_MAP.md).

Questions? See the [FAQ](docs/project/overview/FAQ.md).

---

<a id="explore-the-intergrax-platform"></a>
## Explore the Intergrax Platform

**What is Intergrax built from?** The platform is organized into human-readable
areas below. Each area links to canonical **domain architecture** documents —
the public entry points for *what* a subsystem should do. For cross-layer
capabilities, see [multi-layer feature architecture](#platform-capabilities-and-directions).
For deep engineering registers, use architecture **satellites** (on demand via the
[Technical Documentation Map](docs/project/technical/DOCUMENTATION_MAP.md)) —
not as a first-contact route.

| Platform area | What it provides | Explore |
| --- | --- | --- |
| **Runtime & Orchestration** | Unified execution, workflow orchestration, and Nexus execution paths | [Unified Execution Runtime](docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md) · [Orchestration](docs/project/architecture/ORCHESTRATION.md) · [Nexus Execution Flow](docs/project/architecture/NEXUS_EXECUTION_FLOW.md) |
| **Agents & Reasoning** | Agent contracts, reasoning and cognition, critic verification, adaptive harness intelligence | [Agent Contracts & Assembly](docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [Reasoning & Cognition](docs/project/architecture/REASONING_AND_COGNITION.md) · [Critic Verification](docs/project/architecture/CRITIC_VERIFICATION.md) · [Adaptive Harness Intelligence](docs/project/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| **Knowledge & Retrieval** | Retrieval, grounding, and knowledge-source integration boundaries | [RAG](docs/project/architecture/RAG.md) · [Knowledge Source Integrations](docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) |
| **Memory & Context** | Durable memory, context engineering, and unified context lifecycle | [Memory](docs/project/architecture/MEMORY.md) · [Context Engineering](docs/project/architecture/CONTEXT_ENGINEERING.md) · [Unified Context Lifecycle](docs/project/architecture/UNIFIED_CONTEXT_LIFECYCLE.md) |
| **Tools, Skills & Integrations** | Tools, skills, integrations, LLM adapters, and code-craft surfaces | [Tools](docs/project/architecture/TOOLS.md) · [Skills](docs/project/architecture/SKILLS.md) · [Integrations](docs/project/architecture/INTEGRATIONS.md) · [LLM Adapters](docs/project/architecture/LLM_ADAPTERS.md) · [Code Craft](docs/project/architecture/CODE_CRAFT.md) |
| **Governance, HITL & Reliability** | Policy and approval enforcement, failure handling, and human-in-the-loop | [Governed Execution](docs/project/architecture/GOVERNED_EXECUTION.md) · [Reliability / Failure / HITL](docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md) |
| **Observability & Evidence** | Runtime observability, proof receipts, and reviewable execution records | [Observability](docs/project/architecture/OBSERVABILITY.md) · [Proof Receipts](docs/project/architecture/PROOF_RECEIPTS.md) |
| **Extensibility & Ecosystem** | Governed plugins, agent distribution, marketplace direction, multiplayer collaboration | [Platform Plugins](docs/project/architecture/PLATFORM_PLUGINS.md) · [Agent Distribution](docs/project/architecture/AGENT_DISTRIBUTION.md) · [Multiplayer AI](docs/project/capabilities/architecture/MULTIPLAYER_AI.md) · [Agent Marketplace](docs/project/overview/AGENT_MARKETPLACE.md) |
| **Application Platform** | Tier-3 application environment and application hosting | [Tier-3 Application Environment](docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md) · [Application Hosting](docs/project/architecture/APPLICATION_HOSTING.md) |
| **Platform Foundations & Scale** | Core platform foundation, elastic capacity, modality, and developer experience | [Platform Foundation](docs/project/architecture/PLATFORM_FOUNDATION.md) · [Elastic Capacity & Scaling](docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [Modality](docs/project/architecture/MODALITY.md) · [Experimentation & DX](docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |

Full domain index (24 architecture ↔ plan pairs):
[runtime architecture hub](docs/project/architecture/intergrax_runtime_architecture.md).
Project-level mental model:
[Architecture Overview](docs/project/architecture/ARCHITECTURE_OVERVIEW.md).

### How documentation is organized

| Layer | Answers | Start here |
| --- | --- | --- |
| **First contact** | What is Intergrax, choose a path, platform map | This README |
| **Intent routing** | I want to try / evaluate / build / review | [Public Documentation Map](docs/project/community/PUBLIC_DOCUMENTATION_MAP.md) |
| **Architecture mental model** | Responsibility boundaries and system flow | [Architecture Overview](docs/project/architecture/ARCHITECTURE_OVERVIEW.md) |
| **Domain architecture** | What a platform area should do | `docs/project/architecture/<DOMAIN>.md` |
| **Feature architecture** | Cross-layer capabilities coordinating domains | `docs/project/capabilities/architecture/<FEATURE>.md` |
| **Satellites** | Extended engineering depth (on demand) | Indexed from domain or feature hubs — not first-contact |
| **Technical guides** | How to configure, build, extend, or operate | [Technical guides](docs/project/technical/guides/README.md) |
| **Plans / ADR / proofs** | Implementation status, decisions, bounded evidence | [PROOFS](docs/project/proofs/PROOFS.md) · [Technical Documentation Map](docs/project/technical/DOCUMENTATION_MAP.md) |

---

## Local Knowledge Workspace (LKW)

### Product workflow

```text
approved knowledge
→ LKW workspace
→ ask from Slack / supported client
→ grounded answer
→ sources / persisted evidence
```

**Product Quick Start** is the easiest supported local executable proof path:
indexed Ask V1 over a bundled sample document (`AURORA-17` is the expected
success marker). It does **not** require Slack setup.

**Slack** is the primary daily-use conversational interface direction for LKW
1.0. A bounded DM Ask path is live-verified today; broader Slack workspace,
source-management, and daily-use flows remain under productization.

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="applications/local_workspace_application/docs/assets/lkw-grounded-result-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="applications/local_workspace_application/docs/assets/lkw-grounded-result-light.svg"
  >
  <img
    alt="LKW quickstart flow showing the approved sample file lkw_product_quickstart.txt, the question “What is the project codename?”, the grounded answer “AURORA-17”, its source reference, and persisted Ask-run verification."
    src="applications/local_workspace_application/docs/assets/lkw-grounded-result-light.svg"
  >
</picture>

This visual represents the documented Quick Start, not a finished UI
screenshot; dynamic workspace and Ask-run IDs are omitted.

### What is boundedly proven today

LKW is the **Primary Product Proof**, classified as **Backend Product Alpha /
MVP**, with **PARTIAL** status.

**Primary executable product path:** [Product Quick Start](applications/local_workspace_application/docs/product/QUICKSTART.md)
exercises **indexed Ask V1** — not Hybrid Ask certification, Trusted Ask
durability, or Core Platform Proof.

**Separate bounded technical evidence:** indexed Hybrid Ask branch
(`LKW-HYBRID-ASK-INDEXED`) is a **real application code path** for indexed retrieval.
Hybrid Ask combining indexed and authorized live evidence is **not yet proven**.

**Separate live durability evidence:** [Trusted Ask](applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md#trusted-ask-workspace-mvp-2)
(`LKW-ASK-WORKSPACE-LIVE`) verifies completed grounded Ask outcomes across
restart without resync/reindex.

**Not yet proven:** complete live-provider access, finished end-user packaging,
**real-user validation**, and **commercial validation**. See
[LKW Platform Proof](applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) and
[PROOFS](docs/project/proofs/PROOFS.md).

## Try LKW

One supported command takes you from the repository root to a grounded answer
with a source citation over indexed knowledge — the canonical **Product Quick
Start** path, separate from Slack DM setup. Detailed prerequisites and
troubleshooting live in the [LKW Quick Start](applications/local_workspace_application/docs/product/QUICKSTART.md).

**Windows:**

```bat
applications\local_workspace_application\scripts\run-lkw-product-quickstart-windows.bat
```

**Linux:**

```sh
./applications/local_workspace_application/scripts/run-lkw-product-quickstart-linux.sh
```

**macOS:**

```sh
./applications/local_workspace_application/scripts/run-lkw-product-quickstart-macos.sh
```

**Expected answer marker:** `AURORA-17` · **Expected source file:** `lkw_product_quickstart.txt`

First run may download Docker images and configured models when Ollama is the
selected provider; duration depends on your environment.

<a id="try-lkw"></a>

### LKW routes

| Route | Purpose |
| --- | --- |
| [Product Tour](applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) | Understand what the product experience looks like |
| [Quick Start](applications/local_workspace_application/docs/product/QUICKSTART.md) | Run the canonical product path |
| [Core Platform Proof](applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) | Verify bounded infrastructure/platform behavior |

**Core Platform Proof** is separate from Product Quick Start and Trusted Ask: a
platform-level bounded proof covering startup/readiness, durable knowledge and
execution, background processing, persisted reviewable evidence, hosting/recovery,
and watched-folder indexing — not production readiness, commercial validation,
or all-provider certification.

**Proof families:** Product evaluation (`LKW-PRODUCT-QUICKSTART-*`), Core platform (`LKW-BACKGROUND-TASK`, `LKW-HOSTING`, `LKW-FILE-WATCHER`), Indexed Hybrid Ask (`LKW-HYBRID-ASK-INDEXED`), Trusted Ask (`LKW-ASK-WORKSPACE-LIVE`).

---

## Why this matters

Building an impressive AI demo is easier than operating a controlled AI
application that a team can review and trust. Teams repeatedly rebuild
knowledge access, policy, integrations, approvals, and evidence foundations
around each product.

Intergrax centralizes reusable mechanisms so product teams can focus on the
specialized workflow. Read [Why Intergrax](docs/project/overview/WHY_INTERGRAX.md)
for the category, problem, and fit.

## Responsibility model

The root-level model is about responsibility, not a mandatory execution
sequence. A request uses only the configured resources its product context
selects.

| Responsibility | What it owns |
| --- | --- |
| **Specialized product application** | Product workflow, UX, business semantics, permissions, and acceptance |
| **Intergrax** | Reusable application operating layer for policy and approval boundaries, controlled context, governed execution, recovery, observability, and evidence / provenance |
| **Model / agent** | Reasoning, inference, and decision generation within supplied context and governed boundaries |
| **Knowledge / tools / integrations / models** | Selected resources behind configured access and effect boundaries |
| **Evidence / provenance** | Reviewable receipts, traces, and records produced during execution |

See the [Architecture Overview](docs/project/architecture/ARCHITECTURE_OVERVIEW.md)
for the complete responsibility model.

## AI execution should not be a black box

Meaningful AI execution should be reconstructable, reviewable, and attributable.
Intergrax is designed so important actions do not disappear inside an opaque agent loop.

```text
request → context → agent / plan / decision → policy / approval
       → model / RAG / tool → validation → result → evidence
                              ↓
                 reviewable execution record
```

A governed run can leave correlated runtime events, typed [`DecisionRecord`](docs/project/architecture/REASONING_AND_COGNITION.md) artifacts, and structured [`ProofReceipt`](docs/project/architecture/PROOF_RECEIPTS.md) evidence.
This is execution-level explainability, not hidden model reasoning.
Universal every-path production observability is not claimed.

[Observability](docs/project/architecture/OBSERVABILITY.md) ·
[Reasoning / DecisionRecord](docs/project/architecture/REASONING_AND_COGNITION.md) ·
[Proof Receipts](docs/project/architecture/PROOF_RECEIPTS.md)

**Runnable evidence:** Inspect the current bounded LKW observability proof, including independently inspectable Elasticsearch/Kibana records, controlled Sentry problem signals, and persisted execution evidence.
[LKW bounded observability proof](applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) · [Controlled Sentry proof](applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md)

## What exists today

Status is capability-specific; implementation is not blanket proof of the whole
platform.

| Area | Role | Current status |
| --- | --- | --- |
| **LKW** | Primary Product Proof | **PARTIAL — Backend Product Alpha / MVP** |
| **Other reusable foundations** | Supporting evidence | Varies by capability; inspect [PROOFS](docs/project/proofs/PROOFS.md) |

Platform capability maturity is summarized in
[Platform capabilities and directions](#platform-capabilities-and-directions) below.

## Platform capabilities and directions

Compact index of strategic platform capabilities. Status is bounded and
capability-specific; see linked architecture and proof routes for detail.

| Capability / direction | What it adds | Current maturity | Explore |
| --- | --- | --- | --- |
| **Governed Execution** | Reusable policy and approval enforcement around agent decisions, tool/action boundaries and meaningful side effects, with canonical HITL and plugin-extensible policy rules | **IMPLEMENTED CORE — coverage / qualification ongoing** — complete platform-wide governance and production qualification **not established** | [Governed Execution](docs/project/architecture/GOVERNED_EXECUTION.md) |
| **Observability & Auditability** | Shared observability spine for reconstructable, reviewable governed execution — runtime events, [`DecisionRecord`](docs/project/architecture/REASONING_AND_COGNITION.md) artifacts, [`ProofReceipt`](docs/project/architecture/PROOF_RECEIPTS.md) evidence; execution-level explainability, not hidden chain-of-thought | **IMPLEMENTED CORE + BOUNDED PROOF** — universal every-path production observability **not claimed** | [Observability](docs/project/architecture/OBSERVABILITY.md) · [LKW bounded observability proof](applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) · [Controlled Sentry proof](applications/local_workspace_application/docs/SENTRY_OBSERVABILITY.md) |
| **Token Optimization** | Featured platform-capability proof — policy-governed context and prompt optimization with receipts, fallback, and bounded offline proof | **PARTIAL — bounded** — universal savings and production-proven savings **not established** | [Token Optimization guide](docs/project/capabilities/token_optimization/README.md) · [Claim guardrails](docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md) |
| **Multiplayer AI** | Governed multi-principal collaboration among humans, agents, services, and external agents | **Architecture / roadmap stage** — runtime proof **not yet established** | [Multiplayer AI architecture](docs/project/capabilities/architecture/MULTIPLAYER_AI.md) |
| **Platform Extensibility** | Governed extension/package model across domain-owned contracts | **Canonical architecture frozen** — implementation stages planned; complete third-party install-to-runtime E2E proof **not yet established** | [Platform Plugins](docs/project/architecture/PLATFORM_PLUGINS.md) |
| **Agent Marketplace** | Future ecosystem layer — discovery and distribution over governed Agent Distribution / Platform Extensibility | **FUTURE PRODUCT — NOT SHIPPED TODAY** | [Agent Marketplace concept](docs/project/overview/AGENT_MARKETPLACE.md) |

## License and collaboration

Intergrax is **source-available** under the Intergrax Evaluation and
Collaboration License 1.0. You may clone, install, run, test, and modify the
repository locally for **non-production evaluation**, subject to the
[LICENSE](LICENSE).

Feedback, contribution, permission, and pilot routes are described in
[Collaboration](docs/project/community/COLLABORATION.md) and
[Partners](docs/project/community/PARTNERS.md). Production use, commercial use,
hosting, and redistribution require **explicit written permission or agreement**
under the legally authoritative [LICENSE](LICENSE).
