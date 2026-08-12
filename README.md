# Intergrax

Intergrax helps teams build specialized AI applications that use controlled
knowledge and tools while keeping access, actions, and evidence reviewable.

It is a reusable governed foundation—an application operating layer around
execution boundaries—so product teams do not rebuild policy, approvals,
integrations, recovery, and evidence mechanisms for every workflow.

**Local Knowledge Workspace (LKW)** is the primary product path: add an approved
source, ask over indexed knowledge, and inspect a grounded answer with source
references and persisted evidence.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Source-available](https://img.shields.io/badge/source--available-evaluation-6c5ce7.svg)](LICENSE)
[![Active R&D](https://img.shields.io/badge/active-R%26D-0969da.svg)](#license-and-collaboration)
[![Documented proof paths](https://img.shields.io/badge/documented-proof%20paths-2ea44e.svg)](docs/project/proofs/PROOFS.md)

**[Try LKW](#try-lkw)** · [See the LKW workflow](docs/project/product/lkw/LKW_PRODUCT_TOUR.md) · [Review proof](docs/project/proofs/PROOFS.md)

> Intergrax is **source-available** and under **active R&D**. LKW is a
> **Backend Product Alpha / MVP**. **Real-user validation** and **commercial
> validation** are incomplete.

---

## Local Knowledge Workspace (LKW)

### Product workflow

```text
approved source
→ ingest / index
→ Ask
→ grounded answer
→ source / evidence
```

`AURORA-17` is the expected success marker used by the canonical LKW
quick-start proof.

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="docs/project/assets/public/lkw-grounded-result-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="docs/project/assets/public/lkw-grounded-result-light.svg"
  >
  <img
    alt="LKW quickstart flow showing the approved sample file lkw_product_quickstart.txt, the question “What is the project codename?”, the grounded answer “AURORA-17”, its source reference, and persisted Ask-run verification."
    src="docs/project/assets/public/lkw-grounded-result-light.svg"
  >
</picture>

This visual represents the documented Quick Start, not a finished UI
screenshot; dynamic workspace and Ask-run IDs are omitted.

### What is boundedly proven today

LKW is the **Primary Product Proof**, classified as **Backend Product Alpha /
MVP**, with **PARTIAL** status. Bounded proof exists for indexed knowledge
workflows. **Hybrid Ask** is the production code path used by the
proven indexed branch; the proven scope today is indexed knowledge only.
Mixed indexed + authorized-live Hybrid Ask in one answer is **not yet proven**.

**Proofs:** `LKW-CORE-PLATFORM-WINDOWS`, `LKW-CORE-PLATFORM-LINUX`, `LKW-CORE-PLATFORM-MACOS`

**Not yet proven:** Hybrid Ask combining indexed and authorized live evidence,
complete live-provider access, finished end-user packaging, **real-user validation**,
and **commercial validation**. Mixed indexed + authorized live Hybrid Ask is
**not complete**; complete live-provider access remains incomplete.

See the detailed [LKW Platform Proof](docs/project/proofs/LKW_PLATFORM_PROOF.md)
and the current [PROOFS dashboard](docs/project/proofs/PROOFS.md).

## Try LKW

One supported command takes you from the repository root to a grounded answer
with a source citation over indexed knowledge. Detailed prerequisites and
troubleshooting live in the [LKW Quick Start](docs/project/product/lkw/QUICKSTART.md).

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

First run may download Docker images and the configured local model; duration
depends on your environment and is not externally validated as a fixed target.

<a id="try-lkw"></a>

### LKW routes

| Route | Purpose |
| --- | --- |
| [Product Tour](docs/project/product/lkw/LKW_PRODUCT_TOUR.md) | Understand what the product experience looks like |
| [Quick Start](docs/project/product/lkw/QUICKSTART.md) | Run the canonical product path |
| [Core Platform Proof](docs/project/proofs/LKW_PLATFORM_PROOF.md) | Verify bounded infrastructure/platform behavior |

**Core Platform Proof** is separate from the Product Quick Start: a
platform-level bounded proof covering startup, sentry, elasticsearch,
persistence, background task, application hosting, and file watcher. It does
**not** imply production readiness, commercial validation, or all-provider
certification.

**Proofs:** `LKW-BACKGROUND-TASK`, `LKW-HOSTING`, `LKW-FILE-WATCHER`, `LKW-ASK-WORKSPACE-LIVE`

---

## Choose your path

| You want to… | Start here |
| --- | --- |
| Try the product | [LKW Quick Start](docs/project/product/lkw/QUICKSTART.md) |
| Understand the product first | [LKW Product Tour](docs/project/product/lkw/LKW_PRODUCT_TOUR.md) |
| Verify bounded platform behavior | [Core Platform Proof](docs/project/proofs/LKW_PLATFORM_PROOF.md) |
| Check whether your workflow fits | [Use Cases](docs/project/overview/USE_CASES.md) |
| Review current evidence | [PROOFS](docs/project/proofs/PROOFS.md) |
| Evaluate one claim fairly | [Evaluation Guide](docs/project/builders/EVALUATION_GUIDE.md) |
| Start building | [Builder Quick Start](docs/project/builders/BUILDER_QUICKSTART.md) |
| Plan deeper application composition | [Build With Intergrax](docs/project/builders/BUILD_WITH_INTERGRAX.md) |
| Review architecture | [Architecture Overview](docs/project/architecture/ARCHITECTURE_OVERVIEW.md) |
| Discuss a pilot or design partnership | [Partners](docs/project/community/PARTNERS.md) |
| Contribute, give feedback, or ask about permissions | [Collaboration](docs/project/community/COLLABORATION.md) |
| Perform a deep technical review | [Technical Documentation Map](docs/project/technical/DOCUMENTATION_MAP.md) |

Questions? See the [FAQ](docs/project/overview/FAQ.md). For the complete
public route map, use the [Public Documentation Map](docs/project/community/PUBLIC_DOCUMENTATION_MAP.md).
The [project documentation hub](docs/project/README.md) is the secondary
all-docs entry point. The [public roadmap](docs/project/overview/ROADMAP.md)
describes outcome direction.

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

## What exists today

Status is capability-specific; implementation is not blanket proof of the whole
platform.

| Area | Role | Current status |
| --- | --- | --- |
| **LKW** | Primary Product Proof | **PARTIAL — Backend Product Alpha / MVP** |
| **Token Optimization** | Featured platform-capability proof | **PARTIAL — bounded** |
| **Multiplayer AI** | Strategic platform capability | **Architecture / roadmap stage** — runtime proof not yet established |
| **Platform Extensibility / Plugins** | Strategic platform capability | **Canonical architecture frozen** — implementation stages planned; complete platform-level third-party E2E proof not yet established |
| **Other reusable foundations** | Supporting evidence | Varies by capability; inspect [PROOFS](docs/project/proofs/PROOFS.md) |

## Token Optimization

Token Optimization is a compact example of a reusable platform capability
below the product surface: policy-governed context and prompt optimization with
protected-region validation, receipts, fallback, and a bounded vLLM proof path.

Its current status is **PARTIAL**. Live provider-wide proof, production rollout,
final cross-provider proof, universal savings, and **production-proven savings**
are **not established**. Details belong to the [Token Optimization guide](docs/project/capabilities/token_optimization/README.md)
and its [claim guardrails](docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md).

**Proof:** `RUNTIME-TOKEN-OPTIMIZATION-OFFLINE`

## Multiplayer AI

Intergrax is extending governed execution toward governed multi-principal
collaboration among humans, agents, services, and eventually external agents.
The architectural direction covers identity and membership, delegation of
effective authority, shared work, durable collaborative artifacts, explicit
decisions, principal-scoped context, activity and provenance, and
external-agent interoperability.

Multiplayer AI is broader than one agent calling another. It governs who
participates, what authority is effective, what shared work exists, which
artifact version is authoritative, what decision was made, what context each
principal may see, and what evidence remains.

Current status is **architecture / roadmap stage**. Runtime implementation and
proof are **not yet established**. See the [Multiplayer AI architecture](docs/project/capabilities/architecture/MULTIPLAYER_AI.md).

## Platform Extensibility

Intergrax already exposes extension points across integrations, tools, skills,
RAG, Vendor Knowledge, security, policy, host composition, and other platform
domains. The [Platform Plugins](docs/project/architecture/PLATFORM_PLUGINS.md)
architecture defines how independently packaged extensions can participate
without collapsing those domain contracts into one universal plugin runtime.

The strategic goal is coordinated **package identity**, **discovery**,
**configuration**, **compatibility**, **trust**, **qualification**,
**lifecycle**, and **author experience** — while domain-owned contracts still
govern actual runtime behavior. Extend the platform without modifying its core
while preserving governed capability boundaries.

A basic plugin system answers how code can be loaded. Platform Extensibility
also must answer what capability a package contributes, how it is discovered,
whether it is compatible, how it receives configuration and dependencies, what
trust or qualification state applies, and which domain contract governs
execution.

Canonical architecture is **frozen**; platform-wide harmonization is **not
complete**; a complete third-party install-to-runtime E2E proof is **not yet
established**.

## Agent Marketplace — future ecosystem concept

**Future product concept — not shipped today.**

**Build once. Govern centrally. Install reusable AI capabilities into any
Intergrax application.**

Intergrax is building a governed distribution model for reusable Tier-2 agents
across built-in sources, local and developer sources, private enterprise
catalogs, and future public catalogs. Every source converges on the same
governed lifecycle:

```text
Discover → Trust → Install → Configure → Materialize → Activate → Route
```

Apps install capabilities. Nexus routes work. Intergrax governs execution.

> **Status — FUTURE PRODUCT:** The public marketplace experience, publisher
> portal, catalog product, billing layer, and LKW marketplace UI are **not
> shipped today**. Underlying distribution and platform capabilities — Agent
> Distribution, package verification, application binding, immutable
> materialization, RuntimeRevision activation, AgentRegistry, and Nexus
> capability routing — have mixed **AVAILABLE TODAY** / **ARCHITECTURE FROZEN**
> / **UNDER IMPLEMENTATION** maturity.

Platform Extensibility / Agent Distribution is the governed technical substrate;
Agent Marketplace is the future discovery, distribution, and ecosystem layer
built on that substrate — not an independent execution engine.

Example reusable capability patterns: Research · Legal · Project Management ·
UX Research · Private Enterprise

[Explore the Agent Marketplace concept and reference architecture →](docs/project/product/AGENT_MARKETPLACE.md)

<!-- Compatibility anchors for inbound documentation links -->
<a id="quick-start"></a>
<a id="proof-of-platform"></a>
<a id="start-here"></a>
<a id="harness-ai--the-core-idea"></a>
<a id="the-agent-model--why-architects-choose-intergrax"></a>

---

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
