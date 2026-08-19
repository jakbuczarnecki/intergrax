<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Alternatives and Trade-offs

This page answers: **When should I choose another modern stack, and when might Intergrax be worth evaluating?**

It is **not** a feature matrix, competitor attack page, market ranking, performance comparison, or pricing comparison.

Modern alternatives overlap and may be composed. Intergrax is **not** universally superior. The comparison is about **architectural responsibility** and **integration burden** — which layer owns product semantics, enforcement, consequential effects, canonical history, recovery, and evidence — not about whether a listed capability exists somewhere in a stack.

External product capabilities were verified against vendor primary documentation on **2026-08-18**. Intergrax claims remain bounded by [PROOFS.md](../proofs/PROOFS.md). Category-level comparison (no vendor names): [Where Intergrax fits](WHY_INTERGRAX.md#where-intergrax-fits).

---

## How to read each section

Each alternative follows the same structure. Vendor capability facts come from primary sources; adopter responsibilities are **decision questions your team must settle** — not implied feature gaps.

- **Best fit / strengths** — what the stack is designed to optimize for (vendor capability fact)
- **Choose it when…** — honest recommendation cases
- **What it already solves** — capability facts from primary sources (not an Intergrax gap list)
- **Responsibilities / questions your team still needs to settle** — product semantics, organization-specific permissions, acceptance criteria, how multiple products share foundations, what evidence model reviewers require, and deployment/operational ownership — without implying the named stack lacks common platform facilities
- **How Intergrax approaches responsibility differently** — architectural interpretation only (different center of responsibility, not missing-feature contrast)
- **Current Intergrax evidence boundary** — what is proven today, not a competitiveness score

Common platform facilities — persistence, memory, tracing, guardrails, approvals, workflows, durability, identity, HITL, observability — are **not** treated as Intergrax differentiators by default. Modern stacks may provide them; the comparison is about **where architectural responsibility sits**, not whether a capability exists somewhere in a vendor portfolio.

---

## OpenAI Agents SDK

Primary source: [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/) (verified 2026-08-18).

### Best fit / strengths

Lightweight agent loops, tool use, handoffs, and tracing oriented around OpenAI model APIs and a fast path from prototype to working agent.

### Choose it when…

- you need a **small or simple agent application** with a fast start;
- your team is already standardized on OpenAI models and SDK patterns;
- orchestration depth and cross-product operating-model consolidation are secondary to shipping one agent workflow quickly.

**OpenAI may be the better choice** for a focused agent app where the main problem is running agents well inside one product, not standardizing a multi-application operating model.

### What it already solves

Agent orchestration primitives, tool invocation, multi-agent handoffs, and integrated tracing within the SDK model — per vendor documentation.

### Responsibilities / questions your team still needs to settle

Product semantics and acceptance criteria; organization-specific permissions and policy models beyond what you choose to adopt from SDK patterns; how multiple products share foundations; what evidence model your reviewers require; deployment packaging and operational ownership for your environment.

### How Intergrax approaches responsibility differently

Intergrax targets a **shared application operating model** across multiple specialized products: explicit enforcement boundaries, structural execution identity, classified recovery, and correlated evidence — consolidated as platform responsibilities rather than reassembled per application around an agent SDK.

### Current Intergrax evidence boundary

Bounded LKW and platform-capability proofs — see [PROOFS.md](../proofs/PROOFS.md). No claim that Intergrax is faster to adopt than the Agents SDK for a single simple agent.

---

## Microsoft Agent Framework

Primary source: [Microsoft Agent Framework overview](https://learn.microsoft.com/en-us/agent-framework/overview/) (verified 2026-08-18).

### Best fit / strengths

Agent construction and orchestration aligned with Microsoft and Azure AI services, identity, and enterprise integration patterns.

### Choose it when…

- your environment is **Microsoft- or Azure-centric**;
- you want agent frameworks that fit naturally with existing Azure AI, identity, and operations tooling;
- a vendor-aligned enterprise path matters more than a source-available, self-assembled operating layer.

**Microsoft may be the natural choice** for organizations already committed to Azure AI Foundry, Entra identity, and Microsoft operational practices.

### What it already solves

Agent authoring, orchestration, and integration with Microsoft AI platform services — per vendor documentation.

### Responsibilities / questions your team still needs to settle

Domain workflow semantics and acceptance criteria; organization-specific permissions and policy models for your scenario; how multiple products share foundations; what evidence model auditors require; deployment and operational ownership across your Azure estate.

### How Intergrax approaches responsibility differently

Intergrax is not an Azure-native distribution. It consolidates governed execution, structural observability, and agent contract assembly as a **product-agnostic application operating layer** intended to span multiple specialized applications — with product teams retaining meaning and business responsibility.

### Current Intergrax evidence boundary

Active R&D with partial proofs. No enterprise-readiness or Azure-equivalence claim.

---

## LangGraph + LangSmith platform layer

Primary sources: [LangGraph documentation](https://langchain-ai.github.io/langgraph/), [LangSmith documentation](https://docs.smith.langchain.com/) (verified 2026-08-18).

### Best fit / strengths

Fine-grained **stateful graph orchestration** for agent workflows, with LangSmith providing observability, evaluation, and deployment-adjacent platform tooling in the LangChain ecosystem.

### Choose it when…

- **stateful orchestration** is your primary engineering problem;
- you want graph-based control flow, checkpoints, and LangSmith-backed tracing/evaluation as the center of gravity;
- you are comfortable owning product semantics and governance integration around the LangGraph execution model.

**LangGraph may be the better choice** when orchestration graph design — not multi-product operating-model consolidation — is the main deliverable.

### What it already solves

Graph state machines, durable checkpoints, human-in-the-loop interruption patterns, and platform observability/evaluation tooling — per vendor documentation.

### Responsibilities / questions your team still needs to settle

Product meaning and acceptance criteria; organization-specific permissions and policy models; how multiple products share foundations; what evidence model your auditors require; deployment and operational ownership around the LangGraph/LangSmith stack you operate.

### How Intergrax approaches responsibility differently

Intergrax uses its own Nexus execution model (LangGraph is optional legacy compatibility, not the center). The platform thesis is a **unified governed application layer** — policy enforcement, consequential-effect boundaries, structural execution identity, classified recovery, and evidence spine — rather than graph orchestration as the primary abstraction.

### Current Intergrax evidence boundary

Core harness-path mechanisms exist; not every boundary is publicly proof-qualified. See [PROOFS.md](../proofs/PROOFS.md).

---

## Dapr Agents / durable workflow-oriented stack

Primary sources: [Dapr Workflow](https://docs.dapr.io/developing-applications/building-blocks/workflow/), [Dapr Agents overview](https://docs.dapr.io/developing-applications/develop-agents/) (verified 2026-08-18).

### Best fit / strengths

**Distributed durable workflow** infrastructure — long-running, resumable orchestration across services with operational primitives from the Dapr runtime.

Temporal, DBOS, and Restate may appear in similar decision spaces as durable execution components; this page does not catalog every durable-workflow vendor.

### Choose it when…

- **distributed durable workflow infrastructure** is the dominant requirement;
- microservice-side orchestration and durable state management are core platform requirements;
- agent reasoning is one step inside a broader durable workflow platform you already operate.

**Dapr may be the more natural choice** when durable workflow and distributed runtime operations — not AI application operating-model consolidation — are the primary buying criteria.

### What it already solves

Durable workflow execution, activity orchestration, and agent-oriented building blocks on the Dapr runtime — per vendor documentation.

### Responsibilities / questions your team still needs to settle

AI-specific governance semantics, knowledge boundaries, and agent contracts for your products; acceptance criteria for agent behavior; how multiple AI products share foundations; what evidence model reviewers require; product-layer UX and operational ownership composed with the Dapr runtime you run.

### How Intergrax approaches responsibility differently

Intergrax treats AI application **governance, agent contracts, consequential execution, recovery taxonomy, and canonical execution history** as first-class platform concerns for specialized AI products — consolidated in one application operating layer — while Dapr's documented scope centers durable distributed workflow and runtime building blocks.

### Current Intergrax evidence boundary

Durable operator-queue and production chaos claims are **not** made. Reliability mechanisms exist on bounded harness paths — see architecture docs and [PROOFS.md](../proofs/PROOFS.md).

---

## Google ADK / Agents CLI ecosystem

Primary sources: [Google Agent Development Kit documentation](https://google.github.io/adk-docs/), [Agents CLI — Getting started](https://google.github.io/agents-cli/guide/getting-started/), [Agents CLI — Quickstart tutorial](https://google.github.io/agents-cli/guide/quickstart-tutorial/), [Agents CLI reference](https://google.github.io/agents-cli/cli/) (verified 2026-08-18).

### Best fit / strengths

Agent authoring and tool use via ADK; a documented **create → evaluate → deploy → observe** lifecycle via Agents CLI — per Google's Agents CLI guides and CLI reference.

### Choose it when…

- you want Google's documented agent development, evaluation, deployment, and observability workflow as the center of gravity;
- that Google Cloud–native lifecycle matters more than a self-hosted application operating layer;
- you are building inside Google's agent platform rather than standardizing your own multi-product harness.

**Google provides a documented integrated create/evaluate/deploy/observe lifecycle and may be the more natural choice when that Google Cloud–native lifecycle is the primary requirement.**

### What it already solves

ADK: agent definition and tool integration — per ADK documentation. Agents CLI: project scaffolding, evaluation, deployment, and observability workflows along the documented lifecycle — per Agents CLI getting-started, quickstart tutorial, and CLI reference.

### Responsibilities / questions your team still needs to settle

How multiple products share foundations; self-hosting posture and license fit for your organization; organization-specific permissions and acceptance criteria; what evidence model reviewers require beyond the workflows you adopt from Google's tooling; deployment and operational ownership across your Google Cloud estate.

### How Intergrax approaches responsibility differently

Intergrax is source-available and centers a **governed multi-application operating model** under your product team's control — with explicit separation between product meaning and platform enforcement across specialized applications.

### Current Intergrax evidence boundary

No claim of equivalent Google Cloud integration depth. See [PROOFS.md](../proofs/PROOFS.md) for what is demonstrated today.

---

## Custom best-of-breed stack

Typical composition: agent framework + policy layer + durable workflow engine + telemetry stack + custom identity/application layer.

### Best fit / strengths

Maximum control and the ability to select best-in-class components for each concern.

### Choose it when…

- you have a **strong internal platform team** with capacity to integrate and maintain multiple systems;
- your requirements map cleanly to separable products (orchestration, policy, durability, observability);
- you prefer composing established components over adopting a consolidated application operating layer.

**A strong internal platform team may reasonably prefer a custom best-of-breed stack** when integration cost is acceptable and you want explicit ownership of every layer.

### What it already solves

Each component solves its own primary concern — orchestration, policy, durability, tracing, identity — when integrated deliberately.

### Responsibilities / questions your team still needs to settle

Integration contracts between components; consistent execution identity and recovery taxonomy across the stack; correlated governance evidence; how multiple AI products share foundations without diverging; deployment and operational ownership for each component and the glue layer.

### How Intergrax approaches responsibility differently

Intergrax deliberately **consolidates** product semantics/enforcement separation, governance boundaries, consequential execution, structural identity, classified recovery, and evidence into one shared operating model — reducing the integration surface between separately purchased or built components.

### Current Intergrax evidence boundary

Consolidation is an architectural thesis, not a measured integration-cost reduction. Cross-product reuse remains unproven — see below.

---

## When to evaluate Intergrax

Evaluate Intergrax when the main problem is not merely **"how do we run an agent?"** but **"how do we keep multiple specialized AI applications under a shared, auditable application operating model for product-owned semantics, governed execution, consequential effects, recovery, history, and evidence?"**

That is an architecture thesis — not measured superiority, delivery speed, safety, or cost. Cross-product reuse/compounding value remains a **hypothesis** until measured with additional products and external evidence.

## What to read next

| Route | Use it for |
| ----- | ---------- |
| [Where Intergrax fits](WHY_INTERGRAX.md#where-intergrax-fits) | Category-level responsibility map (no vendor names) |
| [How Intergrax approaches responsibility](WHY_INTERGRAX.md#how-intergrax-approaches-responsibility) | Public differentiation spine |
| [Use Cases](USE_CASES.md) | Concrete workflow fit |
| [PROOFS.md](../proofs/PROOFS.md) | Current evidence and claim limits |
| [Governed Execution](../architecture/GOVERNED_EXECUTION.md) | Policy and enforcement architecture |
| [Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md) | Technical boundary overview |
