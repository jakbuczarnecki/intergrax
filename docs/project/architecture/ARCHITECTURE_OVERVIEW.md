<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Architecture Overview

How Intergrax separates the specialized product application, its application operating layer, model or agent behavior, governed access to knowledge and tools, and reviewable evidence.

> [!NOTE]
> Intergrax is source-available and in active R&D. This overview explains **responsibility boundaries** and **governed execution**; it is not the complete architecture canon or a production-readiness, security-certification, or commercial-validation claim. Use the [Technical Documentation Map](../technical/DOCUMENTATION_MAP.md) for deep implementation architecture.

Primary audience: external architects, Principal or Staff engineers, CTOs, and technical evaluators comparing application operating boundaries.

---

## At a glance

| Boundary | Responsibility | Ownership limit |
| -------- | -------------- | --------------- |
| **Specialized product application** | Domain workflow, UX, product semantics, required identity and permissions, business acceptance, and deployment choices | Defines the business rule and product outcome; does not need to rebuild shared execution controls |
| **Intergrax application operating layer** | Policy and approval mechanisms, context and knowledge boundaries, governed execution, tool and integration boundaries, runtime controls, recovery, observability, and evidence/provenance | Provides reusable enforcement mechanisms; does not decide the product's business permissions or acceptance criteria |
| **Agent and model behavior** | Reasoning, inference, decision generation, and agent-specific domain behavior | Operates within the supplied context and governed execution; does not own policy, permissions, or evidence |
| **Knowledge, tools, integrations, and model systems** | Source data, remote services, business systems, tool effects, and model access | Are selected behind configured boundaries; they do not own the product workflow or end-user experience |
| **Evidence and provenance** | Receipts, traces, provenance, and records for review, debugging, and governance | Is produced during execution; it does not certify production readiness, security, or commercial validation |

## The operating model

```mermaid
flowchart TB
    APP[Specialized product application]

    subgraph IX[Intergrax application operating layer]
        POLICY[Policy and approval boundaries]
        CONTEXT[Context and knowledge boundaries]
        EXEC[Governed execution]
        EVIDENCE[Evidence and provenance]
    end

    AGENT[Agent and model behavior]
    KNOWLEDGE[Knowledge sources]
    TOOLS[Tools and integrations]
    MODELS[Model systems]
    REVIEW[Review, debugging, and governance]

    APP --> IX
    IX -->|governed interaction| AGENT
    IX -->|selected source| KNOWLEDGE
    IX -->|authorized integration| TOOLS
    IX -->|selected model| MODELS
    EVIDENCE --> APP
    EVIDENCE --> REVIEW
```

Intergrax surrounds the application's execution boundaries. A request may use only the model, knowledge source, or tool that its product context and configured controls select; the diagram does not mean every request invokes every resource. Agent or model behavior supplies reasoning and decisions, while the operating layer bounds how that behavior receives context, accesses resources, produces effects, and leaves evidence.

**Harness AI** is useful category shorthand for this operating or execution layer around model and agent behavior. Intergrax is therefore not merely a model provider, an agent reasoning framework, or the product's UX and business workflow.

## Who defines policy and who enforces it?

- **The product team owns** what the business rule means, who should have permission, which actions require approval, the required identity and tenant context, and whether the product outcome is acceptable.
- **Intergrax provides** reusable mechanisms to propagate that identity and context, apply configured policy and approval boundaries, constrain execution and tool access, recover and observe runtime behavior, and record evidence.
- **Agent and model behavior does not replace either responsibility.** It can propose a decision or action, but it does not grant business permission or bypass the governed boundary.

This division lets products keep their domain accountability while sharing an application operating layer instead of rebuilding controls around each workflow.

## Request and evidence flow

```mermaid
flowchart LR
    REQUEST[Request] --> INTENT[Product intent and context]
    INTENT --> EXECUTION[Governed execution]
    EXECUTION -->|selected resources only| RESOURCE[Model, knowledge, or tool interaction]
    RESOURCE --> RESULT[Result]
    RESULT --> RECEIPT[Evidence and provenance]
    RECEIPT --> RESPONSE[Product response or review]
```

The lifecycle is conceptual: the operating layer selects the resources needed for the request, then returns the result with evidence/provenance that can be inspected for debugging, review, and governance. Evidence is a first-class execution output, not an afterthought or an inference reconstructed from ad hoc logs.

## LKW as a product example

Local Knowledge Workspace (LKW) is the **Primary product proof** at **Backend Product Alpha / MVP** with **PARTIAL** public proof status.

| LKW-specific product responsibility | Shared Intergrax foundation |
| ---------------------------------- | ---------------------------- |
| Workspace workflow, approved-source choice, user-facing Ask, and product acceptance | Ingest and knowledge boundaries, governed Ask execution, evidence/provenance, and hosting/runtime mechanisms |

The accepted indexed path demonstrates bounded ingest, indexed knowledge, grounded Ask, source references, and persisted execution evidence. Mixed indexed + authorized live Hybrid Ask remains incomplete; complete live-provider access, real-user validation, and commercial validation are not established. See [LKW Platform Proof](../proofs/LKW_PLATFORM_PROOF.md) and the [PROOFS dashboard](../proofs/PROOFS.md).

## Token Optimization as a secondary capability example

Token Optimization is a **Featured platform-capability proof** with **PARTIAL** public status. It operates inside governed execution as a reusable capability for policy-governed context and prompt optimization, protected-region validation, receipts, and observability. It is not a separate product layer, and universal token savings or production-proven savings are not claimed.

Details belong in the owning [Token Optimization guide](../capabilities/token_optimization/README.md) and its [claim guardrails](../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md).

## Multiplayer AI as a strategic platform direction

Multiplayer AI extends the operating model from a single governed request and execution path toward governed collaboration among multiple principals — humans, agents, services, and eventually external agents. This is architectural direction; today's runtime does not yet complete that evolution.

The planned Multiplayer layer coordinates platform primitives for Principal, Membership, Delegation, Shared Work, Artifacts, Decisions, ContextView, Activity, and AgentDirectory. It reuses existing UCL, Context Engineering, Memory, Knowledge/RAG, Token Optimization, HITL, execution/runtime, and evidence/provenance mechanisms without relabeling them as Multiplayer.

```mermaid
flowchart TB
    PRINCIPALS[Human / Agent / Service / External Agent]
    MP[Multiplayer layer<br/>identity · authority · shared work<br/>artifacts · decisions · context views]
    GOV[Governed Intergrax execution]
    RES[knowledge · tools · models · evidence]

    PRINCIPALS --> MP
    MP --> GOV
    GOV --> RES
```

*Conceptual strategic architecture — not an implemented runtime topology.*

Details belong in the [Multiplayer AI architecture](../capabilities/architecture/MULTIPLAYER_AI.md).

## Architect review path

1. Understand this Architecture Overview as the project-level mental model.
2. Make [PROOFS](../proofs/PROOFS.md) the primary next action: check what has current bounded evidence and what remains incomplete.
3. Use the [Evaluation Guide](../builders/EVALUATION_GUIDE.md) for a bounded technical evaluation.
4. Use the [Technical Documentation Map](../technical/DOCUMENTATION_MAP.md) to route deeper implementation questions to the canonical architecture material it indexes, including the [Harness narrative](../technical/guides/INTERGRAX_HARNESS_NARRATIVE.md) and [runtime architecture hub](../architecture/intergrax_runtime_architecture.md).

The overview summarizes the architecture; the proof documents establish current evidence; the technical map routes implementation-level due diligence. None of these routes turns bounded evidence into a production, security, real-user, or commercial validation claim.
