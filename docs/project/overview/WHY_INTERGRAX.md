<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Why Intergrax

Intergrax exists for teams building specialized agent applications that would otherwise rebuild controlled knowledge access, policy, tools, evidence, context, approvals, recovery, and observability around every model-driven workflow. It is a reusable governed foundation for those application operating boundaries; the project is source-available and in active R&D, with bounded current evidence rather than a universal production-readiness claim.

> [!NOTE]
> This page explains the problem category and evaluation boundary. See [PROOFS.md](../proofs/PROOFS.md) for detailed evidence and claim limits.

---

## At a glance

| Question | Answer |
| -------- | ------ |
| What is Intergrax? | A reusable governed foundation for specialized AI applications. |
| What problem does it address? | Each product otherwise assembles its own control and execution infrastructure around model behavior. |
| Who is it for? | Teams building specialized agent-backed applications, platform engineers evaluating reusable infrastructure, and technical design partners with a concrete workflow to validate. |
| What remains with the product team? | Workflow semantics, UX, deployment choices, required permissions, product-specific validation, and business responsibility. |
| Where does it fit? | Around model or agent behavior, as reusable governed application infrastructure. |

---

## The repeated problem

Model calls and agent behavior are only one part of a specialized application. A product also commonly needs:

- identity and permissions;
- controlled knowledge access;
- policy and human approval;
- tool and integration boundaries;
- context and runtime controls;
- evidence and provenance;
- failure handling and recovery; and
- observability.

These are recurring engineering responsibilities, not a claim that every mechanism above is universally production-proven. Without a reusable operating foundation, each application rebuilds substantial control and execution infrastructure, with the resulting inconsistency and maintenance burden spread across products.

---

## What Intergrax is

Intergrax is a reusable governed foundation, or application operating layer, for specialized AI applications. It concentrates repeatable boundaries around knowledge, policy, tools, evidence, context, and execution so product teams can focus on the concrete domain workflow.

Some teams use **Harness AI** to describe this category: an operating layer around model or agent behavior that constrains, coordinates, records, and recovers execution. The term is useful shorthand, not a universal industry taxonomy.

| Layer | Primary responsibility |
| ----- | ---------------------- |
| Model or agent framework / platform | Model access, agent behavior, orchestration primitives, and often adjacent runtime facilities such as tracing, persistence, guardrails, workflows, or HITL — depending on the chosen stack. |
| Intergrax | Reusable application operating boundaries around knowledge, policy, tools, evidence, context, and governed execution across specialized products. |
| Product application | Domain workflow, UX, deployment, required permissions, and product-specific validation. |

Modern agent frameworks and platforms increasingly bundle runtime facilities that teams once assembled separately. The comparison is not whether those facilities exist somewhere in the stack, but **which layer owns primary responsibility** for product semantics, enforcement boundaries, consequential effects, canonical history, recovery posture, and evidence — and how much integration burden remains with the adopting team.

```mermaid
flowchart LR
    P1[Product A] --> R1[Rebuilt foundations]
    P2[Product B] --> R2[Rebuilt foundations]
    P3[Product C] --> R3[Rebuilt foundations]

    I[Intergrax reusable foundation]
    I --> A[Specialized application A]
    I --> B[Specialized application B]
    I --> C[Specialized application C]
```

The diagram contrasts rebuilding foundations per product with reusing one governed foundation. It does not imply that every future product is already implemented.

---

## Responsibility boundary

Intergrax aims to centralize reusable application foundations:

- policy and governance mechanisms;
- knowledge, context, and evidence boundaries;
- integration and tool-execution boundaries; and
- runtime controls for governed execution.

The adopting product team still owns:

- workflow and product semantics;
- UX and deployment choices;
- required identity, tenant, and product permissions;
- product-specific validation and acceptance criteria; and
- business responsibility for the product and its outcomes.

Intergrax does not remove those responsibilities. It is intended to reduce repeated foundation-building, not to make a domain product or its risk decisions automatic.

---

## How Intergrax approaches responsibility

This section states the public differentiation spine in plain language. It describes an architectural approach — not measured superiority, delivery speed, or proof that every boundary is complete today.

1. **Product owns meaning; platform owns enforcement.** Your application defines what permission means, which actions need approval, and what outcomes are acceptable. Intergrax supplies reusable mechanisms that evaluate and enforce those rules at configured execution boundaries.

2. **Governance spans explicit execution boundaries.** Policy, denial, and human approval attach to named steps in execution rather than living only in scattered application conditionals.

3. **Consequential external effects cross an explicit governed boundary.** Tool calls and meaningful side effects are authorized and recorded through platform mechanisms on wired paths — separate from model reasoning itself.

4. **Execution has structural identity and canonical history.** Runs, attempts, and events carry typed identity so operators can reconstruct what happened without treating an external trace UI as the only source of truth.

5. **Recovery distinguishes retry, idempotency, compensation, degradation, and HITL.** Failure handling is classified and bounded. Agents may express recovery intent; the runtime owns budgets, layers, and stop conditions.

6. **Important execution can produce structured evidence, not telemetry alone.** Governance and execution transitions can be correlated with persisted evidence where mechanisms are connected — beyond unstructured log lines.

7. **Agent authors own domain behavior; agents are not private runtimes.** Agents declare contracts and implement domain decisions. The platform owns safe execution, tracing, budgets, and lifecycle — not a second hidden scheduler or policy engine inside each agent.

Other stacks can implement similar patterns. Intergrax deliberately consolidates these responsibilities into a shared application operating model intended to serve multiple specialized products. Whether that consolidation reduces integration burden in practice remains an evaluation question, not a universal claim.

For named modern frameworks and platforms — including cases where another stack may be the better fit — see [Compare modern alternatives and trade-offs](ALTERNATIVES_AND_TRADEOFFS.md).

---

## Business and strategic thesis

### Repeated organizational cost

When multiple AI applications each build policy boundaries, knowledge access, tool and integration controls, approval handling, evidence and provenance, recovery, and observability separately, the organization duplicates engineering effort. Implementations diverge, maintenance burden spreads across products, and review surfaces become inconsistent. This is a duplication and fragmentation problem — not a claim of measured savings.

### Potential sponsor and adopter profiles

Commercial validation remains incomplete. Potential adopter or sponsor profiles include:

- a CTO or VP Engineering responsible for multiple AI product initiatives;
- an AI platform or enablement team standardizing governed application foundations;
- an enterprise product team building a specialized governed AI application; and
- an integrator or solution engineering group delivering repeated customer-specific AI applications.

### LKW as the current product wedge

LKW is the current product path used to test this thesis. It exercises whether a reusable governed foundation can support a concrete, understandable workflow: approved knowledge → grounded Ask → source and evidence → persistence and reviewability. This is bounded product and platform proof — not product-market fit.

### Compounding value hypothesis

If multiple specialized applications reuse the same policy, integration, context, evidence, recovery, observability, and execution foundations, investment in those shared mechanisms may support more than one product: each additional application can reuse existing governed capabilities, expose missing capability gaps, and strengthen the reusable foundation. This is a strategic hypothesis, not a measured commercial result.

### Commercialization gates

The following remain open before commercial validation can be claimed:

- real-user validation and repeated usage;
- workflow-level value in representative conditions;
- design-partner validation and operational or pilot evidence where authorized;
- commercial willingness to pay or enter agreement; and
- deployment and operational hardening where required.

---

## Who it is for

Intergrax is for:

1. Teams building specialized agent-backed applications that need governance, evidence, knowledge access, or controlled tool execution.
2. Platform or AI engineers evaluating reusable application infrastructure.
3. Architects who need controlled execution and evidence boundaries.
4. Technical design partners with a concrete workflow worth validating.

---

## Where Intergrax fits

These approaches are not mutually exclusive. A real system may combine several of them, and the comparison is about primary responsibility rather than a feature scorecard or superiority claim.

The question is: which layer of responsibility are you trying to buy or build?

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../assets/public/intergrax-category-map-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../assets/public/intergrax-category-map-light.svg"
  >
  <img
    alt="Responsibility map comparing finished AI SaaS, workflow automation platforms, retrieval or knowledge toolkits, agent frameworks, custom in-house foundations and Intergrax by their primary value and what the adopting team still owns."
    src="../assets/public/intergrax-category-map-light.svg"
  >
</picture>

This neutral map explains primary responsibility. Categories overlap, and no universal superiority or feature parity is claimed.

### Choose by your primary need

| Approach | Primary value | Your team still owns | Consider it when |
| -------- | ------------- | -------------------- | ---------------- |
| Finished AI SaaS | Ready-made end-user product | Adoption and workflow fit | You need a ready-made end-user product |
| Workflow automation platform | Connect systems and process steps | AI-specific behavior and evidence semantics | You need process and system automation |
| Retrieval or knowledge toolkit | Retrieval, indexing and grounding components | Product, orchestration, policy and operations | You need retrieval and grounding components |
| Agent framework or agent platform | Compose agent behavior, orchestration, and often adjacent runtime facilities (tracing, persistence, guardrails, workflows, HITL) within that stack's model | Product semantics, multi-application operating model, and how foundations are shared across products | You need agent behavior and orchestration as the primary deliverable |
| Custom in-house foundation | Maximum design control | Every shared layer and its maintenance | You need complete design control and can maintain the platform |
| Intergrax | Reusable governed foundation for specialized AI applications | Product workflow, UX, deployment choices, required permissions, product-specific validation, and business responsibility | You need to build a specialized governed AI application on reusable policy, knowledge, integration, execution, and evidence foundations |

### A simple decision flow

```mermaid
flowchart TD
    A[What do you need most?]
    A -->|A ready-made end-user product| B[Finished AI SaaS]
    A -->|Connected systems and process steps| C[Workflow automation platform]
    A -->|Retrieval and grounding components| D[Retrieval or knowledge toolkit]
    A -->|Agent behavior and orchestration| E[Agent framework]
    A -->|Maximum control and platform capacity| F[Custom in-house foundation]
    A -->|A specialized application on reusable governed foundations| G[Evaluate Intergrax]
    G --> H[Check concrete use-case fit]
    H --> I[Review current proof and limitations]
```

### Fit summary

| Intergrax may fit when | Another approach may fit better when |
| ---------------------- | ------------------------------------ |
| You need governed execution, evidence, and knowledge access in a specialized product | Finished SaaS needed immediately — not building your own application |
| You want reusable platform foundations instead of a new runtime per product | You want a simple prompt prototype only |
| You are evaluating infrastructure for multiple agent-backed applications | You expect a no-code workflow builder |
| You can accept bounded proof and active R&D | You require an unrestricted open-source license |
| Governance, policy, and evidence matter to your reviewers | You have no governance or evidence requirements |
| You want to own a specialized application while reusing its operating foundation | You do not want to own an application or its product-specific validation |

Other frameworks and tools may suit different needs. This guide does not dismiss them. For named modern alternatives and explicit trade-offs — including when another stack may be the better choice — see [ALTERNATIVES_AND_TRADEOFFS.md](ALTERNATIVES_AND_TRADEOFFS.md).

Intergrax may coexist with model providers, retrieval systems, integration tools and application-specific components. The category map does not claim that these approaches are mutually exclusive.

---

## Current evidence and limits

| Area | Classification | Status | Notes |
| ---- | -------------- | ------ | ----- |
| **LKW** | **Primary product proof** | **PARTIAL** | **Backend Product Alpha / MVP** — bounded application and platform proof. |
| **Token Optimization** | **Featured platform-capability proof** | **PARTIAL** | Implemented deterministic mechanisms; bounded executable offline smoke proof. Manual vLLM prefix-cache evidence is separate — no public_evidence_eligible vLLM proof_id today. |

**Evidence and detail:** [LKW Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) · [Token Optimization guide](../capabilities/token_optimization/README.md) · [PROOFS.md](../proofs/PROOFS.md)

The bounded indexed Hybrid Ask path is not the same as a completed mixed indexed + authorized live workflow: **mixed indexed + authorized live Hybrid Ask remains incomplete**. Real-user validation and commercial validation are incomplete. Universal production readiness and universal token savings are not claimed.

Detailed evidence and claim boundaries belong in [docs/project/proofs/PROOFS.md](../proofs/PROOFS.md), not in this category overview.

---

## What to read next

**Primary next action:** [Check your concrete workflow in Use Cases](USE_CASES.md).

If the category appears relevant:

| Route | Use it for |
|-------|------------|
| [ALTERNATIVES_AND_TRADEOFFS.md](ALTERNATIVES_AND_TRADEOFFS.md) | Comparing Intergrax with named modern agent/platform alternatives and trade-offs |
| [docs/project/proofs/PROOFS.md](../proofs/PROOFS.md) | Reviewing current evidence |
| [docs/project/architecture/ARCHITECTURE_OVERVIEW.md](../architecture/ARCHITECTURE_OVERVIEW.md) | Understanding technical boundaries |
| [docs/project/builders/BUILDER_QUICKSTART.md](../builders/BUILDER_QUICKSTART.md) | Beginning a bounded build |
| [docs/project/builders/BUILD_WITH_INTERGRAX.md](../builders/BUILD_WITH_INTERGRAX.md) | Planning deeper application composition |
| [docs/project/community/PARTNERS.md](../community/PARTNERS.md) | Discussing a bounded pilot |
