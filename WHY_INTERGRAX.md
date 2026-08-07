<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Why Intergrax

Intergrax helps teams understand whether reusable governed foundations fit the problem they are trying to solve.

Intergrax helps teams build specialized agent applications without rebuilding the same policy, knowledge, evidence, integration, and execution foundations for every product.

> [!NOTE]
> Intergrax is **source-available** and under **active R&D**. LKW is the current **primary product proof** and remains **Backend Product Alpha / MVP**. **Real-user** and **commercial validation** are incomplete.

---

## At a glance

| Question | Answer |
| -------- | ------ |
| What is Intergrax? | A reusable foundation for governed AI applications. |
| What problem does it address? | Teams repeatedly rebuild permissions, policy, knowledge, tools, evidence, and operational boundaries for every new agent product. |
| Who is it for? | Teams building specialized agent-backed applications, platform engineers evaluating reusable infrastructure, and technical design partners with a concrete workflow to validate. |
| Current proof | LKW as **Primary product proof**; Token Optimization as **Featured platform-capability proof** — both **PARTIAL**. |
| Current maturity | **Source-available**, **active R&D**; bounded proof paths, not universal production readiness. |
| Where does it fit? | A reusable governed application foundation — not a finished SaaS, no-code builder, single-purpose retrieval component or standalone agent framework |

---

## The repeated problem

Building an impressive agent demo is relatively easy. Delivering a controlled application that a team can review, operate, and trust is difficult.

Every new product tends to need the same foundations again: permissions and identity, knowledge access, policy enforcement, human-in-the-loop gates, tool and integration boundaries, trace and evidence collection, testing, and runtime governance. Rebuilding these separately for each product creates inconsistency, slows delivery, and makes governance harder to audit across applications.

---

## The Intergrax approach

Intergrax concentrates repeatable infrastructure into one governed foundation so product teams can focus on the concrete workflow.

| Differentiator | What it means |
| -------------- | ------------- |
| **Application-first** | Real applications and user workflows lead development. |
| **Governed execution by default** | Policy, budgets, human review, and trace surround execution — not bolted on after the demo works. |
| **Reusable delivery foundation** | Multiple products reuse shared knowledge, evidence, integration, and execution infrastructure. |
| **Explicit responsibility boundaries** | Applications, orchestration, agents, and the harness each own a clear slice of the system. |

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

## Who it is for

1. Teams building specialized agent-backed applications that need governance, evidence, knowledge access, or controlled tool execution.
2. AI platform engineers and architects evaluating reusable agent-application infrastructure.
3. Technical design partners with a concrete workflow worth validating.

Intergrax is not aimed at every possible AI project, generic consumers, or teams that only need a one-off demo.

---

## Where Intergrax fits

These approaches are not mutually exclusive. A real system may combine several of them, and the comparison is about primary responsibility rather than a feature scorecard or superiority claim.

The right choice depends on the product and responsibility boundary the team wants to own. Consider the approach that best matches the outcome, control, and maintenance burden required for the work.

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="docs/assets/public/intergrax-category-map-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="docs/assets/public/intergrax-category-map-light.svg"
  >
  <img
    alt="Responsibility map comparing finished AI SaaS, workflow automation platforms, retrieval or knowledge toolkits, agent frameworks, custom in-house foundations and Intergrax by their primary value and what the adopting team still owns."
    src="docs/assets/public/intergrax-category-map-light.svg"
  >
</picture>

This neutral map explains primary responsibility. Categories overlap, and no universal superiority or feature parity is claimed.

### Choose by your primary need

| Approach | Primary value | Your team still owns | Consider it when |
| -------- | ------------- | -------------------- | ---------------- |
| Finished AI SaaS | Ready-made end-user product | Adoption and workflow fit | You need a ready-made end-user product |
| Workflow automation platform | Connect systems and process steps | AI-specific behavior and evidence semantics | You need process and system automation |
| Retrieval or knowledge toolkit | Retrieval, indexing and grounding components | Product, orchestration, policy and operations | You need retrieval and grounding components |
| Agent framework | Compose agent behavior and orchestration | Product controls, runtime governance and evidence | You need agent behavior and orchestration |
| Custom in-house foundation | Maximum design control | Every shared layer and its maintenance | You need complete design control and can maintain the platform |
| Intergrax | Reusable governed foundation for specialized AI applications | Product workflow, UX, deployment choices, product-specific validation and all required permissions remain the product team's responsibility. | You need to build a specialized governed AI application on reusable policy, knowledge, integration, execution and evidence foundations |

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

Other frameworks and tools may suit different needs. This guide does not dismiss them.

Intergrax may coexist with model providers, retrieval systems, integration tools and application-specific components. The category map does not claim that these approaches are mutually exclusive.

**Primary next action after apparent fit:** [Check your concrete workflow in USE_CASES.md](USE_CASES.md).

---

## What exists today

| Area | Classification | Status | Notes |
| ---- | -------------- | ------ | ----- |
| **LKW** | **Primary product proof** | **PARTIAL** | **Backend Product Alpha / MVP** — bounded application and platform proof. |
| **Token Optimization** | **Featured platform-capability proof** | **PARTIAL** | Implemented mechanisms plus **bounded vLLM** proof. |

**Evidence and detail:** [PROOFS.md](PROOFS.md) · [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) · [Token Optimization guide](docs/features/token_optimization/README.md)

---

## What Intergrax does not currently claim

Intergrax does **not** currently claim:

- a finished SaaS;
- completed real-user validation;
- completed commercial validation;
- universal production readiness;
- completed Hybrid Ask;
- universal token savings or production-proven savings.

Detailed proof matrices and claim boundaries remain in [PROOFS.md](PROOFS.md).

---

## Primary next action

**Check concrete workflow fit:** [USE_CASES.md](USE_CASES.md)

## Other routes

| Route | Use it for |
|-------|------------|
| [PROOFS.md](PROOFS.md) | Reviewing current evidence |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | Understanding technical boundaries |
| [BUILDER_QUICKSTART.md](BUILDER_QUICKSTART.md) | Beginning a bounded build |
| [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) | Planning a deeper build or evaluation route |
| [PARTNERS.md](PARTNERS.md) | Discussing a bounded pilot |
