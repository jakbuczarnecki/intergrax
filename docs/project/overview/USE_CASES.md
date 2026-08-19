<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Use Cases

This is the canonical decision guide for a concrete question: **does Intergrax fit my workflow?** It evaluates stable user problems and responsibility fit. It does not replace the proof record, an outcome roadmap, or a product-specific evaluation.

> [!NOTE]
> Current evidence is bounded. Authoritative status and claim limits: [PROOFS.md](../proofs/PROOFS.md).

## At a glance

| Decision question | Current answer |
|-------------------|----------------|
| Strongest current fit | Private governed knowledge workspace — **Primary product proof**, **Backend Product Alpha / MVP**, **PARTIAL** |
| Bounded technical fit | Specialized governed applications and context workflows; a **Reasonable technical evaluation** may be appropriate |
| Not a fit today | Finished SaaS immediately, no-code automation, retrieval primitives only, or unrestricted OSS — another approach may fit better |
| Primary next action | Review [PROOFS](../proofs/PROOFS.md) before evaluation or partner discussion |

## Start with your workflow

Intergrax is more likely to fit when the workflow needs several of these responsibilities:

- explicit identity, permissions, or tenant context;
- governed knowledge access and grounded results;
- controlled tool or integration execution;
- human approval boundaries;
- evidence, provenance, receipts, or trace;
- recovery and observability;
- reusable foundations across more than one workflow.

These are fit signals, not a requirement that every workflow use every mechanism. Ask:

1. What does a person do from the first request to the accepted outcome?
2. Which knowledge and external actions are allowed, and which are forbidden?
3. What must a reviewer be able to inspect afterward?
4. Can the product team own the domain workflow and validate the result in a bounded evaluation?

Use this stable vocabulary:

- **STRONGEST CURRENT FIT** — current product proof directly represents a similar workflow.
- **BOUNDED TECHNICAL FIT** — architecture and supporting mechanisms are plausible, but product-specific validation remains necessary.
- **NOT YET PROVEN** — the user outcome is structurally plausible, but the required end-to-end outcome is not established.
- **NOT A FIT** — another class of solution is more appropriate.

```mermaid
flowchart TD
    A[Concrete workflow] --> B[Private governed knowledge workspace]
    A --> C[Specialized agent application]
    A --> D[Evidence-aware automation or integration]
    A --> E[Governed context optimization]
    A --> F[Simple prototype or ready-made product]

    B --> G[STRONGEST CURRENT FIT]
    C --> H[BOUNDED TECHNICAL FIT]
    D --> H
    E --> H
    F --> I[NOT A FIT]
```

The diagram is a responsibility-based evaluation route, not a list of completed products. The textual sections below are the authoritative explanation.

**Current evidence boundary:** LKW demonstrates bounded indexed Ask through production Hybrid Ask. **Mixed indexed + authorized live Hybrid Ask is not yet established**; complete live-provider access is incomplete. Real-user and commercial validation remain open — see [PROOFS](../proofs/PROOFS.md).

## Strongest current fit

### Private governed knowledge workspace

**User problem:** private or controlled knowledge is distributed across approved sources. People need grounded answers with source references, while access boundaries, persistence, and reviewability remain explicit.

**Who:** internal knowledge workers; expert or operations teams; teams with controlled documentation and reviewer requirements; and the product or IT owner responsible for knowledge access. No existing customer claim is implied.

**Current approach:** manual search across approved sources; separate search or retrieval tools; ad-hoc AI assistants without consistent evidence or access boundaries; and fragmented workflows across files and systems.

**Pain:** difficulty finding trusted current information; weak source traceability; inconsistent access boundaries; repeated manual review; and hard-to-review AI answers.

**Desired outcome:** grounded answers from approved sources; a clear source and evidence trail; reviewable execution; and persistence or repeatability where relevant.

**Success signal:** representative questions completed correctly; expected source references present; forbidden or unauthorized sources not used; evidence retained; workflow repeatable; and target users judge answers useful enough to reuse.

**Responsibility fit:** Intergrax can provide reusable mechanisms around approved indexed sources, grounded Ask execution, evidence and provenance, controlled access context, persistence, and reviewable execution. The product team still defines the workspace workflow, source approval, user experience, business permissions, deployment, and acceptance criteria.

**Current evidence class:** **STRONGEST CURRENT FIT**

**Current proof:** LKW is the **Primary Product Proof**, **Backend Product Alpha / MVP**, with **PARTIAL** proof status. Bounded indexed Ask evidence exists, including the indexed path through production Hybrid Ask.

Canonical verification route: [LKW Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) via [PROOFS — LKW Primary Product Proof](../proofs/PROOFS.md#lkw--primary-product-proof).

## Other bounded technical fits

### Governed knowledge application

**Who:** enterprise product team, AI engineering team, or solution team building a specialized knowledge application.

**Current approach:** product-specific retrieval and governance plumbing combined with custom access and evidence handling.

**Pain:** duplicated platform work and inconsistent evidence or access contracts across products.

**Desired outcome:** controlled sources, grounded answers, evidence, and explicit access boundaries in a specialized application the team owns.

**Success signal:** bounded workflow runs; required controls and evidence present; product-specific behavior remains application-owned.

**Fit:** **BOUNDED TECHNICAL FIT** when the product team can validate its own sources, permissions, user workflow, and acceptance criteria. The LKW evidence is the closest current product reference, but it does not automatically validate another product.

### Specialized agent application

**Who:** enterprise product team, AI engineering team, or solution or integration team.

**Current approach:** build product-specific governance, integration, and runtime plumbing; combine multiple frameworks and custom controls.

**Pain:** duplicated platform work; inconsistent contracts and evidence; product engineers spending time on reusable infrastructure rather than domain workflow.

**Desired outcome:** build specialized product behavior on reusable governed foundations.

**Success signal:** bounded application workflow runs; required controls and evidence present; product-specific code remains application-owned; reusable mechanisms are reused rather than rebuilt; and the team can identify remaining platform gaps.

**Fit:** **BOUNDED TECHNICAL FIT**. Intergrax supplies reusable application operating mechanisms; the adopting team builds and validates the specialized product.

**Current proof:** builder scaffold and current architecture evidence support a reasonable technical evaluation. Start with [Build With Intergrax](../builders/BUILD_WITH_INTERGRAX.md).

### Evidence-aware automation or integration workflow

**Who:** teams where automated actions require reviewability or policy boundaries.

**Current approach:** actions distributed across integrations without one consistent evidence or approval trail.

**Pain:** hard to reconstruct what ran, what was allowed, and what failed across systems.

**Desired outcome:** controlled action with reviewable evidence and explicit failure or recovery boundaries.

**Success signal:** allowed action executes; forbidden action is blocked or rejected as designed; receipt or trace exists; a reviewer can reconstruct the outcome; and the failure path is understandable.

**Fit:** **BOUNDED TECHNICAL FIT** for a bounded technical evaluation. Supporting evidence does not amount to certification, compliance approval, legal attestation, or universal operational readiness. See the [Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md) for the responsibility boundary.

**Current proof:** supporting verification route — [BoundaryAttest case study](case-studies/BOUNDARYATTEST_ATTESTATION_POC.md).

### Governed context and prompt optimization

**Who:** teams needing deterministic, policy-bounded context handling with reviewable receipts.

**Pain:** unnecessary context or prompt material may be included while protected regions and policy constraints must remain intact.

**Desired outcome:** reduce unnecessary context or prompt material while preserving protected regions and producing evidence. Universal savings are not claimed.

**Success signal:** policy constraints preserved; protected regions remain intact; deterministic optimization behavior; receipts produced; and bounded offline proof passes.

**Fit:** **BOUNDED TECHNICAL FIT**. Token Optimization is a **Featured platform-capability proof**, **PARTIAL**, with bounded evidence. See the [Token Optimization guide](../capabilities/token_optimization/README.md).

Bounded technical fits require product-specific validation; the adopting team owns acceptance criteria and repeated-use evidence.

## Not yet proven

The stable outcome **“combine indexed knowledge with authorized live evidence”** is **NOT YET PROVEN**. Bounded indexed Ask evidence exists, but mixed indexed + authorized live Hybrid Ask with unified provenance remains incomplete.

This distinction is about evidence, not whether the problem is meaningful. See [PROOFS](../proofs/PROOFS.md) for the authoritative boundary.

## When another approach is better

Another approach may fit better if the team only needs:

- a simple prototype or prompt demonstration;
- a finished SaaS immediately, without owning a custom application;
- no-code automation;
- retrieval primitives only;
- an unrestricted open-source framework;
- no responsibility for product-specific validation, deployment, or business outcomes.

Intergrax is a reusable foundation, not a universal replacement for a finished product, an automation platform, a retrieval toolkit, or an unrestricted framework.

## Fit matrix

| Workflow need | Fit class | Evaluation question | Decision implication |
|--------------|-----------|---------------------|----------------------|
| Private governed knowledge workspace | **STRONGEST CURRENT FIT** | Can representative questions be answered from approved sources with evidence retained? | Inspect the LKW proof and its limits |
| Specialized governed application | **BOUNDED TECHNICAL FIT** | Can the team reuse foundations instead of rebuilding governance and runtime plumbing? | A reasonable technical evaluation may be appropriate |
| Shared foundations across multiple applications | **BOUNDED TECHNICAL FIT** | Would a second application reuse existing governed capabilities? | Product-specific validation remains required |
| Indexed knowledge combined with authorized live evidence | **NOT YET PROVEN** | Is bounded indexed evidence sufficient without mixed live provenance? | Defer unless the bounded evidence is sufficient for evaluation |
| Generic prototype or finished SaaS needed immediately | **NOT A FIT** | Is owning a specialized governed application required? | Another approach may fit better |

## What Intergrax does not currently offer

These are decision shortcuts — not a substitute for [PROOFS](../proofs/PROOFS.md):

- No finished hosted SaaS
- No complete Hybrid Ask combining indexed and authorized live evidence
- No compliance certification or unrestricted open-source rights
- No universal token or cost reduction

## Responsibility check

Intergrax helps centralize reusable mechanisms for:

- policy and approval boundaries;
- knowledge, context, identity, and tool-access controls;
- governed execution, recovery, and observability;
- evidence, provenance, receipts, and review surfaces.

The product team remains responsible for:

- domain workflow and business semantics;
- UX and product behavior;
- permission decisions and required identity context;
- deployment and operating choices;
- product-specific validation and acceptance;
- business responsibility for the product and its outcomes.

Fit therefore means that Intergrax can reduce repeated foundation-building. It does not transfer ownership of the product or its risk decisions.

## Define a useful evaluation

Before proceeding, define one bounded workflow:

- **Workflow:** what a person does from request to accepted outcome.
- **Data:** which sources may be read, indexed, or used live.
- **Allowed actions:** what the system may do autonomously.
- **Forbidden actions:** what must never happen without approval.
- **Approvals:** where a person must confirm or reject.
- **Evidence:** citations, receipts, trace, provenance, and review surfaces required.
- **Success:** the measurable result that makes the evaluation worthwhile.
- **Repeat use:** what would make the team return to the workflow.

## Evidence separation and decision

Each document answers a different question:

- **USE_CASES:** conceptual and current workflow fit.
- **[PROOFS](../proofs/PROOFS.md):** current evidence status and claim limits.
- **[ROADMAP](ROADMAP.md):** outcome direction, not a provider or module tracker here.
- **[Evaluation Guide](../builders/EVALUATION_GUIDE.md):** how to run a bounded evaluation.
- **[Partners](../community/PARTNERS.md):** pilot or operational discussion route.

After apparent fit, the primary next action is **PROOFS**: [review the current evidence status](../proofs/PROOFS.md).

1. If relevant evidence exists, use the [Evaluation Guide](../builders/EVALUATION_GUIDE.md).
2. If a pilot or operational discussion is appropriate, use [Partners](../community/PARTNERS.md) and review [Collaboration](../community/COLLABORATION.md) as needed.
3. If evidence is insufficient, defer.
4. If the responsibility class is wrong, stop and choose another approach.
