<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Proof Library

**Real problems. Executable evidence.**

The Intergrax Proof Library is a problem-first catalog of **Scenario Proofs** — executable, falsifiable demonstrations that start from difficult real-world AI system problems, not from platform feature demos.

Each accepted Scenario Proof is designed so you can inspect what was tested, how it was attacked, the evidence produced, the verdict, stated limitations, source, and a reproduction path. Scenarios are **not** marketing demos; they exist to make bounded claims inspectable under adversarial conditions.

> [!NOTE]
> The Proof Library is being **bootstrapped**. Accepted Scenario Proofs will appear in the catalog below only after passing canonical acceptance gates. No scenario is listed here until that bar is met.

---

> [!IMPORTANT]
> **Challenge Intergrax with your problem.**
>
> Have a difficult AI problem? Describe the real workflow, what can go wrong, and what a convincing result would require.
>
> Bring the **problem**, not an Intergrax feature request. If the problem is suitable for the Proof Library, maintainers may turn it into an executable Scenario Proof.
>
> **[Propose a scenario →](https://github.com/jakbuczarnecki/intergrax/issues/new?template=scenario_proposal.yml)**

---

## What is a Scenario Proof?

A Scenario Proof follows a canonical path from real failure risk to inspectable evidence:

```text
REAL PROBLEM
→ FAILURE RISK
→ SCENARIO
→ ADVERSARIAL TEST
→ INTERGRAX MECHANISMS
→ EXECUTION
→ EVIDENCE
→ VERDICT
→ REPORT
→ REPRODUCTION
```

Scenario Proofs aim to demonstrate **difficult system guarantees** — governance under conflict, unsafe-action prevention, recovery, evidence admissibility, and similar real-world conditions — rather than basic LLM orchestration or happy-path demos.

For maintainer authoring workflow and technical infrastructure, see the internal [Platform Proof Library](../../../platform_proofs/README.md) gateway.

---

## How to read a proof

Every published Scenario Proof should let you inspect:

| Inspect | What you learn |
| --- | --- |
| **Real problem** | The workflow or decision the system must support |
| **Why failure matters** | Consequences if the AI or system gets it wrong |
| **Bounded claim** | Exactly what is — and is not — being demonstrated |
| **PASS / FAIL criteria** | Objective conditions for verdict |
| **Adversarial conditions** | Failure modes, conflicts, and edge cases exercised |
| **Intergrax mechanisms** | Which platform capabilities the scenario actually uses |
| **Execution / evidence** | What ran and what artifacts were produced |
| **Final verdict** | PASS, FAIL, or bounded partial outcome |
| **Limitations** | Excluded claims and scope boundaries |
| **Source** | Repository paths and proof package location |
| **Reproduction** | How to run or verify locally |
| **Report** | Rich narrative when available |

---

## Scenario catalog

Accepted Scenario Proofs appear here after passing canonical Proof Library acceptance gates.

**Current status:** The library is being bootstrapped. The first flagship Scenario Proof has passed design qualification and is awaiting implementation under the canonical authoring process.

No accepted scenarios are published yet. When entries appear, each will follow this shape:

```text
Scenario title

Problem:
...

Risk:
...

What is demonstrated:
...

Status:
...

[Read scenario] [View report] [Source] [Run locally]
```

Links will be added only when a scenario has passed acceptance — not for placeholder or in-design work.

---

## Proof Library vs evidence dashboard

These are **separate** public surfaces with different jobs:

| Surface | Framing | Answers |
| --- | --- | --- |
| **Proof Library** (this page) | Problem-first | *Show me difficult problems and executable scenarios.* |
| **[PROOFS.md](PROOFS.md)** | Evidence-first | *Show me exactly what is proven and what Intergrax is allowed to claim.* |

An attractive scenario description **never** overrides evidence or claim status. For implementation status, bounded verification, partial capability, planned work, and claims that are not currently supported, use the **[Intergrax Proofs evidence dashboard](PROOFS.md)**.

---

## Related routes

| Route | Purpose |
| --- | --- |
| [PROOFS.md](PROOFS.md) | Public evidence-and-claims dashboard |
| [LKW Product Tour](../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) | Active reference product walkthrough |
| [LKW Quick Start](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) | Runnable product evaluation path |
| [Platform Proof Library (maintainers)](../../../platform_proofs/README.md) | Technical proof infrastructure and authoring |
