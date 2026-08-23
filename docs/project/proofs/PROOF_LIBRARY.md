<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Proof Library

**Real problems. Executable evidence.**

The Intergrax Proof Library is a problem-first catalog of **Scenario Proofs** — executable falsification attempts against bounded real-world system claims.

Intergrax is easier to understand by watching it handle hard problems than by reading capability lists. Each accepted Scenario Proof is designed to be run, inspected, challenged, and reproduced: what was tested, how it was attacked, the evidence produced, the verdict, stated limitations, source, and a reproduction path.

> [!IMPORTANT]
> Scenario Proofs stress-test bounded system guarantees under adversarial conditions and produce inspectable evidence. **Scenario Proofs are not products.** They do not substitute for real-user validation, commercial validation, or production readiness. Product, user, and commercial validation remain separate evidence classes.

---

## A. What is a Scenario Proof?

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

Scenario Proofs aim to demonstrate **difficult system guarantees** — governance under conflict, unsafe-action prevention, recovery, evidence admissibility, false confident diagnosis, and similar real-world conditions — rather than basic LLM orchestration or happy-path demos.

<a href="../assets/public/readme/fullsize/intergrax-scenarios-overview.md">
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../assets/public/readme/intergrax-scenarios-overview-dark.png"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../assets/public/readme/intergrax-scenarios-overview-light.png"
  >
  <img
    src="../assets/public/readme/intergrax-scenarios-overview-light.png"
    alt="Scenario Proof path from real problem through adversarial test and execution to evidence, verdict, and reproduction."
  >
</picture>
</a>

[View full-size diagram](../assets/public/readme/fullsize/intergrax-scenarios-overview.md)

---

## B. What makes a scenario worth publishing?

A scenario belongs in the library when it exposes a **meaningful failure risk** that simple AI is insufficient to handle safely:

| Criterion | Why it matters |
| --- | --- |
| **Real workflow stakes** | Wrong answers or unsafe actions have operational consequences |
| **Adversarial conditions** | Conflicting, stale, incomplete, or misleading evidence is part of the problem |
| **Bounded claim** | Exactly what is — and is not — being demonstrated is explicit |
| **Executable falsification** | A skeptical reviewer can run, inspect, and challenge the result |
| **Honest outcomes** | PASS, FAIL, or bounded UNRESOLVED must be earned — not narrated |

Maintainer authoring workflow and technical infrastructure live in the internal [Platform Proof Library](../../../platform_proofs/README.md) gateway.

---

## C. Scenario catalog

The library operating model is ready. Accepted Scenario Proofs appear here only after passing canonical acceptance gates — with evidence, verdict, report, and reproduction routes.

**Current status:** no accepted Scenario Proofs are published yet. The first flagship scenario is in development.

| Scenario | Status | Public routes |
| --- | --- | --- |
| **AI Incident Investigation with Independent Verification** | **In development** — design accepted for implementation; no executable proof yet | [Scenario design](../../../platform_proofs/scenarios/ai_incident_investigation/README.md) |

When a scenario is accepted, each entry will expose:

```text
Scenario title

Problem:
...

Risk:
...

What is demonstrated:
...

Status: ACCEPTED

[Read scenario] [View report] [Source] [Run locally]
```

Links appear only when artifacts exist — not for placeholder or in-design work.

---

## D. Featured scenario in development

### AI Incident Investigation with Independent Verification

> **Can an AI investigate an operational incident without turning correlation into a confident false diagnosis?**

Initial operational signals make workload overload plausible. Evidence is conflicting, stale, and incomplete. Independent verification must challenge unsupported causality, gather targeted evidence to distinguish competing hypotheses, and produce a bounded **RESOLVED** or honest **UNRESOLVED** outcome.

**Status:** in development — not accepted proof evidence. No report, evidence bundle, or reproduction path exists yet.

<a href="../assets/public/readme/fullsize/scenario-ai-incident-investigation.md">
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../assets/public/readme/scenario-ai-incident-investigation-dark.png"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../assets/public/readme/scenario-ai-incident-investigation-light.png"
  >
  <img
    src="../assets/public/readme/scenario-ai-incident-investigation-light.png"
    alt="AI incident investigation scenario — operational signals, conflicting evidence, independent verification, and bounded RESOLVED or UNRESOLVED outcomes."
  >
</picture>
</a>

[View full-size diagram](../assets/public/readme/fullsize/scenario-ai-incident-investigation.md)

**[Scenario design document](../../../platform_proofs/scenarios/ai_incident_investigation/README.md)**

---

## E. How to read a proof

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

## F. PASS / FAIL / UNRESOLVED semantics

| Verdict | Meaning |
| --- | --- |
| **PASS** | The bounded claim survived adversarial conditions with inspectable evidence |
| **FAIL** | The system did not meet the stated claim under the scenario's conditions |
| **UNRESOLVED** | Critical distinguishing evidence is unavailable or hypotheses remain indistinguishable — no confident guessing |

A scenario visual may show RESOLVED or UNRESOLVED branches as **possible bounded outcomes** — that describes scenario semantics, not a current achieved PASS.

---

## G. Challenge Intergrax

Have a difficult AI system problem?

Describe:

- the workflow
- what can go wrong
- why simple AI is insufficient
- what a convincing result would require

Bring the **problem**, not an Intergrax feature request. If the problem is suitable for the Proof Library, maintainers may turn it into an executable Scenario Proof.

**[Propose a scenario →](https://github.com/jakbuczarnecki/intergrax/issues/new?template=scenario_proposal.yml)**

---

## H. Proof Library vs evidence dashboard

**[PROOFS.md](PROOFS.md)** remains the canonical evidence-status source. These surfaces answer different reader questions:

| Surface | Framing | Answers |
| --- | --- | --- |
| **Proof Library** (this page) | Problem-first | *Show me difficult problems and executable scenarios.* |
| **[PROOFS.md](PROOFS.md)** | Evidence-first | *Show me exactly what is proven and what Intergrax is allowed to claim.* |

Use [PROOFS.md](PROOFS.md) for implementation status, bounded verification, partial capability, planned work, and claims that are not currently supported. Scenario Proofs are distinct from product proofs — for the active reference product and its bounded product paths, see [LKW Product Tour](../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md).

---

## I. Related routes

| Route | Purpose |
| --- | --- |
| [README](../../../README.md) | First contact — Scenarios, Products, Platform entry points |
| [PROOFS.md](PROOFS.md) | Public evidence-and-claims dashboard |
| [LKW Product Tour](../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) | Active reference product walkthrough |
| [LKW Quick Start](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) | Runnable product evaluation path |
| [Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md) | Platform architecture mental model |
| [Platform Proof Library (maintainers)](../../../platform_proofs/README.md) | Technical proof infrastructure and authoring |
