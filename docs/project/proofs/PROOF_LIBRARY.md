<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Proof Library

**Real problems. Executable evidence.**

The Intergrax Proof Library is a problem-first catalog of **Scenario Proofs** - executable falsification attempts against bounded real-world system claims. Each scenario targets a difficult failure mode that a simple LLM workflow is not sufficient to handle safely or reliably.

Intergrax is easier to understand by watching it handle hard problems than by reading capability lists. Accepted Scenario Proofs are designed to be run, inspected, challenged, and reproduced: what was tested, how it was attacked, the evidence produced, the verdict, stated limitations, source, and a reproduction path.

Scenario Proofs stress-test bounded system guarantees under adversarial conditions. They are not products, commercial validation, or evidence of product-market fit.

---

## A. Scenario catalog

Start with the problems. Each scenario below targets a concrete failure mode that a simple LLM workflow is not sufficient to handle safely or reliably.

The library operating model is ready. Accepted Scenario Proofs appear here only after passing canonical acceptance gates - with evidence, verdict, report, and reproduction routes.

**Current status:** no accepted Scenario Proofs are published yet. Flagship scenarios are in development.

| Preview | Scenario | What makes it hard | Status |
| --- | --- | --- | --- |
| <a href="../../../platform_proofs/scenarios/ai_incident_investigation/README.md"><img src="../../../platform_proofs/scenarios/ai_incident_investigation/assets/proof-story-light.svg" width="180" alt="AI Incident Investigation scenario preview"></a> | [**AI Incident Investigation with Independent Verification**](../../../platform_proofs/scenarios/ai_incident_investigation/README.md) | Plausible correlation points to the wrong root cause. The system must gather and falsify evidence instead of confidently guessing. | **In development** - FULL-1 RESOLVED and FULL-2 UNRESOLVED implemented and executable; public Scenario Proof not yet accepted |
| <a href="../../../platform_proofs/scenarios/indirect_prompt_injection/README.md"><img src="../../../platform_proofs/scenarios/indirect_prompt_injection/assets/scenario-overview.png" width="180" alt="Indirect Prompt Injection scenario preview"></a> | [**Indirect Prompt Injection with Governed Action Prevention**](../../../platform_proofs/scenarios/indirect_prompt_injection/README.md) | Hostile retrieved data may genuinely fool the model into requesting a dangerous write. Governance must still stop the real side effect. | **Implementation initialized** - scenario architecture accepted; executable business proof not yet implemented |
| <a href="../../../platform_proofs/scenarios/verified_product_identification/README.md"><img src="../../../platform_proofs/scenarios/verified_product_identification/assets/scenario-overview.png" width="180" alt="Verified Product Identification scenario preview"></a> | [**Verified Product Identification at Catalog Scale**](../../../platform_proofs/scenarios/verified_product_identification/README.md) | Millions of noisy offers and near-identical variants. Top semantic retrieval must not be mistaken for verified product identity. | **Design ready for quality gate** - real 3.77M-offer dataset foundation validated; solution architecture documented; implementation not initialized |

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

Links appear only when artifacts exist - not for placeholder or in-design work.

---

## B. Featured scenarios in development

### AI Incident Investigation with Independent Verification

> **Can an AI investigate an operational incident without turning correlation into a confident false diagnosis?**

Initial operational signals make workload overload plausible. Evidence is conflicting, stale, and incomplete. Independent verification must challenge unsupported causality, gather targeted evidence to distinguish competing hypotheses, and produce a bounded **RESOLVED** or honest **UNRESOLVED** outcome.

**Status:** in development - FULL-1 RESOLVED and FULL-2 UNRESOLVED implemented and executable via platform proof runner. No accepted published evidence bundle, report, or reproduction route yet.

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
    alt="AI incident investigation scenario - operational signals, conflicting evidence, independent verification, and bounded RESOLVED or UNRESOLVED outcomes."
  >
</picture>
</a>

[View full-size diagram](../assets/public/readme/fullsize/scenario-ai-incident-investigation.md)

**[Scenario design document](../../../platform_proofs/scenarios/ai_incident_investigation/README.md)**

---

### Indirect Prompt Injection with Governed Action Prevention

> **Can an AI agent be fooled - and still be prevented from causing real harm?**

An autonomous order assistant legitimately reads external order notes.
A hostile instruction hidden in those notes may fool the real model into proposing a shipping-address change.

The scenario does not rely on the model detecting the attack.

The write tool stays available.
The model may genuinely request it.
Intergrax governance evaluates the tool invocation before execution and blocks the unauthorized write.

The model may lose the battle.
The execution boundary must still win.

**Status:** implementation initialized - scenario architecture accepted; business implementation not yet completed; no verified proof run yet.

<a href="../../../platform_proofs/scenarios/indirect_prompt_injection/assets/scenario-overview.png">
  <img
    src="../../../platform_proofs/scenarios/indirect_prompt_injection/assets/scenario-overview.png"
    alt="Indirect Prompt Injection Defense scenario - hostile external data influences the model while trusted governance prevents unauthorized write execution"
  >
</a>

[View full-size scenario overview](../../../platform_proofs/scenarios/indirect_prompt_injection/assets/scenario-overview.png)

**[Scenario design document](../../../platform_proofs/scenarios/indirect_prompt_injection/README.md)**

---

### Verified Product Identification at Catalog Scale

> **Can a system verify product identity from incomplete descriptions against millions of noisy catalog offers - without mistaking the top search result for a verified match?**

A technician or buyer describes a part imperfectly. The catalog holds 3.77 million real product offers with missing fields, conflicting attributes, and near-identical variants. Several candidates look semantically close. The scenario tests whether independent retrieval channels plus evidence-based verification can establish identity - or honestly refuse when evidence is contradictory, ambiguous, or insufficient.

The WOW moment is uncomfortable and simple:

```text
top search result  ≠  verified product
```

**Status:** design ready for quality gate - real WDC dataset foundation validated (3,770,377 offers); solution architecture documented; implementation not initialized; no executable proof, evidence, or report yet.

<a href="../../../platform_proofs/scenarios/verified_product_identification/assets/scenario-overview.png">
  <img
    src="../../../platform_proofs/scenarios/verified_product_identification/assets/scenario-overview.png"
    alt="Verified Product Identification at Catalog Scale - multi-channel product retrieval, reranking, evidence verification and bounded outcomes"
  >
</a>

[View full-size scenario overview](../../../platform_proofs/scenarios/verified_product_identification/assets/scenario-overview.png)

**[Scenario design document](../../../platform_proofs/scenarios/verified_product_identification/README.md)**

---

## C. What is a Scenario Proof?

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

Scenario Proofs aim to demonstrate **difficult system guarantees** - governance under conflict, unsafe-action prevention, recovery, evidence admissibility, false confident diagnosis, and similar real-world conditions - rather than basic LLM orchestration or happy-path demos.

> [!IMPORTANT]
> Scenario Proofs stress-test bounded system guarantees under adversarial conditions and produce inspectable evidence. **Scenario Proofs are not products.** They do not substitute for real-user validation, commercial validation, or production readiness. Product, user, and commercial validation remain separate evidence classes.

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

## D. What makes a scenario worth publishing?

A scenario belongs in the library when it exposes a **meaningful failure risk** that simple AI is insufficient to handle safely:

| Criterion | Why it matters |
| --- | --- |
| **Real workflow stakes** | Wrong answers or unsafe actions have operational consequences |
| **Adversarial conditions** | Conflicting, stale, incomplete, or misleading evidence is part of the problem |
| **Bounded claim** | Exactly what is - and is not - being demonstrated is explicit |
| **Executable falsification** | A skeptical reviewer can run, inspect, and challenge the result |
| **Honest outcomes** | PASS, FAIL, or bounded UNRESOLVED must be earned - not narrated |

Maintainer authoring workflow and technical infrastructure live in the internal [Platform Proof Library](../../../platform_proofs/README.md) gateway.

---

## E. How to read a proof

Every published Scenario Proof should let you inspect:

| Inspect | What you learn |
| --- | --- |
| **Real problem** | The workflow or decision the system must support |
| **Why failure matters** | Consequences if the AI or system gets it wrong |
| **Bounded claim** | Exactly what is - and is not - being demonstrated |
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
| **UNRESOLVED** | Critical distinguishing evidence is unavailable or hypotheses remain indistinguishable - no confident guessing |

A scenario visual may show RESOLVED or UNRESOLVED branches as **possible bounded outcomes** - that describes scenario semantics, not a current achieved PASS.

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

Use [PROOFS.md](PROOFS.md) for implementation status, bounded verification, partial capability, planned work, and claims that are not currently supported. Scenario Proofs are distinct from product proofs - for the active reference product and its bounded product paths, see [LKW Product Tour](../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md).

---

## I. Related routes

| Route | Purpose |
| --- | --- |
| [README](../../../README.md) | First contact - Scenarios, Products, Platform entry points |
| [PROOFS.md](PROOFS.md) | Public evidence-and-claims dashboard |
| [LKW Product Tour](../../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) | Active reference product walkthrough |
| [LKW Quick Start](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) | Runnable product evaluation path |
| [Architecture Overview](../architecture/ARCHITECTURE_OVERVIEW.md) | Platform architecture mental model |
| [Platform Proof Library (maintainers)](../../../platform_proofs/README.md) | Technical proof infrastructure and authoring |
