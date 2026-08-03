<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Intergrax Public Positioning Contract

This document is the canonical public-positioning contract for Intergrax. It governs:

- root README positioning;
- public overview documents;
- outreach introductions;
- design-partner language;
- repository descriptions;
- future demo and landing-page copy.

It does **not** replace:

- architecture canon;
- implementation plans;
- proof evidence;
- license;
- collaboration rules;
- detailed proof and claims status (owned by a later canonical proof-and-claims document).

---

## 1. Canonical public position

**Primary sentence:**

Intergrax helps teams build specialized agent applications without rebuilding the same policy, knowledge, evidence, integration, and execution foundations for every product.

**Category descriptor:**

Intergrax is a reusable Harness AI foundation for governed agent applications.

---

## 2. Problem

Building an impressive agent demo is relatively easy. Delivering a controlled application that a team can review, operate, and trust is difficult.

Teams repeatedly rebuild the same foundations for every new product: identity and permissions, tool access, knowledge retrieval, memory, policy enforcement, human-in-the-loop gates, trace and evidence collection, testing, and operational boundaries. This repeated infrastructure work slows each new product and makes review and governance inconsistent across applications.

---

## 3. Value

Intergrax allows teams to:

- focus on the concrete application and user workflow;
- reuse governed execution infrastructure;
- reuse knowledge, evidence, integration, and model boundaries;
- improve the shared platform through pressure from real products.

Do not claim a measured delivery-time reduction unless supported by the later proof-and-claims document.

---

## 4. What the customer or partner receives

The customer or partner receives a concrete solution to a concrete problem.
Intergrax is the reusable foundation used to deliver that solution.

Public messaging must not require the reader to buy into “Agent OS” as the product before understanding the concrete solution.

The internal architectural phrase “The harness is the product.” may remain in deep technical material, but it must not be the primary public market message.

---

## 5. Meaning of Harness AI

Harness AI is reusable infrastructure for governed agent applications. It provides shared foundations for:

- governed execution;
- policy and human-in-the-loop (HITL);
- trace and evidence;
- knowledge, RAG, and memory;
- tools and integrations;
- model portability;
- application hosting and runtime boundaries;
- testing and operational verification.

Harness AI is explained after the problem and value because readers should first understand what concrete outcome Intergrax helps deliver. The explanation does not require knowledge of internal tier labels or component names.

---

## 6. Role of LKW

Local Knowledge Workspace (LKW) is:

- the current primary product-development and platform-validation program;
- the first concrete product proof of Intergrax;
- the driver of reusable platform improvements through real product requirements;
- currently a Backend Product Alpha / MVP under active development.

LKW is **not**:

- a finished SaaS;
- a product that has completed real-user validation;
- a product that has completed commercial validation;
- proof that all planned providers, live access paths, or Hybrid Ask capabilities are complete;
- the final definition of every product that can be built with Intergrax.

---

## 7. Primary audiences

The positioning is aimed narrowly at:

1. teams building specialized agent-backed applications that need governance, evidence, knowledge access, or controlled tool execution;
2. AI platform engineers and architects evaluating reusable agent-application infrastructure;
3. technical design partners with a concrete workflow worth validating.

It is **not** aimed at “everyone,” generic consumers, or every possible AI project.

---

## 8. Differentiators

### Application-first

Real applications and user workflows lead development.

### Governed execution by default

Policy, HITL, trace, and evidence are execution concerns, not optional decorations.

### Reusable delivery foundation

Multiple products reuse shared infrastructure instead of rebuilding the same foundations.

### Explicit responsibility boundaries

Applications own product environment; orchestration coordinates work; agents make domain decisions; the harness controls execution and evidence.

---

## 9. Current-stage language

**Allowed public stage descriptions:**

- source-available
- active R&D
- reusable harness/platform baseline
- LKW Backend Product Alpha
- product-validation program
- technical evaluation
- design-partner discovery

**Require qualification when using:**

- production-grade
- enterprise
- secure
- certified
- validated
- complete
- ready

**Stage guardrails:**

- `production-grade Harness AI` is the strategic destination;
- it is not an unrestricted public claim about every current component or deployment;
- maturity scores describe internal evidence models and do not equal product certification;
- real-user validation and commercial validation are not complete.

---

## 10. What Intergrax is not

Intergrax is not currently positioned as:

- a finished SaaS;
- a generic no-code agent builder;
- a direct replacement for LangChain, LangGraph, CrewAI, or every agent framework;
- a production certification;
- a compliance or security certification;
- a commercially validated product;
- a completed implementation of every cataloged integration;
- a guarantee that every future product will be faster or safer without measurement.

Other frameworks and tools may suit different needs. This positioning does not dismiss them.

---

## 11. Messaging order

Public messaging must follow this order:

```text
1. Problem
2. User or team outcome
3. Concrete product or workflow
4. Current proof and honest status
5. Harness AI explanation
6. Architecture deep dive
7. Evaluation, pilot, or technical-review next step
8. License details
```

Legal and technical details remain available but must not dominate the first contact.

---

## 12. Source-of-truth boundaries

| Topic | Owner |
|-------|-------|
| Public positioning | This document |
| Detailed implementation status | Owning implementation plans |
| Detailed proof claims | Later canonical proof-and-claims document |
| License rights | [`LICENSE`](../../LICENSE) |
| Collaboration permissions | [`COLLABORATION.md`](../../COLLABORATION.md) |
| Architecture details | Architecture canon |
| Root README | Will consume this positioning in a later alignment task; not modified by this contract |

Do not duplicate detailed task statuses or proof tables here.
