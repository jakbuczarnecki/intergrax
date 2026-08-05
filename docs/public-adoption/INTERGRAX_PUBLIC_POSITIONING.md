<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Public Positioning Contract

This document is a **Layer 5 maintainer contract**. It governs exact first-contact copy, product hierarchy, audience value, and CTA language for:

- root README positioning (consumed in PX-2);
- public overview documents;
- outreach introductions;
- design-partner language;
- repository descriptions;
- future demo and landing-page copy.

It is **not** a normal public-reader route.

It does **not** replace:

- architecture canon;
- implementation plans;
- proof evidence;
- license;
- collaboration rules;
- detailed proof and claims status (`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`, `PROOFS.md`).

---

## At a glance

| Item | Status |
|------|--------|
| Contract status | ACTIVE |
| Current phase | PX-9 — READY_FOR_REVIEW |
| Primary product path | Local Knowledge Workspace (LKW) |
| Secondary capability path | Token Optimization |
| Current primary CTA | Try LKW |
| Product-orientation CTA | See the LKW workflow |
| Supported product quickstart | PX-3 — ACCEPTED / CLOSED |
| LKW visual evidence | PX-4 — ACCEPTED / CLOSED |
| LKW Product Tour | PX-5 — ACCEPTED / CLOSED |
| Builder Quick Start | PX-6 — ACCEPTED / CLOSED |
| Architect route | `ARCHITECTURE_OVERVIEW.md` |
| Buyer route | `USE_CASES.md` |
| Partner route | `PARTNERS.md` |
| Architect, buyer and partner routes | PX-7 — ACCEPTED / CLOSED |
| Category comparison | PX-8 — ACCEPTED / CLOSED |
| Category comparison owner | `WHY_INTERGRAX.md#where-intergrax-fits` |
| Root README application | APPLIED IN PX-2 — ACCEPTED / CLOSED |
| External reader validation | NOT_STARTED |
| Real-user validation | INCOMPLETE |
| Commercial validation | INCOMPLETE |

---

## Canonical first-contact message

### Primary sentence

Intergrax helps teams build AI applications that can use their knowledge and tools while keeping access, actions, and evidence under control.

### Supporting sentence

Teams reuse shared policy, knowledge, integration, execution, and evidence foundations instead of rebuilding them for every product.

### LKW product sentence

Local Knowledge Workspace (LKW) is the primary product path: a private-by-default workspace for adding approved knowledge sources, asking questions, and receiving grounded answers with source references and inspectable evidence.

### Current-status sentence

LKW is a Backend Product Alpha / MVP under active development. The current bounded product proof covers indexed knowledge workflows; complete live or hybrid access, finished end-user packaging, real-user validation, and commercial validation are not complete.

### Category descriptor

Intergrax is a reusable foundation for governed AI applications.

The category descriptor appears after the concrete problem, outcome, and LKW explanation — not as the opening headline.

---

## Public reader language

Freeze these rules for normal public-reader copy:

- user outcome before subsystem name;
- responsibility before implementation detail;
- one plain-language explanation before a necessary acronym;
- reader-friendly visible link labels even when the target filename is technical;
- provider names only where they identify actual proof scope;
- task IDs only in maintainer and deep technical material;
- current limitations remain visible after simplification;
- simplification must not weaken legal, proof or maturity boundaries;
- diagrams and visuals remain part of the public product surface.

Prohibit in Layer 1–3 normal reader copy:

```text
PX task identifiers
TOKEN phase identifiers
CTX-UCL identifiers
Tier-number architecture labels
READY_FOR_REVIEW
ACCEPTED / CLOSED
BLOCKED_ON_*
Cursor operators
lab host
README Quick start
```

ProofReceipt is a deep technical term. Normal reader pages use persisted execution evidence or execution receipt where needed.

Harness AI is not first-contact or at-a-glance terminology. It may remain in intentionally deep architecture material.

---

## Product and message hierarchy

Public messaging must follow this order:

```text
1. User problem and desired outcome
2. LKW as the primary product path
3. Current bounded result and honest maturity
4. Current primary CTA
5. Intergrax as the reusable foundation
6. Token Optimization as a secondary capability
7. Architecture and proof routes
8. Partner, permission and license routes
```

Frozen ownership:

```text
LKW = primary public product CTA
Token Optimization = secondary platform-capability CTA
Architecture = deeper evaluation route
Full certification proof = reviewer route, not first product introduction
```

Product trial and platform evaluation are different routes. A certification runbook must not replace a product tour.

---

## LKW visual evidence rule

The first product visual shows a concrete indexed LKW result: an approved sample source, managed intake and indexing, a user question, a grounded answer, a source reference, and persisted Ask-run verification. The platform architecture visual remains later in the README, after the concrete product workflow.

The PX-4 visual is a neutral documentation representation, not a UI screenshot. It makes no Hybrid, live-provider, or production claim.

---

## Calls to action

### Current primary CTA

**Try LKW**

The supported product quickstart path is accepted and closed in PX-3. It remains subject to the final executable-claim audit before external sessions and must not become a documentation-only path.

### Builder CTA

**Start building with Intergrax**

→ `BUILDER_QUICKSTART.md`

This is a secondary developer route and must not be promoted above the primary **Try LKW** product CTA.

### Product-orientation CTA

**See the LKW workflow**

→ `LKW_PRODUCT_TOUR.md`

### LKW route separation

```text
Product Tour:
understand the experience

Quick Start:
run the supported product path

Platform Proof:
review bounded technical evidence
```

### Secondary CTA

**Explore Token Optimization**

### Deeper technical CTA

**Review architecture and proofs**

### Prohibited as public primary CTA

The following must **not** be the first product action for normal readers:

- Run echo.basic
- Start lab_application
- Evaluate the lab
- Run the platform smoke test

`echo.basic` is not a primary public CTA. `lab_application` is not a primary public CTA. Echo and lab may remain advanced platform smoke or maintainer diagnostics.

---

## Audiences

| Audience | Primary value they should understand | Primary next action | Primary destination |
|----------|--------------------------------------|---------------------|---------------------|
| Potential LKW user | A governed knowledge-workflow product path exists; LKW is the primary product; maturity is honest and bounded | See the LKW workflow | `LKW_PRODUCT_TOUR.md` |
| AI engineer or developer | Intergrax is a serious foundation for governed AI applications; builder paths exist separately from product trial | Open the bounded builder route | `BUILDER_QUICKSTART.md` |
| Architect or platform engineer | Governance, architecture, and proof boundaries at a high level without first-contact subsystem jargon | Review architecture and current proof boundaries | `ARCHITECTURE_OVERVIEW.md` |
| CTO, product lead or technical buyer | Problem, value, maturity, and honest proof status for decision-making | Assess use-case fit and current evidence | `USE_CASES.md` |
| Partner, integrator or design partner | Partner fit, pilot workflow, and what Intergrax does not promise publicly | Review partner fit and prepare a pilot brief | `PARTNERS.md` |
| Contributor or deep technical reviewer | Contribution boundaries, license scope, and where deep technical material lives | Open the technical documentation map | `docs/DOCUMENTATION_MAP.md` |

The positioning is **not** aimed at “everyone,” generic consumers, or every possible AI project.

```text
Product user:
Try LKW

Product observer:
See the LKW workflow

Builder:
Start building with Intergrax

Architect:
Review architecture and proof boundaries

Buyer:
Assess use-case fit and evidence

Partner:
Review pilot fit and prepare a brief

Deep technical reviewer:
Open the technical documentation map
```

---

## Harness AI terminology

Harness AI is an **optional explanatory term only**. It is:

- not the headline;
- not the primary sentence;
- not required to understand LKW;
- explained after the product and user outcome;
- defined as reusable infrastructure for governed AI applications.

Harness AI must not be presented as a recognized market category without evidence. Legitimate deep technical use of the term may remain in architecture and reviewer material.

---

## Differentiators

Express these through user outcomes, not subsystem names:

1. **Product-first development** — real applications and user workflows lead development.
2. **Controlled execution and evidence** — policy, human-in-the-loop gates, trace, and evidence are execution concerns, not optional decorations.
3. **Reusable foundations across applications** — multiple products reuse shared infrastructure instead of rebuilding the same foundations.
4. **Clear responsibility boundaries** — applications own product environment; orchestration coordinates work; agents make domain decisions; the harness controls execution and evidence.

Do not claim:

- guaranteed faster delivery;
- guaranteed safety;
- universal production readiness;
- enterprise readiness;
- completed market validation;
- commercial validation;
- universal superiority over other frameworks.

Do not claim a measured delivery-time reduction unless supported by the proof-and-claims model.

---

## Category and alternative positioning

The category comparison is owned by [`WHY_INTERGRAX.md#where-intergrax-fits`](../../WHY_INTERGRAX.md#where-intergrax-fits). It compares common solution categories by primary responsibility, what the adopting team still owns, and when the approach may fit.

Freeze these categories:

```text
Finished AI SaaS
Workflow automation platform
RAG or knowledge toolkit
Agent framework
Custom in-house foundation
Intergrax reusable governed application foundation
```

For every category comparison require:

```text
primary value
what the adopting team still owns
when the approach may fit
```

Freeze these rules:

- comparisons are category-level, not vendor-level;
- categories overlap;
- approaches may be combined;
- Intergrax is not universally superior;
- no feature parity is claimed;
- no market validation is inferred;
- no competitor performance, pricing, security or maturity claim is made;
- no vendor logo or trademark is used;
- no scorecard, winner or ranking is published;
- Intergrax maturity and proof boundaries remain visible.

The comparison is a responsibility map, not a claim that Intergrax already provides every listed foundation completely or at production maturity. The detailed route is [WHY_INTERGRAX.md#where-intergrax-fits](../../WHY_INTERGRAX.md#where-intergrax-fits).

## Category clarification

Intergrax is **not**:

- merely a single agent framework;
- only a RAG toolkit;
- currently a finished knowledge SaaS;
- a no-code builder;
- a claim to replace every existing framework.

Intergrax **combines** reusable application foundations with concrete product paths and bounded proofs. Category-level comparison belongs in [WHY_INTERGRAX.md#where-intergrax-fits](../../WHY_INTERGRAX.md#where-intergrax-fits); do not name or attack specific competitors here.

---

## Prohibited first-contact patterns

The canonical first-contact section must **not** contain:

```text
Harness AI
Agent OS
Tier-0
Tier-1
Tier-2
Tier-3
Nexus
echo.basic
lab_application
internal PX/TOKEN/CTX task IDs
a long capability catalog
a full architecture diagram
license-first wording
```

These items may appear in appropriately deeper documents.

---

## Maturity and legal boundaries

**Allowed public stage descriptions:**

- source-available
- active R&D
- reusable harness/platform baseline
- LKW Backend Product Alpha / MVP
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
- real-user validation is incomplete;
- commercial validation is incomplete;
- no production-certification claim;
- no security or compliance certification claim;
- no measured speed or savings claim without evidence.

Value must appear before detailed limitations. Limitations remain visible.

License and collaboration ownership remain in [`LICENSE`](../../LICENSE) and [`COLLABORATION.md`](../../COLLABORATION.md).

---

## Messaging order

The required public sequence:

```text
1. User problem
2. Concrete outcome
3. LKW product workflow
4. One current next action
5. Current maturity and limitations
6. Intergrax foundation explanation
7. Token Optimization capability
8. Architecture and proof depth
9. Partner or builder route
10. License and permission details
```

Do not place license or architecture before the concrete product explanation.

---

## Source-of-truth boundaries

| Topic | Owner |
|-------|-------|
| Exact first-contact copy, product hierarchy, audience value and CTA language | This document (`INTERGRAX_PUBLIC_POSITIONING.md`) |
| PX phase status and experience gates | `PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` |
| Document layers and reader routes | `PUBLIC_DOCUMENTATION_ARCHITECTURE.md` |
| Current public proof status | `PROOFS.md` |
| Public product-validation direction | `ROADMAP.md` |
| Legal rights and restrictions | `LICENSE` |
| Detailed implementation status | Owning implementation plans |
| Detailed proof claims | `PUBLIC_PROOF_AND_CLAIMS_MODEL.md` |
| Collaboration permissions | `COLLABORATION.md` |
| Architecture details | Architecture canon |
| Root README | Consumes this contract in PX-2 and is accepted / closed. |

Do not duplicate detailed task statuses or proof tables here.
