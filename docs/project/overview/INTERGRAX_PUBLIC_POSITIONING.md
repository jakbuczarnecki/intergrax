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
| Current phase | PX-12 — CHANGES_REQUIRED |
| Proof ownership and roadmap deduplication | PX-10 — ACCEPTED / CLOSED |
| Public-language cleanup | PX-9 — ACCEPTED / CLOSED |
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

## Public status summary boundary

- README and first-contact documents may show overall public classification.
- They may link to `PROOFS.md`.
- They do not copy product-roadmap task status.
- They do not copy phase or dependency tables.
- Current implementation detail belongs to the owning roadmap.
- Public proof detail belongs to `PROOFS.md`.
- Summaries change only when their current wording becomes inaccurate.

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

### Root first-screen CTA row

**[Try LKW](#try-lkw)** · [See the LKW workflow](../maintainers/public-adoption/LKW_PRODUCT_TOUR.md) · [Choose your path](#choose-your-path)

`Try LKW` is the only first-screen primary CTA.
Product Tour is the lower-friction orientation route.
Choose your path is the audience router.
Token Optimization remains discoverable in its dedicated lower section.
Architecture and proof links remain available lower in README.
Neither Token Optimization, architecture nor PROOFS appears as an equal first-screen CTA.

### Primary-action rule

Every major reader-facing document must identify:

- one primary next action;
- its exact destination;
- secondary conditional routes separately.

Freeze these reader paths:

```text
General first contact:
Try LKW
→ README.md#try-lkw

Product orientation:
See the LKW workflow
→ LKW_PRODUCT_TOUR.md

Run the primary product:
Run the supported LKW path
→ applications/local_workspace_application/docs/product/QUICKSTART.md

Review LKW after the supported run:
Inspect bounded LKW evidence
→ applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md

Builder:
Start with the Builder Quick Start
→ BUILDER_QUICKSTART.md

Builder after orientation:
Plan a bounded build
→ BUILD_WITH_INTERGRAX.md

Architect:
Review architecture boundaries
→ ARCHITECTURE_OVERVIEW.md

Architect after boundaries:
Review current proof
→ PROOFS.md

CTO, product lead or technical buyer:
Assess concrete workflow fit
→ USE_CASES.md

Buyer after apparent fit:
Review current proof
→ PROOFS.md

Partner, integrator or design partner:
Prepare a pilot brief
→ PARTNERS.md#pilot-brief

Category-level uncertainty:
Compare common approaches
→ WHY_INTERGRAX.md#where-intergrax-fits

Named modern alternatives:
Compare modern alternatives and trade-offs
→ ALTERNATIVES_AND_TRADEOFFS.md

Concrete fit after category orientation:
Check workflow fit
→ USE_CASES.md

Token Optimization evaluator:
Explore Token Optimization
→ docs/project/capabilities/token_optimization/README.md

Deep technical reviewer:
Open the Technical Documentation Map
→ docs/project/technical/DOCUMENTATION_MAP.md

Permission uncertainty:
Review collaboration and legal terms
→ COLLABORATION.md and LICENSE
```

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
| Contributor or deep technical reviewer | Contribution boundaries, license scope, and where deep technical material lives | Open the technical documentation map | `docs/project/technical/DOCUMENTATION_MAP.md` |

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

Express differentiation through outcomes and responsibility boundaries, not subsystem names or market labels. The public spine is:

1. **Product owns meaning; platform owns enforcement** — applications define business rules, permissions, and acceptable outcomes; Intergrax supplies reusable enforcement mechanisms at configured execution boundaries.
2. **Governance spans explicit execution boundaries** — policy evaluation, approval, and denial attach to named execution steps rather than living only in ad hoc application code.
3. **Consequential external effects cross an explicit governed boundary** — meaningful side effects and tool actions are authorized and recorded through platform mechanisms on wired paths.
4. **Execution has structural identity and canonical history** — runs, attempts, and events carry typed identity so history can be reconstructed without treating vendor telemetry as the source of truth.
5. **Recovery distinguishes retry, idempotency, compensation, degradation, and HITL** — failure handling is classified and bounded rather than left to hidden agent loops.
6. **Important execution can produce structured evidence, not telemetry alone** — governance and execution transitions can be correlated with persisted evidence where mechanisms are connected.
7. **Agent authors own domain behavior; agents are not private runtimes** — agents declare contracts and domain decisions; the platform owns safe execution, not a second hidden runtime per agent.

These are architectural responsibility choices. They are not claims that every boundary is universally complete, that competitors cannot implement similar patterns, or that Intergrax is uniquely capable because it implements a listed mechanism.

Do not claim:

- guaranteed faster delivery;
- guaranteed safety;
- universal production readiness;
- enterprise readiness;
- completed market validation;
- commercial validation;
- universal superiority over other frameworks;
- measured cross-product reuse or compounding value without evidence.

Do not claim a measured delivery-time reduction unless supported by the proof-and-claims model.

---

## Category and alternative positioning

Intergrax maintains **two governed comparison surfaces**. They serve different reader questions and must not be merged into one scorecard.

### A. Category comparison

Owner: [`WHY_INTERGRAX.md#where-intergrax-fits`](WHY_INTERGRAX.md#where-intergrax-fits)

The category comparison compares common solution **categories** by primary responsibility, what the adopting team still owns, and when the approach may fit. It does not name vendors and does not publish a feature winner scorecard.

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

Freeze these category rules:

- comparisons are category-level, not vendor-level;
- categories overlap;
- approaches may be combined;
- Intergrax is not universally superior;
- no feature parity is claimed;
- no market validation is inferred;
- no competitor performance, pricing, security or maturity claim is made;
- no vendor logo or trademark is used;
- no scorecard, winner or ranking is published;
- Intergrax maturity and proof boundaries remain visible;
- modern agent frameworks and platforms may include persistence, HITL, tracing, guardrails, workflows, and other runtime facilities — category comparison is about **primary responsibility**, not missing features.

The category comparison is a responsibility map, not a claim that Intergrax already provides every listed foundation completely or at production maturity.

### B. Named alternatives comparison

Owner: [`ALTERNATIVES_AND_TRADEOFFS.md`](ALTERNATIVES_AND_TRADEOFFS.md)

The named-alternatives document may name real frameworks and platforms. It answers when another modern stack may be the better choice and when Intergrax may be worth evaluating. It is decision-oriented, not competitive marketing.

Freeze these named-comparison rules:

1. Primary-source-backed factual claims only.
2. Include an explicit **externally verified on** date for competitor capability facts.
3. No feature winner scorecard.
4. No blanket checkmark or cross comparisons.
5. State where an alternative is a better choice.
6. State where Intergrax has a different responsibility model.
7. Do not claim unique capability merely because Intergrax implements it.
8. No universal superiority claims.
9. Avoid volatile pricing, benchmark, security, maturity, or market-share claims.
10. Separate competitor capability facts, Intergrax architectural interpretation, and Intergrax evidence boundary.
11. If a factual external claim cannot be supported by the accepted COMM-1 source set, omit it rather than infer it.
12. **Harness AI** remains descriptive only, never the differentiator.

Named competitor claims must remain conservative. Intergrax claims in this surface remain bounded by [`PROOFS.md`](../proofs/PROOFS.md).

## Category clarification

Intergrax is **not**:

- merely a single agent framework;
- only a RAG toolkit;
- currently a finished knowledge SaaS;
- a no-code builder;
- a claim to replace every existing framework.

Intergrax **combines** reusable application foundations with concrete product paths and bounded proofs. Category-level comparison belongs in [WHY_INTERGRAX.md#where-intergrax-fits](WHY_INTERGRAX.md#where-intergrax-fits). Named modern alternatives belong in [ALTERNATIVES_AND_TRADEOFFS.md](ALTERNATIVES_AND_TRADEOFFS.md). Do not name or attack specific competitors in first-contact copy or this contract's category section.

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

License and collaboration ownership remain in [`LICENSE`](../../../LICENSE) and [`COLLABORATION.md`](../community/COLLABORATION.md).

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
| Category-level comparison (no vendor names) | `WHY_INTERGRAX.md#where-intergrax-fits` |
| Named modern alternatives and decision trade-offs | `ALTERNATIVES_AND_TRADEOFFS.md` |
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
