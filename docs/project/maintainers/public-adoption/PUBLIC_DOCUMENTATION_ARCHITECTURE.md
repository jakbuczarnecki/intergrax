<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Public Documentation Architecture

This document is the canonical maintainer-facing contract for Intergrax public documentation structure. It owns:

- public documentation layers;
- public document responsibilities;
- reader-intent routing;
- proof classification;
- relationship between the public map and technical map;
- placement rules for LKW and Token Optimization;
- migration rules for the new root README and future public documents.

It does **not** replace:

- `INTERGRAX_PUBLIC_POSITIONING.md` — exact first-contact message, product hierarchy, audience value, CTA language, and placement of Harness AI terminology;
- implementation plans;
- detailed proof claims;
- architecture canon;
- `LICENSE`;
- `COLLABORATION.md`.

---

## 1. Objective

Public documentation must let a reader understand:

```text
what Intergrax solves
→ what can be seen or evaluated
→ what is currently proven
→ where deeper technical information lives
→ what next action is available
```

The reader must not need to understand internal tiers, Cursor workflow, issue automation, or implementation task IDs before understanding project value.

---

## Frozen documentation ownership model

Freeze three documentation layers. The project-documentation/public-experience
stream owns project reader experience and the editorial presentation of project
projections. The relevant module or product stream owns implementation truth.

### PROJECT DOCUMENTATION

This stream owns project-level reader experience, including:

```text
README.md
docs/project/README.md
docs/project/overview/WHY_INTERGRAX.md
docs/project/overview/USE_CASES.md
docs/project/overview/ROADMAP.md
docs/project/overview/FAQ.md
docs/project/architecture/ARCHITECTURE_OVERVIEW.md
docs/project/builders/BUILDER_QUICKSTART.md
docs/project/builders/BUILD_WITH_INTERGRAX.md
docs/project/builders/EVALUATION_GUIDE.md
docs/project/community/PARTNERS.md
docs/project/community/COLLABORATION.md
docs/project/community/PUBLIC_DOCUMENTATION_MAP.md
```

Its purpose is positioning, reader routing, project architecture explanation,
use-case communication, builder experience, buyer and partner experience,
public navigation, and visual communication.

### PROJECT PROJECTIONS

Project projections are reader-facing summaries owned editorially by this
stream. Their factual content must come from accepted module evidence. Examples
include `docs/project/proofs/PROOFS.md`, the LKW Product Tour, README LKW
sections, capability spotlights, project-level capability summaries,
integration/support summaries, and selected diagrams whose meaning depends on
completed capabilities.

Presentation ownership does not transfer implementation truth. A projection
may be promoted only from accepted evidence: roadmap implementation status,
code existence, or unit tests alone is insufficient when a claim requires
live/product proof. Removing or weakening a claim is allowed when the evidence
boundary requires it.

### MODULE SOURCES OF TRUTH

Module or product streams own detailed implementation status, technical proof,
benchmarks, capability architecture, and module-specific execution truth. This
includes, for example:

```text
LKW:
applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md
module architecture, operational proof, and lifecycle details

Token Optimization:
docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md
architecture, model qualification, benchmarks, and technical proof artifacts

Vendor Knowledge:
provider-neutral architecture, provider/plugin roadmaps, reconciliation,
provider proofs, and adapter-specific behavior

Other module families:
UCL / Context, LangChain Independence, runtime/model infrastructure,
and future platform capabilities
```

The project-documentation stream may read, link, cite, and summarize accepted
evidence. It must not casually become the implementation owner of module
sources of truth. Module implementation roadmaps remain module-owned and are not
public claim dashboards.

### Public Evidence Packet

The compact handoff from a module stream to project documentation may provide:

```text
CAPABILITY
STATUS
ACCEPTED SHA
PROVEN
USER-VISIBLE OUTCOME
NOT PROVEN
VERIFICATION PATH
PUBLIC CLAIM CANDIDATE
VISUAL OPPORTUNITY
PUBLIC DOCS POTENTIALLY AFFECTED
```

This is a handoff contract, not a requirement to create evidence-packet files.

### Non-blocking rule and claim promotion

A module still being developed does not globally block unrelated Project
Documentation work. For example, unfinished mixed indexed plus live Hybrid Ask
may block only a mixed indexed-and-live claim, its evidence-result visual, or
promotion of the relevant `PROOFS.md` row. It must not block `WHY_INTERGRAX`,
generic architecture explanation, unrelated use cases, builder structure, FAQ,
partner material, navigation, or the visual system.

The frozen promotion pipeline is:

```text
module implementation
→ module acceptance
→ appropriate user-like/live proof where required
→ accepted evidence
→ project projection
→ optional README promotion
```

Implementation does not directly become a marketing claim.

---

## Proof and roadmap ownership

The public documentation ownership contract is:

```text
Product or capability roadmap
→ detailed implementation progress

Proof document and accepted evidence
→ demonstrated behavior

PUBLIC_PROOF_AND_CLAIMS_MODEL.md
→ status vocabulary and promotion rules

PROOFS.md
→ reader-facing public claims

README and overview documents
→ short compatible summaries
```

`PROOFS.md` may link to roadmaps.
`PROOFS.md` must not reproduce roadmap task tables.
The claims model must not reproduce roadmap phase tables.
Feature guides must not maintain parallel current phase snapshots.
Overview documents must not become status dashboards.
Accepted implementation does not automatically become accepted proof.
A roadmap update does not automatically require a public claim update.
A public claim update requires accepted evidence or a changed claim boundary.

### Roadmap owners

```text
LKW:
applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md

Token Optimization:
docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md
```

Future products must register exactly one detailed roadmap owner before
being added to `PROOFS.md`.

### Proof row contract

Every product or capability entry in `PROOFS.md` may contain:

```text
public classification
accepted evidence
limitation
verification route
detailed roadmap link
```

It may not contain:

```text
active task
next task
review-fix status
dependency graph copied from the roadmap
current subphase table
maintainer review state
```

### Visual ownership

The first `PROOFS.md` Mermaid diagram explains the stable directional
ownership model:

```text
roadmap
→ implementation
→ accepted evidence
→ public claim
→ overview
```

It is conceptual and stable. It must not show individual implementation-task
statuses. The second claim-to-proof lifecycle diagram remains unchanged.

### PX-9 closeout

```text
PX-9 — ACCEPTED / CLOSED

Acceptance evidence:
c9521fb3edace541e76259147073835c37c37b2e
```

### PX-10 closeout

```text
PX-10 — ACCEPTED / CLOSED

Implementation:
072b409ccd9fc73ea06e7b477d12b6a3fbf0a881

Review fix:
18626e91a24aa770f4011ab5294219fdfdcf6144
```

### PX-11 closeout

```text
PX-11 — ACCEPTED / CLOSED

Acceptance evidence:
b942121d0a509d059681d6f1df55ff09d7aaf6a2
```

## Internal readiness-review ownership

```text
PUBLIC_LAUNCH_CHECKLIST.md
→ single internal readiness record

PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md
→ PX phase state

EXTERNAL_READER_VALIDATION_PROTOCOL.md
→ external-session methodology

Product roadmaps
→ detailed implementation progress

PROOFS.md
→ public claims
```

PX-12 does not create another readiness report. Readiness checks are pinned
to one commit. Successful command execution and evidence review are different
modes and must be labeled. A technical failure is routed to its technical
owner; documentation must not hide a failed executable path. The final task
commit may become a PX-13 candidate only after independent acceptance.
PX-12 does not constitute external validation.

---

## 2. Public documentation layers

Freeze five layers.

### Layer 1 — Landing

**Owner:**

```text
README.md
```

**Status:** implemented in PX-2 — ACCEPTED / CLOSED

**Role:**

- first 30 seconds;
- problem and value;
- LKW product workflow;
- selected platform-capability spotlight;
- honest maturity;
- one clear next action;
- routes to deeper documents.

**Not:**

- full architecture canon;
- full implementation roadmap;
- full license;
- public issue automation;
- complete capability catalog.

### Layer 2 — Reader-intent documents

Target document responsibilities:

```text
WHY_INTERGRAX.md
PROOFS.md
ARCHITECTURE_OVERVIEW.md
BUILD_WITH_INTERGRAX.md
USE_CASES.md
PARTNERS.md
COLLABORATION.md
FAQ.md
ROADMAP.md
LKW_PRODUCT_TOUR.md
BUILDER_QUICKSTART.md
```

| Document | Responsibility | Status |
|----------|----------------|--------|
| `WHY_INTERGRAX.md` | Problem, value, audience, category fit and fair alternative positioning without tier jargon | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6; category positioning extended in PX-8 |
| `PROOFS.md` | Consolidated public proof status and claim boundaries | exists — `docs/project/proofs/PROOFS.md` |
| `ARCHITECTURE_OVERVIEW.md` | High-level Harness AI architecture for external reviewers | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `BUILD_WITH_INTERGRAX.md` | Builder route selection and deeper planning | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `USE_CASES.md` | Public use-case fit and applicability | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `PARTNERS.md` | Partner fit and pilot workflow | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `COLLABORATION.md` | Collaboration and permission-request router | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `FAQ.md` | General external-reader FAQ | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `ROADMAP.md` | Public outcome-gated product-validation roadmap | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `LKW_PRODUCT_TOUR.md` | Non-executable, product-first walkthrough of the supported LKW experience and its boundaries | **PX-5 — ACCEPTED / CLOSED** |
| `BUILDER_QUICKSTART.md` | First bounded builder onboarding route: choose a workflow, identify the ownership boundary, use an existing setup or verification path and continue through progressive disclosure | **PX-6 — ACCEPTED / CLOSED** |

Each document owns one primary responsibility. Do not duplicate ownership across reader-intent documents.

### Layer 3 — Proofs and capability spotlights

| Classification | Role |
|----------------|------|
| **LKW Platform Proof** | Primary product proof |
| **Token Optimization Engine** | Featured platform-capability proof |
| **Case studies** | Secondary bounded evidence |

Product proof and platform-capability proof are complementary but not interchangeable. LKW demonstrates a real application workflow; Token Optimization demonstrates a reusable platform mechanism. Neither replaces the other.

### Layer 4 — Technical due diligence

**Owner:**

```text
docs/project/technical/DOCUMENTATION_MAP.md
docs/project/architecture/intergrax_runtime_architecture.md
docs/project/architecture/
docs/project/maintainers/plans/
docs/project/capabilities/
applications/<pkg>/docs/
agents/<agent>/docs/
```

This layer serves developers, architects, reviewers, and implementation agents.
Application-owned technical canon (architecture, plan, build/deploy, ADRs, evidence)
lives under `applications/<pkg>/docs/`. Code-local README files and workflow-adjacent
artifacts may remain at the application or agent root when required by tooling; these
are narrow exceptions, not competing canonical documentation roots.

### Layer 5 — Maintainer controls

**Owner:**

```text
docs/project/maintainers/public-adoption/
```

Includes:

- positioning contract;
- public documentation architecture (this document);
- claim guardrails;
- external reader validation protocol;
- launch checklist;
- outreach kit;
- triage playbooks;
- issue automation.

Frozen responsibilities:

```text
Public product experience transformation
Audience and first-contact success contract
PX-0 through PX-15 phase status
→ PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md

External reader validation methodology
→ EXTERNAL_READER_VALIDATION_PROTOCOL.md

Pre-session and pre-outreach readiness
→ PUBLIC_LAUNCH_CHECKLIST.md

Participant recruitment and session-request templates
→ OUTREACH_KIT.md
```

`PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` is a **Layer 5 maintainer control**. It is indexed from `docs/project/maintainers/public-adoption/README.md`. It is **not** a normal public-reader route and must **not** be added to `docs/project/community/PUBLIC_DOCUMENTATION_MAP.md`. Previous planned external-reader execution steps 9B and 9C are replaced by PX-13 and PX-14 after the product-experience redesign. `EXTERNAL_READER_VALIDATION_PROTOCOL.md` continues to own session methodology, not roadmap phase status.

These controls govern public communication but are **not** the default first-contact path for normal readers.

---

## 3. Reader-intent routing contract

Freeze one primary route for each intent. Secondary links are allowed; every intent has exactly one primary destination.

| Reader intent | Primary destination |
|-------------|---------------------|
| Understand the problem and value | `WHY_INTERGRAX.md` |
| Compare Intergrax with common solution approaches | `WHY_INTERGRAX.md#where-intergrax-fits` |
| Review as an architect or platform engineer | `ARCHITECTURE_OVERVIEW.md` |
| Assess fit as a CTO, product lead or technical buyer | `USE_CASES.md` |
| Explore a partner, integrator or design-partner path | `PARTNERS.md` |
| Understand product-validation direction | `ROADMAP.md` |
| See the first-contact LKW product workflow | `README.md#local-knowledge-workspace-lkw` |
| Understand LKW without running it | `LKW_PRODUCT_TOUR.md` |
| Try LKW (supported product quickstart) | `applications/local_workspace_application/docs/product/QUICKSTART.md` |
| Run or inspect the bounded LKW technical proof | `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` |
| Explore Token Optimization | Token Optimization main guide — `docs/project/capabilities/token_optimization/README.md` |
| Start building with Intergrax | `BUILDER_QUICKSTART.md` |
| Plan a deeper application build | `BUILD_WITH_INTERGRAX.md` |
| Run a bounded evaluation (detailed companion) | `EVALUATION_GUIDE.md` |
| Check current proof status | `docs/project/proofs/PROOFS.md` — canonical proof dashboard |
| Contribute or provide technical feedback | `COLLABORATION.md` |
| Understand practical permission boundaries | `COLLABORATION.md` |
| Read legally authoritative rights and restrictions | `LICENSE` |
| Read general first-contact questions | `FAQ.md` |
| Perform deep technical review | `docs/project/technical/DOCUMENTATION_MAP.md` |

## Primary CTA and reader-progression contract

Freeze the primary reader actions:

```text
README
→ Try LKW

LKW Product Tour
→ LKW Quick Start

LKW Quick Start
→ LKW Platform Proof

Builder Quick Start
→ BUILD_WITH_INTERGRAX

Architecture Overview
→ PROOFS

Use Cases
→ PROOFS after apparent fit

Why Intergrax
→ Use Cases

FAQ
→ LKW Product Tour

Partners
→ prepare pilot brief
```

Every major Layer 1–3 reader document identifies one primary next action.
Secondary links remain available but are visually grouped.
Reference tables are not automatically CTA blocks.
Repeated links must not create a second primary action.
Product trial and platform proof remain separate.
Token Optimization remains a secondary capability route.
Proof and architecture links must not compete with `Try LKW` in the README first-screen CTA row.
A router may expose several intents, but each intent has one primary destination.
No fake urgency, countdowns, unsupported promises or dark-pattern wording are allowed.
Contact is not requested before partner fit and pilot scope are prepared.

## Conversion-path ownership

Freeze these progressions:

```text
Product:
README
→ Product Tour or Quick Start
→ Platform Proof when deeper evidence is needed

Builder:
Builder Quick Start
→ BUILD_WITH_INTERGRAX
→ route-specific evaluation or technical documentation

Architect:
Architecture Overview
→ PROOFS
→ Evaluation Guide or Technical Documentation Map

Buyer:
Use Cases
→ PROOFS
→ Evaluation Guide, Partners, defer or stop

Partner:
Partners
→ pilot brief
→ permission-aware discussion
```

No route is required to continue when the fit is negative.

## CTA visual ownership

```text
docs/project/community/PUBLIC_DOCUMENTATION_MAP.md#start-by-what-you-want-to-do
```

Purpose:

```text
visual reader-intent and next-action routing
```

This is a navigation diagram, not a product architecture claim.
It uses no custom Mermaid styling.
The table remains the accessible textual route.
The diagram must remain consistent with the routing contract.
LKW remains visually and semantically the primary product route.

## Existing compliant documents

PX-11 reviewed but did not need to modify:

```text
ARCHITECTURE_OVERVIEW.md
BUILD_WITH_INTERGRAX.md
EVALUATION_GUIDE.md
PARTNERS.md
PROOFS.md
LKW_PLATFORM_PROOF.md
Token Optimization guide
```

### Audience ownership

```text
ARCHITECTURE_OVERVIEW.md:
high-level responsibility, governance-placement and proof-boundary route for architects and platform engineers

USE_CASES.md:
fit, maturity, risk and evaluation-decision route for CTOs, product leads and technical buyers

PARTNERS.md:
partner qualification, pilot preparation and permission-aware next-step route for partners, integrators and design partners
```

Explicit exclusions:

```text
ARCHITECTURE_OVERVIEW.md does not own deep implementation navigation or proof matrices.

USE_CASES.md does not own technical proof evidence, legal terms or pilot execution.

PARTNERS.md does not grant production, commercial, hosting or redistribution permission.
```

Deep evidence remains owned by `PROOFS.md`.

Legal terms remain owned by `LICENSE`.

Practical permission routing remains owned by `COLLABORATION.md`.

### Category-positioning ownership

```text
WHY_INTERGRAX.md:
category-level responsibility comparison

USE_CASES.md:
concrete workflow fit and buyer decision

ARCHITECTURE_OVERVIEW.md:
technical responsibility boundaries

FAQ.md:
concise route to the owning comparison
```

The category comparison describes primary responsibility, not verified feature parity.

Generic categories may be described. Specific vendors are not compared in PX-8.

No score, ranking, winner, market-leadership or universal-superiority claim is allowed.

The categories may overlap and may be combined.

Reader-intent ownership is frozen as follows:

```text
BUILDER_QUICKSTART.md:
first builder checkpoint

BUILD_WITH_INTERGRAX.md:
builder route selection and deeper planning

EVALUATION_GUIDE.md:
bounded evaluation execution

docs/project/technical/DOCUMENTATION_MAP.md:
deep technical navigation
```

The Builder Quick Start does not own product trial instructions. The LKW Quick Start does not own generic builder onboarding. The Evaluation Guide does not replace the Builder Quick Start. The Technical Documentation Map is not the first builder step.

---

## 4. LKW placement contract

- LKW is the primary product-development and product-proof path.
- README must show LKW through a user workflow before infrastructure details.
- LKW remains Backend Product Alpha / MVP.
- LKW does not claim completed commercial or real-user validation.
- LKW proof details remain owned by LKW documentation and the later proof-and-claims document.

Primary public entry:

```text
README
→ Product Tour
→ Quick Start or Platform Proof

README:
first-contact explanation and primary CTA

LKW_PRODUCT_TOUR.md:
non-executable product walkthrough

QUICKSTART.md:
supported executable product evaluation

LKW_PLATFORM_PROOF.md:
bounded technical reviewer evidence

```

The Product Tour does not own execution instructions. The Quick Start does not own platform certification. The Platform Proof does not replace the Product Tour.

**Frozen CTA placement (PX-1):**

- LKW is the primary public product CTA (**Try LKW**).
- Token Optimization is the secondary capability CTA.
- Product quickstart and platform proof are separate routes (`product quickstart ≠ platform proof`).
- Root README adoption is implemented in PX-2 (ACCEPTED / CLOSED).
- Supported Try LKW quickstart is implemented in PX-3 (ACCEPTED / CLOSED).

---

## 4a. LKW visual evidence ownership

PX-4 owns the neutral documentation visual for the supported indexed LKW Quick Start. It presents a concrete indexed result: the approved sample source, managed intake and indexing, the user question, the grounded answer, the source reference, and persisted Ask-run verification.

The canonical assets are:

```text
applications/local_workspace_application/docs/assets/lkw-grounded-result-light.svg
applications/local_workspace_application/docs/assets/lkw-grounded-result-dark.svg
```

The root `README.md` is the primary placement and `applications/local_workspace_application/docs/product/QUICKSTART.md` is the secondary placement. The visual is not a UI screenshot and makes no Hybrid, live-provider, or production claim.

The canonical alt-text meaning identifies `lkw_product_quickstart.txt`, the question “What is the project codename?”, the grounded answer “AURORA-17”, its source reference, and persisted Ask-run verification.

---

## 5. Token Optimization placement contract

**Canonical main guide:**

```text
docs/project/capabilities/token_optimization/README.md
```

**Classification:**

```text
Featured platform-capability proof
```

Token Optimization is relevant to readers interested in:

- context and prompt cost control;
- deterministic optimization;
- cache-stable prompt assembly;
- cache-aware execution;
- protected-region safety;
- receipts and attribution;
- bounded provider proof paths.

**Safe short description (frozen):**

```text
Intergrax includes a deterministic, policy-governed Token Optimization
Engine with protected-region validation, receipts, cache-stable prompt
assembly, cache-aware execution, and bounded proof paths.
```

Do not include numeric savings. Do not claim:

- universal token reduction;
- production-proven savings;
- automatic cost reduction for every model;
- universal provider cache behavior;
- complete live-provider, provider-wide, rollback, production-rollout or generally available durable compaction behavior;
- universal or production-proven savings.

The first public link must point to `docs/project/capabilities/token_optimization/README.md`.

Secondary technical links may point from that guide to:

```text
docs/project/capabilities/architecture/TOKEN_OPTIMIZATION.md
docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md
docs/project/capabilities/token_optimization/proofs/
docs/project/capabilities/TOKEN_OPTIMIZATION_CLAIMS.md
```

Do not route first-time readers directly to architecture, plan, or audit documents.

---

## 6. Root README placement

Reserve a future root README slot for Token Optimization under:

```text
Featured platform capability
```

or an equivalent neutral heading.

The final wording and any proof links in root README remain governed by `PUBLIC_PROOF_AND_CLAIMS_MODEL.md` § README discovery versus performance promotion and `TOKEN_OPTIMIZATION_CLAIMS.md` § README discovery and promotion boundary.

**Resolved rule (frozen):**

- **Neutral discovery** (capability name, qualified status, main-guide link) is allowed before TOKEN-10H when no numeric savings or universal performance claims are used.
- **Performance promotion** (badges, percentages, universal savings, production-proven claims) remains gated by TOKEN-10G hard gates and TOKEN-10H completion.

The current root README is aligned with the frozen rule: it describes the
bounded durable-compaction mechanism with explicit limitations. Do not weaken
the performance gate.

---

## 7. Current-to-target migration map

| Current document | Current role | Target role | Future owner | Migration task |
|------------------|--------------|-------------|--------------|------------------|
| `README.md` | Layer 1 product-first landing | Layer 1 product-first landing | `README.md` | **implemented** — PX-2 ACCEPTED / CLOSED |
| `LKW_PRODUCT_TOUR.md` | New product-first reader route | Layer 2 reader-intent document | `LKW_PRODUCT_TOUR.md` | **PX-5 — ACCEPTED / CLOSED** |
| `PROOFS.md` | — | Layer 2 proof dashboard | `docs/project/proofs/PROOFS.md` | **implemented** (PUBLIC-DOCS-COMMERCIALIZATION-4) |
| `WHY_INTERGRAX.md` | — | Layer 2 value and fit guide | `WHY_INTERGRAX.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `ARCHITECTURE_OVERVIEW.md` | — | Layer 2 public architecture overview | `ARCHITECTURE_OVERVIEW.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `BUILD_WITH_INTERGRAX.md` | — | Layer 2 evaluation and building router | `BUILD_WITH_INTERGRAX.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `EVALUATION_GUIDE.md` | Bounded evaluation paths | Detailed bounded execution companion | `EVALUATION_GUIDE.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-9A |
| `USE_CASES.md` | Public use-case map | Layer 2 public use-case fit map | `USE_CASES.md` | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `PARTNERS.md` | Partner brief | Layer 2 partner-fit and pilot-workflow guide | `PARTNERS.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `FAQ.md` | Mixed general, architecture and legal FAQ | Layer 2 concise first-contact FAQ | `FAQ.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `ROADMAP.md` | Public product-validation roadmap | Layer 2 outcome-gated public roadmap | `ROADMAP.md` | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `COLLABORATION.md` | Mixed collaboration and maintainer-control document | Layer 2 practical collaboration and permission-request router | `COLLABORATION.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md` | LKW guided proof | Primary product proof | LKW proof docs | stable |
| `docs/project/capabilities/token_optimization/README.md` | Token Optimization guide | Featured platform-capability proof entry | Token Optimization docs | stable (this task) |
| `docs/project/technical/DOCUMENTATION_MAP.md` | Technical/developer navigation | Layer 4 technical map | `docs/project/technical/DOCUMENTATION_MAP.md` | stable |
| `docs/project/maintainers/public-adoption/README.md` | Maintainer adoption controls | Layer 5 maintainer index | `docs/project/maintainers/public-adoption/` | stable (this task) |

`PROOFS.md` is **implemented** at `docs/project/proofs/PROOFS.md` (canonical proof dashboard). `WHY_INTERGRAX.md`, `ARCHITECTURE_OVERVIEW.md`, and `BUILD_WITH_INTERGRAX.md` are **implemented** (PUBLIC-DOCS-COMMERCIALIZATION-6). `EVALUATION_GUIDE.md` is **refreshed** (PUBLIC-DOCS-COMMERCIALIZATION-9A) as the detailed bounded execution companion; `BUILD_WITH_INTERGRAX.md` owns public route selection. `PARTNERS.md`, `COLLABORATION.md`, and `FAQ.md` are **refreshed** (PUBLIC-DOCS-COMMERCIALIZATION-8).

---

## External-validation boundary

A protocol, checklist, automated test, internal audit or maintainer review does not constitute external validation.

External validation requires real independent sessions recorded against a pinned repository revision.

Documentation comprehension validation is not product validation, real-user validation, commercial validation, security review, legal review or production-readiness proof.

The previous planned 9B and 9C execution steps are paused and superseded by PX-13 and PX-14 of `PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md`.

PX-13 owns real external comprehension and trial sessions.

PX-14 owns findings, corrections and required reruns.

The historical names 9B and 9C do not define an additional active execution path.

---

## Public language boundary

Freeze the following language ownership:

```text
Layer 1–3 public-reader documents:
lead with user outcome, responsibility and next action

Layer 4 deep technical documents:
may use exact architecture terms, code symbols and provider details

Layer 5 maintainer controls:
may retain PX, TOKEN, CTX and other task identifiers
```

Product and capability proper nouns are allowed when they identify real public concepts:
LKW, Token Optimization, Hybrid Ask, Slack, Google Workspace, Ollama and vLLM.

Internal task identifiers, implementation-stage labels, review states and maintainer workflow vocabulary are not normal public-reader language.

Freeze these normal-reader replacements:

```text
ProofReceipt
→ persisted execution evidence
  or execution receipt when technically necessary

Tier-3
→ end-to-end application and platform workflow

Harness AI
→ reusable governed application foundation
  unless the reader intentionally entered deep technical material

HITL
→ human approval or human review

RAG
→ retrieval and grounding
  unless defined in a deep technical context
```

Filenames and deep destination paths do not need to be renamed merely because they contain technical vocabulary. Visible link labels should remain reader-friendly.

---

## 8. Link and duplication rules

1. One primary route per reader intent.
2. README links to overview documents, not deep implementation plans.
3. Public maps may link to proof entrypoints.
4. Proof entrypoints own their detailed proof navigation.
5. Planned files are not clickable until created.
6. Status details remain in owning plans and later proof-and-claims contract.
7. Maintainer controls do not become the default public journey.
8. Archived documents are never normal navigation targets.
9. Public documents must not duplicate the complete technical documentation map.
10. Every new reader-facing Layer 1–3 document must be added to both this architecture contract and `docs/project/community/PUBLIC_DOCUMENTATION_MAP.md`. A new Layer 5 maintainer control must be added to this architecture contract and `docs/project/maintainers/public-adoption/README.md`, but must not become a normal public-reader route unless its role is intentionally changed.

`docs/project/community/PUBLIC_DOCUMENTATION_MAP.md` must not list Layer 5 positioning contracts, claim guardrails or maintainer controls as normal public documents.

Those remain owned by `docs/project/maintainers/public-adoption/README.md`.

---

## Visual presentation standard

Public documentation is a product surface. Correct information presented as an unreadable wall of text is not sufficient.

### First-screen rule

Every major reader-facing document should show within its first visible section:

- what the document is for;
- current scope or maturity;
- one primary next action;
- a visual summary when the concept benefits from one.

### Visual hierarchy

Use:

- one clear H1;
- a short introductory sentence;
- an at-a-glance block;
- sections arranged by reader questions;
- short paragraphs;
- tables only when comparison is useful;
- callouts for limitations and warnings.

### Diagrams

GitHub-native Mermaid is preferred for text-native architecture, responsibility, state and workflow diagrams.

Use GitHub-native Mermaid for:

- architecture;
- workflows;
- evidence flows;
- responsibility boundaries;
- state transitions.

Primary diagrams must:

- use short labels;
- avoid excessive nodes;
- remain legible in GitHub light and dark mode;
- avoid custom color styling that breaks dark mode;
- be explained in text immediately after the diagram.

ASCII diagrams may remain in deep technical documents, but new primary public diagrams should use Mermaid.

### Images and screenshots

Future root README and major product documents should use real visual assets when available.

Rules:

- use real product screenshots or intentionally designed neutral illustrations;
- never fabricate product UI;
- provide descriptive alt text;
- store future public assets under `docs/project/assets/public/`;
- use PNG or WebP for screenshots;
- compress assets reasonably;
- avoid screenshots containing credentials, private data, or local absolute paths.

Reviewed SVG is appropriate for intentionally designed neutral product-result visuals when:

- the asset has explicit ownership;
- light and dark variants remain semantically identical;
- descriptive alt text exists;
- surrounding text preserves the essential meaning;
- the asset is not presented as a screenshot;
- there are no external dependencies.

PX-8 category visual ownership:

```text
docs/project/assets/public/intergrax-category-map-light.svg
docs/project/assets/public/intergrax-category-map-dark.svg

Owner:
WHY_INTERGRAX.md#where-intergrax-fits

Purpose:
neutral category-responsibility map

Status:
PX-8 — ACCEPTED / CLOSED

Acceptance evidence:
0580270902ec265b6d7523e9b00d50acb074d815
```

Both variants must remain semantically identical. Surrounding text must preserve essential meaning. These assets are conceptual documentation graphics, not screenshots. Vendor logos and unsupported claims are prohibited.

No binary image asset is required until a reviewed source screenshot is supplied.

### PX-9 visual preservation

Language cleanup must not flatten the public visual experience. Existing reviewed pictures, SVG families, tables and Mermaid diagrams remain unless factually incorrect or owned by a later redesign task. Diagram labels should be simplified before a diagram is removed.

PX-9 updates the existing `BUILD_WITH_INTERGRAX.md` route diagram without adding a new diagram or visual asset.

### Minimum future root README visual set

Freeze the future requirement:

1. one hero product visual;
2. one LKW user-workflow diagram;
3. one simplified Intergrax architecture diagram;
4. one compact proof/status presentation;
5. one Token Optimization capability spotlight.

The README must not become a gallery. Every visual must explain something.

**Delivered in the root README redesign:**

- light/dark neutral hero illustration;
- light/dark neutral LKW grounded-result SVG visual;
- simplified Intergrax architecture Mermaid diagram;
- compact proof/status section;
- Token Optimization capability spotlight and Mermaid diagram.

Real product screenshots remain deferred until reviewed source images exist.

---

## 9. Source-of-truth boundaries

| Topic | Owner |
|-------|-------|
| Public positioning | `INTERGRAX_PUBLIC_POSITIONING.md` — exact first-contact message, product hierarchy, audience value, CTA language, Harness AI placement |
| Public use-case fit | `USE_CASES.md` |
| Public product-validation direction | `ROADMAP.md` |
| Public value and fit | `WHY_INTERGRAX.md` |
| Public architecture overview | `ARCHITECTURE_OVERVIEW.md` |
| Evaluation and building route selection | `BUILD_WITH_INTERGRAX.md` |
| Public documentation architecture | this document |
| Public proof and claims model | `PUBLIC_PROOF_AND_CLAIMS_MODEL.md` |
| Public proof dashboard | `docs/project/proofs/PROOFS.md` |
| Public reader navigation | `docs/project/community/PUBLIC_DOCUMENTATION_MAP.md` |
| Technical/developer navigation | `docs/project/technical/DOCUMENTATION_MAP.md` |
| Token Optimization technical guide | `docs/project/capabilities/token_optimization/README.md` |
| Token Optimization claim boundaries | `TOKEN_OPTIMIZATION_CLAIMS.md` |
| Detailed proof status | `PROOFS.md` + `PUBLIC_PROOF_AND_CLAIMS_MODEL.md` |
| Product implementation status | owning implementation plans |
| Partner fit and pilot workflow | `PARTNERS.md` |
| Collaboration and contribution routes | `COLLABORATION.md` |
| Practical permission-request route | `COLLABORATION.md` |
| Legally authoritative rights and restrictions | `LICENSE` |
| General first-contact questions | `FAQ.md` |
| Public product experience transformation | `PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` |
| Audience and first-contact success contract | `PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` |
| PX-0 through PX-15 phase status | `PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` |
| External reader validation methodology | `EXTERNAL_READER_VALIDATION_PROTOCOL.md` |
| Reader-facing evaluation paths | `EVALUATION_GUIDE.md` |
| Validation readiness | `PUBLIC_LAUNCH_CHECKLIST.md` |
| Validation recruitment templates | `OUTREACH_KIT.md` |
