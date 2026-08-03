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

## 2. Public documentation layers

Freeze five layers.

### Layer 1 — Landing

**Owner:**

```text
README.md
```

**Role:**

- first 30 seconds;
- problem and value;
- LKW product proof;
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
```

| Document | Responsibility | Status |
|----------|----------------|--------|
| `WHY_INTERGRAX.md` | Problem, value, and audience without tier jargon | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `PROOFS.md` | Consolidated public proof status and claim boundaries | exists — root `PROOFS.md` |
| `ARCHITECTURE_OVERVIEW.md` | High-level Harness AI architecture for external reviewers | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `BUILD_WITH_INTERGRAX.md` | Bounded evaluation and builder onboarding path | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `USE_CASES.md` | Public use-case fit and applicability | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `PARTNERS.md` | Partner fit and pilot workflow | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `COLLABORATION.md` | Collaboration and permission-request router | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `FAQ.md` | General external-reader FAQ | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `ROADMAP.md` | Public outcome-gated product-validation roadmap | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |

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
docs/DOCUMENTATION_MAP.md
docs/intergrax_runtime_architecture.md
docs/architecture/
docs/plan/
docs/features/
applications/*/docs/
```

This layer serves developers, architects, reviewers, and implementation agents.

### Layer 5 — Maintainer controls

**Owner:**

```text
docs/public-adoption/
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

`PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md` is a **Layer 5 maintainer control**. It is indexed from `docs/public-adoption/README.md`. It is **not** a normal public-reader route and must **not** be added to `docs/PUBLIC_DOCUMENTATION_MAP.md`. Previous planned external-reader execution steps 9B and 9C are replaced by PX-13 and PX-14 after the product-experience redesign. `EXTERNAL_READER_VALIDATION_PROTOCOL.md` continues to own session methodology, not roadmap phase status.

These controls govern public communication but are **not** the default first-contact path for normal readers.

---

## 3. Reader-intent routing contract

Freeze one primary route for each intent. Secondary links are allowed; every intent has exactly one primary destination.

| Reader intent | Primary destination |
|-------------|---------------------|
| Understand the problem and value | `WHY_INTERGRAX.md` |
| Check use-case fit | `USE_CASES.md` |
| Understand product-validation direction | `ROADMAP.md` |
| See a real product workflow | LKW Platform Proof — `docs/public-adoption/LKW_PLATFORM_PROOF.md` |
| Explore Token Optimization | Token Optimization main guide — `docs/features/token_optimization/README.md` |
| Review high-level architecture | `ARCHITECTURE_OVERVIEW.md` |
| Build or evaluate with Intergrax | `BUILD_WITH_INTERGRAX.md` |
| Run a bounded evaluation (detailed companion) | `EVALUATION_GUIDE.md` |
| Check current proof status | `PROOFS.md` — root proof dashboard |
| Discuss a pilot or design-partner workflow | `PARTNERS.md` |
| Contribute or provide technical feedback | `COLLABORATION.md` |
| Understand practical permission boundaries | `COLLABORATION.md` |
| Read legally authoritative rights and restrictions | `LICENSE` |
| Read general first-contact questions | `FAQ.md` |
| Perform deep technical review | `docs/DOCUMENTATION_MAP.md` |

---

## 4. LKW placement contract

- LKW is the primary product-development and product-proof path.
- README must show LKW through a user workflow before infrastructure details.
- LKW remains Backend Product Alpha / MVP.
- LKW does not claim completed commercial or real-user validation.
- LKW proof details remain owned by LKW documentation and the later proof-and-claims document.

Primary public entry: `docs/public-adoption/LKW_PLATFORM_PROOF.md`

**Frozen CTA placement (PX-1):**

- LKW is the primary public product CTA.
- Token Optimization is the secondary capability CTA.
- Product trial and platform evaluation are separate routes.
- The current primary CTA is **See the LKW workflow**.
- **Try LKW** is gated by PX-3.
- Root README adoption occurs in PX-2.

---

## 5. Token Optimization placement contract

**Canonical main guide:**

```text
docs/features/token_optimization/README.md
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
- completed in-cache compaction;
- completed TOKEN-10G or TOKEN-10H unless later verified.

The first public link must point to `docs/features/token_optimization/README.md`.

Secondary technical links may point from that guide to:

```text
docs/features/architecture/TOKEN_OPTIMIZATION.md
docs/features/plan/TOKEN_OPTIMIZATION.md
docs/features/token_optimization/proofs/
docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md
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

Current root README may still contain pre-reconciliation promotion language; the future README rewrite must align with the frozen rule. Do not weaken the performance gate.

---

## 7. Current-to-target migration map

| Current document | Current role | Target role | Future owner | Migration task |
|------------------|--------------|-------------|--------------|------------------|
| `README.md` | Layer 1 product-first landing | Layer 1 product-first landing | `README.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-5 |
| `PROOFS.md` | — | Layer 2 proof dashboard | root `PROOFS.md` | **implemented** (PUBLIC-DOCS-COMMERCIALIZATION-4) |
| `WHY_INTERGRAX.md` | — | Layer 2 value and fit guide | `WHY_INTERGRAX.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `ARCHITECTURE_OVERVIEW.md` | — | Layer 2 public architecture overview | `ARCHITECTURE_OVERVIEW.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `BUILD_WITH_INTERGRAX.md` | — | Layer 2 evaluation and building router | `BUILD_WITH_INTERGRAX.md` | **implemented** — PUBLIC-DOCS-COMMERCIALIZATION-6 |
| `EVALUATION_GUIDE.md` | Bounded evaluation paths | Detailed bounded execution companion | `EVALUATION_GUIDE.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-9A |
| `USE_CASES.md` | Public use-case map | Layer 2 public use-case fit map | `USE_CASES.md` | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `PARTNERS.md` | Partner brief | Layer 2 partner-fit and pilot-workflow guide | `PARTNERS.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `FAQ.md` | Mixed general, architecture and legal FAQ | Layer 2 concise first-contact FAQ | `FAQ.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `ROADMAP.md` | Public product-validation roadmap | Layer 2 outcome-gated public roadmap | `ROADMAP.md` | **implemented / refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-7 |
| `COLLABORATION.md` | Mixed collaboration and maintainer-control document | Layer 2 practical collaboration and permission-request router | `COLLABORATION.md` | **refreshed** — PUBLIC-DOCS-COMMERCIALIZATION-8 |
| `docs/public-adoption/LKW_PLATFORM_PROOF.md` | LKW guided proof | Primary product proof | LKW proof docs | stable |
| `docs/features/token_optimization/README.md` | Token Optimization guide | Featured platform-capability proof entry | Token Optimization docs | stable (this task) |
| `docs/DOCUMENTATION_MAP.md` | Technical/developer navigation | Layer 4 technical map | `docs/DOCUMENTATION_MAP.md` | stable |
| `docs/public-adoption/README.md` | Maintainer adoption controls | Layer 5 maintainer index | `docs/public-adoption/` | stable (this task) |

`PROOFS.md` is **implemented** (root proof dashboard). `WHY_INTERGRAX.md`, `ARCHITECTURE_OVERVIEW.md`, and `BUILD_WITH_INTERGRAX.md` are **implemented** (PUBLIC-DOCS-COMMERCIALIZATION-6). `EVALUATION_GUIDE.md` is **refreshed** (PUBLIC-DOCS-COMMERCIALIZATION-9A) as the detailed bounded execution companion; `BUILD_WITH_INTERGRAX.md` owns public route selection. `PARTNERS.md`, `COLLABORATION.md`, and `FAQ.md` are **refreshed** (PUBLIC-DOCS-COMMERCIALIZATION-8).

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
10. Every new reader-facing Layer 1–3 document must be added to both this architecture contract and `docs/PUBLIC_DOCUMENTATION_MAP.md`. A new Layer 5 maintainer control must be added to this architecture contract and `docs/public-adoption/README.md`, but must not become a normal public-reader route unless its role is intentionally changed.

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
- store future public assets under `docs/assets/public/`;
- use PNG or WebP for screenshots and SVG for designed diagrams where appropriate;
- compress assets reasonably;
- avoid screenshots containing credentials, private data, or local absolute paths.

No binary image asset is required until a reviewed source screenshot is supplied.

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
- LKW user-workflow Mermaid diagram;
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
| Public proof dashboard | `PROOFS.md` |
| Public reader navigation | `docs/PUBLIC_DOCUMENTATION_MAP.md` |
| Technical/developer navigation | `docs/DOCUMENTATION_MAP.md` |
| Token Optimization technical guide | `docs/features/token_optimization/README.md` |
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
