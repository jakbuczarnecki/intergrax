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

- `INTERGRAX_PUBLIC_POSITIONING.md`;
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
PARTNERS_AND_PILOTS.md
FAQ.md
LICENSE_FAQ.md
ROADMAP.md
```

| Document | Responsibility | Status |
|----------|----------------|--------|
| `WHY_INTERGRAX.md` | Problem, value, and audience without tier jargon | planned |
| `PROOFS.md` | Consolidated public proof status and claim boundaries | planned |
| `ARCHITECTURE_OVERVIEW.md` | High-level Harness AI architecture for external reviewers | planned |
| `BUILD_WITH_INTERGRAX.md` | Bounded evaluation and builder onboarding path | planned |
| `PARTNERS_AND_PILOTS.md` | Design-partner and pilot workflow (successor to split partner docs) | planned |
| `FAQ.md` | Common external questions | exists — root `FAQ.md` |
| `LICENSE_FAQ.md` | License and permission FAQ without full legal text | planned |
| `ROADMAP.md` | Public product-validation program and adoption priorities | exists — root `ROADMAP.md` |

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
- launch checklist;
- outreach kit;
- triage playbooks;
- issue automation.

These controls govern public communication but are **not** the default first-contact path for normal readers.

---

## 3. Reader-intent routing contract

Freeze one primary route for each intent. Secondary links are allowed; every intent has exactly one primary destination.

| Reader intent | Primary destination |
|-------------|---------------------|
| Understand the problem and value | root README now; future `WHY_INTERGRAX.md` |
| See a real product workflow | LKW Platform Proof — `docs/public-adoption/LKW_PLATFORM_PROOF.md` |
| Explore Token Optimization | Token Optimization main guide — `docs/features/token_optimization/README.md` |
| Review high-level architecture | current Harness Narrative in root README; future `ARCHITECTURE_OVERVIEW.md` |
| Run a bounded evaluation | `EVALUATION_GUIDE.md` |
| Check current proof status | future `PROOFS.md` |
| Discuss a pilot or design-partner workflow | `PARTNERS.md`, later `PARTNERS_AND_PILOTS.md` |
| Check permission boundaries | `COLLABORATION.md` and `LICENSE` |
| Perform deep technical review | `docs/DOCUMENTATION_MAP.md` |

---

## 4. LKW placement contract

- LKW is the primary product-development and product-proof path.
- README must show LKW through a user workflow before infrastructure details.
- LKW remains Backend Product Alpha / MVP.
- LKW does not claim completed commercial or real-user validation.
- LKW proof details remain owned by LKW documentation and the later proof-and-claims document.

Primary public entry: `docs/public-adoption/LKW_PLATFORM_PROOF.md`

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

The final wording and any proof links in root README remain gated by the next task:

```text
PUBLIC-DOCS-COMMERCIALIZATION-4
PROOF-AND-CLAIMS-MODEL
```

**Recorded conflict:**

- current root README already links and promotes Token Optimization (Quick start path, evaluation table, badges);
- current `TOKEN_OPTIMIZATION_CLAIMS.md` contains a TOKEN-10G / TOKEN-10H promotion gate: main README Token Optimization promotion is allowed only after TOKEN-10H when a checked-in proof has passed TOKEN-10G hard gates and been independently audited.

Point 4 must reconcile this before the new README is implemented. Do not resolve the gate by silently weakening it in earlier tasks.

---

## 7. Current-to-target migration map

| Current document | Current role | Target role | Future owner | Migration task |
|------------------|--------------|-------------|--------------|------------------|
| `README.md` | Landing, mixed technical index | Layer 1 landing only | root README rewrite | PUBLIC-DOCS-COMMERCIALIZATION-4 + README task |
| `EVALUATION_GUIDE.md` | Bounded evaluation paths | Reader-intent evaluation | `BUILD_WITH_INTERGRAX.md` (partial merge) | future commercialization |
| `USE_CASES.md` | Use-case map | Reader-intent fit check | `WHY_INTERGRAX.md` / use-case section | future commercialization |
| `PARTNERS.md` | Partner brief | Reader-intent partnership | `PARTNERS_AND_PILOTS.md` | future commercialization |
| `FAQ.md` | External FAQ | Reader-intent FAQ | `FAQ.md` (retain) | stable |
| `ROADMAP.md` | Public program roadmap | Reader-intent roadmap | `ROADMAP.md` (retain) | stable |
| `COLLABORATION.md` | Collaboration permissions | Permission boundary | `COLLABORATION.md` + `LICENSE_FAQ.md` | future commercialization |
| `docs/public-adoption/LKW_PLATFORM_PROOF.md` | LKW guided proof | Primary product proof | LKW proof docs | stable |
| `docs/features/token_optimization/README.md` | Token Optimization guide | Featured platform-capability proof entry | Token Optimization docs | stable (this task) |
| `docs/DOCUMENTATION_MAP.md` | Technical/developer navigation | Layer 4 technical map | `docs/DOCUMENTATION_MAP.md` | stable |
| `docs/public-adoption/README.md` | Maintainer adoption controls | Layer 5 maintainer index | `docs/public-adoption/` | stable (this task) |

Planned documents (`WHY_INTERGRAX.md`, `PROOFS.md`, `ARCHITECTURE_OVERVIEW.md`, `BUILD_WITH_INTERGRAX.md`, `PARTNERS_AND_PILOTS.md`, `LICENSE_FAQ.md`) are **not** marked as implemented.

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
10. Every new public document must be added to both this architecture contract and `docs/PUBLIC_DOCUMENTATION_MAP.md`.

---

## 9. Source-of-truth boundaries

| Topic | Owner |
|-------|-------|
| Public positioning | `INTERGRAX_PUBLIC_POSITIONING.md` |
| Public documentation architecture | this document |
| Public reader navigation | `docs/PUBLIC_DOCUMENTATION_MAP.md` |
| Technical/developer navigation | `docs/DOCUMENTATION_MAP.md` |
| Token Optimization technical guide | `docs/features/token_optimization/README.md` |
| Token Optimization claim boundaries | `TOKEN_OPTIMIZATION_CLAIMS.md` |
| Detailed proof status | later proof-and-claims document |
| Product implementation status | owning implementation plans |
| License rights | `LICENSE` |
| Collaboration permissions | `COLLABORATION.md` |
