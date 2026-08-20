# Multi-Product Application Bootstrap Rules

**Document type:** Normative Product Session bootstrap contract  
**Owner:** Portfolio Control  
**Last defined:** 2026-08-20 (MP-14)

This document answers: **What must be true before a newly admitted Intergrax product begins implementation?**

---

## Applies to

| Role | Product |
|------|---------|
| Newly selected applications | Contract-to-Invoice Leakage / Recovery Operator |
| | Supplier Disruption Response Operator |
| | Third-Party Risk Decision Operator |
| | Deployment / Change Guardian |
| Future admissions | Any product formally admitted to the multi-product program |

**Explicit exclusion:** Local Knowledge Workspace (LKW) predates this methodology. It uses the accepted [REFERENCE BASELINE](products/LKW.md), not a retroactive bootstrap or T0.

**Canonical companions (do not duplicate here):**

| Topic | Document |
|-------|----------|
| Program gates and G4 governance | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| T0/T1 reuse methodology, M1–M6, PASS / PARTIAL / FAIL | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Product-first development and capability-audit mechanics | [PRODUCT_FIRST_MVP.md](../plans/PRODUCT_FIRST_MVP.md) |
| Selection record | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |

---

## Core invariants

```text
Same bootstrap method != same product architecture.
Product need first. Platform observation second.
```

This contract creates **methodological comparability** without forcing products into the same architecture.

Objectives:

- preserve product independence;
- create comparable evidence across products;
- prevent platform-first product distortion;
- freeze reuse expectations before implementation;
- make later T1 reuse audit meaningful;
- ensure shared-platform changes are evidence-driven.

The bootstrap must prevent:

- building products to showcase Intergrax;
- copying LKW domain semantics;
- premature scaffold/code before product definition;
- retroactive reuse scoring;
- silent shared-platform changes;
- declaring simple application startup as a meaningful product milestone.

---

## 1. Purpose

A Product Session progresses from selection to normal development only through the sequence in §3. Implementation does not start because a product was selected.

---

## 2. Non-negotiable product-first rule

```text
REAL USER / BUSINESS PROBLEM
    → PRODUCT DEFINITION
    → PRODUCT ARCHITECTURE
    → PLATFORM CAPABILITY AUDIT
    → T0 REUSE BASELINE
    → IMPLEMENTATION
```

**Never:**

```text
PLATFORM FEATURE
    → invent product use case
    → claim reuse success
```

| Rule | Meaning |
|------|---------|
| Market need is independent | Intergrax capability fit must **not** determine whether a market need is real. |
| Architecture may conflict with LKW | A product may require architecture that conflicts with assumptions learned from LKW. |
| Semantics stay product-owned | Product semantics remain product-owned unless evidence supports generalization. |
| Platform learning is an outcome | Platform learning is an outcome of product development, not the product objective. |
| Failure is allowed | Product failure is allowed. Platform generality failure is allowed. |
| Disproof is valid | A valid product must be capable of disproving an Intergrax abstraction. |

**G0 acceptance question:** *Would we still consider this product worth exploring if Intergrax did not exist?* If **NO**, G0 fails.

---

## 3. Standard bootstrap sequence

| Step | Gate / action | May not be skipped |
|------|---------------|--------------------|
| A | Admission / selection evidence | Selection record exists |
| B | **G0 — Product Baseline** | Portfolio Control accepts G0 |
| C | Product wedge freeze | Initial wedge recorded |
| D | **G1 — Product Architecture** | Portfolio Control accepts G1 |
| E | Platform Capability Audit | Audit complete against current boundaries |
| F | **G2 — T0 Reuse Baseline** | T0 accepted and frozen |
| G | Application Scaffold | Only after accepted G2 |
| H | **G3 — First Real Vertical Slice** | Meaningful end-to-end product outcome |
| I | Normal development | Asynchronous; product-owned |
| J | **G4** whenever material shared-platform pressure appears | Before implementing the change |

**Hard order:**

- scaffold may **not** precede accepted G2;
- implementation may **not** precede frozen T0;
- minor exploratory notes are allowed before G2;
- **no product implementation commit** should exist before T0.

G5–G8 remain governed by [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md). This document does not redefine them.

---

## 4. G0 — Product Baseline

G0 freezes product identity independently of Intergrax.

### Required frozen fields

| Field | Required |
|-------|----------|
| Product ID / canonical name | Yes |
| Problem statement | Yes |
| Target user | Yes |
| Economic / operational buyer | Yes |
| Real-world pain | Yes |
| Existing alternatives / incumbent behavior | Yes |
| Why current alternatives are insufficient | Yes |
| Product wedge | Yes |
| Primary workflow | Yes |
| MVP success outcome | Yes |
| Commercial hypothesis | Yes |
| Pilot hypothesis | Yes |
| Known selection caveats | Yes |
| Explicit non-goals | Yes |
| Key product risks | Yes |
| Market evidence references | Yes |
| Selection record reference | Yes — [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |

**Critical rule:** G0 must **not** contain a positive Intergrax-fit requirement.

A product may mention expected technical constraints. G0 **cannot** justify the product because:

- Intergrax already has feature X;
- the product exercises subsystem Y;
- the product improves platform proof.

If the product would not be worth exploring without Intergrax, G0 fails.

---

## 5. Product wedge freeze

After accepted G0, freeze the **initial** market/product wedge **before** architecture.

| Rule | Meaning |
|------|---------|
| Change is explicit | The wedge may later change only through an explicit portfolio/product decision. |
| History is preserved | The historical wedge remains inspectable. |
| No silent rewrite | Do not silently rewrite the wedge after implementation or competitive findings. |
| Selection is input | Initial portfolio context may be referenced; it is not re-derived here. |

**Non-normative wedge illustrations** (selection context only; not architecture):

| Product | Initial wedge sense |
|---------|---------------------|
| Contract Recovery | Detect + evidence + prepare recovery of money lost between contract and actual spend |
| Supply Disruption | Turn disruption into executable mitigation before operations fail |
| Third-Party Risk | Take a vendor request toward a defensible risk decision with evidence |
| Deployment Guardian | Independent production-readiness / authorization decision and enforcement |

Do **not** expand these into product architecture in this document.

---

## 6. G1 — Product Architecture

Product Session owns product architecture.

Architecture must first describe the product, not Intergrax.

| Required description | Before any Intergrax mapping |
|----------------------|------------------------------|
| User-visible workflow | Yes |
| Product domain model | Yes |
| Lifecycle | Yes |
| Authoritative product state | Yes |
| External systems | Yes |
| Actions / side effects | Yes |
| Evidence requirements | Yes |
| Human approval points | Yes |
| Failure / recovery behavior | Yes |
| Security boundaries | Yes |
| Data sensitivity / tenancy | Yes |
| Product API / frontend needs | Yes |
| Operational model | Yes |

Only **after** that architecture is coherent: map it to Intergrax.

```text
Design the right product architecture first.
Then determine what Intergrax can support.
```

**Do not force:**

- workspace / source / binding vocabulary from LKW;
- Hybrid Ask;
- RAG;
- chat / conversation UI;
- agent loops;
- Nexus-specific business semantics;
- renaming product concepts merely to resemble existing Intergrax abstractions.

---

## 7. Platform Capability Audit

Before shared-platform changes or product implementation assumptions, inspect **relevant current** Intergrax boundaries. Planned capability is not implemented capability.

For every material product responsibility ask:

1. Does a shared capability already exist?
2. Is it actually accepted/implemented, or only planned?
3. Can it be consumed unchanged?
4. Can it be consumed through intended configuration?
5. Is the apparent capability actually LKW-owned semantics?
6. Is the need product-owned?
7. Is there a genuine reusable platform gap?
8. Would extending the platform warp LKW or another active product?
9. Would local implementation duplicate an existing platform responsibility?
10. Would any proposed shared change require G4?

Use the LKW [REFERENCE BASELINE](products/LKW.md) as evidence of what existed **before** new-product implementation.

**LKW is evidence/reference, not a template.**

Capability-audit mechanics remain canonical in [PRODUCT_FIRST_MVP.md](../plans/PRODUCT_FIRST_MVP.md). This section defines bootstrap timing and the questions that must be answered before T0.

---

## 8. Responsibility classification during bootstrap

For T0 **expectation** purposes use the exact canonical classifications from [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md):

| Classification | Meaning at T0 |
|----------------|---------------|
| `REUSED_UNCHANGED` | Expected consumption of an existing shared mechanism with no platform modification |
| `REUSED_CONFIGURED` | Expected reuse through intended configuration / policy / adapter / DI contract |
| `EXTENDED_GENERALLY` | Expected general platform gap requiring a product-neutral extension |
| `PRODUCT_OWNED` | Expected product-domain responsibility |
| `PLATFORM_LEAK` | Defect class — not an acceptable planned outcome |

At T0 these are **expectations / hypotheses**. Implementation has not yet happened.

Do **not** present expected classification as a measured outcome.

Final classification belongs to later evidence-driven audit (T1 / G6).

---

## 9. G2 — Preregistered T0 reuse baseline

Every newly selected product **must** have accepted T0 before the first implementation commit.

LKW is excluded: it predates the methodology and has no retroactive T0.

Canonical methodology remains [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md). This section defines bootstrap obligations only.

T0 must freeze at minimum:

| Frozen field | Notes |
|--------------|-------|
| Exact Intergrax starting SHA | Repository state at T0 freeze |
| Product hypothesis | Why the product should exist |
| Target user | Concrete first user |
| Primary workflow | Observable end-to-end steps |
| Diversity from LKW | Why this is not a renamed LKW workflow |
| Accepted product architecture reference | Pointer to accepted G1 |
| Platform responsibility matrix | Every required responsibility; expected classification |
| Critical Reuse Set | See §11 |
| Expected reuse candidates | Expected `REUSED_UNCHANGED` |
| Expected configured reuse | Expected `REUSED_CONFIGURED` |
| Expected general platform gaps | Expected `EXTENDED_GENERALLY` |
| Expected `PRODUCT_OWNED` responsibilities | Product-domain scope |
| Known ambiguity / risk | Open questions frozen as unknown, not scored |
| M1–M6 measurement methodology | As defined by PRODUCT_REUSE_PROOF — do not redefine |
| PASS / PARTIAL / FAIL criteria | As defined by PRODUCT_REUSE_PROOF — do not redefine |
| Timestamp / date | T0 freeze time |
| Evidence / source links | Architecture, audit, selection, baseline |

Do **not** duplicate the full reuse methodology here.

---

## 10. T0 immutability / anti-gaming

After the first implementation commit, do **not** silently modify T0 to:

- remove a failed Critical Reuse item;
- move a responsibility to `PRODUCT_OWNED` because reuse failed;
- add an already-observed successful reuse candidate;
- weaken PASS criteria;
- hide unexpected platform expansion;
- hide private duplication;
- reframe product diversity after implementation.

**Allowed:**

- append a dated annotation;
- record an explicit architecture/product decision;
- supersede through an auditable decision;
- preserve the original T0.

Historical T0 remains inspectable. Versioned deviations follow [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md).

---

## 11. Critical Reuse Set

The Critical Reuse Set contains shared responsibilities that, based on accepted G1 architecture and the current platform audit, are **genuinely important** to product execution and expected to be inherited from Intergrax.

| Rule | Meaning |
|------|---------|
| Product architecture drives the set | Item must map to a real product responsibility |
| No metric padding | Do not inflate the set with trivial library reuse |
| No strength-picking | Do not select items solely because they are known Intergrax strengths |
| No score protection | Do not omit difficult responsibilities merely to protect score |

PASS / PARTIAL / FAIL semantics remain canonical in [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md). This document does not redefine them.

---

## 12. Scaffold gate

Application scaffold starts **only after**:

- G0 **ACCEPTED**
- G1 **ACCEPTED**
- Platform Capability Audit complete
- G2 / T0 **ACCEPTED** and frozen

The scaffold:

- creates product application structure;
- wires approved existing platform boundaries;
- does **not** invent platform abstractions;
- does **not** implement speculative shared-core changes;
- does **not** copy LKW domain code unless an independently shared reusable component is already canonical.

A scaffold starting successfully is **not** meaningful product proof.

---

## 13. G3 — First real vertical slice

G3 must prove **one real user/business outcome** end-to-end.

It **cannot** be satisfied by only:

- application starts;
- health endpoint works;
- one model call succeeds;
- database connects;
- one generic agent executes;
- scaffold tests pass;
- mocked happy-path without product semantics.

The slice must include enough real product semantics to test architecture and platform boundaries.

**Non-normative illustrations** (do not freeze implementation detail here):

| Product | Example of a real slice |
|---------|-------------------------|
| Contract Recovery | Contract/invoice evidence → concrete discrepancy → evidence-backed recovery finding |
| Supply Disruption | Disruption signal → affected item/order → actionable mitigation recommendation |
| Third-Party Risk | Vendor request → relevant evidence → defensible decision/result |
| Deployment Guardian | Release candidate → current evidence → GO/NO-GO decision |

---

## 14. G4 — Material platform pressure

Canonical governance: [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §7–§8.

A Product Session must **stop** before implementing a material shared-platform change and raise G4.

Examples:

- new shared contract;
- change to runtime identity / lifecycle;
- new generalized evidence model;
- shared governance semantics;
- recovery semantics;
- Vendor Knowledge generalization;
- shared persistence abstraction;
- shared action / side-effect boundary;
- product-specific branch proposed inside shared core;
- private reimplementation of a platform responsibility.

Portfolio Control then decides classification.

**No Product Session may self-approve `EXTENDED_GENERALLY`.**

---

## 15. LKW anti-copy rule

Forbidden reasoning:

```text
LKW uses Workspace          → every product needs Workspace
LKW uses Indexed Source     → finance product needs Indexed Source
LKW uses Hybrid Ask         → all products should use Hybrid Ask
LKW uses RAG                → all products should index evidence
LKW uses conversation UI    → every product needs chat
LKW has Live Access Binding → every external authorization is a Live Access Binding
```

Instead ask:

- what is the product's actual domain concept?
- is there a genuinely shared lower-level platform abstraction?
- can LKW and the new product consume that abstraction differently?

LKW is the **reference baseline**, not a product template. RAG, Hybrid Ask, workspace, and conversation semantics are **not** mandated.

---

## 16. Product may disprove the platform

A valid outcome may be:

- `REUSED_UNCHANGED`
- `REUSED_CONFIGURED`
- `EXTENDED_GENERALLY`
- `PRODUCT_OWNED`
- `PLATFORM_LEAK`
- discovery that a supposed shared abstraction was LKW-specific
- discovery that a current platform boundary is incorrectly shaped
- discovery that the product should not use a platform capability at all

The program does **not** require every product to produce a PASS reuse score.

Commercial / product success and platform proof remain **separate scorecards**.

---

## 17. Product Session bootstrap deliverables

Before implementation, every Product Session must produce authoritative **product-owned** semantic artifacts for:

| Artifact | Owner | Accepted by |
|----------|-------|-------------|
| G0 product baseline | Product Session | Portfolio Control |
| Frozen initial wedge | Product Session | Portfolio Control (with G0) |
| G1 product architecture | Product Session | Portfolio Control |
| Product roadmap / implementation plan | Product Session | Portfolio Control (existence/consistency) |
| T0 reuse baseline | Product Session prepares | Portfolio Control |
| First vertical-slice definition | Product Session | Portfolio Control (definition before scaffold; G3 after evidence) |

Exact filenames are **not** prescribed here. Later product-card / session structure may define storage. Required meaning and ownership are binding now.

Portfolio Control reviews and accepts central gate status.

---

## 18. Ownership

| Owner | Owns |
|-------|------|
| Product Session | Product problem, wedge, domain semantics, product architecture, product roadmap, implementation, product evidence, T0 preparation |
| Portfolio Control | Independent gate review; acceptance/rejection of G0/G1/G2/G3/G4; central status; cross-product impact review; final platform-impact classification; protection against platform-first bias |
| Platform sessions | Implementation of accepted shared-platform changes after G4 disposition |

Product Sessions report evidence. They do not self-certify central status or self-approve material shared-platform classification.

---

## 19. Bootstrap failure conditions

Bootstrap fails if any of the following occur:

- product justified primarily by Intergrax fit;
- implementation begins before T0;
- scaffold precedes accepted architecture;
- product architecture copied from LKW without independent domain reason;
- Critical Reuse Set chosen to maximize score;
- expected `PRODUCT_OWNED` scope adjusted after observing failure;
- shared platform modified without G4;
- product privately rebuilds a known platform responsibility;
- first vertical slice has no real product/business semantics;
- current product evidence is confused with target architecture;
- planned platform capability treated as implemented.

---

## 20. Gate summary

| Gate | Question | Must exist before PASS |
|------|----------|------------------------|
| G0 | Is this a real product worth exploring independently of Intergrax? | Product baseline + market/wedge evidence |
| G1 | Do we know how the product should work? | Accepted product architecture |
| G2 | What platform reuse do we expect BEFORE code? | Frozen preregistered T0 |
| Scaffold | Can implementation start safely? | G0 / G1 / G2 accepted |
| G3 | Does first real product workflow work end-to-end? | Meaningful vertical-slice evidence |
| G4 | Does product require material shared-platform change? | Portfolio Control disposition |

G5–G8: [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md). Not redefined here.

---

## 21. Relation to later gates

After bootstrap:

- normal product development continues asynchronously;
- G4 is triggered as needed;
- G5 / G6 / G7 / G8 remain governed by [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md);
- T1 and M1–M6 are later audit outcomes, **not** bootstrap success metrics.

---

## 22. Definition of READY_FOR_IMPLEMENTATION

A new product is **READY_FOR_IMPLEMENTATION** only when **all** of the following are true:

- G0 accepted;
- initial wedge frozen;
- G1 architecture accepted;
- platform capability audit complete;
- T0 accepted and immutable baseline stored;
- exact starting SHA recorded;
- Critical Reuse Set frozen;
- first vertical slice defined;
- no unresolved material platform change bypasses G4.

Then — and only then — scaffold/code begins.
