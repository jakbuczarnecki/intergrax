# Cross-Product Reuse Proof Contract

## Status

**Maintainer-level pre-registered proof contract.**

This document defines **how** Intergrax will later be evaluated for cross-product reuse across a second, meaningfully different real product. It freezes the experiment methodology **before** Product #2 implementation begins.

This document does **not**:

- choose the final Product #2;
- build Product #2;
- claim that cross-product reuse has been demonstrated;
- change public positioning or architecture;
- establish a reuse percentage, faster development, lower cost, commercial validation, or validated second-product demand.

Cross-product reuse remains a **strategic hypothesis** in public documentation ([WHY_INTERGRAX](../../overview/WHY_INTERGRAX.md)) until independently reviewed and accepted T1 evidence exists.

---

## Core principle

```text
Product #2 exists to solve a real product problem.

The reuse proof observes what the product can inherit from Intergrax.

The product must NOT be invented or distorted merely to make Intergrax reuse look good.

Product-first remains authoritative.
```

The [Product-First MVP Development Brief](PRODUCT_FIRST_MVP.md) governs all product development. When a second product is used to test platform reuse, that product must originate from a real product hypothesis — not from a desire to exercise platform features.

---

## Canonical proof question

```text
Can a second, meaningfully different real product keep ownership of its domain
workflow and business semantics while reusing the existing Intergrax application
operating layer without product-specific hacks in shared platform core or
private duplication of platform responsibilities?
```

---

## Product #2 diversity gate

Product #2 must be **meaningfully different** from LKW. A renamed or reshaped LKW workflow does not provide credible cross-product evidence.

This contract does **not** freeze a specific market product. The concrete product must come from product or customer evidence.

### LKW reference profile (current product wedge)

- knowledge-centric;
- primarily Ask / read;
- grounding and source focused.

### Preferred stress-test archetype (example / hypothesis only)

A **strong contrasting archetype** — not an approved future Product #2 — would stress different platform boundaries:

- action / workflow-centric;
- external consequential effects;
- policy / approval boundaries;
- recovery, escalation, and compensation;
- reviewable evidence around actions.

A governed operations-style application is a strong **example** of this archetype. It illustrates what “meaningfully different” means; it is **not** the selected Product #2.

### Diversity gate checklist (T0)

Before implementation begins, the frozen T0 record must explain **why** Product #2 is meaningfully different from LKW across at least:

- primary user job and pain;
- primary workflow shape (read vs act, knowledge vs operations);
- consequential side effects and approval posture;
- evidence and recovery requirements;
- product-owned semantics that LKW does not share.

---

## T0 — Pre-registration baseline

**Before the first Product #2 implementation commit**, freeze a T0 baseline. Once implementation begins, measurement rules may **not** be retroactively changed merely to improve the proof result. Any legitimate rule correction must be explicitly versioned, dated, and explained.

### Required T0 contents

| Field | Description |
| ----- | ----------- |
| **Product hypothesis** | What Product #2 is and why it should exist |
| **Target user** | First concrete user, not a generic segment |
| **Real problem** | Observable pain and current alternatives |
| **Primary workflow** | End-to-end observable steps |
| **Meaningful difference from LKW** | Why this product stresses different boundaries |
| **Starting Intergrax commit SHA** | Exact platform baseline for the experiment |
| **Required platform-responsibility matrix** | Every platform responsibility Product #2 needs (see below) |
| **Expected reuse candidates** | Which responsibilities are expected to be `REUSED_UNCHANGED` or `REUSED_CONFIGURED` |
| **Known expected gaps** | Anticipated `EXTENDED_GENERALLY` or open questions |
| **Measurement methodology** | How M1–M6 will be computed |
| **PASS / PARTIAL / FAIL rules** | Frozen qualitative criteria (this contract) |

T0 should be stored as a dated maintainer record linked from the Product #2 plan. The starting commit SHA must match the repository state at T0 freeze.

---

## Platform-responsibility matrix

For every **platform responsibility** required by Product #2’s workflow, classify the **final** result as exactly one of the following categories. Report **all** categories — not only successful reuse.

### Categories

#### `REUSED_UNCHANGED`

Existing shared mechanism consumed **without** platform modification.

#### `REUSED_CONFIGURED`

Existing mechanism reused through its intended configuration, policy, adapter, or dependency-injection contract **without** changing core platform semantics.

#### `EXTENDED_GENERALLY`

Real product pressure exposed a missing reusable capability; the platform was extended through a **general** contract. This is useful platform evolution but is **not** counted as pure reuse.

#### `PRODUCT_OWNED`

Behavior correctly belongs to the product: domain workflow, UX, business semantics, business acceptance, product-specific policy **meaning**, and similar product accountability.

`PRODUCT_OWNED` responsibilities are **excluded** from reuse-ratio denominators.

#### `PLATFORM_LEAK`

Product-specific branching, private infrastructure duplication, bypass of shared platform contracts, or product-specific behavior leaking into platform core.

### Candidate responsibilities (include only what the workflow requires)

At T0, consider — where Product #2 actually needs them:

- execution identity;
- tenant / principal context;
- policy evaluation;
- authorization / enforcement boundary;
- HITL / human continuation;
- meaningful side-effect authorization;
- tool / integration execution;
- execution evidence / provenance;
- canonical run history;
- observability;
- retry / recovery;
- compensation / escalation (where applicable);
- idempotency;
- hosting / lifecycle;
- application composition;
- other existing platform mechanisms genuinely required by the workflow.

Do **not** require Product #2 to use capabilities it does not need merely to improve reuse metrics.

---

## Metrics

Report exact numerators, denominators, and derived values. Do **not** invent an arbitrary minimum reuse percentage in this contract.

### M1 — Responsibility Reuse Ratio

```text
(REUSED_UNCHANGED + REUSED_CONFIGURED)
/
all required PLATFORM responsibilities in the frozen T0 matrix
```

- **Denominator:** all required platform responsibilities in the frozen T0 matrix.
- **Exclude:** `PRODUCT_OWNED` from the denominator.
- **Report:** numerator, denominator, and percentage.

### M2 — Platform Expansion Ratio

```text
EXTENDED_GENERALLY
/
all required platform responsibilities
```

- **Report:** exact count and percentage.

### M3 — Core Product Hack Count

Number of product-identity or product-workflow-specific branches or special cases introduced into **shared platform core**.

- **Hard target:** `0`.

Conceptual anti-pattern:

```text
if product == "Product2": ...
```

### M4 — Private Platform Duplication Count

Number of required platform responsibilities that Product #2 implements **privately** instead of consuming the shared Intergrax mechanism, without a justified architectural reason documented at T0 or T1.

- **Hard target:** `0`.

Examples of unjustified private duplication: a product-local policy engine, identity system, HITL mechanism, retry/recovery framework, evidence journal, execution runtime, or tool gateway when the shared responsibility should apply.

### M5 — Boundary Integrity

For every new component or meaningful change, record whether it is:

- `PRODUCT_OWNED`, or
- `PLATFORM_OWNED`,

and **why**.

Reject reasoning such as “put it in platform because it may be reusable later.” Placement follows the active product need and the responsibility model in [ARCHITECTURE_OVERVIEW](../../architecture/ARCHITECTURE_OVERVIEW.md).

### M6 — Inherited Capability Set

Record which operational properties Product #2 receives through platform reuse — for example, where applicable:

- identity;
- governance;
- HITL;
- side-effect control;
- evidence;
- history;
- observability;
- recovery;
- hosting;
- integration boundaries.

This is a **capability inventory** for the proof record, not a public marketing claim until the proof is accepted.

---

## Hard fail conditions

Any of the following is an automatic architectural failure:

1. **Product-specific branching in shared platform core** — e.g. product-identity conditionals in platform code paths.

2. **Private rebuild of a platform responsibility** — Product #2 reimplements policy engine, identity, HITL, retry/recovery, evidence journal, execution runtime, tool gateway, or equivalent when the shared responsibility should be used.

3. **Contract bypass** — Product #2 requires violating or bypassing existing platform contracts merely to make the workflow work.

4. **Insufficient diversity** — Product #2 is essentially a renamed or reshaped LKW workflow and does not provide meaningful cross-product evidence.

5. **Retroactive measurement gaming** — measurement rules are chosen or materially rewritten after implementation results are known to improve the apparent proof.

---

## Platform evolution is not automatic failure

Preserve Product-First platform-gap handling ([PRODUCT_FIRST_MVP](PRODUCT_FIRST_MVP.md) §9):

```text
product need
→ bounded architecture audit
→ classify gap
→ general platform contract if justified
→ product consumes shared implementation
→ end-to-end revalidation
```

Legitimate missing reusable capabilities are classified `EXTENDED_GENERALLY`, not `REUSED_UNCHANGED`.

A proof with substantial platform expansion may be **PARTIAL** rather than **PASS**, even when the resulting architecture is sound. Do **not** treat every new platform change as failure.

---

## PASS / PARTIAL / FAIL

Qualitative classification. No minimum reuse percentage is defined in this contract.

### PASS

Requires **all** of:

- Product #2 is meaningfully different from LKW;
- a real vertical slice works end to end;
- shared responsibilities are substantially consumed through existing Intergrax contracts;
- Core Product Hack Count (M3) = `0`;
- Private Platform Duplication Count (M4) = `0`;
- product / domain semantics remain product-owned;
- any platform extensions are general and explicitly classified as `EXTENDED_GENERALLY`;
- existing LKW semantics are not product-specifically altered for Product #2;
- exact responsibility matrix and M1–M6 are reported;
- T0 rules were frozen before implementation.

### PARTIAL

- the product works and architecture remains clean;
- but material platform expansion is required, reuse is weaker than expected, or important abstractions require general redesign.

A PARTIAL result remains useful and must be reported honestly.

### FAIL

- any hard fail condition; or
- a result that cannot credibly test cross-product reuse.

---

## T1 — Post-implementation evidence record

After Product #2 vertical-slice completion, publish a T1 evidence record containing:

| Item | Description |
| ---- | ----------- |
| T0 starting commit SHA | Frozen baseline |
| Product #2 implementation end SHA | Final measured state |
| Final working vertical slice | What was demonstrated end to end |
| Exact responsibility matrix | Final classification per responsibility |
| M1–M6 results | All metrics with numerators and denominators |
| Shared platform files changed for Product #2 | List with rationale |
| Platform-gap decisions | Audit outcomes and `EXTENDED_GENERALLY` items |
| Product-owned implementation scope | What remained in the product |
| Inherited capability set | M6 detail |
| Violations / exceptions | Any documented deviations from targets |
| Final result | PASS / PARTIAL / FAIL |
| Reviewer rationale | Independent review conclusion |

**Public cross-product reuse claims may be promoted only after this result is independently reviewed and accepted.**

Until then, [WHY_INTERGRAX](../../overview/WHY_INTERGRAX.md) correctly describes cross-product reuse as a strategic hypothesis.

---

## Relationship to other documents

| Document | Role |
| -------- | ---- |
| [PRODUCT_FIRST_MVP](PRODUCT_FIRST_MVP.md) | Authoritative product-development rule; links to this contract for reuse experiments |
| [WHY_INTERGRAX](../../overview/WHY_INTERGRAX.md) | Public strategic hypothesis — unchanged by this contract |
| [ARCHITECTURE_OVERVIEW](../../architecture/ARCHITECTURE_OVERVIEW.md) | Responsibility-boundary reference for matrix and M5 classification |
| Product #2 plan (future) | Must exist before T0; not created by this contract |

---

## What this contract does not establish

This document defines a **future evaluation method**. It does **not** establish:

- cross-product reuse as a measured fact;
- faster product development;
- lower implementation cost;
- a reuse percentage target;
- Product #2 existence or selection;
- validated second-product demand;
- commercial validation.
