# Multi-Product Audit Integration

**Document type:** Normative Portfolio Control integration contract  
**Owner:** Portfolio Control  
**Purpose:** Define when and how the multi-product program invokes and consumes the canonical Intergrax audit engine without duplicating it.

---

## 1. Core authority rule

`docs/audit_results/` remains the **single canonical audit engine** and audit source of truth.

Canonical methodology and lifecycle:

| Artifact | Role |
|----------|------|
| [docs/audit_results/README.md](../../../audit_results/README.md) | Global campaign registry and audit entry point |
| [docs/audit_results/AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md) | Canonical adversarial audit methodology (Protocol v2.2) |
| [docs/audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) | Canonical remediation lifecycle |
| `docs/audit_results/<CAMPAIGN>/README.md` | Per-campaign finding register, rollup, and remediation trace |

Portfolio Control artifacts **MUST NOT** duplicate:

- campaign registry;
- finding register;
- finding lifecycle;
- severity model;
- remediation state;
- verification state;
- immutable audit snapshots;
- campaign rollups.

Portfolio Control may only:

- trigger or request an audit;
- record gate-level decision and status;
- link audit campaigns and findings;
- consume accepted audit evidence;
- translate accepted audit evidence into portfolio or platform decisions where another program artifact legitimately owns that decision.

**Principle:** Portfolio Control uses the canonical audit engine. Portfolio Control does not duplicate the canonical audit engine.

---

## 2. Gate review ≠ audit campaign

A **Portfolio Control gate review** is a program decision checkpoint governed by [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) and product bootstrap rules. It evaluates product evidence, architecture, reuse posture, market signals, or portfolio direction.

An **audit campaign** is an adversarial, evidence-producing process governed by [AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md). It produces pinned implementation evidence, conformance matrices, findings, and verdicts under the canonical audit lifecycle.

Not every gate should create an audit campaign. Avoid audit inflation and campaign spam. Invoke the audit engine only when independent adversarial evidence is material to the gate decision.

---

## 3. G0–G8 integration matrix

| Gate | Primary purpose | Canonical control mode | Audit engine |
|------|-----------------|------------------------|--------------|
| **G0 — Product Baseline** | Product/market baseline acceptance | Portfolio Control review | **NO** canonical audit campaign by default |
| **G1 — Product Architecture** | Architecture acceptance | Portfolio Control architecture review | **CONDITIONAL** — bounded DOMAIN/LAYER or CONCEPTUAL/CROSS-DOMAIN audit only when architectural risk or uncertainty justifies adversarial audit |
| **G2 — T0 Reuse Baseline** | Preregistration per [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) | PRODUCT_REUSE_PROOF / Portfolio Control acceptance | **NO** audit campaign — T0 is a frozen experimental baseline, not an audit result |
| **G3 — First Real Vertical Slice** | Product evidence / end-to-end product outcome | Product Session evidence + Portfolio Control gate review | **NO** audit campaign by default — may be triggered if platform-conformance uncertainty is material |
| **G4 — Material Platform Pressure** | Decide product-owned vs platform-owned / general extension | Portfolio Control decision | **CONDITIONAL / recommended** for ambiguity or materiality — PLATFORM CONSUMER AUDIT and/or CONCEPTUAL/CROSS-DOMAIN AUDIT |
| **G5 — MVP / Major Proof** | Major product/proof acceptance | Product Session + Portfolio Control | **CONDITIONAL** — existing audit engine may provide independent proof when material claims require adversarial evidence |
| **G6 — T1 Reuse Audit** | Final independent reuse/platform-consumer evaluation | PRODUCT_REUSE_PROOF T1 + Portfolio Control | **REQUIRED** — MUST use PLATFORM CONSUMER AUDIT as independent implementation evidence; T1 / M1–M6 remain owned by PRODUCT_REUSE_PROOF |
| **G7 — Market Validation** | Customer/market evidence | Portfolio Control / business review | **NO** platform audit campaign |
| **G8 — Continue / Accelerate / Reduce / Pause / Stop** | Portfolio decision | Portfolio Control | **NO** audit campaign by default — may consume prior audit evidence but does not become an audit |

Gate meanings G0–G8 remain as defined in [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md). This document adds audit-integration semantics only.

---

## 4. G4 integration

A G4 decision may be made directly when evidence is simple and ownership is obvious.

**Trigger a canonical audit** when one or more apply:

- the product claims a shared capability is missing but existing mechanisms may already cover it;
- the proposed extension touches multiple shared domains;
- the product proposes a competing universal mechanism;
- ownership between product and platform is ambiguous;
- LKW or another active product may be warped or broken;
- implementation appears to bypass a canonical mechanism;
- a proposed "general" abstraction may actually encode one product's semantics;
- provider/backend abstractions may be compromised;
- material production guarantees are at risk.

**Select audit shape based on the question:**

| Question | Audit shape |
|----------|-------------|
| "Is the product consuming Intergrax correctly?" | **PLATFORM CONSUMER AUDIT** |
| "Is the proposed shared abstraction/ownership model correct across multiple platform domains?" | **CONCEPTUAL / CROSS-DOMAIN AUDIT** |
| Bounded layer-specific architectural risk | **DOMAIN / LAYER AUDIT** (when appropriate) |

Do not invent a special G4 audit shape. Use existing Protocol v2.2 scope types only.

---

## 5. G6 / T1 — required integration

For preregistered new products, G6 requires canonical audit integration. The sequence is:

```text
T0 frozen under PRODUCT_REUSE_PROOF
        ↓
product implementation
        ↓
PLATFORM CONSUMER AUDIT @ exact SHA
        ↓
canonical audit findings / conformance evidence
        ↓
T1 PRODUCT_REUSE_PROOF evaluation
        ↓
M1–M6 + PASS / PARTIAL / FAIL
        ↓
Portfolio Control gate decision
```

**PLATFORM CONSUMER AUDIT does NOT replace T1.**

**T1 does NOT replace PLATFORM CONSUMER AUDIT.**

They answer different questions:

| Process | Question |
|---------|----------|
| PLATFORM CONSUMER AUDIT | Did implementation actually consume Intergrax correctly? |
| T1 (PRODUCT_REUSE_PROOF) | Against the preregistered experiment, what reuse outcome did we obtain? |

T1 methodology, metrics, and outcome rules remain canonical in [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md).

---

## 6. Two classification systems — do not collapse

### Canonical audit consumer classifications

From [AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md) section D3 (platform consumer conformance matrix):

- `REUSED`
- `THIN ADAPTER`
- `JUSTIFIED SPECIALIZATION`
- `DUPLICATED`
- `BYPASSED`
- `MISSING PLATFORM CAPABILITY`
- `NOT APPLICABLE`
- `INSUFFICIENT EVIDENCE`

### PRODUCT_REUSE_PROOF classifications

From [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md):

- `REUSED_UNCHANGED`
- `REUSED_CONFIGURED`
- `EXTENDED_GENERALLY`
- `PRODUCT_OWNED`
- `PLATFORM_LEAK`

> **Audit classification is evidence input to T1. It is NOT an automatic T1 classification.**

Do not collapse these systems. Do not define deterministic one-to-one conversion where evidence is ambiguous.

---

## 7. Interpretation / mapping rules

Conservative guidance for T1 evaluators. Ambiguous cases require explicit reasoning against frozen T0 and accepted audit evidence.

| Audit classification | T1 interpretation guidance |
|----------------------|----------------------------|
| `REUSED` | Strong evidence candidate for `REUSED_UNCHANGED`; may require T0 comparison to determine whether configuration changed semantics |
| `THIN ADAPTER` | May support `REUSED_CONFIGURED`; may coexist with `PRODUCT_OWNED` adapter or domain translation; requires ownership and T0 analysis |
| `JUSTIFIED SPECIALIZATION` | Does not automatically mean `PRODUCT_OWNED`; shared platform responsibility may still be reused beneath product-owned specialization; evaluate responsibility granularity against T0 |
| `DUPLICATED` | Strong evidence of `PLATFORM_LEAK` / M4 violation where T0 classified the responsibility as platform-owned; do not silently reclassify after implementation |
| `BYPASSED` | Strong evidence of `PLATFORM_LEAK` where mandatory platform guarantees were expected; severity and effect depend on exact T0 responsibility and audit findings |
| `MISSING PLATFORM CAPABILITY` | Indicates a genuine gap may exist; does **NOT** automatically equal `EXTENDED_GENERALLY` — that requires a general reusable extension actually accepted and implemented under program governance |
| `NOT APPLICABLE` | Valid where the product truly does not require the concern; compare against T0 — unexpected N/A requires explanation |
| `INSUFFICIENT EVIDENCE` | Cannot be treated as successful reuse; T1 must remain unresolved or qualified for that responsibility until evidence exists |

**`PRODUCT_OWNED`:** determined from product responsibility ownership, T0, and architecture — not generated automatically by consumer audit classification.

---

## 8. Finding authority / no duplication

If an audit campaign creates finding IDs, their canonical lifecycle remains **only** in:

`docs/audit_results/<CAMPAIGN>/README.md`

Do **not** copy full finding state into:

- product control cards;
- [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md);
- [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md);
- Product Session roadmaps;
- other portfolio artifacts.

**Allowed references:**

| Artifact | Allowed content |
|----------|-----------------|
| Control card | Latest relevant audit campaign link; overall or layer verdict; selected finding IDs relevant to current gate; concise implication |
| PORTFOLIO_STATUS | Gate status; latest audit/verdict reference; concise portfolio consequence |
| PLATFORM_IMPACT_LEDGER | Accepted platform-impact conclusions only when that artifact legitimately owns the impact classification; never duplicate audit finding lifecycle |
| Architecture / plan | May reference canonical finding IDs where required by [AUDIT_REMEDIATION_PROTOCOL.md](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) |

Control cards and portfolio status are **link/index surfaces**, not second finding registers.

---

## 9. Remediation ownership

Any remediation of canonical audit findings follows [AUDIT_REMEDIATION_PROTOCOL.md](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md).

Portfolio Control does not create a second remediation workflow.

- Product Session may implement remediation assigned to its product scope.
- Platform or shared work may implement platform remediation.
- Finding status remains canonical in the audit campaign register.

G4/G6 gate status and finding remediation status are **separate concepts**. A gate may be accepted while related audit findings remain open; conversely, remediation progress does not automatically advance a gate.

---

## 10. Audit campaign scope discipline

Portfolio Control should request the **smallest audit scope** capable of falsifying the material claim.

Examples:

- one application consumer audit for G6;
- bounded consumer slice for a G4 ambiguity;
- cross-domain audit for a shared abstraction touching several layers;
- do not automatically audit the whole platform when one product concern is under review.

Reuse existing protocol context and read discipline. No special Portfolio Control audit campaign format.

---

## 11. LKW

Local Knowledge Workspace (LKW):

- remains the existing reference product;
- may receive future PLATFORM CONSUMER AUDIT like any other active application;
- has **no retroactive T0**;
- therefore it **cannot** receive retroactive T1 reuse scoring against a fabricated preregistration baseline.

Audit evidence for LKW is still valid product and platform evidence, but it is not a retroactive PRODUCT_REUSE_PROOF experiment.

---

## 12. New products

For each of the four newly selected products (per [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md)):

1. G0 / G1 / G2 bootstrap occurs before implementation.
2. T0 exists before the first implementation commit.
3. Consumer audit becomes **mandatory at G6**.
4. Resulting audit evidence feeds T1.
5. Cross-product conclusions may then feed Portfolio Control and [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) where appropriate.

No G6 has been executed yet for any preregistered new product.

---

## 13. Public documentation / VIS-3A

Public docs do not consume raw unreviewed findings as marketing truth.

Public capability, platform, and reuse claims should be downstream from accepted product, proof, and audit evidence.

Detailed VIS-3A visual workflow is not defined here. [MP-20](PORTFOLIO_STATUS.md) owns cross-session handoffs.

---

## 14. What this integration does not create

This contract explicitly does **not** create:

- `reviews/*` or any competing audit workspace;
- a new audit protocol;
- a new campaign format;
- a new severity model;
- a new finding status model;
- a new remediation protocol;
- a new consumer conformance matrix;
- duplicate T0/T1 methodology;
- automatic conversion between audit and T1 classification systems.

---

## 15. Decision tree

```text
Is this a Portfolio Control gate?
        ↓
Does it require independent adversarial implementation/architecture evidence?
  NO → normal gate review (no audit campaign)
  YES
        ↓
Is the question primarily "consumer uses platform correctly"?
  YES → PLATFORM CONSUMER AUDIT
  NO
        ↓
Is the question cross-domain / shared-abstraction correctness?
  YES → CONCEPTUAL / CROSS-DOMAIN AUDIT
  NO → bounded DOMAIN / LAYER audit if appropriate

At G6 for preregistered new products:
  always PLATFORM CONSUMER AUDIT → feeds T1 → Portfolio Control gate decision
```

---

## 16. Source-of-truth table

| Concern | Canonical owner |
|---------|-----------------|
| Audit methodology | [docs/audit_results/AUDIT_PROTOCOL.md](../../../audit_results/AUDIT_PROTOCOL.md) |
| Audit campaigns and findings | `docs/audit_results/<campaign>/README.md` |
| Remediation methodology | [docs/audit_results/AUDIT_REMEDIATION_PROTOCOL.md](../../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md) |
| T0 / T1 / M1–M6 reuse experiment | [docs/project/maintainers/plans/PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Gate status / product status | Portfolio Control — [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md), control cards |
| Product architecture / implementation | Respective Product Session |
| Accepted platform impact | [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Public presentation | Downstream docs / VIS stream |

---

## Related documents

| Question | Document |
|----------|----------|
| How the program operates | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| How new products bootstrap | [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) |
| Reuse proof methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Current portfolio state | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
