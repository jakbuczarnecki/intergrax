# Third-Party Risk - Product Session Brief

**Document type:** Durable product-specific session mission artifact  
**Owner:** Third-Party Risk Product Session (future launch)  
**Audience:** Future session operator / MP-22 Session Launch Pack assembler  
**Status:** **SELECTED** - Pre-bootstrap; G0 **PENDING**

> **This is NOT the final session launch prompt.**  
> It is a durable product-specific mission and context artifact consumed later by **MP-22 Session Launch Pack**.  
> Do not treat this file as a conversational bootstrap prompt. Common operating behavior lives in [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md).

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Product Session | Third-Party Risk Product Session |
| Product | Third-Party Risk Decision Operator |
| Short name | Third-Party Risk |
| Program role | Newly selected application |
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| Control card | [products/third-party-risk.md](../products/third-party-risk.md) |

| Bootstrap item | Status |
|----------------|--------|
| G0 Product Baseline | **PENDING** |
| G1 Product Architecture | **NOT STARTED** |
| G2 / T0 reuse baseline | **NOT CREATED** |
| Application scaffold | **NOT CREATED** |
| Implementation | **NOT STARTED** |
| Cross-product reuse evidence | **NONE** |
| Next allowed action | **G0 Product Baseline preparation** |

Portfolio caveat retained: **initial wedge still requires sharpening.**

---

## 2. Mission

Move a **real vendor request** from evidence gathering through review to a **defensible decision** - approve, reject, or conditional - with auditability, not merely collect documents or orchestrate tasks.

**Product-first rule:**

```text
The product is not being built to demonstrate Intergrax.
Intergrax reuse is observed as a consequence of building the product.
```

---

## 3. Why this product exists independently of Intergrax

Third-party risk decisions are cross-functional, evidence-heavy, and audit-sensitive. Risk, security, procurement, and compliance teams need faster cycles **without** weaker defensibility.

End-to-end TPRM orchestration is an established category. The product must earn a **standalone buying need**, not replicate incumbent feature checklists.

---

## 4. Current authoritative starting state

| Item | Status |
|------|--------|
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| G0 | **PENDING** |
| G1 | **NOT STARTED** |
| T0 | **NOT CREATED** |
| Scaffold | **NOT CREATED** |
| Implementation | **NOT STARTED** |
| Product architecture | **Does not exist** |
| Evidence beyond selection | **None** |
| Market / customer / commercial validation | **NOT CLAIMED** |

Authoritative index: [third-party-risk control card](../products/third-party-risk.md), [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5.

---

## 5. Product hypothesis / current product truth

**Pre-G0 hypothesis (subject to G0 validation):**

```text
Move a real vendor request from evidence gathering through review to a
defensible decision.
```

Orchestration alone is **not** claimed as unique. Questionnaire automation is a commodity trap. G0 must **especially sharpen the wedge** before architecture.

---

## 6. Buyer / user and economic or operational job

| Dimension | Value |
|-----------|-------|
| Primary buyer | CISO / Risk / Procurement / Compliance |
| Core job | Reduce time and friction while **increasing defensibility** of vendor-risk decisions |
| Economic consequence | Breach liability, audit findings, delayed vendor onboarding, wrong-vendor acceptance |
| Success horizon | Request-to-decision cycle (days to weeks) - not crisis minutes or deployment gates |
| Value unit | Defensible decision record with evidence trail and accountable human roles |

---

## 7. Primary workflow

Target workflow shape for G0 sharpening (not architecture):

```text
Vendor request initiated
  → evidence gathering (questionnaires, attestations, external signals as scoped)
  → risk reasoning and gap identification
  → review / escalation with accountable roles
  → defensible approve / reject / conditional decision
  → audit-ready decision trace
```

G0 must sharpen **wedge before architecture** - what materially improves decision quality beyond orchestration?

---

## 8. What makes this product different from LKW / other products

| Contrast | Third-Party Risk |
|----------|------------------|
| LKW | Workspace knowledge Q&A - not vendor onboarding decisions |
| Contract Recovery | Post-contract spend leakage - not pre-engagement vendor risk |
| Supplier Disruption | In-flight supply crisis mitigation - not vendor approval gate |
| Deployment Guardian | Release authorization - not third-party risk acceptance |

Evidence semantics center on **decision defensibility and auditability**, not answer provenance or operational exposure runway.

---

## 9. Product-specific wedge / kill questions

- What is the **concrete wedge beyond orchestration**?
- Does the product produce a **defensible decision**, or only collect documents?
- Is this a **standalone buying need** or an incumbent TPRM feature?
- How do we avoid **commodity questionnaire automation**?
- What evidence **materially improves decision quality / auditability**?
- Where must **human accountability** remain non-delegable?

---

## 10. Major failure modes / category traps

- **Questionnaire chatbot** - form filling without decision quality.
- **PDF summarizer** - document recap without risk reasoning.
- **Generic compliance workflow** - tasks without defensible outcome.
- **Evidence collection without decision** - repository, not operator.
- **Broad "AI governance platform" expansion** - scope creep before wedge proof.
- **LKW-style RAG answers** mistaken for vendor-risk decisions.

---

## 11. Platform posture

**Before G1:** Do not deeply shape the product around current Intergrax APIs.

**After product architecture:** Perform Platform Capability Audit.

**Before implementation:** Accepted G2/T0 required.

**During implementation:** Material shared platform change needed → **STOP** → **G4**.

Product Session cannot self-approve `EXTENDED_GENERALLY`, `GENUINE_PLATFORM_GAP`, or shared core product-specific behavior.

VIS-3A owns public presentation - not gate status. COMM does not own Portfolio Control authority.

---

## 12. Current gate / first allowed action

**G0 Product Baseline** - preparation and acceptance per [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md).

G0 must **especially sharpen the wedge** before any architecture work. Do **not** start G1, T0, scaffold, or implementation until G0 is accepted.

---

## 13. Evidence the session must eventually produce

Future evidence target (not claimed today):

```text
Real vendor request
  → gathered evidence
  → risk reasoning
  → defensible approve / reject / conditional decision
  → audit-ready trace
```

Also eventually: G3 vertical slice and reuse evidence when T0 exists.

---

## 14. What the session must NOT claim yet

- Product architecture or TPRM platform integrations.
- Customer, commercial, or market validation beyond selection screening.
- Cross-product reuse or accepted platform-impact records.
- Unique wedge without G0 sharpening (portfolio caveat active).
- Replacement of incumbent TPRM suites without evidence.
- Public product presentation (VIS-3A).

---

## 15. Sources of truth

| Topic | Document |
|-------|----------|
| Common session behavior | [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md) |
| Bootstrap contract | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
| Program governance | [MULTI_PRODUCT_PROGRAM.md](../MULTI_PRODUCT_PROGRAM.md) |
| Selection record | [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5 |
| Live portfolio state | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Control card | [products/third-party-risk.md](../products/third-party-risk.md) |
| Reuse methodology | [PRODUCT_REUSE_PROOF.md](../../plans/PRODUCT_REUSE_PROOF.md) |

Product-owned architecture and roadmap: **do not exist yet.**

---

## 16. Handoff expectations

Contact Portfolio Control when:

- **G0 ready** (wedge sharpened);
- **G1 ready**;
- **G2/T0 ready**;
- **G3**;
- **G4 pressure**;
- **G5**, **G6**, major **G7** evidence;
- **G8** recommendation.

Detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](../CROSS_SESSION_COORDINATION.md).
