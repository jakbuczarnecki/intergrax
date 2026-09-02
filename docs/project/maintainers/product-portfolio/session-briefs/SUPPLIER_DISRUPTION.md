# Supplier Disruption - Product Session Brief

**Document type:** Durable product-specific session mission artifact  
**Owner:** Supplier Disruption Product Session (future launch)  
**Audience:** Future session operator / MP-22 Session Launch Pack assembler  
**Status:** **SELECTED** - Pre-bootstrap; G0 **PENDING**

> **This is NOT the final session launch prompt.**  
> It is a durable product-specific mission and context artifact consumed later by **MP-22 Session Launch Pack**.  
> Do not treat this file as a conversational bootstrap prompt. Common operating behavior lives in [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md).

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Product Session | Supplier Disruption Product Session |
| Product | Supplier Disruption Response Operator |
| Short name | Supplier Disruption |
| Program role | Newly selected application |
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| Control card | [products/supplier-disruption.md](../products/supplier-disruption.md) |

| Bootstrap item | Status |
|----------------|--------|
| G0 Product Baseline | **PENDING** |
| G1 Product Architecture | **NOT STARTED** |
| G2 / T0 reuse baseline | **NOT CREATED** |
| Application scaffold | **NOT CREATED** |
| Implementation | **NOT STARTED** |
| Cross-product reuse evidence | **NONE** |
| Next allowed action | **G0 Product Baseline preparation** |

---

## 2. Mission

Turn an **active supply disruption** into a **mitigation plan** and ultimately **controlled mitigation actions** before operational impact materializes - not stop at alerts or recommendations.

**Product-first rule:**

```text
The product is not being built to demonstrate Intergrax.
Intergrax reuse is observed as a consequence of building the product.
```

---

## 3. Why this product exists independently of Intergrax

Supply disruptions compress decision time while exposure spreads across SKUs, orders, and sites. Operations leaders need a workflow that converts disruption signal into scoped exposure and **controlled mitigation** - not a news feed or generic copilot summary.

Selection recorded severe operational pain, high economic consequence, and crisis-oriented workflow as admission reasons. Integration and GTM complexity remain material caveats.

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

Authoritative index: [supplier-disruption control card](../products/supplier-disruption.md), [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5.

---

## 5. Product hypothesis / current product truth

**Pre-G0 hypothesis (subject to G0 validation):**

```text
Turn an active supply disruption into a mitigation plan and ultimately
controlled mitigation actions before operational impact materializes.
```

Detection plus recommendation alone is **not** claimed as differentiated. Incumbent supply-risk platforms already move toward mitigation workflows. The wedge must be **execution-oriented**.

---

## 6. Buyer / user and economic or operational job

| Dimension | Value |
|-----------|-------|
| Primary buyer | COO / Supply Chain / Procurement |
| Core job | Convert disruption into affected business scope, time/risk picture, and controlled mitigation |
| Economic consequence | Stockouts, line stoppages, revenue loss, expedite cost |
| Success horizon | **Crisis tempo** - hours to days, not financial audit cycles |
| Value unit | Scoped exposure + actionable mitigation with human approval boundaries |

---

## 7. Primary workflow

Target workflow shape for G0 sharpening (not architecture):

```text
Disruption signal (supplier, geo, category, force majeure, etc.)
  → affected supplier / SKU / order scope
  → exposure and runway (time, inventory, revenue at risk)
  → mitigation options ranked by feasibility and impact
  → controlled decision / action with approval trace
```

G0 must sharpen: **disruption → affected supplier/SKU/order → exposure/runway → mitigation options → controlled action.**

---

## 8. What makes this product different from LKW / other products

| Contrast | Supplier Disruption |
|----------|---------------------|
| LKW | Durable knowledge operations - not supply-chain crisis response |
| Contract Recovery | Contract/spend leakage - not operational disruption mitigation |
| Third-Party Risk | Vendor onboarding risk decision - not in-flight supply crisis |
| Deployment Guardian | Software release gate - not physical supply exposure |

Evidence semantics center on **operational exposure and mitigation**, not provenance of answers or financial discrepancy.

---

## 9. Product-specific wedge / kill questions

- Does the workflow reach **mitigation**, or stop at recommendation?
- Can required **ERP / procurement / supplier data** realistically be integrated for a pilot?
- Is the workflow **fast enough for crisis tempo**?
- What actions require **human approval** vs autonomous execution?
- Do **incumbents** already produce the same end-to-end outcome?
- What is valuable **before full autonomous action** exists?

---

## 10. Major failure modes / category traps

- **News / supplier-risk monitoring dashboard** - awareness without mitigation path.
- **Generic alert summarizer** - LLM recap of headlines without scoped business impact.
- **Recommendation-only agent** - plans without controlled execution hooks.
- **Unrealistic autonomous supplier switching** - fantasy automation before trust.
- **Giant integration project** before proving user value in a bounded slice.
- **Copying LKW RAG patterns** for operational crisis workflows.

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

G0 sharpens disruption-to-mitigation chain before architecture. Do **not** start G1, T0, scaffold, or implementation until G0 is accepted.

---

## 13. Evidence the session must eventually produce

Future evidence target (not claimed today):

```text
Realistic disruption scenario
  → impacted items / orders
  → time / exposure quantification
  → mitigation plan
  → controlled decision / action trace
```

Also eventually: G3 vertical slice and reuse evidence when T0 exists.

---

## 14. What the session must NOT claim yet

- Product architecture, ERP integrations, or implementation.
- Customer, commercial, or market validation beyond selection screening.
- Cross-product reuse or platform-impact records.
- Autonomous supplier switching as proven capability.
- End-to-end incumbent parity without evidence.
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
| Control card | [products/supplier-disruption.md](../products/supplier-disruption.md) |
| Reuse methodology | [PRODUCT_REUSE_PROOF.md](../../plans/PRODUCT_REUSE_PROOF.md) |

Product-owned architecture and roadmap: **do not exist yet.**

---

## 16. Handoff expectations

Contact Portfolio Control when:

- **G0 ready**;
- **G1 ready**;
- **G2/T0 ready**;
- **G3**;
- **G4 pressure**;
- **G5**, **G6**, major **G7** evidence;
- **G8** recommendation.

Detailed cross-session handoffs are governed by [CROSS_SESSION_COORDINATION.md](../CROSS_SESSION_COORDINATION.md).
