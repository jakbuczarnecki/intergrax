# Contract Recovery — Product Session Brief

**Document type:** Durable product-specific session mission artifact  
**Owner:** Contract Recovery Product Session (future launch)  
**Audience:** Future session operator / MP-22 Session Launch Pack assembler  
**Status:** **SELECTED** — Pre-bootstrap; G0 **PENDING**

> **This is NOT the final session launch prompt.**  
> It is a durable product-specific mission and context artifact consumed later by **MP-22 Session Launch Pack**.  
> Do not treat this file as a conversational bootstrap prompt. Common operating behavior lives in [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md).

---

## 1. Session identity

| Field | Value |
|-------|-------|
| Product Session | Contract Recovery Product Session |
| Product | Contract-to-Invoice Leakage / Recovery Operator |
| Short name | Contract Recovery |
| Program role | Newly selected application |
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| Control card | [products/contract-recovery.md](../products/contract-recovery.md) |

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

Find **economic leakage** between contracts and actual spend and support **recovery** — turning defensible discrepancies into economically actionable recovery cases, not merely detecting or reporting them.

**Product-first rule:**

```text
The product is not being built to demonstrate Intergrax.
Intergrax reuse is observed as a consequence of building the product.
```

---

## 3. Why this product exists independently of Intergrax

Enterprises lose money when contracted rates, terms, and entitlements diverge from invoiced and paid reality. Finance and procurement teams need a path from contract truth and spend truth to **recoverable value** — not another passive analytics dashboard.

This problem exists whether or not Intergrax is the platform. Selection recorded direct ROI, read-only pilot feasibility, and clear economic value as admission reasons.

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
| Product roadmap | **Does not exist** |
| Evidence beyond selection | **None** |
| Market / customer / commercial validation | **NOT CLAIMED** |

Authoritative index: [contract-recovery control card](../products/contract-recovery.md), [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5.

---

## 5. Product hypothesis / current product truth

**Pre-G0 hypothesis (subject to G0 validation):**

```text
Find economic leakage between contracts and actual spend and support recovery.
```

Simple leakage **detection** is not claimed as unique. Spend intelligence competitors exist. The wedge must be **recovery-oriented**, not reporting-oriented.

---

## 6. Buyer / user and economic or operational job

| Dimension | Value |
|-----------|-------|
| Primary buyer | CFO / Procurement / Finance |
| Core job | Turn contract/spend discrepancies into economically actionable recovery cases |
| Economic consequence | Leakage → unrealized or unrecovered money |
| Success horizon | Financial cycle / audit-driven — not real-time crisis |
| Value unit | Defensible discrepancy → quantified recovery opportunity → human/business action |

---

## 7. Primary workflow

Target workflow shape for G0 sharpening (not architecture):

```text
Contract terms + invoice/spend inputs
  → discrepancy detection with defensible evidence
  → recovery opportunity quantification
  → review / assignment / recovery action trace
```

G0 must sharpen: **problem → discrepancy → evidence → recovery opportunity → human/business action.**

Do not design architecture in this brief.

---

## 8. What makes this product different from LKW / other products

| Contrast | Contract Recovery |
|----------|-------------------|
| LKW | Knowledge Q&A and provenance — not financial reconciliation |
| Supplier Disruption | Operational crisis mitigation — not contract/spend leakage |
| Third-Party Risk | Vendor onboarding decision — not post-contract spend recovery |
| Deployment Guardian | Release authorization — not economic recovery |

Time horizon is **financial/audit**, not seconds-to-mitigate crisis or deployment gate tempo.

---

## 9. Product-specific wedge / kill questions

- Is **recovery** the actual wedge, or merely detection/reporting?
- Can value connect to **money recoverable or recovered**, not just "findings"?
- Can the first pilot operate **safely read-only**?
- Is **buyer ownership** clear (CFO vs procurement vs AP)?
- Does the product escape **generic CLM / spend analytics**?
- What evidence makes a discrepancy **defensible and actionable**?

---

## 10. Major failure modes / category traps

- **Invoice anomaly dashboard** — charts without recovery workflow.
- **Generic document extraction** — OCR/LLM summaries without economic linkage.
- **CLM clone** — repository of contracts without spend reconciliation path.
- **"AI finds discrepancies"** without recovery case management.
- **Architecture theater around RAG** because LKW uses RAG — finance problem first.
- **Detection-only product** marketed as recovery.

---

## 11. Platform posture

**Before G1:** Do not deeply shape the product around current Intergrax APIs.

**After product architecture:** Perform Platform Capability Audit.

**Before implementation:** Accepted G2/T0 required.

**During implementation:** Material shared platform change needed → **STOP** → **G4**.

Product Session cannot self-approve `EXTENDED_GENERALLY`, `GENUINE_PLATFORM_GAP`, or shared core product-specific behavior.

VIS-3A owns public presentation of approved facts — not implementation truth or gate status. COMM does not own Portfolio Control authority.

---

## 12. Current gate / first allowed action

**G0 Product Baseline** — preparation and acceptance per [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md).

G0 should sharpen the recovery chain before any architecture work. Do **not** start G1, T0, scaffold, or implementation until G0 is accepted.

---

## 13. Evidence the session must eventually produce

Future evidence target (not claimed today):

```text
Real contract + invoice/spend input
  → defensible discrepancy
  → quantified recovery opportunity
  → review / action trace
```

Also eventually: G3 vertical slice, reuse evidence per [PRODUCT_REUSE_PROOF.md](../../plans/PRODUCT_REUSE_PROOF.md) when T0 exists.

---

## 14. What the session must NOT claim yet

- Product architecture or implementation.
- Customer, commercial, or market validation beyond frozen selection facts.
- Cross-product reuse or platform-impact classification.
- T0/T1 reuse evidence.
- Unique wedge without G0 sharpening.
- Public marketing copy or README hero (VIS-3A).

---

## 15. Sources of truth

| Topic | Document |
|-------|----------|
| Common session behavior | [PRODUCT_SESSION_OPERATING_MANUAL.md](../PRODUCT_SESSION_OPERATING_MANUAL.md) |
| Bootstrap contract | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
| Program governance | [MULTI_PRODUCT_PROGRAM.md](../MULTI_PRODUCT_PROGRAM.md) |
| Selection record | [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5 |
| Live portfolio state | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Control card | [products/contract-recovery.md](../products/contract-recovery.md) |
| Reuse methodology | [PRODUCT_REUSE_PROOF.md](../../plans/PRODUCT_REUSE_PROOF.md) |

Product-owned architecture and roadmap: **do not exist yet.**

---

## 16. Handoff expectations

Contact Portfolio Control when:

- **G0 ready** for acceptance;
- **G1 ready** (architecture);
- **G2/T0 ready**;
- **G3** vertical slice evidence;
- **G4 pressure** from material platform need;
- **G5**, **G6**, major **G7** evidence;
- **G8** recommendation input.

Detailed coordination rules: **MP-20** (future).
