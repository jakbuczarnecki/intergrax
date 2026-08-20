# Contract-to-Invoice Leakage / Recovery Operator — Portfolio Control Card

**Document type:** Per-product Portfolio Control index  
**Owner:** Portfolio Control Session  
**Card created:** MP-15 (2026-08-20)

---

## 1. Identity

| Field | Value |
|-------|-------|
| Product | Contract-to-Invoice Leakage / Recovery Operator |
| Short name | Contract Recovery |
| Program role | Newly selected application |
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| Baseline type | **None** — bootstrap not started |
| Portfolio recommendation | **CONTINUE** |
| Relative priority | **HIGH** |
| Future session | Contract Recovery Product Session (not yet launched) |

This product was admitted by the MP-1→MP-8 selection pipeline. It is not a reference baseline and has no retroactive T0.

---

## 2. Authoritative sources

This card indexes Portfolio Control state. It does **not** override product-owned sources once they exist.

| Topic | Authoritative document |
|-------|------------------------|
| Selection record | [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5 — Contract-to-Invoice Leakage / Recovery Operator |
| Bootstrap contract | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
| Live portfolio state | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |

**Product-owned architecture and roadmap:** do not exist yet.

---

## 3. Bootstrap position

| Gate | Status |
|------|--------|
| G0 — Product baseline | **Pending** |
| G1 — Product architecture | Not started |
| G2 — T0 reuse baseline | Not started |
| Application scaffold | Not started |

Next portfolio gate: **G0 — product baseline** per [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md).

---

## 4. Selection context (index only)

From [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5:

- **User-facing job:** Find economic leakage between contracts and actual spend; support recovery.
- **Primary buyer:** CFO / Procurement / Finance.
- **Selection rationale:** Direct measurable ROI; relatively low-friction read-only pilot possible; strong demo; clear economic value.
- **Retained caveat:** Spend intelligence and value-leakage detection already have competitors. Simple leakage detection is **not** claimed as unique. Future product work must validate a sharper recovery-oriented wedge.

Do not expand this index into product architecture here.

---

## 5. Coordination interface

Portfolio Control owns this card.

When the Contract Recovery Product Session launches, coordination will follow:

```text
Product Session (accepted product truth)
        ↓
Portfolio Control (verification / central index)
        ↓
approved public facts only
        ↓
VIS-3A public presentation (downstream; not authoritative)
```

This card does **not** assign ownership to VIS-3A. Detailed cross-session handoffs are reserved for MP-20.

---

## 6. Explicit non-claims

This card does **not** prove or claim:

- product architecture;
- implementation;
- customer or commercial validation;
- cross-product reuse;
- platform-impact classification;
- T0 or T1 reuse evidence;
- readiness for public product presentation beyond selection facts already recorded in the selection document.

---

## 7. Next portfolio state

- Remains **SELECTED** / Pre-bootstrap until G0 is accepted.
- Future Contract Recovery Product Session will supply product-owned truth; Portfolio Control will verify and index it here.
- Material shared-platform pressure must pass G4 when it arises.

---

## Related portfolio documents

| Question | Document |
|----------|----------|
| Live portfolio dashboard | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Program governance | [MULTI_PRODUCT_PROGRAM.md](../MULTI_PRODUCT_PROGRAM.md) |
| Bootstrap rules | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
