# Deployment / Change Guardian - Portfolio Control Card

**Document type:** Per-product Portfolio Control index  
**Owner:** Portfolio Control Session  
**Card created:** MP-15 (2026-08-20)

---

## 1. Identity

| Field | Value |
|-------|-------|
| Product | Deployment / Change Guardian |
| Short name | Deployment Guardian |
| Program role | Newly selected application |
| Program State | **SELECTED** |
| Product Stage | Pre-bootstrap |
| Baseline type | **None** - bootstrap not started |
| Portfolio recommendation | **CONTINUE** |
| Relative priority | **HIGH** |
| Future session | Deployment Guardian Product Session (not yet launched) |

This product was admitted by the MP-1→MP-8 selection pipeline. It is not a reference baseline and has no retroactive T0.

---

## 2. Why selected

From [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5:

- Clear modern pain.
- Accessible technical buyer.
- Strong low-risk shadow-mode pilot.
- High-quality demonstrability.
- Independent business case.

---

## 3. Current product hypothesis

Determine whether a software change is safe and authorized to reach production and progressively enforce that decision.

**Pre-G0 hypothesis - subject to G0 validation/refinement.**

---

## 4. Buyer / user context

| Field | Value |
|-------|-------|
| Primary buyer | CTO / VP Engineering / Platform Engineering / SRE |
| User-facing job (selection record) | Determine whether a software change is safe and authorized to reach production; enforce the decision progressively |

---

## 5. Selection caveat / differentiation risk

GitHub, GitLab, Harness, cloud, and DevOps incumbents have strong distribution. Future product work must validate a vendor-neutral cross-system decision/enforcement wedge.

---

## 6. Bootstrap state

| Item | Status |
|------|--------|
| G0 Product Baseline | **PENDING** |
| Initial wedge freeze | **PENDING** |
| G1 Product Architecture | **NOT STARTED** |
| Platform Capability Audit | **NOT STARTED** |
| G2 T0 Reuse Baseline | **NOT STARTED** / **NOT CREATED** |
| Application path | **NOT CREATED** |
| Application scaffold | **NOT CREATED** |
| G3 First Real Vertical Slice | **NOT STARTED** |
| Current accepted platform pressure | **NONE** |
| Cross-product reuse evidence | **NONE** |
| Market validation | Selection evidence only |
| Customer validation | **NOT CLAIMED** |
| Commercial validation | **NOT CLAIMED** |

Next portfolio gate: **G0 - product baseline** per [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md).

---

## 7. Authoritative sources

This card indexes Portfolio Control state. It does **not** override product-owned sources once they exist.

| Topic | Authoritative document |
|-------|------------------------|
| Selection record | [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5 - Deployment / Change Guardian |
| Bootstrap contract | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
| Live portfolio state | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |

**Product-owned architecture and roadmap:** do not exist yet.

---

## 8. Public Presentation Surface

### Public Presentation Document

| Field | Value |
|-------|-------|
| Status | **NOT YET REGISTERED IN PORTFOLIO CONTROL** |
| Owner | Separate VIS/public-documentation stream |
| Template authority | VIS-3A / separate visual documentation stream |
| Future link | **PENDING** |

### Root README product surface

| Field | Value |
|-------|-------|
| Status | **PARALLEL DOCUMENTATION WORK** |
| Owner | VIS-3A / public documentation stream |

**Rules:**

- This Control Card does not define visual layout, template, or hero.
- VIS-3A owns **how** product truth is presented.
- VIS-3A does **not** own implementation truth or gate status.
- Pre-G0 public claims may use only frozen selection facts and explicit pre-bootstrap status.
- Do not guess presentation paths.
- Public docs are never implementation source of truth.

---

## 9. Current evidence / non-claims

**Current evidence:** Selection record only - [PRODUCT_PORTFOLIO_SELECTION.md](../PRODUCT_PORTFOLIO_SELECTION.md) §5.

This card does **not** prove or claim:

- product architecture;
- implementation;
- customer or commercial validation beyond selection screening;
- cross-product reuse;
- platform-impact classification;
- T0 or T1 reuse evidence;
- readiness for public product presentation beyond selection facts already recorded in the selection document.

---

## 10. Portfolio Control Questions

Portfolio Control must revisit these at material gates. **Do not answer here.**

- Is vendor neutrality real across multiple systems?
- Does decision use independent cross-system evidence?
- Can enforcement work without replacing CI/CD?
- Is value greater than native GitHub/GitLab/Harness/cloud controls?
- Can shadow mode show value before consequential control?

---

## 11. Future authoritative product artifacts

Product Session will own detailed product truth. Portfolio Control will verify and index it.

| Artifact | Status |
|----------|--------|
| Accepted G0 Product Baseline | Not yet created |
| Accepted Product Architecture | Not yet created |
| Product roadmap / implementation plan | Not yet created |
| Platform Capability Audit | Not yet created |
| Accepted T0 reuse baseline | Not yet created |
| G3 vertical-slice evidence | Not yet created |
| Later proofs / validation evidence | Not yet created |
| Public Product Presentation Document (once canonical path exists) | Not yet registered |

---

## 12. Next gate

**G0 - Product Baseline** - pending acceptance per [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md).

Product remains **SELECTED** / Pre-bootstrap until G0 is accepted. Material shared-platform pressure must pass G4 when it arises.

---

## Related portfolio documents

| Question | Document |
|----------|----------|
| Live portfolio dashboard | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Program governance | [MULTI_PRODUCT_PROGRAM.md](../MULTI_PRODUCT_PROGRAM.md) |
| Bootstrap rules | [PRODUCT_BOOTSTRAP_RULES.md](../PRODUCT_BOOTSTRAP_RULES.md) |
