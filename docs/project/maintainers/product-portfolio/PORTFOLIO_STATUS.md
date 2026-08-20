# Portfolio Status — Live Current-State Dashboard

**Document type:** Maintainer-level operational control artifact  
**Owner:** Portfolio Control Session  
**Last verified:** 2026-08-20 (MP-13 LKW reference baseline ingestion)

---

## Purpose

This document answers: **Where are all active/selected products right now?**

It is the live, compact current-state dashboard for the entire multi-product program.

**This is NOT:**

- a product roadmap
- a backlog
- a duplicate of product architecture
- a historical log
- a platform impact log

---

## Ownership and verification rule

**Portfolio Control owns this document.**

| Role | May do | May NOT do |
|------|--------|------------|
| Product Session | Report progress and evidence to Portfolio Control | Self-certify central portfolio status |
| Portfolio Control | Independently verify repo evidence; update this dashboard | Accept completion summaries without authoritative evidence where implementation evidence exists |

A Product Session may report progress, but **central portfolio status changes only after Portfolio Control verifies the authoritative repo evidence.**

---

## Controlled vocabulary

### Program State

| State | Meaning |
|-------|---------|
| **SELECTED** | Admitted to the program; baseline/bootstrap not yet accepted |
| **ACTIVE** | Baseline accepted; product development under program control |
| **PAUSED** | Development intentionally suspended; status preserved |
| **STOPPED** | Product removed from active program pursuit |
| **COMPLETED** | Product reached its defined program completion target |

### Portfolio Recommendation

Independent of Program State. Expresses Portfolio Control's current guidance.

| Recommendation | Meaning |
|----------------|---------|
| **ACCELERATE** | Increase attention or pacing relative to peers |
| **CONTINUE** | Maintain current course |
| **REDUCE** | Lower investment or scope emphasis |
| **PAUSE** | Recommend suspension pending evidence or decision |
| **STOP** | Recommend program exit |

### Relative Priority

| Priority | Meaning |
|----------|---------|
| **HIGH** | Among the highest portfolio attention items |
| **MEDIUM** | Standard portfolio attention |
| **LOW** | Lower relative attention within the active portfolio |

Recommendation and priority are **separate dimensions**. Do not mix them.

---

## Active portfolio status

| Product | Role | Program State | Product Stage | Current Milestone | Recommendation | Priority | Platform Evidence State | Last Verified Evidence | Next Portfolio Gate |
|---------|------|---------------|---------------|-------------------|----------------|----------|-------------------------|------------------------|---------------------|
| Local Knowledge Workspace (LKW) | Existing reference product | **ACTIVE** | Advanced existing product | See [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) — current task: LKW-PLUGIN-CAPABILITY-CONFIGURATION-1 (READY_FOR_REVIEW); next: LKW-INDEXED-SOURCE-LIFECYCLE-1 | **CONTINUE** | **HIGH** | Reference baseline accepted at `821eb7f6b2096de142822a29abc4546ee387a158` — [LKW control card](products/LKW.md) | [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md); [LKW control card](products/LKW.md) | Next material product gate / G4 as triggered |
| Contract-to-Invoice Leakage / Recovery Operator | Newly selected | **SELECTED** | Pre-bootstrap | G0 / product baseline pending | **CONTINUE** | **HIGH** | Reuse evidence: not started | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §5; [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §18 | G0 — product baseline |
| Supplier Disruption Response Operator | Newly selected | **SELECTED** | Pre-bootstrap | G0 / product baseline pending | **CONTINUE** | **MEDIUM** | Reuse evidence: not started | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §5; [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §18 | G0 — product baseline |
| Third-Party Risk Decision Operator | Newly selected | **SELECTED** | Pre-bootstrap | Initial wedge still requires sharpening; G0 pending | **CONTINUE** | **MEDIUM** | Reuse evidence: not started | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §5; [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §18 | G0 — product baseline |
| Deployment / Change Guardian | Newly selected | **SELECTED** | Pre-bootstrap | G0 / product baseline pending | **CONTINUE** | **HIGH** | Reuse evidence: not started | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §5; [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §18 | G0 — product baseline |

**Architecture for all four newly selected products:** not started. No new product application folders exist.

**LKW market/commercial status:** Existing implemented reference product under active development; no portfolio-level commercial validation claim recorded.

---

## Outside active portfolio table

These positions are tracked for context but are **not** active program products.

| Position | Product | Notes |
|----------|---------|-------|
| Challenger | Autonomous Agent Governance Operator | Not active unless formally promoted — see [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §6 |
| Future / wildcard | Prior Authorization Operator | Not part of first-wave portfolio — see [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §7 |

---

## Current portfolio risks

Evidence-supported at MP-13 closeout:

- Third-Party Risk initial wedge still requires sharpening (selection caveat retained).
- Supply Disruption commercial/GTM/integration complexity remains material (selection caveat retained).
- No cross-product reuse has yet been proven.
- New product architectures do not yet exist.
- LKW reference baseline ingested; no retrospective accepted `PI-*` records from consumption alone.

---

## Current portfolio actions

At MP-13 closeout:

1. ~~Perform LKW baseline ingestion (G0-equivalent for reference product).~~ **Completed** — [LKW control card](products/LKW.md) at baseline `821eb7f6b2096de142822a29abc4546ee387a158`.
2. **Next:** MP-14 Product Bootstrap Rules for newly selected products.
3. Prepare product-session launch — **future**; not started.

Do not treat future actions as completed until Portfolio Control verifies evidence and updates this document.

---

## Related documents

| Question | Document |
|----------|----------|
| Why products were selected | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| How the program operates | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| What product pressure did to Intergrax | [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Why portfolio direction changed | [DECISION_LOG.md](DECISION_LOG.md) |
