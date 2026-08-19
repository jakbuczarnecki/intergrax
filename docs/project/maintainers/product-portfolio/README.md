# Multi-Product Program — Maintainer Workspace

Maintainer control and documentation area for coordinated development and evaluation of multiple Intergrax applications.

---

## Program intent

```text
Market problems select products first.
Products are developed independently.
Their effect on the shared Intergrax platform is observed and audited second.
Never invent or distort a product merely to exercise a platform capability.
```

The multi-product program exists to develop several independently justified market applications in parallel and learn what that teaches about Intergrax as a platform. Product selection preceded platform-fit analysis.

---

## Scope

Program control covers **all actively developed products** in the program:

| Role | Product |
|------|---------|
| Existing reference product | Local Knowledge Workspace (LKW) |
| Newly selected applications | Contract-to-Invoice Leakage / Recovery Operator |
| | Supplier Disruption Response Operator |
| | Third-Party Risk Decision Operator |
| | Deployment / Change Guardian |

LKW was **not** selected by the MP-1→MP-8 market-selection pipeline. It is included because the portfolio controller must evaluate the effects of **all active Intergrax products** together.

---

## Current state

| Item | Status |
|------|--------|
| MP-1 through MP-8 (market/product selection) | **Completed** |
| Product selection record | **Frozen** — [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| Program governance contract | **Defined** — [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| Operational control artifacts | **Created** (MP-12) — see source-of-truth map below |
| Product architectures for the four new applications | **Not yet designed** |
| New product application scaffolding | **Not started** |
| Cross-product reuse demonstration | **Not demonstrated** |
| LKW baseline ingestion into portfolio control | **Pending** |

---

## Source-of-truth map

| Question | Authoritative document |
|----------|------------------------|
| Why these products were selected | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| How the program operates | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| Where everything is now | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| What product pressure did to Intergrax | [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Why portfolio/program direction changed | [DECISION_LOG.md](DECISION_LOG.md) |
| Per-product reuse methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| Product-first development principle | [PRODUCT_FIRST_MVP.md](../plans/PRODUCT_FIRST_MVP.md) |
| LKW architecture | [LKW ARCHITECTURE.md](../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) |
| LKW implementation roadmap | [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |

Do not duplicate LKW architecture or roadmap content here.

---

## Ownership

| Artifact class | Owner |
|----------------|-------|
| Operational control artifacts (`PORTFOLIO_STATUS`, `PLATFORM_IMPACT_LEDGER`, `DECISION_LOG`) | Portfolio Control Session |
| Product evidence and implementation | Respective Product Session |

Product Sessions report evidence; they do not self-certify central status, declare accepted platform impact, or rewrite portfolio decisions. Portfolio Control verifies repo evidence, updates central artifacts, and preserves append-oriented history.

Where exact implementation evidence exists, completion summaries alone are insufficient.

---

## Planned workspace shape

Later tasks will add, without creating them now:

- per-product control cards (`products/*`)
- checkpoint reviews (`reviews/*`)
