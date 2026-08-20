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

LKW is Product 0 / the existing reference product, baseline-ingested, supervised by Portfolio Control, and a **future independently operated Product Session**. Its historical reference baseline is different from the preregistered T0 used by the four newly selected products.

---

## Session topology

**Six parallel working sessions** at full operating launch:

1. Portfolio Control Session
2. LKW Product Session
3. Contract Recovery Product Session
4. Supplier Disruption Product Session
5. Third-Party Risk Product Session
6. Deployment Guardian Product Session

**Parallel specialist streams** (not part of the six): **VIS-3A** (public visual/documentation presentation) and **COMM** (LKW proof development within its authorized roadmap). Neither is Portfolio Control.

**Public product set:** five products (LKW plus the four newly selected). Portfolio Control is not a public product.

Truth flows from accepted product/proof evidence → Portfolio Control → approved public facts → VIS-3A presentation. Public documentation is never implementation truth.

---

## Current state

| Item | Status |
|------|--------|
| MP-1 through MP-8 (market/product selection) | **Completed** |
| Product selection record | **Frozen** — [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| Program governance contract | **Defined** — [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| Operational control artifacts | **Created** (MP-12) — see source-of-truth map below |
| Product bootstrap contract | **Defined** (MP-14) — [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) |
| Product control cards | **Created** (MP-15) — see control-card table below |
| Audit engine integration | **Defined** (MP-16) — [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) |
| Portfolio Control operating manual | **Defined** (MP-17) — [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md) |
| Product Session operating manual | **Defined** (MP-18) — [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md) |
| Product-specific session briefs | **Created** (MP-19) — see [Product Session briefs](#product-session-briefs) |
| Cross-session coordination contract | **Defined** (MP-20) — [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md) |
| Product architectures for the four new applications | **Not yet designed** |
| New product application scaffolding | **Not started** |
| Cross-product reuse demonstration | **Not demonstrated** |
| LKW baseline ingestion into portfolio control | **Completed** (MP-13) — [LKW control card](products/LKW.md) |

---

## Source-of-truth map

| Question | Authoritative document |
|----------|------------------------|
| Why these products were selected | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| How the program operates | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| How must a new product start? | [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) |
| Where everything is now | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| What product pressure did to Intergrax | [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Why portfolio/program direction changed | [DECISION_LOG.md](DECISION_LOG.md) |
| Per-product reuse methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| How Portfolio Control gates use the audit engine | [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) |
| How does the central Portfolio Control Session operate? | [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md) |
| How does each Product Session operate? | [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md) |
| How do the sessions coordinate? | [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md) |
| Product-first development principle | [PRODUCT_FIRST_MVP.md](../plans/PRODUCT_FIRST_MVP.md) |
| LKW Portfolio Control Card | [products/LKW.md](products/LKW.md) |
| LKW architecture | [LKW ARCHITECTURE.md](../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) |
| LKW implementation roadmap | [LKW IMPLEMENTATION_PLAN.md](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |

Product control cards are concise Portfolio Control indexes. They do not replace product-owned architecture or roadmap. LKW is baseline-ingested as the reference product.

Newly selected products start under [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md). They are not yet in implementation.

Do not duplicate LKW architecture or roadmap content here. Do not duplicate bootstrap rules here.

---

## Ownership

| Artifact class | Owner |
|----------------|-------|
| Operational control artifacts (`PORTFOLIO_STATUS`, `PLATFORM_IMPACT_LEDGER`, `DECISION_LOG`) | Portfolio Control Session |
| Product evidence and implementation | Respective Product Session |

Product Sessions report evidence; they do not self-certify central status, declare accepted platform impact, or rewrite portfolio decisions. Portfolio Control verifies repo evidence, updates central artifacts, and preserves append-oriented history.

Where exact implementation evidence exists, completion summaries alone are insufficient.

---

## Product control cards

Portfolio Control owns all control cards. Product Sessions supply accepted product truth; VIS-3A consumes approved public facts downstream.

| Product | Control card | Baseline status |
|---------|--------------|-----------------|
| Local Knowledge Workspace (LKW) | [products/LKW.md](products/LKW.md) | Reference baseline ingested (MP-13) |
| Contract-to-Invoice Leakage / Recovery Operator | [products/contract-recovery.md](products/contract-recovery.md) | Pre-bootstrap (MP-15) |
| Supplier Disruption Response Operator | [products/supplier-disruption.md](products/supplier-disruption.md) | Pre-bootstrap (MP-15) |
| Third-Party Risk Decision Operator | [products/third-party-risk.md](products/third-party-risk.md) | Pre-bootstrap (MP-15) |
| Deployment / Change Guardian | [products/deployment-guardian.md](products/deployment-guardian.md) | Pre-bootstrap (MP-15) |

Control cards index portfolio state; architecture and roadmap remain product-owned.

## Product Session briefs

Durable product-specific mission artifacts for future Product Session launch. **Not launched sessions** — consumed later by MP-22 Session Launch Pack.

| Layer | Document | Role |
|-------|----------|------|
| Common operating behavior | [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md) | How every Product Session operates |
| Product-specific mission | `session-briefs/<PRODUCT>.md` | What this product is trying to prove/build |
| Future session bootstrap | MP-22 Session Launch Pack | Assembled prompt from common rules + brief + current repo state |

| Product | Session brief | Program state |
|---------|---------------|---------------|
| Local Knowledge Workspace (LKW) | [session-briefs/LKW.md](session-briefs/LKW.md) | ACTIVE reference product |
| Contract Recovery | [session-briefs/CONTRACT_RECOVERY.md](session-briefs/CONTRACT_RECOVERY.md) | SELECTED / Pre-bootstrap |
| Supplier Disruption | [session-briefs/SUPPLIER_DISRUPTION.md](session-briefs/SUPPLIER_DISRUPTION.md) | SELECTED / Pre-bootstrap |
| Third-Party Risk | [session-briefs/THIRD_PARTY_RISK.md](session-briefs/THIRD_PARTY_RISK.md) | SELECTED / Pre-bootstrap |
| Deployment Guardian | [session-briefs/DEPLOYMENT_GUARDIAN.md](session-briefs/DEPLOYMENT_GUARDIAN.md) | SELECTED / Pre-bootstrap |

Do not duplicate brief content in this index.

## Document relationships

| Document | Role |
|----------|------|
| [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) | Constitution |
| [PRODUCT_BOOTSTRAP_RULES.md](PRODUCT_BOOTSTRAP_RULES.md) | New-product bootstrap |
| [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md) | Central control behavior |
| [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md) | Common product behavior |
| [session-briefs/](session-briefs/) | Product-specific mission |
| [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md) | Handoffs and authority between sessions and specialist streams |
| [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md) | Canonical audit integration |

**MP-22** Session Launch Pack — future six launch prompts (Portfolio Control + five Product Sessions). VIS-3A and COMM remain external specialist streams unless a future task creates separate prompts.

## Planned workspace shape

Later program tasks (see [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) for current next step):

- ~~MP-16 Multi-Product Audit Integration~~ — **Completed** — [MULTI_PRODUCT_AUDIT_INTEGRATION.md](MULTI_PRODUCT_AUDIT_INTEGRATION.md)
- ~~MP-17 Portfolio Control Operating Manual~~ — **Completed** — [PORTFOLIO_CONTROL_OPERATING_MANUAL.md](PORTFOLIO_CONTROL_OPERATING_MANUAL.md)
- ~~MP-18 Product Session Operating Manual~~ — **Completed** — [PRODUCT_SESSION_OPERATING_MANUAL.md](PRODUCT_SESSION_OPERATING_MANUAL.md)
- ~~MP-19 Product-Specific Session Briefs~~ — **Completed** — [session-briefs/](session-briefs/)
- ~~MP-20 Cross-Session Coordination Rules~~ — **Completed** — [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md)
- MP-21 Workspace Consistency Audit
- MP-22 Session Launch Pack

Portfolio Control uses the canonical `docs/audit_results/` engine for actual audits. It does not create a competing audit workspace (no `reviews/*` audit system).
