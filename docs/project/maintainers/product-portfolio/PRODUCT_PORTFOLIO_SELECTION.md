# Product Portfolio Selection Record

**Document type:** Maintainer-level historical selection record  
**Evidence class:** A — completed selection pipeline (MP-1→MP-8)  
**Last frozen:** 2026-08-19

---

## Historical integrity rule

This record preserves the selection process as it was completed. Later evidence may **append dated follow-up notes**; it must **not** silently rewrite original reasoning or scores.

- Future product failure does **not** retroactively invalidate the fact that the original market hypothesis was reasonable at selection time.
- Future market success does **not** retroactively strengthen unsupported original claims.
- Competitive positions, buyer maturity, and category boundaries may change; this record captures the state at selection closeout.

---

## 1. Status

| Item | State |
|------|-------|
| Document level | Maintainer selection record |
| Selection phase (MP-1→MP-8) | **Complete** |
| Architecture for new applications | **Not started** |
| Cross-product reuse claim | **None** |

---

## 2. Objective

Select **3–4 independently commercially plausible products** from real market problems **before** evaluating portfolio or platform fit.

Platform capabilities, existing Intergrax code, and reuse potential were explicitly excluded from early selection criteria.

---

## 3. Anti-bias rules

Rules applied throughout MP-1→MP-8:

| Rule | Application |
|------|-------------|
| Market need first | Products must solve identifiable buyer pain before platform analysis |
| Platform fit excluded from early selection | Intergrax feature coverage was not a selection criterion in MP-1→MP-6 |
| No sunk-cost bonus | Existing repo/code received no preference |
| No platform-exercise products | No application may be selected merely because it exercises Intergrax features |
| Intergrax-independent defensibility | Each product must remain defensible if Intergrax did not exist |
| Portfolio diversity last | Diversity entered only after individual commercial screening (MP-7) |

---

## 4. Selection pipeline

### MP-1 — Market Opportunity Landscape

Broad landscape scan: approximately **22 problem hypotheses** across enterprise operations, finance, risk, engineering, and emerging AI governance.

Examples (non-exhaustive): contract-to-invoice leakage, supplier disruption response, third-party risk decisions, deployment safety, prior authorization burden, AR dispute resolution, contract obligation execution, autonomous agent governance, generic contract review, customer-service AI, IT service AI, continuous audit, incident command, contractor access.

**No Intergrax-fit scoring** at this stage.

### MP-2 — Market Evidence

Each hypothesis screened against:

- pain severity
- frequency
- economic impact
- buyer clarity
- budget existence
- current workflow quality
- AI transformation potential
- trust barrier
- time-to-value
- commercial accessibility

### MP-3 — Shortlist

Eight products advanced to competitive and commercial testing:

| # | Product |
|---|---------|
| 1 | Third-Party Risk Decision Operator |
| 2 | Contract-to-Invoice Leakage Hunter |
| 3 | Enterprise AI Governance Operator |
| 4 | Supplier Disruption Response Operator |
| 5 | Deployment / Change Guardian |
| 6 | Prior Authorization Operator |
| 7 | Contract Obligation Execution Operator |
| 8 | AR Dispute Resolution Operator |

### MP-4 — Competitive Kill Test

| Outcome | Detail |
|---------|--------|
| AR Dispute Resolution Operator | **Eliminated** — crowded category; weak differentiation path |
| Contract Obligation Execution Operator | **Reserved** — plausible but execution complexity and incumbent coverage |
| Enterprise AI Governance Operator | **Reframed** toward **Autonomous Agent Governance Operator** (emerging, narrower wedge) |
| Supply Disruption + TPRM | **Retained** with explicit competitive caveats |
| General principle | Competition treated as evidence **against** naive differentiation claims |

### MP-5 — Wow / MVP Test

Key “wow” classes identified:

| Product | Wow class |
|---------|-----------|
| Contract Recovery | Money wow — direct economic recovery |
| Supply Disruption | Crisis wow — urgent operational mitigation |
| Agent Governance | Control wow — governance over autonomous agents |
| Deployment Guardian | Engineering wow — safe change authorization |

### MP-6 — Commercial Viability

Broad tier outcome (no precise market-size or pricing claims):

| Tier | Products |
|------|----------|
| **Tier A** | Contract Recovery, Deployment Guardian, Third-Party Risk, Supply Disruption |
| **Tier B challenger** | Autonomous Agent Governance |
| **Wildcard** | Prior Authorization |

### MP-7 — Portfolio Construction

The selected four are complementary across:

- different buyers
- different business outcomes
- different data domains
- different workflow shapes
- different operational tempo
- different consequential actions

**Explicit constraint:** diversity was applied **only after** each candidate had passed individual market and commercial screening (MP-2→MP-6). Portfolio fit did not rescue weak individual candidates.

### MP-8 — Independence Audit

**Result: PASS**

| Finding | Detail |
|---------|--------|
| Tier A precedence | Selected four were already Tier A **before** portfolio/platform analysis |
| Prior Authorization not selected | Strong diversity value, but healthcare/regulatory/integration burden disproportionate to first multi-product experiment |
| Agent Governance not selected | Unusually strong Intergrax alignment, but hyperscaler/infrastructure competition and immature buyer category reduced commercial confidence |

These counterexamples are intentional anti-bias evidence: neither “best platform fit” nor “best diversity” automatically wins.

---

## 5. Final newly selected products

### Contract-to-Invoice Leakage / Recovery Operator

| Field | Detail |
|-------|--------|
| User-facing job | Find economic leakage between contracts and actual spend; support recovery |
| Primary buyer | CFO / Procurement / Finance |
| Why selected | Direct measurable ROI; relatively low-friction read-only pilot possible; strong demo; clear economic value |
| Important caveat | Spend intelligence and value-leakage detection already have competitors. **Do not claim simple leakage detection is unique.** Future product work must validate a sharper recovery-oriented wedge. |

### Supplier Disruption Response Operator

| Field | Detail |
|-------|--------|
| User-facing job | Turn an active supply disruption into a mitigation plan and ultimately controlled mitigation actions before operational impact materializes |
| Primary buyer | COO / Supply Chain / Procurement |
| Why selected | Severe operational pain; high economic consequence; strong crisis-oriented workflow; materially different from other selected products |
| Important caveat | Existing supply-risk and resilience platforms already move from detection into mitigation workflows. **Do not claim detection + recommendation alone is differentiated.** Future product session must validate a sharper execution-oriented wedge. |

### Third-Party Risk Decision Operator

| Field | Detail |
|-------|--------|
| User-facing job | Move a real vendor request from evidence gathering through review to a defensible decision |
| Primary buyer | CISO / Risk / Procurement / Compliance |
| Why selected | Established enterprise buyer/budget; fragmented cross-functional workflow; high auditability and decision value; credible bounded pilot |
| Important caveat | End-to-end TPRM orchestration is already an active category. **Do not claim orchestration alone is unique.** Future product work must narrow the initial wedge. |

### Deployment / Change Guardian

| Field | Detail |
|-------|--------|
| User-facing job | Determine whether a software change is safe and authorized to reach production; enforce the decision progressively |
| Primary buyer | CTO / VP Engineering / Platform Engineering / SRE |
| Why selected | Clear modern pain; accessible technical buyer; strong low-risk shadow-mode pilot; high-quality demonstrability; independent business case |
| Important caveat | GitHub, GitLab, Harness, cloud, and DevOps incumbents have strong distribution. Future product work must validate a vendor-neutral cross-system decision/enforcement wedge. |

---

## 6. Challenger

### Autonomous Agent Governance Operator

| Field | Detail |
|-------|--------|
| Strategic position | Strategically strong; emerging real problem; excellent demo potential |
| Not selected into first four | Hyperscalers and infrastructure vendors are aggressively entering identity/runtime-governance layers; buyer maturity is still developing |
| Status | **Challenger** — not rejected forever |

---

## 7. Future / wildcard vertical

### Prior Authorization Operator

| Field | Detail |
|-------|--------|
| Pain | Exceptionally severe real-world pain; strong economic and human value |
| Not selected for first portfolio | Credible MVP requires healthcare-specific domain, payer/EHR integration, and regulatory/safety investment disproportionate to this first multi-product experiment |
| Status | **Future wildcard** — not rejected as a business opportunity |

---

## 8. Rejected / reserved categories

Compact retention of evaluated alternatives (evidence that alternatives existed):

| Category | Disposition | Primary reason |
|----------|-------------|----------------|
| Generic contract review | Rejected | Commoditized; weak wedge |
| Generic customer-service AI | Rejected | Crowded; unclear buyer ROI |
| Generic IT service AI | Rejected | Incumbent ITSM coverage |
| AR Dispute Resolution Operator | Eliminated (MP-4) | Competitive kill — weak differentiation |
| Contract Obligation Execution Operator | Reserved | Execution complexity; incumbent coverage |
| Continuous audit | Rejected | Broad category; weak initial wedge |
| Incident commander | Rejected | Crowded incident-management space |
| Contractor access | Rejected | Identity/access incumbents |

---

## 9. Final portfolio relationship

```text
NEWLY SELECTED (MP-1→MP-8 pipeline):
  • Contract-to-Invoice Leakage / Recovery Operator
  • Supplier Disruption Response Operator
  • Third-Party Risk Decision Operator
  • Deployment / Change Guardian

EXISTING REFERENCE PRODUCT (not a selection-pipeline result):
  • Local Knowledge Workspace (LKW)
```

LKW is **not** a fifth result of the selection pipeline. It joins the later multi-product program because the control and audit session must evaluate the effects of **all active applications** on Intergrax.

---

## 10. What this record does NOT prove

| Claim | Status |
|-------|--------|
| These products are validated businesses | **Not claimed** |
| Customer validation | **Not claimed** unless independently obtained later |
| Product architecture for the four new applications | **Does not exist yet** |
| Implementation for the four newly selected applications | **Does not exist yet** |
| Cross-product reuse | **Not proven** |
| Intergrax as validated multi-product platform | **Does not follow from selection alone** |
| Competitive positions and markets | **May change** |

---

## 11. Next step

**MP-11** — define [`MULTI_PRODUCT_PROGRAM.md`](MULTI_PRODUCT_PROGRAM.md), including:

- portfolio control session responsibilities
- product-session independence
- review gates
- platform-change escalation
- asynchronous product pacing
- promotion / pause / stop recommendations
- central status and ledger ownership
