# MP-22 — Session Launch Pack

**Status:** READY TO LAUNCH

**Created from:** `901afb141f1b27140f74363b91eb7034f0cea4f4` (`development`)

---

## Purpose

This pack contains **six ready-to-paste launch prompts** for the Intergrax multi-product program. Each prompt bootstraps one independent working session into the durable operating system created by MP-10→MP-21. The prompts do **not** recreate that operating system — they instruct the session to load canonical repo contracts and operate from current evidence.

**Recommended use:** Open six independent ChatGPT/Cursor-assisted sessions. Paste **exactly one** corresponding prompt into each. Do **not** paste multiple launch prompts into one Product Session.

**Runtime truth:** Prompts dynamically re-read the current repository. The pack assembly SHA above is **historical launch context only** — each session must resolve current `development` HEAD before acting.

**Outside this pack:** VIS-3A (public visual/documentation presentation) and COMM (LKW proof development) remain external specialist streams governed by [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md).

---

## Six sessions

| # | Session | Prompt file | Initial role |
|---|---------|-------------|--------------|
| 1 | Portfolio Control Session | [session-launch/PORTFOLIO_CONTROL.md](session-launch/PORTFOLIO_CONTROL.md) | Central authority — gate acceptance, G4, cross-product impact, status/cards/ledger |
| 2 | LKW Product Session | [session-launch/LKW.md](session-launch/LKW.md) | Continue ACTIVE reference product development |
| 3 | Contract Recovery Product Session | [session-launch/CONTRACT_RECOVERY.md](session-launch/CONTRACT_RECOVERY.md) | New product — economic leakage and recovery |
| 4 | Supplier Disruption Product Session | [session-launch/SUPPLIER_DISRUPTION.md](session-launch/SUPPLIER_DISRUPTION.md) | New product — disruption → mitigation → controlled action |
| 5 | Third-Party Risk Product Session | [session-launch/THIRD_PARTY_RISK.md](session-launch/THIRD_PARTY_RISK.md) | New product — vendor request → evidence → defensible decision |
| 6 | Deployment Guardian Product Session | [session-launch/DEPLOYMENT_GUARDIAN.md](session-launch/DEPLOYMENT_GUARDIAN.md) | New product — change → independent evidence → GO/NO-GO |

**Public product set:** five products (LKW + four newly selected). Portfolio Control is **not** a public product.

---

## Recommended startup order

1. **Portfolio Control** — central authority should be available before new products reach G0/G4 handoffs
2. **LKW** — existing ACTIVE reference product
3. **Four new Product Sessions** — in any order

Sessions are **prepared, not automatically launched**. Opening a prompt file does not start a session until the operator pastes it into a new conversation.

---

## Historical launch context (verify from repo at session start)

| Product | Expected state at pack creation | Notes |
|---------|--------------------------------|-------|
| LKW | **ACTIVE** — existing reference product | Current task per IMPLEMENTATION_PLAN; verify at launch |
| Contract Recovery | **SELECTED** / Pre-bootstrap / G0 pending | No architecture, scaffold, or implementation |
| Supplier Disruption | **SELECTED** / Pre-bootstrap / G0 pending | No architecture, scaffold, or implementation |
| Third-Party Risk | **SELECTED** / Pre-bootstrap / G0 pending | Initial wedge requires sharpening |
| Deployment Guardian | **SELECTED** / Pre-bootstrap / G0 pending | No architecture, scaffold, or implementation |

Each session **must verify** current state from the repository at launch. Do not treat this table as permanent runtime truth.

---

## Validation matrix

| Criterion | Portfolio Control | LKW | Contract Recovery | Supplier Disruption | Third-Party Risk | Deployment Guardian |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Role correct | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Canonical read set defined | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Current-state self-verification | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| First action explicit | sync/report | sync/roadmap | verify G0 | verify G0 | verify G0 | verify G0 |
| Authority boundary | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| G4 behavior | owns G4 | escalate G4 | escalate G4 | escalate G4 | escalate G4 | escalate G4 |
| Audit/T1 behavior | owns gate audit use | report only | per bootstrap | per bootstrap | per bootstrap | per bootstrap |
| Concurrency rules | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Collaboration workflow | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| No automatic cross-session communication | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| No stale permanent SHA assumption | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Public truth boundary | owns eligibility | no public claims | no public claims | no public claims | no public claims | no public claims |

---

## Program completion

After MP-22, central preparation (MP-10→MP-22) is **complete**. Further work occurs in Portfolio Control Session, respective Product Sessions, and existing specialist streams (VIS-3A, COMM) unless future evidence requires a new explicit program task.

---

## Related documents

| Question | Document |
|----------|----------|
| Maintainer workspace index | [README.md](README.md) |
| Live portfolio state | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| Program constitution | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
| Cross-session coordination | [CROSS_SESSION_COORDINATION.md](CROSS_SESSION_COORDINATION.md) |
