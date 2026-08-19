# Portfolio Decision Log

**Document type:** Maintainer-level operational control artifact  
**Owner:** Portfolio Control Session  
**Last updated:** 2026-08-19 (MP-12 initial creation)

---

## Purpose

This document answers: **Why did we change portfolio direction, priority, product status, or a material program-level rule?**

It is an append-oriented program/portfolio decision history.

**This is NOT:**

- a code-change log
- a backlog
- a product roadmap
- a duplicate of [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md)
- a duplicate of [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md)

A later PAUSE or STOP does **not** erase an earlier ACCELERATE decision. History must remain inspectable.

---

## Ownership

| Role | May do | May NOT do |
|------|--------|------------|
| Product Session | Propose evidence and request portfolio decisions | Directly rewrite portfolio decision history |
| Portfolio Control | Record accepted program decisions; append status changes | Rewrite or delete prior decision records |

---

## Stable ID format

| Rule | Detail |
|------|--------|
| Format | `PD-001`, `PD-002`, … |
| Immutability | IDs are never reused |
| Orientation | Append-only; do not rewrite historical decisions |

Later decisions may reverse or supersede earlier ones by reference — not by erasure.

---

## Decision status (controlled vocabulary)

| Status | Meaning |
|--------|---------|
| **ACTIVE** | Currently governing program behavior |
| **SUPERSEDED** | Replaced by a later decision; retained for audit |
| **REVERSED** | Explicitly overturned; retained for audit |
| **CLOSED** | No longer operative; context fully resolved |

---

## Minimum decision schema

Every future `PD-*` entry should include:

| Field | Content |
|-------|---------|
| **ID** | Stable `PD-NNN` identifier |
| **Date** | Decision date |
| **Decision** | What was decided |
| **Trigger / evidence** | What prompted the decision |
| **Reason** | Why the decision was made |
| **Affected products** | Which products the decision touches |
| **Program effect** | What changes in program operation |
| **Status** | ACTIVE / SUPERSEDED / REVERSED / CLOSED |
| **Supersedes / Superseded By** | When applicable |

---

## Initial decision records

### PD-001 — Adopt multi-product program

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Adopt a multi-product Intergrax development/control program. |
| **Trigger / evidence** | MP-1→MP-8 selection pipeline complete; MP-11 governance contract defined |
| **Reason** | Develop independent real products in parallel and observe platform behavior under real product pressure. |
| **Affected products** | All program products |
| **Program effect** | Establishes coordinated multi-product development under Portfolio Control |
| **Status** | **ACTIVE** |

Evidence: [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md); [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md)

---

### PD-002 — Include LKW as reference product

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Include Local Knowledge Workspace (LKW) in the multi-product program as the existing reference product. |
| **Trigger / evidence** | Portfolio must evaluate all active Intergrax applications together |
| **Reason** | LKW provides live product pressure baseline; Portfolio Control requires visibility across all active applications. |
| **Affected products** | LKW |
| **Program effect** | LKW joins program as reference product — **not** as a fifth result of MP-1→MP-8 market selection |
| **Status** | **ACTIVE** |

---

### PD-003 — Admit four newly selected products

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Admit to the program: Contract-to-Invoice Leakage / Recovery Operator; Supplier Disruption Response Operator; Third-Party Risk Decision Operator; Deployment / Change Guardian. |
| **Trigger / evidence** | MP-1→MP-8 selection closeout |
| **Reason** | Four independently commercially plausible products passed market, competitive, and portfolio screening. |
| **Affected products** | Contract Recovery; Supply Disruption; Third-Party Risk; Deployment Guardian |
| **Program effect** | Four new products enter program at SELECTED / pre-bootstrap state |
| **Status** | **ACTIVE** |

Full selection rationale: [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) §5

---

### PD-004 — Agent Governance as challenger

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Keep Autonomous Agent Governance Operator as challenger rather than an active first-wave product. |
| **Trigger / evidence** | MP-6/MP-8 commercial and competitive screening |
| **Reason** | Strong strategic fit but immature buyer category and hyperscaler competition reduced first-wave commercial confidence. |
| **Affected products** | Autonomous Agent Governance (challenger position) |
| **Program effect** | Not in active portfolio status table unless formally promoted |
| **Status** | **ACTIVE** |

---

### PD-005 — Prior Authorization as future/wildcard

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Keep Prior Authorization Operator as future/wildcard rather than an active first-wave product. |
| **Trigger / evidence** | MP-8 independence audit |
| **Reason** | Healthcare/regulatory/integration burden disproportionate to first multi-product experiment despite strong pain. |
| **Affected products** | Prior Authorization (wildcard position) |
| **Program effect** | Not in active portfolio status table |
| **Status** | **ACTIVE** |

---

### PD-006 — Session model

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Use independent Product Sessions plus one Portfolio Control Session. |
| **Trigger / evidence** | MP-11 governance contract |
| **Reason** | Preserve product autonomy while enabling cross-product verification and platform-impact classification. |
| **Affected products** | All program products |
| **Program effect** | Establishes dual session operating model |
| **Status** | **ACTIVE** |

---

### PD-007 — G4 escalation requirement

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Require G4 escalation before material shared-platform changes. |
| **Trigger / evidence** | MP-11 governance contract §7 |
| **Reason** | Prevent unclassified platform leakage and ensure cross-product review before shared-core modification. |
| **Affected products** | All program products |
| **Program effect** | Material platform pressure must pass G4 before implementation |
| **Status** | **ACTIVE** |

---

### PD-008 — Asynchronous pacing and portfolio recommendations

| Field | Value |
|-------|-------|
| **Date** | 2026-08-19 |
| **Decision** | Allow asynchronous product pacing and portfolio recommendations ACCELERATE / CONTINUE / REDUCE / PAUSE / STOP. |
| **Trigger / evidence** | MP-11 governance contract |
| **Reason** | Products advance at different rates; portfolio guidance must be independent of synchronized phase gates. |
| **Affected products** | All program products |
| **Program effect** | Enables independent product pacing with central portfolio guidance |
| **Status** | **ACTIVE** |

---

## Related documents

| Question | Document |
|----------|----------|
| Current portfolio state | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| Platform impact evidence | [PLATFORM_IMPACT_LEDGER.md](PLATFORM_IMPACT_LEDGER.md) |
| Selection history | [PRODUCT_PORTFOLIO_SELECTION.md](PRODUCT_PORTFOLIO_SELECTION.md) |
| Program operations | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) |
