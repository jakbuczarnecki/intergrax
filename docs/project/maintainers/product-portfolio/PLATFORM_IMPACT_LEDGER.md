# Platform Impact Ledger

**Document type:** Maintainer-level operational control artifact  
**Owner:** Portfolio Control Session  
**Last updated:** 2026-08-19 (MP-12 initial creation)

---

## Purpose

This document answers: **What did a real product need reveal, reuse, change, invalidate, or prove about Intergrax?**

It is an append-oriented audit ledger of **material product pressure** on the shared Intergrax platform.

**This is NOT:**

- a speculative backlog
- a list of platform features products might need someday
- an architecture roadmap
- a place to invent reuse before evidence exists

**Candidate future reuse is NOT evidence of reuse.** Only later verified consumption may populate `Later Reuse Evidence`.

---

## Ownership

| Role | May do | May NOT do |
|------|--------|------------|
| Product Session | Detect, report, and evidence platform pressure | Directly declare accepted platform impact in this ledger |
| Portfolio Control | Accept, classify, and append impact records after verification | Add speculative records without product evidence |

Where exact implementation evidence exists, completion summaries alone are insufficient.

---

## Current ledger state

**No accepted multi-product platform impact records exist yet.**

LKW historical/platform pressure will be ingested later in MP-13 through an evidence-based baseline review. Do not retroactively invent ledger items before that review.

---

## Stable ID format

| Rule | Detail |
|------|--------|
| Format | `PI-001`, `PI-002`, … |
| Immutability | IDs are never reused |
| Orientation | Append-only; do not delete accepted history |

If a decision later proves wrong:

- do **not** delete the old entry
- mark it **SUPERSEDED** or **INVALIDATED**
- reference the replacement or new evidence

---

## Record status (controlled vocabulary)

| Status | Meaning |
|--------|---------|
| **OPEN** | Reported; not yet accepted by Portfolio Control |
| **ACCEPTED** | Verified and part of the authoritative impact record |
| **SUPERSEDED** | Replaced by a later accepted record; history preserved |
| **INVALIDATED** | Earlier conclusion overturned by evidence; history preserved |

---

## Classifications (canonical)

Use exact classifications from [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md):

| Classification | Meaning |
|----------------|---------|
| **REUSED_UNCHANGED** | Existing shared mechanism consumed without platform modification |
| **REUSED_CONFIGURED** | Existing mechanism reused through intended configuration, policy, adapter, or DI contract without changing core platform semantics |
| **EXTENDED_GENERALLY** | Real product pressure exposed a missing reusable capability; platform extended through a general contract — useful evolution, not pure reuse |
| **PRODUCT_OWNED** | Behavior correctly belongs to the product (domain workflow, UX, business semantics, product-specific policy meaning) |
| **PLATFORM_LEAK** | Product-specific branching, private infrastructure duplication, bypass of shared contracts, or product-specific behavior leaking into platform core |

Do not invent new reuse classifications.

**Rules:**

- `PLATFORM_LEAK` remains a defect classification.
- `EXTENDED_GENERALLY` requires an accepted G4 decision before implementation proceeds (see [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §7).
- Later unchanged/configured reuse by another product is **stronger** evidence than the originating product alone.
- Failed generalization is valuable evidence and must remain visible.

---

## Minimum record schema

Every future `PI-*` entry must include:

| Field | Required content |
|-------|------------------|
| **ID** | Stable `PI-NNN` identifier |
| **Date** | Date accepted or last status change |
| **Origin Product** | Product that surfaced the pressure |
| **Product Need** | What the product required |
| **Product Evidence** | Exact SHA or authoritative artifact |
| **Platform Area** | Affected shared platform boundary or capability |
| **Classification** | One canonical classification from table above |
| **Decision** | Portfolio Control disposition |
| **Decision Evidence** | Link, SHA, or artifact supporting the decision |
| **Other Active Products Reviewed** | Which peer products were checked for cross-product effect |
| **Cross-Product Outcome** | Controlled conclusion (see below) |
| **Later Reuse Evidence** | Verified later consumption by other products, if any |
| **Status** | OPEN / ACCEPTED / SUPERSEDED / INVALIDATED |
| **Supersedes / Superseded By** | When applicable |

### Cross-product outcome (controlled conclusions)

- unaffected
- compatible by configuration
- requires later adoption
- reveals conflict
- invalidates proposed generalization

---

## Anti-gaming rules

1. Never add speculative impact records merely because a future product might need something.
2. Never classify platform work as reusable before accepted evidence.
3. `PLATFORM_LEAK` remains a defect — not a neutral outcome.
4. `EXTENDED_GENERALLY` requires accepted G4 decision before material shared-platform implementation.
5. Later unchanged/configured reuse by another product outweighs originating-product-only claims.
6. Failed generalization must remain visible; do not erase inconvenient history.

---

## Related documents

| Question | Document |
|----------|----------|
| Reuse methodology | [PRODUCT_REUSE_PROOF.md](../plans/PRODUCT_REUSE_PROOF.md) |
| G4 escalation gate | [MULTI_PRODUCT_PROGRAM.md](MULTI_PRODUCT_PROGRAM.md) §7 |
| Current portfolio state | [PORTFOLIO_STATUS.md](PORTFOLIO_STATUS.md) |
| Program direction changes | [DECISION_LOG.md](DECISION_LOG.md) |
