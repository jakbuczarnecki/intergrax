# ASF/SEF — Operator decision register (OD-SEF)

**Campaign:** ASF-SEF corrective program · **Ticket:** [`governance/tickets/T-IGX-SEF-A.md`](../../../../../governance/tickets/T-IGX-SEF-A.md)  
**Plan SSOT (binding):** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](../../../../../governance/tickets/plans/ASF-SEF-MASTER-PLAN.md)  
**Publication batch:** P1-B1 @ operator authorization 2026-11-05 (plan digest `a500419535c8187d16e4950744fb0cb99fdc77b6ccd1d76092f254d1ccae80d2`)

This register is the **canonical repo store** for OD-SEF rows (pattern: EVOL-X [`EVOL_X_OPERATOR_DECISION_REGISTER_W1.md`](../evol_x/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md), ARCH-FREEZE [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md)).

## Status @ P1-B1 publication

| OD_ID | Decision | Status | Authorized signer | Blocks / enables |
|-------|----------|--------|-------------------|------------------|
| **OD-SEF-01** | Accept architecture **Variant C + D-lite** (separate Engineering Execution Authority vs Scenario Runtime; provider-neutral `CodingExecutionBackend`; no second Nexus/ToolRuntime/PolicyEngine SSOT) | **APPROVED** | Operator @ 2026-11-05 (via plan approval) | Architecture narrative + ADR-SEF-001 draft |
| **OD-SEF-02** | Roadmap **Option 3**: after **ARCH-FREEZE CLOSED**, **SEF-X** ∥ **SCENARIO-READINESS-X**; **Option 4 NOT AUTHORIZED**; **no** automatic SCENARIO-READINESS-X ← SEF-X completion dependency | **APPROVED** | Operator @ 2026-11-05 | Roadmap §5 + §3.0.7a patch (T-P1-005) |
| **OD-SEF-03** | ADR-SEF-001 **preparation only** — publish **`Proposed`**; **`Accepted`** requires separate review | **APPROVED FOR PREP ONLY** | Operator @ 2026-11-05 | [`ADR-SEF-001`](../../../technical/adr/entries/2026-11-05/ADR-SEF-001.md) |
| OD-SEF-04 | Engineering FSM binding to tickets | PENDING | Operator | T-P3-001 |
| OD-SEF-05 | Coding backend allowlist | PENDING | Operator | T-P4 |
| OD-SEF-06 | SEF scoped B6 auditor appointment | PENDING | Operator | T-P12-003 |
| OD-SEF-07 | SCENARIO-READINESS handoff scope | PENDING | Operator | T-P13-005 |
| OD-SEF-08 | Campaign closure + roadmap ledger | PENDING | Operator | T-P13-006 |
| (existing) OD-IC-01 | Internal EVOL/ENT acceptance | COMPLETE | Operator | Does **not** replace formal B6 |
| (existing) EVOL B6 | Protocol 03 independent | OPEN | Independent auditor | ARCH-FREEZE chain |
| **OD-SEF-P1-EXCEPTION** | Accept global `check_harness_adr.py` **24-findings** as **out-of-scope** for ASF-SEF-P1 closure | **AWAITING OPERATOR EXCEPTION APPROVAL** | Operator | P1 ticket closure; does **not** waive **T-IGX-014-A** hygiene |

## Decision text (OD-SEF-01..03)

### OD-SEF-01 — Variant C + D-lite

**Operator decision:** Adopt **Engineering/Delivery Plane** as **Variant C + D-lite**: DevOS control plane + engineering orchestrator + **provider-neutral** `CodingExecutionBackend` adapters for repo-bound engineering jobs. **Scenario Runtime Plane** remains **HostTaskExecutionPort → NexusLoop → PolicyEngine → ToolRuntime**. **Rejected:** Variant A (embedded factory core — violates single Nexus SSOT); Variant B (external factory with alternate governance SSOT).

**Technical basis:** Master Plan Deliverable 1, 5; REQ-ASF-002; GAP-CRIT-05 closure path.

### OD-SEF-02 — Option 3 (parallel post-freeze tracks)

**Operator decision:** After **ARCH-FREEZE CLOSED**, **`SEF-X`** and **`SCENARIO-READINESS-X`** may proceed **in parallel** with explicit schedule coupling in the roadmap ledger. **Forbidden by default:** SCENARIO-READINESS-X blocked on SEF-X completion. Cross-track WBS edges require documented technical justification in ticket + plan digest. **Option 4** (`platform_proofs/` pilot-only factory): **NOT AUTHORIZED** — deferred/rejected in impact analysis only.

**Technical basis:** Master Plan Deliverable 14 Option 3, Deliverable 16; roadmap [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.7a.

### OD-SEF-03 — ADR preparation gate

**Operator decision:** P1-B1 may publish **ADR-SEF-001** with status **`Proposed` only**. Status **`Accepted`** requires separate operator acceptance review and ADR process — not implied by OD-SEF-03 or plan approval.

**Technical basis:** [`docs/project/technical/adr/README.md`](../../../technical/adr/README.md); Master Plan Deliverable 15.

### OD-SEF-P1-EXCEPTION — Global ADR validator @ P1 closure

**Status:** **AWAITING OPERATOR EXCEPTION APPROVAL** (placeholder — **no operator signature**)

**Decision (draft):** For **ASF-SEF-P1** independent acceptance and ticket closure only, treat **`scripts/maintenance/check_harness_adr.py` exit 1** with the **24** documented findings (`.tmp/session/ASF-SEF-P1-CLOSURE/DOSSIER.md` § ADR_GLOBAL_BASELINE_RECONCILIATION) as **pre-existing global index/parser drift**, **not introduced** by commits `e2f88cdf4` / `7de6b41f`. P1 scope added **ADR-SEF-001** with a validator-visible index link. This exception **does not** declare `check_harness_adr.py` PASS, **does not** close **T-IGX-014-A**, and **does not** promote FRZ/B6/ARCH-FREEZE or ADR **`Accepted`**.

**Evidence baselines:** START_HEAD `b342e5e7f` and AUDITED_SHA `7de6b41f` both report **24** identical missing-index messages; logs under `.tmp/session/ASF-SEF-P1-CORRECTIVE/` and `.tmp/session/ASF-SEF-P1-CLOSURE/`.

**Operator signature (required to activate):**

```text
Exception approved: operator @ <ISO8601> (OD-SEF-P1-EXCEPTION — global ADR validator out-of-scope for ASF-SEF-P1 only)
```
