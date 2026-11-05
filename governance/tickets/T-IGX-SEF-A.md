# T-IGX-SEF-A — ASF/SEF corrective program (Master Plan governance)

## Scope

**P1-B0 bootstrap** and (after `plan_approved`) **P1-B1 publication** for the ASF/SEF corrective Master Plan @ Git baseline `e019c39df042a09bca1b998fca2b02dd02ae3654` (`development`).

**Plan binding (repo SSOT):** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](plans/ASF-SEF-MASTER-PLAN.md) — copied from Cursor plan `asf-sef_corrective_master_plan_89133b21.plan.md` @ B0; operator content pin (raw file SHA-256) `db3386be2b139e4340fea347c70851fa8451ec6f8c09f89f18fd34fb1788795e`.

**Operator decisions (intent only until B1):** OD-SEF-01 Variant C+D-lite; OD-SEF-02 Option 3 (SEF-X ∥ SCENARIO-READINESS-X post ARCH-FREEZE); OD-SEF-03 ADR-SEF-001 **Proposed** only; **Option 4 NOT AUTHORIZED**.

## Plan digest

Plan digest: a500419535c8187d16e4950744fb0cb99fdc77b6ccd1d76092f254d1ccae80d2

## Approved

*(P1-B0 STOP — operator must add `Approved: <name> @ <ISO8601>` after final plan review; no agent-generated approval.)*

## Test contract

After operator `Approved:` line (B0-5..B0-7):

```powershell
uv run python scripts/governance/verify_plan_approval.py governance/tickets/T-IGX-SEF-A.md --plan governance/tickets/plans/ASF-SEF-MASTER-PLAN.md
```

P1-B1 gate (documentation batch only):

```powershell
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

P1-B1 until `plan_approved`: ADR-SEF-001 file, roadmap ledger edits, architecture satellites, CAP-049..052 product implementation, Option 4 pilot, Protocol 03 B6 / FRZ / ARCH-FREEZE status changes, EVOL-X formal CLOSED, fake `Approved:`.

## Lifecycle state

`plan_pending`

**Commit trailer:** `Ref: governance/tickets/T-IGX-SEF-A.md`
