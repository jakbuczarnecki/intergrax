# T-IGX-SEF-A — ASF/SEF corrective program (Master Plan governance)

## Scope

**P1-B0 bootstrap** and (after `plan_approved`) **P1-B1 publication** for the ASF/SEF corrective Master Plan @ Git baseline `7f24d0ea7113d7f23698e556695be40c363dd38a` (`development`).

**Plan binding (repo SSOT):** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](plans/ASF-SEF-MASTER-PLAN.md) — copied from Cursor plan `asf-sef_corrective_master_plan_89133b21.plan.md` @ B0; operator content pin (raw file SHA-256) `db3386be2b139e4340fea347c70851fa8451ec6f8c09f89f18fd34fb1788795e`.

**Operator decisions (intent only until B1):** OD-SEF-01 Variant C+D-lite; OD-SEF-02 Option 3 (SEF-X ∥ SCENARIO-READINESS-X post ARCH-FREEZE); OD-SEF-03 ADR-SEF-001 **Proposed** only; **Option 4 NOT AUTHORIZED**.

## Plan digest

Plan digest: a500419535c8187d16e4950744fb0cb99fdc77b6ccd1d76092f254d1ccae80d2

## Approved

Approved: operator @ 2026-11-05T09:08:00+01:00 (ASF-SEF-P0 final Master Plan review — binding governance/tickets/plans/ASF-SEF-MASTER-PLAN.md; Plan digest a500419535c8187d16e4950744fb0cb99fdc77b6ccd1d76092f254d1ccae80d2; unlock P1-B1 publication only — OD-SEF-01/02/03, ADR-SEF-001 Proposed, roadmap Option 3, arch docs; no product/CAP-049..052, Option 4, ADR Accepted, FRZ/B6/ARCH-FREEZE status changes)

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

`plan_approved`

**Commit trailer:** `Ref: governance/tickets/T-IGX-SEF-A.md`
