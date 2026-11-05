# T-IGX-SEF-A — ASF/SEF corrective program (Master Plan governance)

## Scope

**P1-B0 bootstrap** and (after `plan_approved`) **P1-B1 publication** for the ASF/SEF corrective Master Plan @ Git baseline `7f24d0ea7113d7f23698e556695be40c363dd38a` (`development`).

**Plan binding (repo SSOT):** [`governance/tickets/plans/ASF-SEF-MASTER-PLAN.md`](plans/ASF-SEF-MASTER-PLAN.md) — copied from Cursor plan `asf-sef_corrective_master_plan_89133b21.plan.md` @ B0; operator content pin (raw file SHA-256) `db3386be2b139e4340fea347c70851fa8451ec6f8c09f89f18fd34fb1788795e`.

**Operator decisions (published P1-B1):** canonical register [`docs/project/maintainers/qualification/sef/ASF_SEF_OPERATOR_DECISION_REGISTER.md`](../../docs/project/maintainers/qualification/sef/ASF_SEF_OPERATOR_DECISION_REGISTER.md) — OD-SEF-01 Variant C+D-lite; OD-SEF-02 Option 3; OD-SEF-03 ADR **Proposed** only; **Option 4 NOT AUTHORIZED**.

## P1-B1 publication (2026-11-05)

- OD register, ADR-SEF-001 (`Proposed`), roadmap Option 3 (§3.0.7a **SEF-X**), architecture satellites under `docs/project/maintainers/plans/satellites/ASF_SEF_*`.
- **START_HEAD (Phase 1):** `b342e5e7f56fe67aa4dcc7b7f828da5fd4f15bbe`

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

Product CAP-049..052 implementation, Option 4 pilot, ADR **`Accepted`**, Protocol 03 B6 / FRZ row promotion / ARCH-FREEZE CLOSED claims, EVOL-X formal CLOSED, P2–P13 engineering waves until post–P1-B1 gates.

## Lifecycle state

quality_gate

## Checkpoint

**P1-B1 + corrective (@ AUDITED_SHA `7de6b41f637df1d6949985d9dc567602a074e230`):**

- Links: **108 OK / 0 broken** — `.tmp/session/ASF-SEF-P1-CORRECTIVE/check_links_after.log`
- `verify_plan_approval.py` / `verify_ticket_lifecycle.py`: OK — corrective logs
- `igx_gov ticket`: PASS — `.tmp/session/ASF-SEF-P1-CORRECTIVE/igx_gov_ticket_after.log`
- `check_harness_adr.py`: **FAIL (24 global)** — not P1-introduced; **OD-SEF-P1-EXCEPTION** **AWAITING OPERATOR EXCEPTION APPROVAL**; hygiene **T-IGX-014-A** scaffolded
- Acceptance prep: [`ASF_SEF_P1_OPERATOR_ACCEPTANCE_RECORD_PREPARED.md`](../../docs/project/maintainers/qualification/sef/ASF_SEF_P1_OPERATOR_ACCEPTANCE_RECORD_PREPARED.md); closure dossier `.tmp/session/ASF-SEF-P1-CLOSURE/DOSSIER.md`

**Formal closure:** **blocked** until operator approves **OD-SEF-P1-EXCEPTION** and signs acceptance record (**no FORMAL_CLOSURE_SHA** in this batch).

**Commit trailer:** `Ref: governance/tickets/T-IGX-SEF-A.md`
