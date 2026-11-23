# PLAN T-IGX-050-N — T-AF2-RW-PLAN-AUDIT: independent session prep (development tor)

Draft, 2026-11-21. Documentation and ticket scaffold only on the `development` train. No product paths, no pytest, no simulation of the independent auditor session, and no PASS, SCOPE closure, FRZ promotion, or G2 claim anywhere in this plan.

## 1. Purpose

Publish **entry artefacts** for node **T-AF2-RW-PLAN-AUDIT** (MEP-AF2 W2, AUD, REV-MASTER RM-03) so a future **`independent_auditor`** session (AR-12-07 GRANTED) can start without re-deriving inputs from chat. The formal audit verdict remains **BLOCKED** until that session records its outcome in `RW_PLAN_AUDIT_STATUS.md`.

## 2. Binding inputs (read-only)

- [`MEP-AF2_v2.1.md`](../../../docs/project/maintainers/qualification/arch_freeze_program/MEP-AF2_v2.1.md) — RW-PLAN scope and FIND-AF2-23 / phase check 3.
- [`WP12_DAG.md`](../../../docs/project/maintainers/qualification/arch_freeze_program/WP12_DAG.md) — edge `T-AF2-RW-PLAN` → `T-AF2-RW-PLAN-AUDIT` → `SCOPE`.
- [`ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md`](../../../docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md) — Appendix C / FRZ rows for audit recompute.
- [`RC3_EXIT_PREP_CHECKLIST.md`](../../../docs/project/maintainers/qualification/arch_freeze_program/continuation/RC3_EXIT_PREP_CHECKLIST.md) — section RC3-J7 ladder.
- Operator decision **AR-12-07** in [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../../../docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md).

## 3. Write set (this batch)

| Path | Action |
|------|--------|
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/RW_PLAN_AUDIT_PREP_DOSSIER.md` | **Add** — session prep dossier |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/RW_PLAN_AUDIT_STATUS.md` | **Update** — pin refresh; verdict **BLOCKED** only |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/RC1_OPEN_ITEMS_DEV.md` | **Update** — RC1-J1 cites prep dossier |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/CHECKPOINT.md` | **Update** — workstream row `af-dev-rw-plan` |
| `docs/project/maintainers/qualification/arch_freeze_program/continuation/G2_CHECKLIST_DRAFT.md` | **Update** — G2-B3 cites prep dossier (still BLOCKED) |
| `governance/tickets/T-IGX-050-N.md` | **Add** — ticket scaffold |
| `governance/tickets/plans/PLAN_T-IGX-050-N.md` | **Add** — this plan |

## 4. Steps

1. Add `RW_PLAN_AUDIT_PREP_DOSSIER.md` with pin `e69e1ff5819b98c316377bcadf749f61fe00abc2`, J7 ladder, mandatory inputs, and explicit **NOT STARTED** for the auditor session.
2. Refresh `RW_PLAN_AUDIT_STATUS.md` date/pin; keep verdict **BLOCKED** (no PASS line).
3. Cross-link RC1-J1, G2-B3, and `CHECKPOINT.md` parallel table.
4. Scaffold `T-IGX-050-N.md` with lifecycle `draft`; record plan digest in the ticket.

## 5. Out of scope

- Running or pretending to run the independent audit.
- Editing `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`.
- Changing `SCOPE`, G2, FCS, or `ARCH_FREEZE_CLOSED` status claims.
- Publishing or merging `RW_PLAN` body from train onto `development` (cite train SHAs only).

## 6. Verification

- Docs-only batch: no pytest.
- Optional: `uv run python -m scripts.governance.plan_digest.py governance/tickets/plans/PLAN_T-IGX-050-N.md` after plan edits to refresh the ticket digest line.