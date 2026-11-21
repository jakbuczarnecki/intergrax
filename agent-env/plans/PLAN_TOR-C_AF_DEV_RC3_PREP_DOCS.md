# PLAN TOR-C — RC-3 prep dossier (docs, tor `development`)

**Status:** plan for implementer (architect 2026-11-21). **Not** operator-approved. **No** G2 PASS / FCS / CLOSED language.

## 1. Purpose

After RC-2 partial exit (fala 5 on `af-dev-merge`), document **RC-3 entry prep** on the `development` tor only: draft G2 readiness checklist, train↔development gap sourced from `TRAIN_CHERRY_PICK_STATUS.md`, keep **RW-PLAN-AUDIT** BLOCKED. Ticket-less docs under `continuation/` per MEP-AF2 §I.

## 2. Scope

| In | Out |
|----|-----|
| Worktree `D:/Projekty/_wt/intergrax/af-dev-rc1-docs`, branch `af-dev-rc1-docs` rebased on `origin/development` | `intergrax/`, `agents/`, `applications/`, `scripts/`, governance product tickets |
| Edits only under `docs/project/maintainers/qualification/arch_freeze_program/continuation/` | IIAC records, RB-2 re-pin, train merge/PR, RW-PLAN-AUDIT simulation |
| One commit on `af-dev-rc1-docs`; serial merge to `development` via `af-dev-merge`; push `origin/development` | Declaring G2 pin, Q02 PASS, FCS, `ARCH_FREEZE_CLOSED`, or RW-PLAN PASS |

**SSOT inputs (read-only):** `arch-freeze_continuation_9eb1c56f.plan.md` §J RC-3; `MEP-AF2_v2.1.md` ladder `G2`/`R`/`FCS`; on `development` HEAD: `RC2_EXIT_CHECKLIST.md`, `TRAIN_CHERRY_PICK_STATUS.md`, `IIIC_GOVERNANCE_FAST_BLOCKED.md`, `OPERATOR_DECISIONS_AUTONOMY.md`.

## 3. Steps

1. `git fetch`; in `af-dev-rc1-docs`: `git rebase origin/development` (resolve only if continuation conflicts).
2. Refresh pins with `git rev-parse origin/development` and `origin/arch-freeze/rw-train`; optional `git log --oneline origin/arch-freeze/rw-train..origin/development` for gap table.
3. **Add** `RC3_PREP_DOSSIER.md` — title states **prep only, not RC-3 exit**; sections: predecessor (`RC2_EXIT_CHECKLIST` **PARTIAL**), plan §J RC-3 criteria (Q02/RB-2/FCS) as **prerequisites not met**, explicit **G2 does not exist yet**, links to sibling continuation files, werdykt line **DRAFT / NOT STARTED**.
4. **Add** `G2_CHECKLIST_DRAFT.md` — mirror `RC2_EXIT_CHECKLIST` table style; rows from MEP W3→W4 (RW complete, train CI green except expected-red, PT-G01/G02, RW-PLAN-AUDIT PASS, Q02 independent PASS, governance-fast @ G2, 12-commit `development` classification before FCS); every open item `[ ]` or `BLOCKED`; banner **not a G2 pin, not Q02 PASS**.
5. **Add** `TRAIN_DEV_GAP.md` — copy commit queue and CI gate narrative **from** `TRAIN_CHERRY_PICK_STATUS.md`; add merge-base + counts + which slices land on train vs remain dev-only; disposition **QUEUED / STOP** until green `governance-fast` (cite `IIIC_GOVERNANCE_FAST_BLOCKED.md` if present on HEAD).
6. **Update** `CHECKPOINT.md`: G4 → RC-3 prep dossier; refresh SHA pins; RW-PLAN **BLOCKED** unchanged.
7. **Update** `CONTINUATION_AUTONOMY_CHECKPOINT.md`: new fala row (TOR C / fala 6 docs) with development SHA after merge.
8. **Leave** `RW_PLAN_AUDIT_STATUS.md` BLOCKED (date refresh only if HEAD moved).
9. Commit message: `docs(arch-freeze): RC-3 prep dossier and train-dev gap (TOR C)` — **no** `Ref:` (ticket-less).
10. In `af-dev-merge`: merge `af-dev-rc1-docs` → `development`, push `origin/development`.

## 4. Files to edit

| Action | Path |
|--------|------|
| Add | `continuation/RC3_PREP_DOSSIER.md` |
| Add | `continuation/G2_CHECKLIST_DRAFT.md` |
| Add | `continuation/TRAIN_DEV_GAP.md` |
| Update | `continuation/CHECKPOINT.md` |
| Update | `continuation/CONTINUATION_AUTONOMY_CHECKPOINT.md` |
| Touch optional | `continuation/RW_PLAN_AUDIT_STATUS.md` (BLOCKED only) |

**Do not** modify `TRAIN_CHERRY_PICK_STATUS.md` in this tor unless syncing an already-stale copy without semantic change (prefer cite + gap doc).

## 5. Tests / verification

- `git diff --name-only origin/development...HEAD` ⊆ `docs/project/maintainers/qualification/arch_freeze_program/continuation/`.
- `rg -i 'G2 PASS|Q02 PASS|FCS|ARCH_FREEZE_CLOSED|RW-PLAN.*PASS' continuation/` → no false closure claims (allowed: "not G2 PASS", "BLOCKED", "PARTIAL").
- Post-push: confirm `origin/development` contains new files.

## 6. Risks

| Risk | Mitigation |
|------|------------|
| `af-dev-rc1-docs` behind fala 5 docs | Rebase first; if missing `TRAIN_CHERRY_PICK_STATUS.md`, read from `origin/development` only |
| Stale SHAs in tables | Pin full 40-char SHA + short prefix; mark snapshot date 2026-11-21 |
| Accidental closure wording | Explicit headers "nie CLOSED"; OD-AF-AUTO-05 citation |

## 7. Escalation (`/igx-autonomy`)

- Merge conflict outside `continuation/` → **STOP**, parent picks resolution owner.
- `governance-fast` / RB-2 fix attempted in docs → **STOP** (out of scope).
- Operator asks for G2 pin or Q02 record in same commit → **STOP** (forbidden).

## 8. Decision

Single deliverable batch (three new markdown + two checkpoint updates); no parallel doc tickets.