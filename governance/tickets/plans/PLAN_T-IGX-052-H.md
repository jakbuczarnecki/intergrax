# PLAN T-IGX-052-H — PT-Q11 / M3 pin prep (governance refinement)

Status: **DRAFT**. Source: MEP-AF2 v2.1 W6 exit = **M3 recorded** via `PT-Q11` (EBH-7 final pin + independent B6). Base `e69e1ff58`. **Does not pin M3**, set `FREEZE_CERTIFICATION_SHA`, or close ARCH-FREEZE.

## 1. Purpose

MEP §K requires a ticket before the EBH-7 final pin. Prep checklist `M3_PREP_CHECKLIST.md` exists; this batch refines governance artefacts so pin work cannot start without an approved plan, explicit forbidden claims, and a bounded write set. No M3 SHA, no LOCK, no `ARCH_FREEZE_CLOSED`.

## 2. Prerequisites (program — none met)

| Id | Prerequisite | State @ base |
|----|--------------|-------------|
| H-P0 | W6 campaign + requal nodes executed or waived | **BLOCKED** |
| H-P1 | W5 finals (`T-IGX-052-A`..`D`) merged | **OPEN** |
| H-P2 | OD-AF-17 answered (post-freeze ENT baseline path) | **OPEN** |
| H-P3 | `executable_identity_violations` probe @ pre-pin tip (MEP PD3-05) | not run |

## 3. PT-Q11 prep rows (read-only until execution)

| Row | Topic | Prep id |
|-----|-------|---------|
| Identity allowlist + ordering | EBH-7 program docs | M3P-Q11-1 |
| Replay dispatch + B6 (AR-37, AR-53, AR-57) | not scheduled | M3P-Q11-2 |
| Pin commit = merge commit only | orchestrator rule | M3P-Q11-3 |
| `governance/tickets/` frozen after M3 — W6 scaffolds before pin | MEP §K | M3P-Q11-4 |
| `FREEZE_CERTIFICATION_SHA` in ladder | **TBD — no pin** | M3P-Q11-5 |

## 4. This batch (governance refinement only)

| Item | Path | Change |
|------|------|--------|
| Ticket | `governance/tickets/T-IGX-052-H.md` | refine |
| Plan | this file | refine |

No pin commit, no ladder SHA write, no `M3_PREP_CHECKLIST.md` edit in this batch.

## 5. Future pin batch (out of scope)

EBH-7 final pin merge, `FREEZE_CERTIFICATION_SHA` recording, ladder updates, and post-M3 governance freeze are **separate** operator-gated batches after H-P0..H-P3 and OD-AF-17.

## 6. Verification plan (this batch)

- `uv run python -m scripts.governance.igx_gov ticket`
- Plan digest recomputed; ticket updated.
- `verify_ticket_lifecycle` on `T-IGX-052-H.md` when available.

## 7. Out of scope

Performing the pin; W7 `PT-Q13`; editing `governance/tickets/` after M3; any PASS or certification claim; product paths; pytest at refine time.

## 8. Risks

Governance tickets F/G/H must reach `plan_approved` **before** M3 per M3P-Q11-4; late edits after pin are forbidden by MEP — refinement timing matters.