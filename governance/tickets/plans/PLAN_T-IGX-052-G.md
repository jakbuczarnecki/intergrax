# PLAN T-IGX-052-G — W6 family requalification and Campaign C prep (governance refinement)

Status: **DRAFT** for Operator review. Source: MEP-AF2 v2.1 W6; WP-12 `PT-C01`, `PT-Q05`, `PT-Q06`, `PT-Q07`, `PT-Q08`, `PT-Q09`, optional `PT-Q10`, `PT-Q12`. Base `e69e1ff58`. No requal PASS, no independent audit, no family replay executed.

## 1. Purpose

Separates **governance SSOT** for Campaign C sample and family requalification nodes from execution. Refines ticket and plan only so later batches can attach evidence and commands without conflating prep with PASS. Claims no Q05..Q12 verdicts, no IIIC certifier session, no FRZ row promotion.

## 2. Prerequisites (program — not met @ `e69e1ff58`)

| Id | Prerequisite | Notes |
|----|--------------|-------|
| G-P0 | **FCS** recorded; tree ≡ G2 | **BLOCKED** |
| G-P1 | W5 checklists/docs per MEP merged | see W6_PREP_DOSSIER W6-P1 |
| G-P2 | TENANT / TRACE / STATE residuals visible in FRZ matrix | W6R-Q06-1, W6R-Q07-2 cite BLOCKED rows |
| G-P3 | Independent audit commissions (AR-54..57, AR-08/09) named in WP-12 | **not** run |

## 3. Node map (execution deferred)

| Node | Role | Prep checklist |
|------|------|----------------|
| `PT-C01` | Campaign C sample (HIST-85) | `W6_CAMPAIGN_PREP_CHECKLIST.md` §C (W6C-D1..D2) |
| `PT-Q05` | ROADMAP-REPLAY-X counter 0 + DUP-X | `W6_REQUAL_PREP_CHECKLIST.md` |
| `PT-Q06` | STATE-X, TRACE-X | same |
| `PT-Q07` | TENANT-X, TEN-REVAL @ FCS, MCP delta | same |
| `PT-Q08` | TOOL-LIFE-X RA-1 | same |
| `PT-Q09` | DIST-SCALE-X scoped envelope | same |
| `PT-Q10` | Optional batch (DP-04/09 = B) | same |
| `PT-Q12` | Family verification suites @ FCS | same |

**Excluded from this ticket:** `PT-Q11` / M3 pin → T-IGX-052-H. Campaign B omega nodes → T-IGX-052-F.

## 4. This batch (governance refinement only)

| Item | Path | Change |
|------|------|--------|
| Ticket | `governance/tickets/T-IGX-052-G.md` | refine |
| Plan | this file | refine |

No qualification doc edits in this batch.

## 5. Future execution write set (out of scope here)

Evidence and checklist updates under `docs/project/maintainers/qualification/arch_freeze_program/` and per-node qualification paths cited in `W6_REQUAL_PREP_CHECKLIST.md`; optional `tests/` or campaign harness paths per AR rows filled at execution approval. Requires separate ticket state transition and test contract.

## 6. Verification plan (this batch)

- `uv run python -m scripts.governance.igx_gov ticket`
- `uv run python scripts/governance/plan_digest.py` on this plan; update ticket digest line.
- `verify_ticket_lifecycle` on `T-IGX-052-G.md` when available.

## 7. Out of scope

`PT-Q11`, W7/W8 lock work; `intergrax/` product edits in prep or refine batches; independent audit PASS language; editing `W6_REQUAL_PREP_CHECKLIST.md` here.

## 8. Risks

Optional `PT-Q10` DAG may waive nodes; ticket scope must stay explicit so execution batches do not imply Q10 ran when skipped.