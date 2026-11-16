# PLAN T-IGX-050-D — ARCH-FREEZE node T-AF2-FRZ-MATRIX: 225-row planning record and 35 ticket-ready CODE_FIX plans

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementer session of the ARCH-FREEZE program (plan MEP-AF2 v2.1, node T-AF2-FRZ-MATRIX, wave W2, section D). Nothing here is approved, applied or audited.

## 1. Purpose

Turn `docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` into the planning record for all 225 freeze criteria (26 families). Every row gets the 17 planning fields that tie it to its semantic owner, contract, production paths, evidence, tests, negative tests, required fix, required certification, Operator decision, PASS and N/A conditions, Path B status and freeze impact. The product wave (RW train) cannot be planned until the 35 rows rated BLOCKS with track CODE_FIX have ticket-ready plans; this node provides them.

The document is a planning record. It promotes no row, closes no row and certifies nothing. `PASS_condition` cells state what a certifier needs.

## 2. Scope

- Status reconciliation of the existing 14-column table: the `CURRENT_STATUS` cell of the 21 rows that differ from the checklist (FRZ-AUD-01..10, FRZ-EVOL-01..10, FRZ-FRZ-11) is taken from the checklist at `origin/development`. The checklist has 108 `PASS`, 8 off-vocabulary `PASS`-prefixed labels and 109 `OPEN`. No other cell of the table changes.
- Appendix A: the 17 columns for 225 rows, one table per family, keyed by FRZ_ID.
- Appendix B: Dimension A / Dimension B split for FRZ-PRD-01, PRD-02, PRD-03, PRD-08, SEC-04, SEC-05 and SEC-06, as in plan section D, with the OD-AF-18 handling of SEC-05/06.
- Appendix C: a ticket-ready plan for each of the 35 BLOCKS/CODE_FIX rows of WP-12 `ARCH_FREEZE_BLOCKERS.md` section H (ticket, scope, red-first negative test, regression scope, sensitivity). Appendix C2 lists the rows that join the BLOCKS set after the PT-FRZ-RR corrections and DP-36/DP-37.
- Appendix D: the 8 PT-FRZ-RR classification corrections and 9 consistency flags, and the 38 rows protected from a whole-row Path B move.
- Inputs, all read-only: WP-4 classifications (209 rows), PT-FRZ-RR, WP-12 master rows, decisions DP-34..40, OD-AF-08, OD-AF-18, the PDC package (PR #612) and plan sections D, E, G.

## 3. Explicitly out of scope

- The FRZ checklist and the roadmap (Operator-only: PT-D10, PT-D09). Their amendments appear as nodes, not edits.
- Every path under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, `pyproject.toml`, `uv.lock`, the IIAC files and the PDC-X record directories.
- Any PASS, closure, certification, FRZ promotion, N/A grant or independent-verification claim.
- Applying the A/B split or the OD-AF-18 wording to the checklist; reachability measurement (T-AF2-CLOSED-WORLD); the RW-train file-overlap matrix and merge order (T-AF2-RW-PLAN).

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md
governance/tickets/T-IGX-050-D.md
governance/tickets/plans/PLAN_T-IGX-050-D.md
```

## 5. Method

1. Worktree outside the repository on branch `af2/w2-frz-matrix`, cut from `origin/development` at `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2` (it includes the PR #606 edits to the matrix).
2. A generator script in the session output folder reads the checklist, the matrix, the WP-4 rows, the WP-12 master rows and the PDC FRZ coverage matrix, and writes the appendices. Cells quote or compress those sources with attribution (`[WP-4]`, `[RR]`, `[PDC]`, `[Plan]`); session artefacts are not presented as repo facts.
3. The 35 plans are written by hand from the WP-12 findings, the PDC package and plan section G; finding-level negative tests and regression scopes from the WP-12 master rows are attached with attribution, and test paths that do not exist at the base are marked.
4. A `FACT_CHECK.md` in the session output folder records each repo fact as `path@sha:line`.
5. The mechanical check is a `python -I` script in the output folder; its output goes into the PR body.

## 6. Test contract

No product test is run. One docs-only verification step, one process at a time:

- `python -I check_matrix.py <worktree>`: 225 rows and 26 families; 17 fields populated in every row; every checklist status reproduced exactly (108 + 8 / 109); every non-PASS row has a closing node; no row has `path_B` outside L66 unless OD-AF-18 is cited; the 35 plans and the 7 split rows are present; the checklist and roadmap are untouched.
- `python scripts/governance/verify_arch_freeze_plan_traceability.py` (existing validator, reads the matrix).
- `python scripts/docs/check_docs_domain_pairs.py`.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths of section 4.

## 7. Risks

- **Stale or wrong source classifications.** WP-4 covers 209 of 225 rows and predates OD-AF-18; PT-FRZ-RR found 8 differences. Mitigation: the corrections are applied and marked `[RR]`; rows outside WP-4 say `NOT_CLASSIFIED`.
- **Ticket letters can drift.** `T-IGX-043-*` and `T-IGX-051-*` letters follow the orchestrator dashboard; `T-IGX-052-*` letters are not allocated. Mitigation: stated in the appendix limits; T-AF2-RW-PLAN may reassign.
- **Matrix size.** The appendices add about 250 KB. Mitigation: legend codes (PC4, PP0, EV0, NA0, NEG-P, NEG-0) replace repeated boilerplate.
- **`--sync-matrix` interaction.** The validator's `--sync-matrix` option rewrites the `REMEDIATION_TASK` column of rows that are no longer `OPEN`. Appendix rows do not start with `| FRZ-` (the ID is in backticks), so the option does not touch them; the 21 reconciled rows would lose their remediation task in the main table, which Appendix A keeps.
- **Pin validity.** The file lives under `docs/`; it does not change `intergrax/`, `agents/` or `applications/`, so it does not affect the IIAC activation pin. Whether a docs-only change needs an EBH-7 re-pin is decided by the orchestrator under DP-61.
