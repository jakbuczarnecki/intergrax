# PLAN T-IGX-050-B — ARCH-FREEZE program W0: traceability validator amendment (T-AF2-VALIDATOR-AMEND)

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementing session for the orchestrator of MEP-AF2 v2.1. Nothing here is approved, applied or audited. The validator is a tool the certifier will rely on, so it needs an independent review (T-AF2-VALIDATOR-INDEP-REVIEW) before use; this ticket does not provide it.

## 1. Purpose

`scripts/governance/verify_arch_freeze_plan_traceability.py` hard-codes counts (225 FRZ, 26 families, PASS 108, OPEN 109, 54 WBS tasks, 16 qualification runs) and a WBS shape from before IIAC closed EVOL-X and ENT-AUDIT-X. Its `--sync-wbs` and `--sync-matrix` modes rewrite the WBS and the matrix from hard-coded row data. MEP-AF2 v2.1 section C items 1-4 and C.4 amend it. This ticket implements that amendment and nothing else. It certifies nothing, promotes no FRZ criterion and changes no stage status.

## 2. Scope (MEP-AF2 v2.1 section C)

| # | Change | Where |
| --- | --- | --- |
| 1 | Every hard DAG rule is kept with unchanged text and logic; they all still pass on the current WBS without rewiring | `_validate_semantic_dag` |
| 2 | `SUPERSEDED_BY_CURRENT_AUTHORITY` rows stay DAG nodes (edges through them keep counting for reachability) and are recorded as satisfied; the `V21_QUALIFICATION` cell must carry a non-empty `authority_ref:` (placeholders such as `TBD` are rejected) | `_check_qualification_states` |
| 3 | Counters are computed: checklist FRZ / family / PASS / OPEN counts; WBS total = 54 historical rows + the `T-AF2-*` and `PT-*` rows; qualification run count; an unknown checklist status is an error | `collect` |
| 4 | The matrix-vs-checklist cross-check (row sets, status class, header `Rows` / `Families` counts) runs only when the matrix carries the header line `**MATRIX_CHECKLIST_CROSSCHECK:** ENABLED` [PD4-01]; T-AF2-SSOT-RECONCILE-B adds that line, the W0 run does not need it | `_matrix_crosscheck_enabled` |
| 5 | Task IDs use the word-boundary regex `\b(?:T\|PT)-[A-Z0-9-]+`; `PT-` and `T-AF2-` IDs are accepted as WBS rows and dependencies; `PT-Q11` is never read as `T-Q11` and `ENT-B6` never yields `T-B6` | `_TASK_ID_RE` |
| 6 | `FIND-PROD-REACH-DEFERRED` is INFO: removed from the blocker set, required in the gap register with severity INFO, reported as an INFO line | `INFO_FINDINGS` |
| 7 | DAG self-test, run on every validation and with `--self-test`: reference graph acyclic and rule-clean; one mutation per rule is rejected; each of the 20 COORD prerequisites, once cut from COORD, is rejected; a fixture proves the edge `T-B5-EVOL-FRZ <- T-B7-ECAP-2C` is rejected | `dag_self_test` |
| 8 | `--sync-wbs` and `--sync-matrix` exit with code 2 and a message that points to plan section C and touch no file; the functions and the hard-coded row data they used are deleted | `main` |

One additive hard rule [PD2-01]: `T-B5-EVOL-FRZ` and `T-B6-FRZ-AUD` must not follow `T-B7-ECAP-2C` or `T-B7-LOCK-FREEZE-SHA`. It names, in the validator's own words, the violation that the self-test fixture of item 7 needs to show; the existing cycle check already rejects the same edge.

## 3. Explicitly out of scope

- Any edit of the WBS, the matrix, the gap register, the checklist or any other document (the validator only reads them); the matrix header flag is added by T-AF2-SSOT-RECONCILE-B, not here.
- The IIAC material scripts, product paths, the roadmap, `.github/`, GitHub settings.
- The independent review of the validator (T-AF2-VALIDATOR-INDEP-REVIEW), any re-run that would count as qualification (T-B0-TRACE-VALIDATE), and any PASS or certification statement.
- Matching the `**Task count:**` line of the WBS header: it is prose, not checked.

## 4. Write set (the only paths the PR may touch)

```text
scripts/governance/verify_arch_freeze_plan_traceability.py
tests/unit/governance/test_arch_freeze_plan_traceability.py
governance/tickets/T-IGX-050-B.md
governance/tickets/plans/PLAN_T-IGX-050-B.md
```

## 5. Method

1. Worktree outside the repository at `<parent>/_wt/intergrax/w0-validator`, branch `af2/w0-validator` cut from a freshly fetched `origin/development`, as granted by OD-AF-02.
2. Remove `sync_wbs`, `sync_matrix`, `remediation_task` and their helpers; amend the validator; add tests that point the validator at copies of the real qualification files in a temporary directory and mutate them.
3. Every factual sentence added to a record is checked against `origin/development` and listed in `.tmp/session/arch-freeze/T-AF2-VALIDATOR-AMEND/FACT_CHECK.md` (session record).
4. Commit without any AI-tool attribution, with the trailer `Ref: governance/tickets/T-IGX-050-B.md`. Push with the upstream set first and an explicit refspec; open a ready (non-draft) pull request against `development`.

## 6. Test contract

As in the ticket: one `uv run pytest` process under the shared lock, one validator run, one ruff run.

## 7. Risks

- The matrix header flag name `MATRIX_CHECKLIST_CROSSCHECK` is chosen here; T-AF2-SSOT-RECONCILE-B must use exactly that line.
- Reachability-based hard rules are transitive, so a COORD prerequisite can stay reachable through another path; the self-test therefore cuts every edge into the prerequisite instead of one edge.
- The validator is a certifier input: until the independent review passes it must not be cited as qualification evidence.
