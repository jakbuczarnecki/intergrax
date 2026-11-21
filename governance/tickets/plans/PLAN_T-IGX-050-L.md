# PLAN T-IGX-050-L — ARCH-FREEZE program W2: fixes for the independent review of the amended traceability validator (T-AF2-VALIDATOR-FIX)

Status: DRAFT for Operator review. Drafted 2026-11-17 by an implementing session for the orchestrator of MEP-AF2 v2.1. Nothing here is approved, applied or audited. The validator is a tool the certifier will rely on; this ticket fixes review findings and does not itself provide a new independent review.

## 1. Purpose

The independent review of the validator amended by T-IGX-050-B (PR #628, merge `a7f023615`) returned `VALIDATOR_REVIEW_FINDINGS` (2 MAJOR, 2 MINOR, 3 LOW, 0 BLOCKER; review record `.tmp/session/arch-freeze/validator-review3/REVIEW.md`, session record). This ticket fixes F1-F4 and F6; F5 is documented and left unchanged; F7 is out of scope. A second round (review of PR #653, findings MAJOR-1, MINOR-1, MINOR-2, LOW-1, LOW-2) adjusts the F2 trigger and edge list and this text. It certifies nothing, promotes no FRZ criterion and changes no stage status.

## 2. Scope

| Finding | Severity | Change | Where |
| --- | --- | --- | --- |
| F1 | MAJOR | A malformed task ID is an error and never dropped silently: every token of a `DEPENDENCIES` cell must be a well-formed `T-` or `PT-` ID, every row of the WBS task table must have a well-formed `TASK_ID` and at least 14 columns, and the task-ID regex uses a lookbehind so an ID glued to a prefix (`PRE-T-B5-X`) is not read as `T-B5-X` | `_parse_dependencies`, `_parse_wbs`, `_TASK_ID_RE` |
| F2 | MAJOR | The plan C.3 new edges are rules (reachability, like the hard rules): `LOCK` follows `T-AF2-ENT-BASELINE-PREFREEZE`, `T-AF2-ECAP-REVISION`, `T-AF2-MANIFESTS`, `PT-Q11`; all 13 `T-B7-PROMOTE-*` and `T-AF2-REQUAL-HIST-PASS` follow `T-B7-ECAP-2C`; `COORD` follows `REQUAL-HIST-PASS`; `T-AF2-POST-BASELINE` follows `PT-Q13` and `COORD`; `T-AF2-W8-INDEP-CONFIRM` follows `POST-BASELINE` and `PFP-VERIFY`; `FRZ-FRZ` follows `W8-INDEP-CONFIRM`; `T-AF2-SSOT-RECONCILE-B` and `T-AF2-FRZ-MATRIX` follow `PT-D05` (26 edges). ROADMAP <- FRZ-FRZ, ECAP-16 <- ROADMAP and VBN-HEAD not on LOCK are covered by the existing hard rules. The rules are active as soon as any one regenerated row exists (the three marker rows, the four new LOCK prerequisites, `PT-Q13`, `PT-D05`, `T-AF2-SSOT-RECONCILE-B`, `T-AF2-FRZ-MATRIX`), so the current 54-row WBS stays valid; once active, all eleven regenerated rows must exist. One self-test mutation per edge, plus one per missing row, and a post-DAG-REGEN reference graph | `_validate_post_regen_rules`, `_post_regen_reference_graph`, `_post_regen_self_test_cases` |
| F3 | MINOR | Every `V21_QUALIFICATION` state must be one of the section C states (a typo, a lower-case state or a missing cell is an error instead of skipping the `authority_ref` check); per-state counts are reported | `_check_qualification_states`, `QUALIFICATION_STATES` |
| F4 | MINOR | The matrix-vs-checklist cross-check always runs (T-AF2-SSOT-RECONCILE-B is merged); there is no header flag that can be absent or deleted, and the printed counter reads `matrix_crosscheck=ENABLED` | `collect` |
| F6 | LOW | An `authority_ref` must contain an identifier-like token (six or more alphanumerics, or a path); `NO`, `NOT YET`, `SEE BELOW` join the denylist | `_is_placeholder_ref` |
| F5 | LOW | Documented, not changed: `MIN_QUAL_RUNS = 16` is an intentional append-only floor (a deleted run fails); the qualification plan declares no count of its own, so a computed comparison would need a document edit outside this write set | this plan |

## 3. Explicitly out of scope

- F7 (two documents still describing `--sync-matrix` as live): `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` and `PLAN_T-IGX-050-D.md` are outside this write set (the second is another ticket's plan); a documentation ticket is needed.
- Adding the `MATRIX_CHECKLIST_CROSSCHECK` header line to the matrix: no longer needed after F4.
- Edits of the WBS, the matrix, the gap register, the checklist or any other document (the validator only reads them); DAG-REGEN itself; the `OD-AF-16b` / `OD-AF-17` inputs of `T-AF2-POST-BASELINE` (decision IDs, not WBS tasks).
- The IIAC material scripts, product paths, the roadmap, `.github/`, GitHub settings; any re-run that counts as qualification; any PASS or certification statement.

## 4. Write set (the only paths the PR may touch)

```text
scripts/governance/verify_arch_freeze_plan_traceability.py
tests/unit/governance/test_arch_freeze_plan_traceability.py
governance/tickets/T-IGX-050-L.md
governance/tickets/plans/PLAN_T-IGX-050-L.md
```

## 5. Method

1. Worktree outside the repository at `<parent>/_wt/intergrax/validator-fix`, branch `af2/validator-fix` cut from a freshly fetched `origin/development`, as granted by OD-AF-02.
2. Tests first: each fixed finding gets tests that fail on the unfixed validator and pass on the fixed one (one red run of the new tests against the unfixed behaviour, one green run of the whole module).
3. Commit without any AI-tool attribution, with the trailer `Ref: governance/tickets/T-IGX-050-L.md`. Push with the upstream set first and an explicit refspec; open a DRAFT pull request against `development`.

## 6. Test contract

One `uv run pytest` process per step under the shared lock (red run, green run), one validator run, one ruff run; no broad suites.

## 7. Risks

- The post-DAG-REGEN rules are reachability rules: a dropped direct edge is still accepted when the prerequisite stays reachable through another path (same semantics as the hard rules).
- The rules switch on when any one of the eleven regenerated rows exists. DAG-REGEN must add all eleven rows in one change, otherwise the validator reports the missing ones. This includes the mapped `PT-Q11`, `PT-Q13` and `PT-D05` rows: adding one of them alone is reported until the rest exist.
- The F3 state check applies to every WBS row, including the rows DAG-REGEN adds: each added row must carry a section C `V21_QUALIFICATION` state, or the validator reports it.
- The VBN-HEAD rule (`T-B7-VBN-HEAD-DOSSIER` not on LOCK) applies only while `T-AF2-VBN-DOSSIER` stays the same node as `T-B7-VBN-HEAD-DOSSIER`; DAG-REGEN must not split them.
- A direct edge is not required when a transitive path exists; DAG-REGEN may rewire as long as reachability holds.
- The always-on cross-check turns a future matrix/checklist drift red immediately; the current state was checked green.
- The validator remains a certifier input: its green run is not proof that the regenerated DAG is correct until a re-review of the seed set (review section 6) is done.
