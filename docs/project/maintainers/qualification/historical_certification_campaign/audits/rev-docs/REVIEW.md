# REVIEW rev-docs - branch hist-cert/docs-patches (worktree D:\Projekty\_wt\intergrax\int-docs), base 15347255a

Reviewer: rev-docs (read-only, non-author). No repository file was edited, committed or pushed.
Overall verdict: **ACCEPT_WITH_CHANGES** (3 of 4 commits need one edit each; no commit needs to be dropped).

| # | Commit | Campaign id | Verdict |
|---|--------|-------------|---------|
| 1 | 42a5dda0d | HIST-04 | **ACCEPT_WITH_CHANGES** (roadmap row flips a stage label; SHA ladder edit is clean) |
| 2 | 6d85b4d0e | HIST-02 | **ACCEPT** (one optional wording nit) |
| 3 | 7c8de550c | HIST-03 | **ACCEPT_WITH_CHANGES** (FRZ-OBS-04 note quotes wording that is not in that row) |
| 4 | 89cbab394 | HIST-01 | **ACCEPT** |

Diff vs base: 8 files, +15/-10. Only typographic non-ASCII (em dash, ellipsis, middle dot, arrows, section sign, not-equal); no Polish or other non-English text added. No CR characters added.

## Commit 1 - 42a5dda0d (HIST-04) - ACCEPT_WITH_CHANGES

| Check | Evidence | Result |
|-------|----------|--------|
| B6_ACCEPTANCE_PUBLICATION_SHA `1dd76933...` | `git cat-file -t` = commit; `merge-base --is-ancestor ... 15347255a` ok; SHA_LADDER L18 | OK |
| FORMAL_CLOSURE_PUBLICATION_SHA `be77b3a5...` | commit, ancestor of base; SHA_LADDER L26 | OK |
| W7_06_HANDOFF_PUBLICATION_SHA (named only) | SHA_LADDER L27 `7a4f279f...`, ancestor of base | OK |
| effective_from_sha `f8c29fb8...` (unchanged text) | commit | OK |
| OD-EVOL-12 / OD-EVOLX-08 signed by delegated sessions | SHA_LADDER L25; CLOSURE_EVIDENCE_INDEX L9,15,16 | OK |
| FRZ-EVOL-01..10 "promoted per certified dispositions"; FRZ-EVOL-07 qualification-layer only, LIM-05 | Checklist L273-282 (PASS / PASS-SCOPED); L279 LIM-05 | OK |
| SHA ladder L81 OD-EVOL-18 rewrite | names three ladder keys that all exist at L18/L26/L27 | OK (correct, in-scope) |
| Stage label unchanged | **FAILS.** Last cell of roadmap row 489 (EVOL-X) at base: `**[ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY** - **INTERNALLY CLOSED - AUTOMATED INTERNAL AUDIT ACCEPTED (SCOPED)**`. At HEAD: `**[x] CLOSED - INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED) - PUBLISHED** (...) - earlier internal closure @ 2026-11-05 is historical`. (`awk -F'|'` on line 489, field NF-1, base vs HEAD.) | DEFECT |

Finding 1.1 (scope/status, blocking): the commit flips the stage cell from `[ ]` / PENDING_AUTHORITY to `[x]` / CLOSED - INDEPENDENTLY ACCEPTED. This is a stage-label change, in the class the campaign excluded. The commit message ("reconcile stale ... PENDING text", "No FRZ status changes") does not disclose it, and the body of the same commit shows only the long prose cell, so it is easy to miss. The new label is supported by the repo: the same roadmap already carries it at L561, L563, L622 and L2065, and SHA_LADDER L25/L26 carries it. So this is not a false claim. It is a label change made outside the authorised class and must not ride in a "stale text" commit.

Finding 1.2 (consistency): after reverting 1.1, the row's prose says CLOSED while its status cell says PENDING_AUTHORITY, so the row stays internally inconsistent. That is the pre-existing state at base (the prose was the stale part) and is acceptable for a non-status commit. The cell change should go to the Operator/SSOT lane as its own explicitly labelled change (the roadmap already shows the Operator-accepted CLOSED elsewhere).

Finding 1.3 (wording, minor): "Formal closure (superseded wording above is historical):" - "above" has no clear referent inside the cell. Suggest "(earlier PENDING wording is historical)".

## Commit 2 - 6d85b4d0e (HIST-02) - ACCEPT

| Check | Evidence | Result |
|-------|----------|--------|
| `5a87046...` | `git log -1` = "feat(harness): type observability transport and recertify W5"; ancestor of base | OK |
| Gate path | `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py` exists; asserts param `ObservabilityExportPayload` (L138) and rejects `RuntimeEvent` (L150) | OK |
| QUAL-X-FINAL row `CW-HARNESS-W5-BND-06` | `_qual_x_final_invariant_coverage_catalog.py` L141, `frz_criterion="FRZ-BND-06"`, same test_gate | OK |
| "QUAL-X-P5 F01" | `_p5_freeze_selection_catalog.py` L119-135: `QUAL-X-P5-F01-ARCH-BOUNDARIES` lists FRZ-BND-06 and `_HARNESS_W5` module (L33-35) | OK |
| Scope limiter (observability exporters only; other transports not covered) | matches row text of the catalog (OTLP/collector/otlp_event_export_sink only) | OK |
| GR-11 `4e5878f5...` | commit; roadmap L1903 `GOV-X1-GR11`; GOV_X1 doc L13 "GR-11 CLOSED / ACCEPTED" | OK |
| "nine-row" `GR11-*` catalog | `tests/qualification/governance/gr11/catalog.py`: 9 ids (ROOT-ADMISSION ... RELIABILITY-OBSERVATION-EVIDENCE) | OK |
| "former label G11-17 matches no catalog row id" | `grep G11-17` over tests/docs: only the new quote itself | OK |
| Status | FRZ-BND-06 and FRZ-PLG-06 stay `OPEN`; both notes say "not global PASS" | OK |

Optional nit 2.1: for FRZ-PLG-06 the old text "extension package != activation/authorization" is kept and narrowed to "Governance-relevant plugins" - this narrows, not widens, the claim. Fine.

## Commit 3 - 7c8de550c (HIST-03) - ACCEPT_WITH_CHANGES

| Check | Evidence | Result |
|-------|----------|--------|
| FRZ-OBS-04..07 status unchanged (OPEN) | diff touches only the evidence cell; status cell still `OPEN` | OK |
| TRACE-X rows cited exist and are PASS | Checklist L139 TRC-01, L140 TRC-02, L141 TRC-03, L144 TRC-06, L148 TRC-10 (TRACE-X-P6), L149 TRC-11, L150 TRC-12 | OK |
| "superseded by TRACE-X closure" | roadmap L391 TRACE-X `[x] CLOSED / independently accepted` | OK |
| Quoted phrase "remains TRACE-X / CTRL-X" present in each row | OBS-05, OBS-06, OBS-07: yes. **OBS-04 at base: `grep -c 'remains TRACE'` = 0.** Its row ends "Parent-scope contribution only." | **DEFECT** (OBS-04) |
| PROD_Q_PLATFORM L187 path prefixes | the six cited files exist: `applications/local_workspace_application/hosting/readiness.py`, `.../serving/readiness_routes.py`, `.../tests/hosting/test_hosted_readiness.py`, `.../tests/host/test_lkw_fastapi_lifecycle.py`, `intergrax/hosting/engine/components.py`, `health.py` (`git ls-files`) | OK |
| `aacf7f0` (APP-HOST-8B) | `feat(lkw): propagate degraded state in APP-HOST-8B readiness projection` | OK (unchanged text) |
| ROADMAP_REPLAY_X note: `a2ed11f0`, 95 PASS / 130 OPEN | doc L3 AUDITED_HEAD a2ed11f0e811...; metrics table PASS 95 / OPEN 130 | OK |
| Path B section name | Checklist L57 "Path B - architecture-scoped vs live qualification acceptance" | OK |
| E.1 annotation (P3 package): "NOT EXECUTED" at header table, next-stage row, item 2 | P3 package L40, L58, L222 | OK |
| E.1 CLOSED claim, 21 modules / 209 tests, no monolithic artifact, does not close PROD-Q-R1-R7-Q / PROD-Q-R1 / PROD-Q | roadmap L2047 (`PROD-Q-R1-R7-Q-EXEC-E1`) | OK |
| P3 closure scope unchanged | annotation only; no status text edited; scope statements restate existing ruling, add none | OK |

Finding 3.1 (blocking, small): the FRZ-OBS-04 note says "the stale 'remains TRACE-X / CTRL-X' wording above is superseded" but FRZ-OBS-04 contains no such wording. A quoted phrase that is not in the row is a citation error in a record document.

Finding 3.2 (wording): in OBS-05/06/07 the note says the "remains TRACE-X / CTRL-X" wording is superseded "by TRACE-X closure". The phrase also names CTRL-X, whose closure the note does not address. Narrow the claim to what the evidence supports (the TRACE-X part).

Finding 3.3 (observation, no edit required): the TRC-row-to-OBS-row pairings (e.g. TRC-12 for OBS-04, TRC-03 for OBS-06) are the author's judgement, not a repo-recorded mapping. The notes say "not a mapping acceptance" and keep OPEN, which is adequate. Final mapping belongs to the independent audit.

## Commit 4 - 89cbab394 (HIST-01) - ACCEPT

| Check | Evidence | Result |
|-------|----------|--------|
| Old P0 SHA `76a98cb3...` was wrong | `git log -1` = "docs(qualification): publish SKILL-X parent independent acceptance" | confirmed wrong |
| New `f5c93309...` | `git log --diff-filter=A -- TOOL_LIFE_X_P0_CANONICAL_LIFECYCLE_CLOSED_WORLD_PRE_AUDIT.md` = f5c933097 "TOOL-LIFE-X P0 canonical lifecycle closed-world pre-audit"; the only commit touching the file | OK |
| Other `76a98cb3` occurrences in that file | none left | OK |
| QUAL-X "35 modules at `d9ea0d6e8`" | executed `qual_x_p5_dimension_a_pytest_module_targets()` from `git archive d9ea0d6e8` of `_p5_freeze_selection_catalog.py` (scratch dir, outside repo): returned 35 | OK |
| "header erratum F-04" | base header (line 3) lists F-04 ("32 modules" vs 35 at d9ea0d6e8) | OK |
| R2 record unchanged | addition is a separate blockquote; existing "32 modules" lines (L1048, L1584) untouched | OK |

## Cross-commit consistency
- Commits 2 and 3 edit disjoint rows of the same checklist (L85, L108 vs L135-138); no overlap or merge conflict; commit 3 applies on top of commit 2 (index shows 5356c6d6b -> 40912fe04).
- Notes added by commits 2/3 cite `5a87046...` in the same truncated style as the existing text.
- No stale cross-reference introduced, except the row-489 label issue (finding 1.1/1.2).
- Roadmap ladder `EVOL_X_SHA_LADDER.md` reference in commit 1 resolves (file exists, keys exist).

## Gate remarks (item 5)
The diff-scoped gates would not catch: (a) a stage-label flip inside a long table cell (finding 1.1); (b) a quoted phrase absent from its row (finding 3.1); (c) a wrong-but-valid SHA (commit 4 was exactly that class: the old SHA existed but belonged to another commit). Only `git log --diff-filter=A` style checks catch (c).

## Required edits (one consolidated batch)

1. `docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`, row `| **EVOL-X** |` (L489), last cell. Restore the base text:
   - old (HEAD): `**[x] CLOSED — INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED) — PUBLISHED** (IIAC B6 acceptance and OD-EVOLX-08; operator signatures delegated, `process_enforced`; SHAs in `EVOL_X_SHA_LADDER.md`) · earlier internal closure @ 2026-11-05 is historical`
   - new (base): `**[ ] FORMAL PROTOCOL 03 B6 — PENDING_AUTHORITY** · **INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED (SCOPED)**`
   - Keep the prose-cell edit. Route the label change to the Operator as a separate, explicitly labelled commit (L561/L563/L622/L2065 already state CLOSED). Alternative if the Operator expressly approves the label now: keep it but state it in the commit message ("changes the EVOL-X stage cell from PENDING_AUTHORITY to CLOSED per roadmap L561/L2065, Operator-authorised").
2. Same row, prose cell: `**Formal closure (superseded wording above is historical):**` -> `**Formal closure (earlier PENDING wording is historical):**`.
3. `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`, row FRZ-OBS-04, note: replace `the stale "remains TRACE-X / CTRL-X" wording above is superseded by TRACE-X closure; TRACE-X evidence bearing on this row:` with `TRACE-X evidence bearing on this row (TRACE-X is closed; this row's "Parent-scope contribution only" statement is unchanged):`.
4. Same file, rows FRZ-OBS-05, -06, -07 notes: replace `is superseded by TRACE-X closure;` with `is superseded, as far as it concerns TRACE-X, by TRACE-X closure (the CTRL-X part is not addressed by this note);`.
5. Commit messages cannot be edited without a rewrite; instead record in the merge/PR description that commit 1 carries the stage-cell revert (edit 1) and that the OBS notes are non-status.

After edits 1-4: re-run the pre-commit gates; no further review needed for commits 2 and 4.

## Not verified
- Campaign evidence diffs (`hist01..hist04/*.diff`, finding ids F-001, F-002, F-008, F-04) are outside the repository; I verified only the claims they produce, not the finding numbering. F-04 is visible in the QUAL-X header at base.
- Whether CTRL-X is closed (not needed once edit 4 is applied).
- Pre-commit gate results (not available to me; gates not re-run).
- Content of `iiac_evidence/EVOL-X/...` decision records (existence of OD-EVOL-12 / OD-EVOLX-08 signatures taken from SHA_LADDER and CLOSURE_EVIDENCE_INDEX, not from the signed records themselves).

---

## Delta re-review (correction commit 6d1434796; net branch HEAD vs base 15347255a)

Method: `git show 6d1434796`; a cell-by-cell comparison script (scratchpad, read-only) that diffs every changed line of the net diff `15347255a..HEAD` by table column. Result: 8 files, +15/-10 (unchanged vs the previous review); the correction commit itself is 2 files, +5/-5.

| # | Check | Evidence | Result |
|---|-------|----------|--------|
| 1 | Roadmap L489 stage cell equals base; only the prose cell differs | Net diff row EVOL-X: 6 columns at base and HEAD, changed column index 3 only (description). Stage cell at HEAD = `**[ ] FORMAL PROTOCOL 03 B6 — PENDING_AUTHORITY** · **INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED (SCOPED)**`, identical to base. Edit 2 applied: `Formal closure (earlier PENDING wording is historical):` | PASS |
| 2 | FRZ-OBS-04 no longer quotes an absent phrase; OBS-05..07 narrowed to TRACE-X | OBS-04 note now: "TRACE-X evidence bearing on this row (TRACE-X is closed; this row's "Parent-scope contribution only" statement is unchanged)" (that phrase is in the row). OBS-05/06/07: "superseded, as far as it concerns TRACE-X, by TRACE-X closure (the CTRL-X part is not addressed by this note)" | PASS |
| 3 | No FRZ status or stage label changed in the net diff | Every changed table row: FRZ-BND-06, FRZ-PLG-06, FRZ-OBS-04..07 changed only the evidence column (index 6 of 8; status column index 5 = `OPEN` unchanged); PROD-Q "Degraded operation" row changed code/test columns only (4, 5; status column unchanged); TOOL-LIFE P0 row changed the SHA column only; EVOL-X row see check 1. Remaining changes are non-table additions (P3 annotation, QUAL-X note, ROADMAP_REPLAY_X note) and the SHA-ladder OD-EVOL-18 bullet. No `OPEN`/`PASS`/`CLOSED` label altered | PASS |
| 4 | Nothing else changed in the commit | `git diff 6d1434796~1 6d1434796 --stat`: only the roadmap (1 line) and the checklist (4 lines: OBS-04..07); commit message states E1-E4 and nothing else | PASS |
| 5 | English only | Added lines contain only U+2014, U+2026, U+00A7, U+00B7; no Polish or other non-English text | PASS |

Residual (not a defect, as noted in the first review): the EVOL-X row prose says CLOSED while its stage cell says PENDING_AUTHORITY; this is the base-state inconsistency, to be resolved by the Operator in a separate labelled change.

### Final verdicts per commit (branch state at 6d1434796)

| Commit | Verdict |
|--------|---------|
| 42a5dda0d (HIST-04) | **ACCEPT** (stage-cell flip reverted by 6d1434796; SHA ladder edit clean) |
| 6d85b4d0e (HIST-02) | **ACCEPT** |
| 7c8de550c (HIST-03) | **ACCEPT** (OBS notes corrected by 6d1434796) |
| 89cbab394 (HIST-01) | **ACCEPT** |
| 6d1434796 (correction) | **ACCEPT** |

Overall branch verdict: **ACCEPT**. Not verified: pre-commit gate results for 6d1434796 (not available to me).
