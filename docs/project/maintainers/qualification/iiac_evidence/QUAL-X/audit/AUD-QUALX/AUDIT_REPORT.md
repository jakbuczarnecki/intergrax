# AUDIT REPORT — AUD-QUALX (QUAL-X Population A formal closure) — AUDITOR RECORD

Auditor: `aud_qualx` (fresh session; not implementer / closer / earlier campaign worker). Date: 2026-11-07.
Authority: Operator instruction of 2026-11-07 (independent audits only). This is NOT a closure signature, acceptance decision, OD record or FRZ promotion.

## Verdict: PASS_SCOPED (chain_gate: PROCEED)

Scope = QUAL-X **Population A only**, exactly as the stage declares. Excluded by the stage's own scope: global FRZ-REG-01..10 PASS, PROD-Q Dimension B (live), Population E / ENT-AUDIT-X, ARCH-FREEZE, SKILL-X entry.

## SHAs (all fully resolved; all ancestors of base `2f6664d53e3fcd1cb6010e33506e2d22201dc9f3`)

- ACCEPTANCE_SHA `d9ea0d6e87b939e4cefb9971c6bbd4fbcaecc0ce` (2026-10-23 08:03)
- `4c5faa89aad381e293da2210ec3e83566c3a8473` conditional package (10:06), parent = d9ea0d6e8
- `0d52df6e0da10bb1ec36bcb113b54a4866b9b76f` formal closure (10:52), parent = 4c5faa89a
- Topology correction: 4c5faa89 is the *parent* of the closure commit (chain d9ea -> 4c5f -> 0d52), not a "descendant" of it. Both are doc-only descendants of d9ea.

## Answers to the mission's specific points

1. **Independent audit record for the closure commit?** None. The A1–A5 "adjudication" is a table inside closure commit 0d52df6e0 itself, 46 min after 4c5faa89, "no re-run"; all 18 QUAL-X inventory commits since 9a175177 are by one git identity. F02 of the closure mentions a "READ-ONLY audit session" with a procedural violation, but no audit report, identity or manifest exists in the repo. Prior claim "self-published" is **confirmed**. The closure itself says "READY FOR INDEPENDENT AUDIT" and "SKILL-X not authorized until independent acceptance", so the stage never claimed to have one.
2. **Doc-only descendants?** Yes. `git diff --name-status d9ea0d6e8 0d52df6e0` = 3 files, all `.md` (roadmap, FRZ checklist, QUAL-X inventory); d9ea -> 4c5f = 2 `.md`. The product/test tree at 0d52 equals the tree at d9ea.
3. **Mandatory gates reproduced at d9ea0d6e8:** 630 passed, 0 failed, 0 skipped in one pytest run (below).
4. **FRZ promotion:** none. FRZ-REG-01..10 are OPEN at d9ea, 0d52 and base. d9ea -> 0d52 changed only the prose of FRZ-REG-06 (status stays OPEN). "ALREADY PROVEN" wording is Population-A-scoped and says checklist rows stay OPEN (terminology risk only, F-05).
5. **QUAL-X-P0:** see below and F-02.
6. **Original independence:** not independent (item 1). This audit supplies the missing independent exact-SHA replay; the substance genuinely reproduces, hence PASS_SCOPED rather than CONDITIONAL.

## Key finding: the attested evidence did not cover the acceptance SHA

- The 535/535 "full Dimension A" attestation is for `9a175177` (2026-10-22). Between 9a175177 and d9ea0d6e8 there are 16 commits, including **product-code changes** in 7 files under `intergrax/runtime/` (events/store.py, nexus_loop.py, observability_wiring.py, task_events.py, journal_export.py, debug_lab_nexus_loop.py, host_observability_composition.py; BND-03 Option A), and a change to the P5 manifest (`_p5_freeze_selection_catalog.py`) that grew the Dimension A module list from 32 to 35 (added HARNESS_W5, UE_9BR1, EVIDENCE_PERSISTENCE_BOUNDARY).
- The operator-accepted "bridge" was a 49-test delta (4 modules) plus historical BND-03 logs (local `.tmp`, "70/76", limited provenance). It did not execute the changed hardening_3 gate, the P3 inventory test, test_unified_run_journal or test_observability_wiring, nor the other modules, against the changed product code. The closure's "Production: unchanged" statement in the BND-03 section holds for the last commit only, not for the 9a17..d9ea range.
- Raw clone logs of the 535 run were never retained (the operator acknowledged this). The in-doc 535 record is a doc-only commit (cc8d6e850, parent 9a1751774).
- **Resolved by this audit** via a full replay at the exact acceptance SHA.

## Test replay (independent)

- Env: detached scratch worktree `D:/Projekty/_wt/_scratch/aud-qualx` @ d9ea0d6e8 (clean), `uv sync --frozen` (uv 0.13.4). uv.lock sha256 `0ab9b58d92cb4b7a2c706223d8dbc0147b9db73c2406a8e4867a372c4d4fa7b6` equals the attested value; pyproject/uv.lock unchanged 9a17..d9ea. `INTERGRAX_RELATIONAL_DB` isolated. Heavy-test mutex held, then released.
- Command (single run): `uv run pytest -p no:xdist -W default <41 modules>` =
  - 35 modules from `qual_x_p5_dimension_a_pytest_module_targets()` at d9ea (includes the 57 mandatory collectable delegated nodes);
  - final-invariant catalog integrity;
  - 4 BND-03 modules (runtime_event_persistence, unified_run_journal incl. SQLite journal E2E, observability_wiring, nexus_loop_runtime_event_persistence);
  - the W2-C retry-containment module (REG-06 debt item; not in the Dim-A list by design).
- Result: **630 passed in 1080.39 s (18:00); no failures, no skip/xfail/xpass, no warning/error lines.** Log: `logs/pytest-d9ea.log`; targets: `logs/targets.txt`.
- A4 static check: no `RuntimeEventPersistence` import in `intergrax/runtime/execution` (only a keyword-argument name at harness_host_internal_composition.py:33). Gate `test_execution_engine_has_no_runtime_event_persistence_dependency` exists and is inside the green run.
- A1: Path B §2.0.3 is present in the roadmap and consistent with the closure text (QUAL-X predecessor = TENANT-X CLOSED; PROD-Q live slice stays OPEN/BLOCKED). The operator's original decision message is quoted in-repo but not independently available to me, so it is recorded as NOT_REPRODUCED.

## Bounded exclusions (match the stage's declared scope)

Dimension B (PROD-Q live HTTP, R1-SQLITE-ENV-01 HTTP reproduction); Populations B/C/D/E; global FRZ-REG PASS. Not performed: a fresh network `git clone` (the replay used a clean detached worktree of the exact SHA with a fresh venv); the original 9a175177 raw logs.

## Findings (see FINDINGS.json)

- F-01 MEDIUM: A1–A5 self-adjudicated, no independent record (resolved in substance by this audit; Operator signature still required).
- F-02 MEDIUM: QUAL-X-P0 still `[ ] READY FOR AUDIT`, no standalone audit.
- F-03 MEDIUM: evidence-range gap 9a175177 -> d9ea0d6e8 (closed by replay).
- F-04 LOW: 32 vs 35 modules, "70/76" ambiguity, uv.lock hash representations.
- F-05 LOW: "ALREADY PROVEN" next to OPEN rows.
- F-06 LOW: stale in-doc status text.
- F-07 INFO: topology wording.

## QUAL-X-P0 — what is still missing

Roadmap row 478 (QUAL-X-P0, baseline/gate inventory Q01–Q12) is still `[ ] READY FOR AUDIT` at 0d52 and at base. No auditor record for P0 exists; the gate "independent audit of P0 before P1" was never evidenced. Its substance (Q01–Q12 via packages P1–P5 and FINAL) is exercised by the green Dimension A run, but this audit did not re-audit the P0 inventory prose or matrix line by line. Missing: an Operator decision either (a) absorbing P0 into acceptance of this audit, or (b) commissioning a separate P0 verification. This does not block Population A PASS_SCOPED.

## Downstream

At 0d52 the roadmap rows for SKILL-X / TOOL-LIFE-X / DIST-SCALE-X read `[ ] PLANNED / MANDATORY` (not CLOSED). Any later closure of those stages on "QUAL-X = CLOSED" is outside this audit.
