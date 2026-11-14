# PDC-X P6 acceptance record (T-IGX-036-W, Part A)

Status: recorded by a TESTER session. Nothing was repaired, nothing was certified. **This record carries no PASS verdict.**

Ticket: `governance/tickets/T-IGX-036-W.md` (plan `governance/tickets/plans/PLAN_T-IGX-036-W.md`, Part A, deliverables 1-3).

## Identity

| Item | Value |
| --- | --- |
| Run date | 2026-11-14 (UTC) |
| Worktree | `D:/Projekty/_wt/intergrax/pdc-acc`, branch `pdc-x-w-acc`, created from `origin/development` |
| HEAD under test | `884c177adc366004929641d029bba8d129b0ecc8` (merge of PR #531) |
| INTEGRATED_BASELINE_SHA | `a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73` |
| Product tree check | `git diff --name-only a156cbbd5..HEAD -- intergrax agents applications` returned 0 paths (product tree equals the baseline) |
| Tree state | clean before and after every run (`git status --short` empty; logs are under the gitignored `.tmp/`) |
| Python / uv | CPython 3.12.11, uv 0.13.4 |

Environment: `uv sync --extra dev-ci --frozen` once (55 s, log `00_uv_sync.log`; the same sync CI uses) so that `duckdb`, `celery` and `tzdata` are present. `uv run` was used for every run. No `-n`, no `-p no:cacheprovider`, one process at a time, nothing in parallel.

Logs: `.tmp/proof/pdc-x-p6/acceptance/` in the worktree (gitignored, not committed): `00_uv_sync.log`, `01_ecap.log`, `ecap/` (report), `02_pdcx.log`, `03_gates.log`, plus `*.start` / `*.end` UTC markers.

## Run 1: ECAP FULL

```text
uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL --output-dir .tmp/proof/pdc-x-p6/acceptance/ecap
```

`--output-dir` was added to keep the report out of the repo tree (the plan's command has no output directory). Run first, on the clean tree, under `timeout 5400`.

| Item | Value |
| --- | --- |
| Duration | 23 m 24 s (start 06:11:50Z) |
| Status / exit | BLOCKED, exit 1 |
| Checks executed / skipped / planned | 85 / 0 / 85 |
| Checks failed | 1 |
| Report | `.tmp/proof/pdc-x-p6/acceptance/ecap/enterprise_audit_report.md` and `.json`, `enterprise_audit_outcomes.json` |

Findings (1):

- `VBN-REACHABILITY:FAIL` (severity ERROR, new vs. the ECAP baseline `480550d80`), gate file `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py`: 2 failed, 84 passed. Failing tests: `test_ebh5_pkgc_v2_dual_slug_integration[integration.embedding_provider]` and `test_ebh5_pkgc_v2_dual_slug_rpl_cde_equivalence[integration.embedding_provider]`. Cause in the log: `UnknownIntegrationError: Integration slug 'ollama' is not registered in the catalog` (`intergrax/integrations/registry/catalog.py:136`). Classified below (R-04).

The six ECAP-F1 checks and the ECAP-F2 check reported by the B-P6 report of T-IGX-036-S (contract-surface purity, SQLite touchpoint registry) did not fail in this run. The four `duckdb` cases of ECAP-F3 did not fail either (environment synced with `dev-ci`).

## Run 2: `tests/qualification/pdc_x`

```text
uv run pytest tests/qualification/pdc_x -q -rf --tb=short
```

Under `timeout 7200` (120 min hard limit).

| Item | Value |
| --- | --- |
| Pytest duration | 171.29 s (2 m 51 s), start 06:35:45Z |
| passed | 927 |
| failed | 0 |
| skipped | 5 |
| xfailed / xpassed / errors | 0 / 0 / 0 |

XD-01..12 are plain passing tests; there is no `xfail` left in `pdc_x` (the former strict xfails were flipped by T-IGX-036-T, -U and -V). The five skips are not itemised in the output (`-rf` lists failures only); they were not investigated.

Process note: after the summary line was printed the `pytest` process did not exit. Python stayed idle for about 7 minutes (CPU time constant over 10 s, 23 threads in the child, no output). The tester killed the two processes of this run at 06:46:08Z. The wrapper log therefore shows `real 10m22s` and `exit=127`, which is the kill, not a test result. The pytest summary line above is the result. The run of Run 3 exited normally. A thread left running by some `pdc_x` test is the likely cause; it was not investigated (finding for the orchestrator, see R-05).

## Run 3: DUP-X and EBH-4 gates (one command)

Gate files located in the description of PR #514 (B-P6 report of T-IGX-036-S): `tests/qualification/dup_x` (all modules) and the EBH-4 composition-root and bypass gates. All six EBH-4 files exist on HEAD.

```text
uv run pytest tests/qualification/dup_x \
  tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py \
  tests/unit/scripts/test_check_application_production_gates.py \
  tests/unit/runtime/qualification/test_ebh_4_b4_tenant_isolation.py \
  tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py \
  tests/unit/runtime/execution/test_runtime_request_tenant_envelope.py \
  tests/unit/runtime/task/test_unified_task_runner_execution_boundary.py -q -rf --tb=short
```

Under `timeout 3600`.

| Item | Value |
| --- | --- |
| Duration | 67.05 s (start 06:46:22Z), process exited normally, exit 0 |
| passed | 130 |
| failed | 0 |
| skipped / xfailed | 0 / 0 |

The count equals the 130 passed reported in the PR #514 description.

## Classified list of reds

| ID | Red | Run | Class | Note |
| --- | --- | --- | --- | --- |
| R-01 | none in `pdc_x` | 2 | n/a | 0 failures |
| R-02 | none in DUP-X / EBH-4 | 3 | n/a | 0 failures |
| R-03 | no IIAC ACT-N11, FREEZE launcher, `ebh_7`, `roadmap_replay_x`, `dist_scale_x` or T-report baseline test ran red | all | known classes (a)-(d) | those suites were not run (outside the exact runs); no red of these classes appeared |
| R-04 | ECAP `VBN-REACHABILITY`: 2 `test_ebh_5_behavioral_package_c_v2.py` cases, `ollama` slug not in the catalog | 1 | **outside known classes (a)-(d); finding for the orchestrator** | Matches the duplicate `ollama` catalog slug regression of batch B-REG-P (`T-IGX-036-M`, PDC-D1-R022, regression from `7aa9cfa18`), whose fix is deliberately held until MCP-SERVER-X P8 (OD-PDC-SEQ-01). It is in the integration catalog, not in an area changed by the PDC repair tickets T, U, V. No baseline comparison was run; the class rests on the error text and the ticket `T-IGX-036-M` text, not on a re-run at `dc18e36d9`. |
| R-05 | `pdc_x` pytest process did not exit after the summary line | 2 | **unclassified observation; finding for the orchestrator** | Not a test failure. The process was killed by the tester. Cause not investigated. |

Unknown reds (test failures outside the known classes): none beyond R-04, which has a known tracking ticket. Nothing was repaired.

## Not run (by instruction)

`tests/qualification/ebh_7`, `roadmap_replay_x`, `dist_scale_x`, the IIAC activation tests, `ent_audit_x` launcher FREEZE test and any baseline comparison. The Part B test contract (`ebh_7`, `pdc_x_requirements.py --check`) is not part of this record.

## Machine notes

Windows 11 Home, Intel Core i9-13900H (20 logical processors), 63.7 GB RAM; the machine was shared with other sessions during the runs, so durations are indicative only. `D:` had about 114 GB free before the runs. No Docker. All started processes were stopped; none is left running.

This record carries no PASS verdict.
