# AUDIT_REPORT - T-IGX-031-A, S12 (lint normalization, Amendment A1), delta auditor `aud-s12`

Auditor: fresh session (not aud-tests, planner, reviewer, approver, implementer or MASTER). Pinned detached checkout `D:/Projekty/_wt/intergrax/aud-s12` @ `47c29b4c35e46439b86916405557ac0488344320`. All runs are my own, on that SHA, through `uv run --extra dev-ci`, under the heavy lock (one process at a time; lock released at the end).

Read scope: brief, implementer handoff, baseline `aud-tests/AUDIT_REPORT.md`, plan sections S12 / step 12 / R8 / R9, full `git diff 217dd739e 47c29b4c3`, `scripts/ci/run_ruff_changed_paths.py`, the relevant lines of `test_sec_ent.py` and w5_c.
Edit scope: none in the repo. Scratch only: `neg_wrapper.py` in my scratchpad (not in the repo) and logs in `audits/aud-s12/`. No commit, push, merge, rebase, reset, amend. End state: `git status --short` empty, HEAD = 47c29b4c3.
Tests run: ruff check / format --check, the step 12 (b) AST script (positive and negative), pytest steps 1-5. Not run, as ordered: 27-path run, probes A/B.

## VERDICT: PASS_SCOPED

S12 is a behaviour-neutral lint normalization of exactly the four planned files. It fixes the base lint debt without touching any assertion, test body, fixture, marker, `no_ci`, skip or xfail. Steps 1-5 are green with the same counts as the audited tree. Not covered: residuals R8 and R9 (below) and anything outside the five files.

## Per-step evidence

| # | Check | Result | Log |
|---|---|---|---|
| 1 scope | `git diff --name-only 217dd739e 47c29b4c3` | exactly the 4 plan files (w5_b2, w5_c, w5_d, test_sec_ent.py); +17 / -32; nothing under intergrax/ agents/ applications/ scripts/, no helper/ticket/plan/conftest/pyproject | - |
| 1 parent | `git diff --name-only b28398026 217dd739e -- tests/` | empty: the parent carries the audited test content (the parent adds only docs/governance records). The AST script re-checks this for the five files | - |
| 1 digest | `plan_digest.py` of the plan at S12 | `dc0fcf736456534e923000eaf4cce1d569edee9db28ef73386a5bd8284b3cc6b` = amended digest in the brief | - |
| 1 hygiene | `git diff --check 217dd739e 47c29b4c3` | clean; `grep` of added lines for `noqa|skip|xfail|mark`: no hit | - |
| 3 AST (real) | plan step 12 (b) script, extracted by me verbatim from the plan (byte-equal to the implementer's copy), `python -I s12_equiv.py 47c29b4c3...` from the repo | `PARENT 217dd739e...`, `S12 EQUIVALENT`, exit 0 (all `ALLOWED_NOQA_DELTA` = 0) | ast_real.log |
| 3 AST (negative) | same script, unmodified, in a wrapper that returns w5_d's S12 blob with one assertion flipped (`assert sink_a is not sink_b` -> `is`), in memory only | `FAIL tree without imports differs: ...w5_d...`, `S12 NOT EQUIVALENT (1)`, exit 1 | ast_negative.log |
| 4 ruff | `uv run --extra dev-ci ruff check` on the five files (ruff 0.15.20, the locked one) | `All checks passed!`, exit 0 | ruff_check.log, ruff_version.log |
| 4 ruff | `uv run --extra dev-ci ruff format --check` on the five files | `5 files already formatted`, exit 0 | ruff_format.log |
| 5 step 1 | pytest SE | 14 passed, exit 0 (67.7 s) | step1_se.log |
| 5 step 2 | pytest B2 C D | 20 passed, exit 0 | step2_w5.log |
| 5 step 3 | pytest SE B2 C D, one process | 34 passed, exit 0 | step3_isoA.log |
| 5 step 4 | pytest D C B2 SE, one process | 34 passed, exit 0 | step4_isoB.log |
| 5 step 5 | single node `test_security_spine_counters_increment` | 1 passed, exit 0 | step5_node.log |

All five pytest processes exited normally after their summary. No hang, so the AUD-1 class did not occur here.

Note on the negative run: my first attempt (a scratch clone with a flipped assertion committed there) was refused by the harness, and I did not work around it. The in-memory wrapper exercises the same script and the same `git show AFTER:path` read, so it shows the AST check catches a changed assertion. It does not exercise the script's git plumbing on a real second commit.

## Hunk classes (every hunk of 47c29b4c3, read in full)

| Class | Hunks | Files |
|---|---|---|
| A. I001: the blank line after `import pytest` removed (intergrax / testing_support / tests.support imports now follow in the same block) | 1 per file | all 4 |
| B. I001: `lab_application.host.settings` and `lab_application.manifest` moved from before `testing_support` / `tests.support` to a trailing block after them (module level, still in the header, above `pytestmark`) | 1 per file | w5_b2, w5_c, w5_d |
| C. I001: names inside a list reordered, same module (`ExporterKind, ExportError`; `ObservabilityExportSinkFactory, OtlpEventExportSink`) | 2 | w5_d |
| D. The single declared F401 deletion: function-local `from intergrax.runtime.events.event_bus import RuntimeEventBus` in `test_defense_middleware_blocks_cross_tenant_scope` | 1 | test_sec_ent.py |
| E. The declared E402 fix by MOVE: late `from intergrax.contracts.event_delivery import ObservabilityExportPayload` deleted and merged into the existing header statement `... import EventDeliveryPolicy, ObservabilityExportPayload` (no `noqa`) | 2 (merge + deletion) | w5_c |
| F. ruff format: line joins of one signature / call / def, no token changed | w5_c 1, w5_d 3 | w5_c, w5_d |

Nothing else. No hunk touches an assertion, a fixture, a marker, `pytestmark`, a decorator or a function body other than class D.

## Moved imports, read one by one

- Class B: module-level before and after; the moved group stays in the header, above `pytestmark` and every executable statement. The only change is the relative order of {`testing_support`, `tests.support`} and {`lab_application`}. No `sys.path`, `register_*` or bootstrap call sits between them (an import-time side effect is therefore only possible through transitive imports = R9).
- Class C: names of one module, no semantics.
- Class D: the name is NOT used in that function (the body builds `PluginSecurityDefenseMiddleware`, `ToolCallHookPayload`, and so on). The other `RuntimeEventBus` imports in the file (L246, L288) are function-local in other functions and untouched. The blank line after the `async def` line remains, as the plan says, and `ruff format --check` accepts it.
- Class E: `ObservabilityExportPayload` appears once besides the import, as the annotation of `_FailingExportSink.export` (L99), and the file has `from __future__ import annotations` (L5), so the annotation is never evaluated at run time. The same module is already imported in the header statement. Scope kept (module level, header). The AST script confirms the move lands at the position of the `EventDeliveryPolicy` import.

## Pre-commit hook (no `--no-verify`)

`core.hooksPath` is `.githooks`. The implementer's `impl-s12/commit.log` contains the hook output (`[INFO] pre-commit ruff: 4 file(s)`, `All checks passed!`, `4 files already formatted`, the diff-scoped gate and `[OK] diff-scoped gates passed.`). The commit message carries no `--no-verify` mention. A commit object cannot prove the flag was not used; the hook output in the log is consistent with a normal commit and nothing contradicts it. There is no `--no-verify` authorization for S12, and none is needed.

## Pre-push gate (`scripts/ci/run_ruff_changed_paths.py`)

`origin/development` (af323c005) is an ancestor of S12. `git diff --name-only --diff-filter=d origin/development 47c29b4c3 -- '*.py'` lists exactly five files: the helper `tests/support/integration_catalog_isolation.py` and the four S12 files. Those are the files I ran with the locked ruff (`ruff check`, then `ruff format --check`, both exit 0). By construction the gate runs the same two commands on those paths, so it would pass for the branch diff vs origin/development. Not verified: the gate script itself was not executed, and the plain `uv run ruff` (without `--extra dev-ci`) that the script calls was not run (the harness refused my attempt); the gate's real `push_paths()` base, and the `extend-exclude` handling, were read but not run.

## Findings

| ID | Sev | Finding | Fix |
|---|---|---|---|
| none blocking | - | Nothing has to be corrected before merge on the strength of this audit. | - |
| S12-1 | Info | Class B puts `lab_application` after `tests.support`, a different grouping than the manual order the files had before (AUD-3 of the baseline). Ruff's own classification (first-party vs the rest) decides it, and the locked ruff accepts it. Not a defect. | none |
| S12-2 | Info | The harness refused my scratch-clone negative test and a direct run of plain `uv run ruff`; the substitutes are described above. | none |

## Honesty of the handoff

Every checkable claim held: scope (4 files, +17/-32), the single F401 deletion, the E402 move without `noqa`, ruff clean on five files, `S12 EQUIVALENT`, 14 / 20 / 34 / 34 / 1 passed, no hang, nothing pushed. The handoff correctly lists what was not run and claims no PASS. The CRLF-working-tree / LF-blob note is plausible and does not affect the committed diff (`git diff --check` clean). I did not verify the claim about the CRLF hand edit beyond the committed blobs.

## Residuals

- R8 (27-path run not repeated): stands as stated in the plan. My evidence narrows it: all test bodies are AST-identical, the 34-test combined runs pass in both orders, and no file outside the four changed. It is not closed; closing it needs a new Operator authorization for another 27-path run. Per the brief I did not run it.
- R9 (transitive import order of the 62 modules brought in by the reorder of class B): stays open for the untraceable list (function-level imports at import time, `importlib`, third-party side effects, `conftest.py`). The empirical cover is steps 3 and 4 (both orders, 34 passed). I found no sign of a difference. I did not trace the import graph myself.

## What I could not verify

- That the S12 commit was made without `--no-verify` beyond the hook output in the implementer's log.
- The pre-push gate script end to end, and plain `uv run ruff` (see above).
- R8 and R9 beyond the evidence above; the 27-path result at S12; CI behaviour (`no_ci`); the AST script's behaviour on a real second scratch commit.
