# IMPL_TESTS_AUDIT_HANDOFF - T-IGX-031-A (tests-only), implementer -> independent auditor

Implementer session claims NO PASS. Lifecycle stays `plan_approved` in the ticket (not edited); state reached: implementation complete, implementer-run checks below, `ready_for_audit`.

## Implementation
- Branch `hist-cert/impl-tests`, base `origin/development` af323c005. Local commits only, nothing pushed.
- Implementation SHA (HEAD): `b28398026645093a455eeb19b2808e7998d238e2`
  - `232d03175` test(isolation): helper + W5 b2/c/d autouse fixtures
  - `b28398026` test(security): test_sec_ent.py (ledger S1-S11, S1b)
- Frozen audited scope (diff `af323c005..HEAD`, `git diff --check` clean, numstat shows no whole-file rewrite, so LF in the index):
  - `tests/support/integration_catalog_isolation.py` (new, +30)
  - `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py` (+10)
  - `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_c_event_export.py` (+9)
  - `tests/unit/runtime/events/test_enterprise_scale_resilience_w5_d_exporter_composition.py` (+9)
  - `tests/unit/runtime/security/test_sec_ent.py` (+69/-17)
  - Nothing under intergrax/ agents/ applications/ scripts/, no conftest/markers (`no_ci` kept)/pyproject/manifest/ticket/plan edit.
- Working tree is clean. Note: working-tree files on this Windows checkout are CRLF (autocrlf); I preserved CRLF in the existing files so git sees only my hunks.

## DEVIATION requiring auditor attention: commits made with `--no-verify`
The repo pre-commit ruff gate rejected the commits. Cause: defects already present at base `af323c005` (measured with `git show af323c005:<f> | ruff check --stdin-filename`): I001 in all four existing files, F401 (unused `RuntimeEventBus` in a test) in test_sec_ent.py, E402 in w5_c; `ruff format --check` already red at base for w5_c and w5_d. The gate lints whole touched files, so it cannot pass without out-of-ledger edits (import reorder, removing an import, moving a late import). I asked the operator via AskUserQuestion; the answer was "Commit with --no-verify". Both commit messages say so. I did not widen scope.

## Commands run (worktree, one process at a time under heavy.lock; logs in `D:\Projekty\intergrax\.tmp\session\hist-cert-campaign\impl-tests\`)
BEFORE (base file versions swapped in as uncommitted scratch, restored, `git diff` empty afterwards; measured at af323c005):
- SE alone: 7 failed, 6 passed, exit 1 (`before1_se.log`)
- W5 b2+c+d alone: 20 passed, exit 0 (`before2_w5.log`)
- SE then W5 (one process): 15 failed, 18 passed, exit 1 (`before3_isoA.log`)

AFTER on final tree HEAD b28398026 (`final1..5_*.log`):
- Step 1 SE: 14 passed, exit 0
- Step 2 W5: 20 passed, exit 0
- Step 3 isolation A (SE B2 C D): 34 passed, exit 0
- Step 4 isolation B (D C B2 SE): 34 passed, exit 0
- Step 5 single node `test_security_spine_counters_increment`: 1 passed, exit 0 (no registration-order fallback needed)
- (Pre-format-nit runs of steps 1-5 in `step1..5_*.log` were also green; the only later edit was one line collapse in S4 plus an import-position tweak in w5_d, then everything was rerun.)

Step 6 NEGATIVE PROBE A (uncommitted, restored, diff empty): reverted S6/S7 patch targets, S4 input to `object()`, removed S10 identity binding. Actual red set: exactly `test_resolve_encryptor_fails_closed_on_resolution_error` (S6), `test_harness_envelope_encryptor_not_selected_by_resolver` (S7), `test_resolve_encryptor_fails_closed_on_conformance_error` (S4), `test_security_spine_counters_increment` (S10); 4 failed, 10 passed, exit 1 (`probeA.log`). Matches prediction.

Step 7 NEGATIVE PROBE B (uncommitted, restored, diff empty): removed the four autouse fixtures, `_bootstrap_catalog()` body replaced by bare `register_default_integrations()`. Actual: 12 failed, 22 passed, exit 1 (`probeB.log`): SE S6, S7, S8, S9 + b2 x5, c x2, d x1. Matches prediction exactly.

Step 8 ACT-N11 pre-check (after commit): `verify_iiac_policy_activation --require-in-force` exit 0 (IN_FORCE @ IIAC-2026-11-05.1); governance tests (activation_gate, negative_guards, evol_x_real_repo_fresh_clone, g0_inventory): 65 passed, 1 skipped (`test_iiac_activation_gate.py:439` symlink creation unavailable on this host), exit 0, 619 s. The inline-helper fallback was not needed.

Step 9 ruff (`step9_ruff_check2.log`, `step9_ruff_format2.log`, base comparison `ruff_base.log`): on the five files `ruff check` = 6 findings, `ruff format --check` = 2 files, both identical to the base defects listed above (no new finding; the new helper file is clean). The plan's "clean" expectation therefore does NOT hold for these files at base; I did not fix it (out of ledger). I did fix the one format hunk I had introduced (S4 call collapsed to one line).

Step 10 diff guard: `git diff --name-only af323c005..HEAD` = exactly the five files; no intergrax/ agents/ applications/ scripts/ path vs 6734971cf either (ticket/plan files arrived with base).

Step 11 (27-path run): NOT run, per brief. Belongs to the auditor under separate Operator authorization.

## Ledger conformance notes for the auditor
- S1/S1b/S2-S11 implemented as written. S3 uses `_PrebuiltFakeSecretsStore.for_provider(provider_id="sec-ent-fake")` (Pydantic subclass with `ClassVar` store, `get_secret(path, *, version=None)` etc.); it works, so escape clause item S3 did not trigger. S10 needed no event-kind registration fallback.
- S6 uses `bootstrap_application_integration_catalog()` via `_bootstrap_catalog()`; S6/S7 patch `intergrax.applications._shared.security_runtime_bridge.resolve_from_profile` and add the `__cause__` text assertions.
- Import placement: existing files use their own manual import order that already fails I001; I placed new imports by that order (e.g. `testing_support`/`tests.support` after `lab_application` in W5 files), not by ruff's.
- Escape clause: no item left the batch; no product-code edit was needed.

## Still a thesis / not proven
- F-016 is NOT resolved by this evidence: only the minimal SE + W5 reproduction is proven. Whether other polluters exist in the 27-path selection, and whether a later file relied on a populated catalog left behind by the W5 files (clean-after leaves catalogs empty), is unknown until step 11.
- Runs on this mutable worktree are not exact-SHA clean-checkout evidence (COMMON_BRIEF 4); the auditor must re-run on the pinned final SHA.
- R1 (duplicate `ollama` slug), R2 (~72 other bare `register_default_integrations()` callers), R7 (raw `TypeError` for non-contract prebuilt store, pinned by the new test S5), `no_ci` still excludes test_sec_ent.py from GitHub CI: all untouched residuals.

## Known risks
- `--no-verify` commits (see above); the auditor should confirm no new lint beyond the base set.
- The `.venv` in this worktree was created by my first `uv run` (gitignored, expected).
- I once ran `git reset -q` in a command that the harness denied (never executed); the index state was fixed with `git restore --staged`. No reset/rebase/amend/push/stash occurred.

## What the auditor must run
1. Re-run plan section 7 steps 1-10 on the exact final SHA in a clean pinned checkout, including both negative probes and the ACT-N11 pre-check; hunk-by-hunk gate-weakening review against plan section 5 checklist.
2. Step 11 (27-path single-process run, `run1.sh` in `.tmp/session/hist-cert-campaign/hist03/`) after a separate Operator authorization; expected: no `already registered` anywhere (the 6 `p2_c_http_execution_admission` failures, F-015, may remain).
3. Decide on the `--no-verify` deviation and the base lint debt.
