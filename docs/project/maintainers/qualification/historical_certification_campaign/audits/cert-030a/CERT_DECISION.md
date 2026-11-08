# CERT_DECISION - T-IGX-030-A plan review (cert-030a)

Verdict: **APPROVE** (plan only, bound to Plan digest c6d1dca6c129e8f5ce14854698c89e998f600a8d69f11d150649e9591b18e342)
Decided: 2026-11-08T10:42:59+01:00 by cert-030a under DELEG-HIST-030A-2026-11-08 (operator-delegate; NOT an operator-written approval).
Scope of approval: PLAN of T-IGX-030-A only. Not approved/not touched: T-IGX-030-B, T-IGX-030-C (HELD), FRZ promotion, IIAC activation, any stage status, any PASS/closure.

## Evidence per check

0. Delegation integrity. CURRENT (revised, language-only) version: `sha256sum DELEG-HIST-030A-2026-11-08.md` = e17d1cb131cdbf6475b3330021ed85807979aaaab6292322805528db10a80364; equals the `.sha256` file. The revision note names the superseded first version b3e24acfe187f3770edcb4f43a785e53904d6dbd6ee3fbba2627972b2b5af184, which is the hash I had first verified and recorded; the review was started against it. After MASTER's notice I re-read the revised record in full and re-verified the hash: scope (030-A plan approval only), bound digest, expiry 2026-11-08T23:59:59+01:00, designated-session constraints and the required line form are unchanged, so the verdict and the written Approved line stand. The record is narrow; I am a new session, not impl-tlx/MASTER/HIST.
1. Plan digest. `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-030-A.md` (cwd = worktree impl-tlx) -> c6d1dca6...e342. Equals brief value, delegation value, and ticket line 18 `Plan digest:`. PASS.
2. Tests-only scope. `git diff --name-only 492e361c0 78657d445` -> only governance/tickets/T-IGX-030-{A,B,C}.md and plans/PLAN_T-IGX-030-{A,B,C}.md; grep for `^(intergrax|agents|applications)/` -> NONE. Gate semantics re-read: `iiac_common.PRODUCT_PATH_PREFIXES = ("intergrax/","agents/","applications/")`; `iiac_activation_gate._material_changes` only flags 6 listed governance/protocol scripts+doc; neither tests/ nor tickets/ trip ACT-N11. Plan files for the planned phase-2 commits (2 test files) are likewise outside both sets. PASS.
   Note: the plan commit also carries the HELD B/C tickets/plans (docs only). Both state "HELD ... do not approve or implement"; I do not approve them.
3. Mechanical integration. Base 492e361c0 vs source parents:
   - F1 tests/unit/runtime/tools/test_tools_side_effect_safety.py: base blob 05b5fccc4... == 0adb73380^ blob 05b5fccc4...; `git diff 492e361c0 30c049c12 -- F1` = 0 lines. C1 post blob 2325c4167...
   - F2 tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py: base blob c18e28cab... == 10089121e^ blob c18e28cab... (also unchanged by C1); `git diff 492e361c0 0adb73380 -- F2` = 0 lines. C2 post blob 23483824...
   - Each source commit changes exactly one test file besides its own old ticket/plan files (not carried over). No merge work. PASS.
4. Assertion ledger vs real diff (`git diff 30c049c12 0adb73380 -- F1`, `git diff 0adb73380 10089121e -- F2`):
   - #1 TIMEOUT -> ABANDONED exact + new exact `error_message == "abandoned"`; success/error/ledger UNCERTAIN/replay-raises kept. Matches ledger.
   - #2 VALIDATION_ERROR -> ABANDONED exact; `external_effects["n"]==1`, UNCERTAIN, replay raises, n==1 kept. Matches ledger.
   - #3, #4 stdlib substring -> exact code + exact message + NEW ledger `UNCERTAIN` assertion; `executor.calls == 1` kept. Matches ledger (`_`->`store` rename present; `store` already bound in #4).
   - #5 new N2 test: ABANDONED, UNCERTAIN, executor.calls==0, replay raises InvocationUncertaintyError, calls still 0. Pure addition.
   - #6 import reorder + one signature wrap: formatting only; no behaviour change (line-length 120, select E,F,I,UP in pyproject).
   - Weakening scan on added lines (skip|xfail|`in (`/`in {`|except): none. No assertion removed other than the replaced ones. No set-widening, no dropped effect-count/ledger/replay check, no swallowed failure.
   - C2: `started` asyncio.Event -> threading.Event; sleep(0.06) -> event-driven wait (8 s bound, raises TimeoutError); `step_id` added; new `invoker.results` non-empty and `_is_error_text(catalog) is None`; accepted set `{"execution_cancelled","execution_admission_denied"}` unchanged (only `, catalog_payload` message added); 2 new negative tests (child never runs; `missing_execution_binding` or `execution_cancelled`); no assertion removed; remaining hunks import/format. Matches plan section 5. The "cancel-before-admission" test accepts two outcomes but both assert `not invoker.results`; acceptable for a new test.
   - Code basis: Batch C a5f813a2c is an ancestor of base; `disposition_message` returns the enum value `"abandoned"`, `runtime_error_for_disposition(ABANDONED)` -> `RuntimeErrorCode.ABANDONED`; invoker.py uses ABANDONED disposition (L1435, L1485, L1615-1616). The code is consistent with the new assertions, but I did NOT run tests, so the runtime path for each test is unconfirmed.
5. Stale IDs. C1 blob has exactly 5 `T-IGX-023-A` comments (L518, 645, 1255, 1326, 1335), all historical comments; ID map recorded in plan and ticket (023-A->030-A C1, 023-B->030-A C2, 024-A->030-C). ACCEPTABLE: keeping them byte-identical preserves provable blob equality; they are comments only. Comment-only rename is a possible follow-up. Minor cost: a reader sees 023-A, which collides with the published EVOL-X W7-06 ticket; mitigated by the recorded map.
6. Test contract. All named test files exist at 492e361c0 (4 TLX/p0 modules, 3 IIAC governance tests, verify_iiac_policy_activation.py). One process at a time, own exit code via `; echo EXIT=$?` not via pipe, logs under .tmp/session/t-igx-030-a/, no --basetemp, reuse .venv (exists), heavy lock dir exists. Bounded: 3 consecutive F3 runs, explicit stop-and-report on ACT-N11 red. Runnable and bounded. I did not run it. PASS.
   Approval-line parsers (`verify_plan_approval._APPROVED_RE`, `verify_ticket_lifecycle._APPROVED_RE` = `^Approved:\s*.+$`) accept the prescribed delegate form.
7. 030-B/030-C: HELD, excluded, untouched.

## Residual risks
- Delegation is an UNSIGNED transcription by MASTER of the Operator's chat words (assurance: process_enforced). The ticket text says the approval line is "written only by the Operator"; the delegated line is intentionally distinguishable. The Operator can override.
- ABANDONED replaces TIMEOUT and the contract `error_mapping` VALIDATION_ERROR for admitted side-effecting calls (decisions OD-TLX-A / OD-TLX-C Option A). After this change no assertion in the touched file expects TIMEOUT or VALIDATION_ERROR for those cases; if other tests elsewhere still do, phase 2 combined run will show it.
- C1 locks in ABANDONED as the CURRENT fail-closed result for post-close() submit failure; it does not fix a possible durable RUNNING record (HELD 030-C).
- Ruff-forced import ordering puts `intergrax.*` among third-party imports; the plan's lint step must be run and any ruff output applied/reported. If ruff disagrees, the blob-equality claim changes; the plan already says to report.
- F3 flake needs an earlier test in the same process; mitigated by whole-module 3x runs.
- ACT-N11 pre-check is expected green by my reading of the gate code but is not executed by me.

## Could not verify
- OD-TLX-A / OD-TLX-C signed records (outside the repo); the semantic correctness of ABANDONED vs TIMEOUT is taken from the delegated decisions.
- Any runtime behaviour (no tests run by design): that the 4 reconciled tests and the 3 new tests actually pass, that F3 is no longer flaky, that ruff is clean, that ACT-N11/IIAC checks are green.
- That the delegation reflects the Operator's chat exactly (only MASTER's transcription was available; the Operator wrote in Polish and the record gives an English rendering by MASTER; the original chat transcript is held by MASTER).
- Phase-2 outcome; the plan's own "STOP if HEAD moved" check remains the implementer's duty.

## Action taken
APPROVE line written under `## Approved` in D:\Projekty\_wt\intergrax\impl-tlx\governance\tickets\T-IGX-030-A.md (nothing else changed, not committed, plan file untouched).
