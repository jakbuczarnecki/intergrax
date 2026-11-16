# REVIEW-PLAN2 - independent review of Amendment A1 (S12 lint normalization), T-IGX-031-A

Reviewer: fresh session (authored nothing in this chain). Subject: branch `hist-cert/plan-amend`, commit `00df54795` (parent `efc9a61e4`). Read-only on the repo; ruff run only as `--diff` / `--check` / stdin (locked ruff 0.15.20 from the worktree `.venv`, the version in `uv.lock`); no pytest; scratch work only under the session scratchpad (removed at the end). Evidence below = command + result.

## VERDICT: ACCEPT_WITH_CHANGES

The design is sound and every measurable claim of the planner was reproduced. The AST check is not vacuous (it caught 17 of 17 negative cases) and a faithful simulation of the S12 steps ends clean and EQUIVALENT. Changes needed are documentation/precision fixes to the plan (F1-F5, all LOW); none changes the design. Because F1-F5 edit the plan, the digest changes again and the approval must be requested for the corrected digest only (do not approve 980394...a120 and then edit).

## 1. Diff and process state - PASS

- `git diff --name-only efc9a61e4 00df54795` = `governance/tickets/T-IGX-031-A.md`, `governance/tickets/plans/PLAN_T-IGX-031-A.md`. No test or product file.
- Digest: `python -I scripts/governance/plan_digest.py` on the committed blob (`git show 00df54795:<plan>`) = `980394592125b524504dfd081631735839242a7455f1f18b585acf5625e5a120` = ticket line 20 `Plan digest:`. The old blob at `efc9a61e4` gives `a83842...7509` (the superseded digest, matches the planner).
- `scripts/governance/verify_plan_approval.py` run on the committed ticket+plan (scratch copies): digest stage passes, then `[ERROR] ticket missing Approved: line`, exit 1. That is the correct state for `plan_pending`.
- `grep -n '^Approved:'` on the committed ticket: no match (the verifier regex requires `Approved:` at line start followed by text). The old approval is a line starting `Previous approval (SUPERSEDED ...)`; `## Approved` shows `(pending ...)`. Lifecycle: `plan_pending`.
- Format: both blobs: non-ASCII 0, tab 0, backslash 0, CR 0.
- Every hunk of the plan diff (old vs new) is part of A1; none is unrelated:
  1. header "Amendment A1" paragraph - A1
  2. section 1, new item 3 - A1
  3. section 2, "Amendment A1 (S12) adds no file" paragraph - A1
  4. section 5, ledger row S12 - A1
  5. section 5, new sub-section "S12 detail" - A1
  6. section 6, "Amendment A1 roles" paragraph - A1
  7. section 7, step 9 expected/baseline cells reworded - A1
  8. section 7, new step 12 row + embedded script + explanation - A1
  9. section 9, residuals R8, R9 - A1
  10. section 9, "Could not verify" sentence extended - A1
  Unchanged: steps 1-8, 10, 11, all of S1-S11, the forbidden list, gate-weakening checklist (the original one), R1-R7.
  Ticket hunks: A1 note, digest line, Approved section (history + pending), verification-command lines (step 9 comment + S12 line), lifecycle - all A1.

## 2. Defect list - PASS (reproduced)

`git show efc9a61e4:<file> | .venv/Scripts/ruff.exe check|format --check --stdin-filename <file> -` (config from the worktree):

| File | ruff check | format --check |
|---|---|---|
| test_sec_ent.py | I001 (L5), F401 `RuntimeEventBus` (L265) | clean |
| w5_b2 | I001 | clean |
| w5_c | I001, E402 (L100) | RED |
| w5_d | I001 | RED |
| tests/support/integration_catalog_isolation.py | clean | clean |

Total 6 errors (I001 x4, E402 x1, F401 x1), 2 files to reformat. Matches the brief and the planner. (A global ruff 0.16.1 on this machine gives the same lint result; see F4 on pinning.)

`ruff check --select I,F401 --diff` and `ruff format --diff` match the planner's description exactly: blank line after `import pytest` removed in all four (intergrax is classified third-party); in b2/c/d `testing_support.*` and `tests.support.*` move above `lab_application.*`, which becomes a trailing block; in w5_d `ExporterKind`/`ExportError` and `ObservabilityExportSinkFactory`/`OtlpEventExportSink` swap (same module, same names); F401 deletes exactly one line; format: one hunk in w5_c (signature joins to 1 line), four hunks in w5_d (line joins <= 120 cols). Nothing else.

## 3. Behaviour neutrality - PASS (one claim in the report is wrong, F2)

(a) F401: `RuntimeEventBus` appears in test_sec_ent.py only at L247/249, L265 (the unused one), L290/293. In `test_defense_middleware_blocks_cross_tenant_scope` the name is never used. A function-local import cannot be re-exported. The module `intergrax.runtime.events.event_bus` is imported at module top by `intergrax/runtime/nexus/nexus_loop.py:67` (and test_sec_ent.py imports `NexusLoop` at top), so even the import side effect (module load) is already done before the test body runs. Removal changes nothing.

(b) E402 in w5_c: the late import (L100) of `ObservabilityExportPayload` is used once, as an annotation (L104), the file has `from __future__ import annotations` (L6) so it is never evaluated; the same module `intergrax.contracts.event_delivery` is imported at L18, so module load order is unchanged. Moving the name into the L18 statement only moves the name binding earlier. No import-time behaviour can change. The fallback `# noqa: E402` is acceptable behaviourally but is incompatible with the embedded script (F1).

(c) I001 reordering. Beyond reading the five files the planner read, I ran a static trace (scratch, AST, top-level imports resolved to files under repo root, `applications/`, `agents/`, including package `__init__`):
- closure of the test file's own `intergrax.*` imports = 2965 modules (loaded before either reordered group in BOTH orders);
- group A (`testing_support.runtime_events`, `tests.support.integration_catalog_isolation`): closure 2401, of which only 4 modules are not already in that set (`testing_support`, `testing_support.runtime_events`, `tests`, `tests.support.integration_catalog_isolation`);
- group B (`lab_application.host.settings`, `lab_application.manifest`): closure 2456, 58 new modules (lab_application, echo, lab, problem_radar, research, signoff_probe agents, a few `intergrax.agents.*`, `intergrax.applications.*`, `intergrax.fastapi_core.*`, `intergrax.runtime.wiring.harness_governance`);
- new-A and new-B are DISJOINT (0 shared modules);
- module-level statements in all 62 new modules, flagged by type (calls in expression/assignment position, subscript/attribute assignment, loops, any decorator other than dataclass/pytest/property-like): none after filtering (only `@dataclass(...)` and one `TypeVar(...)`).
So swapping the relative order of A and B cannot change any shared state: neither group registers anything, and A's modules need only already-loaded modules.
NOT traced (R9 stays open for these): imports executed inside functions at import time, `importlib`/entry-point discovery, third-party package side effects, `conftest.py` imports, and the runtime (non-import) behaviour of the reordered modules' functions. The test files' own top-level statements are covered by the AST check (see 4).

## 4. AST-equivalence script (plan step 12 (b)) - not vacuous; three gaps (F3)

Extracted the 76-line script from the plan (parses). It uses `git show` / `git diff`, so I ran an identical copy with only `git()` replaced by a directory reader (logic untouched) on scratch copies. The "after" copies were produced by the real S12 steps with the locked ruff on the real blobs (F401 fix on test_sec_ent.py, hand merge of the late import in w5_c, `--select I --fix`, `format`, via stdin).

Positive control: pure S12 output, base vs after -> `S12 EQUIVALENT`, exit 0. All five files then pass `ruff check` and `ruff format --check`.

Negative cases (each applied to the S12 output; all exit 1 with the stated reason):
- assertion changed / assertion deleted / decorator removed / module marker `no_ci` dropped / `@pytest.mark.skip` added / string literal changed / body statements reordered / default argument changed / import wrapped in `if` -> "tree without imports differs"
- module-level import added; function-local import added; other import removed; import aliased; import moved function -> module scope; declared F401 removal NOT applied -> "FAIL imports"
- `# noqa` added -> "noqa count changed"
- extra file changed -> "files outside the four changed"
Result: 17 of 17 detected. Body, decorator, marker, assertion, import-set and noqa changes are all caught.

Gaps (not detected, exit 0): G1 a module-level import moved to the END of the file (import position relative to executable code is not checked); G3 a function-local import moved to after its first use (would raise NameError, caught only by the step 12 (c) rerun); G2 a non-`noqa` comment added or changed (for example `# pragma: no cover`, `# type: ignore`). G1 is the exact class of the E402 move itself, which is therefore only covered by the hunk-by-hunk reading. These are LOW because the reader and the rerun backstop them, but the plan claims the script "proves" more than it does (F3).

## 5. Gate weakening - PASS with notes

- S12 permits only: import statements, line breaks/whitespace, one deleted unused local import. The script blocks everything else (4 above). No assertion/body/marker/`no_ci`/skip/xfail change can pass it.
- "Exactly the four files differ": the script tests `changed - FILES` (subset), i.e. it forbids any other file and does not demand all four differ. That is the right direction. It also means the S12 commit may not carry the checkpoint/ticket files; the plan says S12 is "a separate commit ... touching only the four files", consistent.
- `ruff --fix` on whole files: the plan scopes it (`--select F401`, `--select I`, `ruff format` on the four files only). I reproduced those exact commands' effect: the diff is exactly the hunks listed in section 2 plus the one leftover blank line (F2). No extra rule is applied because `--select` overrides the configured set. Risk that remains: a different ruff version (global 0.16.1 vs locked 0.15.20) could format differently; the plan uses `uv run --extra dev-ci ruff`, which is the locked one - good, make it explicit (F4).
- "S12 as a separate third commit, no squash": agree. It is the only way the script has a clean parent and the audited commits stay verifiable. Squashing would make step 12 (b) unrunnable. The script depends on the auditor passing the right PARENT (F4).

## 6. Residuals R8 and R9 - honest; recommendation

R8 text is honest (the authorization was used; the run predates S12; what S12 cannot be proved to leave unchanged is import-time ordering). R9 is honest ("not traced"). My static trace (section 3c) narrows R9: for the four reordered files the order of the two groups is provably irrelevant to static top-level state; I recommend the plan cite that as reviewer evidence but keep R9 open for the untraceable list above.
Recommendation on another 27-path run: NOT needed for S12. Bodies/fixtures/markers are AST-identical, the reorder is between two side-effect-free, disjoint, non-overlapping module sets, and step 12 (c) reruns the four files in two orders in one process. An extra run (about 8 minutes plus a new authorization) buys little. The only reason to spend it is the separate question of claiming F-016 resolved on the FINAL SHA; that is the Operator's policy call and is independent of S12. If the Operator wants that claim, schedule one run once, after S12, on the final tree, instead of arguing about S12.

## 7. Format and process statements - PASS

ASCII only, LF, no tabs, no backslashes in either file (counted on the committed blobs). Process statements true: no `Approved:` line anywhere; the previous approval is labelled SUPERSEDED and bound to the old digest; `## Approved` is pending; the planner is stated to write no approval; the reviewer role split (implementer / delta auditor / fresh approver) is consistent with section 6.

## Findings

| # | Sev | Location | Finding | Exact fix |
|---|---|---|---|---|
| F1 | LOW | plan S12 detail row "W5 c / E402" (fallback) vs step 12 (b) script | The documented fallback (`# noqa: E402`, keep the import in place) makes the script print `FAIL noqa count changed` (reproduced; this is its only failure). The fallback is therefore unusable as written. | Add to the script `ALLOWED_NOQA_DELTA = {FILES[2]: 0}` default and say: if the fallback is taken, the auditor runs the script with that entry set to 1 (and reads the single added `noqa` line); or drop the fallback and make "move" mandatory. State which. |
| F2 | LOW | AMEND_REPORT "Decisions" item 2; plan S12 detail row F401 | The report says the blank line left after deleting the import "ruff format normalizes". False: with ruff 0.15.20 the first empty line of the function body stays and `ruff format --check` passes (verified: line 264 is an empty line after the `def`). The plan does not repeat the false claim, but does not mention the blank line either; the hunk-by-hunk reader will see one extra whitespace change. | In the F401 row add: "the blank line that followed the deleted import stays (ruff format keeps it); accepted, whitespace only", or tell the implementer to delete it by hand. Choose one and say so. |
| F3 | LOW | plan step 12 (b) explanation ("The check proves ...") | Over-claim: the script does not check import POSITION relative to code (G1, G3) nor comments other than `noqa` (G2). The E402 move is in the unchecked class. | Reword "proves" to list what it does not prove, and add to the auditor duties: "read every moved import: it stays in the same scope, module-level ones stay above any executable statement or sit in the original header block, function-local ones stay before their first use". Optional hardening: key each import also by the count of non-import statements preceding it in its body, with the one declared w5_c move whitelisted. |
| F4 | LOW | plan step 12 (b)/(a), S12 commands | (i) PARENT is supplied by hand; a wrong PARENT gives meaningless evidence. (ii) Ruff version not pinned in prose (global 0.16.1 exists on this machine, lock has 0.15.20). | (i) Require `PARENT = $(git rev-parse <S12_SHA>^)` and `git diff b28398026 <PARENT> -- <the five files>` empty (the parent carries the audited test content). (ii) State "locked ruff via `uv run --extra dev-ci ruff`; a global ruff is not evidence". |
| F5 | LOW | plan S12 detail ("base `af323c005`") | The plan header says Base `6734971cf`; `af323c005` (merge of PR #415, parent of the first implementation commit) is introduced by A1 without definition. | Add "implementation base `af323c005` (merge of PR #415, parent of `232d03175`)" in the S12 detail "Why" paragraph. |

No HIGH or MEDIUM finding. Nothing found that makes S12 weaken a gate.

## What I could not verify

- Any actual test result after S12 (no pytest by brief); the claim "steps 1-5 give 14/20/34/34/1 after S12" is a prediction until the implementer's rerun.
- Import behaviour that is dynamic or lives outside top-level static imports (function-level imports run at import, `importlib`, entry points, third-party packages, `conftest.py`).
- That the real S12 commit will contain exactly the simulated edits (I simulated with the locked ruff on the real blobs, not on the real worktree).
- The pre-push gate itself (`scripts/ci/run_ruff_changed_paths.py` was read: it runs `uv run ruff check` then `ruff format --check` on changed Python paths; not executed).
- Whether step 10's `git diff --name-only 6734971cf..HEAD` guard still matches reality after the merge of PR #415 (pre-existing text, not part of A1; mentioned only so nobody trips on it).
