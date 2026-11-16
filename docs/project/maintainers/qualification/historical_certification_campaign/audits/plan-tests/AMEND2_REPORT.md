# AMEND2 REPORT - T-IGX-031-A, review rev-plan2 findings F1-F5 applied

Worktree `D:/Projekty/_wt/intergrax/plan-tests`, branch `hist-cert/plan-amend`. New local commit `1c9240802` (parent `00df54795`; no amend, not pushed). Files: ticket and plan only. No test or product file touched; no pytest run.

New plan digest: `dc0fcf736456534e923000eaf4cce1d569edee9db28ef73386a5bd8284b3cc6b` (`python -I scripts/governance/plan_digest.py`, recomputed on the committed blob, equal to the ticket `Plan digest:` line).
Ticket: `plan_pending`; no line starts with `Approved:` (0 matches on the committed blob). History kept: the first superseded digest `a83842545c...` (approved, superseded) and a new paragraph saying digest `980394592125...` (commit `00df54795`) was reviewed by rev-plan2 but NEVER approved. ASCII, LF, no tabs, no backslashes (both blobs).

## Findings applied

- F1 (noqa fallback): the script has `ALLOWED_NOQA_DELTA` (0 for every file). The plan states that MOVE is the PRIMARY fix; the `# noqa: E402` fallback is only for the case where the move is shown to change behaviour, and then the auditor sets the w5_c entry to 1 and reads the single added noqa line. The E402 row, the step 12 text and the script say the same.
- F2 (blank line): the F401 row states that the blank line after the deleted import stays (ruff format 0.15.20 keeps it), accepted, whitespace only, reviewer-verified, not to be deleted by hand. (My earlier report claim that `ruff format` normalizes it was wrong; the plan never repeated it.)
- F3 ("proves"): reworded into "what the script checks" (7 points) and "what it does NOT prove" (comment/blank-line placement, same-body import order, dynamic imports, runtime behaviour), plus the auditor duty to read every moved import (same scope; module-level above executable code or in the original header; function-local before first use; no noqa/skip/xfail/marker in added lines). Hardening added (simple, tested): import position (= number of non-import statements before it in its body) with the declared w5_c move whitelisted to land at the position of the existing `EventDeliveryPolicy` import; multiset of non-`noqa` comments.
- F4: PARENT is computed by the script (`git rev-parse <S12_SHA>^`) and it fails unless `git diff --name-only b28398026 <PARENT> -- <five files>` is empty; the locked ruff via `uv run --extra dev-ci ruff` is mandatory (0.15.20 in `uv.lock`, confirmed in the worktree: `ruff 0.15.20`), a global ruff is not evidence. Step 12 (a) and the S12 "Commands" paragraph say so.
- F5: S12 detail "Why" defines "implementation base `af323c005` = merge of PR #415, parent of `232d03175`", distinct from the plan's planning base `6734971cf`.
- R9: cites the reviewer's static trace (section 3c: 2965-module closure, group A adds 4, group B adds 58, disjoint, no side-effecting top-level statements in the 62 new modules) and keeps R9 OPEN for the untraceable list (function-level imports at import time, importlib/entry points, third-party side effects, conftest, runtime behaviour).

## Script test on scratch copies (outside the repo)

Scratch git repo in the session scratchpad; the locked ruff 0.15.20 applied through stdin to the real audited blobs `b28398026:<file>` (config from the worktree), then the S12 steps (F401 fix, hand merge or noqa, `--select I --fix`, `format`). The embedded script is byte-identical to the tested file (checked).

| Case | Expected | Result |
|---|---|---|
| move variant, all deltas 0 | EQUIVALENT | EQUIVALENT, exit 0 |
| fallback (noqa) variant, w5_c delta 1 | EQUIVALENT | EQUIVALENT, exit 0 |
| fallback variant, delta 0 | FAIL | FAIL (noqa delta) |
| move variant, w5_c delta 1 | FAIL | FAIL (noqa delta) |
| module-level import moved to end of file (G1) | FAIL | FAIL (import position changed) |
| function-local import moved after a statement (G3) | FAIL | FAIL (import position changed) |
| extra non-noqa comment (G2) | FAIL | FAIL (comments changed) |
| assertion changed | FAIL | FAIL (tree differs) |
| noqa added elsewhere | FAIL | FAIL (noqa delta) |
| extra file changed | FAIL | FAIL (files outside the four) |
| parent not carrying audited content | FAIL | FAIL (parent check) |

Both variants end clean under the locked `ruff check` and `ruff format --check` (no "LINT NOT CLEAN" output). This is evidence about the script only; the implementer's real run counts.

## Residuals / open questions

- R8 (27-path run not repeated) and R9 (open for the untraceable list) are unchanged in substance; "after" results remain predictions.
- The ticket in this worktree is checked out with CRLF in the working tree (blob is LF); I normalized on read and wrote LF; git prints the LF-to-CRLF warning on add. Harmless.
- Question for MASTER: approval must be requested for digest `dc0fcf73...cc6b` only; the digests `a83842...` and `980394...` are both superseded.
