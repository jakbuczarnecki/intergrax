# AMEND REPORT - T-IGX-031-A, Amendment A1 (S12 lint normalization), Operator option A

Planner session `plan-tests`, worktree `D:/Projekty/_wt/intergrax/plan-tests`, branch `hist-cert/plan-amend` (from `efc9a61e4`).
New local commit: `00df54795` (parent `efc9a61e4`; no amend, not pushed). Files: `governance/tickets/T-IGX-031-A.md`, `governance/tickets/plans/PLAN_T-IGX-031-A.md` only. Docs only: no test or product file touched, no test run.

New plan digest: `980394592125b524504dfd081631735839242a7455f1f18b585acf5625e5a120` (computed with `python -I scripts/governance/plan_digest.py`, recomputed on the committed blob, equal to the ticket `Plan digest:` line). Old digest (superseded): `a83842545c6e94534fc6d9a1a5b1990bd35a7676d4c7d84d6ce95107ed267509`.
Ticket: lifecycle `plan_pending`; NO line starts with `Approved:`; the old approval is kept as history under "Previous approval (SUPERSEDED ... old digest ...)"; the `## Approved` section is pending for the new approval. ASCII, LF (blob CR = 0), no tabs, no backslashes.

## What was added to the plan

- Ledger row S12 and a new sub-section "S12 detail" (section 5): defect table with exact fix and neutrality reason per defect, exact commands, import-order analysis, S12 gate-weakening review.
- Step 12 (section 7): equivalence evidence the auditor must produce, with the exact AST check (76-line script embedded; it parses).
- Step 9 now says: red at the base and at `efc9a61e4`, clean only after S12.
- Section 6: S12 implemented by an implementer session, delta-audited by a FRESH auditor, amended plan approved by a FRESH delegated session; the planner writes no `Approved:`.
- Section 9: residuals R8 (27-path not repeated) and R9 (transitive import graph not traced). Sections 1 and 2 and the ticket scope carry a short Amendment A1 note; the file set is unchanged (no new file).

## Defect list (confirmed read-only: `ruff check` / `ruff format --check`, no fix, no tests)

Run on the base blobs (`git show af323c005:<file>` into ruff with `--stdin-filename`) and on the tree `efc9a61e4`:

| File | At base `af323c005` | At `efc9a61e4` |
|---|---|---|
| `test_sec_ent.py` | I001; F401 `RuntimeEventBus` (L223); format ok | I001; F401 (L265); format ok |
| W5 b2 | I001; format ok | I001; format ok |
| W5 c | I001; E402 (L91); format RED | I001; E402 (L100); format RED |
| W5 d | I001; format RED | I001; format RED |
| helper | n/a (new) | clean |

Totals at `efc9a61e4`: 6 ruff errors (I001 x4, E402 x1, F401 x1), 2 files would be reformatted. Matches the brief.

## Decisions and evidence per defect

1. I001 (four files): `ruff check --select I --fix`. Dry run (`--diff`) shows only: a blank line after `import pytest` removed (ruff treats `intergrax` as third-party because `src` lists directories); in b2/c/d the `testing_support.*` and `tests.support.*` imports move above `lab_application.*`; in w5_d two name lists are re-sorted (same module, same names).
   Order dependence: top level of `lab_application/host/settings.py`, `lab_application/manifest.py`, `testing_support/runtime_events.py`, `tests/support/integration_catalog_isolation.py`, `intergrax/runtime/events/event_bus.py` read/grepped: no top-level registration/bootstrap/`sys.path`/`atexit`/`os.environ`; only definitions, `@dataclass`, a module `logger`. Nothing excluded or pinned. NOT traced: their transitive imports (R9).
2. F401 `RuntimeEventBus` in `test_sec_ent.py`: REMOVE. Read: the name occurs only at L247/L249 and L290/L293, in two other functions with their own local imports; in the affected function it is unused; a function-local import cannot be re-exported; no other string occurrence (no `patch("...RuntimeEventBus")`). Dry run shows exactly one deleted line (and a leftover blank line that `ruff format` normalizes; step 12 (a) confirms).
3. E402 in w5_c: MOVE, no `noqa`. Read: no comment explains the late position; the name is used once, as an annotation (L104), under `from __future__ import annotations` (never evaluated); the same module `intergrax.contracts.event_delivery` is already imported at L18, so module load order is unchanged. Fix: merge into L18 (hand edit), then isort. Fallback in the plan: `# noqa: E402` with a reason if step 12 (b) or (c) disagrees.
4. Format (w5_c one hunk, w5_d four hunks): `ruff format`, only line-break collapses within 120 columns (dry run read).
5. Nothing else is touched: no assertion, body, marker, `no_ci`, skip/xfail.

## Equivalence evidence required from the delta audit (step 12)

(a) `ruff check` and `ruff format --check` clean on all five files. (b) The embedded AST check with `<PARENT_SHA> <S12_SHA>`: per-scope import multiset equal except the one declared F401 removal; syntax tree with imports stripped identical; `noqa` count unchanged; only the four files differ; prints `S12 EQUIVALENT`. (c) Steps 1-5 rerun, same results (14 / 20 / 34 / 34 / 1 passed). (d) Probes A and B not required again (S12 must not touch test bodies; (b) proves it). The 27-path run is NOT repeated.

## Residuals

- R8: no 27-path run after S12 (authorization was for one run, used). Why S12 cannot change the F-016 conclusion: bodies, fixtures, decorators, markers, assertions are AST-identical; differences are import order/grouping, line breaks, one deleted unused local import; none registers anything in a catalog. Not excluded: an effect of changed leaf-import order on import-time ordering inside a large combined process (R9). Closing it needs a new Operator authorization.
- R9: transitive import graphs untraced; covered empirically by the step 12 (c) rerun, not by reading.
- Everything "after" is a design from reading; I measured only the defect list.

## Observations

- In this worktree `governance/tickets/T-IGX-031-A.md` is checked out with CRLF (working tree only; the committed blob is LF, CR = 0, verified with `git cat-file`). The plan file is LF in the working tree. I matched and wrote LF; git warns "LF will be replaced by CRLF" for both files on `git add`. Harmless, but whoever edits the ticket on Windows should write LF.
- The earlier ticket text "ready_for_audit" is reset to `plan_pending` as ordered.

## Open questions for MASTER / Operator

1. Who approves the amended plan (fresh delegated session or Operator)? The old delegation covers only the old digest.
2. Should S12 stay a separate third implementation commit (assumed, so that the AST check has a clean parent) or be squashed later? Squashing would break check (b); I recommend not squashing.
3. Is one more 27-path run wanted after S12, or is the residual R8 accepted?
