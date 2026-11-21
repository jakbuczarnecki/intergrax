# EQO pre-publication audit

Audited tip: `c089f134739a161439d52ae8e3db60ecdf6dac7b`. `origin/development`: `17fc79228eb0b46880ecad17ff52354622a81e9e`. 42 commits ahead, 0 behind, at audit time. Worktree: `D:/Projekty/_wt/intergrax/af-dev-merge`. This file correction is a later docs commit. **Do not publish `c089f1347`.** No as-is exception exists.

## Commit audit

42 commits. 40 carry `Ref: governance/tickets/`. Missing `Ref:`: merge `6fa8f2955` and `d1da3b3c1` (A4-034 block record). Eleven commits contain `Co-authored-by: Cursor <cursoragent@cursor.com>`: `5cceebb69`, `537a06296`, `5e9496611`, `b02278643`, `d1da3b3c1`, `c600f79a6`, `c15588151`, `ffec97177`, `531b4e323`, `960205aaa`, `321957488`. High-signal secret scan of the full diff: no private-key block and no `AKIA` id. Product behavior under test was not changed after `84fc00a16`; later commits are docs plus the already-merged PR #738 test fixture (`g09_tenant_evaluation_registry.py`, two architecture test lines).

Scope besides EQO bookkeeping and tickets `T-IGX-055-A` / `T-IGX-055-B`:

- Product: `fastapi_router.py`, `schemas.py`, `check_runtime_event_tenant_propagation.py`, two unit tests.
- Side docs: `CRITIC_VERIFICATION.md` stub (5 lines), roadmap §3.0.6 status line, freeze checklist FRZ-CTL-05 label, `AGENT_INSTRUCTIONS.md` critic link.

`intergrax-core.md` Attribution forbids a tool `Co-Authored-By`. Human `Co-Authored-By` is allowed only on request. No exception mechanism authorizes these eleven tool trailers. Publishing as-is is not offered.

## Evidence SHAs (not a formal settlement)

| Blob | SHA | What it proves |
|------|-----|----------------|
| Missing-tenant behavior | `537a06296` | Has a forbidden tool trailer |
| Whitespace fail-closed | `1aa91803b` | No tool trailer |
| Gate rebind | `960205aaa` | Has a forbidden tool trailer |
| UAEP path assertion | `f3371ca6f` | No tool trailer |
| Pytest of the two files plus A4-026 file | `84fc00a16` | 19 passed. Not re-run on `c089f1347` |
| Named replay / governance | `84fc00a16` | 64 passed and 27 passed |

A4-006 and A4-027 stay `TECHNICAL_REVIEW_PASS` only. Formal settlement still requires a clean published SHA, a re-run of the two test files on that SHA, and an independent certifier. No `CLOSED`, `FULLY_MET`, or FRZ-TEN-10.

## OD-EQO-09 does not authorize A4-042

`EQO_OPERATOR_DECISION_PACKET.md` defines OD-EQO-09 as "Confirm proposed EDER owners / One batch approval". It does not mention `intergrax/globals` or locale defaults. WB06 follow-up 2 still asks the operator to assign an owner and choose stay, configuration, or neutral. No record shows that choice was made. A4-042 code is not READY.

## A4-031 and A4-037

A4-031: `uaep_executor.py:386` skips `assemble_uaep_session_messages` when `context_engine` or `llm_adapter` is None, and the call continues. Both dependencies default to unset. Fail-closed would stop every UAEP execution that does not inject both, including current partial and lab wiring. Not authorized.

Negative tests, to be written only if fail-closed is chosen (they would be red today):

- engine None, adapter set: raise, do not assemble.
- adapter None, engine set: raise, do not assemble.
- both None: raise.
- both set: assembly still runs.

A4-037: `nexus_loop.py:334` already raises `HumanDecisionPersistenceNotConfiguredError` in `production_mode` when the store is missing and `disable_human_decision_persistence` is false, and when the store is in-memory. `persist_human_decision` (`human_response.py:91`) still returns when the store is None. That return is the lab path and the explicit acknowledgement path. Raising there would close both. Not authorized.

Negative tests that match the current guard (green if added; not added in this audit):

- `production_mode`, store None, acknowledgement false: `HumanDecisionPersistenceNotConfiguredError`.
- `production_mode`, in-memory store: same error.
- `production_mode`, acknowledgement true: construction succeeds; `persist_human_decision` returns without writing.
- not `production_mode`, store None: `persist_human_decision` returns.

## Deferral bases (not closure)

A decision here does not close the row and does not start a Composer.

| ID | Basis | Allowed next state |
|----|-------|--------------------|
| A4-034 | FRZ-REL-02 has no sentence that names one qualified LLM owner across the inline budget, unwired `LocalProviderRetryBudget`, and `StepRetryBudget`. Deletion is forbidden. | Stay `EXACT_BLOCKER` until that sentence exists outside this packet. |
| A6-006 | `ENT_AUDIT_X_SEMANTIC` skip is the qualification contract, not a product defect. | Stay deferred. No gate edit. Not closed. |
| EDER-F-0016, F-0020, F-0023, F-0059, F-0076, F-0080, F-0084, F-0091, F-0107, F-0116, F-0124, F-0154, F-0158, F-0171, F-0182, F-0186, F-0188, F-0192, F-0200, F-0207, F-0209, F-0210, F-0222, F-0236 | Each register row has no explicit `tests/*.py` path. | Requalify only when FCS names a test. No invented test. |
| EDER-D-0001, D-0002, D-0003, D-0004, D-0005, D-0006, D-0007, D-0008, D-0085 | Same: no explicit test path. | Same. |
| EDER-D-0086 | Cited path is an ellipsis, not a file. | Same. Not promotable to a named test. |

## Decisions that unblock work

| ID | Choice required | READY after that choice |
|----|-----------------|-------------------------|
| OD-REWRITE | Separate consent to rewrite the eleven tool trailers and to add `Ref:` on `d1da3b3c1` before any push. No as-is alternative. | Orchestrator rewrites only the unpublished stack, re-checks trailers, then gives the operator the new tip and the push commands. No push in this session. |
| OD-042 | Explicit choice: keep Poland defaults, move them to configuration, or make them neutral. OD-EQO-09 is not this choice. | Composer edits `intergrax/globals/settings.py` only for the chosen option, plus one unit test. |
| OD-031 | Explicit fail-closed on the unset UAEP assembly path, knowing partial hosts will stop. | Composer changes `uaep_executor.py` and adds the four tests above. |
| OD-037-GUARD | Authorize tests that lock the existing `production_mode` guard. No change to `persist_human_decision`. | Composer adds the four tests above. No production edit. |
| OD-037-PERSIST | Only if the acknowledgement and lab return must also raise. | Separate package. Not recommended on the current code. |

A4-040 / OD-EQO-04 stays open and is not READY. A4-034, A6-006, and the 34 rows do not become READY by confirming this table.
