# CERT_DECISION - cert-031a, plan approval of T-IGX-031-A

Session: cert-031a (new, dedicated; not planner, not rev-plan, not MASTER, not implementer/auditor of this batch).
Time: 2026-11-08T16:44:00+01:00. Read-only checks (git -C / show / diff / grep, plan_digest.py); NO pytest/ruff/uv run.

## VERDICT: APPROVE

One line written under `## Approved` of `D:/Projekty/_wt/intergrax/plan-tests/governance/tickets/T-IGX-031-A.md` (disclosed delegate form, sha f1d85253c verified with `git cat-file -e`). Nothing else changed; plan file untouched; nothing committed.

## The four R-2 points (as stated at the start)
1. I am a new session dedicated only to this decision; I did not write the plan.
2. I am not planner / plan reviewer / MASTER / implementer / auditor of this batch.
3. I may REJECT; approval was not pre-agreed.
4. My authority is the delegation DELEG-HIST-031A (plan approval of T-IGX-031-A only, plus the meaning of "rec."); T-IGX-030-B/C stay HELD and are not approved here; I run no tests.

## Evidence per check
0. Delegation: `sha256sum DELEG-HIST-031A-2026-11-08.md` = 7b5897d1420ac3fc612ce9f1e23a2a3191adf1035a5721c734a0f3962021314e = the `.sha256` file = brief. MATCH. The record is a MASTER transcription (not Operator-signed); S-1 is struck per R-6, so this is not a validity condition. The four "rec." readings are matched by the plan: shared helper file (plan s.2 item 5), clean-before/after (s.5, risk 1), R1 and R7 to the product batch (s.9), `no_ci` stays (s.2, s.9 risk 3).
1. Digest: `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-031-A.md` (cwd = plan-tests) = a83842545c6e94534fc6d9a1a5b1990bd35a7676d4c7d84d6ce95107ed267509 = ticket `Plan digest:` line. `git diff --stat 9318bf1bb HEAD -- <plan>` empty; blob sha256 at 9318bf1bb = working file sha256 (a838...). HEAD = f1d85253c7d66a413e62f5d4f9d733eba786e3da; `git diff --name-only 6734971cf..HEAD` = the published delegation, the ticket, the plan, nothing else. Worktree was clean before my edit.
2. Scope: plan s.2 lists 5 files, all under `tests/` (4 existing, 1 new `tests/support/integration_catalog_isolation.py`). Nothing under intergrax/ agents/ applications/ scripts/; none of the six MATERIAL_GOVERNANCE_PATHS (`iiac_activation_gate.py` L40-49 read); no conftest, markers, pyproject. `no_ci` is at test_sec_ent.py L41 and stays. A new tests file is outside the ACT-N11 product prefixes; plan step 8 (incl. test_iiac_g0_inventory) plus an inline fallback cover the residual doubt.
3. Review findings (diff cc9b17ff6..9318bf1bb, 2 files, 22+/20-):
   - F1 false venv-approval claim: FIXED, plan L32 now says no approval was requested/given and the planner did not stop and ask. (PLAN_REPORT.md lines are outside git; not checked, not part of the approved plan.)
   - F2 F-016 overstated: FIXED, plan L12 "sufficient polluter ... does not exclude other polluters", step 11 REQUIRED for the audit, R3 open until clean; ticket scope text matches.
   - F3 S6/S7 too weak: FIXED, `__cause__` text assertions added (L74-75); probe A expects S6, S7, S4, S10.
   - F4 probe B count: FIXED, "predicted 12 failed / 22 passed" (L105).
   - F5 S5 vs fail-closed docstring: FIXED, note in S5 and residual R7 (L73, L133).
   - F6 S1 clears-the-store claim: FIXED, parenthesis removed, S1b extends `_reset_store` (L69).
   - F7 helper subset: FIXED, `reset_entry_point_spec_cache_for_tests` added (exists, discovery.py:145); the other seven helper names exist (grep verified).
   - F8 non-ASCII: FIXED, `git grep -nP '[^\x00-\x7F]'` on both files at 9318bf1bb: no hits; 0 CR.
   - F9 count: FIXED, "about 72 (76 including the four)"; `git grep -c` total = 76, 4 in test_sec_ent.py (L106,127,196,211).
   - F10 info: nothing to change.
4. Ledger S1-S11, S1b, W5 fixtures, helper: none weakens a gate. No skip/xfail/marker; no set-widening or `in`; no dropped effect/ledger/replay check (S3 still asserts vaulted value and envelope); no swallowed failure; S4 keeps its negative probe (category-contract object without the three methods, same message, now actually reaching the conformance gate); S5 adds a probe; S6/S7 are strengthened with cause-text asserts and bind the patch where the name is looked up; S8/S9 assertions byte-kept with the sanctioned bootstrap (`integration_wiring.py` L13-30, docstring says prefer it); S10 supplies the identity the production API demands (`bind_/reset_active_execution_identity`, `testing_support/runtime_events.py:137`, reset in `finally`); fixtures and helper are isolation only. Fakes are not vacuous (real conformance and encryptor effect still asserted).
5. Test contract (s.7): `uv run --extra dev-ci pytest ...`, one process at a time under the heavy lock, exit code captured, no `-p no:cacheprovider`, no `--basetemp`, no xdist. Counts consistent (13+1=14; 5+6+9=20; 34). Order independence tested in two orders; probes A and B are meaningful and now deterministic; step 8 ACT pre-check with stop rule; step 11 is an AUDITOR step covered by AUTH-HIST-27PATH-2026-11-08 (sha256 = 1cf2c0a85415852d45e8c5d4eec2c62a39649db462234b8f8848607d1184c6b2, verified; it limits use to the independent auditor, not the approver or implementer, one run, one identical retry for infra error only).
6. Roles: s.6 requires implementer / auditor / approver as three different fresh sessions. The plan and ticket claim no PASS, no closure, and explicitly say F-016 may not be called resolved before step 11 is clean.
7. Escape clause (s.8): honest. Names the real unknowns (S10 publish/schema, S8/S9 infisical factory, S3 Protocol), forbids product edits, no skip/xfail to get green, partial result reported. R1 (`ollama` slug shared by two providers) and R7 (raw TypeError) are labelled product-level, unfixed, and routed to W1-PLAN step 3, as the delegation says.
8. T-IGX-030-B/C: not touched, not approved.

## Residual risks
- Every "after" column (14 / 20 / 34 passed, probe sets) is a prediction; no test was run by the planner, the reviewer or me. The implementer/auditor must produce them. If a candidate in the escape clause materialises, that item leaves the batch (partial result).
- Clean-before/clean-after leaves the catalogs empty after these files; a later file relying on a populated catalog could fail. Only step 11 shows this (REQUIRED at audit).
- Other polluters in the 27-path selection are not excluded (stated in the plan); F-016 stays open until step 11.
- `test_sec_ent.py` stays `no_ci`: even green, GitHub CI does not run it.
- Product-level R1 and R7 remain until the product batch.

## What I could not verify
- Any runtime behaviour (pydantic subclass with ClassVar satisfying the SecretsStore Protocol, real `infisical` factory in S8/S9 without credentials, spine publish/schema path in S10, ACT-N11 / G0 inventory with the new file committed).
- That the delegation reflects the Operator's actual words: it is a MASTER transcription (per its own header); I did not see the chat. The brief itself accepts this under R-6/S-1.
- PLAN_REPORT.md corrections for finding 1 (outside git).
- Whether the lines `Approved:` will satisfy `verify_plan_approval` (not run; form copied exactly from the brief).
