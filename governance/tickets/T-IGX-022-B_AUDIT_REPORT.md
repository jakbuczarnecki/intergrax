# T-IGX-022-B - Independent audit of the B-4 delta introduced by T-IGX-022-A

Auditor: independent audit session (Claude agent, fresh context, read-only on the repository).
Audited revision: development @ cc86e4d805ed6a601bbe642ddf8f09243a4d36d0 (merge of PR #396; implementation commit 00cc594a9, parent 65ccf681a).
Accepted B-4 baseline compared against: fdb1e7a6215f8eb03bc3ca8d00cce9659dc5c3db.
Work location: fresh clone D:/Projekty/_wt/intergrax/_scratch/aud022b/c (removed after the audit); main checkout not touched; no repository file edited, no commit, no branch, no push.

## 1. Scope

Questions 1-6 of the audit brief (delta integrity, discrimination of the two edited tests under mutation, Gate A/B, IIAC activation, targeted test run, role separation). Out of scope (per T-IGX-022-B): formal re-certification of B-4, IIAC manifests, certifier disposition, the 520-test run, roadmap_replay_x failures, ent-audit-fast.

## 2. Commands run and exit codes

| # | Command (in clone, PYTHONPATH=clone, venv python 3.12 via `uv sync`) | Exit |
|---|---|---|
| C1 | git diff fdb1e7a62..cc86e4d80 -- preflight script, .gitattributes, test_evol_x_closure_preflight_b4.py | 0 (one hunk, see 3.1) |
| C2 | git diff 65ccf681a cc86e4d80 --stat | 0 (3 files, see 3.1) |
| C3 | git hash-object of preflight script at HEAD vs git rev-parse fdb1e7a62:<script> | identical blob 356dc59a4ed6615e63ed0f35863ede7d50384e48 |
| C4 | md5 of .gitattributes at fdb1e7a62 vs HEAD | identical (4f053aa46179ac11779b089f98e72ab8) |
| C5 | pytest (one process) test_evol_x_closure_preflight_b4.py::test_ra_b4_03_real_register_rows_still_classify_correctly + test_evol_x_b6_attestation_contract.py::test_published_records_keep_formal_b6_pending | 0 - `2 passed in 0.44s` |
| C6 | scripts/qualification/evol_x_closure_publication_preflight.py --gate formal-closure | 0 - `EVOL-X closure preflight: PASS`, Gate A PASS, Gate B PASS (repository canonical, policy mode IIAC) |
| C7 | scripts/governance/verify_iiac_policy_activation.py --require-in-force | 0 - `IN_FORCE @ IIAC-2026-11-05.1 from f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d` |
| C8 | 30 single-test pytest probe runs (one process at a time), each followed by `git checkout -- .` in the scratch clone | see section 4 |
| C9 | 4 preflight runs on a mutated working tree (see 4.3) | 1 (expected) |

## 3. Delta integrity (Q1)

3.1 Changes since the accepted B-4 implementation fdb1e7a62 for the three guarded paths:
- scripts/qualification/evol_x_closure_publication_preflight.py: NO change (blob hash identical at fdb1e7a62 and HEAD; `git log fdb1e7a62..cc86e4d80` on the path is empty).
- .gitattributes: NO change (identical hash).
- tests/unit/qualification/test_evol_x_closure_preflight_b4.py: exactly one hunk, +2/-1 lines, only in test_ra_b4_03_real_register_rows_still_classify_correctly: a comment line added and `assert not pf.decision_recorded(...)` flipped to `assert pf.decision_recorded(...)` for OD-EVOL-12 and OD-EVOLX-08. The OD-EVOL-05b/10/13 positive loop and the OD-EVOL-14/15/17 `assert not` loop are unchanged.
- P2 pin: `P2_CERTIFICATION_PUBLICATION_SHA = "ad0877715ed079ce920ff9d925fda03555b2c04e"` (script line 158) unchanged, which follows from the byte-identical script.
- `git diff 65ccf681a cc86e4d80 --stat`: governance/tickets/T-IGX-022-A.md (+50), test_evol_x_b6_attestation_contract.py (+19/-3), test_evol_x_closure_preflight_b4.py (+2/-1). Nothing else.
- Test inventory of the preflight test file: 220 test definitions and 9 `skip|xfail` textual occurrences at both fdb1e7a62 and HEAD; the diff contains no removed or added test function and no added or removed skip marker. No other test in the file was weakened, skipped or removed.
- Material governance paths (the six MATERIAL_GOVERNANCE_PATHS of iiac_activation_gate.py): no change between fdb1e7a62 and HEAD. The diff of tests/unit/qualification, scripts/qualification, scripts/governance and governance lists only the two test files, igx_gov/{cli,doctor,inventory}.py, sync_agent_env.py and unrelated governance docs, tickets, policy-registry and TRUSTED_POLICY_PIN (the OD record states igx_gov/* and sync_agent_env.py do not void OD-3; the other governance files are outside the OD-3 condition 3 list).

Conclusion: the delta against the accepted B-4 state is exactly one preflight-test function edit; the script, its P2 constant and .gitattributes are byte-identical, so no new publication of the script is needed.

## 4. Mutation probes (Q2, Q3) - scratch clone only, each reverted with git checkout

4.1 test_ra_b4_03_real_register_rows_still_classify_correctly (T1). Baseline pass. All mutants FAIL (exit 1) as required:

| Mutant | Result |
|---|---|
| OD-EVOL-12 status -> PENDING | FAIL (AssertionError OD-EVOL-12) |
| OD-EVOLX-08 status -> PENDING | FAIL |
| OD-EVOL-12 -> NOT APPROVED (negated) | FAIL |
| OD-EVOL-12 approved but date removed | FAIL |
| OD-EVOL-14 -> plain OPERATOR APPROVED (approved-looking, scope text removed) | FAIL (`assert not True`) |
| OD-EVOL-15 -> plain OPERATOR APPROVED | FAIL |
| OD-EVOL-17 -> plain OPERATOR APPROVED | FAIL |
| OD-EVOL-05b -> PENDING | FAIL |
| OD-EVOL-10 -> PENDING | FAIL |
| OD-EVOL-13 -> PENDING | FAIL |
| script decision_recorded() forced to True | FAIL (OD-EVOL-14 not rejected) |
| script decision_recorded() forced to False | FAIL (OD-EVOL-05b not accepted) |

The edited test still discriminates in both directions (signed rows recorded; pending, negated, undated and scope-limited rows not recorded).

4.2 test_published_records_keep_formal_b6_pending (T2). Baseline pass.

| Mutant | Result |
|---|---|
| evol_x_closed true, b6_acceptance_publication_sha null | FAIL (sha fullmatch) |
| b6 sha malformed ("1dd7693") | FAIL |
| ladder FORMAL_CLOSURE_PUBLICATION_SHA row removed (renamed) | FAIL |
| ladder FORMAL_CLOSURE sha replaced by placeholder `0000` | FAIL |
| OD-EVOL-12 PENDING while closed | FAIL |
| OD-EVOLX-08 PENDING while closed | FAIL |
| OD-EVOL-12 approved without date | FAIL |
| ECAP amendment: PENDING_REQUIRED_APPROVALS text removed | FAIL |
| ECAP amendment: NOT_IN_FORCE changed to IN_FORCE | FAIL |
| governance ecap_amendment_activation -> ACTIVE | FAIL |
| governance normative_effect -> IN_FORCE | FAIL |
| evol_x_closed false but b6 sha still set | FAIL (open branch requires null sha) |
| evol_x_closed false, sha null, register rows approved | FAIL (open branch requires PENDING rows) |
| evol_x_closed as string "true" (not boolean True) | FAIL (falls into the open branch, sha not null) |
| ladder sha well-formed but not a commit (40 x f) | PASS (gap F-1) |
| every `**OD-EVOL-14**` register row renamed | PASS (gap F-2) |

4.3 Preflight Gate probes (formal-closure) on an uncommitted mutated tree: all four mutants (nonexistent ladder sha, b6 sha null, OD-EVOL-12 PENDING, ladder row removed) exit 1, but three of them were rejected by the dirty-tree binding (A-21) before the specific check; only the OD-EVOL-12 mutant reached the specific check (`B-07 operator decision OD-EVOL-12 not recorded as decided`). I did not create throwaway commits (brief: never commit), so Gate B's specific B-08 / B-10 reaction to a non-ancestor ladder SHA was verified by code reading only (script lines ~2556-2570: every ladder SHA must be a commit and an ancestor of HEAD, the closure SHA must be a strict descendant of the B6 SHA, and a claimed closure without the ladder row is B-10; lines ~2422-2426: the B6 SHA must descend from the certification SHA and be an ancestor of HEAD). On the real tree Gate B passes.

## 5. Findings

| ID | Severity | Evidence |
|---|---|---|
| F-1 | LOW | T2 (test_published_records_keep_formal_b6_pending) checks the ladder FORMAL_CLOSURE_PUBLICATION_SHA and the governance b6 SHA only for 40-hex shape, not for commit existence or ancestry of HEAD (probe "40 x f" passes). The gap is covered by preflight Gate B (B-08, B-10), which passes on the real tree; the unit test is a state guard, not the closure gate. Optional hardening: add an ancestry assertion, or keep relying on the Gate B CI check. |
| F-2 | LOW | T2 assertion `"OD-EVOL-14" in register` is a bare substring check and survives removal of the OD-EVOL-14 table rows (probe passes). It pre-dates the delta and is unchanged by it; not a regression. |
| F-3 | LOW (informational) | The closed branch of T2 does not cross-check that the governance b6 SHA equals the ladder B6 SHA; preflight B-02 covers it. The open branch preserves the original strict pre-closure assertions, so a premature or inconsistent CLOSED claim is caught in both directions. |
| F-4 | LOW (process) | Session independence cannot be proven from git: the author of 00cc594a9 is `jakbuczarnecki` (GitHub noreply identity, the same account as the repo git user), merges are by the operator account. Role separation is process_enforced (section 7). |

No MEDIUM or HIGH finding. No evidence that the delta weakened the B-4 corrective.

## 6. Answers in brief

1. Delta = a single hunk in one preflight-test function (plus the other test file and the ticket outside the guarded set). Script and .gitattributes byte-identical, P2 pin unchanged, no test removed, skipped or added.
2. Yes, the edited T1 discriminates (12 of 12 mutants killed).
3. Yes, T2 catches premature CLOSED claims (14 of 16 probes killed; the two survivors are F-1 and F-2, covered by Gate B or pre-existing).
4. Gate A and Gate B PASS (exit 0); IIAC IN_FORCE (exit 0).
5. The two edited tests: 2 passed.

## 7. Role separation (Q6)

`git log` shows 00cc594a9 authored by `jakbuczarnecki` on 2026-11-07 22:07 +0100 under ticket T-IGX-022-A. That ticket states the implementer must not self-certify the audit and names T-IGX-022-B as a separate-session audit; T-IGX-022-B states the implementer does not set the verdict. This audit ran in a new session with no prior involvement in EVOL-X or T-IGX-022-A (no implementation, publication, certification or signing by me), started from the T-IGX-022-B task prompt. Both the implementer and I act through the same operator repository account and git metadata cannot distinguish sessions, so independence is process_enforced and NOT proven. I signed nothing and wrote no `Approved` line. The scratch clone stayed at cc86e4d80 on branch development with no persisted modifications.

## 8. Not verified

- Gate B reaction to a non-ancestor / non-commit ladder SHA by execution (code reading only, to avoid scratch commits).
- The full 520-test run of T-IGX-022-A and the GitHub CI status of `evol-x-b4-preflight`; not repeated (out of scope).
- Whether the operator requires a formal certifier re-disposition of B-4 after this audit (separate operator-ordered step).

Verdict: PASS_WITH_FINDINGS

Reason: the delta since the accepted B-4 implementation fdb1e7a62 is exactly one edit to one preflight test function (two assertions flipped to match the signed OD-EVOL-12 / OD-EVOLX-08 rows); the preflight script, its P2 pin and .gitattributes are byte-identical, no test was removed, skipped or weakened, and the edited test still discriminates in both directions under 12 mutation probes. The second edited test still rejects premature or inconsistent CLOSED claims in 14 of 16 probes, with two LOW gaps (no ancestry check of the ladder SHA in the unit test, which Gate B covers and which passes; a weak substring assertion that pre-dates the delta). Gate A/B and IIAC activation pass. The verdict is conditional on the stated limits: session independence is process_enforced, and the Gate B ancestry reaction was verified by code reading, not execution.
