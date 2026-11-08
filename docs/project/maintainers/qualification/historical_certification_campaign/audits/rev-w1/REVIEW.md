# REVIEW-W1 - independent non-author review of branch hist-cert/cert-w1

Reviewer: fresh session (rev-w1). Authored nothing in this stage. Read-only: only `git show/diff/log/grep/ls-tree/merge-base/cat-file` against `D:/Projekty/_wt/intergrax/cert-w1`; no checkout, edit, commit, test or governance script run.
Reviewed: base `70cceae21`, tip `8ccbcef4450ea6b015bb3535a4f31950c29483b9` (2 commits). Line numbers below refer to `OPERATOR_DECISION_RECORDS_W1.md` at the tip unless stated.

## Verdict: ACCEPT_WITH_CHANGES

Every factual claim I re-checked is true, with five small exceptions (F-1..F-5), and none of them changes a SIGNED verdict. The records do not promote anything, do not approve any ticket, and do not exempt any rows from the checklist rule. cert-w1's own findings 3, 5 and 8 are all disclosed honestly in the records. The changes are wording or citation fixes. They can go in as an erratum commit on top of the tip. Do not amend: the four `Approved` lines carry the sha of commit 1 and a time stamp, but not a hash of the body, so a follow-up commit does not invalidate them.

## Check 1 - Diff inventory: PASS

- `git log 70cceae21..8ccbcef44`: 2 commits, linear (`8ccbcef44` -> `cbb511ec7` -> `70cceae21`).
- `git diff --name-status 70cceae21 8ccbcef44`: exactly two `A` entries, `.../historical_certification_campaign/decisions/OPERATOR_DECISION_RECORDS_W1.md` and `.../historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md`. 178 insertions, 0 deletions.
- Nothing under `intergrax/ agents/ applications/ scripts/ tests/ governance/`. No checklist, roadmap, WBS, ticket or IIAC file in the diff.

## Check 2 - Delegation copy and Approved lines: PASS

- `sha256sum` of the original `D:\Projekty\intergrax\.tmp\session\hist-cert-campaign\delegations\DELEG-HIST-W1-2026-11-08.md` = `747dce743947fa2d88d00b85d66631b2d562fc5939d3c8e83e0a44e89a4c9a1c`. `git show 8ccbcef44:<path> | sha256sum` gives the same value. Byte-identical. The brief's value matches too. (The `.sha256` file exists next to the original; I did not open it.)
- `git cat-file -e cbb511ec7...:docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W1-2026-11-08.md` -> exists. `git merge-base --is-ancestor cbb511ec7 8ccbcef44` -> true. The path in all four lines is the same repo-relative path, and the sha is 40-hex.
- Form: all four lines (47, 85, 111, 139) read `Approved <ID>: operator-delegate cert-w1 @ <ISO-8601+01:00> | Delegation: <path>@<sha>`, which is the form required by the delegation (S-2). Time stamps per item: 15:51:46 (R1), 15:51:49 (R2), 15:51:50 (R3), 15:51:53 (PLAN). They lie between the commit times (commit 1 15:46:54, commit 2 15:53:51). Line 4 says explicitly that every `Approved` line is a delegate line and "must never be cited as an Operator-written line". No wording anywhere presents them as operator-written.
- W1-PLAN (lines 116-140): "This record certifies a sequence. It approves no ticket plan." Verdict line 137: "SIGNED (sequence acceptance only)". Matches the delegation's MASTER reading 2.
- Accepted-content text of R1/R2/R3 and the six PLAN steps (lines 26, 54, 92, 118) matches delegation lines 14-17 word for word in substance.

## Check 3 - Re-verified claims (evidence = command + result)

| Claim in record | Result |
|---|---|
| Checklist L222 SEC-05: criterion text, owners CTRL-X/PROD-Q, OPEN, note wording (a597445, d44a238db..., deaafee1a..., "Scoped capability evidence only ... PROD-Q still required ... new global FRZ PASS = 0") | TRUE (`git show <tip>:...CHECKLIST.md \| sed -n 222p`) |
| Checklist L384 PHYSQ stage row lists SEC-05 and says all listed criteria remain OPEN | TRUE |
| Checklist L66 does not list SEC-05; it lists PRD-01, PRD-02 and the live parts of PRD-03, PRD-08, SEC-04 | TRUE |
| PHYSQ JSON: 106 lines, `reference-substrate-qualification`, allowed target reachable in control and qualified phases, denied target reachable in control and `reachable:false` in qualified, redirect `escaped:false`, `correlation_result` PASS, 3 sessions destroyed, 2026-09-29, no `netns`/`nftables` string | TRUE (`cat -n`; `grep -ci` -> 0) |
| `deaafee1a` and `d44a238db` are ancestors of base; JSON history = 2 commits of 2026-09-29 | TRUE (`merge-base --is-ancestor` rc 0 both; `git log` shows `deaafee1a`, `7a7b63a24`) |
| HIST-03 matrix L26 SEC-05 = A, obligations as quoted; L70 wording | TRUE |
| Vendor rule: "Contract stubs, fakes, or in-process doubles are not real vendor integration evidence - label scope accordingly"; no Docker migration, no new vendor integrations, no production configuration; R7-Q/PROD-Q/FRZ statuses unchanged by rule/tooling edits | TRUE (note: the "label scope" sentence sits in the "Vendor integration qualification" section, the freeze items in the "C10/C15" section; the record's wording at line 34 merges them but each quote is accurate) |
| Decision-03: 68 lines, status line, 2026-10-18 K8s selection, section 8 text | TRUE (`cat -n`) |
| OD-D03-A-01: 35 lines, "mechanism class only", no service selection, D03-B/C mandatory | TRUE |
| Conditional infrastructure direction: Spanner `eur3` + Azure RA-GZRS North Europe "accepted only as a preferred service choice", OD-D03-A-03..08 not ratified, cloud execution NOT AUTHORIZED, offline child CLOSED - SCOPED | TRUE |
| L218 SEC-01 counts: Decision-03 x1, D03-A x5, Spanner x1, Kubernetes x1; L230, L232, L258 contain none | TRUE (per-line `grep -o`). Also none of L222/264/265/271 contains "population"/"ENT-AUDIT-X"; L268 (REG-06) contains "Population" twice. |
| L264/265/268/271 = REG-02/03/06/09, all OPEN, owner QUAL-X (REG-02 also EBH-5) | TRUE |
| QUAL-X inventory: Populations A-D at 2357-2360, E at 2361 "Full freeze checklist rollup / REG-01 PLATFORM BLOCKED - ENT-AUDIT-X"; section is "Population boundaries" | TRUE (`### §2 Population boundaries`) |
| WBS L38 `T-B7-L3-FREEZE` = `run_enterprise_audit.py --mode FREEZE`, "FREEZE PASS"; L60 `T-B7-PROMOTE-REG` REG-01..10, depends on L3-FREEZE | TRUE. Workflow `enterprise-audit.yml` is named "Enterprise audit (ENT-AUDIT-X)" and calls `scripts/enterprise_audit/run_enterprise_audit.py run` |
| HIST-01 handoff L18-22: option (a)/(b)/(c), recommends (b) for REG-07/10, (a) for REG-01/04/05, nothing on REG-02/03/06/09 | TRUE |
| REG matrix: the four rows class gap/required, Population A only | consistent; `needs_head_replay` = True for REG-02, 03, 05, 06, 07, 09, 10 (see F-1) |
| 19 candidates; REG-10 is the only REG row; none of REG-02/03/06/09 | TRUE (`counts.by_classification.VALID_SCOPED_PASS_CANDIDATE` = 19; scan of the matrix rows gives only FRZ-REG-10) |
| W0 DR-4 text and "Residual risks (1)" quoted; DR-9d NOT_SIGNED / AUTHORITY_REQUIRED | TRUE |
| DR-10 (L101-103) and DR-12 (L124-127) texts; "not assumed" | TRUE |
| `iiac_common.py` L22 `PRODUCT_PATH_PREFIXES`, L95-110 `product_paths_changed_between` | TRUE |
| `iiac_activation_gate.py` L40-49 `MATERIAL_GOVERNANCE_PATHS` (six members as listed), L329-344 `_material_changes`, ACT-N11 messages at 569 and 576 (record says 567-576) | TRUE. Tests-only/docs/tickets change matches neither prefix list nor the six paths, so neither ACT-N11 rule trips. |
| Tickets: `T-IGX-030-B.md`, `T-IGX-030-C.md` exist, `## Lifecycle state` = `plan_pending`, HELD, no approval line; `T-IGX-030-A.md` exists with a `cert-030a` delegate line, excludes F-006/F-016 | TRUE |
| No ticket for F-006/F-016 test-only batch, P2-C, BND-03; `T-IGX-021-B/C` absent, named only in `T-IGX-021-A.md` L47-48 | TRUE (`git ls-tree`; `git grep` over `governance/tickets`: only `plans/ASF-SEF-MASTER-PLAN.md:204` mentions `intergrax/contracts/external_work.py`, an unrelated CAP row) |

## Check 4 - Overstatement: none that changes a verdict

- No record promotes anything or approves a ticket. Rows "stay OPEN" in R1, R2, R3; L66 unchanged; PLAN "approves no ticket plan".
- W1-R2 does NOT treat itself as exempting the (b) parts of SEC-01/REL-03/REL-05 from the checklist rule; line 77 says the opposite and points to a later ruling. Correct (I confirmed the rule at checklist L68: `OPEN = 0` applies to architecture-scoped criteria, and L66 lists only PRD-01/02 and the live parts of PRD-03/08 and SEC-04).
- W1-R3 does not claim the three rows' text mentions Population E (line 104 says the link rests on the HIST-01 shape and on REG-06). The identification "FREEZE-mode run = Population E rollup" is stated as the reviewer's own reading of three documents (line 98). I agree it is a reading, not a sentence in any document.
- cert-w1 finding 3 (R2 does not exempt) -> disclosed at line 77. Finding 5 (SEC-05 tension HIST-03 A vs "PROD-Q still required") -> disclosed at line 39. Finding 8 (F-006/F-016 tests-only is a thesis) -> disclosed at line 130. Findings 2, 4, 6, 7, 9 also appear (lines 72-74, 36, 100, 126). Honest.
- "Could not verify" blocks are present for every item and match what I could not verify either (see below).

## Findings (numbered; none blocking)

**F-1 (LOW, factual) - line 105, W1-R3 residual risk 3.** The record says `needs_head_replay` is true "for REG-02, 06, 09". `HIST01_FRZ_REG_PER_ID_MATRIX.json` has `needs_head_replay: true` for REG-03 as well (REG-02, 03, 05, 06, 07, 09, 10 are true; 01, 04, 08 false). The omitted id is REG-03, one of the four rows W1-R3 is about. Fix: replace "for REG-02, 06, 09" with "for REG-02, 03, 06, 09". (The W0 DR-4 record has the same incomplete list at its line 103; that is not part of this branch and is not re-opened here.)

**F-2 (LOW, citation) - line 67, REL-03 mapping.** "Decision-03 section 1(b) and section 4 (PIS as the one logically global linearizable intent authority; stable CREATE identity)". Section 1(b) is right. "Stable CREATE identity" is not a section 4 requirement of Decision-03; it is a prior decision listed in Decision-03's `Prior` field (line 9, "Stable CREATE Identity Decision 01") and is quoted in checklist L230. Fix: "section 1(b) and section 4 (one authoritative PIS concurrency domain); stable CREATE identity comes from the prior Stable CREATE Identity Decision 01 cited in Decision-03's Prior field".

**F-3 (LOW-MEDIUM, wording that could be over-read) - line 63, (a) list.** The record puts "the four-part never-call-unless rule" of Decision-03 section 1 under (a) "testable in Dimension A". Parts (a) and (b) of that rule (a trusted independent continuity authority proven for the environment; one logically global linearizable intent authority across all replicas) cannot be proven without the anchor backend and a multi-replica environment, which the record itself places in (b). Line 73 gives the correct qualifier (offline/simulated, labelled, no Kubernetes or exactly-once claim) but only for D03-B/D03-C. Fix: at line 63 add after the rule: "as a fail-closed contract with offline or simulated evidence only; proof against a real continuity authority or real replicas is part (b)".

**F-4 (LOW-MEDIUM, naming collision not disclosed) - lines 54-70.** In Decision-03 section 6, "Gate D03-A" is "trusted anchor backing and trust domain" (naming the actual backend, CAS semantics, recovery boundary, issuer authorization). The delegation's (a) says "contract and the trust/PIS semantics". By Decision-03's own text the Gate D03-A body is a (b) item (the record maps backend and trust domain to (b) at line 64). A later reader of "(a) = trust semantics" may read the D03-A gate as (a). Fix: add one sentence to the "What maps cleanly" block: "'Trust' in (a) means only the contract-level fail-closed predicate and the ELCR-IIRW class as a contract; Decision-03 Gate D03-A (anchor backing and trust domain, section 6) is itself part (b)."

**F-5 (INFO) - line 72, evidence for REL-05/PRD-08 link.** The record cites HIST-03 matrix "lines 22, 34, 70-71". The matrix rows for REL-05 (L36) and PRD-08 (L50) do not mention Decision-03 at all; only the narrative at L71 names the four rows. This strengthens the record's own point 1 at line 72. Optional fix: "(rows L22 and L34 name Decision-03; for REL-05 (L36) and PRD-08 (L50) only L71 does)".

**F-6 (INFO, no change to the branch) - merge method.** Each `Approved` line points to the sha of commit 1. If the PR is squash- or rebase-merged, `cbb511ec7` is no longer reachable from the target branch and the `Delegation:` reference dangles. `CERT_W1_REPORT.md` line 46 says so; the record itself does not. Fix (optional): add to line 4 "the PR must be merged with a merge commit so that `cbb511ec7` stays reachable". MASTER should enforce the merge method either way.

**F-7 (INFO) - time stamps.** The four stamps lie within 7 seconds (15:51:46..53) although the record describes a separate verification per item. This is consistent with stamping at the signing step after the verification, and each stamp is distinct as required. No change; noted so nobody reads the stamps as the verification time.

## Check 5 - Format: PASS

- LF only: 0 CR bytes in both files. Both end with a single `\n`. 0 lines with trailing spaces in the records file.
- ASCII only: `tr -d '\000-\177' | wc -c` -> 0 for the records file (the delegation file is byte-identical to the original, which the original author's report records as ASCII; its sha equals).
- No backslash characters in either file (`grep -nF '\'` -> none). Paths use `/`.
- No secrets: `grep -iE 'password|secret|token|api[_-]?key'` -> no hit in either file.
- English only; Polish appears nowhere (the delegation states the original is in the chat transcript).

## What I could not verify

- That the netns + nftables enforcement really ran (the JSON does not name them; I read the checklist's words, as did cert-w1). Nothing was executed.
- Whether any verifier other than the two gate files (`verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, ACT-N13 test-matrix bytes at `iiac_activation_gate.py` L437-538) binds particular test files. I saw that `test_matrix_path`/`test_matrix_digest` exist, not what they cover; cert-w1 said the same. The ACT-N11 claim is confirmed for ACT-N11 only.
- The scope of the ENT-AUDIT-X Population E rollup and whether it will include the four REG rows; whether one IIAC re-activation can cover both streams (step 2 of the plan); feasibility of any listed ticket.
- Whether the PHYSQ and D03-A offline-provider evidence is current at HEAD (not replayed; no test allowed by the brief).
- Whether F-006/F-016 can be fixed by tests alone (a thesis by design).
- The `.sha256` sidecar file content (I hashed the original myself instead).
- Anything about the operator's chat reply itself: the delegation is a transcription; I verified the record against the transcription, not the transcript.
