# Historical enterprise certification campaign - FINAL REPORT
Generated 2026-11-08T11:20:57Z. Result: **PARTIAL / EXACT BLOCKERS**. This is NOT `HISTORICAL_CAMPAIGN_CLOSED_VERIFIED`. Language: English only (project rule).

## 1. Result in one paragraph
All four workstreams completed their inventories and every one of the 85 historical FRZ rows has a reasoned disposition with evidence, owner and remaining gate. **No row was closed independently (0) and none was promoted (0 scoped acceptances)**: the checklist is unchanged (all 85 rows OPEN). The campaign found real defects that were hidden behind historical closure labels, corrected the documentation it could correct, integrated and independently audited one test-only corrective (T-IGX-030-A, PASS_SCOPED), and stopped where the next step needs an Operator decision or an IIAC re-audit / re-activation.

## 2. Start and end state
|  | start | end |
|---|---|---|
| repository | origin/development `ba07ebaa3` | origin/development `41908e3fb` |
| FRZ checklist (225 rows) | 120 OPEN / 105 PASS | 120 OPEN / 105 PASS (no status changed) |
| historical 85 rows | 85 OPEN | 85 OPEN (all dispositioned) |
| Operator decisions | - | 5 answered, 7 open |
Expected family counts reconciled with the checklist at the start: REG 10, BND 6, OWN 5, CTR 6, TYP 6, PLG 8, RPL 4, OBS 7, SEC 10, REL 11, PRD 12 (matches: REG 10, BND 6, OWN 5, CTR 6, TYP 6, PLG 8, RPL 4, OBS 7, SEC 10, REL 11, PRD 12).

## 3. Sessions
Workers (separate Claude Code processes, separate worktrees and branches, pinned at `ba07ebaa3`):
| workstream | terminal | Claude session id | FRZ scope | branch |
|---|---|---|---|---|
| HIST-01 | t7 | 8cca8db3-455a-4d8d-9c63-d356410c6ef8 | REG-01..10 | hist-cert/hist01-qualification |
| HIST-02 | t12 | f13fa3c1-138a-4d38-9ff4-19244367cc49 | BND-01..06 OWN-01..05 CTR-01..06 TYP-01..06 PLG-01..08 RPL-01..04 | hist-cert/hist02-architecture |
| HIST-03 | t13 | 5a9e8ba9-967f-49ab-bccd-cf3651c93abc | OBS-01..07 SEC-01..10 REL-01..11 PRD-01..12 | hist-cert/hist03-controls |
| HIST-04 | t14 | 303eeffe-0910-416b-b572-c2f1f67d1240 | none (governance / SSOT) | hist-cert/hist04-governance |
Sessions retired before producing any evidence (first launch used the wrong terminal host - see section 12): | workstream | retired session id | host |
|---|---|---|
| HIST-01 | 32721904-8016-4992-a9f6-960becef8c85 | herdr CLI server w5 |
| HIST-02 | ed3bd0ff-b79c-4f1d-b095-13b85fa25cab | herdr CLI server w5 |
| HIST-03 | 58b95a10-d45a-467a-878a-8f9138eaa1f3 | herdr CLI server w5 |
| HIST-04 | aceb4ed9-0151-40d9-8e7c-02353812f0c2 | herdr CLI server w5 |
Other sessions:
| name | Claude session id | role |
|---|---|---|
| impl-tlx | 4e77efc7-e2c5-4334-9974-392ee06497ca | planner / implementer, T-IGX-030-A |
| cert-030a | 929ea8c8-4988-4fc1-bcc6-6e66b265b0ac | designated certifying session (plan approval) |
| rev-docs | 2fa420b1-3289-4211-bdde-8f7a59068b02 | independent reviewer, PR 405 |
| rev-rules | 36de2835-33cd-4117-a4a4-0f201f898370 | independent reviewer, PR 406 |
| aud-030a | cdb0c03b-866b-433c-b899-a8d0ccbcc05f | independent auditor, T-IGX-030-A |
Session ids are transcript file names; the host does not expose them.

## 4. What was implemented and published
| PR | branch | head | merge commit | content | independent verdict |
|---|---|---|---|---|---|
| 405 | hist-cert/docs-patches | 6d1434796 | 41908e3fb | documentation-only evidence / citation corrections, no status changes | ACCEPT (rev-docs) |
| 406 | hist-cert/rules-update | 937c39820 | feee82b2e | IIAC Amendment 01 revision (S-1 / S-3 struck), delegation template, plan approvers + advisory pin, README | ACCEPT (rev-rules) |
| 407 | hist-cert/impl-tlx | 90fae38aa | 4e48eba82 | T-IGX-030-A: two test files (blob-identical to the source commits), plans of HELD tickets 030-B / 030-C | PASS_SCOPED (aud-030a) |
Merged by the Operator; each head equalled the pushed tip and CI on that head was green (jobs that failed with an HTTP 504 on the `uv` download were re-run).

## 5. Findings
62 findings: by class {"SSOT_DRIFT": 13, "QUALIFICATION_GAP": 12, "ACCEPTED_LIMITATION": 6, "EVIDENCE_GAP": 13, "AUTHORITY_GAP": 5, "UNVERIFIED": 3, "IMPLEMENTATION_GAP": 4, "INDEPENDENT_AUDIT_GAP": 3, "NO_REMAINING_OBLIGATION": 1, "PATH_B_LIVE_DEFERRED": 1, "VALID_SCOPED_PASS_CANDIDATE": 1}; by severity {"HIGH": 15, "MEDIUM": 20, "LOW": 12, "MAJOR": 7, "MINOR": 5, "BLOCKER": 1, "INFO": 2}; by workstream {"HIST-01": 16, "HIST-02": 14, "HIST-03": 16, "HIST-04": 16}.
Material ones (HIGH / MAJOR / BLOCKER) are listed per id in the four `HISTNN_FINDINGS_AND_RESIDUALS.json` files (`dossiers/`). The ones that matter most:
- **EBH-4 R1 nexus-encapsulation gate is red on the base** (HIST02-F-013, MAJOR): `intergrax/runtime/tools/tool_operation_recovery.py` imports `runtime.nexus` (TOOL-LIFE-X commit `a5f813a2c`). The accepted EBH-6 / EBH-7 runs did not execute this gate.
- **PROD-Q P2-C HTTP admission tests are red** (HIST03-F-015, HIGH, worker thesis: code defect): 6 of 7 tests fail even in isolation; the error text was introduced by SKILL-X Batch C `d1040f485`, which the SKILL-X acceptance did not catch. Not verified: that the tests were green at the P2-C accepted SHA.
- **A Tier-0 module imports a Tier-3 demo fixture** recorded as a "ratified tier exception" with no decision id (HIST02-F-001, MAJOR): breaks the hard boundary in `AGENTS.md`.
- **Authority gaps (HIST-04):** the EVOL-X chain was signed by delegated sessions in plain form although the campaign delegations exclude EVOL-X (F-H4-01); Amendment 01 was self-contradictory and its personal-signature safeguards were agent-proposed (F-H4-02, since revised by PR 406); the mechanism cannot prove authorship (F-H4-03).
- **Label drift:** TOOL-LIFE-X is labelled "CLOSED - INDEPENDENTLY ACCEPTED" against a signed CONDITIONAL / STOP audit (H01-F001, F-H4-08); the EVOL-X roadmap row prose says CLOSED while its stage cell says PENDING_AUTHORITY.
- **28 rows have no existing current-HEAD executable selection owner** (HIST02-F-011).
- **FRZ-RPL-01..04 cannot move until the independent VBN acceptance exists** (HIST02-F-007, the only BLOCKER-severity finding; WBS FIND-VBN-PARTIAL / T-B7-VBN-INDEPENDENT-ACCEPT, BLOCKED_EXTERNAL); no PASS is inferred from EBH-7.

## 6. Disposition of the 85 rows (derived)
By classification: {"QUALIFICATION_GAP": 34, "EVIDENCE_GAP": 20, "VALID_SCOPED_PASS_CANDIDATE": 19, "AUTHORITY_GAP": 1, "IMPLEMENTATION_GAP": 1, "SSOT_DRIFT": 5, "INDEPENDENT_AUDIT_GAP": 3, "PATH_B_LIVE_DEFERRED": 2}.
By state: {"QUALIFICATION_REQUIRED": 51, "AUDIT_REQUIRED": 29, "AUTHORITY_REQUIRED": 2, "IMPLEMENTATION_REQUIRED": 1, "PATH_B_DEFERRED": 2}.
Counters: open architecture blockers 52; outstanding audit requirements 29; Path B primary 2; rows named by the checklist Path B list (L66) 5; rows with a worker-identified live dependency or Path B primary 19 (FRZ-REG-08 was reclassified out of Path B by the independent integration audit, F-01); AUTHORITY_REQUIRED 2; findings requiring implementation 4; SSOT discrepancies 13.
`VALID_SCOPED_PASS_CANDIDATE` (19 rows) means "an independent audit and an exact-SHA replay are the only thing missing", not "passed".

## 7. Closed obligations
None of the 85 rows. What was closed or established: the documentation inconsistencies of PR 405; the Amendment 01 revision text of PR 406 (merged by the Operator; its status as policy remains an open Operator decision, DR-2 / OD-HIST04-B); conditions C-1 / C-2 of the TOOL-LIFE-X audit are addressed by T-IGX-030-A (PASS_SCOPED; does not close TOOL-LIFE-X).

## 8. Qualification results (per-path results and summaries are in the dossiers; raw logs are NOT published - see the evidence registry, section 4 - so these figures are worker-reported and cannot be reproduced from published files)
- HIST-02 replay of 27 architecture modules at `ba07ebaa3` (pinned checkout, heavy lock): 818 tests collected; first run 813 passed / 5 failed (3 missing `duckdb` extra, 1 harness artefact caused by MASTER's `--basetemp`, 1 real: EBH-4 R1); re-run of the 4 non-product failures with `--extra dev-ci`: 4 passed. Net: 817 passed, 1 real failure.
- HIST-03 replay of 27 modules: combined run 331 passed / 21 failed = 7 `test_sec_ent.py` (patch-target defect confirmed statically, runtime confirmation pending, F-006) + 8 W5 b2/c/d (registry pollution; pass alone, F-016) + 6 P2-C (worker thesis: code defect, F-015, masked by the pollution in this run). Caveat: the run used a shared pin whose venv HIST-02 extended while the run was in progress; it is not a clean-room exact-SHA run (HIST03 register, test_execution). In isolation W5 b2 / c / d: 5 + 6 + 9 passed; P2-C: 1 passed / 6 failed.
- aud-030a at `683f95572`: side-effect module 37 passed; F3 module 5 passed x3; combined set 76 passed; ruff clean; IIAC verifier IN_FORCE; governance tests 56 passed / 1 skipped (symlinks). Negative probe: the pre-change file fails 4 of 36 tests against the same code.
- Not run: the broad N6 replay of TOOL-LIFE-X, `igx_gov ticket`, the QUAL-X Dimension A replay (approved, sequenced after the product batch).

## 9. Operator decisions
Answered (5): OD-HIST01-A, OD-HIST02-B, OD-INT-01, OD-PUB-01, OD-ACT-N11. Open (7): OD-HIST04-A; OD-HIST04-B; OD-HIST04-C; OD-HIST02-A; OD-HIST03-A; OD-HIST03-B; OD-HIST01-B. Mapping to the decision request: OD-HIST01-A = DR-10, OD-HIST02-B = DR-5, OD-INT-01 = DR-7, OD-PUB-01 = DR-11, OD-ACT-N11 = DR-12 (answered); OD-HIST04-A = DR-1, OD-HIST04-B = DR-2, OD-HIST04-C = DR-3, OD-HIST01-B = DR-4, OD-HIST03-B = DR-6, OD-HIST02-A = DR-8, OD-HIST03-A = DR-9 (open). The answers are relayed from the Operator's chat by the MASTER orchestrator; no Operator-signed record exists in the repository. Full text, evidence, options and recommendations: `HISTORICAL_AUTHORITY_AND_OPERATOR_DECISIONS.md`.

## 10. Independent audit of this package
The independent integration audit (aud-final, a fresh read-only session that authored, applied, reviewed and approved nothing in this campaign) returned **ACCEPT_WITH_CHANGES** for package commit `2a7aec93d`: 13 findings (1 HIGH - FRZ-REG-08 wrongly classified as Path B; 8 MEDIUM; 4 LOW). The central claim (PARTIAL / EXACT BLOCKERS, 85 rows dispositioned, none closed or promoted) was confirmed; no false closure language, no misrepresented Operator signature, no change outside the package directory and no material IIAC path touched were found. All 13 findings were applied in `6c973c93a`. The text-only re-read of that commit returned **ACCEPT_WITH_CHANGES** with a single mechanical edit left (R-1: four dashboard counters still showed the pre-reclassification values); it was applied in `63ff33812` and checked by MASTER with `git diff` (only those four numbers and the generation timestamp changed). The auditor stated that the package is ACCEPT once R-1 is applied and that no further audit round is needed; none was run. It also confirmed that 19 is the right count of rows with a worker-identified live dependency or Path B primary. The audit covers the fidelity of the consolidation, not the correctness of the 85 underlying worker judgements, which remain unaudited. Report with the re-read section: `audits/aud-final/AUDIT_REPORT.md`.

## 11. Authority risks that remain
- The EVOL-X delegated signature lines have no valid authority until the Operator ratifies or re-signs them (DR-1); 10 FRZ-EVOL rows are PASS on the checklist on that chain.
- Delegated records are `process_enforced`: the identity of a delegate cannot be proven (one git identity in the chain); with S-3 struck nothing is non-delegable, including IIAC activation.
- Validators still read "operator" literally and contradict the revised policy (OD-ENT-01 delegate path demands an operator-signed line; EVOL-X preflight rejects the delegate form; activation gate and protocol are literal and are material paths). Alignment is HELD for the IIAC re-audit / re-activation batch.
- `verify_plan_approval.py` accepts any `Approved:` line: a green verifier does not prove authority.
- The semantic basis of the T-IGX-030-A assertions (OD-TLX-A / OD-TLX-C Option A, delegated, signed outside the repository) could not be verified from the repository.

## 12. Errors made by the orchestrator (kept for transparency)
1. Workers were first launched on the `herdr` CLI server, a different host from the one `herdr-app` shows; the Operator saw no panes. Relaunched through `hc`; the first sessions never produced evidence.
2. Replay commands carried `-p no:cacheprovider` and an outside `--basetemp`; both caused avoidable failed runs.
3. A rebased documentation patch was reported as status-neutral after inspecting a truncated preview; it flipped the EVOL-X roadmap stage cell. The independent review caught it; reverted in PR 405.
4. "The Operator must sign personally" was presented as a binding rule; it came from agent-proposed safeguards (S-1 / S-3) the Operator never confirmed. Corrected after the Operator's objection; revised in PR 406.
5. Operator-facing documents were first written in Polish; translated (English-only rule).
6. A failure message of the first combined run was attributed to the wrong tests; HIST-03 later separated it.
7. A SHA-256 check in a temporary tree disagreed with the repository's pin verifier because of line endings; the verifier was authoritative and the pin was correct.
Worker-side incidents: one worker removed another's live heavy lock (self-reported; HIST-03's combined run overlapped HIST-02's rerun in the same venv, so HIST-03 results carry an interference caveat; see the HIST-02 report, "Lock protocol incident"); an aborted run left a hung process tree (HK-01, termination left to the Operator).

## 13. Global acceptance criteria (section 19 of the mission)
| # | criterion | status | basis |
|---|---|---|---|
| 1 | All four workstreams completed their inventories | MET | 4 worker dossiers + DONE flags |
| 2 | All 85 original FRZ criteria were reconciled | MET | 85/85 rows dispositioned in HISTORICAL_FRZ_85_DISPOSITION_MATRIX.json (dispositioned, NOT closed) |
| 3 | No applicable mandatory architecture invariant remains OPEN without an authorized resolution | NOT MET | all 85 rows OPEN; 0 closed independently |
| 4 | No unresolved critical implementation blocker remains | NOT MET | 4 IMPLEMENTATION_GAP findings (EBH-4 R1 gate red, PROD-Q P2-C red, and others) |
| 5 | All material correctives have independent acceptance | PARTIAL | T-IGX-030-A (tests only) PASS_SCOPED; product correctives (030-B, 030-C, P2-C) not yet implemented |
| 6 | Required authority exists | NOT MET | 7 Operator decisions open (EVOL-X ratification, Amendment 01 status, labels, scope rulings) |
| 7 | No delegated signature is misrepresented | MET (campaign-created records); earlier chains UNRESOLVED | campaign delegation lines disclose themselves; EVOL-X chain stays AUTHORITY_GAP |
| 8 | Historical scoped closures retain their limitations | MET | no scope was widened in any campaign output |
| 9 | All applicable supersession records agree | PARTIAL | EVOL-X roadmap row and TOOL-LIFE-X label still disagree with their evidence |
| 10 | Canonical roadmap and FRZ checklist are reconciled | NOT MET | documentation patches merged (PR 405) but label/scope decisions DR-1/3/4/9 are open |
| 11 | Required negative regression tests are green | NOT MET | known red gates at the base: EBH-4 R1 (1 test), PROD-Q P2-C (6 tests), test_sec_ent (7 tests) |
| 12 | Publication is complete on the authorized branch | PARTIAL | PRs 405-407 merged; final package PR pending |
| 13 | Independent integration audit has passed | MET (conditional edit applied) | aud-final: ACCEPT_WITH_CHANGES at 2a7aec93d, re-read ACCEPT_WITH_CHANGES at 6c973c93a with one mechanical edit (R-1, four dashboard numbers); the auditor stated the package is ACCEPT once R-1 is applied and that a git-diff check is sufficient; R-1 applied in 63ff33812 and the diff shows only those numbers and the generation timestamp. No second audit round was run |
| 14 | All residual Path B live obligations remain correctly OPEN | MET | 5 rows named by checklist L66 plus 14 worker-identified live dependencies; all OPEN |
| 15 | Handoff is suitable for ARCH-FREEZE readiness review | MET (as a NOT-READY handoff) | ARCH_FREEZE_HISTORICAL_PREDECESSOR_READINESS_HANDOFF.md |

## 14. Next critical steps toward ARCH-FREEZE
1. Operator decisions DR-1, 2, 3, 4, 6, 8, 9. 2. One IIAC re-audit / re-activation batch carrying T-IGX-030-B (EBH-4 R1), T-IGX-030-C (invoker fix + red-first test) and the validator alignment. 3. Fresh independent audit of that batch; then the QUAL-X Dimension A replay on the final SHA. 4. Independent audit of the AUDIT_REQUIRED row dispositions before any checklist promotion. Details: `ARCH_FREEZE_HISTORICAL_PREDECESSOR_READINESS_HANDOFF.md` and `HISTORICAL_REMAINING_EXACT_BLOCKERS.md`.
