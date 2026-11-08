# Operator decision records W0 - DR-1, DR-2, DR-3, DR-4, DR-6, DR-8, DR-9

Written by the designated certifying session `cert-w0` on 2026-11-08, under IIAC Amendment 01 (R-1..R-6, S-1 and S-3 struck, S-2 kept) and the delegation record `docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md`.
Every `Approved` line below is a **delegate line**. It is not written by the Operator and must never be cited as an Operator-written line. Assurance level: `process_enforced`. Each line ends with `@6a0a531fb6b94d41d6a00ce8bd800c7504749560`: the commit that carries the English-only revision of the delegation record (section 11). The first version of the lines pointed to `cef4aea9af5e91891c57478323191b0ff83e92e3` (first copy of the record, sha256 ea101bf8...); they were rebound by MASTER in a later commit, nothing was amended. The time stamps (15:04:58+01:00) are the original batch stamp of the session and were not changed.

## 0. Session statements (R-2) and method

- `cert-w0` is (1) a **new** session, (2) **dedicated** solely to certifying these seven decisions, (3) **not a continuation** of any other session, (4) **not an implementation session**. It is not MASTER, not a HIST worker and not an auditor of this stage.
- Delegation verified before any work: SHA-256 of `DELEG-HIST-W0-2026-11-08.md` = `ea101bf8f104cd6cd8db0089d7937d8de44d01f4da7877ad3b9257ca718cd660` (equals the value in the brief and in the `.sha256` file). The published copy is byte-identical (the blob in git hashes to the same value).
- Baseline: worktree `cert-w0`, branch `hist-cert/cert-w0`, base `origin/development` = `991a3e4104bf71f5ebf6a30cb02ddbea184fb2cd` (re-fetched: `git fetch origin development`, `git rev-parse origin/development` = the same SHA).
- Method: read-only verification of repository files and git history; worker conclusions treated as theses. **No test was run.** One governance command was run: `python -I scripts/governance/verify_trusted_policy_pin.py` -> `[INFO] verify_trusted_policy_pin: OK`, exit 0 (before the patches; re-run after them, see section 10).
- Consent is not pre-agreed as a result. `cert-w0` signs a decision only where its own verification supports the consented answer, and says what it could not verify. Summary: DR-1, DR-2, DR-3, DR-4, DR-6, DR-8 SIGNED; DR-9 SIGNED in part (three sub-rulings), with two sub-rulings NOT_SIGNED.

| DR | Verdict | Patch |
|---|---|---|
| DR-1 | SIGNED | `decisions/EVOL_X_DELEGATED_SIGNATURES_RATIFICATION.md` (new file) |
| DR-2 | SIGNED | status note in `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` |
| DR-3 | SIGNED | annotation in roadmap row L482 (TOOL-LIFE-X) |
| DR-4 | SIGNED | record only |
| DR-6 | SIGNED | record only (commissions Q-11) |
| DR-8 | SIGNED | record only |
| DR-9 | SIGNED IN PART: P-03, P-04, waivers; NOT_SIGNED: SEC-05, Decision-03 / D03-A | record only |

---

## DR-1 (OD-HIST04-A) - ratification of the delegated EVOL-X plain-form signatures

**Consented answer.** Option 1: ratify per candidate C1, effective from the original timestamps (Amendment 01 section 1.2, "Earlier chains").

**Verification (read-only).**
- The five lines exist verbatim in the tracked records: `grep -n '^Approved'` on the three records under `iiac_evidence/EVOL-X/` and on `evol_x/EVOL_X_PRE_P3_OPERATOR_DECISIONS_AND_PUBLISHER_DELEGATION.md`: OD-EVOLX-P3-B6 15:41:06, OD-EVOLX-PRE-P3 15:41:06, OD-EVOLX-B6-ACCEPT 15:41:06, OD-EVOL-12 18:17:34, OD-EVOLX-08 18:21:06 (all 2026-11-07, +01:00).
- Content binding recomputed by me, not copied from HIST-04: `git show <signing commit>:<record> | grep -v '^Approved ' | sha256sum` equals the digest in the delegation record for all five (P3 `a1c14c35...`, PRE-P3 `7bf1ad16...`, B6 `520bf25a...`, OD-EVOL-12 `5c9da7e4...`, OD-EVOLX-08 `1befe97c...`): 5 of 5 MATCH. Signing commits `27d905027`, `1dd769331`, `109029b2c`, `5f9727032` and the closure `be77b3a5f` are ancestors of the base.
- The three EVOL-X delegation records are tracked, name the plain form, carry the Operator's quoted chat commission and an expiry of 2026-11-07T23:59:59+01:00; all five signature times are before the expiry and in the W7 order (OD-EVOL-12 < OD-EVOLX-08 < closure publication 18:27:25).
- `FRZ-EVOL-01..10` are `PASS` or `PASS - SCOPED` in the checklist (rows 273-282) on the base.
- Amendment 01 section 1.2 and section 3 allow exactly this route: the Operator may confirm an earlier delegated line by commission, the original timestamps do not change, nothing is rewritten; S-3 (the former bar on delegating such acts) is struck.
- Re-binding of closure SHAs: **determined not to be required for option 1.** The ratification is a new file outside `iiac_evidence/EVOL-X/`; the SHA ladder binds publication commits (not content of this file); the preflight scans only the EVOL-X evidence tree (`evol_x_closure_publication_preflight.py` lines 153, 1415, 1678). Option 2 (re-signing) might have needed re-binding; it was not chosen and not assessed.

**Residual risks.**
- The repository cannot show that the Operator gave the quoted commissions, that the signing sessions were new and dedicated, or that the delegation records existed before the signatures (they were added in the same commits as the signatures). The ratification rests on the Operator's consent and stays `process_enforced`.
- The lines remain in the plain form; any citation of EVOL-X must keep the caveat in section 4 of the ratification record.
- Ratifying authority is not an audit of the EVOL-X closure content; no FRZ status changes.

**Could not verify.** (1) The campaign delegations `DELEG-CERTCAMPAIGN-SIGN-2026-11-07` / `DELEG-2026-11-08` that exclude EVOL-X: gitignored, outside my read scope; known only through HIST-04's quotation. (2) That the preflight and the EVOL-X tests still pass after a new file appears elsewhere in `docs/` (not run).

**Verdict: SIGNED.** Patch: `decisions/EVOL_X_DELEGATED_SIGNATURES_RATIFICATION.md` (new; rewrites no earlier record).
```text
Approved DR-1: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-2 (OD-HIST04-B) - status of IIAC Amendment 01

**Consented answer.** Option (c): Amendment 01 is in force as process-enforced policy for certifications started after `d82f8b722` (S-1 and S-3 struck, S-2 kept as the default) **and** T-IGX-021-B/C is a separate stage.

**Verification.**
- Text merged by PR 406 (read in full, `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md`, 169 lines before the note): the header says "IN FORCE AS OPERATOR-ADOPTED POLICY TEXT ... S-1 and S-3 are STRUCK; S-2 is kept as a strikeable default"; section 5.3 says "in force as process-enforced operator policy"; section 6 says "The rule of section 1 applies from d82f8b722 regardless of steps 1-6"; section 2 strikes S-1 and S-3 and keeps S-2; section 4.5 lists T-IGX-021-B and -C as follow-up tickets. `grep -n -i 'proposed\|not in force'` finds only the words "agent-proposed" (sections 2 and S-2): **the header-versus-5.3/6 contradiction reported by HIST-04 no longer exists in the merged text.** The remaining gap is only that the header does not say that T-IGX-021-B/C is a separate, unstarted stage.
- The note I added is a single paragraph in the header block (2 inserted lines, 0 deleted; `git diff --stat`). It states (a) in force as process-enforced policy, (b) T-IGX-021-B/C a separate stage, and that it is a delegate note. R-1..R-6 and the safeguards are untouched.
- Material paths: the six `MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40-47`) are the protocol, `iiac_activation_gate.py`, `iiac_common.py` and the three `verify_iiac_*.py`. The commit touches only the amendment file (`git show --stat`): none of the six.
- Tested or pinned artifacts: `grep -rIl IIAC_AMENDMENT_01_OPERATOR_DESIGNATED` over the tree (excluding `.git`, `.tmp`) finds only this campaign's documents, `IIAC_OPERATOR_SIGNING_DELEGATION.template.md` and `governance/tickets/T-IGX-021-A.md`; no script or test references the file; `governance/TRUSTED_POLICY_PIN.yaml` pins seven governance YAML files and not this document. `python -I scripts/governance/verify_trusted_policy_pin.py` exit 0.
- S-2 kept: consistent with the consented answer; S-2 stays strikeable by the Operator.

**Residual risks.** Mechanical enforcement does not exist: validators still read "operator" literally (Q-12, ENT delegate path requiring a signed delegation line). The note does not fix that; it is the separate stage (W4, batched with the IIAC re-audit / re-activation per DR-12 = B). Whether the note passes a non-author document review is for the review MASTER orders.

**Could not verify.** That no ticket-level test contract (e.g. T-IGX-021-A's guards) digests the amendment text: no tests run.

**Verdict: SIGNED.**
```text
Approved DR-2: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-3 (OD-HIST04-C) - TOOL-LIFE-X roadmap label

**Consented answer.** Option A: annotation on roadmap row L482, **no status change**.

**Verification.**
- The row is `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` line 482; its status cell reads `**[x] CLOSED - INDEPENDENTLY ACCEPTED** (acceptance baseline @ 1548640708c8...; publication **READY FOR INDEPENDENT PUBLICATION AUDIT** on GitHub)` (the file uses an em dash in the label).
- Signed DS-3 record (`CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md`, OD-ROUTE-DS3-2026-11-07, signed text lines 296-300): "TOOL-LIFE-X is neither accepted nor waived"; AUD-TOOLLIFE at `1548640708c850b3fbfca3f2db0e5ecfee4274dd` returned CONDITIONAL / chain gate STOP, C-1 four red tests, C-2 the F3 parent-cancel test; a re-audit is authorised after the correctives; "No roadmap label is changed by it; how the existing label is treated while the verdict is CONDITIONAL is not decided here". My annotation states only those facts. I **dropped** the HIST-04 sentence "no external review artefact exists in the repository", which I did not verify.
- Patch check: `git diff` is one hunk, one line; a character-level comparison of old and new line 482 shows a **pure insertion** (0 characters removed) between `on GitHub` and `) |`; the bold status token `**[x] CLOSED — INDEPENDENTLY ACCEPTED**` is intact (checked by string search). Annotation wording includes the required phrase "NOT supported by an independent PASS and must not be cited as current-HEAD proof".
- Pinned or tested artifacts: the roadmap is not in the trusted policy pin; roadmap tests assert other rows (`ROADMAP-REPLAY-X`, `_FORBIDDEN_DRIFT_SUBSTRINGS`; none of the forbidden substrings is introduced); the `HISTORICAL_LEDGER_ROWS` TOOL-LIFE-X entry in `tests/qualification/roadmap_replay_x/_historical_rows.py:1789` is a separate static data row, not compared with L482 by the tests I read. Ledger row L2059 and other mentions of the label are **not** annotated (out of the consented scope).

**Residual risks.** The label itself is unchanged and the cell still says "CLOSED"; the annotation is the only mitigation. The re-audit (Q-07) and a later keep-or-option-B decision remain open. L2059 and the HIST-03/HIST-04 cross-references still carry the old label without the annotation.

**Could not verify.** That no roadmap test breaks (not run); that no other tool parses the long status cell.

**Verdict: SIGNED.**
```text
Approved DR-3: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-4 (OD-HIST01-B) - promotion scope for FRZ-REG-01..10

**Consented answer.** The HIST-01 split: (b) for REG-07 and REG-10 only after the DR-10 replay; (a) for REG-01, REG-04, REG-05; REG-08 stays OPEN (not Path B); no REG promotion now.

**Verification.**
- Checklist line 66: "Live-scoped criteria ...: FRZ-PRD-01, FRZ-PRD-02, and the live-environment / operator-production-profile obligations within FRZ-PRD-03, FRZ-PRD-08, and FRZ-SEC-04". No FRZ-REG row is on it.
- WBS: `T-B7-PROMOTE-REG` (`ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` line 60) covers REG-01..10 as ordinary promotion after the FREEZE audit; `T-PATHB-EXCL` (line 45) lists only FRZ-PRD-01 and FRZ-PRD-02. REG-08 is in neither Path B artifact.
- HIST-01 handoff (`dossiers/hist01/HIST01_ACCEPTANCE_HANDOFF.md` lines 18-22) recommends exactly: (b) for REG-07/10 after N4, (a) for REG-01/04/05, REG-08 deferred. The consented split matches it. HIST-01's per-id matrix classes REG-08 as `PATH_B_LIVE_DEFERRED`; the campaign has already corrected that (`HISTORICAL_PATH_B_DEFERRED_OBLIGATIONS.md` line 10 and the disposition matrix: REG-08 is not on L66, the PROD-Q live dependency is recorded only as a dependency).
- Population A is closed only for its own scope (`d9ea0d6e8`), and REG-07 / REG-10 / REG-05 / REG-02 / REG-06 / REG-09 are marked `needs_head_replay` in HIST-01's matrix: consistent with sequencing (b) after the DR-10 replay on the final integrated SHA.

**Ruling recorded.**
1. REG-07, REG-10: scope (b): promotion considered only on Population A-D evidence at a pinned current SHA from the DR-10 replay, with a stated Population E residual; never before that replay.
2. REG-01, REG-04, REG-05: scope (a): promotion only on the ENT-AUDIT-X Population E rollup.
3. REG-08: stays OPEN; **not** Path B; L66 is not widened; the PROD-Q live dependency stays a recorded dependency.
4. No REG row is promoted now. Any later promotion needs an exact SHA, code, tests and an independent audit that did not author or certify the stage (checklist rule). No checklist, WBS, ticket or roadmap change is made by this ruling.

**Residual risks / gaps.** (1) **REG-02, REG-03, REG-06 and REG-09 are not covered by the HIST-01 split** and therefore by the consented answer: they stay OPEN with no scope assignment, and need their own ruling (or the Operator's explicit reading that they follow (a)). (2) REG-05 and REG-07 need a replay on current HEAD regardless of scope. (3) The split is HIST-01's recommendation, a worker thesis; this record certifies only that it is internally consistent with L66, the WBS and the handoff.

**Could not verify.** The ENT-AUDIT-X Population E rollup scope (another stream's material; out of scope).

**Verdict: SIGNED.**
```text
Approved DR-4: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-6 (OD-HIST03-B) - red PROD-Q P2-C tests: owner and route

**Consented answer.** Option 1: first an independent read-only root-cause check (Q-11), then a ticket with an owner.

**Verification (read-only; no test run, as instructed).**
- The question is still open on the base: the test file `tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` exists (7 test functions); its last commit is `e1312dcae` (2026-10-19), i.e. before `d1040f485` (SKILL-X Batch C, 2026-10-23T15:50:54+02:00, an ancestor of the base). `git blame -L485,493` on `execution_pinning_persistence.py` attributes the whole `wire_skill_execution_pinning_store` guard to `d1040f485`, unchanged since. Three later same-day SKILL-X Batch C commits (`963d652ff`, `595eb8191`, `4a466a568`, 15:57-16:14) touch the two files named in the finding. This ordering is consistent with a regression introduced by Batch C, but it does not prove the root cause (that is Q-11).
- The error text is at `intergrax/skills/execution_pinning_persistence.py:492` on `origin/development` (`"wire_skill_execution_pinning_store accepts kv_store or document_store, not both"`, raised when both stores are non-None, lines 485-493).
- No ticket exists for it: `governance/tickets/` has T-IGX-030-A/B/C only; none names F-015 or this function (grep on `governance docs` finds the string only in campaign documents).

**Ruling recorded.** Q-11 is commissioned: a read-only investigation by a fresh session (not HIST-03, not an implementer, not this session) of (i) whether the defect is a configuration conflict or a code defect, (ii) whether the test was green at the accepted PROD-Q P2-C SHA, (iii) which store is authoritative. Only after the Q-11 report may a ticket with an owner (SKILL-X / HIST-01 area or PROD-Q) be proposed; the ticket's plan approval is a separate act, and any `intergrax/ applications/` edit falls under DR-12 (batched with the IIAC re-activation). `cert-w0` does not perform Q-11 and cannot launch sessions: **MASTER must commission it.**

**Residual risks.** Until Q-11 and the ticket, evidence for rows citing P2-C HTTP admission (SEC-04, PRD-08, PRD-09) stays invalid. The worker root-cause thesis is unverified, as is "green at the accepted SHA".

**Could not verify.** The worker thesis (deliberately not tested).

**Verdict: SIGNED.**
```text
Approved DR-6: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-8 (OD-HIST02-A) - Tier-0 to Tier-3 import (FRZ-BND-03)

**Consented answer.** Option 2: a ticket to move the Tier-0 -> Tier-3 demo composition; an `ACCEPTED_LIMITATION` only if moving proves infeasible.

**Verification.**
- `intergrax/cli/external_work.py:338` on `origin/development` (991a3e410): `from applications.governed_contractor_application.host.collaborative_work_local_fixture import (` (a lazy import inside a function, lines 337-340 region). Still present.
- `AGENTS.md` lines 12-15: "Hard boundaries (never violate): intergrax/ MUST NOT import from agents/ or applications/".
- The "ratified tier exception" has no decision id: the only trace is `EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md` line 46 (`F10-C1-TIER-EXC-01`, "ratified in AC1-AC6 architecture audit"), an audit document.

**Ruling recorded.** Commission a ticket to move the CLI demo composition out of Tier-0 (to Tier-3). If, after a feasibility assessment inside that ticket's plan, moving is infeasible, an `ACCEPTED_LIMITATION` with a decision id is the fallback. The ticket's plan approval is a separate act and is not part of this delegation. Any `intergrax/` edit is a product-path change under DR-12 and ACT-N11: batched with the IIAC re-audit / re-activation. FRZ-BND-03 stays OPEN.

**Residual risks.** The cost of moving has not been estimated. Until the ticket lands, the exception stays unrecorded.

**Could not verify.** Feasibility or cost of the move.

**Verdict: SIGNED.**
```text
Approved DR-8: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## DR-9 (OD-HIST03-A) - PROD-Q / SEC scope rulings

**Consented answer.** "rec.": split SEC-06 without widening checklist L66; PRD-04 architecture-scoped; no waivers; PARTIAL/CONDITIONAL records unrewritten; SEC-05 and Decision-03 / D03-A rulings "as recommended in the decisions document".

**Key finding.** The decisions document (`HISTORICAL_AUTHORITY_AND_OPERATOR_DECISIONS.md`, DR-9) contains a recommendation for P-03, P-04 and the waivers, but **no recommendation for SEC-05 or for Decision-03 / D03-A**: it says only that "the meaning of the scope needs your ruling" and "which parts are Dimension A and which are infrastructure/live". The HIST-03 matrix likewise lists them as "ruling needed" (`HIST03_ARCHITECTURE_VS_LIVE_SCOPE_MATRIX.md` lines 70-71). The Operator's consent to "rec." cannot reach a ruling that has no recommendation, and `cert-w0` will not invent one in the Operator's name.

### DR-9a - P-03: split SEC-06, do not widen L66 - SIGNED
Verification: checklist L66 does not list FRZ-SEC-06; the row (line 223) is `OPEN` with note "E2B/Modal/Daytona production provider qualification remains OPEN (PROD-Q); AW-7C reference substrate does not close this row". The split is consistent with the checklist: (a) plugin-trust architecture, Dimension A, must reach PASS or N/A-with-evidence at ARCH-FREEZE; (b) per-provider live qualification, PROD-Q-P1, Dimension B. L66 is not widened; the row stays OPEN. **The checklist is not edited**; the candidate row note (HIST-03 P-03) stays a candidate for a separately reviewed change.

### DR-9b - P-04: PRD-04 architecture-scoped - SIGNED
Verification: checklist L66 does not list FRZ-PRD-04; by the checklist's own rule an unlisted criterion is architecture-scoped; the mechanical record `tests/qualification/prod_q/_p0_frz_reconciliation.py` names PRD-04 (lines 20, 81, 395) and, per the disposition matrix, hard-pins it as an in-scope blocker (I read the matrix text, not the logic of that file). Ruling: PRD-04 is architecture-scoped (the stricter reading) until the startup evidence is executed. Row stays OPEN. Caveats: the mechanical record needs an audited change once the startup evidence runs (not done, not part of this ruling); roadmap L478 places PROD-Q-R1 in the Path B live slice (F-004); the P2 evidence cited by the row includes P2-C, which is red (DR-6).

### DR-9c - SEC-05 meaning - NOT_SIGNED
Blocker: no recommendation exists in the decisions document (see key finding). The evidence is the Linux reference substrate only (checklist row 222: "global criterion remains OPEN"; HIST-03: scope meaning needs a ruling). Needs an explicit Operator ruling; the row stays OPEN and `AUTHORITY_REQUIRED`.

### DR-9d - Decision-03 / D03-A split (SEC-01, REL-03, REL-05, PRD-08) - NOT_SIGNED
Blocker: same. Which parts of the Kubernetes trust-anchor / global-PIS / Spanner + Azure-witness obligations are Dimension A and which are infrastructure or live has no recommendation. HIST-03 records Decision-03 as ratified but "NO BACKEND OR GLOBAL PIS AUTHORIZED" and D03-A as not ratified. Needs an explicit Operator ruling; the four rows stay OPEN and `AUTHORITY_REQUIRED`.

### DR-9e - waivers and unrewritten records - SIGNED
Verification: `HIST03_PRODQ_RESIDUAL_QUALIFICATION.md` lines 17, 30: AUTH-08 HTTP PARTIAL, G-08 async ContextVar NOT PROVEN, IG-F01 HTTP re-entry NOT PROVEN, "no waiver recorded"; DIM16 S2b is named in the decision request. Ruling: **no waiver is granted** for AUTH-08 HTTP, G-08, IG-F01 or DIM16 S2b; none is treated as waived. Whether each is architecture-mandatory or live-scoped is left to the audit that evaluates the row. No PARTIAL or CONDITIONAL record (P2-C partial log, AUTH-08 PARTIAL, TOOL-LIFE-X CONDITIONAL) is rewritten.

**Residual risks.** Rows depending on 9c and 9d stay `AUTHORITY_REQUIRED`: FRZ-SEC-05, FRZ-SEC-01, FRZ-REL-03, FRZ-REL-05, FRZ-PRD-08. No checklist change was made or is implied; if the checklist text must change for 9a/9b, a candidate patch needs an independent review.

**Could not verify.** Whether any validator hard-codes the DR-9 rows beyond the PRD-04 mechanical record.

**Verdict: SIGNED IN PART** (9a, 9b, 9e). The line below carries its scope in brackets so it cannot be read as a full DR-9 signature.
```text
Approved DR-9 [partial: P-03, P-04, waivers only; SEC-05 and Decision-03/D03-A NOT signed]: operator-delegate cert-w0 @ 2026-11-08T15:04:58+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-W0-2026-11-08.md@6a0a531fb6b94d41d6a00ce8bd800c7504749560
```

---

## 10. Final checks and what this session did not do

- Files changed by `cert-w0`: the delegation copy, the ratification record, this file, one paragraph in the Amendment 01 header block, one inserted annotation in roadmap row L482. Nothing under `intergrax/ agents/ applications/ scripts/ tests/`.
- No plan approval, FRZ promotion, checklist change, IIAC activation, ENT-AUDIT-X material, PR, push, merge, rebase, reset or amend.
- No test was run. `verify_trusted_policy_pin.py`: exit 0 before the patches; the post-patch run is reported in `CERT_W0_REPORT.md`.

## 11. Corrections after independent review (MASTER, 2026-11-08)

Written by the MASTER orchestrator after the independent review `rev-w0` (verdict ACCEPT_WITH_CHANGES, `audits/rev-w0/REVIEW.md`) and the Q-11 investigation (`audits/q11/Q11_REPORT.md`). This section is a correction record: no signed line above was re-stamped, and no signature was added or withdrawn.

- **F-1 (DR-3 wording).** The roadmap annotation quoted in the DR-3 section above ("this label is NOT supported by an independent PASS ...") overstated the evidence: the repository records an external review (`TOOL_LIFE_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md`, audit finding F-03: no review artefact exists in the repository). The annotation in the roadmap now reads: the only recorded independent acceptance is that external review, and the independent exact-SHA audit at the same SHA returned CONDITIONAL; the label must not be cited as current-HEAD proof. The quoted wording in the DR-3 section is superseded by this paragraph. The decision itself (option A, annotation only, status cell unchanged) is unchanged.
- **F-2 / F-8 (delegation record).** The delegation record was revised: the Polish quotation was removed (English-only rule) and its DR-9 row no longer claims a recommendation for SEC-05 or Decision-03 / D03-A (there is none; they were never signed). The revised record has sha256 `dc1ba29882b12241a260bf7a9a900e1a19d3c1c05a0899a8e1f19cd897f5217a`; the first version (sha256 `ea101bf8f104cd6cd8db0089d7937d8de44d01f4da7877ad3b9257ca718cd660`) is superseded. The seven `Approved` lines are rebound to the commit that carries the revised copy.
- **F-8 (authority).** The authority for these signatures is MASTER's transcription of one chat instruction and MASTER's reading that consent means the recommendations. For DR-1 in particular, Amendment 01 section 1.2 expects the Operator to confirm that the earlier EVOL-X lines were written on the Operator's authority; that confirmation is inferred, not explicit. An explicit Operator confirmation (chat line) is requested before the pull request is merged; it is to be transcribed in a new delegation record, not by editing this one.
- **F-3.** The Amendment 01 status note was reworded (section references and the "not in force" phrase).
- **F-4.** The `Delegation:` references point to a commit. The base history uses merge commits; the pull request must be merged with a merge commit (not squash or rebase), or a follow-up record must point to the post-merge sha.
- **F-5.** DR-6 section: `governance/tickets/` also holds T-IGX-020-A, 021-A, 022-A/B, 023-A and 024-A; the conclusion (no ticket names F-015 or `wire_skill_execution_pinning_store`) is unchanged. DR-8 section: the hard-boundary rule is in `AGENTS.md` at line 11 (heading) and line 14 (the rule), not lines 12-15.
- **F-6 / F-7.** All lines carry one batch time stamp; a future mechanical validator (T-IGX-021-B) needs a mapping from `DR-n` to the `OD-*` register ids - add it to that ticket's scope.
- **DR-6 follow-up (Q-11 result, read-only, fresh investigator).** The worker thesis is CONFIRMED by code reading and one run (1 failed, 1 passed, EXIT_CODE=1 at ba07ebaa3): code defect, not configuration or fixture leakage. Strict production composition always supplies both a KV store and a document store; the SKILL-X Batch C guard in `execution_pinning_persistence.py` rejects both; sibling wirings resolve coexisting stores to KV. Introduced by `d1040f485`. Recorded green at the P2-C accepted SHA `8e43b1604` (documents only; not re-run). Recommended owner: SKILL-X; recommended authoritative store: KV. These recommendations are NOT yet a decision: the ticket and its owner need a plan approval (DR-12 applies: the fix is under `intergrax/`). The optional empirical check (a run at the parent of `d1040f485`) was not performed.

## 12. Operator confirmation (MASTER, 2026-11-08)

The Operator answered the F-8 request with an explicit confirmation in chat. It is transcribed in the new record \`delegations/DELEG-HIST-W0-CONFIRMATION-2026-11-08.md\` (not signed by the Operator; assurance \`process_enforced\`). It confirms the separate-session reading, that DR-1 is covered, and that SEC-05 and Decision-03 / D03-A stay open. It contains no ruling on SEC-05, Decision-03 / D03-A or the scope of FRZ-REG-02/03/06/09: those remain open. No signed line above was changed.
