# PDC-X — Final acceptance report (PDC-P9-01) — answers recorded, awaiting the Operator's final confirmation

> **Operator's answers recorded (chosen by the Operator in chat (2026-11-16), awaiting final confirmation).** Transcription: [`operator_decisions/OD-PDC-X-CLOSE.md`](./operator_decisions/OD-PDC-X-CLOSE.md) (English rendering of the Operator's chat answer of 2026-11-16; unsigned: `Signed: <pending — the Operator confirms the final text in chat>`). Outcome A; all 18 "Operator decision" exceptions declared; the 13 rows open with no exception are documented debt (rows stay `PARTIAL`, no new tickets); FRZ option N; debt register accepted; limitations acknowledged; ARCH-FREEZE handoff published, ARCH-FREEZE not entered.

> **DRAFT — proposal for the Operator, not a verdict.** This report consolidates facts (each with its source file) and states the decisions the Operator is asked to sign **once**. It claims no certification of the stage, no closure, no FRZ promotion, no B6 verdict and no production readiness. Nothing in it is signed. Prepared at `origin/development` `f6f335aaef3d1e9ee204bd531f52f8a779566326` (PR #601 merge, 2026-11-16). `UNVERIFIED` marks anything I could not check against a file or `git`.

## 1. Decisions the Operator signs (single signature) and the answers chosen in chat

The orchestrator prepared the package under `DELEGATION-PDC-CLOSE.md` item 3; the Operator signs once. The "Answer" column shows what the Operator chose in chat on 2026-11-16 (chosen by the Operator in chat (2026-11-16), awaiting final confirmation); nothing is signed until the Operator confirms the final text of `operator_decisions/OD-PDC-X-CLOSE.md`.

| # | Decision | Answer (chosen by the Operator in chat (2026-11-16), awaiting final confirmation) | Where the exact text is |
| --- | --- | --- | --- |
| 1 | **Closure outcome.** A: close in the IIAC vocabulary, scoped (`PASS_SCOPED` covers the scoped rows only). B: the master-plan wording. C: do not close, report `PARTIAL_EXACT_BLOCKERS`. Roadmap §3.2 exit criteria are **not met literally** (global audit is `PASS_SCOPED`, not `PASS`; P7 domain verdicts were not re-graded; the certifier's manifest is not a stage certification); the choice is an Operator reading of those criteria. | **A** — scoped closure, IIAC canonical wording | `PDC_ROADMAP_CLOSURE_TEXT.md` sections 1, 2, 3, 6 |
| 2 | **Exceptions of the "Operator decision" kind**, outside `DELEGATION-PDC-P7-REPAIR.md` and `DELEGATION-PDC-CLOSE.md`: EX-D1-03, EX-D1-07, EX-D2-01, EX-D2-06, EX-D2-07, EX-D3-02, EX-D4-02, EX-D4-03, EX-D4-04, EX-D4-05, EX-D5-04, EX-D5-05, EX-D6-03, EX-D6-04, EX-D7-02, EX-D7-03, EX-D8-03, EX-D8-04. Declare them as exceptions (rows `PASS_SCOPED`), all or a named subset. | **All 18 declared as exceptions** | `PDC_EXCEPTION_REGISTER.md` section 2 |
| 3 | **Rows open with no exception** (D1 R009, R015, R017; D2 R004; D3 R005, R018, R019; D5 R002; D6 R007, R010, R015; D7 R015, R022): per row, an exception, a debt acknowledgement, or a small ticket. | **Documented debt; the 13 rows stay `PARTIAL`; no new tickets** | `PDC_DEBT_REGISTER.md` section 13; `PDC_EXCEPTION_REGISTER.md` sections 5, 6 |
| 4 | **Exceptions already authorized, recorded here for the signature:** `DELEGATION-PDC-CLOSE.md` decision 1 (F-D1-04, F-D5-02 remainder, F-D7-03 with row PDC-D7-R014 `PASS_SCOPED`, F-04 family and missing-feature findings, four D3 `fixture_strict_adapter_parity` tests) and `DELEGATION-PDC-EXCEPTIONS.md` (G-05 residual 77 scenario reds, E1-E11, G-10 residual `trace_bridge.v1`, G-28 Cypher store). Each has a reopen condition and owner path. | **Confirmed** | `PDC_EXCEPTION_REGISTER.md` sections 1-4 |
| 5 | **FRZ.** Option N (no promotion, evidence notes only; counters 225 / 26) or Option P (only after a further FRZ-row-level independent audit names the FRZ ids; none exists today). | **N** — no promotion; 225 / 26 | `PDC_FRZ_PROMOTION_PROPOSAL.md` section 4 |
| 6 | **Debt register** (LOW and INFO findings, WAVE2 decision-5 items, G-13..G-27) accepted as documented debt, not repaired. | **Accepted** | `PDC_DEBT_REGISTER.md` |
| 7 | **Limitations stated as facts** (no decision on their truth, only acknowledgement): the EBH-7 L3 replay (run `43738431866`) failed and was **not re-run**, so **B6 is not passed**; the EBH-7 identity gate stays a known red class; the C10/C15 freeze is **not lifted**; **no production-readiness claim**; rows were not re-certified one by one; ECAP FULL, the contract suites and the combined pytest were last run at `d39a520dd`, not at `G`. | **Acknowledged** | section 7 below |
| 8 | **ARCH-FREEZE handoff** published as written; ARCH-FREEZE is **not entered**. | **Published as written; ARCH-FREEZE not entered** | `PDC_ARCH_FREEZE_HANDOFF.md` |

**Signature block** (not filled by the orchestrator, an auditor, the certifier or a delegate; the Operator's answers above are transcribed in `operator_decisions/OD-PDC-X-CLOSE.md`):

```text
Signed: <pending — the Operator confirms the final text in chat>
```

## 2. Status

| Item | Value |
| --- | --- |
| Stage | PDC-X, Platform Domain Completeness and Certification (roadmap §3.2) |
| Overall result | **Not decided by this document** (the Operator chose outcome A in chat; final confirmation pending). Facts: global audit `PASS_SCOPED` at `G`; certifier binding manifest `PASS` (scoped, 19 limitations); stage closure pending the Operator's signature |
| Global audit (P8, IA-GLOBAL) | **`PASS_SCOPED`** at `G` = `969cbd964c679d563b02e55f065b452d46fdf90b`, "only" with the declared exceptions; "not a PASS", "not a repeat-run PASS", "not a verdict on B6 or on certification" (`PDC_INDEPENDENT_AUDIT.md` section 15, merge #597 `a73d15ab3`). It was `PARTIAL` at `d39a520dd` (13.7) and at `969cbd964` before the exception record (14.6) |
| Certifier (CERT-PDC) | binding manifest `PASS`, scoped, 19 limitations; "not a verdict on B6, not a certification of the whole PDC-X stage and not an FRZ or roadmap action" (`PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md`; matrix `PDC_DOMAIN_CERTIFICATION_MATRIX.md`) |
| IIAC activation (ACT-N11) | `IN_FORCE` from phase-1 commit `9f05749bc23592868f3d0a7f505f9edbb2cfd025`, phase 2 `293bb5b3c`, merged as PR #601 `f6f335aae` (`IIAC_POLICY_ACTIVATION_RECORD.json`). Any later product-path change voids it |
| EBH-7 L3 replay (final re-pin at `d39a520dd`) | run `43738431866`: 2 failed, 720 passed, 9 skipped; the two failures were test defects (K and O gate tests append duplicate paths), repaired by T-IGX-041-Q (#593) but **never replayed** (`DELEGATION-PDC-LOOP2.md` condition); **not a B6 PASS** |
| Roadmap closure text | [`PDC_ROADMAP_CLOSURE_TEXT.md`](./PDC_ROADMAP_CLOSURE_TEXT.md) (proposal; roadmap not edited) |
| FRZ | [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md) (proposal; checklist not edited; counters 225 / 26) |
| Registers | [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md) |
| ARCH-FREEZE handoff | [`PDC_ARCH_FREEZE_HANDOFF.md`](./PDC_ARCH_FREEZE_HANDOFF.md) |

## 3. SHA ladder (facts; the authoritative table is `PDC_SHA_LADDER.md`)

| Role | SHA | Source |
| --- | --- | --- |
| BASELINE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | `PDC_SHA_LADDER.md` |
| INTEGRATED_BASELINE_SHA (P6 product tree) | `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (PR #537) | `PDC_SHA_LADDER.md` |
| MCP-RC joint re-pin 2 | `61531828f5728acc8f2ed357b4b835d7f5dc4ebc` (PR #554) | `git log origin/development` |
| `Dn_AUDITED_SHA` (P7, all eight domains; pre-repair) | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (PR #557) | `PDC_P7_D<n>_AUDIT_REPORT.md` |
| P7 repair merges T-IGX-041-A..I | #574 `4e0fe053f`, #575 `4a0f2e197`, #569 `58916e145`, #571 `904433080`, #568 `aa3b3e189`, #573 `a2b24bac3`, #570 `996671a05`, #572 `dd58baa4d`, #576 `63005e290` | `git log origin/development --merges`; audit section 2 |
| P8 audit start | `cba32d2479cd134911d2fdf4fb1db4587c8a94d8` (PR #578) | audit header |
| Repair loop 1 (P8): T-IGX-041-K #581 `2d4e99dd6`, -L #582 `fe19bbf0b`, -N #583 `6257e3de5`, -M #584 `258bbf6ab`, -O #585 `d39a520dd`; EBH-7 final re-pin (T-IGX-041-J) at `d39a520dd`: #580 `bc7c1a685`, W1' #586, W2' record #598 `6c62db426` | `d39a520dd` is the frozen audit SHA of audit section 13 and the EBH-7 pin | `git log`; audit 13; `PDC_SHA_LADDER.md` |
| Repair loop 2: T-IGX-041-P #594 `7f1859e0f`, T-IGX-041-Q #593 | `969cbd964c679d563b02e55f065b452d46fdf90b` = **`G`** (`GLOBAL_AUDITED_SHA`, `corrective_implementation_sha`, `test_execution_sha`) | audit 14; certifier report |
| `E` = `audit_execution_sha` = `evidence_publication_sha` | `a73d15ab341b30e9748aad03894f6faa75ad4e09` (PR #597) | certifier report |
| `C` = `certification_publication_sha` | `fd2ad28efde6216be38194b020188d116264adb6` | certifier report |
| `governance_implementation_sha` | `2ae1014116d353bc7742985e62f2bccc2bfac798` | certifier report |
| Activation phase 1 / phase 2 / merge | `9f05749bc23592868f3d0a7f505f9edbb2cfd025` / `293bb5b3c` / `f6f335aaef3d1e9ee204bd531f52f8a779566326` (PR #601) | `IIAC_POLICY_ACTIVATION_RECORD.json`; `git log` |
| `development` tip at drafting | `f6f335aaef3d1e9ee204bd531f52f8a779566326` | `git rev-parse origin/development`; **re-read before signing** |
| FINAL_CLOSURE_PUBLICATION_SHA | `<FINAL_CLOSURE_PUBLICATION_SHA>` — exists only after the closure pull request merges | P9 merge |

## 4. Domain dispositions

P7 results are at the **pre-repair** audited SHA. The "P8 re-verification" column is the IA-GLOBAL count of repaired findings (certifier matrix section 3, which applies the audit's section 13.2 changes). The certifier disposition is **quoted**, not re-derived; the P7 verdicts are not rewritten; **no row-level re-grade exists** (so the D5 row PDC-D5-R015, a P7 `FAIL`, has no new grade: UNVERIFIED).

| Domain | P7 verdict | P7 rows | P7 findings | Repairs (PR) | P8 re-verification (fixed / partial) | Certifier disposition (quoted from the matrix) |
| --- | --- | --- | --- | --- | --- | --- |
| D1 AGENT-LIFECYCLE | PARTIAL | 10 PASS, 1 PASS_SCOPED, 11 PARTIAL, 0 FAIL | 3 HIGH, 6 MEDIUM, 7 LOW, 4 INFO | T-IGX-041-A #574; -M #584 | 10 / 2 (F-D1-02 and F-D1-03 fixed **at service level only, inert in production**; F-D1-11 durable CAS partial) | "PASS_SCOPED only inside the IA-GLOBAL PASS_SCOPED"; HIGH items stay open by authorization, "inert in production, not fixed" |
| D2 CONTEXT-KNOWLEDGE | PARTIAL | 10 PASS, 1 PASS_SCOPED, 7 PARTIAL | 0 HIGH, 8 MEDIUM, 10 LOW, 2 INFO | T-IGX-041-B #575; -M #584; -P #594 | 7 / 3 (F-D2-08, F-D2-10, F-D2-18) | "PASS_SCOPED inside the global verdict"; G-28 accepted exception |
| D3 MODEL-RUNTIME | PARTIAL | 12 PASS, 3 PASS_SCOPED, 9 PARTIAL | 1 HIGH, 6 MEDIUM, 6 LOW, 3 INFO | T-IGX-041-C #569; -M #584 | 5 / 2 (F-D3-02 pins, F-D3-05 stale tests) | "PASS_SCOPED inside the global verdict. The 77 reds stay red" |
| D4 HOST-RUNTIME | PARTIAL | 9 PASS, 3 PASS_SCOPED, 8 PARTIAL | 0 HIGH, 9 MEDIUM, 7 LOW, 4 INFO | T-IGX-041-D #571 | 9 / 0 | "PASS_SCOPED inside the global verdict" |
| D5 EVALUATION | **FAIL** (one row, PDC-D5-R015) | 11 PASS, 6 PARTIAL, 1 FAIL | 1 HIGH, 7 MEDIUM, 6 LOW, 4 INFO | T-IGX-041-E #568; -M #584 | 11 / 0 (F-D5-01 `VERIFIED_FIXED` for writer, reader and the three host call sites) | "PASS_SCOPED inside the global verdict. The row verdict PDC-D5-R015 FAIL is not rewritten"; strict host with `tenant_id=None` accepted (EX-L-09) |
| D6 RUNTIME-SECURITY | PARTIAL | 14 PASS, 3 PASS_SCOPED, 11 PARTIAL | 0 HIGH, 6 MEDIUM, 9 LOW, 5 INFO | T-IGX-041-F #573; -M #584 | 5 / 0 (F-D6-12 fixed by M) | "PASS_SCOPED inside the global verdict" |
| D7 BACKGROUND-SCHED | PARTIAL | 10 PASS, 1 PASS_SCOPED, 11 PARTIAL, 1 FAIL (PDC-D7-R014) | 3 HIGH, 10 MEDIUM, 7 LOW, 3 INFO | T-IGX-041-G #570; -M #584 | 10 / 2 (F-D7-07, F-D7-12; F-D7-02 fixed by M) | "PASS_SCOPED inside the global verdict"; PDC-D7-R014 `PASS_SCOPED` by authorization |
| D8 ARTIFACT-LIFECYCLE | PARTIAL | 8 PASS, 1 PASS_SCOPED, 10 PARTIAL | 1 HIGH, 6 MEDIUM, 9 LOW, 4 INFO | T-IGX-041-H #572; -M #584 | 7 / 3 (F-D8-03, F-D8-04, F-D8-17) | "PASS_SCOPED inside the global verdict" |
| Total | | | | | 64 / 12 by the certifier's recount; **63 / 13 by the auditor** (audit 13.2); unresolved difference of one row, changes no disposition | |
| TENANT-X carrier-count re-pin | n/a | n/a | F-D2-03, F-D3-06, F-D6-06 (test-side) | T-IGX-041-I #576 | TENANT-X 213 passed at `cba32d247` (not re-run at `G`) | "not re-verified here" |
| Cross-domain XD-01..XD-12 | not part of P7 | | | P6 repairs | XD-01..XD-11 green with no mocks on mandatory paths and XD-12 red at `cba32d247` (G-06, now `VERIFIED_FIXED`); `pdc_x` suite had no red at `d39a520dd` except an environment artifact (audit 13.2); **not re-run at `G`**; a restated "XD PASS" is UNVERIFIED | |

Sources for the P7 counts: the "Overall verdict", "Rows" and "Findings" lines of each audit report; the D8/D6 counts differ from earlier orchestrator notes only where the reports state it.

## 5. Repairs and audit chain (facts)

| Step | Fact | Source |
| --- | --- | --- |
| P7 audit | eight independent domain audits at `c08efe46a` (PRs #559-#566): 7 `PARTIAL`, D5 `FAIL` | audit reports |
| Repair scope | Option B (real defects and hygiene fixed; "no production caller" and "missing feature versus claim" become declared exceptions; one TENANT-X re-pin) | `DELEGATION-PDC-P7-REPAIR.md` |
| P7 repairs | T-IGX-041-A..I merged #568-#576 | `git log` |
| Closing wave | declared exceptions, one global audit with one repair loop, merges, activation, package for one signature | `DELEGATION-PDC-CLOSE.md` (#578) |
| P8 global audit | first verdict `PARTIAL` at `cba32d247`: 1 HIGH (G-01) and 11 MEDIUM defects or register inaccuracies; 76 repaired rows audited | `PDC_INDEPENDENT_AUDIT.md` sections 1, 7, 9 |
| Repair loop 1 | T-IGX-041-K, -L, -N, -M, -O (#581-#585): closed G-01, G-02, G-03, G-06 and F-D6-12; partly G-04, G-05, G-10; excluded E1-E11 | audit 13 |
| Repair loop 2 (Operator: "one loop, no further loop"; condition: only the tests that cover the fixes) | T-IGX-041-P (G-23, graph evidence visibility; write set widened to three files by the Operator's "Yes, 3 files") and T-IGX-041-Q (K/O gate test de-duplication); delta audit: 64 targeted tests passed, mutation executed | `DELEGATION-PDC-LOOP2.md`; audit 14 |
| Exception record | "Yes, all as exceptions" (2026-11-16) | `DELEGATION-PDC-EXCEPTIONS.md` |
| Final audit verdict | `PASS_SCOPED` at `G` | audit 15 |
| Certification / activation | binding manifest `PASS` (19 limitations); activation line by the delegate under `DELEGATION-PDC-P8-ACT.md` (a)-(c); `IN_FORCE` | certifier report; activation record |
| Implementation checkpoints | tickets T-IGX-041-A, -D, -E, -G, -H, -I had no "Implementation checkpoint" section and all nine A..I read `plan_approved` when read at `cba32d247`; tickets J..Q were **not read** for this (UNVERIFIED) | exception register section 6, item 6 |

## 6. Requirement coverage

| Metric | Value |
| --- | --- |
| Requirement rows | 172 (D1 22, D2 18, D3 24, D4 20, D5 18, D6 28, D7 23, D8 19); `pdc_x_requirements.py --check` OK per the audit (check 9); not re-run here |
| Row-level final rating counts | **None exist.** The audit did not re-certify rows one by one (check 15 `PASS_SCOPED`); the certifier records domain dispositions only. A row-level count would be this package's proposal and is not claimed |
| Rows proposed `PASS_SCOPED` by exception (draft) | `PDC_EXCEPTION_REGISTER.md` section 5 |
| Rows open with no exception (draft) | D1 R009, R015, R017; D2 R004; D3 R005, R018, R019; D5 R002; D6 R007, R010, R015; D7 R015, R022 (decision 3) |
| Other audit checks (global audit section 6, at `cba32d247`) | ECAP FULL 85/85, 0 findings; DUP-X/EBH-4 gates 130 passed; TENANT-X 213 passed. At `d39a520dd`: one bounded pytest process 118 failed, 7691 passed (classified in audit 13.3). At `969cbd964`: targeted tests only (64 passed) |

## 7. Limitations (stated as facts; the Operator acknowledges them in decision 7)

1. Exceptions: [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md) (EX-G-01..07, EX-D1..D8, EX-L-01..14), each with a reopen condition. Until the signature they are **proposed**, not approved.
2. Debt: [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md); known reds and process debt in its section 11.
3. **No production readiness**, no live provider qualification (Path B, `OD-PDC-PATHB-01`); `FRZ-PRD-01`, `-PRD-02` and the live parts of `FRZ-PRD-03`, `-PRD-08`, `FRZ-SEC-04` stay `OPEN`.
4. **`dist_scale_x` and Docker (C10/C15) are not qualified; the freeze is not lifted.**
5. **B6 is not passed.** The EBH-7 L3 replay failed (run `43738431866`) and was not re-run; the EBH-7 identity gate stays a known red class. No B6 claim is made.
6. `PASS_SCOPED` means **the scoped rows only**. Product commits after `G` void the audit for the changed paths and the IIAC activation (ACT-N11).
7. Not re-certified row by row; not re-run at `G`: ECAP FULL, the contract suites, TENANT-X, the XD suite (Operator condition of `DELEGATION-PDC-LOOP2.md`).
8. The F-D1-02 / F-D1-03 repairs are inert in production; F-D5-01 leaves the strict-host `tenant_id=None` case; the streaming resilience claim does not hold (EX-L-08); multi-instance queue claims are limited (EX-L-03).
9. MCP-RC carry-overs (owner of the strict xfail, RA-02..RA-04 acceptance) are open per the certifier matrix; F-35 and F-56 were not re-verified by the certifier (UNVERIFIED).
10. The held dependency-bot / `uv.lock` pull requests: merge state **UNVERIFIED** (not read in this pass).

## 8. Independence and provenance

| Role | Session | Note |
| --- | --- | --- |
| Implementers (`T-IGX-036-*`, `T-IGX-041-*` lanes) | labels in `iiac_evidence/PDC-X/P8/activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_P8_04_ACTIVATION.md`; the certifier notes they are "labels, not verified application ids" | never wrote `Approved:` or activation lines |
| Plan signers (delegates) | `Approved: operator-delegate @ ...` lines in the tickets (T-IGX-041-A..Q) | did not implement or audit what they signed (per the delegation text; not independently verified here) |
| Domain auditors IA-D1..IA-D8 | the eight `iiac_audit_session_manifest.*` files | clean sessions; separate from implementers |
| Global auditor IA-GLOBAL | `PDC-P8-IA-GLOBAL-INDEPENDENT-AUDITOR-6573093f-9178-4c2c-93a3-64b0a5b6c657` | one session for sections 1-15; did not implement, sign or certify |
| Certifier CERT-PDC | `local_a6815d6f-adea-401e-8a67-bd6c5e4b7791` (agent id `certification-governance-agent`) | not auditor, not implementer, not delegate |
| Activation delegate | UNVERIFIED id (named in the activation record's handoff) | wrote the plain-form activation line `@ 2026-11-16T09:06:54+01:00` |
| Orchestrator | session `local_80099b72-d308-4d1a-a3cb-99adbd0abe51` | prepared this package; never edits an auditor or certifier verdict; **assurance level `process_enforced`: one git identity signs all commits, separation rests on session separation and manifest fields** |
| Operator | | single signature at the end: `<Operator signature and date>` |
