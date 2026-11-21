# PDC-X — ARCH-FREEZE handoff (PDC-P9-05) — DRAFT for the Operator's signature

> **DRAFT — proposal for the Operator, not a verdict.** This document claims no certification, no closure, no FRZ promotion and no approval, and **does not enter ARCH-FREEZE**. It lists what ARCH-FREEZE may rely on and what it must not assume. Prepared at `origin/development` `f6f335aaef3d1e9ee204bd531f52f8a779566326` (PR #601 merge, 2026-11-16). Companion drafts: [`PDC_FINAL_ACCEPTANCE_REPORT.md`](./PDC_FINAL_ACCEPTANCE_REPORT.md) (decisions the Operator signs), [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md), [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md), [`PDC_ROADMAP_CLOSURE_TEXT.md`](./PDC_ROADMAP_CLOSURE_TEXT.md).

## 1. Status of this handoff

| Item | Value |
| --- | --- |
| Handoff status | **`HANDOFF READY (SCOPED)`** if the Operator confirms closure outcome A (chosen in chat on 2026-11-16, awaiting final confirmation; decision 8: handoff published as written, ARCH-FREEZE not entered); `PARTIAL_EXACT_BLOCKERS` only if the Operator does not confirm |
| P8 global audit verdict | **`PASS_SCOPED`** at `G` = `969cbd964c679d563b02e55f065b452d46fdf90b` (`PDC_INDEPENDENT_AUDIT.md` section 15), only with the declared exceptions; not a PASS |
| Certifier | binding manifest `PASS` scoped (19 limitations); not a certification of the stage |
| IIAC activation | `IN_FORCE` from phase-1 commit `9f05749bc23592868f3d0a7f505f9edbb2cfd025` |
| EBH-7 L3 replay (final re-pin) | run `43738431866` **FAILED** (2 failed, 720 passed, 9 skipped), **not re-run**; **not a B6 PASS and not a certification verdict** |
| FINAL_CLOSURE_PUBLICATION_SHA | `feaa2a87d41c3c8ae95724590291f360d0232016` (PR #603) |
| Operator signature | `<Operator signature — not written by the orchestrator, an auditor, the certifier or a delegate>` |

ARCH-FREEZE is **not entered** by this document. `PDC-X` does not perform the freeze-SHA audit or VBN acceptance (roadmap §3.2, "Non-negotiable boundaries"); those remain with ARCH-FREEZE.

## 2. Exact SHAs (read from `origin/development`)

| Role | SHA | Source |
| --- | --- | --- |
| `development` tip at drafting | `f6f335aaef3d1e9ee204bd531f52f8a779566326` (PR #601 merge) | `git rev-parse origin/development`; **re-read before signing**, it moves |
| BASELINE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | `PDC_SHA_LADDER.md` |
| INTEGRATED_BASELINE_SHA (P6 product tree) | `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (PR #537) | `PDC_SHA_LADDER.md`; superseded as the EBH-7 pin by `d39a520dd` |
| MCP-RC joint re-pin 2 (MCP-SERVER-X ACT-N11 on the integrated tree) | `61531828f5728acc8f2ed357b4b835d7f5dc4ebc` (PR #554) | `git log`; the global audit confirms it is an ancestor of the audit SHA (audit section 4) |
| P7 audited SHA (all eight domain audits; pre-repair) | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (PR #557) | the eight `PDC_P7_D<n>_AUDIT_REPORT.md` |
| P7 audit report merges | #559 `35c3a21c1` (D7), #560 `c000d42a0` (D1), #561 `4a2f6b428` (D3), #562 `3b371ef9d` (D5), #563 `380037300` (D2), #564 `1a4a29746` (D8), #565 `cede66986` (D6), #566 `1c864ea19` (D4); repair-scope record #567 `f87df3b2a` | `git log origin/development --merges` |
| P7 repair merges (T-IGX-041-A..I) | section 3 | `git log` |
| EBH-7 final re-pin `P` (T-IGX-041-J) | `d39a520dda965981a38f8e1a90ec5135ea5d508e` (merge of PR #585; pin PRs #580, #586, record #598) | `PDC_SHA_LADDER.md` row `EBH7_P7_FINAL_REPIN_TEST_EXECUTION_SHA` |
| P8 global audit start | `cba32d2479cd134911d2fdf4fb1db4587c8a94d8` (PR #578) | audit header |
| P8 repair loop 1 (T-IGX-041-K, -L, -N, -M, -O) | #581 `2d4e99dd6`, #582 `fe19bbf0b`, #583 `6257e3de5`, #584 `258bbf6ab`, #585 `d39a520dd` | `git log` |
| P8 repair loop 2 (T-IGX-041-P, -Q) | #594 `7f1859e0f`, #593 `969cbd964` | `git log` |
| **GLOBAL_AUDITED_SHA** `G` (= `corrective_implementation_sha` = `test_execution_sha`) | `969cbd964c679d563b02e55f065b452d46fdf90b` | audit 14, 15; certifier report |
| `E` = `audit_execution_sha` = `evidence_publication_sha` | `a73d15ab341b30e9748aad03894f6faa75ad4e09` (PR #597) | certifier report |
| `C` = `certification_publication_sha` | `fd2ad28efde6216be38194b020188d116264adb6` | certifier report |
| Activation phase 1 / phase 2 / merge | `9f05749bc23592868f3d0a7f505f9edbb2cfd025` / `293bb5b3c` / `f6f335aae` (PR #601) | `IIAC_POLICY_ACTIVATION_RECORD.json` |
| FINAL_CLOSURE_PUBLICATION_SHA | `feaa2a87d41c3c8ae95724590291f360d0232016` | P9 merge |

## 3. Repair merges on `development`

Ticket mapping follows `DELEGATION-PDC-P7-REPAIR.md` (A = D1 … H = D8, I = TENANT-X re-pin) and was checked against each plan header and merge (PR number and merge SHA from `git log origin/development --merges`).

| Ticket | Domain | PR | Merge SHA |
| --- | --- | --- | --- |
| T-IGX-041-A | D1 AGENT-LIFECYCLE | #574 | `4e0fe053f68cc68c99891cccc73aa59c3ed01e87` |
| T-IGX-041-B | D2 CONTEXT-KNOWLEDGE | #575 | `4a0f2e197c76a9dc900f82d96500547e9b047f26` |
| T-IGX-041-C | D3 MODEL-RUNTIME | #569 | `58916e145f99878f287e76d365d21df80a9ac649` |
| T-IGX-041-D | D4 HOST-RUNTIME | #571 | `904433080abcb3b7d0e240fab0f985e0a160573b` |
| T-IGX-041-E | D5 EVALUATION | #568 | `aa3b3e1893ce3106972a43fc016c31d2585cea6e` |
| T-IGX-041-F | D6 RUNTIME-SECURITY | #573 | `a2b24bac35f2854dcb71de078374be4afd5ebef2` |
| T-IGX-041-G | D7 BACKGROUND-SCHED | #570 | `996671a05eccf2d3a2227d7d1647ba605bcd591f` |
| T-IGX-041-H | D8 ARTIFACT-LIFECYCLE | #572 | `dd58baa4da6f04d1fcb8c7be26ca3360ec3db847` |
| T-IGX-041-I | TENANT-X carrier-count re-pin | #576 | `63005e2902ed7724dcafdc036564bcbd9c3deb08` |
| T-IGX-041-J | EBH-7 final re-pin (pin `d39a520dd`) | #580 `bc7c1a685`, #586 `8115de877`, record #598 `6c62db426` | pin `d39a520dd` |
| T-IGX-041-K, -L, -N, -O, -M | P8 repair loop 1 | #581, #582, #583, #585, #584 | see section 2 |
| T-IGX-041-P, -Q | P8 repair loop 2 | #594, #593 | `7f1859e0f`, `969cbd964` |

Earlier P6 facts that stay valid history: T-IGX-036-X product fix PR #537 `146e7acf1`; omnibus debt repair T-IGX-036-Z PR #549 `622a61b962ee9941e4d8d4e4823bdd54063a3e87`.

## 4. What ARCH-FREEZE may rely on

| Item | Where | Notes |
| --- | --- | --- |
| Stage chain P0-P9 | `PDC_SHA_LADDER.md` | the global audit verified 45 / 45 ladder SHAs as existing ancestors at `cba32d247` (check 10); rows added by this package are read from `git log` and the activation record, **ancestry not re-run here** (UNVERIFIED) |
| Domain dispositions D1-D8 | `PDC_FINAL_ACCEPTANCE_REPORT.md` section 4; `PDC_DOMAIN_CERTIFICATION_MATRIX.md` | each "PASS_SCOPED inside the global verdict"; **not** row-level grades |
| Cross-domain scenarios XD-01..12 | `tests/qualification/pdc_x/xd/` | historical run at `884c177ad` (927 passed, 5 skipped); XD-01..XD-11 green and XD-12 red at `cba32d247`, XD-12 fixed by T-IGX-041-M; **not re-run at `G`** |
| Global audit | [`PDC_INDEPENDENT_AUDIT.md`](./PDC_INDEPENDENT_AUDIT.md) | `PASS_SCOPED` at `G`; ECAP FULL 85/85 at `cba32d247` (the only FULL run; not repeated by Operator condition) |
| Certification and activation | [`PDC_DOMAIN_CERTIFICATION_MATRIX.md`](./PDC_DOMAIN_CERTIFICATION_MATRIX.md), [`PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md`](./PDC_P8_04_ACTIVATION_CERTIFIER_REPORT.md) | binding manifest `PASS` scoped; not a stage certification |
| Exception register | [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md) | declared, labelled exceptions with reopen conditions; none silently upgraded; valid only after the Operator signs |
| Debt register | [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md) | documented debt |
| FRZ proposal | [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md) | counts stay 225 / 26; no FRZ row is promoted by this package |

## 5. What the freeze must not assume (limitations carried by PDC-X)

1. **No production readiness.** PDC-X certifies Dimension A (architecture) only; live provider and environment qualification stays with `PROD-Q` (Path B, `OD-PDC-PATHB-01`). `FRZ-PRD-01`, `FRZ-PRD-02` and the live portions of `FRZ-PRD-03`, `FRZ-PRD-08`, `FRZ-SEC-04` stay `OPEN`.
2. **`dist_scale_x` and Docker (C10/C15) are not qualified; the freeze is not lifted** (`DELEGATION-PDC-WAVE2.md` decision 4; `DELEGATION-PDC-EXCEPTIONS.md`): original Windows environment, no Docker migration, no new vendor integrations, no production configuration.
3. **Declared exceptions** of `DELEGATION-PDC-CLOSE.md` decision 1, `DELEGATION-PDC-P7-REPAIR.md` (option B) and `DELEGATION-PDC-EXCEPTIONS.md`: see the exception register for ids, affected rows (proposed `PASS_SCOPED`) and reopen conditions.
4. **Pre-existing authorized exclusions** `OD-PDC-SCOPE-02`, `OD-PDC-SCOPE-03`, `OD-PDC-PATHB-01`.
5. **B6 is not passed.** The EBH-7 L3 replay (run `43738431866`) failed and was not re-run. The EBH-7 identity gate remains a known red class. No B6 claim is made (`DELEGATION-PDC-LOOP2.md`; `DELEGATION-PDC-EXCEPTIONS.md`).
6. **`PASS_SCOPED` covers the scoped rows only**; it is not a PASS and rows were not re-certified one by one.
7. **Known red classes at `G`** (audit 13.3, certifier report): EBH-7 identity gate and F10 classifier (9 at `d39a520dd`), `roadmap_replay_x` (5), `dist_scale_x` (3, PostgreSQL handshake/Docker), the 77 incident scenario reds (EX-L-01), the `trace_bridge.v1` test (EX-L-13), order-dependent reds in combined runs (G-24, debt). IIAC activation staleness is **resolved** by the P8-04 rebind.
8. **Pin validity.** IIAC activation is bound to `G`; any change under `intergrax/`, `agents/` or `applications/` after `audit_execution_sha` invalidates it (ACT-N11).
9. **Held dependency-bot / `uv.lock` pull requests** are merged in one batch after the final re-pin is recorded (`DELEGATION-PDC-WAVE2.md` decision 7): batch state **UNVERIFIED**.
10. **Inert or limited repairs** (facts from the audit): F-D1-02/-03 fixed at service level only and inert in production; strict untenanted host (G-09); streaming resilience claim (G-11); queue recovery not CAS (G-07).

## 6. Dependency record for ARCH-FREEZE

The ARCH-FREEZE dependency on `PDC-X` is satisfied **only** by the roadmap status published after the closure pull request (`OD-PDC-ARCHF-01`: note-only WBS header, task count stays 54; closure checked at `T-B8-ROADMAP` through the roadmap section 5 ledger). This document edits neither `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md`, the freeze checklist nor the roadmap. The FRZ traceability validator (`scripts/governance/verify_arch_freeze_plan_traceability.py`, `FRZ=225 families=26`) must stay green on the closure pull request: **not run in this docs-only pass** (`<validator result at the closure SHA>`).

## 7. Session end

PDC-P9-06: publish; verify merged remote HEAD and CI; session-end branch check (`git branch --show-current`). `<result, after the closure merge>`.
