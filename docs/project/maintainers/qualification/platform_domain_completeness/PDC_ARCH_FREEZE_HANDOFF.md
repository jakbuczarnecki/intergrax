# PDC-X — ARCH-FREEZE handoff (PDC-P9-05) — SKELETON

> **DRAFT — proposal for the Operator, not a verdict.** This skeleton claims no certification, no closure, no FRZ promotion and no approval. Every `<placeholder>` is filled only after the named result exists (P8 audit, EBH-7 replay, Operator signature). Prepared under `operator_decisions/DELEGATION-PDC-CLOSE.md` item 3 for the Operator's single signature at the end of the wave. Companion drafts: [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md), [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md), [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md), [`PDC_ROADMAP_CLOSURE_TEXT.md`](./PDC_ROADMAP_CLOSURE_TEXT.md), [`PDC_FINAL_ACCEPTANCE_REPORT.md`](./PDC_FINAL_ACCEPTANCE_REPORT.md).

## 1. Status of this handoff

| Item | Value |
| --- | --- |
| Handoff status | `<HANDOFF READY (SCOPED) | PARTIAL_EXACT_BLOCKERS>` — decided only after the P8 verdict and the Operator's signature |
| P8 global audit verdict | `<P8 verdict>` |
| GLOBAL_AUDITED_SHA | `<GLOBAL_AUDITED_SHA>` |
| EBH-7 replay (final re-pin) | run id `<replay run id>`; artifact SHA-256 values `<recorded in the W-style replay record>`; **not a B6 PASS and not a certification verdict** |
| FINAL_CLOSURE_PUBLICATION_SHA | `<FINAL_CLOSURE_PUBLICATION_SHA>` |
| Operator signature | `<Operator signature — not written by the orchestrator, an auditor or a delegate>` |

ARCH-FREEZE is **not entered** by this document. `PDC-X` does not perform the freeze-SHA audit or VBN acceptance (roadmap §3.2, "Non-negotiable boundaries"); those remain with ARCH-FREEZE.

## 2. Exact SHAs known today (read from `origin/development`)

| Role | SHA | Source |
| --- | --- | --- |
| `development` tip at drafting | `cba32d2479cd134911d2fdf4fb1db4587c8a94d8` (PR #578 merge) | `git rev-parse origin/development` at drafting; **re-read before signing**, it moves |
| BASELINE_SHA | `a695c9c835bb5b4b206e6e3e00bb2aee23ca7b6b` | `PDC_SHA_LADDER.md` |
| INTEGRATED_BASELINE_SHA (P6 product tree) | `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (PR #537) | `PDC_SHA_LADDER.md` — **superseded as the EBH-7 pin by the final re-pin `P` below** |
| MCP-RC joint re-pin 2 (MCP-SERVER-X ACT-N11 on the final integrated tree) | `61531828f5728acc8f2ed357b4b835d7f5dc4ebc` (PR #554 merge, 2026-11-15 06:31 +01:00) | `git log origin/development`; ancestor of the `development` tip (`git merge-base --is-ancestor` verified at drafting); the MCP-SERVER-X closure dossier is on `development` at `1382cbef1` |
| P7 audited SHA (all eight domain audits) | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (PR #557 merge) | the eight `PDC_P7_D<n>_AUDIT_REPORT.md`; **pre-repair** |
| P7 audit report merges | #559 `35c3a21c1` (D7), #560 `c000d42a0` (D1), #561 `4a2f6b428` (D3), #562 `3b371ef9d` (D5), #563 `380037300` (D2), #564 `1a4a29746` (D8), #565 `cede66986` (D6), #566 `1c864ea19` (D4); Operator decision record for the P7 repair scope #567 `f87df3b2a` | `git log origin/development --merges` |
| P7 repair merges (T-IGX-041-A..I) | see section 3 | `git log origin/development --merges` |
| Operator records of the closing wave | #577 `a478eb235` (P7 re-pin record), #578 `cba32d247` (closing wave record) | `git log origin/development --merges` |
| Final EBH-7 re-pin SHA `P` (tickets T-IGX-041-J, -K, -L) | `<UNKNOWN today: no ticket file T-IGX-041-J/K/L exists on development at drafting>` | `DELEGATION-PDC-P7-REPIN.md` names the tickets; `ls governance/tickets` shows only 041-A..I |
| P8 re-pin (only if the P8 repair loop changes product paths) | `<P8 repair SHAs, if any>` | `DELEGATION-PDC-CLOSE.md` item 2 |
| GLOBAL_AUDITED_SHA | `<GLOBAL_AUDITED_SHA>` | `PDC_SHA_LADDER.md` row; "must contain the MCP-SERVER-X closure publication" |
| FINAL_CLOSURE_PUBLICATION_SHA | `<FINAL_CLOSURE_PUBLICATION_SHA>` | P9 merge |

## 3. Repair merges on `development` (all merged before this draft)

Ticket mapping follows `DELEGATION-PDC-P7-REPAIR.md` (A = D1 … H = D8, I = TENANT-X re-pin) and was checked against each plan header and merge during drafting (PR number and merge SHA from `git log origin/development --merges`).

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

Earlier P6 facts that stay valid history: T-IGX-036-X product fix PR #537 `146e7acf1`; omnibus debt repair T-IGX-036-Z PR #549 `622a61b962ee9941e4d8d4e4823bdd54063a3e87` (in `PDC_DEBT_REGISTER.md`).

## 4. What ARCH-FREEZE may accept (to be completed after P8)

| Item | Where | Notes |
| --- | --- | --- |
| Stage chain P0-P9 | `PDC_SHA_LADDER.md` | every role SHA set and ancestry verified (PDC-P9-02) — `<verification result>` |
| Domain dispositions D1-D8 | `PDC_FINAL_ACCEPTANCE_REPORT.md` section 3 | `<per-domain final disposition after P8>` |
| Cross-domain scenarios XD-01..12 | `tests/qualification/pdc_x/xd/` | historical run at `884c177ad` (927 passed, 5 skipped) — `<run at GLOBAL_AUDITED_SHA>` |
| Global audit | `<PDC_INDEPENDENT_AUDIT.md path>` | `<P8 verdict>` at `<GLOBAL_AUDITED_SHA>` |
| ECAP section 2.C Full Platform run | `<log path>` | the only known baseline run (`884c177ad`) was `BLOCKED` on 1 check (VBN-REACHABILITY, since fixed for the one gate file at `146e7acf1`); result at the audited SHA `<ECAP result>` |
| Exception register | [`PDC_EXCEPTION_REGISTER.md`](./PDC_EXCEPTION_REGISTER.md) | declared, labelled exceptions with reopen conditions; none silently upgraded |
| Debt register | [`PDC_DEBT_REGISTER.md`](./PDC_DEBT_REGISTER.md) | documented debt |
| FRZ proposal | [`PDC_FRZ_PROMOTION_PROPOSAL.md`](./PDC_FRZ_PROMOTION_PROPOSAL.md) | counts stay 225 / 26; `<Operator decision on each proposed row>` |

## 5. What the freeze must not assume (limitations carried by PDC-X)

1. **No production readiness.** PDC-X certifies Dimension A (architecture) only; live provider and environment qualification stays with `PROD-Q` (Path B, `OD-PDC-PATHB-01`). `FRZ-PRD-01`, `FRZ-PRD-02` and the live portions of `FRZ-PRD-03`, `FRZ-PRD-08`, `FRZ-SEC-04` stay `OPEN`.
2. **`dist_scale_x` and Docker (C10/C15) are not qualified, deferred** (`DELEGATION-PDC-WAVE2.md` decision 4). The C10/C15 restriction is unchanged: original Windows environment, no Docker migration, no new vendor integrations, no production configuration. P4 and P6 of `dist_scale_x` stay blocked.
3. **Declared exceptions** of `DELEGATION-PDC-CLOSE.md` item 1 and `DELEGATION-PDC-P7-REPAIR.md` (option B): see the exception register for ids, affected rows (proposed `PASS_SCOPED`) and reopen conditions.
4. **Pre-existing authorized exclusions** `OD-PDC-SCOPE-02`, `OD-PDC-SCOPE-03`, `OD-PDC-PATHB-01` (APP-HOST-6/7/9E, LLM provider plugin layer and non-VENV_BUNDLE runtime factories, live Path B items).
5. **Known red classes at the audited SHA** `<list from the P8 report>`; the previously known classes were IIAC activation staleness (until the P8-04 rebind) and the EBH-7 identity gate (until the final re-pin is recorded).
6. **Pin validity.** IIAC activation is bound to an audit pin; any change under `intergrax/`, `agents/` or `applications/` after `audit_execution_sha` invalidates it (ACT-N11). `<state of the P8-04 rebind>`.
7. **Held dependency-bot / `uv.lock` pull requests** are merged in one batch only after the final re-pin is recorded (`DELEGATION-PDC-WAVE2.md` decision 7): `<batch result>`.
8. **No B6 verdict and no certification** is implied by the EBH-7 replay (`DELEGATION-PDC-WAVE2.md` decision 1, scope).

## 6. Dependency record for ARCH-FREEZE

The ARCH-FREEZE dependency on `PDC-X` is satisfied **only** by the roadmap status published after the closure pull request (`OD-PDC-ARCHF-01`: note-only WBS header, task count stays 54; closure checked at `T-B8-ROADMAP` through the roadmap section 5 ledger). This document edits neither `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md`, the freeze checklist nor the roadmap. The FRZ traceability validator (`scripts/governance/verify_arch_freeze_plan_traceability.py`, `FRZ=225 families=26`) must stay green on the closure pull request. `<validator result at the closure SHA>`.

## 7. Session end

PDC-P9-06: publish; verify merged remote HEAD and CI; session-end branch check (`git branch --show-current`). `<result>`.
