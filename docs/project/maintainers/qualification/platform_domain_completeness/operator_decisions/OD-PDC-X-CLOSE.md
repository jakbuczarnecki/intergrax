# OD-PDC-X-CLOSE — Operator answers to the single-signature questions of the PDC-X closing wave

> **Status: answers recorded from chat; the Operator confirmed the final text in chat on 2026-11-16 (see the signature line).** This record is a transcription prepared by the orchestrating session. It is **not signed**, contains no `Approved` line and writes no closure line. It becomes the Operator's decision only when the Operator confirms the final text in chat.

| Field | Value |
| --- | --- |
| **Record ID** | OD-PDC-X-CLOSE |
| **Stage** | PDC-X P9 (PDC-P9-03, PDC-P9-04, PDC-P9-05) |
| **Basis** | `PDC_FINAL_ACCEPTANCE_REPORT.md` section 1 (decisions 1-8); `DELEGATION-PDC-CLOSE.md` item 3 ("the Operator signs once at the end"); `DELEGATION-PDC-EXCEPTIONS.md`; `DELEGATION-PDC-LOOP2.md`; `OD-PDC-IIAC-SCOPE.md` (verdict vocabulary) |
| **Date of the answers** | 2026-11-16 |
| **Label** | English rendering of the Operator's chat answer of 2026-11-16 (the original answers were given in Polish; the orchestrating session relayed them to the preparing session in English) |
| **State of the repository when the answers were given** | `origin/development` `665760ceb06aee062cf254f485e895d7e928ec15` (PR #602 merge, the P9 closure package drafts) |

## Facts the answers rest on (read from the package; not new claims)

- Global audit `PASS_SCOPED` at `969cbd964c679d563b02e55f065b452d46fdf90b` with declared exceptions; certifier binding manifest `PASS` scoped (19 limitations); IIAC activation `IN_FORCE` from `9f05749bc23592868f3d0a7f505f9edbb2cfd025` (`PDC_FINAL_ACCEPTANCE_REPORT.md` section 2).
- The report (section 1, decision 1) and the closure text (section 1) state that the roadmap section 3.2 exit criteria are **not met literally** (audit is `PASS_SCOPED`, not `PASS`; P7 domain verdicts were not re-graded; the certifier's manifest is not a stage certification) and that the closure choice is an Operator reading of those criteria. The answer to decision 1 below was given with that statement in the package.

## Decision table (as relayed)

| # | Decision | Operator's answer |
| --- | --- | --- |
| 1 | Closure outcome (A scoped closure in the IIAC canonical wording; B master-plan wording; C do not close) | **A** — scoped closure, IIAC canonical wording |
| 2 | The 18 exceptions of the "Operator decision" kind: EX-D1-03, EX-D1-07, EX-D2-01, EX-D2-06, EX-D2-07, EX-D3-02, EX-D4-02, EX-D4-03, EX-D4-04, EX-D4-05, EX-D5-04, EX-D5-05, EX-D6-03, EX-D6-04, EX-D7-02, EX-D7-03, EX-D8-03, EX-D8-04 | **All 18 declared as exceptions** |
| 3 | The 13 rows open with no exception: D1 R009, R015, R017; D2 R004; D3 R005, R018, R019; D5 R002; D6 R007, R010, R015; D7 R015, R022 | **Documented debt. The rows stay `PARTIAL`. No new tickets.** |
| 4 | Exceptions already authorized (`DELEGATION-PDC-CLOSE.md` decision 1; `DELEGATION-PDC-EXCEPTIONS.md`) | **Confirmed** |
| 5 | FRZ: option N (no promotion) or P | **N** — no promotion; counters stay 225 / 26 |
| 6 | Debt register | **Accepted** |
| 7 | Limitations stated as facts: the EBH-7 L3 replay failed and was not re-run, B6 is not passed; the EBH-7 identity gate stays a known red class; the C10/C15 freeze is not lifted; no production-readiness claim; rows were not re-certified one by one; ECAP FULL, the contract suites and the combined pytest were last run at `d39a520dd`, not at `G` | **Acknowledged** |
| 8 | ARCH-FREEZE handoff | **Published as written; ARCH-FREEZE not entered** |

## Effect (once the Operator confirms this text)

- Closure outcome A is the text of `PDC_ROADMAP_CLOSURE_TEXT.md` section 3 (Variant A). The roadmap edit is proposed in a separate commit that can be dropped; it is not part of this record.
- The 18 exceptions of decision 2 and the exceptions of decision 4 are declared exceptions with the reopen conditions, owner paths and source findings of `PDC_EXCEPTION_REGISTER.md`; their rows are rated `PASS_SCOPED` only by this authorization.
- The 13 rows of decision 3 stay `PARTIAL` and are documented debt (`PDC_DEBT_REGISTER.md` section 13).
- No FRZ status changes; the checklist is untouched (225 / 26).
- This record does **not** lift the C10/C15 freeze, does **not** state a verdict on B6 (the replay failed and was not re-run), is not a production-readiness claim, not a certification of every requirement row and not ARCH-FREEZE entry.

## Not covered (unchanged)

OD signatures other than this record; lifting C10/C15; the verdict on B6 or on certification; a third repair loop; any exception beyond those listed.

## Separation

The orchestrating session, the implementers, the auditors, the certifier and the delegates did not write the signature line. The Operator confirmed the text in chat; the signature line below transcribes that confirmation, as the earlier delegation records transcribe the Operator's chat answers.

Signed: Operator, in chat on 2026-11-16, after seeing PR #603 (OD-PDC-X-CLOSE, the registers and the Variant A roadmap text). English rendering of the Operator's answer: "Yes, I sign" (original wording in Polish: "Tak, podpisuję").
