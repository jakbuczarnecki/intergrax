# GA.2R Operator items (ready packet)

Ticket: `T-IGX-078-B`. Prepared 2026-11-27. Status: **not decided, not applied**.

This packet changes no disposition, signs no card and edits no decision record. The orchestrator takes it to the Operator before RW-PLAN-AUDIT round 2. Evidence: [`RED_FIRST_GA2R_RESULT.md`](RED_FIRST_GA2R_RESULT.md).

## Item 1: quote or relay reference for the six existing dispositions (audit M5)

**Fact.** [`RED_FIRST_V5_RED_CLASSES_DECISION.md`](RED_FIRST_V5_RED_CLASSES_DECISION.md) (recorded 2026-11-26) marks PLG-02, PLG-07, RPL-04, REL-07 and PRD-08 `RED_FIRST_OUT`, and FRZ-FRZ-10 `RED_FIRST_LATER_WAVE`. It holds no verbatim Operator statement, and the Operator decision register has no entry for it. Commit `a61eeaedf` applied the dispositions and also changed the checker and its test.

| Option | Content | Consequence |
| --- | --- | --- |
| A | The Operator supplies the original statement or its relay reference (session, date). A decision register entry quotes it as a labelled English rendering. | The six dispositions keep their date and become traceable. |
| B | The Operator confirms the six dispositions again, as a new dated decision, in a register entry. The 2026-11-26 record stays unchanged. | Traceable from the new date; the record says that the confirmation is later than the application. |
| C | The Operator does not confirm. | The six rows return to `RED_FIRST_UNREPRODUCIBLE`; the checker rejects that status and audit round 2 cannot pass until each row has evidence. |

**Recommendation:** A if the original statement can be located, otherwise B. For the register entry, also state whether the checker change in `a61eeaedf` is covered by the same decision or needs a separate review.

## Item 2: rows without red-first evidence (audit B1, H7, H8, H9, M4, M6)

**Fact.** GA.2R re-measured 21 rows at read pin `aabd3925c`. Twelve have no evidence that names a negative test failing at a pre-fix SHA on the declared FRZ mechanism. They are recorded as `EVIDENCE_MISSING`. This is not a waiver. The reason for each row is in the evidence record.

| FRZ row | Evidence | Red-first due at |
| --- | --- | --- |
| FRZ-BND-05 | [record](RED_FIRST_GA2R_RESULT.md#frz-bnd-05) | `T-IGX-043-H` |
| FRZ-BND-06 | [record](RED_FIRST_GA2R_RESULT.md#frz-bnd-06) | `T-IGX-043-J` |
| FRZ-DEBT-02 | [record](RED_FIRST_GA2R_RESULT.md#frz-debt-02) | `T-IGX-051-H` |
| FRZ-DEBT-03 | [record](RED_FIRST_GA2R_RESULT.md#frz-debt-03) | `T-IGX-051-H` |
| FRZ-DEBT-04 | [record](RED_FIRST_GA2R_RESULT.md#frz-debt-04) | `T-IGX-051-F` |
| FRZ-EXE-02 | [record](RED_FIRST_GA2R_RESULT.md#frz-exe-02) | `T-IGX-043-A` |
| FRZ-PLG-05 | [record](RED_FIRST_GA2R_RESULT.md#frz-plg-05) | `T-IGX-043-A` |
| FRZ-SEC-06 | [record](RED_FIRST_GA2R_RESULT.md#frz-sec-06) | `T-IGX-051-B` |
| FRZ-TEN-02 | [record](RED_FIRST_GA2R_RESULT.md#frz-ten-02) | `T-IGX-078-D` |
| FRZ-TYP-01 | [record](RED_FIRST_GA2R_RESULT.md#frz-typ-01) | `T-IGX-078-C` |
| FRZ-TYP-03 | [record](RED_FIRST_GA2R_RESULT.md#frz-typ-03) | `T-IGX-078-C` |
| FRZ-TYP-05 | [record](RED_FIRST_GA2R_RESULT.md#frz-typ-05) | `T-IGX-051-G` |

| Option | Content | Consequence |
| --- | --- | --- |
| A | A new `T-AF2-RED-CLASSES` entry: red-first is due at the listed ticket. The ticket records the exact negative node, red at its pre-fix SHA and green after the fix, before its row can be certified. | No evidence is claimed now. `RED_FIRST_EVIDENCE` stays `INCOMPLETE` until the tickets deliver. The checker would need a separate approved change if the new entry is to count as a disposition. |
| B | All twelve need red-first evidence before audit round 2. | New negative tests must be written and replayed first. GA.2R does not write lane tests, so round 2, GA.3 and the lanes wait. |
| C | Move rows to `RED_FIRST_OUT` or `RED_FIRST_LATER_WAVE`. | Removes or postpones the obligation without a measurement. |

**Recommendation:** A.

**Sub-question for FRZ-DEBT-02 and FRZ-DEBT-03.** The nodes `test_rw_dep_01a_core_requirement_has_declared_owner[jsonschema]`, `...[tzdata]` and `test_rw_dep_01b_project_reports_no_core_ownership_violation` fail at test-only prefix `6c9c8155d` and pass at the pin. The matrix text for NT-DEBT-02-1 asks for a gate that is red at the base and green after the fix. The MEP red-first rule says that a gate already red does not count, and the seeded node NT-RW-DEP-01c passes at the prefix.

- Keep `EVIDENCE_MISSING` until a seeded case is shown red (**recommended**: it follows the MEP rule).
- Or the Operator accepts the matrix wording for these rows and records that the MEP rule does not apply to them. FRZ-DEBT-03 would also need a statement on compatibility seams, because no node inventories them.

## Item 3: EBH-6 Q2-D2 gate inversion for FRZ-TYP-01 and FRZ-TYP-03 (Q-WF-04)

**Fact.** The cited versioned plan reserves to the Operator the inversion of `tests/qualification/ebh_6/_q2d2_technical_qualification.py` and `tests/qualification/ebh_6/test_ebh_6_q2d2_dual_path.py`, because it changes what the gate means. It blocks the merge of the typed exposure seam part (PT-WF Part JB, 8 product paths). The same rows (and FRZ-RPL-04) are now owned by `T-IGX-078-C`. The DW0-03 work item (WL-K1-B-02) lists the typing ratchet files and does not list the Nexus exposure files.

| Option | Content | Consequence |
| --- | --- | --- |
| A | The Operator approves the inversion; it lands in the same change set as the `T-IGX-078-C` product fix, after independent review. | The typed seam fix can merge; the gate never disagrees with the code. |
| B | The Operator refuses the inversion. | `T-IGX-078-C` must show another way to close FRZ-TYP-01 and FRZ-TYP-03. GA.2R did not assess whether one exists. |
| C | Defer the decision. | `T-IGX-078-C` may be planned, but the part that needs the inversion cannot merge; both rows stay open. |

**Recommendation:** A, decided no later than the plan approval of `T-IGX-078-C`.
