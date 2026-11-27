# T-AF2-RED-CLASSES — operator decision record

**Status:** operator **DECIDED** for this card only.  
**Packet:** [`RED_FIRST_V5_RED_CLASSES_PACKET.md`](RED_FIRST_V5_RED_CLASSES_PACKET.md)  
**Packet publication SHA:** `eba9bb47a3bb282c1e001c9a20ad8ed6b596ddbb`  
**Companion:** [`RED_FIRST_V5_CLASSIFICATION_ROOT.md`](RED_FIRST_V5_CLASSIFICATION_ROOT.md)

The operator authorized these historical red-first dispositions and no others:

| FRZ | Disposition | Machine status |
| --- | --- | --- |
| `FRZ-PLG-02` | OUT of mandatory historical red-first evidence | `RED_FIRST_OUT` |
| `FRZ-PLG-07` | OUT of mandatory historical red-first evidence | `RED_FIRST_OUT` |
| `FRZ-RPL-04` | OUT of mandatory historical red-first evidence | `RED_FIRST_OUT` |
| `FRZ-REL-07` | OUT of mandatory historical red-first evidence | `RED_FIRST_OUT` |
| `FRZ-PRD-08` | OUT of mandatory historical red-first evidence | `RED_FIRST_OUT` |
| `FRZ-FRZ-10` | Re-home to `T-B7-PFP-ACTIVATE` (W5) and `T-B7-PFP-VERIFY` (W8) | `RED_FIRST_LATER_WAVE` |

Historical `red_first_status` remains recorded as `red_first_historical_status=RED_FIRST_UNREPRODUCIBLE`. Those outcomes are not `RED_AT_PREFIX`. The 39-row inventory stays. Platform obligations for plugin lifecycle, authority enforcement, vendor replaceability, bounded retry, production bypass / PROD-Q, and post-freeze policy stay with their canonical owners.

This record does not set `RW_PLAN_AUDIT`, `SCOPE`, `G2`, `R`, `FCS`, `M3`, `ENTRY`, or `ARCH-FREEZE CLOSED`.

**Recorded:** 2026-11-26.

---

## Addendum 2026-11-27 (T-IGX-078-F, append-only)

The text above is unchanged. This addendum records two later Operator decisions given in chat on 2026-11-27 ("Accept all (Recommended)" to the OP-B packet).

**Confirmation of the six dispositions (OP-B B-1, option B).** The Operator re-confirmed PLG-02, PLG-07, RPL-04, REL-07 and PRD-08 (`RED_FIRST_OUT`) and FRZ-FRZ-10 (`RED_FIRST_LATER_WAVE`) as a new decision dated 2026-11-27. The confirmation is later than the application of the dispositions in commit `a61eeaedf`, and it also covers the checker change in that commit. See register row OD-AF-CLOSE-B1 in [`ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`](../../ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md). The carrying ticket for FRZ-FRZ-10 is `T-IGX-052-I` (orchestrator allocation, not an Operator decision).

**Twelve EVIDENCE_MISSING rows (OP-B B-2, option A).** Red-first is due at the listed owner ticket. That ticket records the exact negative node red at its pre-fix SHA and green after the fix before the row can be certified. No evidence is claimed now, and `RED_FIRST_EVIDENCE` stays `INCOMPLETE` until the tickets deliver.

| FRZ | Owner ticket |
| --- | --- |
| `FRZ-BND-05` | `T-IGX-043-H` |
| `FRZ-BND-06` | `T-IGX-043-J` |
| `FRZ-DEBT-02` | `T-IGX-051-H` |
| `FRZ-DEBT-03` | `T-IGX-051-H` |
| `FRZ-DEBT-04` | `T-IGX-051-F` |
| `FRZ-EXE-02` | `T-IGX-043-A` |
| `FRZ-PLG-05` | `T-IGX-043-A` |
| `FRZ-SEC-06` | `T-IGX-051-B` |
| `FRZ-TEN-02` | `T-IGX-078-D` |
| `FRZ-TYP-01` | `T-IGX-078-C` |
| `FRZ-TYP-03` | `T-IGX-078-C` |
| `FRZ-TYP-05` | `T-IGX-051-G` |

`FRZ-DEBT-02` and `FRZ-DEBT-03` stay `EVIDENCE_MISSING` until a seeded red case is shown (the MEP red-first rule applies). This addendum is not a waiver and sets no gate.
