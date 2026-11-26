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
