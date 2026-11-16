# OD-SFX-21 — Pre-freeze closure scope: human usability pilot removed from the SFX documentation closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-21 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **RECORDED — OPERATOR INSTRUCTION (NOT SIGNED)** |
| **Provenance** | Transcribed by the SFX closeout implementer session as a labelled English rendering of the operator's Polish scope correction of 2026-11-13 (SFX pre-freeze closeout plan). The agent wrote this record. It carries no operator signature line and no `Approved:` line. |
| **Ticket** | `governance/tickets/T-IGX-037-F.md` |
| **Numbering note** | The closeout plan called this record `OD-SFX-17`. That number is already used by the PROPOSED decision on approval-recording doors (UX-Q-01 and SFX-GAP-25) in `SFX_GAP_ANALYSIS.md`, and OD-SFX-18..20 are PROPOSED there too. To avoid two decisions sharing one ID, this record takes the next free number, 21. Its content is the one the plan describes. |

## Question

Which human-participant work belongs to the current pre-freeze documentation campaign, and what does the campaign status `SFX_PRE_FREEZE_DOCS_CLOSED` mean?

## Decision (English rendering of the operator's instruction)

1. The following are removed from the current pre-freeze campaign: the five-user test, usability sessions, participant recruitment, TTFS measured with people, and reports from those sessions.
2. Their absence is **not** an automatic blocker for `SFX_PRE_FREEZE_DOCS_CLOSED`.
3. Result fields in [`SFX_USABILITY_PROTOCOL.md`](../SFX_USABILITY_PROTOCOL.md) that read `NOT_RUN` stay `NOT_RUN`. No agent records `PASS` for them.
4. What stays in the campaign: automated tests, schema validation, agent review, an independent review of this campaign, evidence checks and exact SHAs.
5. The requirements for the future qualification of the finished product do not change in wording. That covers SFX-P12-05, usability rounds 1–3, the OD-SFX-07 targets at P12 acceptance and the product end audit.
6. `SFX_PRE_FREEZE_DOCS_CLOSED` (this campaign) and `SFX_PRE_IMPLEMENTATION_READY` (a later, separate operator decision) are two different statuses. This campaign does not grant the second.

## Effect on pre-freeze criteria (this sense only)

| Artifact | Change |
| --- | --- |
| `SFX-MASTER-PLAN.md`, unit SFX-P3-01 | Annotation on the round 0 / TTFS baseline criterion for the pre-freeze campaign. Rows SFX-P12-05 and OD-SFX-07 are unchanged. |
| `SFX_WBS.md`, row SFX-P3-01 | Same annotation. |
| `T-IGX-037-D.md` | Checkpoint note. The historical `NOT_RUN` sentence and the `Approved:` line are unchanged. |
| `SFX_USABILITY_PROTOCOL.md` | Banner above the results templates. Every result field stays `NOT_RUN`. |

## Explicit non-actions

- Does not sign or accept any decision, and does not write an `Approved:` line.
- Does not accept any ADR-SFX-001..006, and does not change FRZ or IIAC state.
- Does not grant `SFX_PRE_IMPLEMENTATION_READY` and does not set `PRE_IMPLEMENTATION_READY_SHA`.
- Does not replace the open IDE toolkit v0 operator walkthrough after the REF-1 text. That walkthrough is a different item from the five-user test, and the dry run does not replace it either. It stays open on the ready path.
- Does not satisfy GATE-SFX-ENTRY and does not authorize SFX-P4-B or SFX-P5+.
- Does not relax or tighten SFX-P12 or the OD-SFX-07 targets.
