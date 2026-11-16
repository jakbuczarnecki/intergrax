# ENT-AUDIT-X ← HIST campaign handoff (formal external dependencies)

| Field | Value |
|---|---|
| From | Historical certification campaign (HIST-01..04 master plan execution) |
| To | ENT-AUDIT-X stream owner |
| Date | 2026-11-09 |
| HIST publication SHA | `087bd17693db5b4b8a14cb618389ee1d6c2aaa3d` |
| HIST campaign verdict | **`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`** — **not** ENT formal closure |

## What HIST completed (do not re-do)

- Product batch C1–C4 merged (`3c91694c6`); P4 independent audit published (`468b17576`).
- IIAC re-activation for HIST product paths: record **IN_FORCE** @ `8045915a`; P5 verify green on POST_ACTIVATION tree.
- P6 Dimension A replay **POST_ACTIVATION**: 595 passed, exit 0 @ `bf03550aa` — evidence in [`HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md`](HISTORICAL_FINAL_QUALIFICATION_EVIDENCE.md).
- P8 dossier audit PASS + certifier ACCEPT partial label.

## What HIST explicitly did **not** do (ENT-owned)

| Item | State | ENT action |
|---|---|---|
| Shared re-activation **consent** | ENT counter-sign **absent** | Counter-sign [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md) or publish acceptance of Operator P2 supersession |
| **Population E** REG rollup | Not executed | Run ENT-AUDIT-X Population E qualification per operator REG split (HIST-01 handoff options a/b) |
| **T-IGX-021-B/C** validator alignment | Not started (Q-12) | Schedule with ENT stream; material governance paths |
| **ENT-AUDIT-X formal closure** | Not claimed | Continue EVOL-X / ENT W7 evidence chain; no HIST counter-sign |

## Sequencing record

- Operator authorized HIST to proceed on **ENT-first / shared-reactivation Option 1** sequencing without ENT in-repo consent: [`delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md`](delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md).
- ENT silence is **not** supersession; ENT may still counter-sign later.

## Evidence bundle for ENT intake

1. [`HISTORICAL_CAMPAIGN_TERMINAL_CLOSURE_REPORT.md`](HISTORICAL_CAMPAIGN_TERMINAL_CLOSURE_REPORT.md)
2. [`HISTORICAL_CAMPAIGN_BLOCKER_DOSSIER.md`](HISTORICAL_CAMPAIGN_BLOCKER_DOSSIER.md)
3. [`iiac_evidence/HIST-P5-2026-11-09/`](iiac_evidence/HIST-P5-2026-11-09/) (pre/post activation audit package)
4. [`HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json`](HISTORICAL_FRZ_85_FINAL_DISPOSITION_MATRIX.json) — 85 rows; no promotion by HIST

## Unblock path for global historical `CLOSED_VERIFIED`

Requires ENT consent (or formal Operator supersession covering ENT), Population E rollup, and unchanged Path B / RPL external gates — see master plan §J. HIST does not assert these are satisfied.
