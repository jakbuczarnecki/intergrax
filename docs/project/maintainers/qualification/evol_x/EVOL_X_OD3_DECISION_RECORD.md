# EVOL-X OD-3 - decision record: acceptance of blocker B-4 (independent corrective implementation and re-audit)

**Status:** `DECIDED - B4_TECHNICALLY_RESOLVED_ACCEPT_WITH_CONDITIONS` · **Date:** 2026-11-07 · **Ticket:** `governance/tickets/T-IGX-019-A.md`
**Provenance:** on 2026-11-07 the operator wrote in the chat of the implementer session `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`: „Otrzymujesz wszystkie moje zgody i akceptacje, które są wymagane do zamknięcia tego etapu i wszystkich zadań, które wchodzą w jego skład. Wykonaj teraz ostateczne domknięcie całego zadania i etapu, tak abym mógł odznaczyć je jako wykonane.” The implementer records the operator's acceptance of the certifier's recommendation below. It is a delegated, process-enforced entry (assurance `process_enforced`), not a handwritten or cryptographic signature; the operator may supersede this file at any time. It is **not** read by the closure preflight and does not change any gate result.

## Evidence accepted

* Corrective implementation pin `fdb1e7a62` (final pin; first consolidated corrective `9f8b04d2b`), five audit rounds by the independent auditor `local_e360e892-825d-4e0f-81f6-b4dc46ed4df9`; round 5 found no blocker (PARTIAL, LOW/INFO residuals only). Auditor evidence is published unchanged at `bec0a0948`.
* Certifier report `EVOL_X_W4_01_CERTIFIER_REPORT.md` (published at `b59052993`) by the third session `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18`: verdict **ACCEPT_WITH_CONDITIONS**, no blocker in tiers T1-T3, fresh-clone runs reproduced the expected two-blocker output (`B-4`, `B6-P3/P4/P5`), B4 suite 493 passed, governance suite 145 passed / 1 skipped, no material governance path changed since START_HEAD `5afaad9ea`.

## Decision

B-4 is accepted as **technically resolved at `fdb1e7a62`**, subject to the certifier's conditions:

| # | Condition | State on 2026-11-07 |
|---|---|---|
| 1 | Operator ratification of OD-1 and OD-2; amend OD-2 environment wording (C-01) | done in the ratification sections of `EVOL_X_OD1_DECISION_RECORD.md` / `EVOL_X_OD2_DECISION_RECORD.md` and the amendment in OD-2 |
| 2 | Dry-run the real B-4 resolution (and the P3 record text, W401-RA-05) on a throwaway clone before publishing it | **open** - assigned to the certifier session; the acceptance is effective only if the dry run ends with `--gate b6-readiness` failing on the `B6-P3/P4/P5` `A-18` alone |
| 3 | Keep the pin immutable (preflight script, P2 constant, `scripts/governance/*`) | in force; any change voids this record and needs its own ticket and audit |
| 4 | Re-open OD-1 and OD-2 before P3, and re-run the fresh-clone procedure at the SHA that carries the resolution commit | in force |
| 5 | EVOL-X stays `NOT_CLOSED`; B-5 (OD-EVOL-12 / OD-EVOLX-08), B6 acceptance, FRZ promotion and closure stay operator-only | in force |

References to this evidence must cite commit SHAs, not paths (W404-04).

## Limits

This record publishes no blocker resolution and changes no certification record, gate, material governance path, activation record or protocol. It does **not** authorise or perform P3 / B6 acceptance, does not promote any FRZ status, does not take OD-EVOL-12 or OD-EVOLX-08 and does not close EVOL-X. `--gate b6-readiness` keeps failing until the B-4 resolution is published and until P3 is authorised.
