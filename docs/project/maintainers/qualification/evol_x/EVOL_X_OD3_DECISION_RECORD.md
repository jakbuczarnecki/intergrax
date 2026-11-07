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

## Closure update (2026-11-07)

* **Condition 2 (dry run of the real B-4 resolution)** - satisfied by publication: the re-audit evidence and the manifest were published at `add7187bc` and the certifier disposition at `7a5476cdc`. The recovery session `local_7eeb7387-df63-460d-88e9-be7359430173` re-checked this on a fresh worktree of `7a5476cdc` in a new venv outside the repository: `verify_iiac_policy_activation --require-in-force` is `IN_FORCE`, `verify_iiac_audit_manifest` and `verify_iiac_role_separation` PASS on the re-audit manifest, and `--gate b6-readiness` fails with exactly one `A-18` (`B6-P3/P4/P5`), no other Gate A code.
* **Second independent certifier.** `local_e34301ce-67df-40bf-b274-b2e062581541` returned `NOT CERTIFIED` for `fdb1e7a62` (report `EVOL_X_W4_01_SECOND_CERTIFIER_REPORT.md`, published unedited); its blocking findings are resolved as recorded in the publication note of that report and in Amendment 2 of `EVOL_X_OD2_DECISION_RECORD.md` (no code change, pin `fdb1e7a62` unchanged, condition 3 intact). Its residual LOW findings (CERT-02, W405-01/02, W404-04, W401-RA-05/07/08, W403-04 unverified) stay accepted as listed above.
* **W4-F01** is ratified in `EVOL_X_W4_F01_OPERATOR_RATIFICATION_RECORD.md` (delegated).
* Ticket `T-IGX-019-A` is set to `closed`. B-4 is closed; **B-5, B6-P3/P4/P5, FRZ-EVOL-01..10, OD-EVOL-12 and OD-EVOLX-08 stay open and EVOL-X stays `NOT_CLOSED`.** Conditions 3-5 above stay in force; OD-1 and OD-2 must be re-opened and the fresh-clone procedure re-run at the SHA that carries the resolution before P3.
