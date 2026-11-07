# EVOL-X OD-1 — decision record: governance of the P2 pin in the closure preflight

**Status:** `DECIDED — ACCEPT_RESIDUAL_NO_MATERIAL_CHANGE` · **Date:** 2026-11-06 · **Ticket:** `governance/tickets/T-IGX-019-A.md`
**Provenance:** the operator instructed the implementer agent session `local_4b071fe5-82d0-48bc-b524-146a3daf5c45` in chat to carry out the OD-1 step ("Wykonaj w takim razie kroki 3 i 4", after the step list that named OD-1 as "czy skrypt preflight trafia na listę materialnych ścieżek"). The agent chose the option below and recorded it on the operator's behalf. This is a delegated, process-enforced entry (assurance `process_enforced`), not a handwritten or cryptographic signature; the operator can overrule it by superseding this file. It is **not** a record read by the closure preflight and takes none of OD-EVOL-12, OD-EVOLX-08 or P3.

## Question (finding W401-RA-08)

`P2_CERTIFICATION_PUBLICATION_SHA` lives in `scripts/qualification/evol_x_closure_publication_preflight.py`, which is not in `MATERIAL_GOVERNANCE_PATHS`; the repository has no CODEOWNERS file. Changing the pin together with matching evidence is therefore one unreviewed commit.

## Options considered

| Option | Effect | Cost |
|---|---|---|
| A. Add the preflight script and its tests to `MATERIAL_GOVERNANCE_PATHS` | the pin is governed like the activation gate | edits `scripts/governance/iiac_activation_gate.py` (a material path): triggers the full IIAC governance re-audit and, possibly, re-activation; blocks this whole corrective for an unrelated-in-time change |
| B. Add a CODEOWNERS rule only | review routing for the script, tests and evidence | useful only with branch protection requiring owner review; `development` is not protected today, so it would be a paper control |
| C. Accept the residual, no material change now | the independent audits (this one included) re-check the pin against `git log --diff-filter=A` of the certification record, and the preflight already fails closed if they differ | the control relies on review, not on a mechanism |

## Decision

**Option C**: the residual is accepted without a material change. Rationale: the pin is cross-checked at run time against the first publication of the certification record (a content-derived commit), so editing the constant alone does not move the baseline; moving both the pin and the evidence needs a rewrite that independent audit and the SHA ladder would expose; and option A would force a re-audit of the whole IIAC gate and delay the corrective without closing an exploit of the shipped code (the auditor classed W401-RA-08 as INFO, "not an exploit").

**Review trigger (mandatory):** the decision is re-opened before the P3 B6 acceptance is authorised. At that point the operator decides again between A and B (B only together with branch protection). Any change to `P2_CERTIFICATION_PUBLICATION_SHA`, to the baseline derivation or to `scripts/governance/*` before then requires its own ticket and an independent audit.

## Limits

No material governance path, activation record, gate or protocol changed by this record. It authorises no B6 step, promotes no FRZ status and does not close EVOL-X.

## Operator ratification (2026-11-07)

The independent certifier session `local_dbad2215-acac-4a9e-aaf6-ad3be89aee18` (report `EVOL_X_W4_01_CERTIFIER_REPORT.md`, finding C-02) noted that this decision was taken by the implementer under a blanket authorisation. On 2026-11-07 the operator wrote in the chat of the implementer session `local_4b071fe5-82d0-48bc-b524-146a3daf5c45`: „Otrzymujesz wszystkie moje zgody i akceptacje, które są wymagane do zamknięcia tego etapu i wszystkich zadań, które wchodzą w jego skład. Wykonaj teraz ostateczne domknięcie całego zadania i etapu, tak abym mógł odznaczyć je jako wykonane.” The implementer records that statement here as the operator's ratification of this decision as written (the original text above is unchanged). Like the first entry this is a delegated, process-enforced record (assurance `process_enforced`), not a handwritten or cryptographic signature; the operator may replace it at any time. It takes none of OD-EVOL-12, OD-EVOLX-08 or P3.
