# DELEGATION-SFX-R2-OD-CANONICAL-ADOPTION-SIGNING — Operator commission for OD-SFX-R2-CANONICAL-ADOPTION

| Field | Value |
| --- | --- |
| **Record ID** | DELEGATION-SFX-R2-OD-CANONICAL-ADOPTION-SIGNING |
| **Stage** | SCENARIO-FACTORY-X (SFX) R2 canonical overlay adoption (Option B) |
| **Basis** | IIAC Amendment 01, rules R-1..R-6. Precedent: signed ODs under `DELEGATION-SFX-OD-SIGNING` (OD-SFX-01..04, 15, 16 only — **not** in scope here). |
| **Date** | 2026-11-23 |
| **Campaign** | IGX-SFX-R2-ADOPTION-FINAL-AUTH-003 |
| **Ticket** | `T-IGX-070-C` |
| **Target OD** | `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/OD-SFX-R2-CANONICAL-ADOPTION.md` |
| **Source of commission** | Operator chat commission **IGX-SFX-R2-ADOPTION-FINAL-AUTH-003**, section 0 and G3 |
| **Provenance** | Transcribed by the adoption publication implementer session. Not operator-signed. Assurance: `process_enforced`. |

## Commission (transcription)

The operator commissioned a **new, independent, dedicated Operator Decision Signer session** for the sole purpose of reading, verifying, and signing **`OD-SFX-R2-CANONICAL-ADOPTION.md`** when its text is fully conformant with Option B (section 0 of the commission).

The operator explicitly **did not** reuse `DELEGATION-SFX-OD-SIGNING.md` (that delegation covers OD-SFX-01..04, 15, and 16 only).

## Scope of the delegation

- A new, dedicated signing session may update **`OD-SFX-R2-CANONICAL-ADOPTION.md` only** to record operator adoption of the R2 overlay decision, using the plain OD form: `Approved: OD-SFX-R2-CANONICAL-ADOPTION @ <ISO8601>`.
- Precede the operator line with a provenance paragraph citing **this** record as `<path>@<commit sha>`.
- Set the record's signature status to **APPROVED** when signed.
- Before signing, the session confirms all of the following:
  1. The decision text matches Option B: R2 overlay on R1; SFX business layer; SEF-X engineering authority; Runtime execution authority; REF-1 assisted; HO-2 unchanged; ECAP §16 OPEN; gates unchanged; no implementation authorization.
  2. The record does **not** authorize SFX-P5+, HO-2 relaxation, autonomous multi-submit, production execution, SEF ownership changes, or false Enterprise certification.
  3. Published plan approval for `T-IGX-070-C` exists with valid delegate provenance.
  4. This signing session differs from the adoption implementer and from the plan-approval delegate.
- The session may **refuse** to sign if any check fails.

## Explicit non-delegated items

- Any OD other than OD-SFX-R2-CANONICAL-ADOPTION.
- Ticket plan approval (see DELEGATION-SFX-R2-070-C-PLAN-APPROVAL).
- G6 independent audit certification.
- Product implementation, merges, or GATE-SFX-ENTRY closure.

## Separation (R-1..R-6)

- The adoption implementer, plan-approval delegate, and G6 auditor never write the OD operator line.
- The signing session implements no product code and does not certify its own publication commit as G6 PASS.