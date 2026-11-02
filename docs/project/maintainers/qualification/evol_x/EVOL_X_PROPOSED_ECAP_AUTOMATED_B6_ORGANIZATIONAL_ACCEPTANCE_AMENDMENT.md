# ECAP-P4-EVOL-SCOPE-AUTO-B6 — direction approved, activation pending

**Amendment ID:** `ECAP-P4-EVOL-SCOPE-AUTO-B6`  
**Amendment version:** `2026-11-02.1-pending`  
**Operator decision:** **OD-EVOL-14** — direction and implementation approved (2026-11-02)  
**Normative activation:** **`PENDING_REQUIRED_APPROVALS`**  
**Normative effect:** **`NOT_IN_FORCE`**

Controlling acceptance rule until activation is `IN_FORCE`: ECAP-§2.B step **B6** (independent exact-SHA acceptance recorded in the roadmap §5 ledger) and ECAP-§1.6 (implementation agents never issue final PASS). This file does not replace that text.

An implementation commit of this amendment is not maintainer publication and does not put the amendment in force.

## Authority gate

| Question | Result |
| --- | --- |
| May the operator record the automation-first direction? | **Yes.** OD-EVOL-14 records that direction for EVOL-X `AUTOMATION_FIRST_SCOPED` only. |
| May this session enact the ECAP change as in force? | **No.** |
| Why | ECAP-§1.6 separates the implementation agent from the independent auditor and the operator. The activation conditions below are not all satisfied. No signed security review and no independent key custodian are present in the repository. ECAP-P4 forbids the implementing agent from self-accepting the normative change. |

### Approvals still required before `IN_FORCE`

1. **ECAP maintainer publication** — a publication act distinct from the implementer commit that added this file, stating this version is in force.
2. **Security review** — signed review of the trust boundary, signature contract, replay ledger, and key lifecycle in this amendment.
3. **Independent key custody** — a signing identity whose private key is not generated or held in an implementer-controlled workspace, with a published trust root, rotation, and revocation path.

OD-EVOL-11 (named human auditor) stays **identity not named**. OD-EVOL-12 and OD-EVOLX-08 stay **PENDING**. This amendment does not sign either decision and does not declare EVOL-X CLOSED.

## Scope

Applies only to EVOL-X closure under `certification_profile=AUTOMATION_FIRST_SCOPED` and OD-EVOL-13, and only after this amendment's activation is `IN_FORCE`.

Does not authorize:

- implementer or Cursor-agent self-approval;
- promotion of any `FRZ-*` row, including FRZ-EVOL-07;
- an EVOL-X CLOSED declaration;
- reuse of `NON_PRODUCTION_TEST_ONLY` credentials as formal acceptance;
- treating a second worktree, subagent, or unsigned JSON file as organizational independence.

## Normative text (inactive until `IN_FORCE`)

> **ECAP-P4-EVOL-SCOPE-AUTO-B6:** For EVOL-X under `AUTOMATION_FIRST_SCOPED` only, organizational B6 acceptance MAY be recorded when a custody-separated attestor, distinct from the evidence producer and from the implementer, emits a schema-valid Ed25519 attestation bound to the pinned `TEST_EXECUTION_SHA`, `EVIDENCE_FINAL_PUBLICATION_SHA`, policy id `ECAP-P4-EVOL-SCOPE-AUTO-B6`, policy version, CI provenance, Mode A/B results, findings disposition, and FRZ scope declaration. The attestor's execution identity MUST have read-only access to qualification inputs and MUST NOT have write access to auditor configuration or signing keys used by implementers. Verification logs MUST be immutable or independently verifiable. The trust root MUST support public-key verification, rotation, and revocation. Replay protection MUST bind the attestation to a one-time input identity. Missing, stale, revoked, conflicting, unsigned, or implementer-controlled attestations MUST fail closed. A technical PASS MUST NOT by itself create formal acceptance eligibility. The attestation MUST NOT promote `FRZ-*` and MUST NOT declare EVOL-X CLOSED. Operator decisions W7-03..W7-06 remain mandatory after B6.

Machine-readable status: [EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json).  
Deployment blockers: [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](./EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md).
