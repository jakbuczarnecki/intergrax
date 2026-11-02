# EVOL-X — authority and approval matrix

**Authority:** [EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md](./EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md).

| Decision / gate | Owner | Acceptance mechanism | Evidence artifact | Blocks if missing |
|-----------------|-------|----------------------|-------------------|-------------------|
| OD-EVOL-14 direction | Operator | **APPROVED** 2026-11-02 | Operator register | — |
| OD-EVOL-15 execution scope | Operator | **APPROVED** 2026-11-02 | Wave reports | — |
| ECAP-P4-EVOL-SCOPE-AUTO-B6 **IN_FORCE** | ECAP maintainer | Publication act distinct from implementer commit | [EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md](./EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md) | W7-01 Path B |
| Security review | Independent reviewer | Signed record | [EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md) | Attestor signing |
| Key custody | Key custodian (non-implementer) | Trust root + custody attestation | External trust root JSON (not in repo private key) | `PRODUCTION_INDEPENDENT` |
| OD-EVOL-11 | Operator | Named identity in register | Register row | G-02 |
| Independent attestor | Infra operator | `independent_attestation_environment: true` | Inventory JSON + logs | Path B eligibility |
| Formal B6 | Attestor / named auditor | Protocol 03 or signed attestation | `B6_ACCEPTANCE_PUBLICATION_SHA` | W7-03+ |
| FRZ-EVOL-07 Outcome A/B | Independent + operator | W7-03 record | FRZ dossier | FRZ-07 PASS |
| OD-EVOL-12 | Operator | Signature | OD package | W7-04 |
| OD-EVOLX-08 | Operator | Signature | Closure package | W7-05 |
| W7-05 closure | Operator | Authorization + publication commit | `FORMAL_CLOSURE_PUBLICATION_SHA` | ENT handoff final |
| Implementer / Cursor agent | — | **Must not** self-approve B6, IN_FORCE, FRZ, CLOSED | Governance JSON | Always |

## Role separation (ECAP-§1.6)

| Role | Permitted | Forbidden |
|------|-----------|-----------|
| Implementation agent | Code, docs prep, technical replay, unsigned evidence | B6 PASS, maintainer publication, security signature |
| Independent auditor / attestor | Mode A/B replay, signed acceptance | Implement evidence under review; FRZ promotion |
| Operator | ODs, stage closure authorization | Impersonate independent reviewer |
| ECAP maintainer | Amendment activation/revocation | Substitute for security or custody |

## Proof of authority (how each party demonstrates standing)

| Party | Proof |
|-------|--------|
| ECAP maintainer | Signed publication act naming amendment version; org roster or maintainer registry entry |
| Security reviewer | Signed review document + conflict disclosure; not same principal as implementer commit author for attestation tooling |
| Key custodian | HSM/KMS audit log or custodian attestation letter; public key in trust root with `custody_domain=INDEPENDENT_EXTERNAL` |
| Attestor principal | IAM/runner identity without repo admin; workflow change control outside implementer |
| Operator | `OPERATOR APPROVED — OD-*` in register matching chat or signed record |

## Path A vs Path B

| Path | Independent acceptance | ECAP amendment |
|------|------------------------|----------------|
| **B (preferred)** | Machine attestation + `ELIGIBLE` | Required IN_FORCE |
| **A (fallback)** | Human Protocol 03 @ OD-EVOL-11 named auditor | Not required; ECAP-§2.B B6 default |

Path A does not remove OD-EVOL-12, OD-EVOLX-08, or W7-05 requirements.
