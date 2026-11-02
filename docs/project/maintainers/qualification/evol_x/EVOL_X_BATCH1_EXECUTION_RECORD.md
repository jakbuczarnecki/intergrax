# EVOL-X Batch 1 — external authority execution record

**Wave:** Path B trust activation  
**START_HEAD:** `9c893139e89ed885e8f30e66c14b3cc363a8ac15`  
**Disposition:** **IMPLEMENTER SCOPE COMPLETE — EXTERNAL ACTS REQUIRED**

---

## Implementer-delivered artifacts

| Artifact | Path |
|----------|------|
| Maintainer publication act template | [EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md](./EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md) |
| Security sign-off template | [EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md) |
| Attestor deployment plan | [EVOL_X_INDEPENDENT_ATTESTOR_DEPLOYMENT_PLAN.md](./EVOL_X_INDEPENDENT_ATTESTOR_DEPLOYMENT_PLAN.md) |
| Infrastructure inventory | [b6_technical_qualification/EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json](./b6_technical_qualification/EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json) |

---

## External actions (not performed by implementer)

| Action | Status |
|--------|--------|
| ECAP maintainer `IN_FORCE` publication | **OPEN** |
| Independent security reviewer signature | **OPEN** |
| Production key custody + trust root publish | **OPEN** |
| Attestor environment provisioned | **OPEN** |

---

## Governance JSON (unchanged — honest)

[EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json):

- `normative_effect`: `NOT_IN_FORCE`
- `independent_attestation_environment`: `false`
- `required_approvals_present.ecap_maintainer_publication`: `false`
- `required_approvals_present.security_review`: `false`
- `required_approvals_present.independent_key_custody`: `false`

**Implementer did not set `IN_FORCE` or production custody flags without authority.**

---

## Batch 1 exit criteria

| Criterion | Result |
|-----------|--------|
| Templates and runbooks published | **PASS** |
| Maintainer act signed | **BLOCKED** |
| Security review signed | **BLOCKED** |
| Custody + attestor live | **BLOCKED** |
| `independent_attestation_environment: true` | **NOT SET** (correct) |

**Batch 1 implementer todo:** **COMPLETE**  
**Batch 1 formal trust activation:** **BLOCKED_EXTERNAL_AUTHORITY** — operator must execute external checklist in [EVOL_X_INDEPENDENT_ATTESTOR_DEPLOYMENT_PLAN.md](./EVOL_X_INDEPENDENT_ATTESTOR_DEPLOYMENT_PLAN.md) §4.
