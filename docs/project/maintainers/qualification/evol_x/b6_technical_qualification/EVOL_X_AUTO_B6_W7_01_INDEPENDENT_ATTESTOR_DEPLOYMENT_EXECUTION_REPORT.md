# EVOL-X W7-01 — Independent AUTO-B6 attestor deployment execution report

**Disposition:** **`AUTO-B6 INDEPENDENT ATTESTOR — IMPLEMENTED / DEPLOYMENT READY / EXTERNAL AUTHORITY PENDING`**

**Not claimed:** production independent attestor live · ECAP `IN_FORCE` · signed security review · external key custody · `B6_ACCEPTANCE_PUBLICATION_SHA` · formal Protocol 03 PASS · FRZ promotion · OD-EVOL-12 / OD-EVOLX-08 · EVOL-X CLOSED · E4 execution

| Role | SHA |
|------|-----|
| START HEAD | `da897149ab0c298e66fd81565c2bbdac05966d48` |
| END HEAD | *(set at publication commit for this report)* |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` (unchanged) |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` (unchanged) |
| Qualified CI | `41337077334` — 290/290 (not re-dispatched) |

Machine inventory: [EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json](./EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json).  
Deployment package: [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md](./EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md).  
Trust boundary: [../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md).

## 1. Infrastructure discovered (Batch A)

| Asset | Observation | Class |
|-------|-------------|-------|
| GitHub repo `jakbuczarnecki/intergrax` | Reachable; default branch `main`; work on `development` | AVAILABLE |
| Caller GitHub identity | `admin`, `push`, `workflow` scope | AVAILABLE_BUT_NOT_INDEPENDENT |
| Self-hosted runners | `0` | REQUIRES_PROVISIONING (for isolated attestor runner) |
| GitHub Environments | None listed for repo | REQUIRES_PROVISIONING |
| Repo Actions | Enabled; `allowed_actions=all` | AVAILABLE |
| Secrets (metadata) | `REPO_METADATA_TOKEN` only; no B6 signing secret names | No production custody |
| Cloud KMS / HSM | Not observable from repo credentials | REQUIRES_PROVISIONING |
| Immutable external audit log | Not provisioned | REQUIRES_PROVISIONING |

**Selected topology:** in-repository **reference** attestation contracts + read-only evaluators; **production attestor** must run under a principal and key custody **outside** implementer admin control (see dossier).

## 2. Trust boundary (Batch C)

```text
[Implementer: development push, Cursor, .tmp sessions]
        |  technical Mode A/B, contract tests, evaluate-only
        v
[In-repo verifier + governance JSON (NOT_IN_FORCE)]
        |  requires external proof for PRODUCTION_INDEPENDENT
        v
[Independent attestor — NOT PROVISIONED]
        |  Ed25519 sign, replay ledger append, immutable log
        v
[formal_acceptance_eligibility=ELIGIBLE]  -->  B6_ACCEPTANCE_PUBLICATION_SHA (external act)
```

Implementer **cannot** satisfy `credential_class=PRODUCTION_INDEPENDENT` or `custody_domain=INDEPENDENT_EXTERNAL` by supplying matching JSON fields alone; `evol_x_b6_attestation.py` and governance conjunction fail closed.

## 3. Components deployed in repository (Batch B)

| Component | Path | Role |
|-----------|------|------|
| End-to-end orchestration | `scripts/qualification/evol_x_b6_independent_verification.py` | Worktrees @ FINAL/TES; authority + attestation gates |
| Schema / verify / test sign | `scripts/qualification/evol_x_b6_attestation.py` | Ed25519, replay, governance |
| External evaluator | `scripts/qualification/evol_x_b6_evaluate_signed_attestation.py` | Read-only eligibility JSON |
| Governance record | `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json` | `independent_attestation_environment: false` |
| Trust / ledger templates | `templates/evol_x_b6_trust_root.example.json`, `templates/evol_x_b6_replay_ledger.example.json` | Public shapes only |

No production private keys were generated or committed.

## 4. Qualification executed this wave (Batch E)

| Check | Result | Notes |
|-------|--------|-------|
| Attestation contract tests | **29/29 PASS** @ START HEAD | `tests/unit/qualification/test_evol_x_b6_attestation_contract.py` |
| Mode B VL3 290 replay | **Not re-run** | Qualified TES unchanged; prior wave PASS retained |
| Verifier Mode A @ FINAL | **PASS** | Session `.tmp/session/w7-01-auto-b6-attestor/` |
| Adversarial invalid `--evidence-ref` | **PASS (fail-closed)** | exit 2; control `invalid_evidence_ref_equals_tes` |
| Verifier with `--skip-mode-b` | `technical_verification=INCOMPLETE` (expected) | Full automation PASS not re-claimed without Mode B |
| Formal eligibility | **`NOT_ELIGIBLE`** | Authority + attestation gates |

Negative controls remain covered by the **29** contract tests (forged key, revoked key, replay, implementer attestor, policy drift, FRZ promotion, etc.).

## 5. ECAP and security (Batch F)

| Field | Value |
|-------|-------|
| Amendment | `ECAP-P4-EVOL-SCOPE-AUTO-B6` `2026-11-02.1-pending` |
| `ecap_amendment_activation` | `PENDING_REQUIRED_APPROVALS` |
| `normative_effect` | **`NOT_IN_FORCE`** |
| Signed independent security review | **ABSENT** |
| Maintainer publication act | **ABSENT** |

Source of truth: [EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](../EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json), [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md).

## 6. Formal B6 (Batch G)

**FORMAL B6 = PENDING**

| Gate | Status |
|------|--------|
| ECAP maintainer `IN_FORCE` | BLOCKED |
| Signed security review | BLOCKED |
| Independent key custody + attestor environment | BLOCKED |
| OD-EVOL-11 named auditor (G-02) | BLOCKED |
| Externally signed `PRODUCTION_INDEPENDENT` attestation | ABSENT |
| `B6_ACCEPTANCE_PUBLICATION_SHA` | **null** |

## 7. E4 (Batch H)

**Not executed.** W7-03..W7-06 remain blocked until authentic formal B6 and operator decisions OD-EVOL-12 / OD-EVOLX-08. Handoff prep unchanged: [EVOL_X_E4_W7_HANDOFF_DOSSIER.md](./EVOL_X_E4_W7_HANDOFF_DOSSIER.md).

## 8. External provisioning checklist

1. Record named OD-EVOL-11 auditor or AUTO-B6 custodian (not implementer/agent).
2. Complete signed security review over amendment + schema + key ceremony.
3. ECAP maintainer publication → `IN_FORCE` (separate from implementer commits).
4. Provision attestor principal (read-only @ pinned TES/FINAL), external Ed25519 custody, published trust root, append-only replay ledger, immutable log.
5. Attestor runs Mode A+B, signs attestation; evaluator reports `ELIGIBLE`.
6. Authorized operator publishes Protocol 03 PASS + `B6_ACCEPTANCE_PUBLICATION_SHA`.

## 9. Unresolved external blockers (exact)

- `MISSING_ECAP_MAINTAINER_PUBLICATION`
- `MISSING_SECURITY_REVIEW`
- `MISSING_INDEPENDENT_KEY_CUSTODY`
- `ATTESTATION_ENVIRONMENT_NOT_PROVISIONED`
- `POLICY_NOT_IN_FORCE`
- G-02 OD-EVOL-11 auditor identity not named
- OD-EVOL-12 PENDING · OD-EVOLX-08 PENDING · FRZ-EVOL-01..10 OPEN

**EVOL-X parent status:** OPEN
