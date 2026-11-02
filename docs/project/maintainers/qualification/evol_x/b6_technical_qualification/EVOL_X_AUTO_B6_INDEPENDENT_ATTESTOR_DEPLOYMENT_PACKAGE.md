# EVOL-X AUTO-B6 — independent attestor deployment package

**Disposition:** **`DEPLOYMENT_READY / EXTERNAL PROVISIONING PENDING`**

This package is safe to commit: no private keys, no production attestation, no ECAP activation. It extends [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md) with concrete artifacts an infrastructure owner can deploy outside Cursor and outside implementer write access.

## Principal and access

| Requirement | Deployment specification |
|-------------|-------------------------|
| Distinct principal | Dedicated service account or human attestor identity **not** used for `development` pushes |
| Repository access | Read-only clone or read-only GitHub token; pinned refs only |
| Write access | **Forbidden** to qualification tooling, governance JSON, and ECAP amendment files |
| Signing key | Generated and stored **outside** this repository (HSM/KMS or offline ceremony) |
| Trust root | Publish `evol_x_b6_trust_root.v1` JSON with `custody_domain=INDEPENDENT_EXTERNAL` |

Pinned inputs (must not drift):

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | `41337077334` (290 passed, profile `evol_x_p3_l3`) |

## Artifact templates (public only)

| File | Purpose |
|------|---------|
| [templates/evol_x_b6_trust_root.example.json](./templates/evol_x_b6_trust_root.example.json) | Trust-root shape; replace placeholders after key ceremony |
| [templates/evol_x_b6_replay_ledger.example.json](./templates/evol_x_b6_replay_ledger.example.json) | Append-only replay binding store (JSON array of hex strings) |

## Read-only verification tools (in repo)

| Tool | Role |
|------|------|
| `scripts/qualification/evol_x_b6_independent_verification.py` | Mode A @ FINAL + Mode B @ TES orchestration; **does not** sign formal B6 |
| `scripts/qualification/evol_x_b6_attestation.py` | Schema + Ed25519 verify + governance conjunction |
| `scripts/qualification/evol_x_b6_evaluate_signed_attestation.py` | External evaluator: loads attestation + trust root + optional replay ledger; emits immutable result JSON; **never** mints production keys |

### Attestor workflow (external)

1. Check out **read-only** at TES and FINAL SHAs above.
2. Run full verification (or witness an operator-run report) and capture Mode A/B `PASS` in session JSON under **attestor-controlled** storage (not `.tmp/` in implementer workspace).
3. Build attestation body per `evol_x_b6_attestation.v1`; set `credential_class=PRODUCTION_INDEPENDENT`, `formal_acceptance_disposition=ACCEPTED_AUTOMATED_B6`, bind CI + findings + FRZ scope fields to pinned values.
4. Sign with custody-separated Ed25519 private key; append `replay_binding` to **independent** append-only ledger before publication.
5. Run `evol_x_b6_evaluate_signed_attestation.py` with `--attestation`, `--trust-root`, `--replay-ledger`, and observed Mode A/B — expect `formal_acceptance_eligibility=ELIGIBLE` only when governance is `IN_FORCE` and all approval flags are true.
6. **Only then** may an authorized operator publish `B6_ACCEPTANCE_PUBLICATION_SHA` (not this implementer wave).

## Immutable verification result model

The evaluator returns a single JSON object with separate fields:

- `technical_verification`
- `authority_verification`
- `attestation_verification`
- `formal_acceptance_eligibility` / `formal_acceptance_blockers`

Implementer runs always leave `b6_acceptance_publication_sha` null.

## Revocation and replay

- Trust-root key `status=REVOKED` → verifier fails closed (`REVOKED_SIGNING_KEY`).
- Duplicate `replay_binding` in ledger → `REPLAYED_ATTESTATION`.
- Revocation publisher and ledger appender must be **independent** of the evidence producer.

## ECAP maintainer activation (Batch D)

Amendment [EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md](../EVOL_X_PROPOSED_ECAP_AUTOMATED_B6_ORGANIZATIONAL_ACCEPTANCE_AMENDMENT.md) remains **`PENDING_REQUIRED_APPROVALS` / `NOT_IN_FORCE`**. Maintainer publication is a separate act from this package.

## Security review (Batch C)

Unsigned technical evidence: [EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md). A signed organizational review is still **ABSENT**.

## W7-01 deployment wave

Infrastructure inventory and orchestrator execution report (external authority still pending): [EVOL_X_AUTO_B6_W7_01_INDEPENDENT_ATTESTOR_DEPLOYMENT_EXECUTION_REPORT.md](./EVOL_X_AUTO_B6_W7_01_INDEPENDENT_ATTESTOR_DEPLOYMENT_EXECUTION_REPORT.md).

## Explicit non-claims

- No production attestation environment is provisioned in this repository.
- `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json` keeps `independent_attestation_environment: false`.
- This package does **not** activate ECAP-P4-EVOL-SCOPE-AUTO-B6.
