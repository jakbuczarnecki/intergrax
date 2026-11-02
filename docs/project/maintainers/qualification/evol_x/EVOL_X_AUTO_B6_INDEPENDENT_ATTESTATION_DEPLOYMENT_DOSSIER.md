# EVOL-X automated B6 — independent attestation deployment dossier

**Disposition:** **`AUTOMATED B6 IMPLEMENTED / FORMAL ATTESTATION DEPLOYMENT BLOCKED`**

A Cursor subagent, a second git worktree, or an implementer-run verifier is not an independent attestor. This repository does not contain an acceptable independent attestation environment.

## Trust boundary

| Side | What it may do |
| --- | --- |
| Implementer workspace (this repo, Cursor agents, `development` worktrees) | Change qualification code and docs. Run technical Mode A/B. Evaluate attestation records. Mint **only** `NON_PRODUCTION_TEST_ONLY` signatures with disposition `NOT_ACCEPTED`. |
| Independent attestor (not provisioned) | Read pinned TES/FINAL inputs. Hold the signing key. Append the replay ledger. Emit the signed attestation. |

The verifier process must not be the key custodian. No production private key is generated or stored by this change.

## Environment checklist

| Control | Repository state |
| --- | --- |
| Distinct execution identity | **Absent.** No service account or runner identity is dedicated to B6 attestation. |
| Separate credential custody | **Absent.** `sign_attestation` refuses production keys. No external KMS/HSM trust root is configured. |
| Read-only qualification inputs | **Specified, not deployed.** The future attestor must verify pinned SHAs `3c4a641b3e48a13f437e830a0562e2a80b03aabb` and `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` without write access to `development`. |
| Immutable or independently verifiable logs | **Absent.** Session logs under `.tmp/` are implementer-local and are not an audit ledger. |
| Authorization policy | **Recorded, not in force.** OD-EVOL-14 plus amendment `2026-11-02.1-pending` with activation `PENDING_REQUIRED_APPROVALS`. |
| Signature validation | **Implemented** in `scripts/qualification/evol_x_b6_attestation.py` (Ed25519, exact schema, fail closed). |
| Key lifecycle | **Specified, not deployed.** Trust-root keys need `ACTIVE`/`REVOKED`, a published public key, rotation, and a custodian distinct from the implementer. |
| Recovery / revocation | **Specified, not deployed.** Revoked keys fail closed. There is no production revocation publisher. |
| Replay protection | **Contract implemented.** Formal eligibility requires an external replay ledger. This workspace does not own that ledger and does not append one. |

## What an authorized party must provision

1. Name a key custodian who is not the EVOL-X implementer and not a Cursor agent. Record that identity in a successor to OD-EVOL-11 or in an OD-EVOL-14 supplement.
2. Complete a signed security review of this amendment, the attestation schema `evol_x_b6_attestation.v1`, and the custodian's key ceremony.
3. ECAP maintainer publication that sets amendment `ECAP-P4-EVOL-SCOPE-AUTO-B6` version `2026-11-02.1-pending` (or a successor version) to activation `IN_FORCE`, by an act that is not the implementer commit of the draft.
4. Publish a trust root (`evol_x_b6_trust_root.v1`) with `custody_domain=INDEPENDENT_EXTERNAL` and at least one `ACTIVE` Ed25519 public key bound to the attestor identity. Private key material stays outside this repository.
5. Provide an append-only replay ledger and an immutable verification log that the implementer cannot rewrite.
6. Run Mode A at FINAL and Mode B at TES from that identity, then sign an attestation whose `credential_class` is `PRODUCTION_INDEPENDENT` and whose `formal_acceptance_disposition` is `ACCEPTED_AUTOMATED_B6`.
7. Only after the evaluator reports `formal_acceptance_eligibility=ELIGIBLE` for that independently produced record, publish `B6_ACCEPTANCE_PUBLICATION_SHA`.

Until those steps exist, formal B6 stays **PENDING** and test credentials stay unclassified for acceptance.

Deployable templates and read-only evaluator entry points: [b6_technical_qualification/EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md](./b6_technical_qualification/EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md).

## Explicit non-claims

- Governance file `independent_attestation_environment` is **false**.
- `credential_class_authorized_for_formal_b6` is **null**.
- No attestation private key in this workspace authorizes formal B6.
