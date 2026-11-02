# EVOL-X — independent attestor deployment plan

**Authority:** [EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md](./EVOL_X_COMPLETE_CLOSURE_AND_CERTIFICATION_MASTER_PLAN.md).  
**Related:** [EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](./EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md) · [b6_technical_qualification/EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md](./b6_technical_qualification/EVOL_X_AUTO_B6_INDEPENDENT_ATTESTOR_DEPLOYMENT_PACKAGE.md).

**Status:** **DEPLOYMENT READY (in-repo reference)** · **PRODUCTION ATTESTOR NOT LIVE**

---

## 1. Trust boundary

```text
Implementer (admin/push, Cursor, .tmp sessions)
    → technical verification only (Mode A/B, 29 contract tests)
In-repo verifier (read-only evaluate)
    → requires PRODUCTION_INDEPENDENT + IN_FORCE + external custody
Independent attestor (TO PROVISION)
    → Ed25519 sign, replay ledger, immutable log
Formal B6 publication commit (authorized human publisher)
```

**Inventory:** [EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json](./b6_technical_qualification/EVOL_X_AUTO_B6_INFRASTRUCTURE_INVENTORY.json) — `topology_independence: INSUFFICIENT_UNTIL_EXTERNAL_PROVISIONING`.

---

## 2. Target topology (production)

| Component | Requirement |
|-----------|-------------|
| **Execution principal** | Read-only clone @ `3c4a641b…` (TES) and `bbf3e33b…` (FINAL); no write/admin on `jakbuczarnecki/intergrax` |
| **Runner** | Self-hosted runner **or** separate org workflow; implementer cannot modify workflow + secrets jointly |
| **Signer** | Ed25519 private key in KMS/HSM or custodian vault; never in implementer workspace |
| **Trust root** | Published JSON: public keys, rotation, revocation list |
| **Replay ledger** | Append-only; attestation id bound once |
| **Audit log** | Immutable store (object lock / WORM / external SIEM) for verify + sign events |
| **Verifier** | Repo script `evol_x_b6_evaluate_signed_attestation.py` — read-only |

---

## 3. In-repo components (deployed)

| Path | Role |
|------|------|
| `scripts/qualification/evol_x_b6_independent_verification.py` | Orchestration |
| `scripts/qualification/evol_x_b6_attestation.py` | Schema, verify, test-only sign |
| `scripts/qualification/evol_x_b6_evaluate_signed_attestation.py` | Eligibility evaluation |
| `EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json` | Honest activation state |
| `templates/evol_x_b6_trust_root.example.json` | Shape reference only |
| `templates/evol_x_b6_replay_ledger.example.json` | Shape reference only |

---

## 4. Provisioning checklist (Batch 1 → attestor live)

1. Complete [EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md](./EVOL_X_ECAP_P4_MAINTAINER_PUBLICATION_ACT.md) (maintainer).
2. Complete signed [EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_SIGNOFF_RECORD.md).
3. Key ceremony → publish trust root outside repo; record custodian.
4. Provision runner + read-only GitHub token or fork mirror pinned to SHAs.
5. Deploy attestor job: run orchestrator; emit signed attestation JSON.
6. Run evaluator → expect `formal_acceptance_eligibility=ELIGIBLE`.
7. Update governance JSON: `independent_attestation_environment: true` (only after steps 1–6 real).

---

## 5. Key lifecycle

| Event | Procedure |
|-------|-----------|
| Rotation | New key in trust root; old key `revoked`; attestor refuses revoked kid |
| Revocation | Immediate ledger + trust root update; re-attestation required |
| Compromise | Halt signing; operator incident; W7-02 if evidence re-qualification needed |

Contract tests: `test_revoked_signing_key_is_rejected`, `test_unknown_trust_root_is_rejected`.

---

## 6. Safe retries

- Attestor jobs idempotent on same input digest; replay ledger rejects duplicate attestation id.
- Orchestrator may re-run Mode B; basetemp **outside** nested worktree (F-001).

---

## 7. Failure recovery

| Failure | Recovery |
|---------|----------|
| Mode B fail on attestor | Fix environment; do not change TES without W7-02 |
| Evaluator NOT_ELIGIBLE | Read blockers JSON; fix governance/custody/policy |
| Partial deploy | Keep `evol_x_closed: false`; document in Batch 1 execution record |

---

## 8. What does **not** satisfy independence

- Second Cursor session or worktree as attestor
- GitHub-hosted workflow on same repo where implementer has admin
- `NON_PRODUCTION_TEST_ONLY` test signatures
- Unsigned governance flag edits
