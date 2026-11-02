# EVOL-X AUTO-B6 — security review technical evidence (UNSIGNED)

**Status:** **TECHNICAL EVIDENCE ONLY — NOT A SIGNED INDEPENDENT SECURITY REVIEW**

**Scope:** `AUTOMATION_FIRST_SCOPED` · amendment `ECAP-P4-EVOL-SCOPE-AUTO-B6` (pending) · attestation schema `evol_x_b6_attestation.v1`

## Threat model summary

| Topic | Control (implemented) | Formal gate |
|-------|----------------------|-------------|
| Attestor identity | Contract fields + trust-root binding | Independent custodian not provisioned |
| AuthN / AuthZ | Ed25519 verify; governance `IN_FORCE` check | Policy not in force |
| Signing key lifecycle | ACTIVE/REVOKED in trust root | No external trust root deployed |
| Privilege boundaries | Implementer `sign_attestation` refuses production class | Verified in unit tests |
| Trust-root ownership | Requires `INDEPENDENT_EXTERNAL` custody domain | Absent |
| Supply-chain / workflow tampering | P03 Mode A/B git-bound digests | Mode A replay PASS @ FINAL |
| Evidence tampering | ARTIFACT_DIGESTS + manifest binding | Covered by P03 |
| Forged / replayed attestation | Signature verify + replay ledger set | Ledger not independent |
| Emergency revocation | Revoked keys fail closed | Unit tests PASS |
| Audit log retention | Specified in deployment dossier | Not deployed |

## Automated adversarial coverage (implementer workspace)

Executed: `tests/unit/qualification/test_evol_x_b6_attestation_contract.py` — **29 passed** @ wave END HEAD.

Categories include: unauthorized signer, forged signature, revoked key, wrong trust root, implementer custody, policy version mismatch, wrong TES/FINAL, stale window, replay, altered Mode A/B, CI binding, false M-09H PASS, FRZ promotion, unsigned record, test credential formal claim.

## Verifier session (orchestrated)

Session root: `.tmp/session/evol-x-auto-b6-full-wave-v2/` (gitignored). Mode A **PASS**; adversarial invalid ref **PASS**; authority **PENDING_REQUIRED_APPROVALS**; formal eligibility **NOT_ELIGIBLE**.

## Required independent act

A reviewer **distinct from the implementer** must sign an organizational security approval before ECAP activation. This document is **not** that signature.
