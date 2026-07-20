# Execution Evidence & Host Attestation

**Status:** Platform capability (2026-07-20)  
**Role:** Primary reference for post-proof host attestation  
**Upstream boundary:** [`governed_external_execution.md`](governed_external_execution.md) ends at `GovernedProofProfile`  
**Tracker:** [`execution_evidence_implementation_plan.md`](execution_evidence_implementation_plan.md)  
**ADRs:** ADR-RUNTIME-POLICY-BUNDLE-001 · ADR-EXECUTION-BOUNDARY-EVENT-001 · ADR-HOST-ATTESTATION-001

---

## Motivation

Governed External Execution closes at descriptive `GovernedProofProfile`. Partner validation still needs:

1. a host-signed execution boundary event  
2. one portable attested export joining policy, invocation, identity, and proof  
3. explicit runtime policy pack identity and digest  

This capability closes those gaps **without** reopening GEC ownership.

---

## Capability boundary

```text
policy ALLOW → provider execution → GovernedProofProfile
======== GEC / platform governed-external boundary ========
GovernedProofProfile
  → ExecutionBoundaryEvent (governed_execution_boundary_event.v1)
  → HostAttestation
  → ProofReceipt (execution_evidence.proof_receipt.v1)
  → offline verification
```

| Owns | Owner |
|------|-------|
| `GovernedProofProfile` composition | Tier-2 |
| `ExecutionBoundaryEvent` composition | Host |
| Signing | Host (`HostAttestor`) |
| Receipt production | Host |
| Verification | Provider-neutral verifier (no authz) |

---

## Ownership invariants

1. Policy evaluation precedes meaningful side effects (GEC).  
2. Policy decisions used for attested receipts identify an immutable pack.  
3. Provider execution precedes proof composition.  
4. Proof composition precedes boundary event composition.  
5. Boundary event precedes signing.  
6. Only host-owned code signs receipts.  
7. Tier-2 never signs or persists receipts.  
8. Providers never sign Intergrax receipts.  
9–12. Evidence / proof / receipt / verification never authorize execution.  
13–14. Task, run, correlation, and idempotency identities are preserved.  
15. Mutation of any signed field invalidates verification.  
16. Attestation failure must not retry provider side effects.  
17. No attestation without successful governed proof.  
18. No unsigned artefact may be described as host-attested.  
19. Secrets/credentials never enter the signed event.  
20. Schema and signing algorithm are explicit and versioned.

---

## Reuse vs new contracts

| Existing | Role | Reuse? |
|----------|------|--------|
| Harness `ExecutionBoundaryEventV1` (`execution_boundary_event.v1`) | Tool/step BoundaryAttest export | **Sibling** — not overloaded |
| `HostAttestationSealer` / Ed25519 | Harness EBE-9 signing | Crypto + `canonical_json` patterns reused |
| Live `RuntimePolicyBundle` dataclass | Nexus wiring composition | **Not** the attested pack |
| `intergrax.proofs.receipts.ProofReceipt` | DocumentStore LKW persistence | **Not** the portable attested export |

New:

- `ImmutableRuntimePolicyBundle` (`runtime_policy_bundle.v1`)  
- `ExecutionBoundaryEvent` (`governed_execution_boundary_event.v1`)  
- `HostAttestation` + `HostAttestor`  
- portable `ProofReceipt` (`execution_evidence.proof_receipt.v1`)  
- `VerificationResult`

---

## Runtime policy bundle

Immutable pack: `bundle_id`, `version`, ordered `rules`, `issued_at`, `canonical_digest`.  
`PolicyDecision` may carry `policy_bundle_id` / `policy_bundle_version` / `policy_bundle_digest`.  
When attestation is required, missing pack identity fails closed.

---

## Canonicalization

Reuse `intergrax.runtime.attestation.canonical_json`: UTF-8 JSON, `sort_keys=True`, separators `(",", ":")`.  
Digest: `sha256:<hex>`. Schema id is part of signed bytes.

---

## Attestation model

`HostAttestor.attest(payload: bytes, *, schema: str) -> HostAttestation`  
Default test/local: Ed25519 over canonical event bytes. DI-replaceable by KMS/HSM. No production key custody claimed.

---

## Receipt & verification

`ProofReceipt` binds the full event + host attestation.  
Verifier recalculates bytes/digest and checks signature offline. No provider network. No authorization side effects.

---

## Failure semantics

| Case | Result |
|------|--------|
| Policy DENY | No provider call; no success receipt |
| Provider failure | No success receipt |
| Proof composition failure | No signed receipt |
| Attestation failure after success | `execution_succeeded=True`, `attestation_succeeded=False` — **do not** retry provider |
| Missing attestor when required | Fail closed (no attested claim) |

---

## Non-goals

DB receipt store, distributed audit, replay, public registry, blockchain, payments, live providers, wallets, public HTTP API, key rotation service, remote KMS/HSM, CA, transparency log.

---

## Relation to Governed External Execution

GEC ownership ends at proof. This capability is **downstream** and reusable outside External Work. Do not move receipt/signing responsibilities into Tier-2 or GEC trackers.
