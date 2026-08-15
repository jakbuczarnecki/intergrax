# ADR-HOST-ATTESTATION-001: Host attestor and portable ProofReceipt

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / Execution Evidence |
| **Related** | ADR-EXECUTION-BOUNDARY-EVENT-001 · EBE-9 HostAttestationSealer · docs/project/technical/platform/execution_evidence_and_host_attestation.md |

## Context

Partner validation needs one portable attested export binding the governed boundary event. Existing pieces:

- `HostAttestationSealer` / `HostAttestationEnvelopeV1` — Ed25519 signing for harness EBE (reuse crypto + canonical statement pattern)
- `intergrax.proofs.receipts.ProofReceipt` (`intergrax.proof_receipt.v1`) — DocumentStore persistence for LKW-style proof results — **not** a signed EBE export

## Decision

1. Introduce injectable `HostAttestor` Protocol + `HostAttestation` contract for execution-evidence payloads (algorithm, key_id, payload_digest, signature, signed_at).
2. Default test/local implementation: Ed25519 over canonical payload bytes (or digest envelope), DI-replaceable by KMS/HSM later — no production key custody claimed.
3. Introduce portable `ProofReceipt` under `intergrax.contracts.execution_evidence` with schema `execution_evidence.proof_receipt.v1` binding the full governed event + host attestation.
4. Do **not** overload `intergrax.proofs.receipts.ProofReceipt` (persistence product).
5. Verifier recalculates canonical bytes/digest and checks signature; never authorizes execution; offline; no provider network.
6. Tier-2 must not import signing implementations; providers never sign Intergrax receipts.

## Consequences

### Positive

- Reuses Ed25519 + canonical_json patterns from EBE-9
- Clear separation from DocumentStore ProofReceipt

### Negative

- Name collision at the English product level — always qualify by schema/module

## Compliance

- Attestation failure must not retry provider side effects
- Unsigned artefacts must never be marked attested/verified

## Implementation notes

- `intergrax/runtime/execution_evidence` attestor + verifier
- Host orchestration after Tier-2 returns proof
