# EBE-9 — Host-side boundary event signing (BoundaryAttest handoff)

**Audience:** BoundaryAttest verifier authors · **Golden vector:** [`ebe9_golden_vector.v1.json`](ebe9_golden_vector.v1.json)

## Overview

When `host_signing_enabled=true` on `ExecutionBoundaryExportProfile`, Intergrax emits **one Ed25519 host attestation per** `execution_boundary_event.v1`. BoundaryAttest verifies the host signature, then may add its own separate `client_observed` wrapper.

## Canonical event digest

1. Take the complete unsigned event (`signed: false`; exclude `host_attestation` if present).
2. Serialize with UTF-8 JSON: `sort_keys=True`, `separators=(",", ":")`, `default=str`.
3. `signed_payload_hash = "sha256:" + sha256(bytes).hexdigest()`.

## Host-attestation statement (signed bytes)

```json
{
  "context": "boundaryattest.host-attestation.v1",
  "payload_schema_id": "execution_boundary_event.v1",
  "signed_payload_hash": "sha256:<hex>",
  "signature_algorithm": "Ed25519",
  "public_key_id": "attestation-demo-host-1",
  "signed_at": "2026-06-19T12:00:00+00:00"
}
```

Canonicalize with the **same JSON rules** as the event digest. Ed25519-sign the canonical statement bytes.

## Wire format (`boundary_events[]` element)

```json
{
  "...": "execution_boundary_event.v1 fields",
  "signed": true,
  "host_attestation": {
    "schema_id": "host_attestation_envelope.v1",
    "context": "boundaryattest.host-attestation.v1",
    "payload_schema_id": "execution_boundary_event.v1",
    "signed_payload_hash": "sha256:<hex>",
    "signature_algorithm": "Ed25519",
    "public_key_id": "attestation-demo-host-1",
    "signed_at": "2026-06-19T12:00:00+00:00",
    "signature": "<base64>"
  }
}
```

When signing disabled: `signed: false`, `host_attestation: null` (v2 unsigned unchanged).

## Verification (BoundaryAttest)

1. Recompute event digest → must equal `host_attestation.signed_payload_hash`.
2. Rebuild statement from envelope fields (excluding `schema_id` and `signature`).
3. Verify Ed25519 signature with pinned `public_key_ed25519` from handoff.

## PoC configuration

| Item | Value |
|------|-------|
| Profile | `host_signing_enabled=true` in `attestation_demo` manifest |
| Public key id | `attestation-demo-host-1` |
| Pinned pubkey | see golden vector `public_key_ed25519` |
| Dev signing key | `INTERGRAX_EBE_HOST_SIGNING_KEY` or documented PoC seed in golden vector |

## Non-claims

Host signature is a **runtime/host attestation** only. It does not prove truth, authorization, uncompromised runtime, or final business outcome.

## Test matrix (Intergrax)

| Case | Expected |
|------|----------|
| Valid | verify passes |
| Tampered event | digest mismatch |
| Wrong key | signature verify fails |
| Unknown `public_key_id` | partner policy decision; Intergrax still emits stated id |
| Unsigned (`host_signing_enabled=false`) | `signed: false`, no envelope |
