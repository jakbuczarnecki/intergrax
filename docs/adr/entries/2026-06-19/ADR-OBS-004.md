# ADR-OBS-004: EBE-9 Host-Side Boundary Event Signing

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-19 |
| **Deciders** | Harness platform + BoundaryAttest partner |
| **Related** | [ADR-OBS-002](2026-06-13/ADR-OBS-002.md) · [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §18 · `applications/attestation_demo/partner_handoff/` |

## Context

PoC v2 (EBE-8) delivers unsigned `execution_boundary_event.v1` facts with optional `client_observed` receipts on the partner side. BoundaryAttest (formerly AgentReceipt) requested a minimal **host/runtime attestation** layer (EBE-9) that:

- Signs each complete boundary event via a canonical host-attestation **statement** (not raw digest only)
- Keeps unsigned v2 compatible when `host_signing_enabled=false`
- Preserves one signature per event and dual claims on tool failure
- Leaves partner `client_observed` wrapper separate from host signature

## Decision

Add optional **EBE-9 host signing** to Execution Boundary Export:

1. **Profile:** `host_signing_enabled` + `host_signing_public_key_id` on `ExecutionBoundaryExportProfile`
2. **Key material:** `INTERGRAX_EBE_HOST_SIGNING_KEY` (32-byte Ed25519 seed, base64 or hex); dev PoC fallback seed documented in golden vector
3. **Digest:** canonical JSON of unsigned `execution_boundary_event.v1` → `signed_payload_hash`
4. **Signature:** Ed25519 over canonical JSON statement:
   - `context`: `boundaryattest.host-attestation.v1`
   - `payload_schema_id`, `signed_payload_hash`, `signature_algorithm`, `public_key_id`, `signed_at`
5. **Wire:** `boundary_events[]` element includes `signed: true` and `host_attestation` envelope; when disabled: `signed: false`, `host_attestation: null`
6. **Trust label:** `host_attested` in API `trust_model` when signing enabled — not `server_attested` and not proof of business truth

## Consequences

### Positive

- Partner can verify host claim independently with pinned public key
- Statement binding prevents envelope metadata substitution
- Unsigned v2 path unchanged for hosts that disable signing

### Negative

- Key management deferred to host operator (PoC uses env seed)
- Partner must implement statement + event hash verification

## Compliance

- HOS spine unchanged
- Tier boundaries preserved (`intergrax/runtime/attestation/`)
- Webhook delivery (EBE-7) remains deferred

## Verification

```bash
uv run pytest tests/unit/runtime/attestation/ applications/attestation_demo/attestation_demo_tests -q
```

Golden vector: `applications/attestation_demo/partner_handoff/ebe9_golden_vector.v1.json`
