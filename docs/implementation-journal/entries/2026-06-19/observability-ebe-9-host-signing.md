---
id: IJ-2026-06-19-006
date: 2026-06-19
tiers:
  - tier-1
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - EBE-9
status: completed
commit: 181ded35
adr: docs/adr/entries/2026-06-19/ADR-OBS-004.md
---

# EBE-9 — Host-side boundary event signing for BoundaryAttest

## Operator request

Implement full EBE-9 PoC on development: optional Ed25519 host signing per `execution_boundary_event.v1` using canonical host-attestation statement (per BoundaryAttest agreement), golden vector, attestation_demo enablement, tests, docs, and Docker verification.

## Summary

Added `canonical_json.py`, `host_attestation.py` (sealer + verifier), profile flags `host_signing_enabled` / `host_signing_public_key_id`, buffer sealing at append, dynamic `trust_model` (`host_attested`), golden vector `ebe9_golden_vector.v1.json`, and comprehensive tests (valid/tampered/wrong-key/unsigned + live PoC paths). Restored `boundary_demo_agent` `allowed_tools` for records.put after ACP skill migration regression.

## Project impact

Partners can verify Intergrax host/runtime claims with pinned Ed25519 keys while keeping unsigned v2 and separate BoundaryAttest `client_observed` wrappers.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/OBSERVABILITY.md` §18 EBE-9 |
| Plan | `docs/plan/OBSERVABILITY.md` EBE-9 |
| ADR | `docs/adr/entries/2026-06-19/ADR-OBS-004.md` |
| Handoff | `applications/attestation_demo/partner_handoff/EBE-9_HOST_SIGNING.md` |

## Changed artifacts

- `intergrax/runtime/attestation/` — canonical JSON, host sealer, buffer, schema
- `intergrax/applications/contracts/environment_profile/sub_profiles.py`
- `applications/attestation_demo/` — manifest, router, tests, golden vector
- `agents/boundary_demo/boundary_demo_agent.py` — allowed_tools fix
- `pyproject.toml` — `cryptography` dependency

## Verification

```bash
uv run pytest tests/unit/runtime/attestation/ applications/attestation_demo/attestation_demo_tests -q
```

Result: 40 passed. Docker live PoC run: 2 signed events, `host_attested` trust model.

## Risks and follow-ups

- EBE-7 webhook remains deferred.
- Production key management (KMS/HSM, rotation) out of PoC scope.
