# ADR-SEC-001: Security & Trust Planes and SecurityDefensePlugin entry points

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-19 |
| **Deciders** | Harness platform (idea audit SEC-PLANES) |
| **Related** | [`intergrax_runtime_architecture.md`](../../intergrax_runtime_architecture.md) · [UAEP §42.45](../../architecture/UNIFIED_EXECUTION_RUNTIME.md#4245-security-and-data-governance) · plan SEC-PLANES |

## Context

Operators requested a modular, provider-backed security layer without introducing a parallel `SecurityEngine` tier. Existing V-SEC middleware, `policy_rules`, and integration catalog already covered much capability but lacked a canonical plane model and a first-class extension point for custom runtime defenses.

## Decision

1. Document **Security & Trust Planes** (S1 Trust, S2 Defense, S3 Governance) as a logical index inside UAEP — not a new domain pair or runtime loop.
2. Keep **composition root** at Tier-3 `SecurityEnvelope` + existing UAEP hook timeline.
3. Add **`intergrax.security_defenses`** entry point group with `SecurityDefensePlugin` protocol wrapping into `MiddlewarePipeline` (S2).
4. Ship **`harness.strict_injection`** bundled defense and **`SecurityEnvelope.production()`** / `harness_defense_stack()` presets.
5. Add **encryption enforcement bridge** (`EncryptionEnforcementMiddleware`) for `DataClassification.RESTRICTED` when `secrets_store` is required on strict hosts.

**Rejected:** standalone SecurityEngine; harness-native blockchain; bypass of PolicyEngine/ToolRuntime.

## Consequences

### Positive

- Extension authors have a typed S2 plugin surface aligned with §42.21.
- Wire-time assembly validates defense bundle ids and strict encryption prerequisites.
- Production presets compose S1+S2 without agent code changes.

### Negative

- Third-party defense plugins must be loaded before strict assembly validation (bootstrap order matters).
- Encryption bridge gates on integration profile presence — not field-level KMS encryption yet.

## Compliance

- Tier boundaries preserved (Tier-2 agents do not register defenses).
- Policy-first path unchanged — defenses emit trace via middleware BLOCK.
- Linked architecture and plan docs updated (SEC-PLANES).

## Implementation notes

- Code: `intergrax/runtime/security/defense_*.py`, `encryption_*.py`, `intergrax/core/security_bootstrap.py`
- CI: `scripts/check_harness_security_defense_plugins.py`, `scripts/check_harness_encryption_policy.py`
- Tests: `tests/unit/runtime/security/test_sec_planes.py`
