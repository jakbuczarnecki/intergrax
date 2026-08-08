# ADR-RUNTIME-POLICY-BUNDLE-001: Immutable attested policy bundle identity

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / Execution Evidence |
| **Related** | ADR-POLICY-SIDE-EFFECT-001 · docs/project/technical/platform/execution_evidence_and_host_attestation.md |

## Context

Partner validation requires an explicit, digestable runtime policy pack identity on meaningful side-effect decisions. The existing `intergrax.runtime.policy.policy_bundle.RuntimePolicyBundle` is a **live wiring composition** (tool access, budgets, plan-loop engines) — mutable references, not a signed-ready immutable pack.

## Decision

1. Introduce an immutable contract `ImmutableRuntimePolicyBundle` (`runtime_policy_bundle.v1`) with `bundle_id`, `version`, ordered `rules`, `issued_at`, and `canonical_digest`.
2. Do **not** replace or reshape the live wiring dataclass.
3. Extend `PolicyDecision` with optional `policy_bundle_id` / `policy_bundle_version` / `policy_bundle_digest` (plus existing `policy_rule_id` / `action`).
4. When host attestation is required, missing bundle identity on an ALLOW decision fails closed — no attested receipt.
5. Digest uses existing `canonical_json` (sorted keys, UTF-8) over the bundle excluding the digest field; rule order is part of the digest input (tuple order preserved as a JSON array).

## Consequences

### Positive

- Decisions identify the exact pack used without rebuilding the policy engine
- Live Nexus wiring remains unchanged

### Negative

- Two policy-bundle concepts exist; docs must disambiguate by module path

## Compliance

- Provider-neutral; no transport payloads in the bundle
- Tier-2 does not own pack issuance

## Implementation notes

- `intergrax/contracts/runtime_policy_bundle.py`
- `intergrax/contracts/runtime_policy.py` field extensions
- Host/demo evaluators stamp bundle refs onto `PolicyDecision`
