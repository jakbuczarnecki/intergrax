---
id: IJ-2026-06-11-041
date: 2026-06-11
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-3
status: completed
commit: pending
adr: none — environment-scoped alias registry; minimal Tier-1 intake persist hook
---

# Sprint 16 — CapabilityAlias registry and sunset routing (APP-EVOL-3)

## Operator request

Continue Tier-3 application architecture sprint queue: APP-EVOL-3 — capability alias registry with deprecation routing per UAEP §42.27 and §49.3.

## Summary

- `capability_alias.py` — `CapabilityAlias`, `CapabilityDescriptor`, `CapabilityGovernanceProfile`.
- `capability_governance_profile` on `ApplicationEnvironmentProfile`.
- `capability_alias_wiring.py` — registry build, sunset resolution, manifest canonical check, 14-day window validation.
- `CapabilityAliasMiddleware` (priority 34) + `apply_capability_alias_wiring` on `build_harness_host_runtime`.
- `TaskMetadataKey.CAPABILITY_ALIAS_REDIRECT` + intake persist in `nexus_lifecycle_hooks.py`.
- `scripts/check_capability_alias_registry.py` wired into production gates.

## Project impact

Tier-3 hosts can declare legacy→canonical capability aliases with bounded sunset windows; STRICT intake blocks expired aliases; manifest roster validation rejects alias strings on `AgentBinding`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §49.3 · UAEP §42.27 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` APP-EVOL-3 · §6.2y step 13 |

## Changed artifacts

- `intergrax/applications/contracts/capability_alias.py`
- `intergrax/applications/contracts/environment_profile.py`
- `intergrax/applications/_shared/capability_alias_wiring.py`
- `intergrax/applications/_shared/capability_alias_middleware.py`
- `intergrax/applications/_shared/capability_alias_intake_wiring.py`
- `intergrax/applications/_shared/harness_host_runtime.py`
- `intergrax/runtime/task/task_metadata_keys.py`
- `intergrax/runtime/hooks/nexus_lifecycle_hooks.py`
- `scripts/check_capability_alias_registry.py`
- `scripts/check_application_production_gates.py`

## Verification

```bash
uv run pytest tests/unit/applications/test_capability_alias_wiring.py \
  tests/unit/applications/test_capability_alias_middleware.py \
  tests/unit/scripts/test_check_capability_alias_registry.py \
  tests/unit/scripts/test_check_application_production_gates.py -q
uv run python scripts/check_capability_alias_registry.py
python scripts/check_implementation_journal.py
```

Result: pass.

## Risks and follow-ups

- Reference hosts ship with empty alias lists — enable aliases per product when migrating capabilities (e.g. `research.pipeline` → `research.orchestrate`).
- Next queue item: APP-EVOL-4 agent certification STRICT roster gate.
