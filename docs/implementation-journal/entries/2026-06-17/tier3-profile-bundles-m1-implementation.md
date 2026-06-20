---
id: IJ-2026-06-17-005
date: 2026-06-17
tiers:
  - tier-3
scope: TIER3_APPLICATION_ENVIRONMENT
plan_ref:
  - APP-EVOL-8.1
  - APP-EVOL-8.2
  - APP-EVOL-8.3
  - APP-EVOL-8.4
  - APP-EVOL-8.5
  - APP-EVOL-8.7
status: completed
commit: 9538f616
adr: ADR-APP-003 — implementation follows accepted decision; no new ADR
---

# Tier-3 — APP-EVOL-8 M1 hierarchical profile bundles

## Operator request

Accept Layer Completion proposals 1–7 and 10; implement APP-EVOL-8 M1 (nested bundles + flat shims) without YAML-first or second composition root.

## Summary

Replaced monolithic `environment_profile.py` with `environment_profile/` package: seven nested bundles on `ApplicationEnvironmentProfile`, flat property shims with setters for wire compat, bundle-normalized snapshot/diff digests, `ProfileInvariantValidator`, deploy-time snapshot cache, shared `reference_capability_bundle`, and schema gate script.

## Project impact

Tier-3 hosts keep flat `spec_version` 1.x wire shape and existing wiring entry points; authors may construct profiles via nested bundles. `EnvironmentSnapshot` digests are stable across flat vs nested semantic equivalence.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` §22.6 |
| Plan | `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — APP-EVOL-8 |
| ADR | `docs/adr/entries/2026-06-17/ADR-APP-003.md` |
| Root | `intergrax/applications/contracts/environment_profile/root.py` |
| Normalization | `intergrax/applications/contracts/environment_profile/normalization.py` |
| Gate | `scripts/check_environment_profile_bundle_schema.py` |

## Changed artifacts

- `intergrax/applications/contracts/environment_profile/` — package (bundles, root, normalization, domain_policy)
- `intergrax/applications/_shared/reference_capability_bundle.py`
- `intergrax/applications/_shared/environment_snapshot_wiring.py` — bundle digest + deploy cache
- `intergrax/applications/_shared/environment_diff_wiring.py` — bundle_dump diff
- `intergrax/applications/_shared/environment_conformance.py` — ProfileInvariantValidator
- `tests/unit/applications/test_environment_profile_bundles.py`
- `scripts/check_environment_profile_bundle_schema.py`
- `scripts/generate_domain_audit_prompts.py` · `docs/audit/TIER3_APPLICATION_ENVIRONMENT.md`

## Verification

```bash
uv run pytest tests/unit/applications/test_environment_profile.py tests/unit/applications/test_environment_profile_bundles.py -q
uv run python scripts/check_environment_profile_bundle_schema.py
```

## Remaining

- APP-EVOL-8.6 — `spec_version` 2.0 nested canonical wire (M3)
- Full `tests/unit/applications/` suite requires FastMCP server optional dep in local venv

## Risks and follow-ups

- M3 wire break (`spec_version` 2.0) needs migration tooling before STRICT hosts adopt nested canonical JSON.
- Local CI without `fastmcp[server]` cannot collect manifest parametrized tests that import legal/research factories.
- `bundle_normalized_payload` strips null leaves — document if external digest consumers assumed null-preserving JSON.
