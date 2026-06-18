---
id: IJ-2026-06-17-013
date: 2026-06-17
tiers:
  - tier-0
  - tier-3
scope: OBSERVABILITY
plan_ref:
  - OBS-EVOL-9.10
status: completed
commit: 24e09990
adr: ADR-OBS-003
---

# OBS-EVOL-9.10 — Declarative ObservabilityProfile event subscriptions

## Operator request

Continue OBS-EVOL-9 — declarative bus subscriptions on ObservabilityProfile (SAR-03).

## Summary

Added `EventSubscriptionSpec` and `ObservabilityProfile.event_subscriptions`, handler registry for Tier-3 callbacks, `wire_observability_event_subscriptions()` wired from `build_harness_host_runtime`, and unit tests.

## Project impact

Tier-3 hosts can declare taxonomy-filtered bus handlers in the environment profile without imperative subscribe calls scattered in factories.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/OBSERVABILITY.md` §4.4.11 |
| Plan | `docs/plan/OBSERVABILITY.md` OBS-EVOL-9.10 |

## Changed artifacts

- `intergrax/applications/contracts/environment_profile/sub_profiles.py`
- `intergrax/applications/contracts/environment_profile/__init__.py`
- `intergrax/applications/_shared/event_subscription_registry.py`
- `intergrax/applications/_shared/observability_wiring.py`
- `intergrax/applications/_shared/harness_host_runtime.py`
- `tests/unit/applications/test_observability_event_subscriptions.py`
- `docs/architecture/OBSERVABILITY.md`
- `docs/plan/OBSERVABILITY.md`

## Verification

```bash
uv run pytest tests/unit/applications/test_observability_event_subscriptions.py -q
```

## Risks and follow-ups

- Handlers must be registered before `build_harness_host_runtime` when profile lists subscriptions.
- OBS-EVOL-9.11 W3C Trace Context remains the next planned item.
