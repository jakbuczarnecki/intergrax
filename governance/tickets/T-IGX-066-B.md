# T-IGX-066-B — Wave 06 OD-042: neutral GlobalSettings defaults (wariant W1)

## Scope

EQO Wave 06 **AUTH-03**. Implements operator choice **W1** for EQOA-A4-042: neutral default language/locale/region/timezone in `intergrax/globals/settings.py`, UTC as SSOT default timezone for framework defaults, preserved `INTERGRAX_DEFAULT_*` env overrides, unit tests for defaults and tenant-env isolation. No global migration of `os.getenv`; no HostProfile contract change.

Operator decision id: **OD-042**. OD-EQO-09 does not replace this approval.

## Plan digest

Plan digest: 61a983729d71e0e6cea291e0596a0890fcba28d3b33ae06dbccbe99e27836006

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-066-B.md`](plans/PLAN_T-IGX-066-B.md).

## Approved

Approved: OD-042 — T-IGX-066-B wariant W1; plan digest 61a983729d71e0e6cea291e0596a0890fcba28d3b33ae06dbccbe99e27836006; neutral globals + env override preserved.

Approved: operator @ 2026-11-23T07:45:00+01:00

## Write set

```text
intergrax/globals/settings.py
tests/unit/globals/test_global_settings_neutral_defaults.py
tests/unit/globals/test_global_settings_env_override_isolation.py
```

## Test contract

```text
uv run pytest tests/unit/globals/test_global_settings_neutral_defaults.py tests/unit/globals/test_global_settings_env_override_isolation.py -q
uv run pytest tests/qualification/config_x/test_config_x_qualification_gates.py tests/qualification/config_x/test_config_x_r1_r1_semantic_constant_resolution.py -q --tb=line
```

## Out of scope

Vendor calendar tests with explicit `Europe/Warsaw`; CONFIG-X parent re-close; FRZ promotion; repo-wide locale sweep.

## Lifecycle state

in_progress

## Dependencies

May run after **T-IGX-066-A** in same session when both approved; no file overlap.

## Checkpoint

Tests: `.tmp/session/w06/pytest_w06_post_impl.log` (globals), `.tmp/session/w06/pytest_w06_066b_config.log` (CONFIG subset 25 passed).

## Lifecycle state

quality_gate