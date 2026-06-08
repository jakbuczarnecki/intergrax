# `cost.budget_guardian`

**Bundle:** `cost` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Run budget enforcement for governed agent hosts.

## How it works

cost.* tools via CostBackend; resolved at SkillResolver registration.

## How to use

cost_skill_profile(); enable on trusted operator hosts.

## What you get

Quota checks before expensive tool loops.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cost.check_quota` | Check remaining quota |
| `cost.get_run_budget` | Fetch run budget |
| `cost.forecast_spend` | Forecast spend trajectory |

## Related skills

- `billing.usage_tracker`
- `ops.trace_debug`
