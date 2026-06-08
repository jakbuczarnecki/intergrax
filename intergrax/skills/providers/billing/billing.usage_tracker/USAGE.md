# `billing.usage_tracker`

**Bundle:** `billing` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Usage metering and run cost correlation.

## How it works

billing.* + harness.get_run_cost for trace correlation.

## How to use

billing_skill_profile(); platform metering hosts.

## What you get

Chargeback visibility for multi-tenant deployments.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `billing.list_usage` | List usage records |
| `billing.record_usage` | Record usage event |
| `harness.get_run_cost` | Fetch run cost from trace |

## Related skills

- `cost.budget_guardian`
- `metrics.run_observer`
