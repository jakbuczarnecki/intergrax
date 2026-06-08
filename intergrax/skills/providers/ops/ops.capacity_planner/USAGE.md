# `ops.capacity_planner`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Capacity planning from metrics, cost forecast, and run history.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `metrics.query_range`, `cost.forecast_spend`, `harness.list_runs`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `metrics.query_range` | Catalog tool |
| `cost.forecast_spend` | Catalog tool |
| `harness.list_runs` | Catalog tool |

## Related skills

- Other `ops` bundle skills — see bundle [USAGE.md](../USAGE.md)
