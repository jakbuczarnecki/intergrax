# `harness.cost_analyst`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Run cost analysis with compare and instant metrics.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `harness` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `harness.get_run_cost`, `harness.compare_runs`, `metrics.query_instant`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `harness.get_run_cost` | Catalog tool |
| `harness.compare_runs` | Catalog tool |
| `metrics.query_instant` | Catalog tool |

## Related skills

- Other `harness` bundle skills - see bundle [USAGE.md](../USAGE.md)
