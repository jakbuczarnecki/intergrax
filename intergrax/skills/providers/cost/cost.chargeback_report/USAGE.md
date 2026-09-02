# `cost.chargeback_report`

**Bundle:** `cost` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Chargeback report from run budget, billing usage, and workspace export.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `cost` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `cost.get_run_budget`, `billing.list_usage`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cost.get_run_budget` | Catalog tool |
| `billing.list_usage` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `cost` bundle skills - see bundle [USAGE.md](../USAGE.md)
