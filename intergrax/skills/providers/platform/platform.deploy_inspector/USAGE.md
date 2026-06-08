# `platform.deploy_inspector`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Deploy inspection: workflow runs, check suites, and logs.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `platform` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `platform.list_workflow_runs`, `platform.list_check_suites`, `logs.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `platform.list_workflow_runs` | Catalog tool |
| `platform.list_check_suites` | Catalog tool |
| `logs.search` | Catalog tool |

## Related skills

- Other `platform` bundle skills — see bundle [USAGE.md](../USAGE.md)
