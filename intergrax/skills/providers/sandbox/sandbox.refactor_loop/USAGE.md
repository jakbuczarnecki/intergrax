# `sandbox.refactor_loop`

**Bundle:** `sandbox` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Iterative refactor: exec, write, and workspace search.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `sandbox` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `sandbox.exec`, `workspace.write_file`, `workspace.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `sandbox.exec` | Catalog tool |
| `workspace.write_file` | Catalog tool |
| `workspace.search` | Catalog tool |

## Related skills

- Other `sandbox` bundle skills - see bundle [USAGE.md](../USAGE.md)
