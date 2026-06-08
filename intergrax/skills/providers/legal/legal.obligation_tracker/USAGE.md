# `legal.obligation_tracker`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Track contractual obligations in task memory and workspace.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `legal` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `memory.write`, `memory.read`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `memory.write` | Catalog tool |
| `memory.read` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `legal` bundle skills — see bundle [USAGE.md](../USAGE.md)
