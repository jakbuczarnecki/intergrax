# `ops.postmortem_writer`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Postmortem drafting from harness run metadata and logs.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `harness.get_run`, `logs.search`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `harness.get_run` | Catalog tool |
| `logs.search` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `ops` bundle skills — see bundle [USAGE.md](../USAGE.md)
