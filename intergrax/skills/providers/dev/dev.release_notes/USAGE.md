# `dev.release_notes`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Release notes from issue search and workspace export.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `dev` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `issues.search`, `workspace.write_file`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.search` | Catalog tool |
| `workspace.write_file` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `dev` bundle skills — see bundle [USAGE.md](../USAGE.md)
