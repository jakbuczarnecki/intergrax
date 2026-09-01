# `legal.redline_draft`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Contract redline drafting with retrieval and workspace IO.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `legal` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `rag.retrieve`, `workspace.read_file`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Catalog tool |
| `workspace.read_file` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `legal` bundle skills - see bundle [USAGE.md](../USAGE.md)
