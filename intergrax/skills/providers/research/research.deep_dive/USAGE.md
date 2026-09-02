# `research.deep_dive`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Deep web research with batch fetch and report workspace export.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `research` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `websearch.fetch_batch`, `websearch.read_url`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `websearch.fetch_batch` | Catalog tool |
| `websearch.read_url` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `research` bundle skills - see bundle [USAGE.md](../USAGE.md)
