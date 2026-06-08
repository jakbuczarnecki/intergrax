# `research.source_validator`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Validate sources against index and parse previews.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `research` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `websearch.query`, `rag.retrieve`, `document.parse_preview`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `websearch.query` | Catalog tool |
| `rag.retrieve` | Catalog tool |
| `document.parse_preview` | Catalog tool |

## Related skills

- Other `research` bundle skills — see bundle [USAGE.md](../USAGE.md)
