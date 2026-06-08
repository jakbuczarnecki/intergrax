# `research.report_compiler`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Compile citation-backed reports from retrieval and web evidence.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `research` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `rag.retrieve`, `websearch.query`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Catalog tool |
| `websearch.query` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `research` bundle skills — see bundle [USAGE.md](../USAGE.md)
