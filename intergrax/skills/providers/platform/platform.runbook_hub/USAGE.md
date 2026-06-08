# `platform.runbook_hub`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Platform hub: skill resolve, agent roster, and retrieval.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `platform` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `skill.resolve`, `agent.list_agents`, `rag.retrieve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `skill.resolve` | Catalog tool |
| `agent.list_agents` | Catalog tool |
| `rag.retrieve` | Catalog tool |

## Related skills

- Other `platform` bundle skills — see bundle [USAGE.md](../USAGE.md)
