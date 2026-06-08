# `agent.capability_mapper`

**Bundle:** `agent` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Map agent contracts to catalog tools and skill packs.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `agent` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `agent.get_contract`, `skill.resolve`, `catalog.describe_tool`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `agent.get_contract` | Catalog tool |
| `skill.resolve` | Catalog tool |
| `catalog.describe_tool` | Catalog tool |

## Related skills

- Other `agent` bundle skills — see bundle [USAGE.md](../USAGE.md)
