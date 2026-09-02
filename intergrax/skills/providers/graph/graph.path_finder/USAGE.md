# `graph.path_finder`

**Bundle:** `graph` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Graph path exploration with node fetch and session memory.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `graph` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `graph.run_query`, `graph.get_node`, `memory.read`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `graph.run_query` | Catalog tool |
| `graph.get_node` | Catalog tool |
| `memory.read` | Catalog tool |

## Related skills

- Other `graph` bundle skills - see bundle [USAGE.md](../USAGE.md)
