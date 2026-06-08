# `health.full_stack_probe`

**Bundle:** `health` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Full-stack health probe: graph store, message bus, object storage, search provider.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `health` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `health.check_graph_store`, `health.check_message_bus`, `health.check_object_storage`, `health.check_search_provider`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `health.check_graph_store` | Catalog tool |
| `health.check_message_bus` | Catalog tool |
| `health.check_object_storage` | Catalog tool |
| `health.check_search_provider` | Catalog tool |

## Related skills

-
 
`
h
e
a
l
t
h
.
*
`
 
p
e
e
r
s
 
i
n
 
s
a
m
e
 
b
u
n
d
l
e
