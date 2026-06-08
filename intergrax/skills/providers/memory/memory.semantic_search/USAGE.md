# `memory.semantic_search`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Semantic memory search across session memory and LTM index.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `memory` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `memory.search`, `memory.read`, `ltm.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `memory.search` | Catalog tool |
| `memory.read` | Catalog tool |
| `ltm.search` | Catalog tool |

## Related skills

-
 
`
m
e
m
o
r
y
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
