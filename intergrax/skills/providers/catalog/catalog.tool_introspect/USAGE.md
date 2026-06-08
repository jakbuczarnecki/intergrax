# `catalog.tool_introspect`

**Bundle:** `catalog` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Tool catalog introspection: list tools, describe contracts, resolve skills.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `catalog_skill_profile()`; Enable bundle `catalog` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `catalog.list_tools`, `catalog.describe_tool`, `skill.resolve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `catalog.list_tools` | Catalog tool |
| `catalog.describe_tool` | Catalog tool |
| `skill.resolve` | Catalog tool |

## Related skills

-
 
`
c
a
t
a
l
o
g
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
