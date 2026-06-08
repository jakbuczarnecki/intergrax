# `vector_store.purge`

**Bundle:** `vector_store` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Vector store purge: delete vectors with count and collection listing.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `vector_store` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `vector_store.delete`, `vector_store.count`, `vector_store.list_collections`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `vector_store.delete` | Catalog tool |
| `vector_store.count` | Catalog tool |
| `vector_store.list_collections` | Catalog tool |

## Related skills

-
 
`
v
e
c
t
o
r
_
s
t
o
r
e
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
