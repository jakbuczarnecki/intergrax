# `cache.key_admin`

**Bundle:** `cache` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Cache key administration: list, get, and delete session cache keys.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `cache` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `cache.list_keys`, `cache.get`, `cache.delete`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cache.list_keys` | Catalog tool |
| `cache.get` | Catalog tool |
| `cache.delete` | Catalog tool |

## Related skills

-
 
`
c
a
c
h
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
