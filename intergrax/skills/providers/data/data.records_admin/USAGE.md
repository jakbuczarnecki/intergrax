# `data.records_admin`

**Bundle:** `data` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Records store admin: put, delete, and count documents.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `data` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `records.put`, `records.delete`, `records.count`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `records.put` | Catalog tool |
| `records.delete` | Catalog tool |
| `records.count` | Catalog tool |

## Related skills

-
 
`
d
a
t
a
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
