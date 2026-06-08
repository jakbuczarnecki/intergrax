# `storage.object_lifecycle`

**Bundle:** `storage` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Object storage lifecycle: exists check, presigned URLs, and delete.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `storage` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `storage.exists`, `storage.presigned_url`, `storage.delete`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `storage.exists` | Catalog tool |
| `storage.presigned_url` | Catalog tool |
| `storage.delete` | Catalog tool |

## Related skills

-
 
`
s
t
o
r
a
g
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
