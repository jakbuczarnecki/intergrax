# `platform.secret_admin`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Secret lifecycle admin: put, delete, and get runtime secrets.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `platform` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `platform.put_secret`, `platform.delete_secret`, `platform.get_secret`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `platform.put_secret` | Catalog tool |
| `platform.delete_secret` | Catalog tool |
| `platform.get_secret` | Catalog tool |

## Related skills

-
 
`
p
l
a
t
f
o
r
m
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
