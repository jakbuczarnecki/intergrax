# `filesystem.local_io`

**Bundle:** `filesystem` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Local filesystem IO: read/write text, glob paths, and list directories.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `filesystem_skill_profile()`; Enable bundle `filesystem` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `filesystem.read_text`, `filesystem.write_text`, `filesystem.glob`, `filesystem.list`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `filesystem.read_text` | Catalog tool |
| `filesystem.write_text` | Catalog tool |
| `filesystem.glob` | Catalog tool |
| `filesystem.list` | Catalog tool |

## Related skills

-
 
`
f
i
l
e
s
y
s
t
e
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
