# `notify.batch_dispatch`

**Bundle:** `notify` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Batch notification dispatch with due scheduling and pending list.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `notify` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `notify.send_batch`, `notify.dispatch_due`, `notify.list_scheduled`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `notify.send_batch` | Catalog tool |
| `notify.dispatch_due` | Catalog tool |
| `notify.list_scheduled` | Catalog tool |

## Related skills

-
 
`
n
o
t
i
f
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
