# `ops.log_tail`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Live log tailing with search and error capture for incident response.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `logs.tail`, `logs.search`, `errors.capture`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `logs.tail` | Catalog tool |
| `logs.search` | Catalog tool |
| `errors.capture` | Catalog tool |

## Related skills

-
 
`
o
p
s
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
