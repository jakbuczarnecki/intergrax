# `cloud_platform.resolver`

**Bundle:** `cloud_platform` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Cloud platform resolution: health probe, endpoint resolve, and integration check.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `cloud_platform_skill_profile()`; Enable bundle `cloud_platform` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `cloud_platform.health`, `cloud_platform.resolve`, `health.check_integration`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cloud_platform.health` | Catalog tool |
| `cloud_platform.resolve` | Catalog tool |
| `health.check_integration` | Catalog tool |

## Related skills

-
 
`
c
l
o
u
d
_
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
