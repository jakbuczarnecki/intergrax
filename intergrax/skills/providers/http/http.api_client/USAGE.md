# `http.api_client`

**Bundle:** `http` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

HTTP API client: outbound requests with error capture and log correlation.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `http_skill_profile()`; Enable bundle `http` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `http.request`, `errors.capture`, `logs.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `http.request` | Catalog tool |
| `errors.capture` | Catalog tool |
| `logs.search` | Catalog tool |

## Related skills

-
 
`
h
t
t
p
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
