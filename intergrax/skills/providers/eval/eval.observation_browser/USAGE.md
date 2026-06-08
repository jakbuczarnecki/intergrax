# `eval.observation_browser`

**Bundle:** `eval` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Eval observation browser: list observations, record new, and correlate traces.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `eval` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `eval.list_observations`, `eval.record_observation`, `observability.query_traces`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `eval.list_observations` | Catalog tool |
| `eval.record_observation` | Catalog tool |
| `observability.query_traces` | Catalog tool |

## Related skills

-
 
`
e
v
a
l
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
