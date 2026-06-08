# `platform.workflow_cancel`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

CI workflow cancellation: cancel run, fetch details, list runs.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `platform` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `platform.cancel_workflow_run`, `platform.get_workflow_run`, `platform.list_workflow_runs`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `platform.cancel_workflow_run` | Catalog tool |
| `platform.get_workflow_run` | Catalog tool |
| `platform.list_workflow_runs` | Catalog tool |

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
