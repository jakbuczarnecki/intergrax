# `harness.run_comparator`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Harness run comparison: list runs, fetch details, and compare outcomes.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `harness` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `harness.list_runs`, `harness.get_run`, `harness.compare_runs`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `harness.list_runs` | Catalog tool |
| `harness.get_run` | Catalog tool |
| `harness.compare_runs` | Catalog tool |

## Related skills

-
 
`
h
a
r
n
e
s
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
