# `harness.run_exporter`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Harness run export: bundle export with events and run metadata.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `harness` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `harness.export_run_bundle`, `harness.get_run_events`, `harness.get_run`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `harness.export_run_bundle` | Catalog tool |
| `harness.get_run_events` | Catalog tool |
| `harness.get_run` | Catalog tool |

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
