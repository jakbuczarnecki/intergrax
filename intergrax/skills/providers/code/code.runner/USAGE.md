# `code.runner`

**Bundle:** `code` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Controlled code execution: script run, code exec, and sandbox operation listing.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `code_skill_profile()`; Enable bundle `code` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `code.exec`, `script.run`, `sandbox.list_operations`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `code.exec` | Catalog tool |
| `script.run` | Catalog tool |
| `sandbox.list_operations` | Catalog tool |

## Related skills

-
 
`
c
o
d
e
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
