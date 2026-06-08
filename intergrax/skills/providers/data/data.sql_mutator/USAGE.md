# `data.sql_mutator`

**Bundle:** `data` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

SQL mutation runner: execute statements with schema guard and query fallback.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `data` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `database.execute`, `database.describe_schema`, `database.query`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `database.execute` | Catalog tool |
| `database.describe_schema` | Catalog tool |
| `database.query` | Catalog tool |

## Related skills

-
 
`
d
a
t
a
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
