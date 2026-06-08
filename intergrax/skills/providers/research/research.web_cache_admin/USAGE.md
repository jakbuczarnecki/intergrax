# `research.web_cache_admin`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Web search cache admin: invalidate cache, query, and batch fetch.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `research` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `websearch.invalidate_cache`, `websearch.query`, `websearch.fetch_batch`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `websearch.invalidate_cache` | Catalog tool |
| `websearch.query` | Catalog tool |
| `websearch.fetch_batch` | Catalog tool |

## Related skills

-
 
`
r
e
s
e
a
r
c
h
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
