# `knowledge.confluence_navigator`

**Bundle:** `knowledge` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Confluence deep navigation: get page, search pages, and cross-search.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `knowledge` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `confluence.get_page`, `confluence.search_pages`, `confluence.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `confluence.get_page` | Catalog tool |
| `confluence.search_pages` | Catalog tool |
| `confluence.search` | Catalog tool |

## Related skills

-
 
`
k
n
o
w
l
e
d
g
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
