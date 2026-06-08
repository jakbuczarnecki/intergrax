# `browser.interactive_run`

**Bundle:** `browser` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Interactive browser automation: run browser, fetch page, parse preview.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `browser` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `browser.run`, `browser.fetch_page`, `document.parse_preview`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `browser.run` | Catalog tool |
| `browser.fetch_page` | Catalog tool |
| `document.parse_preview` | Catalog tool |

## Related skills

-
 
`
b
r
o
w
s
e
r
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
