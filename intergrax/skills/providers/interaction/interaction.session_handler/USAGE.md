# `interaction.session_handler`

**Bundle:** `interaction` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

User session handling: list sessions, read history, and post replies.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `interaction_skill_profile()`; Enable bundle `interaction` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `interaction.list_sessions`, `interaction.get_session_history`, `interaction.post_reply`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `interaction.list_sessions` | Catalog tool |
| `interaction.get_session_history` | Catalog tool |
| `interaction.post_reply` | Catalog tool |

## Related skills

-
 
`
i
n
t
e
r
a
c
t
i
o
n
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
