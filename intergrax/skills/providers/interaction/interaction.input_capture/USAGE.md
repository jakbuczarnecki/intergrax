# `interaction.input_capture`

**Bundle:** `interaction` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Capture last user input, post reply, and persist to task memory.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `interaction_skill_profile()`; Enable bundle `interaction` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `interaction.get_last_input`, `interaction.post_reply`, `memory.write`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `interaction.get_last_input` | Catalog tool |
| `interaction.post_reply` | Catalog tool |
| `memory.write` | Catalog tool |

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
