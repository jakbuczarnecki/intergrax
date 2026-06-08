# `gitlab.issue_creator`

**Bundle:** `gitlab` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

GitLab issue creation with dedup search and stakeholder notification.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `gitlab_skill_profile()`; Enable bundle `gitlab` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `gitlab.create_issue`, `issues.search`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `gitlab.create_issue` | Catalog tool |
| `issues.search` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

-
 
`
g
i
t
l
a
b
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
