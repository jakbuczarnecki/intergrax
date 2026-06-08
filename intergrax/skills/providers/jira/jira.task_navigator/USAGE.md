# `jira.task_navigator`

**Bundle:** `jira` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Jira task navigation: search tasks, fetch issues, and add comments.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Use `jira_skill_profile()`; Enable bundle `jira` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `jira.search_tasks`, `jira.get_issue`, `jira.add_comment`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `jira.search_tasks` | Catalog tool |
| `jira.get_issue` | Catalog tool |
| `jira.add_comment` | Catalog tool |

## Related skills

-
 
`
j
i
r
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
