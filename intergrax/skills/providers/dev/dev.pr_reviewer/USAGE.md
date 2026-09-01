# `dev.pr_reviewer`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

PR/issue review with search, fetch, and mail notification.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `dev` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `issues.search`, `issues.get_issue`, `collaboration.send_mail`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.search` | Catalog tool |
| `issues.get_issue` | Catalog tool |
| `collaboration.send_mail` | Catalog tool |

## Related skills

- Other `dev` bundle skills - see bundle [USAGE.md](../USAGE.md)
