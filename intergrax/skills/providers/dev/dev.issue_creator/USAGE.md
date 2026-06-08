# `dev.issue_creator`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Create tracker issues from agent findings with dedup notify.

## How it works

issues.create_issue + search; notify on create.

## How to use

ops_skill_profile(); skills=[DEV_ISSUE_CREATOR].

## What you get

Discovery-to-ticket loop for automation agents.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.create_issue` | Create issue |
| `issues.search` | Dedup search |
| `notify.send` | Notify assignee |

## Related skills

- `dev.issue_triage`
- `platform.cicd_inspector`
