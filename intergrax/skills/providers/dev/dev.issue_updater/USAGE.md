# `dev.issue_updater`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Update existing tracker issues from agent remediation loops.

## How it works

issues.update_issue + add_comment + get_issue.

## How to use

ops_skill_profile(); skills=[DEV_ISSUE_UPDATER].

## What you get

Close-the-loop updates complementing issue_creator.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.update_issue` | Update issue fields |
| `issues.add_comment` | Add comment |
| `issues.get_issue` | Fetch issue details |

## Related skills

- `dev.issue_creator`
- `dev.issue_triage`
