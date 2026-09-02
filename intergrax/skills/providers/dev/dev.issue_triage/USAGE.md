# `dev.issue_triage`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Issue tracker triage** across Jira/GitLab/Linear via provider-agnostic `issues.*` tools: search backlog, read issue details, add comments, notify assignees. Use for developer-assistant agents that reduce context switching - not for full project management workflows.

## How it works

1. `issues.search` builds queries against configured `IssueTracker` integration.
2. `issues.get_issue` fetches full issue payload.
3. `issues.add_comment` posts triage notes (policy-checked).
4. `notify.send` pings assignee channel when needed.
5. Vendor-specific `jira.*` / `gitlab.*` tools remain available separately; this skill uses the generic surface.

## How to use

```python
from intergrax.skills.providers.dev.manifests import DEV_ISSUE_TRIAGE
from intergrax.applications._shared.skill_wiring import ops_skill_profile

AgentContract(id="triage_bot", skills=[DEV_ISSUE_TRIAGE], ...)
```

Wire `issue_tracker` integration slug on host.

## What you get

| Benefit | Detail |
|---------|--------|
| **Tracker-agnostic** | Same skill for Jira or GitLab hosts |
| **Comment + notify loop** | Close the triage feedback cycle |
| **Atomic tools** | LLM picks one operation per turn |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.search` | Search issues by filters |
| `issues.get_issue` | Fetch issue by key/id |
| `issues.add_comment` | Add triage comment |
| `notify.send` | Notify assignee or channel |

## Related skills

- `collaboration.outreach` - email-side follow-up
- `ops.trace_debug` - link incidents to tracker issues
