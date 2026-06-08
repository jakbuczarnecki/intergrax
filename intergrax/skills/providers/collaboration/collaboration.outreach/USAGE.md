# `collaboration.outreach`

**Bundle:** `collaboration` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Email outreach and thread context** via collaboration suite integrations (Microsoft 365, Google Workspace): list messages, read threads, send mail. Use for agents that draft follow-ups, summarize inbox context, or send notifications — not for bulk spam automation.

## How it works

1. Tools bind to `CollaborationSuite` on `ToolWiringContext`.
2. `collaboration.list_messages` / `get_message` provide thread context for the LLM.
3. `collaboration.send_mail` sends outbound mail (policy + HITL may apply on host).
4. Atomic per-tool invocation; skill only merges allow-list.

## How to use

```python
from intergrax.skills.providers.collaboration.manifests import COLLABORATION_OUTREACH

AgentContract(id="outreach_agent", skills=[COLLABORATION_OUTREACH], ...)
```

Wire `collaboration_suite` integration slug; enable collaboration tools on `tool_profile`.

## What you get

| Benefit | Detail |
|---------|--------|
| **Inbox-aware drafts** | Read before send |
| **Suite-agnostic** | Same skill across M365/GWS backends |
| **Policy-governed send** | Through `ToolRuntime`, not raw SMTP in agent |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `collaboration.send_mail` | Send email message |
| `collaboration.list_messages` | List inbox/thread messages |
| `collaboration.get_message` | Read single message body |

## Related skills

- `dev.issue_triage` — tracker comment after email thread
- `notify.send` — lightweight alerts without full collaboration suite
