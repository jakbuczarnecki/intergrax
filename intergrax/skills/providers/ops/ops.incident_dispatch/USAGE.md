# `ops.incident_dispatch`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**On-call incident workflow**: gather log context, trigger PagerDuty incident, notify stakeholders. Use when an agent or harness workflow must escalate failures to human operators — not for routine chat.

## How it works

1. `logs.search` collects context around failure window.
2. `pagerduty.trigger_incident` opens or triggers on-call routing via `NotificationChannel` binding.
3. `notify.send` dispatches supplementary alerts (Slack, email, etc.).
4. High `risk_tier` — restrict to trusted agents and production-gated hosts.

## How to use

```python
from intergrax.skills.providers.ops.manifests import OPS_INCIDENT_DISPATCH

AgentContract(id="reliability_bot", skills=[OPS_INCIDENT_DISPATCH], risk_level=AgentRiskLevel.HIGH, ...)
```

Wire `notification_channel` with `pagerduty` slug + log backend on integration profile.

## What you get

| Benefit | Detail |
|---------|--------|
| **End-to-end escalation pack** | Context + page + notify in one skill |
| **Provider-agnostic notify** | PagerDuty via integration swap |
| **Governance** | High risk tier flags policy review |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `pagerduty.trigger_incident` | Open on-call incident |
| `notify.send` | Secondary notification channel |
| `logs.search` | Failure context for incident body |

## Related skills

- `ops.trace_debug` — investigate before dispatch
- `ops.security_audit` — security-specific escalation path
