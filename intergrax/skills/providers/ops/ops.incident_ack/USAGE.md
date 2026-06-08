# `ops.incident_ack`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

PagerDuty incident acknowledge with trigger and notify escalation path.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or list this manifest on `AgentContract.skills`.

## What you get

Governed access to: `pagerduty.acknowledge_incident`, `pagerduty.trigger_incident`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `pagerduty.acknowledge_incident` | Catalog tool |
| `pagerduty.trigger_incident` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

-
 
`
o
p
s
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
