# `agent.roster_introspect`

**Bundle:** `agent` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Agent roster introspection for hub and concierge agents.

## How it works

agent.list_agents + agent.get_contract + skill.resolve.

## How to use

agent_roster_skill_profile(); platform hub hosts.

## What you get

Self-describing harness without hardcoded agent lists.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `agent.list_agents` | List registered agents |
| `agent.get_contract` | Fetch agent contract |
| `skill.resolve` | Resolve skill pack |

## Related skills

- `platform.concierge`
- `harness.skill_registry`
