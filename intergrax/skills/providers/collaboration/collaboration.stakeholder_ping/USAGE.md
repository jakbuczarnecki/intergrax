# `collaboration.stakeholder_ping`

**Bundle:** `collaboration` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Stakeholder outreach with CRM context, mail, and notify.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `collaboration` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `crm.get_account`, `collaboration.send_mail`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `crm.get_account` | Catalog tool |
| `collaboration.send_mail` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `collaboration` bundle skills — see bundle [USAGE.md](../USAGE.md)
