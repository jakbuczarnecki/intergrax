# `health.identity_probe`

**Bundle:** `health` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Extended health sweep: identity, cache, notify, wiki backends.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `health` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `health.check_identity_provider`, `health.check_key_value_cache`, `health.check_notification_channel`, `health.check_wiki_knowledge`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `health.check_identity_provider` | Catalog tool |
| `health.check_key_value_cache` | Catalog tool |
| `health.check_notification_channel` | Catalog tool |
| `health.check_wiki_knowledge` | Catalog tool |

## Related skills

- Other `health` bundle skills — see bundle [USAGE.md](../USAGE.md)
