# `platform.flag_rollout`

**Bundle:** `platform` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Feature-flag rollout with metrics probe and notify.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `platform` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `platform.evaluate_feature_flag`, `notify.send`, `metrics.query_instant`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `platform.evaluate_feature_flag` | Catalog tool |
| `notify.send` | Catalog tool |
| `metrics.query_instant` | Catalog tool |

## Related skills

- Other `platform` bundle skills - see bundle [USAGE.md](../USAGE.md)
