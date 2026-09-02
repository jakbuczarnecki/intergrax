# `eval.regression_guard`

**Bundle:** `eval` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Regression guard: compare releases, summarize, and alert.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `eval` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `eval.compare_releases`, `eval.summarize_release`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `eval.compare_releases` | Catalog tool |
| `eval.summarize_release` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `eval` bundle skills - see bundle [USAGE.md](../USAGE.md)
