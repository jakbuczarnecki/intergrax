# `eval.baseline_runner`

**Bundle:** `eval` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Baseline eval recording with Braintrust and run listing.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `eval` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `eval.record_observation`, `braintrust.log_eval`, `harness.list_runs`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `eval.record_observation` | Catalog tool |
| `braintrust.log_eval` | Catalog tool |
| `harness.list_runs` | Catalog tool |

## Related skills

- Other `eval` bundle skills — see bundle [USAGE.md](../USAGE.md)
