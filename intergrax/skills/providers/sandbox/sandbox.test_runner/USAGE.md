# `sandbox.test_runner`

**Bundle:** `sandbox` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Sandbox test execution with workspace input and error capture.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `sandbox` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `sandbox.exec`, `workspace.read_file`, `errors.capture`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `sandbox.exec` | Catalog tool |
| `workspace.read_file` | Catalog tool |
| `errors.capture` | Catalog tool |

## Related skills

- Other `sandbox` bundle skills — see bundle [USAGE.md](../USAGE.md)
