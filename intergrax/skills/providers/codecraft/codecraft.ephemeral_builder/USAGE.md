# `codecraft.ephemeral_builder`

**Bundle:** `codecraft` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Harness ephemeral Code Craft loop: start an isolated builder session, iterate generated code, promote vetted artifacts, and dispose ephemeral tools safely.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy and Code Craft governance fragments.

## How to use

Enable bundle `codecraft` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to the ephemeral builder lifecycle plus workspace I/O for promoted artifacts.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `codecraft.start` | Catalog tool |
| `codecraft.iterate` | Catalog tool |
| `codecraft.get_state` | Catalog tool |
| `codecraft.promote` | Catalog tool |
| `codecraft.dispose` | Catalog tool |
| `codecraft.list_ephemeral_tools` | Catalog tool |
| `workspace.read_file` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `codecraft` bundle skills - see bundle [USAGE.md](../USAGE.md)
