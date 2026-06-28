# `local.workspace.synthesize`

**Bundle:** `local` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Synthesize reports and drafts from retrieved evidence; write artifacts only to the shadow workspace.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy. Invokes `workspace.write_file` against the shadow workspace bound on `ToolWiringContext`.

## How to use

Enable bundle `local` on `SkillProfile` or attach this manifest to `AgentContract.skills` for `LocalSynthesizerAgent`. Requires shadow workspace wiring in the host environment profile.

## What you get

Governed access to: `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workspace.write_file` | Create or overwrite draft files in shadow workspace |

## Related skills

- `local.workspace.search` — source evidence for synthesis
- `workspace.authoring` — broader shadow workspace drafting pack
- Other `local` bundle skills — see bundle [USAGE.md](../USAGE.md)
