# `modality.audio_transcript`

**Bundle:** `modality` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Audio transcript pipeline with parse preview and workspace export.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `modality` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `speech.transcribe`, `document.parse_preview`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `speech.transcribe` | Catalog tool |
| `document.parse_preview` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `modality` bundle skills — see bundle [USAGE.md](../USAGE.md)
