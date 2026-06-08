# `storage.artifact_sync`

**Bundle:** `storage` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Object storage sync with shadow workspace import/export.

## How it works

storage.* + workspace import/export via integrations.

## How to use

Wire object_storage slug; enable storage + workspace tools.

## What you get

Durable artifacts across runs without agent-local IO.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `storage.get` | Fetch object |
| `storage.put` | Upload object |
| `workspace.export_artifact` | Export to storage |
| `workspace.import_artifact` | Import from storage |

## Related skills

- `workspace.authoring`
- `research.citation_synthesis`
