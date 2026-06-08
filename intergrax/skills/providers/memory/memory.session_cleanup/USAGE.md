# `memory.session_cleanup`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Session memory hygiene: list, read, delete stale keys.

## How it works

memory.delete_key is destructive; list/read support safe purge.

## How to use

memory_skill_profile(); task KV enabled on MemoryProfile.

## What you get

Prevents unbounded task memory in long sessions.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `memory.list_keys` | Enumerate keys |
| `memory.delete_key` | Delete record |
| `memory.read` | Read before delete |

## Related skills

- `memory.task_scratchpad`
- `cache.session_cache`
