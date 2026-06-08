# `cache.session_cache`

**Bundle:** `cache` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

KV cache with task memory read for session acceleration.

## How it works

cache.* via KeyValueCache; memory.read as fallback.

## How to use

cache_skill_profile(); wire key_value_cache integration.

## What you get

Fewer duplicate tool calls within a session.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cache.get` | Read cache key |
| `cache.set` | Write cache key |
| `memory.read` | Session memory fallback |

## Related skills

- `memory.task_scratchpad`
- `rag.hybrid_qa`
