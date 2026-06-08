# `memory.ltm_curator`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Long-term memory fact curation across sessions.

## How it works

ltm.write_fact + ltm.search via LTM store binding.

## How to use

memory_skill_profile(); enable LTM on MemoryProfile.

## What you get

Durable facts without ad-hoc memory tool lists.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `ltm.write_fact` | Persist durable fact |
| `ltm.search` | Search LTM index |
| `memory.read` | Read session context |

## Related skills

- `memory.task_scratchpad`
- `memory.session_cleanup`
