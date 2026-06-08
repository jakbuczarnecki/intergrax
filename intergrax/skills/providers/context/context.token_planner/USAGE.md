# `context.token_planner`

**Bundle:** `context` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Context budget planning before LLM assembly.

## How it works

context.estimate_tokens + context.summarize with memory.read fallback.

## How to use

context_skill_profile(); pair with ContextProfile.budget on host.

## What you get

Proactive trimming instead of hard failures.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `context.estimate_tokens` | Estimate token count |
| `context.summarize` | Summarize overflow text |
| `memory.read` | Read session context |

## Related skills

- `memory.task_scratchpad`
- `rag.hybrid_qa`
