# `memory.task_scratchpad`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Task-scoped key-value scratchpad** for multi-step agents: read, write, and list keys in the current task namespace. Use in dispute simulation pipelines, long chat turns (IAA), and any agent that must remember intermediate results without custom memory wiring.

## How it works

1. Resolves `memory.read`, `memory.write`, `memory.list_keys`.
2. Tools bind to `PolicyScopedMemoryView` on `ToolWiringContext` (MEM platform wiring).
3. Namespace isolation follows Nexus task_id / delegation rules.
4. Skill does not store data itself - only declares which memory tools the agent may call.

## How to use

```python
from intergrax.skills.providers.memory.manifests import MEMORY_TASK_SCRATCHPAD
from intergrax.applications._shared.skill_wiring import dispute_skill_profile, platform_skill_profile

AgentContract(id="dispute_analyst", skills=[MEMORY_TASK_SCRATCHPAD], ...)
```

Requires `MemoryProfile` with task KV enabled on `ApplicationEnvironmentProfile`.

## What you get

| Benefit | Detail |
|---------|--------|
| **Cross-step continuity** | Analyst → strategist handoff via shared task keys |
| **Low risk tier** | Safe default for lab and staging agents |
| **Policy-scoped writes** | Memory middleware enforces namespace rules |
| **No agent-local stores** | Uses platform MEM layer only |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `memory.read` | Read a key from task memory |
| `memory.write` | Write or merge a task memory record |
| `memory.list_keys` | Enumerate keys in the task namespace |

## Related skills

- `rag.hybrid_qa` - read memory during Q&A
- `workspace.authoring` - write draft pointers to memory
- `platform.concierge` - hub chat with session recall
