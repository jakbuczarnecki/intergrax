# Context Engineering

**Status:** Canonical architecture (decomposed from platform canon)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../IDEAL_HARNESS_AI_ARCHITECTURE.md)

---

# 28. Context Management

Context is expensive and dangerous when uncontrolled.

Nexus must control what context each agent receives.

Rules:

- pass only relevant context
- avoid dumping entire history into agents
- use summaries when needed
- separate task context from user memory
- separate evidence from interpretation
- preserve provenance

## 28.1 Context Engineering (Harness Terminology)

> **Deep dive:** [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) [§7](PLATFORM_FOUNDATION.md)–[§9](ORCHESTRATION.md) (read path, compression ladder, strategy selection) · audit map [§16](AGENT_CONTRACTS_AND_ASSEMBLY.md)

**Context engineering** is the deliberate design of what enters each LLM call: bounded memory reads, summary tiers, evidence vs interpretation, and provenance. In Intergrax this is implemented by Tier-1 — not by ad-hoc prompt concatenation in Tier-2 agents.

### Mechanisms

| Mechanism | Module | Role |
|-----------|--------|------|
| Per-agent context assembly | `ContextManager.build_agent_context()` | Applies `TaskContextAssemblyOptions` |
| Summary tiers | `FULL` / `SUMMARY_ONLY` / `STRUCTURED_ONLY` / `MINIMAL` | Limits prior task noise |
| Memory stores | Session, user LTM, task KV, shared handoff | See §27; access via `MemoryView` only |
| Context injection tools | `rag.retrieve`, `websearch.query` (`injects_context: true`) | Evidence into prompt via tool path |
| **Context budget** | `ContextBudgetPolicy` | Central `max_chars` / trim with `CONTEXT_TRIMMED` events |

### Rules

- Agents MUST NOT assemble unbounded chat history for LLM calls.
- Every trim or tier downgrade MUST be traceable (provenance in `AgentContextBundle`).
- Task context, user memory, and tool-retrieved evidence MUST remain logically separated in the bundle.

Agent workflow reference: [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) Appendix G · full memory canon: [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md).

---

