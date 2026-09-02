# `rag.hybrid_qa`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Default **question-answering over an indexed knowledge base** with optional session context. Use when an agent must answer from RAG chunks, fetch the full source document, and read prior task memory - without hand-picking three separate tools on every contract.

Typical hosts: LKW (`local_search`), Legal, Research, Intergrax Assistant, any Tier-3 profile with a vector index.

## How it works

1. **Registration:** `RagSkillPlugin` registers the manifest in `SkillCatalog` → `SkillRegistry` when bundle `rag` is enabled on `SkillProfile`.
2. **Resolution:** At `AgentRegistry.register`, `SkillResolver` unions `tool_ids` into `AgentContract.allowed_tools`.
3. **Runtime:** The LLM invokes atomic catalog tools only (`rag.retrieve`, `rag.get_document`, `memory.read`) through `ToolRuntime` + policy - the skill is never called as a function.
4. **Prompt ref:** `rag.hybrid_qa.system` is declared for Prompt Registry / future SK-BRIDGE; wire explicitly in agent steps until automatic merge ships.

## How to use

### Enable on Tier-3 host

```python
from intergrax.skills.registry.profile import SkillProfile
from intergrax.applications._shared.skill_wiring import rag_skill_profile, lkw_skill_profile

profile = rag_skill_profile()  # enabled_bundles=["rag"]
# or lkw_skill_profile() which includes rag + workspace + memory + knowledge
```

Set `ApplicationEnvironmentProfile.skill_profile = profile`. Enable the three required tools on host `tool_profile`.

### Declare on agent contract

```python
from intergrax.skills.providers.rag.manifests import RAG_HYBRID_QA

AgentContract(
    id="my_agent",
    skills=[RAG_HYBRID_QA],
    extra_tools=[],
)
```

### Minimal wiring test

```python
from intergrax.skills.registry import build_registry_from_profile, SkillProfile
from intergrax.skills.resolver import SkillResolver

registry = build_registry_from_profile(SkillProfile(enabled_bundles=["rag"]))
pack = SkillResolver(registry).resolve(["rag.hybrid_qa"])
assert "rag.retrieve" in pack.tool_ids
```

## What you get

| Benefit | Detail |
|---------|--------|
| **Reusable allow-list** | Same three tools on every Q&A agent - no copy-paste |
| **Conformance** | `EnvironmentSkillToolConsistencyCheck` validates roster ⊆ environment |
| **Traceability** | `SKILL_RESOLVED` event lists merged `tool_ids` |
| **Composable** | Merge with other skills; `requires_skills` not used on this pack |

## Tools unlocked

| `tool_id` | Role in this skill |
|-----------|-------------------|
| `rag.retrieve` | Hybrid retrieval over configured vector index |
| `rag.get_document` | Fetch full indexed document text/metadata by id |
| `memory.read` | Read task-scoped memory for follow-up context |

## Integrations required

Wire via Tier-3 `IntegrationProfile`: `vector_store`, `embedding_provider` (and optional rerank) so `rag.retrieve` resolves. Session memory requires `MemoryProfile` with task KV enabled.

## Related skills

- `rag.document_ingest` - populate the index before Q&A
- `memory.task_scratchpad` - write-side task memory
- `research.literature_scan` - adds web search to retrieval
