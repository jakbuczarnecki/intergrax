# `knowledge.openai_strict`

**Bundle:** `knowledge` · **Version:** 1.0.0 · **Risk:** `medium` · **Bundle status:** BETA

## Purpose

**Strict grounded Q&A via OpenAI hosted vector store** (`file_search`) - vendor-managed retrieval separate from harness `rag.retrieve`. Use when Tier-3 hosts target OpenAI Responses API with managed vector stores instead of self-hosted RAG stack.

## How it works

1. Resolves single tool `openai.file_search.query`.
2. Does **not** use harness `RetrievalService` / `vectorstore_manager` - OpenAI backend only.
3. Prompt ref: `knowledge.openai_strict.system` enforces citation-only answers.
4. Bundle marked STABLE - pair with integration OpenAI credentials on host.

## How to use

```python
from intergrax.skills.providers.knowledge.manifests import KNOWLEDGE_OPENAI_STRICT
from intergrax.applications._shared.skill_wiring import knowledge_skill_profile

env.skill_profile = knowledge_skill_profile()
AgentContract(id="openai_rag_bot", skills=[KNOWLEDGE_OPENAI_STRICT], ...)
```

Wire OpenAI LLM adapter + vector store tools on `tool_profile`; enable `openai.*` tool bundle.

## What you get

| Benefit | Detail |
|---------|--------|
| **Vendor-hosted RAG** | No self-managed chunk pipeline |
| **Strict grounding mode** | Prompt tuned for file_search citations |
| **Parallel to harness RAG** | Choose OpenAI vs `rag.retrieve` per host |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `openai.file_search.query` | Query OpenAI managed vector store |

## Related skills

- `knowledge.wiki_navigator` - internal wiki path (harness tools)
- `rag.hybrid_qa` - self-hosted index Q&A
