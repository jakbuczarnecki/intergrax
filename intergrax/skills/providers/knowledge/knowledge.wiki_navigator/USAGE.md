# `knowledge.wiki_navigator`

**Bundle:** `knowledge` · **Version:** 1.0.0 · **Risk:** `medium` · **Bundle status:** STABLE

## Purpose

**Internal documentation navigation**: search wiki/knowledge base, fetch full pages, and search Confluence. Use for enterprise agents answering from runbooks, architecture docs, and team wikis - not for open-web research.

## How it works

1. `knowledge.search` / `knowledge.get_page` use provider-agnostic `WikiKnowledge` binding.
2. `confluence.search` uses Confluence-specific wiki backend when that slug is wired.
3. Prompt ref: `knowledge.wiki_navigator.system`.
4. Distinct from `rag.retrieve` - optimized for structured wiki pages, not chunked legal corpora.

## How to use

```python
from intergrax.skills.providers.knowledge.manifests import KNOWLEDGE_WIKI_NAVIGATOR
from intergrax.applications._shared.skill_wiring import lkw_skill_profile

AgentContract(id="doc_bot", skills=[KNOWLEDGE_WIKI_NAVIGATOR], ...)
```

Wire `wiki_knowledge` / Confluence on `IntegrationProfile`; enable knowledge + confluence tools.

## What you get

| Benefit | Detail |
|---------|--------|
| **Runbook Q&A** | Search + full page read |
| **Confluence-ready** | Dedicated search tool alias |
| **Complements RAG** | Wiki pages vs vector chunks |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `knowledge.search` | Search internal knowledge base |
| `knowledge.get_page` | Fetch wiki page content |
| `confluence.search` | Confluence-specific search |

## Related skills

- `legal.case_research` - includes `knowledge.search` for cases
- `harness.integration_bridge_smoke` - smoke test for knowledge.search
- `rag.hybrid_qa` - vector index alternative
