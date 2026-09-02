# `legal.case_research`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

**Case and regulatory research** for dispute simulation and legal analysts: RAG over case materials, internal wiki/knowledge search, and open-web evidence. Use in DSW `dispute_analyst` / `dispute_strategist` pipelines.

## How it works

1. Unions `rag.retrieve`, `knowledge.search`, `websearch.query`.
2. `knowledge.search` uses `WikiKnowledge` binding - Confluence/internal docs.
3. No `requires_skills` - can be used standalone or with `legal.contract_review`.
4. Prompt ref: `legal.case_research.system`.

## How to use

```python
from intergrax.skills.providers.legal.manifests import LEGAL_CASE_RESEARCH
from intergrax.applications._shared.skill_wiring import dispute_skill_profile

env.skill_profile = dispute_skill_profile()
AgentContract(id="dispute_analyst", skills=[LEGAL_CASE_RESEARCH], ...)
```

Wire `wiki_knowledge` or Confluence integration + RAG index for case files.

## What you get

| Benefit | Detail |
|---------|--------|
| **Three-source research** | Index + wiki + web in one pack |
| **DSW-ready** | Matches dispute sim agent roster |
| **Independent of contract review** | No requires_skills dependency |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Case file / chronology retrieval |
| `knowledge.search` | Internal wiki search |
| `websearch.query` | Public regulatory / precedent search |

## Related skills

- `knowledge.wiki_navigator` - deeper wiki navigation
- `research.web_evidence` - heavier web fetch without RAG
- `memory.task_scratchpad` - persist research notes across dispute steps
