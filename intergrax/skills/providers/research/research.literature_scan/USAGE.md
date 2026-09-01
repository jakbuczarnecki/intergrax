# `research.literature_scan`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Literature scan baseline** for `ResearchAgent`: indexed retrieval plus web search for complementary sources. The original research domain skill - used by `research_application` and lab hosts.

## How it works

1. Resolves `rag.retrieve` + `websearch.query`.
2. Prompt ref: `research.literature_scan.system`.
3. Agent UAEP pipeline calls tools via `ToolRuntime`; skill sets allow-list at register.
4. Pairs with `research.web_evidence` for deeper URL fetch without replacing this pack.

## How to use

```python
from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN

# agents/research/research_agent.py
AgentContract(id="research", skills=[RESEARCH_LITERATURE_SCAN], ...)
```

Enable `research` bundle via `research_skill_profile()` or `lab_skill_profile()`.

## What you get

| Benefit | Detail |
|---------|--------|
| **Research SKU default** | Standard tool pair for literature tasks |
| **Hybrid evidence** | Corpus + web in one declaration |
| **Problem Radar ready** | Same pattern for scan-style agents |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Literature index retrieval |
| `websearch.query` | Supplementary web sources |

## Related skills

- `research.web_evidence` - URL read + batch fetch
- `research.citation_synthesis` - report generation
- `rag.hybrid_qa` - index Q&A with memory read
