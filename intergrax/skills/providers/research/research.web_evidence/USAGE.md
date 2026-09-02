# `research.web_evidence`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Web-grounded evidence collection**: search, read individual URLs, and batch-fetch multiple pages. Use when research agents need full page text beyond search snippets - complements `literature_scan` without requiring RAG index hits.

## How it works

1. Resolves full `websearch` trio: `query`, `read_url`, `fetch_batch`.
2. `websearch_executor` or `SearchProvider` binding from Tier-3 integration profile.
3. Prompt ref: `research.web_evidence.system`.
4. No index dependency - works with web-only hosts.

## How to use

```python
from intergrax.skills.providers.research.manifests import RESEARCH_WEB_EVIDENCE

AgentContract(
    id="research",
    skills=[RESEARCH_LITERATURE_SCAN, RESEARCH_WEB_EVIDENCE],
)
```

Wire `search_provider` integration; enable websearch tools on host `ToolProfile`.

## What you get

| Benefit | Detail |
|---------|--------|
| **Deep web capture** | Beyond query snippets |
| **Batch efficiency** | `fetch_batch` for literature lists |
| **Composable** | Add alongside literature_scan without overlap conflict |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `websearch.query` | Web search snippets |
| `websearch.read_url` | Single URL full text |
| `websearch.fetch_batch` | Multi-URL fetch |

## Related skills

- `browser.research_fetch` - JS-rendered pages
- `research.citation_synthesis` - synthesize fetched evidence
- `legal.case_research` - legal variant with RAG + wiki
