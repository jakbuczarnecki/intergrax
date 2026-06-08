# `research.citation_synthesis`

**Bundle:** `research` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Citation-backed report writing** for `SummaryAgent` and research pipelines: combine index retrieval and web sources, export structured report to shadow workspace. Closes the research loop from scan → synthesis artifact.

## How it works

1. Unions `rag.retrieve`, `websearch.query`, `workspace.write_file`.
2. Agent gathers citations via retrieval/search tools, then writes report file.
3. Prompt ref: `research.citation_synthesis.system`.
4. Requires workspace + RAG + search wired on host.

## How to use

```python
from intergrax.skills.providers.research.manifests import RESEARCH_CITATION_SYNTHESIS

AgentContract(id="summary", skills=[RESEARCH_CITATION_SYNTHESIS], ...)
```

Use `research_skill_profile()` + shadow workspace enabled.

## What you get

| Benefit | Detail |
|---------|--------|
| **Report export** | Standard write path for summaries |
| **Dual sources** | Index citations + live web refs |
| **SummaryAgent fit** | Matches `research.summarize` capability |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Cite indexed passages |
| `websearch.query` | Cite live web sources |
| `workspace.write_file` | Export report markdown/PDF path |

## Related skills

- `research.literature_scan` — upstream scan
- `workspace.authoring` — edit drafts post-synthesis
- `data.sql_analyst` — embed tabular results in reports
