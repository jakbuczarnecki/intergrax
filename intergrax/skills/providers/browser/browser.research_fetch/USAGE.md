# `browser.research_fetch`

**Bundle:** `browser` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Rich web capture for research**: fetch JS-rendered pages via browser automation, read static URLs, and preview parsed document structure. Use when `websearch.query` snippets are insufficient and agents need full page content or parse previews.

## How it works

1. `browser.fetch_page` uses `BrowserAutomation` integration (Playwright, Browserbase, etc.).
2. `websearch.read_url` fetches simpler HTTP pages as fallback.
3. `document.parse_preview` shows chunk structure before full ingest.
4. Included in `research_skill_profile()` alongside `research.web_evidence`.

## How to use

```python
from intergrax.skills.providers.browser.manifests import BROWSER_RESEARCH_FETCH
from intergrax.applications._shared.skill_wiring import research_skill_profile

AgentContract(id="research", skills=[BROWSER_RESEARCH_FETCH, RESEARCH_WEB_EVIDENCE], ...)
```

Wire `browser_automation` + `search_provider` on integration profile.

## What you get

| Benefit | Detail |
|---------|--------|
| **JS-heavy sites** | Browser backend for SPAs |
| **Progressive capture** | Search → read → browser escalate path |
| **Ingest preview** | Validate parse before `rag.document_ingest` |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `browser.fetch_page` | Headless browser page capture |
| `websearch.read_url` | HTTP fetch + text extraction |
| `document.parse_preview` | Preview parsed document chunks |

## Related skills

- `research.web_evidence` - search + batch URL fetch
- `rag.document_ingest` - index captured content
