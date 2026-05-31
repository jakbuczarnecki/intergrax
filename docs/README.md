# Intergrax — Documentation

**Last updated:** 2026-05-27

The `docs/` folder holds the canonical platform documentation.

---

## Documents

| Document | Purpose |
|----------|---------|
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, contracts, forbidden patterns |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps, priority, readiness (Appendix A), technical debt backlog (Appendix B) |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| [**INTEGRATIONS.md**](INTEGRATIONS.md) | **Integration catalog** — all implemented providers, contracts, wiring, usage links |
| [**TOOLS.md**](TOOLS.md) | **Tool catalog** — LLM-facing tools, engine status, unified tool model, planned backlog |
| [**LLM_ADAPTERS.md**](LLM_ADAPTERS.md) | **LLM adapter catalog** — providers, streaming, tools, env vars, registry |
| **This file** | Navigation and update rules |

```text
Architecture (what)        →  intergrax_runtime_architecture.md
Implementation (status)    →  INTERGRAX_IMPLEMENTATION_PLAN.md
Agent workflow (how)       →  AGENT_CREATION_GUIDE.md
Integrations (catalog)     →  INTEGRATIONS.md
Tools (catalog)            →  TOOLS.md
LLM adapters (catalog)     →  LLM_ADAPTERS.md
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand the platform | Implementation plan §0, then architecture canon §1–§5 |
| See current phase and what's next | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §1–§4 |
| Check readiness for business agents | Implementation plan **Appendix A** |
| Review technical debt before Tier-1 work | Implementation plan **Appendix B** |
| Wire external systems (DB, Slack, Jira, …) | [INTEGRATIONS.md](INTEGRATIONS.md), then architecture canon §7.1 |
| Wire agent-callable tools (RAG, web search, Jira, …) | [TOOLS.md](TOOLS.md), then architecture canon §7.1.6–§7.1.7 |
| Configure LLM providers (OpenAI, Claude, Bedrock, …) | [LLM_ADAPTERS.md](LLM_ADAPTERS.md), then architecture canon §5.2.2 |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-05-30)

| Phase | Status |
|-------|--------|
| Phase L — Agent OS certification | **Done** |
| Phase M — Integration Library (Tier-0 catalog) | **Done** (beta) — **73** providers with English `USAGE.md`; see [INTEGRATIONS.md](INTEGRATIONS.md) |
| Phase O — Tool Library & unified tool model | **Done** — 11 catalog tools; see [TOOLS.md](TOOLS.md) |
| Phase M-LLM — LLM adapter layer | **Done** (beta) — 19 providers, resilience, tenant metrics, PR guard; see [LLM_ADAPTERS.md](LLM_ADAPTERS.md) |
| Phase N — Application environment scaffold | **Done** (N.0–N.10) — see implementation plan Phase N |
| Phase K — Problem Radar / Vendor Discovery | **Ready to open** (product decision) |

Gate: `uv run pytest -m gate -q` — **363 passed** (full gate); CI workflow runs **335** of them (~20s pytest; see `.github/workflows/unit-tests.yml`)

---

## Update rules

1. **Architecture** → `intergrax_runtime_architecture.md`, then sync §0 in the implementation plan.
2. **Phase / status / readiness** → `INTERGRAX_IMPLEMENTATION_PLAN.md` only.
3. **Agent workflow** → `AGENT_CREATION_GUIDE.md` only.
4. **Integration catalog** → `INTEGRATIONS.md` when adding or changing providers.
5. **Tool catalog** → `TOOLS.md` when adding or changing catalog tools or tool engine contracts.
6. **Do not add** new markdown files to `docs/` without updating this index.

Product-specific roadmaps belong under `agents/<name>/` (e.g. `agents/legal/`), not in `docs/`.
