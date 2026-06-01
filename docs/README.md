# Intergrax — Documentation

**Last updated:** 2026-06-01

The `docs/` folder holds the canonical platform documentation.

---

## Documents

| Document | Purpose |
|----------|---------|
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, retry (§31), observability & trace storage (§33), RAG stack (§7.1.2) |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps, priority, readiness (Appendix A), technical debt backlog (Appendix B) |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| [**INTEGRATIONS.md**](INTEGRATIONS.md) | **Integration catalog** — all implemented providers, contracts, wiring, usage links |
| [**TOOLS.md**](TOOLS.md) | **Tool catalog** — LLM-facing tools, engine status, unified tool model, planned backlog |
| [**LLM_ADAPTERS.md**](LLM_ADAPTERS.md) | **LLM adapter catalog** — providers, streaming, tools, env vars, Prometheus/governance |
| [**../infra/README.md**](../infra/README.md) | **Local Docker infra** — compose profiles, `manage.sh` |
| [**../infra/PORTS.md**](../infra/PORTS.md) | Host port matrix for integration backends |
| **This file** | Navigation and update rules |

```text
Architecture (what)        →  intergrax_runtime_architecture.md
Implementation (status)  →  INTERGRAX_IMPLEMENTATION_PLAN.md
Agent workflow (how)     →  AGENT_CREATION_GUIDE.md
Integrations (catalog)   →  INTEGRATIONS.md
Tools (catalog)          →  TOOLS.md
LLM adapters + metrics   →  LLM_ADAPTERS.md
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
| Start local Redis / Qdrant / Neo4j / … | [../infra/README.md](../infra/README.md), [../infra/PORTS.md](../infra/PORTS.md) |
| Wire agent-callable tools (RAG, web search, Jira, …) | [TOOLS.md](TOOLS.md), then architecture canon §7.1.6–§7.1.7 |
| RAG engine (RetrievalService, RagProfile, metrics) | Architecture §7.1.2 · Phase M-RAG in implementation plan |
| Configure LLM providers (OpenAI, Claude, Bedrock, …) | [LLM_ADAPTERS.md](LLM_ADAPTERS.md), then architecture canon §5.2.2 |
| LLM/RAG Prometheus, trace DB defaults | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) · architecture §33 |
| Nexus retry layers | Architecture §31.1 |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-06-01)

| Phase | Status |
|-------|--------|
| **Phase Q — Harness quality & consolidation** | **In progress** — Waves 1–6 done; remaining: `NexusLoop` &lt;600 lines, metadata bridge caller migration, Phase K — [Appendix C](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-c--harness-audit-traceability-phase-q) |
| Phase L — Agent OS certification | **Done** |
| Phase M / M-LLM / M-RAG / N / O | **Done** (beta where noted) |
| Phase K — Problem Radar / Vendor Discovery | **After Q** (product decision; may override) |

Gate: `uv run pytest -m gate -q` — **415 passed** (2026-06-01); run after each Q PR (must stay green)

---

## Update rules

1. **Architecture** (including observability, retry semantics, trace storage, RAG metrics) → `intergrax_runtime_architecture.md`, then sync §0 in the implementation plan.
2. **Phase / status / readiness** → `INTERGRAX_IMPLEMENTATION_PLAN.md` only.
3. **Agent workflow** → `AGENT_CREATION_GUIDE.md` only.
4. **Integration catalog** → `INTEGRATIONS.md` when adding or changing providers.
5. **Tool catalog** → `TOOLS.md` when adding or changing catalog tools or tool engine contracts.
6. **LLM providers and LLM metrics** → `LLM_ADAPTERS.md` only.
7. **Do not add** new markdown files to `docs/` without explicit approval and an index row here.

Product-specific roadmaps belong under `agents/<name>/` (e.g. `agents/legal/`), not in `docs/`.
