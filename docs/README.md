# Intergrax — Documentation

**Last updated:** 2026-06-01

The `docs/` folder holds the canonical platform documentation.

---

## Documents

| Document | Purpose |
|----------|---------|
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, retry (§31), observability & trace storage (§33), RAG stack (§7.1.2) |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps; Appendix A–D, **E (Phase R / Harness AI)** |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| [**INTEGRATIONS.md**](INTEGRATIONS.md) | **Integration catalog** — all implemented providers, contracts, wiring, usage links |
| [**TOOLS.md**](TOOLS.md) | **Tool catalog** — atomic LLM/MCP tools, engine status, four-layer stack |
| [**SKILLS.md**](SKILLS.md) | **Skill Library** — composable capability packs, registry, importers |
| [**../README.md**](../README.md) | **GitHub landing** — tiers, Integration/Tool/Skill/Agent stack, links to canon and plan |
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
Skills (catalog)         →  SKILLS.md
LLM adapters + metrics   →  LLM_ADAPTERS.md
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand the platform | Implementation plan §0, then architecture canon §1–§5 |
| See current phase and what's next | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §4 — **Phase K** (business agents) |
| Post-audit hardening tracker | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Appendix D** (Q+ **Done**) |
| Harness AI alignment (skills, context) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Appendix E** (R **Done**) · canon §5.3, §7.1.8 |
| Check readiness for business agents | Implementation plan **Appendix A** |
| Review technical debt before Tier-1 work | Implementation plan **Appendix B** |
| Wire external systems (DB, Slack, Jira, …) | [INTEGRATIONS.md](INTEGRATIONS.md), then architecture canon §7.1 |
| Start local Redis / Qdrant / Neo4j / … | [../infra/README.md](../infra/README.md), [../infra/PORTS.md](../infra/PORTS.md) |
| Wire agent-callable tools (RAG, web search, Jira, …) | [TOOLS.md](TOOLS.md), then architecture canon §7.1.6–§7.1.7 |
| Understand Integration vs Tool vs Skill vs Agent | Architecture §5.3, §7.1.6–§7.1.8 · [SKILLS.md](SKILLS.md) |
| RAG engine (RetrievalService, RagProfile, metrics) | Architecture §7.1.2 · Phase M-RAG in implementation plan |
| Configure LLM providers (OpenAI, Claude, Bedrock, …) | [LLM_ADAPTERS.md](LLM_ADAPTERS.md), then architecture canon §5.2.2 |
| LLM/RAG Prometheus, trace DB defaults | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) · architecture §33 |
| Nexus retry layers | Architecture §31.1 |
| Nexus orchestration modules | `intergrax/runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, …) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Scaffold a new skill | `python -m intergrax.scaffold new-skill <skill_id>` |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-06-01)

| Phase | Status |
|-------|--------|
| **Phase Q+ — Harness hardening** | **Done** — [Appendix D](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-d--post-audit-hardening-traceability-phase-q) |
| **Phase R — Harness AI alignment** | **Done (MVP)** — [Appendix E](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-e--harness-ai-alignment-traceability-phase-r) |
| Phase Q — Harness quality | **Done** (Appendix C) |
| Phase L — Agent OS certification | **Done** |
| Phase M / M-LLM / M-RAG / N / O | **Done** (beta where noted) |
| **Phase K — business agents** | **Next** — product agents on certified harness |

Gate: `uv run pytest -m gate -q` — **450 passed** (2026-06-01)

Harness CI also runs: `python scripts/check_harness_no_getattr.py` (zero grandfathered paths)

---

## Update rules

1. **Architecture** (including observability, retry semantics, trace storage, RAG metrics) → `intergrax_runtime_architecture.md`, then sync §0 in the plan.
2. **Status / phases / gaps** → `INTERGRAX_IMPLEMENTATION_PLAN.md` (§0, phase sections, appendices).
3. **Agent author workflow** → `AGENT_CREATION_GUIDE.md`.
4. **Integration or tool catalog changes** → `INTEGRATIONS.md` or `TOOLS.md` respectively.
5. **Skill packs / importers** → `SKILLS.md` + plan Phase R.
6. After each merged harness PR: run gate + getattr audit; update §0 gate count in the plan footer.
