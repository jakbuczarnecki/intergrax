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
| Architecture [**§7.1.8**](intergrax_runtime_architecture.md) | **Skill Library** — composable packs (Phase R); catalog `SKILLS.md` when R-Skill.6 ships |
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
Skills (architecture)    →  intergrax_runtime_architecture.md §7.1.8 · plan Phase R
LLM adapters + metrics   →  LLM_ADAPTERS.md
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand the platform | Implementation plan §0, then architecture canon §1–§5 |
| See current phase and what's next | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) §4 — **Q+** → **Phase R** → Phase K |
| Post-audit hardening tracker | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Appendix D** |
| Harness AI alignment (skills, context) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase R**, **Appendix E** · canon §5.3, §7.1.8 |
| Check readiness for business agents | Implementation plan **Appendix A** |
| Review technical debt before Tier-1 work | Implementation plan **Appendix B** |
| Wire external systems (DB, Slack, Jira, …) | [INTEGRATIONS.md](INTEGRATIONS.md), then architecture canon §7.1 |
| Start local Redis / Qdrant / Neo4j / … | [../infra/README.md](../infra/README.md), [../infra/PORTS.md](../infra/PORTS.md) |
| Wire agent-callable tools (RAG, web search, Jira, …) | [TOOLS.md](TOOLS.md), then architecture canon §7.1.6–§7.1.7 |
| Understand Integration vs Tool vs Skill vs Agent | Architecture §5.3, §7.1.6–§7.1.8 · plan Appendix E |
| RAG engine (RetrievalService, RagProfile, metrics) | Architecture §7.1.2 · Phase M-RAG in implementation plan |
| Configure LLM providers (OpenAI, Claude, Bedrock, …) | [LLM_ADAPTERS.md](LLM_ADAPTERS.md), then architecture canon §5.2.2 |
| LLM/RAG Prometheus, trace DB defaults | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) · architecture §33 |
| Nexus retry layers | Architecture §31.1 |
| Nexus orchestration modules | `intergrax/runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, …) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-06-01)

| Phase | Status |
|-------|--------|
| **Phase Q+ — Harness hardening** | **Open** — [Appendix D](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-d--post-audit-hardening-traceability-phase-q) |
| **Phase R — Harness AI alignment** | **Open** — Skill Library, context budget, delegation — [Appendix E](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-e--harness-ai-alignment-traceability-phase-r) |
| Phase Q — Harness quality | **Done** (Appendix C) |
| Phase L — Agent OS certification | **Done** |
| Phase M / M-LLM / M-RAG / N / O | **Done** (beta where noted) |
| **Phase K — business agents** | **After Q+ Waves 1–3 + R-Skill core + R-Context.1** |

**Start implementation:** Q+ Wave 3 → **R Wave R0/R1** (`R-Skill.1`–`R-Skill.5`, `R-Context.1`)

Gate: `uv run pytest -m gate -q` — **410 passed** (2026-06-01, Waves 1–2 partial)

---

## Update rules

1. **Architecture** (including observability, retry semantics, trace storage, RAG metrics) → `intergrax_runtime_architecture.md`, then sync §0 in the plan.
2. **Phase / status / readiness** → `INTERGRAX_IMPLEMENTATION_PLAN.md` only.
3. **Agent workflow** → `AGENT_CREATION_GUIDE.md` only.
4. **Integration catalog** → `INTEGRATIONS.md` when adding or changing providers.
5. **Tool catalog** → `TOOLS.md` when adding or changing catalog tools or tool engine contracts.
6. **LLM providers and LLM metrics** → `LLM_ADAPTERS.md` only.
7. **Do not add** new markdown files to `docs/` without explicit approval and an index row here.

Product-specific roadmaps belong under `agents/<name>/` (e.g. `agents/legal/`), not in `docs/`.

