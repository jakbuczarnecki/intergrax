# Intergrax — Documentation

**Last updated:** 2026-06-02 (Phase V hardening; Phase W-ML modality docs; K.1/K.2 deferred)

The `docs/` folder holds the canonical platform documentation.

---

## Documents

| Document | Purpose |
|----------|---------|
| [**INTERGRAX_DEVELOPMENT_STRATEGY.md**](INTERGRAX_DEVELOPMENT_STRATEGY.md) | **Strategic goal** — decision hierarchy, lab vs production harness, work cycle |
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, retry (§31), observability & trace storage (§33), RAG stack (§7.1.2) |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps; Appendix A–G + Phase V hardening streams |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| [**INTEGRATIONS.md**](INTEGRATIONS.md) | **Integration catalog** — all implemented providers, contracts, wiring, usage links |
| [**TOOLS.md**](TOOLS.md) | **Tool catalog** — atomic LLM/MCP tools, engine status, four-layer stack |
| [**SKILLS.md**](SKILLS.md) | **Skill Library** — composable capability packs, registry, importers |
| [**MODALITY.md**](MODALITY.md) | **Model & modality plane** — vision (YOLO/ONNX/…), audio/speech, classical ML, Hugging Face roles |
| [**HARNESS_ENVIRONMENT.md**](HARNESS_ENVIRONMENT.md) | **Harness environment** — lab stack, OTLP, skills preset, verification |
| [**../README.md**](../README.md) | **GitHub landing** — tiers, Integration/Tool/Skill/Agent stack, links to canon and plan |
| [**LLM_ADAPTERS.md**](LLM_ADAPTERS.md) | **LLM adapter catalog** — providers, streaming, tools, env vars, Prometheus/governance |
| [**IDEAL_HARNESS_AI_ARCHITECTURE.md**](IDEAL_HARNESS_AI_ARCHITECTURE.md) | **Target Harness AI architecture** — ideal Agent OS reference model for Integrax alignment |
| [**../infra/README.md**](../infra/README.md) | **Local Docker infra** — compose profiles, `manage.sh` |
| [**../infra/PORTS.md**](../infra/PORTS.md) | Host port matrix for integration backends |
| **This file** | Navigation and update rules |

```text
Strategy (why / priority)   →  INTERGRAX_DEVELOPMENT_STRATEGY.md
Architecture (what)         →  intergrax_runtime_architecture.md
Implementation (status)   →  INTERGRAX_IMPLEMENTATION_PLAN.md
Agent workflow (how)      →  AGENT_CREATION_GUIDE.md
Integrations (catalog)    →  INTEGRATIONS.md
Tools (catalog)           →  TOOLS.md
Skills (catalog)          →  SKILLS.md
Modality / ML / vision    →  MODALITY.md
Harness environment       →  HARNESS_ENVIRONMENT.md
LLM adapters + metrics    →  LLM_ADAPTERS.md
Ideal Harness AI target   →  IDEAL_HARNESS_AI_ARCHITECTURE.md
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand strategic direction | [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| Understand the platform | Strategy doc, then implementation plan §0, then architecture canon §1–§5 |
| See what to implement next | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **§6.1 + §6.2** (harness only, including Phase V) — **not** §6.3 unless product reprioritizes |
| Understand Phase V sequence/dependencies | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase V — Execution matrix** |
| See Phase V KPI thresholds | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase V — KPI thresholds and acceptance metrics** |
| See L3/L4 architecture maturity gates | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase V — L3/L4 gate evidence** |
| Business agents / new product apps (end of plan) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **§6.3** — deferred; Appendix A when starting K.* |
| Harness AI terminology | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §5.3 |
| Post-audit hardening tracker | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Appendix D** (Q+ **Done**) |
| Harness AI alignment (skills, context) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Appendix E** (R **Done**) · canon §5.3, §7.1.8 |
| Harness environment (lab stack, OTLP, ops) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase S/T** + **Appendix F** · `HARNESS_ENVIRONMENT.md` |
| Harness production hardening (security, policy, contracts) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase U** + **Appendix G** |
| Harness architecture hardening (capability graph, lifecycle, metrics, prompt/eval/context/security/cost) | [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase V** · [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §53 |
| Vision / audio / ML modality architecture (YOLO, ElevenLabs, HF, ONNX) | [MODALITY.md](MODALITY.md) · canon §7.1.9 · plan **Phase W-ML** |
| Configure multimodal LLM vs dedicated CV | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) (Plane A) · [MODALITY.md](MODALITY.md) (Planes B/C) |
| Check readiness for business agents | Implementation plan **Appendix A** |
| Review technical debt before Tier-1 work | Implementation plan **Appendix B** |
| Wire external systems (DB, Slack, Jira, …) | [INTEGRATIONS.md](INTEGRATIONS.md), then architecture canon §7.1 |
| Start local Redis / Qdrant / Neo4j / … | [../infra/README.md](../infra/README.md), [../infra/PORTS.md](../infra/PORTS.md) |
| Wire agent-callable tools (RAG, web search, Jira, …) | [TOOLS.md](TOOLS.md), then architecture canon §7.1.6–§7.1.7 |
| Understand Integration vs Tool vs Skill vs Agent | Architecture §5.3, §7.1.6–§7.1.8 · [SKILLS.md](SKILLS.md) |
| RAG engine (RetrievalService, RagProfile, metrics) | Architecture §7.1.2 · Phase M-RAG in implementation plan |
| Configure LLM providers (OpenAI, Claude, Bedrock, …) | [LLM_ADAPTERS.md](LLM_ADAPTERS.md), then architecture canon §5.2.2 |
| Evaluate Integrax against ideal Harness AI architecture | [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) |
| LLM/RAG Prometheus, trace DB defaults | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) · architecture §33 |
| Nexus retry layers | Architecture §31.1 |
| Nexus orchestration modules | `intergrax/runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, …) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Scaffold a new skill | `python -m intergrax.scaffold new-skill <skill_id>` |
| Deep-dive UAEP / hooks / governance | Architecture canon §42 |

---

## Current focus (2026-06-02)

| Phase | Status |
|-------|--------|
| **Phase Q+ — Harness hardening** | **Done** — [Appendix D](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-d--post-audit-hardening-traceability-phase-q) |
| **Phase R — Harness AI alignment** | **Done (MVP)** — [Appendix E](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-e--harness-ai-alignment-traceability-phase-r) |
| Phase Q — Harness quality | **Done** (Appendix C) |
| Phase L — Agent OS certification | **Done** |
| Phase M / M-LLM / M-RAG / N / O | **Done** (beta where noted) |
| **Phase S — Harness environment GA** | **Done** (2026-06-01) — [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md) · [Appendix F](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-f--harness-environment-traceability-phase-s) |
| **Phase T / U — Harness cleanliness + production hardening** | **Done** (2026-06-01) — [Appendix G](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-g--harness-production-audit-traceability-phase-u) |
| **Phase V — Harness architecture hardening** | **Active** — Phase V in [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) · canon §53 (`V-CG.1`, `V-AM.1`, `V-ALG.1` done) |
| **Phase W-ML — Model & modality plane** | **Docs Done** · implementation Planned — [MODALITY.md](MODALITY.md) · canon §7.1.9 |
| **Phase V execution controls** | **Defined** — execution matrix, KPI thresholds, cadence, ownership, L3/L4 gates in Phase V section |
| **Harness completion (§4.1)** | **Done** (2026-06-02) |
| **Phase K — Business agents** | **End of plan** — §6.3; **not** default next |

Gate: `uv run pytest -m gate -q` — **481 passed** (2026-06-02)

Harness CI also runs: `python scripts/check_harness_no_getattr.py` (zero grandfathered paths)

---

## Update rules

1. **Strategy** (goal, hierarchy, work cycle) → `INTERGRAX_DEVELOPMENT_STRATEGY.md`.
2. **Architecture** (including observability, retry semantics, trace storage, RAG metrics) → `intergrax_runtime_architecture.md`, then sync §0 in the plan.
3. **Status / phases / gaps** → `INTERGRAX_IMPLEMENTATION_PLAN.md` (§0, phase sections, appendices).
4. **Agent author workflow** → `AGENT_CREATION_GUIDE.md`.
5. **Integration or tool catalog changes** → `INTEGRATIONS.md` or `TOOLS.md` respectively.
6. **Skill packs / importers** → `SKILLS.md` + plan Appendix E (and Phase S when prod proof).
7. **Modality / vision / speech / ML** → `MODALITY.md` + canon §7.1.9 + plan Phase W-ML.
8. **Harness AI terms** → `intergrax_runtime_architecture.md` §5.3 only (single source of truth).
9. After each merged harness PR: run gate + getattr audit; update §0 gate count in the plan footer.
