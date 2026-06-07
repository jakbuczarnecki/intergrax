# Intergrax — Documentation

**Last updated:** 2026-06-06 (FAUDIT-32 remediation Done; Band 2ad closed; §6.1 maintenance)

The `docs/` folder holds the canonical platform documentation.

**Discovery & AI context files** (repository root — route here, do not duplicate canon):

| File | Purpose |
|------|---------|
| [../llms.txt](../llms.txt) | Concise LLM project map |
| [../llms-full.txt](../llms-full.txt) | Extended LLM context map |
| [../AGENTS.md](../AGENTS.md) | Instructions for AI coding agents |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide |
| [../CITATION.cff](../CITATION.cff) | Citation metadata |

---

## Documents

| Document | Purpose |
|----------|---------|
| [**INTERGRAX_DEVELOPMENT_STRATEGY.md**](INTERGRAX_DEVELOPMENT_STRATEGY.md) | **Strategic goal** — decision hierarchy, lab vs production harness, work cycle |
| [**intergrax_runtime_architecture.md**](intergrax_runtime_architecture.md) | **Architecture canon** — tiers, Nexus, UAEP §42, retry (§31), observability & trace storage (§33), RAG stack (§7.1.2) |
| [**INTERGRAX_IMPLEMENTATION_PLAN.md**](INTERGRAX_IMPLEMENTATION_PLAN.md) | **Implementation map** — phases, status, gaps; **FAUDIT-32 Done** (Band 2ad, 23/23 + §6.1ai); **M.6 P6 Done** (Band 2ac, 32/32); gate **901**; default queue = **§6.1 maintenance** |
| [**AGENT_CREATION_GUIDE.md**](AGENT_CREATION_GUIDE.md) | **Agent workflow** — scaffold → register → run → inspect → evaluate |
| [**INTEGRATIONS.md**](INTEGRATIONS.md) | **Integration catalog** — **167** providers, contracts, wiring, usage links |
| [**TOOLS.md**](TOOLS.md) | **Tool catalog** — **36** LLM/MCP tools in **16** bundles, engine status, four-layer stack |
| [**SKILLS.md**](SKILLS.md) | **Skill Library** — composable capability packs, registry, importers |
| [**EXTENSION_AUTHOR_GUIDE.md**](EXTENSION_AUTHOR_GUIDE.md) | **Tier-0 plugins** — integrations, tools, skills; entry points, bootstrap |
| [**MODALITY.md**](MODALITY.md) | **Model & modality plane** — vision (YOLO/ONNX/…), audio/speech, classical ML, Hugging Face roles |
| [**HARNESS_ENVIRONMENT.md**](HARNESS_ENVIRONMENT.md) | **Harness environment** — lab stack, OTLP, skills preset, verification |
| [**../README.md**](../README.md) | **GitHub landing** — tiers, Integration/Tool/Skill/Agent stack, links to canon and plan |
| [**LLM_ADAPTERS.md**](LLM_ADAPTERS.md) | **LLM adapter catalog** — providers, streaming, tools, env vars, Prometheus/governance |
| [**IDEAL_HARNESS_AI_ARCHITECTURE.md**](IDEAL_HARNESS_AI_ARCHITECTURE.md) | **Target Harness AI architecture** — ideal Agent OS reference model for Integrax alignment |
| [**NEXUS_EXECUTION_FLOW_REFERENCE.md**](NEXUS_EXECUTION_FLOW_REFERENCE.md) | **Nexus execution flow** — operational narrative, diagrams, edge cases, evaluation hooks, plan traceability ([Phase FLOW](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-flow--nexus-execution-depth) **Done** 17/18 · **FLOW-8 Deferred**) |
| [**adr/ADR-FLOW-001.md**](adr/ADR-FLOW-001.md) | **Delegation semantics** — `DELEGATES_TO` graph expansion (Option C); **implemented** (`FLOW-2`, `FLOW-14`) |
| [**adr/ADR-FLOW-002.md**](adr/ADR-FLOW-002.md) | **Lifecycle semantics** — reserved `WAITING_FOR_RESOURCES` / `EXPIRED` states (`FLOW-10`) |
| [**adr/ADR-FLOW-003.md**](adr/ADR-FLOW-003.md) | **`MODIFY_PLAN` semantics** — reserved v1; `MODIFY_PLAN_NOT_SUPPORTED` without handoff (`FLOW-16`) |
| [**ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md**](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) | **Adaptive Harness Intelligence (AHI)** — L4 closed-loop architecture RFC, business case, Phase W-ADAPT roadmap |
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
Nexus execution flow      →  NEXUS_EXECUTION_FLOW_REFERENCE.md · Appendix I · canon §42.43
Adaptive Harness (L4)     →  ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md · canon §54
```

---

## Start here

| I want to… | Read |
|------------|------|
| Understand strategic direction | [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| Understand the platform | Strategy doc, then implementation plan §0, then architecture canon §1–§5 |
| Infrastructure vs business scope | [INTERGRAX_IMPLEMENTATION_PLAN.md §4.0a](INTERGRAX_IMPLEMENTATION_PLAN.md#40a-implementation-scope-split-infrastructure-vs-business) |
| See what to implement next (harness) | [§6.1](INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1) maintenance only · [§6.3](INTERGRAX_IMPLEMENTATION_PLAN.md#63-end-of-plan--deferred-product-work-only) product work (deferred) |
| Implement Adaptive Harness Intelligence (L4 runtime) | [ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) · [Phase W-ADAPT](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · [Appendix K](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-k--adaptive-harness-intelligence-traceability-phase-w-adapt) |
| **Agent assembly (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix N](AGENT_CREATION_GUIDE.md#appendix-n--agent-assembly-control-plane) · [Phase AS](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-as--agent-assembly-control-plane-closeout) |
| **Registry architecture (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix O](AGENT_CREATION_GUIDE.md#appendix-o--registry-architecture-control-plane) · [Phase REG](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-reg--registry-architecture-control-plane-closeout) |
| **Capability graph (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix P](AGENT_CREATION_GUIDE.md#appendix-p--capability-graph-control-plane) · [Phase CG](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-cg--capability-graph-control-plane-closeout) |
| **Observability wiring (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix Q](AGENT_CREATION_GUIDE.md#appendix-q--observability-control-plane-closeout) · [Phase OBS](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-obs--observability-control-plane-closeout) |
| **Reliability wiring (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix R](AGENT_CREATION_GUIDE.md#appendix-r--reliability-control-plane-closeout) · [Phase REL](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-rel--reliability-control-plane-closeout) |
| **Security wiring (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix S](AGENT_CREATION_GUIDE.md#appendix-s--security-control-plane-closeout) · [Phase SEC](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-sec--security-control-plane-closeout) |
| **Cost governance (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix T](AGENT_CREATION_GUIDE.md#appendix-t--cost-governance-control-plane-closeout) · [Phase COST](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-cost--cost-governance-control-plane-closeout) |
| **Evaluation wiring (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix U](AGENT_CREATION_GUIDE.md#appendix-u--evaluation-control-plane-closeout) · [Phase EVAL](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-eval--evaluation-control-plane-closeout) |
| **Policy, governance & observability (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix H](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · canon [§42.11](intergrax_runtime_architecture.md#4211-policy-engine) · [`HARNESS_ENVIRONMENT.md`](HARNESS_ENVIRONMENT.md) |
| **Full Nexus execution flow (intake → result)** | [**NEXUS_EXECUTION_FLOW_REFERENCE.md**](NEXUS_EXECUTION_FLOW_REFERENCE.md) · plan traceability §23 (`FLOW-GAP.*`) |
| **Orchestration, graphs & delegation (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix I](AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) · [**flow reference**](NEXUS_EXECUTION_FLOW_REFERENCE.md) · canon [§42.43](intergrax_runtime_architecture.md#4243-multi-agent-collaboration-flow-reference) · [Appendix C](AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs) |
| **Tools & skills (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix J](AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) · [Phase TS](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-ts--tools--skills-control-plane-closeout) |
| **Integration & RAG (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix K](AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · [Phase INT](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-int--integration-control-plane-closeout) · [Phase RAG](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-rag--rag-retrieval-control-plane-closeout) |
| **Context engineering (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix L](AGENT_CREATION_GUIDE.md#appendix-l--context-engineering-control-plane) · [Phase CTX](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-ctx--context-engineering-control-plane-closeout) |
| **Prompt registry (control plane)** | [`AGENT_CREATION_GUIDE.md` Appendix M](AGENT_CREATION_GUIDE.md#appendix-m--prompt-registry-control-plane) · [Phase PE](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-pe--prompt-registry-control-plane-closeout) |
| Audit policy / observability layers | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §5, §21 · [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) |
| Audit orchestration / graph / subagent layers | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) §7–§10 · [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) |
| Memory platform (STM/LTM/context/hooks) | [Phase MEM](INTERGRAX_IMPLEMENTATION_PLAN.md#phase-mem--memory-platform-completion) · **Done** 48/48 |
| Business / product backlog only | [§6.3a](INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated) — after explicit product decision |
| **Local Knowledge Workspace (LKW)** — first business product | [applications/local_workspace_application/ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md) · agents `local_indexer`, `local_search`, `local_synthesizer` · plan **LKW.*** |
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
| Plugin catalogs (integrations, tools, skills) — **Done** | [EXTENSION_AUTHOR_GUIDE.md](EXTENSION_AUTHOR_GUIDE.md) · [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) **Phase P-Ext** · **Appendix I** |
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
| Design or implement L4 adaptive harness (closed loops) | [ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) · [canon §54](intergrax_runtime_architecture.md#54-adaptive-harness-intelligence-ahi--l4-runtime-addendum) |
| LLM/RAG Prometheus, trace DB defaults | [LLM_ADAPTERS.md](LLM_ADAPTERS.md) · architecture §33 |
| Nexus retry layers | Architecture §31.1 |
| Nexus orchestration modules | `intergrax/runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, …) |
| Create a new agent | [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) |
| Scaffold a new skill | `python -m intergrax.scaffold new-skill <skill_id>` |
| Deep-dive UAEP / hooks / governance | [`AGENT_CREATION_GUIDE.md` Appendix H](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · canon §42 |

---

## Current focus (2026-06-06)

| Phase | Status |
|-------|--------|
| **Phase FAUDIT-32 — Full architecture audit** | **Done** (Band 2ad) — 23/23 remediation + [§6.1ai](INTERGRAX_IMPLEMENTATION_PLAN.md#61ai-harness-implementation-queue--faudit-32-follow-up-closed) follow-up |
| **Phase W-ADAPT — Adaptive Harness Intelligence** | **Done** (Band 2y) — **70/70 Done** (Wave 0–7) · [AHIA](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) · [ADR-ADAPT-001](adr/ADR-ADAPT-001.md) |
| **Phase M-LLM-R — LLM completion envelope** | **Done** (Band 2z, 39/39) |
| **Phase M.6 P6 — Integration expansion** | **Done** (Band 2ac, 32/32) |
| **Phase Q+ — Harness hardening** | **Done** — [Appendix D](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-d--post-audit-hardening-traceability-phase-q) |
| **Phase R — Harness AI alignment** | **Done (MVP)** — [Appendix E](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-e--harness-ai-alignment-traceability-phase-r) |
| Phase Q — Harness quality | **Done** (Appendix C) |
| Phase L — Agent OS certification | **Done** |
| Phase M / M-LLM / M-RAG / N / O | **Done** (beta where noted) |
| **Phase S — Harness environment GA** | **Done** (2026-06-01) — [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md) · [Appendix F](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-f--harness-environment-traceability-phase-s) |
| **Phase T / U — Harness cleanliness + production hardening** | **Done** (2026-06-01) — [Appendix G](INTERGRAX_IMPLEMENTATION_PLAN.md#appendix-g--harness-production-audit-traceability-phase-u) |
| **Phase V — Harness architecture hardening** | **Done** — [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) Phase V |
| **Phase W-ML — Model & modality plane** | **Done** — [MODALITY.md](MODALITY.md) · canon §7.1.9 |
| **Phase V execution controls** | **Defined** — execution matrix, KPI thresholds, cadence, ownership, L3/L4 gates in Phase V section |
| **Harness completion (§4.1)** | **Done** (2026-06-02) |
| **Phase AA — Agents & applications conformance** | **Platform Done** (2026-06-02) |
| **Phase MEM — Memory platform** | **Done** (48/48) |
| **GOV-AUDIT — Governance control plane** | **Done** (docs) — Appendix H |
| **Phase ORCH — Orchestration closeout** | **Done** (2026-06-05) — ORCH-1→4 |
| **Phase TS — Tools/skills closeout** | **Done** (2026-06-02) — TS-1→3 |
| **Phase INT — Integration closeout** | **Done** (2026-06-02) — INT-1→2 |
| **Phase RAG — RAG retrieval closeout** | **Done** (2026-06-02) — RAG-1 |
| **Phase CTX — Context engineering closeout** | **Done** (2026-06-02) — CTX-1→2 |
| **Phase LEG — Legacy tool plan closeout** | **Done** (2026-06-02) — LEG-1→3 |
| **Phase PE — Prompt registry closeout** | **Done** (2026-06-02) — PE-1→3 |
| **Phase CLEAN — Legacy module closeout** | **Done** (2026-06-02) — CLEAN-1→4 |
| **Phase AS — Agent assembly closeout** | **Done** (2026-06-02) — AS-1→3 |
| **Phase REG — Registry architecture closeout** | **Done** (2026-06-02) — REG-1→3 |
| **Phase CG — Capability graph closeout** | **Done** (2026-06-02) — CG-1→3 |
| **Phase K — Business agents** | **End of plan** — §6.3; **not** default next |

Gate: `uv run pytest -m gate -q` — **901 passed** (2026-06-06)

**Default harness queue:** [§6.1 maintenance](INTERGRAX_IMPLEMENTATION_PLAN.md#61-harness-platform-maintenance-default--band-1) only — product work deferred to [§6.3](INTERGRAX_IMPLEMENTATION_PLAN.md#63-end-of-plan--deferred-product-work-only).

Harness CI also runs: `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py`

---

## Update rules

1. **Strategy** (goal, hierarchy, work cycle) → `INTERGRAX_DEVELOPMENT_STRATEGY.md`.
2. **Architecture** (including observability, retry semantics, trace storage, RAG metrics) → `intergrax_runtime_architecture.md`, then sync §0 in the plan.
3. **Status / phases / gaps** → `INTERGRAX_IMPLEMENTATION_PLAN.md` (§0, phase sections, appendices).
4. **Agent author workflow** → `AGENT_CREATION_GUIDE.md` ([Appendix H — governance](AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane) · [Appendix I — orchestration](AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) · [Appendix J — tools/skills](AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) · [Appendix K — integration/RAG](AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · [Appendix L — context engineering](AGENT_CREATION_GUIDE.md#appendix-l--context-engineering-control-plane) · [Appendix M — prompt registry](AGENT_CREATION_GUIDE.md#appendix-m--prompt-registry-control-plane)).
5. **Integration or tool catalog changes** → `INTEGRATIONS.md` or `TOOLS.md` respectively.
6. **Skill packs / importers** → `SKILLS.md` + plan Appendix E (and Phase S when prod proof).
7. **Modality / vision / speech / ML** → `MODALITY.md` + canon §7.1.9 + plan Phase W-ML.
8. **Harness AI terms** → `intergrax_runtime_architecture.md` §5.3 only (single source of truth).
9. **Adaptive Harness Intelligence (L4 runtime)** → `ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md` + canon §54; Phase W-ADAPT **Done** — maintenance via §6.1 + `phase_w_adapt_closeout_gate.py`.
10. **Nexus execution flow narrative** (diagrams, edge cases, `FLOW-GAP.*`) → `NEXUS_EXECUTION_FLOW_REFERENCE.md`; sync plan §23 when scheduling `FLOW-*` rows.
11. After each merged harness PR: run gate + getattr audit; update §0 gate count in the plan footer.
