# Tier-2 agents (`agents/`)

**Role:** Reusable domain capabilities — contracts, typed step loop (`on_next_step`), prompts.  
**Hosts:** Tier-3 applications under `applications/` mount agents via `AgentBinding.mount(...)`.  
**Workflow:** [`docs/guides/AGENT_CREATION_GUIDE.md`](../docs/guides/AGENT_CREATION_GUIDE.md) · Appendix AC  
**Architecture:** [`docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 · **§32.0** readability  
**Implementation plan:** [`docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Phase **ACP** — waves §6.1aw  

**Migration (2026):** full fleet program — plan **Wave 8** (`ACP-MIG-*`). Bridge compat in Wave 4; **body migration** per-agent via tiered batches (T0→T4). Tracker: [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) fleet migration tracker. New agents: **READ → UPDATE → DECIDE** + scoreboard (`ACP-PROD-12`).

```text
agents/<slug>/     →  capability modules (no applications/ imports)
applications/      →  deployable environments that compose agents
```

### ACP fleet migration (Wave 8)

| Tier | Batch | Agents | Status |
|------|-------|--------|--------|
| **T0** harness | MIG-3 pilot | `echo`, `signoff_probe` | **Done** — typed `ReflexAgent` + UAEP shim |
| **T1** staging read | MIG-3 pilot | `research` | **Done** |
| **T1** staging read | MIG-4 | `summary`, `local_search` | **Done** |
| **T2** staging mutating | MIG-4 | `legal`, LKW trio, DSW quartet | **Done** |
| **T4** long-running | MIG-5 | `organization_worker`, `intergrax_assistant`, K-path agents | **Done** |

Inventory: `uv run python scripts/audit_agent_fleet_legacy.py` → `build/agent_fleet_inventory.json`.  
CI gate: `uv run python scripts/check_agent_acp_close_ci.py` (fleet migration + scoreboard blockers; ACP-CLOSE-CI-1/3).  
Scoreboard (ACP-PROD-12): `uv run python scripts/report_agent_production_readiness.py --roster` → `build/agent_production_readiness.json`.  
Fleet closure (ACP-LEG-2): `uv run python scripts/check_agent_production_readiness.py --require-fleet-migration-closure --regenerate`.

---

## Agent roster

| Agent | Capability | Lifecycle | Host application(s) | Docs |
|-------|------------|-----------|---------------------|------|
| **EchoAgent** | `echo.basic` | production | `lab_application`, `poc_template_application` | [`echo/`](echo/) |
| **SignoffProbeAgent** | (harness probe) | staging | `lab_application` | [`signoff_probe/`](signoff_probe/) |
| **ResearchAgent** | `research.web_search`, `research.pipeline` | staging | `research_application`, `lab_application` | [`research/`](research/) |
| **SummaryAgent** | `research.summarize` | staging | `research_application` | [`research/`](research/) |
| **LegalAgent** | `legal.review` | staging | `legal_application`, `lab_application` | [`legal/`](legal/) |
| **LocalIndexerAgent** | `local.workspace.index` | staging | `local_workspace_application` | [`local_indexer/`](local_indexer/) |
| **LocalSearchAgent** | `local.workspace.search` | staging | `local_workspace_application` | [`local_search/`](local_search/) |
| **LocalSynthesizerAgent** | `local.workspace.synthesize` | staging | `local_workspace_application` | [`local_synthesizer/`](local_synthesizer/) |
| **DisputeIntakeAgent** | `dispute.intake` | staging | `dispute_sim_application` | [`dispute_intake/`](dispute_intake/) |
| **DisputeAnalystAgent** | `dispute.analyze` | staging | `dispute_sim_application` | [`dispute_analyst/`](dispute_analyst/) |
| **DisputeStrategistAgent** | `dispute.strategy` | staging | `dispute_sim_application` | [`dispute_strategist/`](dispute_strategist/) |
| **DisputeScenarioAgent** | `dispute.scenario` | staging | `dispute_sim_application` | [`dispute_scenario/`](dispute_scenario/) |
| **OrganizationWorkerAgent** | `org.vendor_report` | development | `lab_application` (optional flag) | [`organization_worker/`](organization_worker/) |
| **IntergraxAssistantAgent** | `platform.assist` | development | `intergrax_assistant_application` | [`intergrax_assistant/`](intergrax_assistant/) |
| **ProblemRadarAgent** | `problem_radar.scan` | experimental | certified K.1 deploy path | [`problem_radar/`](problem_radar/) |
| **VendorDiscoveryAgent** | `vendor_discovery.search` | experimental | certified K.2 deploy path | [`vendor_discovery/`](vendor_discovery/) |
| **Lab mock agents** | harness fixtures | — | `lab_application` tests | [`lab/mock_agents.py`](lab/mock_agents.py) |

---

## By product environment

### Local Knowledge Workspace (LKW)

| Agent | Capability | Pipeline step |
|-------|------------|---------------|
| `local_indexer` | `local.workspace.index` | Ingest paths → RAG index |
| `local_search` | `local.workspace.search` | Retrieve + answer |
| `local_synthesizer` | `local.workspace.synthesize` | Shadow artifact drafts |

**Host:** [`applications/local_workspace_application/`](../applications/local_workspace_application/) · **Architecture:** [ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md)

### Dispute Simulation Workspace (DSW)

| Agent | Capability | Pipeline step |
|-------|------------|---------------|
| `dispute_intake` | `dispute.intake` | Classify materials, chronology, case RAG |
| `dispute_analyst` | `dispute.analyze` | Argument matrix, strengths/weaknesses |
| `dispute_strategist` | `dispute.strategy` | Attack/defense lines, emphasis map |
| `dispute_scenario` | `dispute.scenario` | Court variants, correspondence review |

**Host:** [`applications/dispute_sim_application/`](../applications/dispute_sim_application/) · **Architecture:** [ARCHITECTURE.md](../applications/dispute_sim_application/ARCHITECTURE.md)

### Legal review (single-agent SKU)

| Agent | Capability |
|-------|------------|
| `legal` | `legal.review` |

**Host:** [`applications/legal_application/`](../applications/legal_application/) — contract review; distinct from DSW dispute lifecycle.

### Research (multi-agent)

| Agent | Capability |
|-------|------------|
| `research` | `research.web_search`, `research.pipeline` |
| `summary` | `research.summarize` |

**Host:** [`applications/research_application/`](../applications/research_application/)

### Intergrax Assistant (harness chat hub)

| Agent | Capability | Role |
|-------|------------|------|
| `intergrax_assistant` | `platform.assist` | Conversational hub — default chat entry |

Optional specialists (Legal, Research, …) are mounted in the same Tier-3 host via env flags; Nexus delegates — hub does not call them directly.

**Host:** [`applications/intergrax_assistant_application/`](../applications/intergrax_assistant_application/) · **Architecture:** [ARCHITECTURE.md](../applications/intergrax_assistant_application/ARCHITECTURE.md)

---

## Harness / lab agents

Use **`lab_application`** (port `8090`) to experiment with any registered agent — debug API, trace inspection, optional plugin discovery.

| Agent | Typical use |
|-------|-------------|
| `echo` | Minimal UAEP smoke |
| `signoff_probe` | Policy / harness sign-off probes |
| `organization_worker` | Long-running / HITL harness demo |

---

## Per-agent documentation

Each agent folder ships:

| File | Purpose |
|------|---------|
| `README.md` | Quick start, capabilities, registration |
| `ARCHITECTURE.md` | Purpose, layout, runtime contracts |
| `IMPLEMENTATION_PLAN.md` | Local task queue |
| `adr/` | Agent-level architecture decisions (when needed) |

---

## Create a new agent

```bash
python -m intergrax.scaffold new-agent my_agent --capability domain.action
# Or agent + application bundle:
python -m intergrax.scaffold new-stack my_feature --profile lab --capability my_feature.basic
```

Full workflow: [`docs/guides/AGENT_CREATION_GUIDE.md`](../docs/guides/AGENT_CREATION_GUIDE.md)
