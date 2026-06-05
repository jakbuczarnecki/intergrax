# Intergrax — Runtime Implementation Plan

**The single implementation map** — phases, status, gaps, priority, and readiness checklist.

Status: Working draft (2026-06-05) — **Harness platform bands 1–2i Done**; Phase V + V-REM **closed**; **scope split** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business); product **Deferred** [§6.3a](#63a-business-backlog-register-consolidated); gate **571 passed**; **operational L3 signed off** (`release_cycles.json` ≥2 + `phase_w_ops_evidence.py --enforce`)  
Strategy: [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md)  
Architecture canon: [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md)  
Agent workflow: [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md)  
Navigation: [`README.md`](README.md)  

Principle: **evolve, not rewrite** · **reuse Tier-0** (canon §5.2)

---

## Documentation model

Do not maintain separate status/readiness/roadmap files. This plan is the **only** live implementation document:

| Topic | Where |
|-------|--------|
| Strategic goal, decision hierarchy, work cycle | [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| Full architecture specification | `intergrax_runtime_architecture.md` |
| Phase status, gaps, priority | **This file** — **§4** ladder; **§4.0a** infrastructure vs business; **§6.3** / **§6.3a** = product work only |
| Tier-0 integration catalog (what / where) | Architecture canon §7.1.1–§7.1.5 |
| Tier-0 integration implementation (how) | **This file** Phase M |
| Tier-0 tool catalog (what / where) | Architecture canon §7.1.6–§7.1.7, §22 |
| Tier-0 tool implementation (how) | **This file** Phase O |
| Agent creation workflow | `AGENT_CREATION_GUIDE.md` |
| Governance / policy / observability control plane (authoring) | `AGENT_CREATION_GUIDE.md` **Appendix H** · canon §42.11 · `EXTENSION_AUTHOR_GUIDE.md` §10 (`intergrax.policy_rules`) |
| Orchestration / graph / delegation control plane (authoring) | `AGENT_CREATION_GUIDE.md` **Appendix I** · canon §42.3–§42.15, §42.43 · R-Delegate **Done** |
| Tier-3 application environment (self-contained deploy) | Architecture canon §7.4.8–§7.4.10 |
| Tier-3 composition engine (manifest, wiring API) | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| Tier-3 application hosts (`applications/<app>/`) | [`applications/USAGE.md`](../applications/USAGE.md) |
| Application scaffold & deploy plan | **This file** Phase N |
| Business-agent go/no-go checklist | **Appendix A** (below) |
| Technical debt backlog (analysis only) | **Appendix B** (below) |
| Harness quality audit (2026-06-01) → Phase Q tracker | **This file** Phase Q + **Appendix C** |
| Post-audit hardening (typing, legacy, monoliths) | **This file** Phase Q+ + **Appendix D** |
| Harness GA / consolidation (no new OS features) | **This file** Phase Q / Q+ |
| Harness AI alignment audit (2026-06-01) → Phase R | **This file** Phase R + **Appendix E** + canon [§5.3](intergrax_runtime_architecture.md#53-harness-ai-alignment-conceptual-model) |
| Harness environment GA (2026-06-01) → Phase S | **This file** Phase S + **Appendix F** (K.1/K.2 → §6.3 end-of-plan) |
| Harness production hardening (2026-06-01 audit) → Phase U | **This file** Phase U + **Appendix G** (**Done**; does **not** schedule K.1/K.2 — see §6.3) |
| Skill / Tool / Integration layering (canon) | Architecture §5.3, §7.1.6–§7.1.8 |
| Skill catalog | `SKILLS.md` |
| Model & modality plane (vision, audio, ML) | Architecture canon §7.1.9 · [`MODALITY.md`](MODALITY.md) · **Phase W-ML** (below) |
| Plugin catalogs (integrations, tools, skills) | **This file** Phase P-Ext + **Appendix I** · [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) |
| Harness maturity audit (2026-06-02) → operational L3 | **Phase W-OPS** (below) · **§6.2w** · source: maturity audit 2026-06-02 (conversation) |
| Tier-3 application environment audit → full configurability | [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) → **Phase H-APP** · **§6.2x** |
| Developer authoring UX audit (LangGraph-like entry, measurable TTFRun) | **Phase DX** (below) · **§6.2y** · source: harness DX audit 2026-06-03 (conversation + H-APP gap analysis) |
| Agents & applications conformance audit (structure, scaffold, per-agent/app docs, deploy) | **Phase AA** (below) · **§6.2z** · source: Tier-2/Tier-3 audit 2026-06-03 (conversation) |
| Memory platform audit (STM/LTM/org/task/context/hooks/persistence) | **Phase MEM** (below) · **§6.2aa** · **§6.1aa** · source: memory audit 2026-06-02 (conversation) |
| Phase V runtime remediation (2026-06-05 audit) → close Partial gaps | **Phase V-REM** (below) · **Appendix J** · **§6.1z** · **§6.2v** · source: plan/code audit vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` |
| Phase V remediation traceability (audit gap → V-REM ID) | **Appendix J** (below) |
| Full architecture audit procedure (32 layers) | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) · prompt: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) |
| Infrastructure vs business scope split | **§4.0a** · [§6.1aa](#61aa-harness-implementation-queue-memory-platform) · [§6.3a](#63a-business-backlog-register-consolidated) |

**Note on audit source documents:** Some historical audit narratives (e.g. `HARNESS_APPLICATION_LAYER_AUDIT.md`) may live outside the repo. **Task traceability in this plan is canonical** — H-APP (43 tasks), W-OPS, MEM, DX, AA registers below; do not re-derive scope from missing files.

---

## 0. Architecture at a glance

Condensed from the canon. For full contracts and forbidden patterns, read `intergrax_runtime_architecture.md`.

### 0.1 Strategic objective

Intergrax is an **Agent Operating System / Harness AI runtime** — not a collection of business agents. **Priority 1:** production-grade Harness AI (see [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md)).

Current optimization targets:

- **harness environment GA** (Phase S) · **developer authoring UX** (Phase DX) · experimentation speed · agent creation speed · runtime stability
- orchestration quality · observability · composability · skill/platform packs (Integration → Tool → Skill)

**Harness GA (Phase L):** Agent OS certified — Appendix A **20/20**. New agents ship via scaffold without Nexus edits.

**Harness environment (Phase S):** **Done** (2026-06-01) — stable stack, OTLP profile, `harness.*` skills, `HARNESS_ENVIRONMENT.md`, CI smoke. **Did not include** business agents (K.1/K.2).

**Harness cleanliness (Phase T):** **Done** (2026-06-01) — unified `lab_harness_preset()`, typed reference agents, native `CatalogToolPlanner`, expanded stable stack. See Phase T.

**Harness production hardening (Phase U):** **Done** (2026-06-01) — auth surfaces, strict harness profile, `HarnessReferenceAgent`, typed policy bundle, planner decoupling, sandbox opt-in. **U-Leg** legacy removal remains tracked in Appendix G. See Phase U + **Appendix G**.

**Product agents (Phase K):** Problem Radar (K.1), Vendor Discovery (K.2) — **end of plan** (§4.0 Band 3, §6.3); not default next. K.3–K.5 platform hardening **Done**.

**Platform quality (Phase Q):** Done (2026-06-01) — first harness audit remediation; gate was **417 passed** at close (see Appendix C).

**Harness hardening (Phase Q+):** **Done** (2026-06-01) — Protocols (zero grandfathered `getattr` in harness paths), legacy stack removal, Nexus decomposition, monolith splits. See Appendix D.

**Harness AI alignment (Phase R):** **Done (MVP)** (2026-06-01) — **Skill Library**, context-engineering API, graph-native delegation, unified policy bundle. See Appendix E. **Phase S** hardens the **environment** agents run in; product agents follow in **Phase K**.

**Skill layer decision (ADR R.0.1):** **Do not** collapse skills into tools. Tools remain **atomic LLM-invokable operations**; skills are **composable capability packs** (tools + prompts + policy + metadata) with **import adapters** for external skill formats (e.g. Cursor `SKILL.md`). See architecture §7.1.8.

**Plugin catalogs (Phase P-Ext):** **Done** (2026-06-02) — protocols, `bootstrap_catalogs()`, EP fixture, conflict policy, scaffold CLI, 13/13 `ToolPlugin`, 3/3 `SkillPlugin`. See Appendix I · [EXTENSION_AUTHOR_GUIDE.md](EXTENSION_AUTHOR_GUIDE.md).

### 0.6 When Tier-1 (Nexus) changes are required

**Default (Tier-2 agent):** register + `AgentContract` + UAEP steps — **no** edits to `intergrax/runtime/`.

**Extend Tier-1** only when the need is **reusable across many future agents**, not one product:

| Situation | Action |
|-----------|--------|
| New agent with existing capabilities, memory, graph, HITL, sandbox | **Tier-2 only** — `agents/<slug>/` |
| New capability id, prompts, domain tools | **Tier-2** (+ Tier-0 adapter if new external integration) |
| New orchestration primitive (e.g. new graph node type, new lifecycle state) | **Tier-1** — must serve multiple agents; update canon §42 first |
| New platform concern (new store, queue, notification channel) | **Tier-0** — `intergrax/` shared module |
| Agent-specific product wiring (routes, env, which agents active) | **Tier-3** — `applications/<product>/` |
| One agent needs special-case branch in `NexusLoop` | **Anti-pattern** — refactor to contract/metadata or Tier-0 |
| Reusable workflow pack (tools + instructions + policy) for many agents | **Tier-0 Skill** — `intergrax/skills/` + `SkillManifest`; agent composes `skill_ids` |
| Import external skill pack (Cursor, internal markdown) | **Tier-0** — `SkillImporter` adapter; MUST validate against `SkillManifest` |
| Context budget / trim policy for all agents | **Tier-1** — `ContextBudgetPolicy` in `ContextManager` (**Done**, R-Context) |
| Delegated child run (subagent semantics) | **Tier-1** — `ExecutionGraph` delegation node + isolated memory namespace (**Done**, R-Delegate) |

If the answer to “will another agent need this?” is **no**, it does not belong in Nexus.

### 0.2 Four tiers

| Tier | Folder | Role | Analogy |
|------|--------|------|---------|
| **Tier-0** | `intergrax/` | Platform — LLM, storage, queues, logging, adapters | Kernel drivers |
| **Tier-1** | `intergrax/runtime/` | **Nexus Agent OS** — orchestration, lifecycle, trace, memory, HITL | Operating system |
| **Tier-2** | `agents/` | Reusable agent capabilities — domain logic, prompts, tools | Applications |
| **Tier-3** | `applications/` | Self-contained execution environments — env, Docker, wiring, routes | Deployable product/lab host |

### 0.3 Execution path

```text
HTTP / CLI / Worker
    → Tier-3 Application (optional)
    → UnifiedTaskRunner
    → NexusLoop (Tier-1)
    → AgentEngine / UAEP
    → Tier-2 Agent (get_steps → run_step → decide_after_step)
    → ToolRuntime / MemoryView / Validation
    → Trace + RuntimeEvents + TaskResult
```

### 0.4 Agent OS rule

New agents integrate via **`AgentRegistry.register()`** — never by editing `NexusLoop`, `GraphExecutor`, or task lifecycle code.

### 0.5 Maturity dashboard

| Scope | Score | Blocks new agent? | Notes |
|-------|-------|-------------------|-------|
| **Harness GA (functional)** | **Done** | **No** | L certified; scaffold + lab + gate |
| **Harness quality (Phase Q)** | **Done** (Wave 9) | **No** | Appendix C — gate **417 passed** (2026-06-01) |
| **Harness hardening (Phase Q+)** | **Done** (2026-06-01) | **No** | Appendix D — typing, legacy, monoliths, getattr audit |
| **Harness AI alignment (Phase R)** | **Done** (MVP, 2026-06-01) | **No** | Appendix E — Skill Library, context, delegation, policy |
| Skill Library (Tier-0) | **MVP Done** | **No** (extend catalog) | R-Skill.1–10; importers + trace events |
| Context engineering API | **Done** | No | R-Context.1–4; `CONTEXT_*` events |
| Graph delegation (subagent model) | **Done** | No | R-Delegate.1–4; `DelegationSpec` + memory namespace |
| RuntimePolicyBundle narrative | **Done** (Tier-3 + Nexus RuntimeConfig/Context + ToolRuntime) | No | R-Policy.1–2; `runtime_config_bridge.py`, `tool_policy_resolution.py` |
| Canon §1–41 (tiers, Nexus, graph, repo split) | **~98%** (post-Q+) | No | Q+-N, Q+-L, Q+-T Done |
| §42 Unified Execution Runtime | **~99%** (post-Q+-T) | No | UAEP Protocol, harness getattr audit |
| Laboratory workflow | **~99%** (post-Q+-O) | No | Metrics parity, planner observability |
| Agent OS certification (Phase L) | **Done** | No | Appendix A |
| **Harness environment GA (Phase S)** | **Done** (2026-06-01) | No (blocks K.1/K.2 only) | S-Ops + S-H + S-Doc; gate green |
| **Harness cleanliness (Phase T)** | **Done** (2026-06-01) | No | T-Ops + T-H; gate green |
| **Harness production hardening (Phase U)** | **Done** | No | U-Sec + U-Pol + U-Con + U-Typ + U-Arch + U-CI; Appendix G |
| **Harness architecture hardening (Phase V)** | **Done** | No (harness-only) | Phase V-REM closeout complete (2026-06-05) |
| **Phase V runtime remediation (V-REM)** | **Done** | No (harness-only) | All V-REM rows closed; §6.1z queue closed |
| **Operational harness L3 (Phase W-OPS)** | **Done** (code) | No (harness-only) | Ops sign-off: `release_cycles.json` or `W_OPS_RELEASE_CYCLES>=2` + `phase_w_ops_evidence.py --enforce` |
| **Application environment profile (Phase H-APP)** | **Done** (2026-06-03) | No (harness-only) | [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7 — 43 tasks; memory bridge gap → [Phase MEM](#phase-mem--memory-platform-completion) |
| **Memory platform (Phase MEM)** | **Done** (~3,5/5 post-closeout) | No (harness-only) | Memory platform **48/48** — gate **571** |
| Regression gate | **571 passed** | No | Must stay green after each harness PR |

---



## 1. Plan Objective



Transform Intergrax into an **internal agent experimentation laboratory** (§2, §35) aligned with the canonical architecture:



```text

hypothesis → capability → contract → registration → Nexus → trace → evaluation → decision

```



**Success metric:** time from idea to first running experiment **< 1 hour**.

**Capability model:** Integration → Tool → **Skill** → Agent (Harness AI alignment). Skill Library **MVP Done** — see §0, Phase R, Appendix E, architecture §7.1.8, [SKILLS.md](SKILLS.md).



**Current alignment** (synced with §0.5, 2026-06-05):

| Scope | Score | Notes |
|-------|-------|-------|
| Architecture §1–41 (tiers, Nexus, graph, repo split) | **~98%** | Phases A–O, N, Q+ complete |
| §42 Unified Execution Runtime | **~99%** | UAEP Protocol, hook parity, planner split (`step_planner/`), harness getattr audit |
| Laboratory workflow (inspect, decide) | **~99%** | D.1–D.5, metrics parity, debug API |
| Harness quality (Phase Q) | **Done** | Appendix C — gate **417** at Phase Q close |
| Harness hardening (Phase Q+) | **Done** | Appendix D — typing, monolith splits, zero grandfathered `getattr` |
| Harness AI alignment (Phase R MVP) | **Done** | Appendix E — Skill Library, context, delegation, policy bundle |
| Regression gate | **571 passed** | `pytest -m gate`; also `scripts/check_harness_no_getattr.py` |
| Harness environment GA (Phase S) | **Done** (2026-06-01) | S-H.* + S-Ops + S-Doc |
| Harness cleanliness (Phase T) | **Done** (2026-06-01) | T-Ops + T-H |
| Harness production hardening (Phase U) | **Done** | Appendix G audit → U.* (U-Leg residual) |
| Harness architecture hardening (Phase V) | **Done** | Phase V-REM runtime enforcement complete (2026-06-05) |
| Phase V runtime remediation (V-REM) | **Done** | 10/10 tasks — capability graph, lifecycle routing, prompt governance, V-SEC wiring, EvalRunner gate |
| Model & modality plane (Phase W-ML) | **Done** (2026-06-02) | Vision/speech profiles, tools, remote adapters, `harness.vision_qa` — canon §7.1.9 |
| **Harness completion backlog** | **Done** (2026-06-02) | §4.1 — U-Leg, typing/CI, platform skills, research UAEP parity |
| **Plugin catalogs (Phase P-Ext)** | **Done** (2026-06-02) | [Phase P-Ext](#phase-p-ext--plugin-catalogs-integrations-tools-skills) · [P-Ext.6 paydown](#p-ext6--production-closure-paydown) · Appendix I |
| **Application environment (Phase H-APP)** | **Done** (2026-06-03) | [Phase H-APP](#phase-h-app--tier-3-application-environment-full-configurability) · 43 tasks from application-layer audit |
| **Developer authoring UX (Phase DX)** | **Done** (2026-06-02) | [Phase DX](#phase-dx--developer-authoring-experience-fast-environment--agent-builds) · **47/47 Done** — [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **Agents & applications conformance (Phase AA)** | **Platform Done** (2026-06-02) | [Phase AA](#phase-aa--agents--applications-conformance-scaffold-docs-deploy) · platform **Done**; domain **Deferred** — [§6.3a](#63a-business-backlog-register-consolidated) |
| **Memory platform (Phase MEM)** | **Done** (2026-06-02) | [Phase MEM](#phase-mem--memory-platform-completion) · **48/48** |
| Product agents (Phase K) | **Deferred** | K.1/K.2 — end of priority list |
| Tier-3 product applications | **Deferred** | New apps / product routes — after harness backlog |



---



## 2. Map: Architecture → Implementation Status



| Section | Requirement | Status | Location |

|---------|-------------|--------|----------|

| §5.1 Four tiers | Tier-0..3 model | **Done** | architecture doc + `agent_kit/tiers.py` |

| §5.2 Reuse Tier-0 | No redundant platform | **Doc + process** | §5.2, §8.8, §39.8 |

| §9.1 Nexus Loop | Global orchestration | **Done** | `nexus_loop.py` |

| §9.2 Local agent loop | Bounded UAEP steps | **Done** | Echo, Research, Legal `thin_steps` / `dynamic_steps` |

| §12–16 Contracts / Registry | AgentContract, capabilities | **Done** | `intergrax/contracts/`, `runtime/registry/` |

| §22 ToolRuntime | Policy gateway | **Done** | `tool_runtime.py`, `ToolAccessPolicy` |

| §7.1.8 Skill Library | Composable capability packs, importers | **MVP Done** | `intergrax/skills/` · `docs/SKILLS.md` |
| §7.1.9 Model & Modality Plane | Vision (YOLO/ONNX/…), speech, classical ML, HF roles | **Done** | `docs/MODALITY.md` · Phase W-ML |

| §5.3 Harness AI alignment | Scaffold, skill/tool split, context, delegation | **Done** | `scaffold/new_skill.py`, `intergrax/skills/`, Appendix E |

| §23 Task lifecycle | States + trace + typed contract | **Done** | `task/`, `task_contract.py`, `TaskContextAssemblyOptions`, `task_metadata_bridge.py` |

| §24–25 Execution graph | Multi-agent | **Done** | `execution/`, `GraphExecutor` |

| §29 Validation | Nexus + agent | **Done** | `NexusValidationEngine` |

| §31 Retry | Runtime-managed | **Done** | `RetryEngine` |

| §33 Observability | Trace + events | **Done** (lab scope) | Trace + runtime events + metrics export (`B.08`–`B.11`); OTel provider beta |

| §42 Execution runtime | UAEP, hooks, governance, tool gateway | **Done (~99%)** | UAEP Protocol, planner events, tool gateway, harness getattr audit |
| §19 Debug / experiments | CLI, API, registry, cost | **Done** | D.1–D.5 ✅ |

| §7.4 Repo split | agents / applications | **Done** | `agents/legal`, `applications/legal_application` |
| §7.1 Integration Library | Catalog + contracts + providers | **Done** (beta) | Phase M core + M.6 P1/P2/P3; on-demand slugs only |

| §19 Debug surface | CLI / API | **Done** | D.1 CLI + D.2 API ✅ |

| §32 HITL | Approval / reject / escalate | **Done** | F.3 + `runtime/human/` |

| §26 Long-running tasks | Checkpoint / resume | **Done** (baseline) | Scheduler + partial results API + UAEP mid-step (`B.01`, `B.02`) |
| §18 Slack / Teams | Interaction adapters | **Done** (product baseline) | Outbound + `POST /v1/interactions/intake` on lab/legal/research/poc; verifier via env |
| §27 Memory model | Bounded task / agent memory | **Done** | I.1–I.5: TaskMemory, MemoryView, SharedTaskContext, handoff, ContextManager v2 |
| §42.9 Pause / Resume | `RuntimeCheckpoint` | **Done** (baseline) | HITL pause + full snapshot (`plan_snapshot`, `graph_snapshot`, UAEP cursor) |
| §41 Unified entry | Single run lifecycle | **Done** (lab scope) | `UnifiedTaskRunner` on all Tier-3 hosts; legacy `AgentEngine` opt-out removed |

| §20–21 Shadow / Sandbox | Isolated exec | **Done** | F.1 ShadowWorkspace + F.2 SandboxRuntime ✅ |



---



## 3. Implementation Phases



### Phase A — Foundation Stabilization



| # | Deliverable | Status |

|---|-------------|--------|

| A.1 | Unified run lifecycle | **Done** |

| A.2 | Task trace persistence | **Done** |

| A.3 | NexusLoop production path | **Done** |

| A.4 | EvalRunner integration (NexusEvalRunner + gate coverage) | **Done** |

| A.4.1 | NexusEvalRunner integration tests + inclusion in gate | **Done** (2026-06-05 — `tests/integration/eval/test_nexus_eval_runner.py`) |

| A.5-min | Pre-P4.2 regression gate | **Done** |

| A.5 | Full regression suite (Legal E2E, all steps) | **Deferred** |

| A.6 | Shim cleanup | **Done** | Removed `applications/legal_agent/`; docs + duplicate `legal_application/tests/` cleaned |



**A.5-min completion criteria (gate before P4.2):**



```bash

uv run pytest tests/ -m gate -q

```



| Test area | File |

|-----------|------|

| TaskLifecycle transitions | `tests/unit/runtime/task/test_task_lifecycle.py` |

| TaskTraceEmitter + RuntimeEventBus | `tests/unit/runtime/task/test_task_trace_event_bus.py` |

| trace_bridge mapping | `tests/unit/runtime/events/test_trace_bridge.py` |

| AgentEngine.run / run_with_result | `tests/integration/agents/test_agent_engine_*.py` |

| NexusLoop + Echo (lifecycle + events) | `tests/integration/runtime/test_nexus_loop_echo.py` |

| GraphExecutor sequential stub | `tests/integration/runtime/test_graph_executor_stub.py` |



**Infrastructure fixes included:** circular import (`tool_runtime` ↔ `runtime_state`), missing `RegistryToolExecutor`, `ExecutionGraph` pydantic imports, lazy pipeline imports in `tests/conftest.py`.



**Explicitly not required before P4.2:** Legal through NexusLoop, full Nexus step matrix, E2E with real LLM.



---



### Phase B — Extended Nexus



| # | Deliverable | Status |

|---|-------------|--------|

| B.1–B.7 | Classifier, planner, validation, retry, tool policy, composer | **Done** |



---



### Phase C — Multi-Agent Readiness



| # | Deliverable | Status |

|---|-------------|--------|

| C.1–C.6 | ExecutionGraph, GraphExecutor, ContextManager, Research pipeline | **Done** |



---



### Phase D — Observability and Experiments



**Goal:** §19, §35 — laboratory tooling (not SaaS UI).



| # | Deliverable | Status | Notes |

|---|-------------|--------|-------|

| D.0 | §42 P4.1 Event Bus wiring | **Done** | `RuntimeEventBus`, `trace_bridge`, NexusLoop |

| D.1 | Debug CLI | **Done** | `python -m intergrax.debug tasks list\|show\|trace` |

| D.2 | Minimal debug API | **Done** | FastAPI `GET /debug/tasks` on trace store |

| D.3 | Experiment registry | **Done** | SQLite registry; CLI + `GET/POST /debug/experiments` |

| D.4 | Notebook templates | **Done** | `notebooks/experiments/`, `experiments/workflow.py` |

| D.5 | Cost in trace | **Done** | `AgentExecutionResult.cost` from LLM usage / runtime stats |



---



### Phase E — Legal Agent Refactoring (parallel)



| # | Deliverable | Status |

|---|-------------|--------|

| E.1 | Thin sequential Legal — domain steps as UAEP `AgentStep` list | **Done** |

| E.2 | ToolRuntime via gateway (no direct Nexus step imports in bridge) | **Done** (P4.4) |

| E.3 | Governance on UAEP decision path | **Done** (P4.3) |

| E.4 | Thin dynamic Legal (`LegalDynamicPipeline` routing) | **Done** |



**E.4 delivered (2026-05-27):** `agents/legal/uaep/dynamic_steps.py` — 5 UAEP macro-steps (setup → tool plan → route → waves → finalize); `legal_execution_loop` phase functions extracted. Gate: 34 tests.



**E.1 delivered (2026-05-27):** `agents/legal/uaep/thin_steps.py` — 8 UAEP steps (setup → finalize); `LegalAnalysisPipeline` reuses same runners; dynamic mode keeps single pipeline boundary. Gate: 33 tests.



---



### Phase F — Advanced / On-Demand

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| F.1 | ShadowWorkspace | **Done** | `runtime/workspace/`; UAEP + NexusLoop integration |
| F.2 | SandboxRuntime | **Done** | `runtime/sandbox/`; `sandbox.exec` via BoundToolGateway |
| F.3 | Advanced HITL (reject/escalation store) | **Done** | `runtime/human/` store + NexusLoop reject/escalate |
| F.4 | Long-running tasks / Slack-Teams | **Done (partial)** | Checkpoints ✅; Slack/Teams = notification stub only |

| F.5 | Typed task contract | **Done** | `TaskExecutionOptions`, `TaskRuntimeState`, `TaskResultSummary`, bridge |

Long-running **full** §26 (scheduler, UAEP mid-step) and Slack/Teams **full** §18 — see Phase G–H below.



---



### Phase P4 — §42 Unified Execution Runtime



| Step | Deliverable | Status |

|------|-------------|--------|

| P4.1 | Event bus + trace bridge | **Done** |

| P4.2 | UAEP in AgentEngine | **Done** |

| P4.3 | Governance (interrupt, HITL) | **Done** |

| P4.4 | Tool gateway unification | **Done** |

| P4.5 | Agent migration (Echo, Research, Legal) | **Done** |



**P4.5 delivered (2026-05-27):** `uaep_pipeline.py`; Research, Summary, Legal agents on UAEP (`get_steps` / `run_step` / `decide_after_step`); integration tests + NexusLoop research. Gate: 31 tests.



**P4.4 delivered (2026-05-27):** `RuntimeToolGateway`, `ToolRuntime.invoke_request`, Legal bridge via `ToolRequest`; UAEP `BoundToolGateway`. Gate: 25 tests.



**P4.3 delivered (2026-05-27):** `runtime/interrupts/`, `runtime/human/`, policy in UAEP + NexusLoop.



---

### Phase G — §42 Runtime Convergence

**Goal:** Close largest gaps vs §42.9, §42.10, §42.24, §42.40 (evolve, not rewrite).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| G.1 | `RuntimeCheckpoint` contract | **Done** | §42.9.2 | Plan + graph node states + UAEP step index |
| G.2 | UAEP mid-execution resume | **Done** | §42.9.3 | Skip re-run paused step on resume |
| G.3 | HITL middleware hooks | **Done** | §42.10 | `BEFORE/AFTER_HUMAN_APPROVAL` in NexusLoop |
| G.4 | `HumanRequest` v2 fields | **Done** | §42.10.1 | Typed urgency, deadline propagation, timeout stub |
| G.5 | RuntimeEvent-first observability | **Done** | §42.24 | `RuntimeEventPersistence` + `store.py` (`open_runtime_event_store`, env `INTERGRAX_RUNTIME_EVENTS_DB` only) |
| G.6 | Debug API: HITL + checkpoints | **Done** | §19 | Pluggable stores; events/checkpoints/HITL resume |
| G.7 | Graph failure recovery | **Done** | §42.40, §30 | Skip completed nodes; checkpoint on graph fail |
| G.8 | Cooperative cancellation | **Done** | §42.26 | Cancel propagation through graph / UAEP |

---

### Phase H — Interaction Surfaces (§18)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| H.1 | Outbound webhook delivery | **Done** | §18 | Pluggable delivery + formatters; HTTP opt-in |
| H.2 | `InteractionAdapter` protocol | **Done** | §18 | Inbound → normalized `Task` |
| H.3 | Slack inbound lab path | **Done** | §18 | Debug API intake + signature stub |
| H.4 | HITL notification templates | **Done** | §42.10 | Reusable template + `notify_hitl_pause`; Slack/Teams formatters |
| H.5 | Teams parity | **Done** | §18 | Activity parser + HMAC verifier + debug intake tests |
| H.6 | Organization Worker demo | **Done** | §38 | E2E lab: intake → HITL → notification → resume |

---

### Phase I — Memory & Context (§27–28)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| I.1 | `TaskMemory` store | **Done** | §27 | Contract + coordinator; `store.py` (`open_task_memory_store`, env `INTERGRAX_TASK_MEMORY_DB` only) |
| I.2 | `MemoryView` gateway | **Done** | §42.35 | `PolicyScopedMemoryView` + UAEP wiring + `MEMORY_*` events |
| I.3 | `SharedTaskContext` | **Done** | §42.14 | Contract + `ContextManager` + graph merge + memory bridge |
| I.4 | Agent handoff | **Done** | §42.15 | `AgentHandoff` + `HandoffCoordinator` + graph path + `HANDOFF_*` events |
| I.5 | ContextManager v2 | **Done** | §28 | Provenance + summary tiers + `TaskContextAssemblyOptions` on `TaskExecutionOptions.context` |

---

### Phase J — Unified Execution Entry (§41)

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| J.1 | NexusLoop default in apps | **Done** | §41 | Legal + Research: `UnifiedTaskRunner` only (legacy `AgentEngine` removed, B.14) |
| J.2 | RunService → UnifiedTaskRunner | **Done** | §41 | `NexusTaskExecutionAdapter` + `CreateRunRequest.payload` → Task |
| J.3 | Worker queue Task v2 | **Done** | §41 | `QueuedNexusExecutionAdapter`, `nexus.task.v2` Celery handler, checkpoint resume |
| J.4 | Long-running scheduler | **Done** | §26 | `LongRunningScheduler`, delayed resume + HITL timeout enforcement |
| J.5 | Partial results API | **Done** | §26 | `GET /debug/tasks/{id}/progress`, `TASK_PROGRESS` events, notification template |

---

### Phase K — Hardening & Reference Agents

**Harness prerequisites:** L, Q+, R, S, T, U, and §4.1 **Done** — platform is ready **when** product chooses to start Band 3 (§6.3).

**Scheduling rule (2026-06-02):** K.1/K.2 are **end-of-plan** (§4.0 Band 3, §6.3). Completing harness phases does **not** auto-schedule business agents as the next implementation task.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| K.1 | Problem Radar prototype | **Deferred** | §36 | Wave-1 scaffold frozen (`agents/problem_radar/`); resume after harness backlog |
| K.2 | Vendor Discovery prototype | **Deferred** | §37 | After Phase S; product decision |
| K.3 | Policy engine facade | **Done** | §42.11 | `PolicyEngine` + `coerce_replay_policy_engine`; `ExecutionGuard` uses `evaluate_replay` (2026-05-27) |
| K.4 | Dual `AgentDecision` cleanup | **Done** | §42.7 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` alias (TYP-06, 2026-06-02) |
| K.5 | ChatAgent / legacy removal | **Done** | §39 | Production paths use Nexus only; `check_production_chat_agent_imports.py` gate (2026-05-27) |
| K.6 | A.5 full Legal E2E gate | **Deferred** | — | Real LLM; not blocking lab — product/CI decision |

---

### Phase L — Agent OS Certification

**Directive:** L1 certification recorded in Appendix A. K.1/K.2 are **Phase K product work** — **last** in the plan (§6.3), not concurrent with harness bands 1–2.  
**Agent workflow:** [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md)

| # | Deliverable | Status | Req | Notes |
|---|-------------|--------|-----|-------|
| L.1 | UAEP-first agent scaffold | **Done** | R2 | `python -m intergrax.scaffold new-agent` |
| L.2 | Agent creation guide | **Done** | R2 | Single canonical how-to |
| L.3 | Lab application (Tier-3) | **Done** | R1 | `applications/lab_application/` |
| L.4 | Reference technical agents | **Done** | R5 | Echo + `agents/lab/mock_agents.py` |
| L.5 | Agent OS acceptance suite | **Done** | R1 | `tests/acceptance/agent_os/` (+ `05b` mid-step UAEP) |
| L.6 | Runtime independence verification | **Done** | R5 | Register + run without Nexus edits |
| L.7 | Application composition verification | **Done** | R5 | Agents ≠ applications |
| L.8 | Certification checklist | **Done** | R1 | Appendix A (this file) |
| L.9 | **Sign-off exercise** | **Done** | — | `agents/signoff_probe/` — Appendix A record |

**Acceptance tests (L.5):**

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
```

| # | Scenario | Test |
|---|----------|------|
| 1 | Single agent | `test_acceptance_01_single_agent_execution` |
| 2 | Sequential multi-agent | `test_acceptance_02_sequential_multi_agent` |
| 3 | Parallel multi-agent | `test_acceptance_03_parallel_multi_agent` |
| 4 | HITL approve/resume | `test_acceptance_04_human_approval_flow` |
| 5 | Checkpoint recovery | `test_acceptance_05_checkpoint_recovery` |
| 6 | Retry / alternate agent | `test_acceptance_06_retry_flow` |
| 7 | Partial results | `test_acceptance_07_partial_results` |
| 8 | Memory / shared context | `test_acceptance_08_memory_handoff` |
| 9 | Sandbox tools | `test_acceptance_09_sandbox_tool_execution` |
| 10 | Shadow workspace | `test_acceptance_10_shadow_workspace` |

---

### Phase M — Integration Library (Tier-0 Catalog)

**Canon:** §7.1.1–§7.1.5  
**Goal:** One discoverable integration catalog so platform teams ship adapters and agent teams compose them in Tier-3 — without duplicating Redis/Postgres/Slack clients per agent.

**Principle:** evolve existing modules (`queueing/`, `distributed/`, `websearch/`, …) into catalog providers; do not fork parallel stacks.

**Out of scope:** `intergrax/llm_adapters/` — LLM providers are **not** part of the Integration Library (§7.1.2).

### Phase M-LLM — LLM Adapter Layer (Tier-0)

**Canon:** §5.2.2 · **Doc:** [LLM_ADAPTERS.md](LLM_ADAPTERS.md)  
**Goal:** One `LLMAdapter` contract, lazy registry, streaming + native tools + structured output across commercial and self-hosted providers.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-LLM.1 | Shared `_shared/` (messages, tools, retry, conformance) | **Done** | 2026-05-30 |
| M-LLM.2 | Seven core providers hardened | **Done** | OpenAI, Claude, Azure, Gemini, Mistral, Bedrock, Ollama |
| M-LLM.3 | Groq + vLLM (OpenAI-compatible) | **Done** | `openai_compat_providers.py` |
| M-LLM.4 | Bedrock Converse + tools + stream | **Done** | `INTERGRAX_BEDROCK_USE_CONVERSE`, `converse_stream` |
| M-LLM.5 | Conformance tests in CI gate | **Done** | `tests/unit/llm_adapters/` |
| M-LLM.6 | `LLM_ADAPTERS.md` + README section | **Done** | 19 providers |
| M-LLM.7 | OpenAI-compat expansion + Vertex + `LLMProfile` | **Done** | Together, Fireworks, OpenRouter, DeepSeek, xAI, llama.cpp, Cohere, Vertex |
| M-LLM.8 | Optional network smoke workflow | **Done** | Weekly schedule + `workflow_dispatch` |
| M-LLM.9 | Azure refactor (Chat Completions base) | **Done** | Thin `AzureOpenAIChatAdapter` |
| M-LLM.10 | Production hardening | **Done** | Metrics, builtin conformance, `LLMProfile`, Bedrock tools stream, `cohere_native`, `azure_ai_inference` |
| M-LLM.11 | Production ops layer | **Done** | OTLP/Prometheus routes, tenant metrics, rate limit + circuit breaker, secrets map, PR guard, extended network smoke |
| M-LLM.12 | Nexus + governance wiring | **Done** | `llm_tenant_scope`, runtime metrics plugin, `INTERGRAX_LLM_TENANT_MAX_TOKENS` quota |
| M-LLM.13 | Observability + secrets + distributed limits | **Done** | Pushgateway, `LLM_ADAPTERS.md` § Observability, Vault loader, Redis rate limit, governance warn |

### Phase M-RAG — RAG Engine (Tier-0)

**Canon:** §5.2.2 · **Architecture:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) (RAG stack)  
**Goal:** One configurable retrieval path for `rag.retrieve`, Nexus `ContextBuilder`, and ingest — no duplicate dense-only shortcuts; parsers/chunkers/rerankers selected via profile and Integration Library slugs (never hardcoded to a single vendor).

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M-RAG.1 | `RagProfile` + env (`INTERGRAX_RAG_*`) | **Done** | `intergrax/rag/profiles/rag_profile.py` |
| M-RAG.2 | `RetrievalService` (route → retrieve → rerank) | **Done** | `intergrax/rag/retrieval/`; wired to `rag.retrieve` + Nexus |
| M-RAG.3 | Adaptive `QueryRouter` (fast / standard / deep) | **Done** | `intergrax/rag/routing/query_router.py` |
| M-RAG.4 | `IngestPipeline` + configurable chunking strategy | **Done** | `intergrax/rag/ingest/`; `rag.ingest_document` |
| M-RAG.5 | Contextual chunk enricher (optional LLM) | **Done** | `INTERGRAX_RAG_CONTEXTUAL_ENRICH`; injected `LLMAdapter` |
| M-RAG.6 | Query expansion (`deterministic` / `llm`) | **Done** | `MultiQueryRetriever` + `query_expander.py` |
| M-RAG.7 | Evaluation metrics (`recall@k`, MRR) | **Done** | `intergrax/rag/evaluation/metrics.py` |
| M-RAG.8 | `create_default_rag_stack()` bootstrap | **Done** | `intergrax/rag/bootstrap/rag_stack_bootstrap.py` |
| M-RAG.9 | Tool/Nexus wiring (`retrieval_service`, profile on `ToolWiringContext`) | **Done** | `RuntimeConfig.retrieval_service` |
| M-RAG.10 | Native sparse / BM25 in vector backends | **Done** | `LexicalHybridSupport` + `query_hybrid` on InMemory/Qdrant/Weaviate; RRF fusion |
| M-RAG.11 | RAG eval CI gate + golden datasets | **Done** | `tests/fixtures/rag_golden/`, `golden_harness.py`, `rag-guard.yml` |
| M-RAG.12 | GraphRAG (`GraphStore` contract) | **Done** (beta) | `graph/` + `graph_rag` retriever + heuristic indexer |
| M-RAG.13 | Platform agentic retrieval loop (budgeted) | **Done** | `AgenticRetrievalLoop` on deep tier + `INTERGRAX_RAG_AGENTIC_*` |
| M-RAG.14 | Qdrant native sparse vectors + RRF fusion | **Done** | `INTERGRAX_RAG_QDRANT_SPARSE`, `bm25_sparse_encoder.py` |
| M-RAG.15 | Weaviate native `query.hybrid` | **Done** | Live client + `INTERGRAX_RAG_WEAVIATE_NATIVE_HYBRID`; fallback to in-memory |
| M-RAG.16 | LLM graph indexer (optional adapter) | **Done** | `INTERGRAX_RAG_GRAPH_INDEXER_MODE=llm\|heuristic_then_llm` |
| M-RAG.17 | LLM agentic query refinement | **Done** | `INTERGRAX_RAG_AGENTIC_QUERY_MODE=llm` + injected `LLMAdapter` |
| M-RAG.18 | Neo4j GraphRAG backend | **Done** | `Neo4jRagGraphStore` + `INTERGRAX_RAG_GRAPH_STORE=neo4j` |
| M-RAG.19 | SPLADE / learned sparse encoder | **Done** | `sparse_encoder.py`; `INTERGRAX_RAG_SPARSE_ENCODER=splade` (optional `fastembed`) |
| M-RAG.20 | Weaviate prod hardening | **Done** | `schema.py` — migration, multi-tenant, metadata filters |
| M-RAG.21 | Extended golden datasets | **Done** | graph_rag, multi_hop, agentic scenarios in `retrieval_cases.json` |
| M-RAG.22 | RAG observability metrics | **Done** | `INTERGRAX_RAG_METRICS_ENABLED`, runtime plugin on `TASK_COMPLETED` |

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| M.0 | Integration backlog + categories approved | **Done** | Canon §7.1.3 catalog table |
| M.1 | Scaffold `intergrax/integrations/` package | **Done** | `contracts/`, `registry/`, `_shared/`, `providers/` |
| M.2 | Category contracts (P0 set) | **Done** | 7 P0 contracts + re-exports for queueing/notifications/interactions |
| M.3 | `IntegrationRegistry` + `IntegrationProfile` | **Done** | `catalog.register_integration`, `resolve`, env/mapping profile |
| M.4 | P0 providers — wrap existing | **Done** | See **M.4 provider tracker** below |
| M.5 | Provider conformance test harness | **Done** | `tests/unit/integrations/`, `_shared/conformance.py` |
| M.6 | P1 providers (on demand) | **Done** (beta) | postgresql, mysql, jira, confluence, prometheus, ms365_graph, aws, azure, gcp — see M.4/M.6 trackers |
| M.6 P2 | Extended providers (on demand) | **Done** (beta) | All P2/P3 slugs shipped 2026-05-30 — see **M.6 P2 tracker**; `_shared/p2/` + thin `providers/<slug>/` shells |
| M.7 | Agent Creation Guide § integrations | **Done** | Appendix E — capabilities/tools vs `IntegrationProfile` / `wire_lab_integrations()` |
| M.8 | Lab `IntegrationProfile` example | **Done** | `applications/lab_application/` — `wire_lab_integrations()` + `log` provider |

**M.4 delivery workflow (one provider per iteration):**

1. Implement `providers/<category>/<slug>/` (wrap legacy module — no fork).
2. Register via `register_<slug>_integration()` + `register_default_integrations()`.
3. Unit tests under `tests/unit/integrations/providers/`.
4. Add `providers/<slug>/USAGE.md` — English usage guide (factory + `IntegrationProfile` + API invoke example). Extend `scripts/generate_integration_usage_docs.py` and run `uv run python scripts/generate_integration_usage_docs.py`.
5. Update canon §7.1.3 status + this tracker + migration map row.
6. Next slug in priority order.

#### M.4 provider tracker

| Slug | Category | Status | Package | Legacy source |
|------|----------|--------|---------|---------------|
| `redis` | key_value_cache | **Done** | `providers/redis/` — `create_redis_integration()` (KV, idempotency, rate limit, semaphore, rerank) |
| `sqlite` | relational_store | **Done** | `providers/sqlite/` — `create_sqlite_integration()` (trace, events, checkpoints, HITL, …) |
| `kafka` | message_bus | **Done** (+ adopcja) | `providers/kafka/` — runtime transport delegates here |
| `celery` | message_bus | **Done** | `providers/celery/` — `create_celery_integration()` (inject `app` or broker/backend env) |
| `google_cse` | search_provider | **Done** | `providers/google_cse/` — `create_google_cse_integration()` (legacy `GOOGLE_CSE_*` env) |
| `bing` | search_provider | **Done** | `providers/bing/` — `create_bing_integration()` (legacy `BING_SEARCH_V7_API_KEY`) |
| `slack` | notification + interaction | **Done** (+ adopcja) | `providers/slack/` — runtime wiring delegates here |
| `teams` | notification + interaction | **Done** (+ adopcja) | `providers/teams/` — runtime wiring delegates here |
| `webhook` | notification_channel | **Done** (+ adopcja) | `providers/webhook/` — generic HTTP + `GenericJsonPayloadFormatter` |
| `lab_json` | interaction_surface | **Done** (+ adopcja) | `providers/lab_json/` — lab intake; runtime channel ``lab`` |
| `rabbitmq` | message_bus | **Done** (+ adopcja) | `providers/rabbitmq/` — `create_rabbitmq_integration()` (requires `kv_store`) |
| `log` | notification_channel | **Done** (+ adopcja) | `providers/log/` — wraps `LoggingNotificationAdapter`; lab profile default |
| `postgresql` | relational_store | **Done** (beta) | `providers/postgresql/` — `RelationalStore` via psycopg3; only `opens.py` connects |
| `mysql` | relational_store | **Done** (beta) | `providers/mysql/` — `RelationalStore` via pymysql; only `opens.py` connects |
| `databricks` | relational_store | **Done** (beta) | `providers/databricks/` — SQL Warehouse via databricks-sql-connector; only `opens.py` connects |
| `mongodb` | document_store | **Done** (beta) | `providers/mongodb/` — flexible JSON `DocumentStore`; PyMongo only in `opens.py` |
| `pinecone` | vector_store | **Done** (beta) | `providers/pinecone/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `qdrant` | vector_store | **Done** (beta) | `providers/qdrant/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `chroma` | vector_store | **Done** (beta) | `providers/chroma/` — catalog bridge to `rag/`; SDK only in `opens.py` |
| `s3` | object_storage | **Done** (beta) | `providers/s3/` — put/get/delete/presigned_url; boto3 only in `opens.py` |
| `jira` | issue_tracker | **Done** (beta) | `providers/jira/` — REST v3; only `opens.py` creates httpx client |
| `confluence` | wiki_knowledge | **Done** (beta) | `providers/confluence/` — REST wiki; only `opens.py` creates httpx client |
| `prometheus` | observability_backend | **Done** (beta) | `providers/prometheus/` — PromQL query API; only `opens.py` creates httpx client |
| `elasticsearch` | observability_backend | **Done** (beta) | `providers/elasticsearch/` — `_search` aggregations; only `opens.py` creates httpx client |
| `ms365_graph` | collaboration_suite | **Done** (beta) | `providers/ms365_graph/` — Graph mail/calendar/directory; only `opens.py` creates httpx client |
| `cassandra` | document_store | **Done** (beta) | `providers/cassandra/` — CQL get/put/delete/query; only `opens.py` creates driver session |
| `aws` | cloud_platform | **Done** (beta) | `providers/aws/` — IAM/STS auth + category defaults; only `opens.py` creates boto3 session |
| `azure` | cloud_platform | **Done** (beta) | `providers/azure/` — MI / service principal + category defaults; only `opens.py` creates credential |
| `gcp` | cloud_platform | **Done** (beta) | `providers/gcp/` — ADC / service account + category defaults; only `opens.py` creates credentials |

#### M.6 P2 — Extended provider tracker (canon §7.1.3 P2)

Deliver after M.6 P1 priorities unless a product app blocks on a specific slug. Each P2 provider follows the same workflow as M.4 (contract → `providers/<slug>/` → tests → catalog row).

| Slug | Category | Status | Rationale / notes |
|------|----------|--------|-------------------|
| **`cassandra`** | **document_store** | **Done** (beta) | High-volume log / event retention; CQL driver via `opens.py` single entry |
| **`elasticsearch`** | **observability_backend** | **Done** (beta) | Log search / aggregations (`_search` + Lucene `query_string` via ObservabilityBackend); complements `prometheus` |
| **`databricks`** | **relational_store** | **Done** (beta) | Lakehouse SQL Warehouse; PAT via `opens.py`; `execute` / `fetch_all` for analytics agents |
| **`mongodb`** | **document_store** | **Done** (beta) | Flexible JSON documents; partition-scoped get/put/delete/query via PyMongo |
| **`pinecone`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/pinecone_vector_store.py` |
| **`qdrant`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/qdrant_vector_store.py` |
| **`chroma`** | **vector_store** | **Done** (beta) | Catalog bridge to `rag/vectorstore/providers/chroma_vector_store.py` |
| **`s3`** | **object_storage** | **Done** (beta) | AWS S3 blobs; boto3 only in `opens.py` |
| **`azure_blob`** | **object_storage** | **Done** (beta) | Azure Blob; `providers/azure_blob/` + shared `CatalogObjectStorage` |
| **`gcs`** | **object_storage** | **Done** (beta) | GCS via `_shared/p2/gcs_blob.py` |
| **`dynamodb`** | **document_store** | **Done** (beta) | boto3 table facade in `_shared/p2/factories.py` |
| **`oracle`** / **`mssql`** / **`azure_sql`** / **`cloud_sql`** | **relational_store** | **Done** (beta) | SQL adapters via `_shared/p2/clients.py` |
| **`memcached`** / **`elasticache`** | **key_value_cache** | **Done** (beta) | pymemcache / Redis-compatible duck client |
| **`sqs`** / **`service_bus`** / **`pubsub`** | **message_bus** | **Done** (beta) | `CloudTaskQueue` over cloud SDK facades |
| **`email_smtp`** | **notification_channel** | **Done** (beta) | stdlib SMTP in factory open path |
| **`otel`** | **observability_backend** | **Done** (beta) | OTLP-oriented metrics facade (beta noop exporter default) |
| **`github`** / **`linear`** / **`azure_devops`** | **issue_tracker** | **Done** (beta) | REST issue trackers via httpx |
| **`notion`** / **`sharepoint`** | **wiki_knowledge** | **Done** (beta) | REST wiki adapters |
| **`google_workspace`** | **collaboration_suite** | **Done** (beta) | Gmail / Calendar REST |
| **`brave`** / **`serpapi`** | **search_provider** | **Done** (beta) | Shared `_shared/rest_search.py` hit mappers |
| **`playwright`** | **browser_automation** | **Done** (beta) | `contracts/browser_automation.py` + Playwright factory |

#### M.6 P3 / M.7 — Harness integrations (Done beta, 2026-05-29)

**M.11 harness defaults (Done beta):** default `notify_channel` injection from lab wiring (`task_defaults.py`, `LAB_HARNESS` enricher on lab run + interaction intake).

**M.10 harness Tier A (Done beta):** composite observability (`observability_backends` + role-based `resolve_observability_backend`), HITL→PagerDuty runtime path (`create_harness_notification_adapter`, `LAB_HARNESS`), integration tests.

**M.9 harness depth (Done beta):** full adapters (LangSmith, OpenSearch, Vespa, GitLab, PagerDuty, Braintrust), tools (`gitlab.create_issue`, `pagerduty.trigger_incident`, `braintrust.log_eval`), `slash_command`, lab harness profile, CI harness-smoke job. Catalog: **99**.

**M.8 harness gap (Done beta):** +14 slugs via `_shared/p4/factories.py`

**M.7 harness (Done beta):** +21 slugs via `_shared/p3/factories.py` (incl. **sentry**).

#### M.7 — Document parser catalog bridge (2026-05-30)

Vendor document parsing moved from `intergrax/rag/document_loaders/parsers/` into `integrations/providers/document_parser/`. RAG uses `CatalogDocumentParser` + `resolve_document_parser()`.

**Wave 2 (2026-05-30):** `openpyxl`, `whisper`, `yt_dlp`; `cohere_rerank` / `jina_rerank`; Bing/Google CSE implementations under `integrations/.../web_client.py` (websearch re-exports); `ParserPipeline` ingestion trace; tool `rag.ingest_document`; `IntegrationProfile.legal_product()` / `research_product()` / `lab()` with `document_parser=docling`; lab `GET /v1/lab/integrations/docling/health`.

**Wave 3 (2026-05-30):** `reddit`, `google_places` search providers; Chroma/Qdrant/Pinecone SDK in `integrations/.../rag_store.py` (RAG shims); runtime SQLite delivery ledger via `sqlite/opens`; `rag.ingest_document` env flags for legal/research; parser trace export to Langfuse/Sentry.

**Wave 4 (2026-05-30):** `inmemory` vector store SDK in `integrations/.../inmemory/rag_store.py`; SQLite observability via `integration_profile_wiring` + `wire_nexus_observability(integration_profile=…)` with default-path fallback; parser pipeline spans appended to `RunTraceWriter` (`parser_trace_span.py`); vendor import governance script + CI gate; Phase Q scaffold defaults (`IntegrationProfile`, `ToolProfile` with `websearch.read_url`).

**Wave 5 (2026-05-30):** Phase P wave 3 tools (`websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces`); full `IntegrationProfile` on legal/research products; Weaviate/Milvus `rag_store.py`; Redis SDK cleanup in distributed/rag shims; governance extended to `agents/` + `rag/`; parser trace export on `RunTraceWriter.finalize_run`; Phase Q scaffold wave 2 (lab vs product ToolProfile, env profile override).

| Slug | Status | Notes |
|------|--------|-------|
| `docling` | **Done** (beta) | local + server; `opens.py` only Docling/httpx imports |
| `pymupdf` | **Done** (beta) | PDF + optional Tesseract OCR |
| `unstructured` | **Done** (beta) | HTML loader |
| `python_docx` | **Done** (beta) | Word `.docx` |
| `openpyxl` | **Done** (beta) | Excel/CSV via pandas |
| `whisper` | **Done** (beta) | Audio + YouTube (uses yt_dlp opens) |
| `yt_dlp` | **Done** (beta) | YouTube audio/video download |
| `cohere_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `jina_rerank` | **Done** (beta) | RAG rerank via integration resolver |
| `reddit` | **Done** (beta) | Reddit OAuth2 search |
| `google_places` | **Done** (beta) | Google Places text search |

#### M.6 P3 — Legacy backlog note (superseded)

Slugs below were **already in** `IntegrationSlug` unless marked *proposed*. Prioritize when a product app blocks; otherwise deliver after P2.

| Priority | Slug(s) | Category | Why agents/apps need it |
|----------|---------|----------|-------------------------|
| **High** | `mongodb` | document_store | Session state, flexible agent memory, JSON artifacts at scale |
| **High** | `pinecone`, `qdrant`, `chroma` | vector_store | Production RAG — unify Tier-3 `IntegrationProfile.vector_store` with existing `rag/` backends |
| **High** | `s3`, `azure_blob`, `gcs` | object_storage | Checkpoint blobs, sandbox exports, document ingestion pipelines |
| **High** | `email_smtp` | notification_channel | HITL and report delivery without Slack/Teams |
| **Medium** | `notion`, `sharepoint` | wiki_knowledge | Runbooks and internal docs (Confluence complement) |
| **Medium** | `github`, `linear` | issue_tracker | Dev workflows, PR/issue-aware agents |
| **Medium** | `google_workspace` | collaboration_suite | Google-tenant mail/calendar parity with MS365 |
| **Medium** | `otel` | observability_backend | Export runtime traces/metrics to Grafana Cloud, Datadog, etc. |
| **Medium** | `playwright` | browser_automation | JS-heavy sites, authenticated flows beyond static fetch |
| **Medium** | `brave`, `serpapi` | search_provider | Rate-limit / vendor diversity for research agents |
| **Low** | `oracle`, `mssql`, `azure_sql`, `cloud_sql` | relational_store | Enterprise DB deployments |
| **Low** | `dynamodb`, `memcached`, `elasticache` | document_store / KV | AWS-native persistence tiers |
| **Done (beta)** | `weaviate`, `milvus`, `snowflake`, `vault` | vector_store / relational_store / secrets | `integrations/providers/vector_store/weaviate/`, `vector_store/milvus/`, `relational_store/snowflake/`, `secrets/vault/` |

**Vector-store rule (pinecone / qdrant / chroma):** implementation **stays** in `intergrax/rag/vectorstore/`. Integration Library adds `providers/<slug>/` as a **thin registry adapter**: `opens.py` is the only module that imports vendor SDK; `bundle.create_*_vector_store()` delegates to the existing RAG provider. Tier-3 selects slug via `IntegrationProfile.vector_store`; RAG pipeline code unchanged.

**MongoDB — suggested implementation sketch (greenfield):**

```text
providers/mongodb/
├── config.py                   # INTERGRAX_MONGODB_URI, DATABASE, COLLECTION_PREFIX
├── client.py                   # PyMongo collection wrapper (internal — no driver outside opens.py)
├── adapter.py                  # MongoDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs MongoClient
├── bundle.py                   # create_mongodb_document_store()
├── register.py
└── tests/                      # mocked collection; integration_live optional
```

**Prerequisite (mongodb):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Partition key maps to MongoDB `_id` or compound `{tenant_id, key}` index.

**Pinecone — suggested implementation sketch (catalog bridge):**

```text
providers/pinecone/
├── config.py                   # INTERGRAX_PINECONE_API_KEY, INDEX, NAMESPACE, ENV
├── adapter.py                  # Thin VectorStore registry facade (delegates to rag/)
├── opens.py                    # ONLY place that imports pinecone SDK / builds Pinecone client
├── bundle.py                   # create_pinecone_vector_store() → rag PineconeVectorStore
├── register.py
└── tests/                      # mocked delegate; guard: no pinecone import outside opens.py
```

**Prerequisite (pinecone):** `contracts/vector_store.py` — **Done** (re-exports `rag/vectorstore/contracts/vector_store.py`). Registered under `IntegrationCategory.VECTOR_STORE`.

**Cassandra — suggested implementation sketch (greenfield):**

```text
contracts/document_store.py     # DocumentStore — get/put/delete/query by partition key
providers/cassandra/
├── config.py                   # INTERGRAX_CASSANDRA_CONTACT_POINTS, KEYSPACE, USER, PASSWORD
├── client.py                   # CQL session (internal — no direct driver import outside opens.py)
├── adapter.py                  # CassandraDocumentStore implements DocumentStore
├── opens.py                    # ONLY place that constructs cassandra driver session
├── bundle.py                   # create_cassandra_integration()
├── register.py
└── tests/                      # testcontainers or mocked session; integration_live optional
```

**Prerequisite (cassandra):** `DocumentStore` contract — **Done** (`contracts/document_store.py`). Runtime event / trace backends remain SQLite-first until an explicit adoption milestone names Cassandra as a target store.

**Elasticsearch — suggested implementation sketch (greenfield):**

```text
providers/elasticsearch/
├── config.py                   # INTERGRAX_ELASTICSEARCH_URL, USER, PASSWORD, INDEX_PREFIX
├── client.py                   # REST search client (internal — no httpx outside opens.py)
├── adapter.py                  # ElasticsearchObservabilityBackend implements ObservabilityBackend
├── opens.py                    # ONLY place that constructs httpx client / ES connection
├── bundle.py                   # create_elasticsearch_observability_backend()
├── register.py
└── tests/                      # mocked _search / ES|QL responses; integration_live optional
```

**Contract note:** start with `ObservabilityBackend` (`query_instant` / `query_range`) mapped to ES\|QL or index-scoped aggregations where feasible; add optional `search_logs(query, *, limit)` on the contract in a follow-up if PromQL-shaped methods prove awkward for log-only clusters.

**Databricks — suggested implementation sketch (greenfield):**

```text
providers/databricks/
├── config.py                   # INTERGRAX_DATABRICKS_HOST, HTTP_PATH, TOKEN, CATALOG, SCHEMA
├── client.py                   # SQL connection wrapper (internal — no driver import outside opens.py)
├── adapter.py                  # DatabricksRelationalStore implements RelationalStore
├── opens.py                    # ONLY place that opens databricks-sql-connector / REST session
├── bundle.py                   # create_databricks_relational_store()
├── register.py
└── tests/                      # mocked cursor / Statement Execution API; integration_live optional
```

**Contract note:** implements existing `RelationalStore` (`connect`, `execute`, `fetch_all`, `close`). Optional `tenant_schema` maps to Unity Catalog ``catalog.schema`` (default schema per connection). Not a replacement for domain runtime stores (SQLite-first) — target is analytics / reporting agents and batch read paths.


1. Create package skeleton:

```text
intergrax/integrations/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   └── base.py              # IntegrationMetadata, HealthStatus, IntegrationError
├── registry/
│   ├── __init__.py
│   ├── catalog.py           # slug → provider entry (lazy import)
│   └── factory.py           # resolve(category, slug | env)
├── _shared/
│   ├── config.py            # pydantic BaseIntegrationConfig
│   └── health.py
└── providers/
    └── .gitkeep
```

2. Add `IntegrationMetadata` dataclass: `slug`, `categories`, `status` (`stable` | `beta` | `deprecated`), `env_prefix`.

3. Register package in `pyproject.toml` / existing import paths (no new top-level dependency unless provider-specific).

#### M.2 — Category contracts (step-by-step)

For each category in §7.1.2, implement a **minimal** Protocol in `integrations/contracts/`:

| Contract | Minimum methods | Notes |
|----------|-----------------|-------|
| `RelationalStore` | `connect()`, `execute()`, `fetch_all()`, `close()` | **Done** — `contracts/relational_store.py`; sqlite/postgresql/mysql/**databricks** (beta) |
| `KeyValueCache` | `get`, `set`, `delete`, `set_if_absent` | Maps to existing `IdempotencyStore` / Redis helpers |
| `MessageBus` | `enqueue`, `get_status`, `get_result` | Re-export / implement `queueing.contracts.TaskQueue` |
| `SearchProvider` | `search(query, *, limit)` → `SearchResult[]` | Align with `websearch/providers/base.py` |
| `NotificationChannel` | `notify(message)` | Align with `runtime/notifications/adapter_contract.py` |
| `InteractionSurface` | `can_handle`, `to_inbound`, `channel` | Align with `runtime/interactions/adapter_contract.py` |
| `CloudPlatform` | `slug`, `default_region`, `resolve(category)`, `health` | **Done** — `contracts/cloud_platform.py`; **`aws`**, **`azure`**, **`gcp`** providers (beta) |
| `CollaborationSuite` | `get_message`, `list_messages`, `send_mail`, `list_calendar_events`, `get_user` | **Done** — `contracts/collaboration_suite.py`; `ms365_graph` provider |
| `DocumentStore` | `get`, `put`, `delete`, `query` (partition-scoped) | **Done** — `contracts/document_store.py`; `cassandra`, **`mongodb`** (beta) providers |
| `VectorStore` | `add_documents`, `query`, `delete`, … | **Done** — `contracts/vector_store.py` re-exports `rag/`; **`pinecone`**, **`qdrant`**, **`chroma`** (beta) |
| `ObjectStorage` | `put`, `get`, `delete`, `presigned_url` | **Done** — `contracts/object_storage.py`; **`s3`** (beta) |
| `IssueTracker` | `get_issue`, `add_comment`, `search_issues` | **Done** — `contracts/issue_tracker.py`; `jira` provider |
| `WikiKnowledge` | `get_page`, `search_pages` | **Done** — `contracts/wiki_knowledge.py`; `confluence` provider |
| `ObservabilityBackend` | `query_instant`, `query_range` | **Done** — `contracts/observability_backend.py`; `prometheus`, **`elasticsearch`** (beta) providers |

**Rule:** if a contract already exists elsewhere, **re-export or inherit** — do not define a third variant.

#### M.3 — IntegrationRegistry (step-by-step)

1. `catalog.py` — static registry:

```python
INTEGRATION_ENTRIES: dict[str, IntegrationEntry] = {
    "sqlite": IntegrationEntry(categories=("relational_store",), factory="..."),
    "redis": IntegrationEntry(categories=("key_value_cache",), factory="..."),
    # ...
}
```

2. `factory.py`:

```python
def resolve(category: str, slug: str | None = None, *, config: Mapping[str, Any] | None = None) -> Any:
    """slug defaults from env INTERGRAX_INTEGRATION_<CATEGORY> or IntegrationProfile."""
```

3. `IntegrationProfile` — pydantic model loaded from env or YAML in Tier-3 `settings.py`.

4. `health_check_all(profile)` — optional startup probe for lab/production.

#### M.4 — Adding a new provider (checklist for implementers)

Copy this checklist into every `providers/<slug>/README.md`:

```text
[ ] 1. Pick category contract(s) from integrations/contracts/
[ ] 2. Create providers/<slug>/ with adapter.py, config.py, config.example.yaml
[ ] 3. Implement contract — no business logic, no Nexus imports
[ ] 4. Register slug in registry/catalog.py
[ ] 5. Add unit tests with fakes or testcontainers (default: no live vendor)
[ ] 6. Optional: pytest -m integration_live with CI secrets
[ ] 7. Wire in one Tier-3 application as reference (lab or product)
[ ] 8. Update canon §7.1.3 status column
```

**Example — wrapping existing Redis idempotency store:**

```text
providers/redis/
├── adapter.py       # RedisKeyValueCache implements KeyValueCache
├── config.py        # REDIS_URL, REDIS_PREFIX
└── tests/
    └── test_redis_cache.py  # fakeredis or mock
```

Delegate to `intergrax/distributed/providers/redis_idempotency_store.py` internally.

**Example — new Jira provider (greenfield):**

```text
providers/jira/
├── adapter.py       # JiraIssueTracker implements IssueTracker
├── config.py        # JIRA_BASE_URL, JIRA_API_TOKEN
├── config.example.yaml
├── README.md
└── tests/
    └── test_jira_issue_tracker.py  # responses mocked from fixtures/
```

Expose agent tools via Tier-0 tool registration (`jira.get_issue`, `jira.create_comment`) — ToolRuntime policy in Tier-1.

#### M.4b — Cloud platform providers (aws / azure / gcp)

Each platform folder exposes **one auth entry point** and registers sub-service slugs:

```text
providers/aws/
├── adapter.py       # CloudPlatform: IAM profile, region, resolve("object_storage") → S3
├── config.py        # AWS_REGION, AWS_PROFILE, AWS_ROLE_ARN
├── services/        # thin wrappers delegating to category contracts
│   ├── s3.py
│   ├── sqs.py
│   └── dynamodb.py
└── tests/

providers/azure/
├── adapter.py       # Managed identity + service principal
├── services/
│   ├── blob.py
│   └── service_bus.py
└── ...

providers/gcp/
├── adapter.py       # ADC + service account
├── services/
│   ├── gcs.py
│   └── pubsub.py
└── ...
```

**Checklist:** implement infrastructure services (S3, SQS, Blob, GCS, Pub/Sub, …) only. LLM wiring stays in `intergrax/llm_adapters/` — do not register Bedrock, Azure OpenAI, or Vertex under `integrations/`.

#### M.5 — Migration map (legacy → catalog)

| Legacy location | Target slug | Action |
|-----------------|-------------|--------|
| `distributed/providers/redis_kv_store.py` (+ siblings) | `redis` | **Done** — single entry `integrations/providers/key_value_cache/redis/create_redis_integration()` |
| `queueing/providers/kafka/` | `kafka` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/kafka/` |
| `queueing/providers/celery/` | `celery` | **Done** — `integrations/providers/message_bus/celery/create_celery_integration()` |
| `queueing/providers/rabbitmq/` | `rabbitmq` | **Done** — runtime transport + tests delegate to `integrations/providers/message_bus/rabbitmq/` |
| `websearch/providers/google_cse_provider.py` | `google_cse` | **Done** — `integrations/providers/search_provider/google_cse/create_google_cse_integration()` |
| `websearch/providers/bing_provider.py` | `bing` | **Done** — `integrations/providers/search_provider/bing/create_bing_integration()` |
| `runtime/notifications/adapters/webhook_adapter.py` | `webhook` | **Done** — `integrations/providers/notification_channel/webhook/create_webhook_integration()` |
| `runtime/notifications/adapters/logging_adapter.py` | `log` | **Done** — `integrations/providers/notification_channel/log/`; factory delegates |
| `runtime/notifications/adapters/` | `slack`, `teams` | **Done** — runtime delegates |
| `runtime/interactions/adapters/lab_json_adapter.py` | `lab_json` | **Done** — `integrations/providers/interaction_surface/lab_json/create_lab_json_integration()` |
| `runtime/*/stores/sqlite_*.py` (+ store openers) | `sqlite` | **Done** — single entry `integrations/providers/relational_store/sqlite/create_sqlite_integration()` |
| (new) | `postgresql` | **Done** — `integrations/providers/relational_store/postgresql/`; **only** `opens.py` calls `psycopg.connect` |
| (new) | `mysql` | **Done** — `integrations/providers/relational_store/mysql/`; **only** `opens.py` calls `pymysql.connect` |
| (new) | `jira` | **Done** — `integrations/providers/issue_tracker/jira/`; **only** `opens.py` creates httpx client |
| (new) | `confluence` | **Done** — `integrations/providers/wiki_knowledge/confluence/`; **only** `opens.py` creates httpx client |
| (new) | `prometheus` | **Done** — `integrations/providers/observability_backend/prometheus/`; **only** `opens.py` creates httpx client |
| (new) | `ms365_graph` | **Done** — `integrations/providers/collaboration_suite/ms365_graph/`; **only** `opens.py` creates httpx client + token fetch |
| (new) | `cassandra` | **Done** — `integrations/providers/document_store/cassandra/`; **only** `opens.py` creates driver session |
| (new) | `aws` | **Done** — `integrations/providers/cloud_platform/aws/`; **only** `opens.py` creates boto3 session |
| (new) | `azure` | **Done** — `integrations/providers/cloud_platform/azure/`; **only** `opens.py` creates Azure credential |
| (new) | `gcp` | **Done** — `integrations/providers/cloud_platform/gcp/`; **only** `opens.py` creates Google credentials |
| (new) | `elasticsearch` | **Done** — `integrations/providers/observability_backend/elasticsearch/`; **only** `opens.py` creates httpx client |
| (new) | `databricks` | **Done** — `integrations/providers/relational_store/databricks/`; **only** `opens.py` calls `databricks.sql.connect` |
| (new) | `mongodb` | **Done** — `integrations/providers/document_store/mongodb/`; **only** `opens.py` calls `pymongo.MongoClient` |
| `rag/vectorstore/providers/pinecone_*` | `pinecone` | **Done** — `providers/pinecone/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/qdrant_*` | `qdrant` | **Done** — `providers/qdrant/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/providers/chroma_*` | `chroma` | **Done** — `providers/chroma/` catalog bridge; RAG impl stays in `rag/` |
| `rag/vectorstore/bootstrap/vectorstore_bootstrap.py` | integration catalog | **Done** — `create_default_vectorstore_manager()` resolves via `IntegrationProfile.vector_store` |
| `rag/vectorstore/providers/*` | other vector slugs | Catalog entry only until bridge provider ships |

**Not migrated to `integrations/`:** `intergrax/llm_adapters/` — LLM providers are a separate Tier-0 concern (§7.1.2 out-of-scope table).

#### M.6 — Testing strategy

| Layer | Location | Marker |
|-------|----------|--------|
| Contract unit tests | `tests/unit/integrations/` | default gate |
| Provider unit tests | `intergrax/integrations/providers/<slug>/tests/` | default gate |
| Registry / factory | `tests/unit/integrations/test_registry.py` | gate |
| Live vendor smoke | `tests/integration/integrations/` | `integration_live` (CI optional) |

Conformance test pattern: given a fake backend, assert all Protocol methods behave consistently (including error types).

#### M.7 — Agent Creation Guide (Appendix E)

Documented in [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) Appendix E:

- Agents: `capabilities`, `allowed_tools`, `ToolRequest` — no integration slug imports.
- Applications: `IntegrationProfile`, `wire_lab_integrations()`, `register_default_integrations()`.
- Env: `INTERGRAX_INTEGRATION_<CATEGORY>` overrides.

Tier-3 composition example (product factory):

```python
# applications/my_app/factory.py
from intergrax.integrations import (
    IntegrationCategory,
    IntegrationProfile,
    register_default_integrations,
)

def create_app():
    register_default_integrations()
    profile = IntegrationProfile.lab()  # or build_profile_from_env()

    cloud = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)       # aws | azure | gcp
    db = profile.resolve(IntegrationCategory.RELATIONAL_STORE)        # sqlite | postgresql
    cache = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
    storage = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
    notifier = profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)
    # wire into Nexus factories, not into agents/
```

Agents reference capabilities in `AgentContract` (e.g. `allowed_tools=["websearch.query"]`) — not integration slugs.

#### M.8 — Definition of done (Phase M incremental)

Each provider PR is **done** when:

1. Contract conformance tests pass.
2. Registered in `catalog.py` with metadata.
3. `providers/<slug>/USAGE.md` — English: env vars, factory call, `IntegrationProfile` resolve, minimal invoke example.
4. At least one Tier-3 app or lab factory can select it via `IntegrationProfile`.
5. No new direct vendor imports added under `agents/`.

Szablony utrzymywane przez `scripts/generate_integration_usage_docs.py` (regeneracja po dodaniu providera).

---

### Phase N — Application Environment & Deploy Scaffold (Tier-3)

**Canon:** §7.4.8–§7.4.10  
**Goal:** From agent POC to **docker-pushable** dedicated lab/product host in minutes — same ergonomics as `new-agent`, with isolated `.env.example`, manifest, and Docker.

**Prerequisite:** Phase L complete; Phase M.3 (`IntegrationProfile`) available.

**Delivery rule (this phase):** One step per iteration — implement → summarize → update docs → present next step (see **§6.1**).

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| N.0 | Architecture & plan documented | **Done** | §7.4.8–§7.4.10 | This section + runtime canon (2026-05-30) |
| N.1 | `ApplicationManifest` + `AgentBinding` models | **Done** | §7.4.10 | `intergrax/applications/contracts/manifest.py` |
| N.2 | Manifest conformance harness + unit tests | **Done** | §7.4.10 | `intergrax/applications/_shared/wiring.py` |
| N.2.1 | Unified agent initialization (builders / factories / context) | **Done** | §7.4.10 | `ApplicationBuildContext`, `build_application_registry`; lab + legal migrated |
| N.2.2 | Strongly typed `AgentBinding.mount(AgentClass, factory=...)` | **Done** | §7.4.10 | `type[Agent]` + callable factory; `deserialize()` for scaffold strings only |
| N.3 | `python -m intergrax.scaffold new-application` (profile `lab`) | **Done** | §7.4.8 | `new_application.py`, `agent_catalog.py`, `cli.py`; lab templates + smoke |
| N.4 | Scaffold profile `product` (fastapi_core skeleton) | **Done** | §7.4.8 | `new_application_product.py`; FastAPI Core + auth stub + `/health`; `--agents` list |
| N.5 | Docker templates under `applications/<app>/docker/` | **Done** | §7.4.8 | Dockerfile + `.dockerignore` + `docker-compose.yml` + `build-docker.sh` / `.bat`; monorepo-root context |
| N.6 | Reference app `poc_template_application` (committed example) | **Done** | §7.4.8 | `applications/poc_template_application/`; README three-command quickstart; gate smoke |
| N.7 | Backfill `.env.example` on existing apps | **Done** | §7.4.8 | `lab_application`, `legal_application`, `research_application`, `poc_template_application` |
| N.8 | `AGENT_CREATION_GUIDE.md` Step 4E (dedicated application) | **Done** | — | Step 4E + Appendix F cross-links; gate doc test |
| N.9 | Acceptance `test_scaffold_application` (gate) | **Done** | — | `test_scaffold_acceptance.py` — lab/product E2E, CLI profiles, docker scripts |
| N.10 | Optional `new-stack` (agent + application in one CLI) | **Done** | — | `intergrax/scaffold/new_stack.py`; gate test in `test_scaffold_acceptance.py` |

#### N — Step-by-step implementation sequence

Execute **strictly in order**; do not skip ahead without completing acceptance for the current step.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | N.1 | Add `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` (Pydantic) | Unit tests pass; no scaffold yet |
| 2 | N.2 | Add `applications/_shared/conformance.py` (or mirror integrations pattern) | Manifest load + minimal registry build test |
| 3 | N.3 | Implement `new_application.py` + `lab` profile templates | `uv run python -m intergrax.scaffold new-application test_lab --profile lab --agents echo` creates tree; smoke test green |
| 4 | N.3b | Wire `build_parser()` subcommand; post-create hints (uvicorn, pytest, docker) | CLI prints next commands; gate test added (N.9 partial) |
| 5 | N.5 | Add Docker/docker-compose + build scripts to scaffold | `applications/<app>/docker/build-docker.sh` (or `.bat`) builds image from repo root |
| 6 | N.6 | Commit `applications/poc_template_application/` from scaffold | README three-command quickstart verified |
| 7 | N.7 | Add per-app `.env.example` to legal, research, lab | Vars match each `settings.py`; no secrets committed |
| 8 | N.4 | Add `product` profile to scaffold | **Done** — `test_scaffold_product_application.py`; FastAPI Core + `/health` |
| 9 | N.8 | Update agent guide Step 4E | **Done** — scaffold lab/product, Docker scripts, three-command quickstart |
| 10 | N.9 | Full acceptance + `pytest -m gate` | **Done** — runtime E2E + `test_scaffold_acceptance.py` |

**Scaffold CLI (target interface):**

```bash
python -m intergrax.scaffold new-application my_lab \
  --profile lab \
  --agents echo,my_agent \
  --port 8091 \
  --prefix /v1/my_lab
```

**Out of scope for Phase N:**

- Separate `pyproject.toml` per application (stay monorepo + `pythonpath`)
- Auto-discovery of agents in `lab_application` (keep explicit wiring; manifest is declarative, not magic)
- Runtime sandbox (Tier-1) changes — only document distinction (§7.4.9)

#### Tier-3 application layer — readiness (2026-05-30)

**Status: ready** to generate new applications via scaffold. Checklist: [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md).

| Track | ID | Status | Notes |
|-------|-----|--------|-------|
| Engine | N.1–N.2.2 | **Done** | manifest, `build_application_registry`, conformance |
| Scaffold | N.3–N.4, N.10 | **Done** | `lab` + `product` + `new-stack` |
| Deploy | N.5–N.7 | **Done** | Docker scripts, `BUILD_AND_DEPLOY`, `.env.example` |
| Docs + gate | N.8–N.9 | **Done** | Step 4E, `test_scaffold_acceptance`, legal/research/lab manifest tests |
| Hardening | A.1–A.2 | **Done** | `test_legal_manifest_wiring`, tool_wiring assertions on scaffold |
| Optional CI Docker | B.1 | **Done** | `tests/integration/applications/test_poc_template_docker_build.py` (not in gate) |
| Product maturity | — | **Reference** | `legal_application` chat routes — extend scaffold `product` manually |

**Verify:**

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest -m gate -q
```

---

### Phase O — Tool Library & Unified Tool Model (Tier-0)

**Canon:** §7.1.6–§7.1.7, §22, §42.12  
**Goal:** Ship a reusable **Tool Library** catalog (mirror Integration Library) and migrate legacy pipeline flags (`use_rag`, `use_websearch`) to explicit catalog tools.

**Prerequisite:** Phase M.3 (`IntegrationProfile`) available; tool engine (`ToolRegistry`, `RuntimeToolInvoker`) exists.

**Catalog reference:** [`TOOLS.md`](TOOLS.md)

**Delivery rule:** One domain or migration slice per iteration — implement → gate → update `TOOLS.md` → next step.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| O.0 | Architecture & catalog documented | **Done** | §7.1.6–§7.1.7, §22 | Runtime canon + `TOOLS.md` + this section (2026-05-30) |
| O.1 | Extended `ToolContract` | **Done** | §22 | `ToolRiskLevel`, `ToolRetryPolicy`, metadata fields; invoker timeout/retry/trace (2026-05-30) |
| O.2 | `ToolCatalog` + `ToolProfile` + `ToolWiringContext` | **Done** | §7.1.6 | `intergrax/tools/registry/`; `build_registry_from_profile`; RuntimeConfig wiring (2026-05-30) |
| O.3 | Context tools: `rag.retrieve`, `websearch.query` | **Done** | §7.1.7, §22.1 | `providers/rag/`, `providers/websearch/` (2026-05-30) |
| O.4 | Reference domain: `jira.*` tools | **Done** | §7.1.6 | `get_issue`, `add_comment`, `search_tasks` over `IssueTracker` (2026-05-30) |
| O.4b | Catalog domain bundles: `confluence.*`, `notify.send`, observability, `sandbox.exec` | **Done** | §7.1.6 | All first-party catalog tools registered (2026-05-30) |
| O.5 | **Unified tool model migration** | **Done** | §7.1.7, §22.2 | `tool_ids` on plans; RagStep/WebsearchStep → catalog shims (2026-05-30) |
| O.6 | Schema exporters (OpenAI + MCP) | **Done** | §7.1.6 | `tools/exporters/`; MCP catalog mount on lab/poc_template (2026-05-30) |
| O.7 | Migrate legacy `ToolBase` → `ToolContract` | **Done** | §5.2.2 | `ChatAgent` → registry; `tools_base` deprecated (2026-05-30) |
| O.8 | `ToolProfile` in Tier-3 scaffold | **Done** | §7.4.8 | `tool_wiring.py` template; lab + poc_template reference (2026-05-30) |
| O.9 | Agent Creation Guide Appendix E update | **Done** | — | Unified model + ToolProfile examples (2026-05-30) |
| O.10 | Gate tests for catalog conformance | **Done** | — | `tests/unit/tools/providers/` — all catalog bundles (2026-05-30) |
| O.11 | Phase P wave 2 context tools: `websearch.read_url`, `confluence.search` | **Done** | §7.1.7, §22.1 | `providers/websearch/read_url_*`, confluence alias (2026-05-30) |
| O.12 | Phase P wave 3 tools: `websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces` | **Done** | §7.1.7, §22.1 | Extended `ObservabilityBackend.query_traces`, vector `list_collections` (2026-05-30) |

#### O — Step-by-step implementation sequence

Execute **strictly in order** for foundation (O.1–O.4); O.5–O.10 may overlap after O.4 reference tools land.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | O.1 | Extend `ToolContract` + update `RuntimeToolInvoker` for new fields | Unit tests pass; backward compatible defaults |
| 2 | O.2 | Add `tools/registry/catalog.py`, `profile.py`, `ToolWiringContext` dataclass | `register_default_tools()` no-op registry; profile enables subset |
| 3 | O.3 | Implement `providers/rag/` and `providers/websearch/` handlers | **Done** — `rag.retrieve`, `websearch.query` + tests |
| 4 | O.4 | Implement `providers/jira/` bundle (3 tools) | **Done** — conformance tests with mocked `IssueTracker` |
| 4b | O.4b | Implement remaining catalog bundles (`confluence`, `notify`, `observability`, `sandbox`) | **Done** — all tool_ids in `register_default_tools()` |
| 5 | O.5a | Add `tool_ids` to plan models; map legacy booleans → tool_ids | **Done** — `ToolInvocationPlan`, `LegalToolPlan` |
| 6 | O.5b | `RagStep` / `WebsearchStep` delegate to catalog tools | **Done** — `catalog_context.py` shim |
| 7 | O.5c | Update `LegalToolPlan` / engine plans to tool list | **Done** — bridge passes `tool_ids` |
| 8 | O.6 | MCP + OpenAI exporters from single catalog | **Done** — `tools/exporters/` |
| 9 | O.7 | Remove `ToolBase` usage from production paths | **Done** — `ChatAgent` uses registry `ToolRegistry` |
| 10 | O.8–O.10 | Scaffold, docs, gate | **Done** |

#### O.4 — Adding a new tool provider (checklist)

Copy into every `tools/providers/<domain>/USAGE.md`:

```text
[ ] 1. Define Input/Output Pydantic models (LLM-friendly field names)
[ ] 2. Implement ToolHandler — compose integration contract(s), no vendor SDK
[ ] 3. Build ToolContract per tool (description tuned for model selection)
[ ] 4. register_<domain>_tools(registry, ctx: ToolWiringContext)
[ ] 5. Register in tools/registry/catalog.py
[ ] 6. Unit tests with fakes (no live vendor in default gate)
[ ] 7. Wire in lab or poc_template via ToolProfile
[ ] 8. Update TOOLS.md status + this plan tracker
```

#### O.5 — Unified tool model (migration design)

**Problem:** Two parallel mechanisms — boolean plan flags dispatching pipeline steps vs `ToolRegistry` for function tools.

**Target:** One registry, one invoker, one policy surface.

```text
BEFORE (legacy):
  plan.use_rag=True        → RagStep (direct)
  plan.use_websearch=True  → WebsearchStep (direct)
  plan.use_tools=True      → ToolsStep → ToolRegistry

AFTER (canonical):
  plan.tool_ids=["rag.retrieve", "websearch.query", "jira.search_tasks"]
      → ToolRuntime.invoke_request (per id)
      → RuntimeToolInvoker → handler
      → integration / RAG module
```

**Compatibility (O.5a):** `ToolInvocationPlan.from_legacy(use_rag=…)` maps booleans to default tool_ids. Emit deprecation trace when legacy fields used.

**Context injection:** `rag.retrieve` and `websearch.query` set `injects_context=true`; invoker callback or Nexus hook merges bounded output into prompt assembly (§22.1).

**Out of scope for Phase O:**

- Domain-specific tools inside `agents/` (stay Tier-2; register via `ToolProvider` if reusable)
- Replacing `ToolsAgent` planner — it remains the LLM loop over `ToolRegistry`
- New integration categories (still Phase M / §5.2.4)

---

### Phase Q — Harness Quality & Consolidation (audit remediation)

**Source:** Harness implementation audit (2026-06-01) — Nexus, LLM, RAG, memory, observability, legacy, tests, docs.  
**Goal:** Remove bugs, technical debt, dead code, monoliths, dual-path semantics, and documentation drift **without** new business agents or integration catalog breadth.  
**Principle:** evolve, not rewrite · one deliverable per PR · gate green after each step · §0.6 (Tier-1 only when reusable).

**Out of scope for Phase Q:**

- Phase K.1/K.2 business agents (product)
- K.6 / B.15 Legal live LLM E2E (product/CI)
- New integration slugs (Phase M on-demand)
- New Tier-0 universal mechanisms (§5.2.4 human approval)
- Replacing `ToolsAgent` planner (Phase O out of scope)

**Delivery rule:** Same cadence as §6.1 — implement **one Q.* ID** → summarize → update this table + Appendix C status → next ID.

**Phase Q complete when:** All rows below **Done**; Appendix C 100% **Done** or **Won't fix** (documented); §0.5 Harness quality row **Done**; gate unchanged or increased.

---

#### Q.0 — Program governance

| # | Deliverable | Status | Tier | Audit ref | Done when |
|---|-------------|--------|------|-----------|-----------|
| Q.0.1 | Appendix C traceability matrix (audit → Q ID) | **Done** | Docs | C-all | Appendix C below; each row has owner phase |
| Q.0.2 | Phase Q execution order + PR sizing guide | **Done** | Docs | — | §4 + subsection **Q execution order** below |
| Q.0.3 | Gate policy: no Q PR without `pytest -m gate` | **Done** | CI | — | Documented in Q DoD; CI unchanged paths |

---

#### Phase Q-N — Nexus, loops, orchestration, error handling

**Components:** `intergrax/runtime/nexus/`, `intergrax/runtime/execution/`, `intergrax/runtime/hooks/`, `intergrax/runtime/interrupts/`, `intergrax/runtime/policy/`, `intergrax/runtime/nexus/retry/`, `intergrax/agents/agent_engine.py`, `intergrax/agents/uaep.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-N.1 | **Decompose `NexusLoop`** — extract HITL runner, long-running coordinator calls, event publisher, shadow/sandbox cleanup into dedicated modules; `NexusLoop` orchestrates only | **Done** | High | `nexus/orchestration/` (`graph_runner`, `task_events`, `lifecycle_bridge`, …) | `nexus_loop.py` ~586 lines; gate green |
| Q-N.2 | **Fix duplicate `_normalize_human_response`** — single call in `_handle_task_impl` | **Done** | High | `nexus_loop.py` L229–231 | Duplicate call removed (2026-06-01) |
| Q-N.3 | **Retry semantics document + facade** — one doc section: `RetryEngine` (graph/validation/alternate agent) vs `RuntimeConfig.max_run_retries` (LLM/tool in `RuntimeEngine`); optional `RetryCoordinator` delegating both | **Done** | High | `nexus/retry/`, `nexus/config.py`, architecture §31.1 | Doc merged; no duplicate retry without trace event |
| Q-N.4 | **Unify policy injection** — `PolicyEngine` only in public Nexus/UAEP APIs; remove `RuntimePolicyEngine` union from external signatures; `coerce_policy_engine` internal | **Done** | Medium | `nexus_loop.py`, `uaep.py`, factories | Type check / mypy clean on factories; gate green |
| Q-N.5 | **§42 hook parity — decision / interrupt / retry** — wire `BEFORE/AFTER_DECISION`, `BEFORE/AFTER_INTERRUPT`, `BEFORE/AFTER_RETRY` in NexusLoop + UAEP + `RetryEngine`; update `hooks/parity.py` to **WIRED** or **Won't fix** with canon amendment | **Done** | Medium | `hooks/`, `nexus_loop.py`, `uaep.py`, `retry_engine.py` | `parity.py` no NOT_WIRED for these six OR canon §42.20 amended + tests |
| Q-N.6 | **§42 hook parity — trace persist** — `BEFORE/AFTER_TRACE_PERSIST` **WIRED** at trace finalize path; `parity.py` → **WIRED** | **Done** | Medium | `hooks/`, `task_trace.py`, trace emitter | Parity test; hook invoked in integration test |
| Q-N.7 | **Rename Nexus context helpers module** — `runtime_steps/tools.py` → `runtime_steps/tool_context_helpers.py` (or merge into `tools_step.py`); update imports | **Done** | Low | `tool_context_helpers.py` + shim `tools.py` | Backward-compatible re-export (2026-06-01) |
| Q-N.8 | **Split `RuntimeConfig`** — `ModelRuntimeConfig`, `RetrievalRuntimeConfig`, `ToolsRuntimeConfig`, `PlanningRuntimeConfig`, `TraceRuntimeConfig`; composed `RuntimeConfig`; `validate()` cross-field | **Done** | High | `nexus/config.py` | Backward-compatible properties or migration shim one release; all factories updated |
| Q-N.9 | **Type `integration_profile`** — `IntegrationProfile` from `intergrax.integrations` on `RuntimeConfig` / wiring contexts | **Done** | Medium | `nexus/config.py`, `engine/runtime_context.py` | No `Optional[object]` for profile in public config |
| Q-N.10 | **`production_mode` lab default** — `lab_application` / scaffold sets `production_mode=False`; document in Step 4E | **Done** | Low | Tier-3 factories, `AGENT_CREATION_GUIDE.md` | `harness_production_mode()` in `applications/_shared/runtime_defaults.py` |
| Q-N.11 | **Graph callback typing** — `ExecutionNode` instead of `object` in `GraphExecutor` / NexusLoop node callbacks | **Done** | Low | `execution/graph_executor.py`, `nexus_loop.py` | Mypy/ruff on execution package |
| Q-N.12 | **Interrupt handler hygiene** — remove duplicate `InterruptType` import; add unit test for interrupt → policy path | **Done** | Low | `interrupts/handler.py` | Duplicate import removed (2026-06-01) |
| Q-N.13 | **`AgentEngine` static UAEP** — document or inject `event_bus` for `AgentEngine.run` static path; no silent missing events | **Done** | Low | `agents/agent_engine.py` | `_resolve_static_executor`; `tests/unit/agents/test_agent_engine_event_bus.py` |
| Q-N.14 | **Unit tests for `NexusLoop` helpers** — `_finish_task`, lifecycle transitions, HITL branch stubs (mock deps) | **Done** | High | `tests/unit/runtime/nexus/test_nexus_loop.py` | New file; ≥15 focused tests; marker `gate` |
| Q-N.15 | **`GraphExecutor` unit coverage** — failure recovery, skip completed, handoff edge (beyond stub integration) | **Done** | Medium | `tests/unit/runtime/execution/` | `test_graph_executor_coverage.py` + checkpoint skip in `test_runtime_checkpoint.py` |

---

#### Phase Q-L — LLM adapters

**Components:** `intergrax/llm_adapters/`, `docs/LLM_ADAPTERS.md`, governance plugin.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-L.1 | **Remove or complete `tracked_llm_call`** — if kept: `finally` calls `usage.end_call`; if removed: delete `tracked_call.py` + references | **Done** | Medium | `_shared/tracked_call.py` | File removed (unused) (2026-06-01) |
| Q-L.2 | **Public API surface** — re-export `LLMAdapter`, `LLMProvider`, `LLMAdapterRegistry`, `LLMProfile` from `llm_adapters/__init__.py` | **Done** | Low | `llm_adapters/__init__.py` | Public re-exports (2026-06-01) |
| Q-L.3 | **Provider catalog table in docs** — 19 rows: slug, adapter class, env vars, tools/stream/structured, native vs compat | **Done** | High | `docs/LLM_ADAPTERS.md` | Table matches `LLMProvider` enum + conformance list |
| Q-L.4 | **Fix `LLMProfile` docstring** — `max_retries` only via `options={}`; align examples in guide | **Done** | Low | `registry/profile.py`, tests | Example fixed (2026-06-01) |
| Q-L.5 | **Per-provider `supports_streaming()` / `supports_structured_output()`** — override defaults (`False` base default for streaming); table in Q-L.3 | **Done** | Medium | Each `providers/*.py`, ABC defaults | Conformance reads flags; no false positives |
| Q-L.6 | **`PolicyEngine` + `llm_cost_evaluation`** — rule hook on `TASK_COMPLETED` or policy replay; or remove “next step” from docs until done | **Done** | Medium | `governance/`, `observability_bridge.py`, `policy_engine.py` | Test: over-quota/warn triggers policy decision or structured log contract |
| Q-L.7 | **Usage tracking doc** — distinguish adapter `LLMAdapterUsageLog` vs runtime `LLMUsageTracker` | **Done** | Low | `docs/LLM_ADAPTERS.md` § Observability | Two-layer table |
| Q-L.8 | **Conformance: structured output** — parametrize providers with `supports_structured_output`; mock SDK | **Done** | Medium | `tests/unit/llm_adapters/` | Added to gate subset in `llm-adapters-guard.yml` |
| Q-L.9 | **Bedrock `context_window_tokens`** — lookup table or model metadata for common `model_id` | **Done** | Low | `providers/aws_bedrock_adapter.py` | `_CONTEXT_WINDOWS` + prefix fallback; `test_bedrock_context_window.py` |
| Q-L.10 | **OpenAI-compat adapter init** — replace `__dict__.update` with explicit delegation or composition wrapper | **Done** | Low | `openai_compat_providers.py`, factory | `_delegate` + `__getattr__` composition |
| Q-L.11 | **Central env appendix** — single table: `INTERGRAX_LLM_*`, secrets map, per-provider overrides | **Done** | Medium | `LLM_ADAPTERS.md` appendix | Cross-links from each `providers/*/USAGE.md` |

---

#### Phase Q-R — RAG pipeline & Nexus RAG integration

**Components:** `intergrax/rag/`, `runtime/nexus/context/context_builder.py`, `runtime_steps/rag_step.py`, `history_step.py`, `pipelines/no_planner_pipeline.py`, `tools/providers/rag/`, `agents/legal/*` plan flags.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-R.1 | **Delete dead code in `ContextBuilder`** — `_build_backend_where`, `_map_hits_to_chunks`, unused `VectorStoreHit` import | **Done** | High | `context_builder.py` | Dead helpers removed (2026-06-01) |
| Q-R.2 | **Single retrieval per turn (design)** — ADR in plan: either (A) retrieval only in `RagStep`/`rag.retrieve`, or (B) only in `HistoryStep`; remove duplicate vector calls | **Done** | High | `history_step.py`, `context_builder.py` | `HistoryStep` uses `perform_retrieval=False` (2026-06-01) |
| Q-R.3 | **`ContextBuilder` respects plan `use_rag`** — `_should_use_rag` checks plan/engine `use_rag` when present, not only `enable_rag` | **Done** | High | `context_builder.py` | `request.metadata["use_rag"]`; unit test (2026-06-01) |
| Q-R.4 | **`NoPlannerPipeline` conditional `RagStep`** — include `RagStep` only when plan/tool_ids require RAG | **Done** | High | `no_planner_pipeline.py`, `pipeline_factory.py` | Pipeline test matrix |
| Q-R.5 | **Prefetch vs final `top_k`** — `RetrievalRequest.prefetch_k` optional; Nexus passes `max_docs_per_query` as `final_k` only; service uses profile `prefetch_top_k` when unset | **Done** | High | `retrieval_request.py`, `retrieval_service.py` | `test_retrieval_request_prefetch.py` (2026-06-01) |
| Q-R.6 | **Unify RAG config surface** — map `RuntimeConfig.max_docs_per_query` / threshold → `RagProfile` at factory wire time; deprecate duplicate fields with shim + trace | **Done** | High | `nexus/config.py`, `RetrievalRuntimeConfig`, `rag_profile.py` | One source of truth documented |
| Q-R.7 | **`RagProfile.extras`** — use for vendor knobs or remove field | **Done** | Low | `rag_profile.py` | No unused field in frozen profile |
| Q-R.8 | **`INTERGRAX_RAG_METRICS_ENABLED` in `rag_profile_from_env`** or documented exclusion | **Done** | Low | `rag_profile.py`, architecture §7.1.2 | `extras.metrics_enabled` from env (2026-06-01) |
| Q-R.9 | **`rag/answers/` deprecation path** — mark package deprecated; redirect doc to `RetrievalService`; no new imports from Nexus | **Done** | Medium | `rag/answers/`, `chat_agent` removal (Q-X.1) | Grep: zero imports from `runtime/` and `agents/` except tests |
| Q-R.10 | **`UserProfileManager` LTM via `RetrievalService`** — same metadata scope / `RagProfile` chunking policy | **Done** | Medium | `memory/user_profile_manager.py` | Unit test with fake `RetrievalService` |
| Q-R.11 | **Naming guide — three “context builders”** — table in `AGENT_CREATION_GUIDE` or `intergrax/rag/README.md`: Nexus `ContextBuilder`, `ContextManager`, `DefaultContextBuilder` | **Done** | Low | Docs | Linked from architecture §28 pointer |
| Q-R.12 | **Legacy `use_rag` plan flags** — migrate Legal/Nexus plans to `tool_ids` including `rag.retrieve`; emit deprecation `RuntimeEvent` on boolean | **Done** | Medium | `engine_plan_models.py`, `legal/*`, `tool_runtime.py` | Legal tests use `tool_ids`; booleans shim one release |

---

#### Phase Q-M — Memory

**Components:** `intergrax/memory/`, `runtime/task_memory/`, `runtime/nexus/context/`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-M.1 | **Memory architecture one-pager** — four stores: session history, user LTM, task KV (`TaskMemory`), shared graph context; diagram + when to enable SQLite | **Done** | High | `docs/` section in plan §0 or `AGENT_CREATION_GUIDE` Appendix | Linked from §0.3 execution path |
| Q-M.2 | **Task memory visibility in scaffold** — `wire_task_memory` in lab/product templates; env `INTERGRAX_TASK_MEMORY_DB` in `.env.example`; Step 4E paragraph | **Done** | Medium | `applications/*`, scaffold, guide | Scaffold acceptance asserts task memory path optional |
| Q-M.3 | **`resolve_task_memory_persistence` defaults** — log warning when None in lab; debug API hint | **Done** | Low | `task_memory/store.py`, `lab_application` factory | Doc + single integration test |

---

#### Phase Q-O — Observability & metrics

**Components:** `runtime/events/`, `runtime/nexus/tracing/`, `runtime/metrics/`, `debug/`, `llm_adapters/tracking/`, `rag/tracking/`, `applications/_shared/platform_wiring.py`.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-O.1 | **Register RAG observability plugin in default bootstrap** — `register_rag_observability_plugin(plugins)` alongside LLM in `platform_wiring.py` | **Done** | **Critical** | `platform_wiring.py` | `test_platform_wiring_observability.py` (2026-06-01) |
| Q-O.2 | **RAG observability bridge tests** — mirror `test_observability_bridge.py` (LLM) | **Done** | High | `tests/unit/rag/tracking/` | `test_rag_observability_bridge.py` (2026-06-01) |
| Q-O.3 | **Parser trace export strategy** — route `parser_trace_flush` through `ObservabilityBackend` **or** document intentional bypass + single env table | **Done** | Medium | `parser_trace_flush.py`, `parser_trace_exporter.py`, integrations | Documented in architecture §7.1.2 RAG observability |
| Q-O.4 | **`metrics/export.py` typed trace summary** — use `DiagnosticPayload` / `trace_models` schema ids instead of substring heuristics | **Done** | Medium | `runtime/metrics/export.py` | Unit test with synthetic trace events |
| Q-O.5 | **Lint `metrics/export.py`** — remove duplicate `ExecutionMetrics` import | **Done** | Low | `metrics/export.py` | Ruff clean (2026-06-01) |
| Q-O.6 | **`export_run_metrics` behavioral field** — populate from governance/replay or remove from DTO | **Done** | Low | `metrics/export.py` | `ExecutionMetrics` from trace events in `export_run_metrics` |
| Q-O.7 | **Mount LLM metrics routes on lab** — `register_llm_metrics_routes(app)` when `INTERGRAX_LLM_METRICS_ENABLED` | **Done** | Medium | `lab_application/host/factory.py` | Routes registered at factory (2026-06-01) |
| Q-O.8 | **Observability env profile doc** — one table: trace DB, runtime events DB, LLM/RAG metrics, parser trace, integration observability slug | **Done** | High | New subsection §0 or `infra/README` cross-link | All Tier-3 `.env.example` reference same names |
| Q-O.9 | **RAG metrics parity decision** — implement log-only parity **or** `register_rag_metrics_routes` + optional Pushgateway | **Done** | Medium | `rag/tracking/`, architecture §7.1.2 | Matches documented behavior |
| Q-O.10 | **Unify phase mapping** — `trace_bridge` delegates phase to `phase_coverage.py`; single source | **Done** | Medium | `events/trace_bridge.py`, `phase_coverage.py` | Unit test: same `ExecutionPhase` for sample events |
| Q-O.11 | **Debug router type imports** — explicit imports for `DebugHitlResumeService`, `AgentRegistry` in annotations | **Done** | Low | `debug/router.py`, `debug/app.py` | Explicit imports in `debug/router.py` |
| Q-O.12 | **`trace_bridge` unit tests** | **Done** | Medium | `tests/unit/runtime/events/test_trace_bridge.py` | Gate marker |
| Q-O.13 | **Clarify dual Prometheus** — in-process scrape vs `integrations` PromQL backend | **Done** | Low | `docs/LLM_ADAPTERS.md` § Observability | Prevents operator confusion |
| Q-O.14 | **Event/trace store adoption** — SQLite-first default; scale-out criteria for `cassandra` / `elasticsearch` | **Done** | Low | Architecture §33.1 + `cassandra/USAGE.md` | No separate ADR file |

---

#### Phase Q-X — Legacy removal & code hygiene

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-X.1 | **`ChatAgent` removal** — migrate remaining tests to `RuntimeEngine`/`NexusLoop`; delete `intergrax/chat_agent.py`; keep import guard script as negative test | **Done** | High | `chat_agent.py`, `tests/unit/chat_agent/` | Grep zero production imports; gate green |
| Q-X.2 | **`task_metadata_bridge` shrink** — migrate callers to typed `Task` metadata; deprecate flat bridge with warning event | **Done** | Medium | `task_metadata_bridge.py`, `uaep.py` | `execution_options_for_request`; legacy warnings; Task hydrates typed fields |
| Q-X.3 | **Copyright / naming consistency** — `Intergrax` header; fix `Integrax` typo in `chat_agent` (or file deleted in Q-X.1) | **Done** | Low | Affected files from audit | Spot-check script or ruff rule |
| Q-X.4 | **`tools_base` deprecation timeline** — document removal after Q-R.12; no new imports | **Done** | Low | `tools/tools_base.py`, governance script | Module docstring + `DeprecationWarning` on import |
| Q-X.5 | **Sync M.6 “Future” slugs table** — weaviate, milvus, snowflake, vault → **Done (beta)** with paths | **Done** | Low | This plan M.6 P3 section | Table matches repo `integrations/providers/` |

---

#### Phase Q-T — Test harness gaps

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-T.1 | NexusLoop unit suite | **Done** | High | See Q-N.14 | — |
| Q-T.2 | `test_rag_profile_from_env` | **Done** | Medium | `tests/unit/rag/profiles/` | Gate (2026-06-01) |
| Q-T.3 | `test_context_builder_retrieval` | **Done** | High | `tests/unit/runtime/nexus/context/` | `test_context_builder.py` (2026-06-01) |
| Q-T.4 | `test_user_profile_manager` | **Done** | Medium | `tests/unit/memory/` | Index + search |
| Q-T.5 | **Catalog vs legacy RAG path** — integration test one pipeline run, retrieval call count ≤1 | **Done** | High | `tests/integration/runtime/` | Implements Q-R.2 acceptance |
| Q-T.6 | **Observability wiring E2E** — lab factory bootstraps LLM+RAG plugins | **Done** | High | `tests/integration/runtime/test_platform_wiring_observability.py` | Q-O.1 (2026-06-01) |

---

#### Phase Q-D — Documentation & plan sync

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-D.1 | Update `docs/README.md` current focus → Phase Q | **Done** | High | `docs/README.md` | — |
| Q-D.2 | Canon §52 Phase L status → **Done** (pointer to Phase Q) | **Done** | Low | `intergrax_runtime_architecture.md` §52 | — |
| Q-D.3 | §2 architecture map — §42 row points to Phase Q-N.5–Q-N.6 | **Done** | Low | This file §2 | — |
| Q-D.4 | `AGENT_CREATION_GUIDE` — Q-M.1 memory diagram + Q-R.11 naming | **Done** | Medium | Guide appendices | — |
| Q-D.5 | **§5.2 reuse enforcement** — document existing gates (`check_agents_vendor_imports`, `check_integration_vendor_imports`, `check_production_chat_agent_imports`) in AGENT_CREATION_GUIDE anti-patterns | **Done** | Low | Guide + `scripts/` | New agent authors see one list |

---

#### Phase Q — Definition of done (global)

1. Deliverable row **Done** with PR link/date in Appendix C paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **No new** duplicate Tier-0 mechanism (§5.2).
4. **Tests** for behavior change (unit or integration); not docs-only for code fixes.
5. Update **Appendix C** status column for audit ID.

---

#### Phase Q — Recommended execution order

Execute in order unless a row is marked parallel. Critical path for harness stability:

```text
Wave 1 (bugs + critical):  Q-O.1 → Q-N.2 → Q-R.5 → Q-R.1
Wave 2 (RAG semantics):    Q-R.3 → Q-R.4 → Q-R.2 → Q-T.5 → Q-R.6
Wave 3 (observability):    Q-O.2 → Q-O.4 → Q-O.7 → Q-O.10 → Q-O.12 → Q-O.8
Wave 4 (Nexus structure):  Q-N.14 → Q-N.1 → Q-N.3 → Q-N.8
Wave 5 (LLM docs/debt):    Q-L.3 → Q-L.1 → Q-L.5 → Q-L.8 → Q-L.11
Wave 6 (memory + legacy):  Q-M.1 → Q-M.2 → Q-R.10 → Q-X.1 → Q-R.9
Wave 7 (hooks + policy):   Q-N.5 → Q-N.6 → Q-L.6 → Q-N.4
Wave 8 (cleanup):          Q-N.7 → Q-X.2 → Q-X.3 → Q-X.5 → Q-D.*
Parallel anytime:          Q-L.2, Q-L.4, Q-L.9, Q-L.10, Q-O.5, Q-O.6, Q-O.11, Q-O.13, Q-N.10–Q-N.13, Q-N.15
```

**Historical (Phase Q only):** Do not start Phase K.1/K.2 until Q Waves 1–3 were **Done** — **met** (2026-06-01). Phase S focuses on harness environment; K.1/K.2 wait until S Done.

---

### Phase Q+ — Harness Hardening (post-audit 2026-06-01)

**Source:** Technical debt audit after Phase Q — architecture compliance, typing, observability gaps, legacy parallel stacks, Nexus/planning monoliths.  
**Goal:** Intergrax as a **strong, typed, observable harness** comparable in discipline to Cursor / Claude Code / Google ADK-style agent labs — not merely “gate green”.  
**Principle:** evolve, not rewrite · explicit `Protocol` / Pydantic at boundaries · **zero new `getattr` in `runtime/nexus` and `agents/`** (integrations/LLM SDK edges exempt) · one Q+.* ID per PR · gate green.

**Relationship to Phase Q:** Phase Q closed the **first** audit (Appendix C). Phase Q+ closes the **second** audit (Appendix D). Do not reopen Q.* rows unless a regression is found.

**Out of scope for Phase Q+:**

- Phase K.1/K.2 product agents (unless explicitly prioritized — record in Appendix D)
- K.6 / B.15 Legal live LLM E2E
- New integration catalog slugs (Phase M on-demand)
- Rewriting all LLM provider adapters (only isolate SDK reflection — Q+-I.*)
- Mandatory Cassandra / multi-tenant scale-out (architecture §33.1 criteria only)

**Phase Q+ complete when:** All Q+ rows **Done** or **Won't fix** (canon amendment); Appendix D 100%; §0.5 Harness hardening **Done**; gate unchanged or increased; grep gate: no new `getattr` in `runtime/nexus/` + `agents/` (CI script Q+.0.3).

---

#### Q+.0 — Program governance

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+.0.1 | **Appendix D** — audit topic → Q+ ID matrix (P0–P3) | **Done** | High | This file Appendix D | Every audit section mapped |
| Q+.0.2 | **Q+ execution order** — Waves 1–5 below | **Done** | High | §4 Priority Order | Team follows wave sequence |
| Q+.0.3 | **CI grep gate** — fail on new `getattr`/`setattr` in `intergrax/runtime/nexus/`, `intergrax/agents/` | **Done** | High | `scripts/check_harness_no_getattr.py` + gate workflow | Zero grandfathered harness paths (2026-06-01) |

---

#### Q+-T — Typing & explicit contracts (P0)

**Audit:** loose coupling, `getattr`, `Any` on harness paths, classes not implementing Protocols.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-T.1 | **`UAEPAgent` Protocol** — `get_steps`, `run_step`, optional `resume_step`, `decide_after_step`; replace `supports_uaep()` duck typing | **Done** | **Critical** | `agents/uaep_protocol.py`, `agents/uaep.py` | Standalone `@runtime_checkable` Protocol; no `getattr` in UAEP |
| Q+-T.2 | **`ToolInvokerProtocol`** — explicit `registry`; remove `catalog_context` invoker chain `getattr` | **Done** | **Critical** | `runtime/nexus/tools/`, `catalog_context.py` | Typed invoker only |
| Q+-T.3 | **`RuntimeState` trace hook** — `trace_event: Optional[TraceEmitterFn]`; remove `getattr(state, "trace_event")` | **Done** | High | `tool_access_policy.py` | `TraceEmittingRuntimeState` Protocol |
| Q+-T.4 | **`Agent.can_handle(TaskContext)`** — replace `task_context: Any` on `Agent` ABC | **Done** | High | `agents/agent_contract.py`, product agents | Production agents use `TaskContext` |
| Q+-T.5 | **`EnginePlan` / tool plan union** — `tool_runtime` reads `tool_ids` without `getattr(source, …)` | **Done** | High | `tool_runtime.py`, `engine_plan_models.py` | `ToolPlanLike` + `EnginePlan.resolved_tool_ids()` |
| Q+-T.6 | **`long_running_bridge`** — `RuntimeEventPublisher` accepts `RuntimeEvent` only (not `object`) | **Done** | Medium | `orchestration/long_running_bridge.py` | Align with `NexusRuntimeEventPublisher` |
| Q+-T.7 | **`context_builder` session snapshot** — typed session view; no `getattr(session, attr)` loop | **Done** | Medium | `context/context_builder.py` | `ChatSession` fields directly |
| Q+-T.8 | **`rag_step_policy`** — use `NexusPlan` / `EnginePlan` fields only | **Done** | Low | `pipelines/rag_step_policy.py` | `isinstance(plan, EnginePlan)` |

---

#### Q+-N — Nexus decomposition & retry (P0–P1)

**Audit:** `nexus_loop` still owns intake/classification/planning; no `RetryCoordinator`; thin graph tests.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-N.1 | **`NexusIntakeRunner`** — resume/long-running preamble + HITL verdict branches extracted from `nexus_loop` | **Done** | High | `orchestration/intake_runner.py` | `nexus_loop` delegates; behavior unchanged |
| Q+-N.2 | **`NexusPlanningRunner`** — classify → plan → pre-graph HITL; hooks + runtime events | **Done** | High | `orchestration/planning_runner.py` | `nexus_loop` slimmed; graph phase unchanged |
| Q+-N.3 | **`RetryCoordinator`** (optional facade) — delegate `RetryEngine` + `RuntimeConfig.max_run_retries` with `RETRY_SCHEDULED` events | **Done** | Medium | `nexus/retry/coordinator.py`, architecture §31.1 | Graph emits `RETRY_SCHEDULED`; run retries use coordinator |
| Q+-N.4 | **`GraphExecutor` integration tests** — handoff edge, validation retry + alternate agent | **Done** | Medium | `tests/integration/runtime/test_graph_executor_handoff_retry.py` | Handoff + alternate-agent retry |
| Q+-N.5 | **Planner failure observability** — `engine_planner` errors → `RuntimeEventType.PLAN_FAILED` (narrow exceptions) | **Done** | Medium | `planning/engine_planner.py`, `planner_events.py` | `test_engine_planner_plan_failed.py` |

---

#### Q+-O — Observability parity (P1)

**Audit:** metrics heuristics, RAG HTTP metrics asymmetry, lab `production_mode` not wired.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-O.1 | **`export_run_metrics` typed-only** — remove getattr/substring fallbacks; require `DiagnosticPayload` / schema ids | **Done** | High | `runtime/metrics/export.py` | `TraceEvent` / `SerializedTraceEvent` only |
| Q+-O.2 | **Wire `harness_production_mode()`** in lab + scaffold factories | **Done** | Medium | `scaffold/new_agent.py`, Tier-2 lab agents | Lab/scaffold agents use `harness_production_mode()` |
| Q+-O.3 | **RAG metrics HTTP decision** — implement `register_rag_metrics_routes` **or** document Won't fix + unified `/metrics` scrape | **Won't fix** (core) | Medium | architecture §7.1.2 | No default `/metrics/rag`; log + plugin scrape |
| Q+-O.4 | **Ingestion path events** — consistent `RuntimeEvent` on ingest failures | **Done** | Low | `ingestion_events.py`, `ingestion_service.py` | `INGESTION_FAILED` + gate test |

---

#### Q+-L — Legacy & duplicate stacks (P0–P2)

**Audit:** `tools_agent`, `supervisor`, `chains`, `openai/rag`, `rag/answers` parallel Tier-0 paths.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-L.1 | **`tools_agent` deprecation enforcement** — extend `check_*_imports`; zero new production imports outside `agents/legal` migration | **Done** | **Critical** | `scripts/check_tools_agent_imports.py` | CI fails on new imports |
| Q+-L.2 | **Legal agent → catalog `ToolRuntime`** — remove runtime dependency on `ToolsAgent` / `ToolsStep` planner loop | **Done** | **Critical** | `agents/legal/`, `catalog_tool_planner.py` | Legal uses `CatalogToolPlanner` + `tool_planner` |
| Q+-L.3 | **`RuntimeConfig` default tools** — no default `ToolsAgent` in `config` / `config_sections` | **Done** | High | `nexus/config.py`, `config_sections.py` | `tool_planner: ToolPlannerProtocol` only |
| Q+-L.4 | **`supervisor` boundary** — move to `experiments/supervisor` or hard-deprecate with import guard | **Done** | Medium | `intergrax/supervisor/__init__.py`, gate import test | Not imported from runtime/applications |
| Q+-L.5 | **`chains/langchain_qa_chain`** — experimental only or remove from default paths | **Done** | Medium | `intergrax/chains/__init__.py`, gate import test | Tests-only imports |
| Q+-L.6 | **`rag/answers` e2e** — migrate `tests/e2e/rag` to `RetrievalService`; package import guard | **Done** | Medium | `tests/e2e/rag/test_rag_full_runtime_e2e.py` | No `rag.answers` import |
| Q+-L.7 | **`openai/rag/rag_openai.py`** — bridge to `RetrievalService` or delete if unused | **Won't fix** | Low | `openai/rag/rag_openai.py` | Zero production imports; legacy sample only |

---

#### Q+-M — Task metadata & bridge (P1)

**Audit:** automatic legacy hydrate on every `Task()`; bridge still central.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-M.1 | **Opt-in metadata hydrate** — `Task.from_metadata()` / factory; remove automatic `model_validator` hydrate | **Done** | High | `task/task.py`, `task_metadata_bridge.py` | Hydrate only when legacy keys / `_hydrate_legacy` |
| Q+-M.2 | **Tier-3 uses typed `Task.options` only** — lab/scaffold run path sets contract without flat keys | **Done** | Medium | `task_intake.py`, lab `fastapi_router.py` | `graph_id` via orchestration state |

---

#### Q+-P — Planning monoliths (P2)

**Audit:** `step_planner.py` ~683 lines, `engine_planner.py` ~623 lines — hard to extend.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-P.1 | **Split `engine_planner`** — parse / validate / LLM call modules; each &lt; ~300 lines | **Done** | Medium | `engine_planner_parse.py`, `engine_planner_messages.py`, `engine_planner_diagnostics.py`, `engine_planner_orchestrator.py` | Orchestration + traces extracted |
| Q+-P.2 | **Split `step_planner`** — strategy registry vs executor | **Done** | Medium | `planning/step_planner/` (`config`, `step_factory`, `assembly`, `strategies`, `planner`) | Package import stable; gate tests |
| Q+-P.3 | **Structured plan parse errors** — no silent `except Exception: pass` without trace | **Done** | Medium | `engine_planner_parse.py` | Narrow `ValueError` / `JSONDecodeError` only |

---

#### Q+-S — Session monolith (P2)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-S.1 | **Decompose `session_manager`** — storage vs summarization vs org instructions | **Done** | Low | `session_profile_instructions.py`, `session_consolidation.py`, `session_lifecycle.py` | Profile, consolidation, lifecycle coordinators |

---

#### Q+-I — Integration / LLM SDK edges (P3)

**Audit:** acceptable `getattr` inside provider SDK shims — isolate, do not spread.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-I.1 | **SDK reflection quarantine** — document per-provider `*_sdk_bridge.py`; no new getattr in `runtime/` | **Done** | Low | Architecture §5.2.2 | Vendor SDK bridges quarantined to provider modules |

---

#### Q+-D — Documentation (Phase Q+)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| Q+-D.1 | Canon §9 — orchestration module list includes intake/planning runners (when done) | **Done** | Low | `intergrax_runtime_architecture.md` | — |
| Q+-D.2 | `AGENT_CREATION_GUIDE` — anti-pattern: `getattr`, `ToolsAgent`, flat metadata | **Done** | Medium | Guide § anti-patterns | Linked from §0.6 |
| Q+-D.3 | `docs/README.md` focus → Phase Q+ Wave 1 | **Done** | High | `docs/README.md` | Wave 2 focus |

---

#### Phase Q+ — Definition of done

1. Q+ row **Done** with date in Appendix D paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **No new** `getattr`/`setattr` in harness paths (Q+.0.3).
4. **Tests** for each behavior change.
5. Update Appendix D status.

---

#### Phase Q+ — Recommended execution order

```text
Wave 1 (P0 contracts):     Q+.0.3 → Q+-T.1 → Q+-T.2 → Q+-T.3 → Q+-T.4 → Q+-T.5
Wave 2 (P0 legacy):      Q+-L.1 → Q+-L.2 → Q+-L.3 → Q+-M.1
Wave 3 (P1 Nexus+obs):   Q+-N.1 → Q+-N.2 → Q+-O.1 → Q+-O.2 → Q+-N.3 → Q+-N.4 → Q+-N.5
Wave 4 (P2 monoliths):     Q+-P.1 → Q+-P.2 → Q+-S.1 → Q+-L.4 → Q+-L.5 → Q+-L.6
Wave 5 (P3 + docs):        Q+-L.7 → Q+-I.1 → Q+-O.3 → Q+-O.4 → Q+-D.*
Parallel anytime:         Q+-T.6, Q+-T.7, Q+-T.8, Q+-M.2
```

**Gate before Phase K scale:** Waves 1–3 **Done** (typing + Legal off ToolsAgent + Nexus intake/planning split + metrics typed).

---

### Phase R — Harness AI Alignment (post-audit 2026-06-01)

**Source:** Harness AI philosophy audit (scaffold, harness, LLM, tool vs skill, context engineering, subagents, policy) — traceability in **Appendix E**.  
**Status:** **Done (MVP)** (2026-06-01). **Prerequisite met:** Phase **Q+ Done**.  
**Goal:** Intergrax vocabulary and Tier-0 modules align with industry harness terminology **without** breaking Integration → Tool → Agent stack; add **Skill Library** for reuse and external compatibility.  
**Principle:** evolve, not rewrite · skills **compose** tools (never replace `ToolRuntime`) · one R.* ID per PR · gate green.

**Out of scope for Phase R:**

- Nested full harness per child (Cursor 1:1 subagent OS) — use graph delegation first (R-Delegate)
- Auto-discovery of skills from filesystem without validation
- Mandatory migration of all Tier-2 agents to skills in one release

**Phase R (MVP) complete:** Appendix E 100% **Done** or **Won't fix**; §0 Phase R row **Done**; gate **450 passed** (2026-06-01). Further skill catalog expansion is product work, not a harness gate.

---

#### R.0 — Canon, ADR, terminology (do first)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R.0.1 | **ADR: Skill layer Option 2** — reject “skills = tools only”; document four-layer model | **Done** | **Critical** | Architecture §7.1.8, §5.3 | Option 1 listed as rejected with rationale |
| R.0.2 | **Canon sections** — §5.3 Harness mapping, §7.1.8 Skills, §28.1 Context engineering, §42.14.3 Delegation, §42.11.4 Policy bundle | **Done** | **Critical** | `intergrax_runtime_architecture.md` | Cross-linked from plan §0 |
| R.0.3 | **Remove tool/skill conflation** in code docstrings | **Done** | High | `tools/core/contracts.py` | `ToolContract` describes **tool** only |
| R.0.4 | **README navigation** — Phase R, skills layer in root + docs README | **Done** | Medium | `/README.md`, `docs/README.md` | GitHub landing + docs index mention skills |

**Delivery rule:** Same as §6.1 — one R.* ID → PR → update Appendix E status → gate.

---

#### R-Skill — Skill Library (Tier-0)

**Problem:** Integrations and tools are production-grade; **skills are not**. Agents duplicate prompts, tool allow-lists, and policy fragments. External harness ecosystems (Cursor skills, internal markdown packs) cannot plug in without a **validated manifest**.

**Target layout:**

```text
intergrax/skills/
├── core/                   # SkillContract, SkillManifest, SkillProvider protocol
├── registry/               # SkillCatalog, SkillProfile, register_default_skills()
├── importers/              # cursor_skill_md.py, … (validate → SkillManifest)
├── _shared/
└── providers/
    └── <domain>/           # e.g. legal/, research/
        ├── manifest.py     # SkillManifest instance(s)
        ├── prompts.yaml    # or Prompt Registry refs
        └── USAGE.md
```

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Skill.1 | **`SkillManifest` + `SkillContract`** — frozen manifest: `skill_id`, `version`, `description`, `tool_ids`, `prompt_instruction_ids`, `policy_fragment_id`, `risk_tier`, `tags` | **Done** | **Critical** | `intergrax/skills/core/contracts.py` | Pydantic/jsonschema round-trip test |
| R-Skill.2 | **`SkillRegistry` + `SkillProfile` + `SkillCatalog`** — mirror Tool registry pattern | **Done** | **Critical** | `intergrax/skills/registry/` | `build_registry_from_profile()` |
| R-Skill.3 | **`SkillResolver`** — given `skill_ids`, produce resolved `allowed_tools` ∪, prompt pack refs, policy fragments; **no LLM execution** in resolver | **Done** | **Critical** | `intergrax/skills/resolver.py` | Unit: two skills merge tool lists with conflict rules |
| R-Skill.4 | **Tier-3 wiring** — skill profile in `ApplicationBuildContext`, `skill_wiring.py`, legal host | **Done** | High | `applications/_shared/skill_wiring.py` | Legal registry resolves skills |
| R-Skill.5 | **`AgentContract.skill_ids`** + validation against registry at register time | **Done** | High | `intergrax/contracts/`, `AgentRegistry` | Unknown skill_id → register error |
| R-Skill.6 | **`docs/SKILLS.md`** — catalog, layering diagram, import rules | **Done** | Medium | `docs/SKILLS.md`, `docs/README.md` index row | Approved index entry |
| R-Skill.7 | **Scaffold `new-skill`** | **Done** | Medium | `intergrax/scaffold/new_skill.py` | `python -m intergrax.scaffold new-skill <id>` |
| R-Skill.8 | **`CursorSkillImporter`** — parse `SKILL.md` + frontmatter → `SkillManifest` (best-effort; reject on schema fail) | **Done** | High | `intergrax/skills/importers/cursor_skill_md.py` | Fixture test with sample SKILL.md |
| R-Skill.9 | **Pilot skill pack** — `legal.contract_review` (tool_ids + prompt refs + policy fragment) | **Done** | High | `intergrax/skills/providers/legal/` | Legal agent lists `skill_ids`; gate green |
| R-Skill.10 | **Nexus trace events** — `SKILL_RESOLVED`, `SKILL_IMPORT_FAILED` | **Done** | Low | `runtime/events/context_skill_recording.py` | `record()` on register + import service |

**Skill vs tool enforcement:**

| Rule | Enforcement |
|------|-------------|
| Skill MUST NOT be a `ToolContract` | CI: no `ToolHandler` named `skill.*` without ADR |
| Skill MAY reference only registered `tool_id`s | `SkillResolver` validates against `ToolRegistry` |
| LLM tool-calling surface = **tools only** | Skills expand allow-list before run, not at invoke time |
| External skill without manifest validation | **Rejected** at import — no silent attach |

---

#### R-Context — Context engineering (Tier-1)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Context.1 | **`ContextBudgetPolicy`** — `max_chars`, `max_tokens_estimate`, `summary_tier` defaults; applied in `ContextManager.build_agent_context()` | **Done** | **Critical** | `runtime/nexus/context/context_budget.py` | Test: over-budget input trimmed |
| R-Context.2 | **Trace events** — `CONTEXT_ASSEMBLED`, `CONTEXT_TRIMMED` with before/after sizes | **Done** | High | `ContextManager` + `context_skill_recording` | Emitted when `event_bus` wired |
| R-Context.3 | **AGENT_CREATION_GUIDE** — “Context engineering” subsection links canon §28.1 | **Done** | Medium | `AGENT_CREATION_GUIDE.md` Appendix G | No duplicate truth |
| R-Context.4 | **Finish unified tool path** — residual `use_rag` / `RagStep` callers → `rag.retrieve` | **Done** | High | `tool_gateway.py`, legal bridge, `context_builder.py` | Bridge uses `tool_ids`; LLM booleans sync in `LegalToolPlan` only |

---

#### R-Delegate — Graph-native delegation (subagent equivalent)

Intergrax does **not** implement Cursor-style nested harness in Phase R. **Delegation** = Nexus graph node with isolated memory namespace and bounded context assembly.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Delegate.1 | **`DelegationSpec` on `ExecutionNode`** — `child_agent_id`, `isolated_memory_namespace`, `context_assembly_override` | **Done** | High | `contracts/delegation.py`, `execution_graph.py` | Schema + validation |
| R-Delegate.2 | **Memory namespace isolation** — child reads/writes under `task_id/delegation/{node_id}/` via `MemoryView` | **Done** | High | `delegation_memory.py`, UAEP | Unit test |
| R-Delegate.3 | **Trace linkage** — `parent_run_id`, `parent_node_id` on child run metadata | **Done** | Medium | `graph_executor.py` | Request metadata on child node |
| R-Delegate.4 | **Integration tests** — two-agent graph with delegation node | **Done** | Medium | `test_graph_executor_delegation.py` | Gate |

---

#### R-Policy — Unified policy bundle (Tier-1 + Tier-3)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| R-Policy.1 | **`RuntimePolicyBundle`** — aggregates tool, memory, budget, HITL, plan-loop; optional `domain_fragments: dict[str, Any]` | **Done** | High | `runtime/policy/policy_bundle.py` | Import via `policy_bundle` module (not `policy.__init__`) |
| R-Policy.2 | **Tier-3 composition** — lab/product factories build bundle once per app | **Done** | High | `policy_wiring.py`, lab/legal `wiring.py` | `ApplicationBuildContext.policy_bundle` |
| R-Policy.3 | **Canon §42.11.5** — “how to read policy for a run” operator section | **Done** | Medium | Architecture §42.11.5 | Operator runbook table |

---

#### Phase R — Definition of done

1. R row **Done** with date in Appendix E paydown log.
2. **Gate:** `uv run pytest -m gate -q` green.
3. **Skills:** at least one first-party skill pack + one importer test (R-Skill.8 or Won't fix with reason).
4. **No** new `ToolContract` entries that represent multi-step business workflows without ADR.
5. Update Appendix E status.

---

#### Phase R — Recommended execution order

```text
Wave R0 (canon):           R.0.1 → R.0.2 → R.0.3 → R.0.4
Wave R1 (skill core):      R-Skill.1 → R-Skill.2 → R-Skill.3 → R-Skill.5 → R-Skill.4
Wave R2 (skill ecosystem): R-Skill.8 → R-Skill.7 → R-Skill.9 → R-Skill.6 → R-Skill.10
Wave R3 (context):         R-Context.1 → R-Context.2 → R-Context.4 → R-Context.3
Wave R4 (delegate):        R-Delegate.1 → R-Delegate.2 → R-Delegate.3 → R-Delegate.4
Wave R5 (policy):          R-Policy.1 → R-Policy.2 → R-Policy.3
```

**Gate before Phase K.1/K.2 scale:** **Met** — Q+ **Done**, R-Skill.1–R-Skill.5 and R-Context.1 **Done**.

---

### Phase S — Harness Environment GA (post-R 2026-06-01)

**Source:** Architecture audit (2026-06-01); strategic pivot — **full harness environment** before business agents.  
**Status:** **Done** (2026-06-01). **Prerequisites met:** Phase L, Q, Q+, R (MVP).  
**Goal:** Make the **Harness AI environment** (Tier-0 + Tier-1 + lab/product wiring) **ops-ready and complete** — stable integration paths, observability, platform skills, operator docs — using **existing** reference agents (echo, research, legal, signoff_probe), not new product agents.  
**Principle:** evolve, not rewrite · Tier-1 only via §0.6 · one S.* ID per PR · gate green.

**Explicitly out of scope for Phase S:**

- **K.1 Problem Radar / K.2 Vendor Discovery** — **Phase K** (after U Done)
- Multi-tenant SaaS (canon §50 — future)
- Nested full harness per child — graph delegation remains default (R-Delegate)
- `stable` on all 99 integration slugs — only the **lab harness stack** (see S-Ops.1)

**Deferred from old Phase S scope → Phase K:** S-K.* (reference business agent proof).

#### S.0 — Canon & strategy sync

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S.0.1 | **Development strategy** document + docs index | **Done** | Critical | `INTERGRAX_DEVELOPMENT_STRATEGY.md`, `docs/README.md` | Linked from plan + root README |
| S.0.2 | **Canon §2 / §50–§51** — laboratory + harness narrative | **Done** | Critical | `intergrax_runtime_architecture.md` | No contradiction with strategy |
| S.0.3 | **Canon §52** — Phase S harness question | **Done** | High | Canon §52 | Environment GA, not K.1/K.2 |
| S.0.4 | **Plan pivot** — Phase S = harness only; K.1/K.2 deferred | **Done** | Critical | This file §0, §4, Phase K, Appendix F | 2026-06-01 |

#### S-Ops — Integration & observability (harness stack)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Ops.1 | **Integration stable track** — lab harness stack (`sqlite`, `redis`, `qdrant`, `slack`, `sentry`, …) marked `stable` in catalog | **Done** | **Critical** | `harness_lab_stack.py`, `INTEGRATIONS.md` | `test_harness_lab_stable_stack.py` |
| S-Ops.2 | **OTLP / observability** — lab profile wires `otel` when `LAB_OTEL_ENABLED`; document noop vs export | **Done** | High | `IntegrationProfile.harness_environment()`, `.env.example` | `test_lab_harness_environment_wiring.py` |
| S-Ops.3 | **Harness-smoke CI** — expand M.12+ coverage for stable stack (network optional) | **Done** | Medium | `.github/workflows/unit-tests.yml` | harness-smoke includes S unit tests |
| S-Ops.4 | **Legal live LLM E2E** | **Deferred** | Low | K.6 / B.15 | Not blocking harness environment |

#### S-H — Platform harness capabilities (no business agents)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-H.1 | **Platform skill bundle `harness`** — ≥3 skills (e.g. `harness.tool_smoke`, `harness.context_demo`, `harness.trace_read`) | **Done** | **Critical** | `intergrax/skills/providers/harness/`, `SKILLS.md`, bootstrap | `test_harness_skill_bundle.py` |
| S-H.2 | **Lab wiring** — `SkillProfile` + `ToolProfile` + policy bundle documented as canonical harness preset | **Done** | High | `skill_wiring.py`, `HARNESS_ENVIRONMENT.md` | lab enables `harness` bundle |
| S-H.3 | **Cursor SKILL.md importer** in gate | **Done** | Medium | `tests/unit/skills/importers/test_cursor_skill_md.py` | `pytest.mark.gate` |
| S-H.4 | **`rag.answers` test migration** — no deprecation warnings in gate | **Done** | Low | `tests/integration/rag/answers/` | `RetrievalService` only |
| S-H.5 | **Echo/signoff path** — lab run proves skills + trace + policy bundle (existing agents) | **Done** | High | `tests/acceptance/agent_os/test_lab_application.py` | gate + harness wiring tests |

#### S-Doc — Operator & author surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Doc.1 | **`HARNESS_ENVIRONMENT.md`** — lab stack, env vars, stable integrations, OTLP, policy bundle read order | **Done** | **Critical** | `docs/HARNESS_ENVIRONMENT.md`, `docs/README.md` index | Linked from plan §6 |
| S-Doc.2 | **Context / trace operator section** — `CONTEXT_*` events, debug API, metrics routes | **Done** | Medium | `HARNESS_ENVIRONMENT.md` | Pointers to canon §28.1 |

#### Phase S — Definition of done

1. **Stable** integration list for lab harness stack published and tested (S-Ops.1).
2. **OTLP path** documented and wired for lab when env configured (S-Ops.2).
3. **≥ 3** `harness.*` platform skills + legal/research bundles registered (S-H.1).
4. **`HARNESS_ENVIRONMENT.md`** complete; lab wiring matches doc (S-H.2, S-Doc.1).
5. Gate: `uv run pytest -m gate -q` green; `python scripts/check_harness_no_getattr.py` OK.
6. §0.5 **Harness environment GA** row **Done** with date; Appendix F updated.
7. **K.1/K.2 remain Deferred** — not required for Phase S close.

#### Phase S — Recommended execution order

```text
Wave S0 (docs):      S.0.* (Done)
Wave S1 (ops):       S-Ops.1 → S-Ops.2 → S-Ops.3
Wave S2 (platform):  S-H.1 → S-H.2 → S-H.3
Wave S3 (proof):     S-H.5 → S-Doc.1 → S-Doc.2
Wave S4 (cleanup):   S-H.4
Parallel:            S-Ops.4, domain skill growth (legal/research) — not required for S Done
```

**After Phase S Done (historical):** Harness environment was ready for product agents. **Scheduling (2026-06-02):** K.1/K.2 remain **§6.3 end-of-plan** until explicit product prioritization.

---

### Phase T — Harness Cleanliness (post-S 2026-06-01)

**Status:** **Done** (2026-06-01). **Prerequisites:** Phase S **Done**.  
**Goal:** Close harness technical debt — unified lab preset, typed Tier-2 agents, native catalog planner, expanded stable stack, gate hygiene — without new business agents.

| # | Deliverable | Status | Location | Acceptance |
|---|-------------|--------|----------|------------|
| T-Ops.1 | **`lab_harness_preset()`** — default lab profile (sqlite + log + lab_json + OTEL; optional redis/qdrant) | **Done** | `IntegrationProfile`, `integration_wiring.py`, `settings.py` | `test_lab_harness_preset.py` |
| T-H.1 | **Echo/signoff `skill_ids`** — `harness.tool_smoke` on `AgentContract` | **Done** | `agents/echo`, `agents/signoff_probe` | `test_harness_reference_agent_skills.py` |
| T-H.2 | **`rag.answers` gate hygiene** — gate uses `RetrievalService` only; legacy tests marked `legacy_rag_answers` | **Done** | `tests/integration/rag/answers/` | No `rag.answers` in `-m gate` |
| T-H.3 | **Typed `TaskContext` in Tier-2 agents** — no `getattr` on capability/message content in `agents/` | **Done** | echo, research, signoff, org worker, lab mocks | `check_harness_no_getattr.py` scans `agents/` |
| T-Ops.5 | **`CatalogToolPlanner`** without `ToolsAgent` wrapper | **Done** | `tool_planning_service.py`, `catalog_tool_planner.py` | `test_catalog_tool_planner.py` |
| T-Ops.6 | **Tier-2 stable stack** — `postgresql` + `sentry` in `HARNESS_LAB_STABLE_SLUGS` | **Done** | `harness_lab_stack.py`, postgresql `register.py` | `test_harness_lab_stable_stack.py` |

#### Phase T — Definition of done

1. Lab default wiring uses `lab_harness_preset()` (OTEL on unless env disables).
2. Echo and signoff_probe declare `harness.tool_smoke` via `skill_ids`.
3. Gate RAG path is `RetrievalService`-only; legacy `rag.answers` tests excluded from gate.
4. `python scripts/check_harness_no_getattr.py` passes with `agents/` in scan roots.
5. `CatalogToolPlanner` does not import `ToolsAgent`.
6. `postgresql` stable in catalog and harness stack list.

**After Phase T Done (historical):** Harness cleanliness complete. **Scheduling (2026-06-02):** product milestone K.1/K.2 is **deferred** (§6.3), not the default next step.

---

### Phase U — Harness Production Hardening (post-T 2026-06-01)

**Source:** Harness-system audit (2026-06-01) — security, contracts, policy wiring, typing, legacy, CI; **no business agents** (K.1/K.2 out of scope).  
**Status:** **Done** (2026-06-01). **Prerequisites:** Phase T **Done**. **Residual:** U-Leg.* (legacy module removal) — optional follow-up; does not block K.  
**Goal:** Close the gap between **laboratory harness** (fast iteration) and **production harness** (strategy doc: governance, persisted trace, secured surfaces, typed contracts, single policy path) without starting product agents.

**Explicitly out of scope for Phase U:**

- **K.1 Problem Radar / K.2 Vendor Discovery** — remain **Phase K** (after U Done)
- Multi-tenant SaaS (canon §50)
- New domain skills beyond harness platform pack
- Legal/product application feature work (except shared harness wiring used by lab)

#### U.0 — Audit & plan sync

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U.0.1 | **Appendix G** — audit findings → U.* IDs (100% mapped) | **Done** | Critical | This file Appendix G | Every audit row has U ID |
| U.0.2 | **§0.5 / §4 / §6** — Phase U as **NOW**; K.1/K.2 gated on U Done | **Done** | Critical | This file | No contradiction with strategy |

#### U-Sec — Lab & debug security surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Sec.1 | **AuthZ on lab surfaces** — optional API key / bearer for `POST /v1/lab/run`, `/debug/*`, MCP mount; default **deny** when `INTERGRAX_HARNESS_API_KEY` set | **Done** | **Critical** | `harness_auth.py`, lab/debug/MCP routes | `test_harness_auth.py` |
| U-Sec.2 | **MCP default opt-in** — `LAB_INCLUDE_MCP=false` default for strict profile; document in `HARNESS_ENVIRONMENT.md` | **Done** | High | `LabApplicationSettings`, `.env.example` | `test_lab_application_settings_phase_u.py` |
| U-Sec.3 | **Sandbox tool policy** — lab enables `sandbox.exec` only when `SandboxSessionManager` wired; document risk | **Done** | High | `tool_wiring.py`, harness docs | Unit: sandbox omitted without session |
| U-Sec.4 | **`strict_harness` runtime profile** — `production_mode=True`, `GovernanceService`, persisted `trace_db_path`, OTEL; env `LAB_STRICT_HARNESS=true` | **Done** | **Critical** | `lab_runtime_config.py`, lab wiring | `test_lab_strict_harness.py` |

#### U-Pol — Unified policy path (lab + Tier-1)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Pol.1 | **`apply_policy_bundle` in lab** — `build_lab_runtime_config(ctx)` applies `ApplicationBuildContext.policy_bundle` to every UAEP `RuntimeConfig` (echo, signoff, mocks) | **Done** | **Critical** | `lab_runtime_config.py`, `runtime_config_bridge.py` | Reference agents use `build_lab_agent_runtime_context` |
| U-Pol.2 | **Policy engine vs bundle** — single composition root: Nexus `policy_engine` + `RuntimeConfig.policy_bundle` documented and wired from same `build_runtime_policy_bundle()` in lab | **Done** | High | `policy_wiring.py`, lab registry | Bundle passed via `ApplicationBuildContext` |
| U-Pol.3 | **Typed `RuntimePolicyBundle`** — replace `budget: Any`, `plan_loop: Any` with concrete policy types or `Protocol` refs | **Done** | Medium | `runtime/policy/policy_bundle.py` | `BudgetPolicy` / `PlanLoopPolicy` fields |

#### U-Con — Agent / UAEP contract unification

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Con.1 | **`HarnessReferenceAgent` base** — `class HarnessReferenceAgent(Agent):` + required UAEP methods; echo/signoff/mock inherit | **Done** | **Critical** | `intergrax/agents/harness_reference_agent.py` | Echo/signoff/mocks inherit |
| U-Con.2 | **Register-time UAEP check** — `AgentRegistry.register()` rejects agents that fail `isinstance(agent, UAEPAgent)` when manifest marks `requires_uaep: true` | **Done** | High | `agent_registry.py`, lab manifest | `test_agent_registry_uaep.py` |
| U-Con.3 | **Skill runtime proof** — gate test: lab registry resolves `harness.tool_smoke` → non-empty `allowed_tools` and tool step can plan | **Done** | High | `test_harness_reference_agent_skills.py`, acceptance lab | Echo/signoff declare `harness.tool_smoke` |

#### U-Typ — Strong typing & getattr hygiene

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Typ.1 | **Fix `ToolsAgentConfig`** — remove erroneous tuple defaults (`temperature = None,`); use `@dataclass` or explicit `__init__` | **Done** | **Critical** | `intergrax/tools/tools_agent.py` | Extends `ToolPlanningConfig` |
| U-Typ.2 | **`ToolPlanningConfig` in Tier-1** — planner prompts/config in `runtime/nexus/tools/`; `ToolPlanningService` does not import `tools.tools_agent` | **Done** | High | `runtime/nexus/tools/` | `test_catalog_tool_planner.py` |
| U-Typ.3 | **`ToolPlannerTrackable` protocol** — replace `isinstance(tool_planner, CatalogToolPlanner)` in `runtime_state` | **Done** | Medium | `tool_planner_trackable.py`, `runtime_state.py` | Protocol-based LLM tracker |
| U-Typ.4 | **Extend getattr audit** — `integrations/registry/profile.py`, `sandbox/service.py` | **Done** | Medium | Typed profile + `SandboxSession` | Harness nexus/agents paths clean |
| U-Typ.5 | **Remove `hasattr` on harness paths** — `shared_task_context`, `engine_plan_models`, `platform_wiring` trace_store resolution | **Done** | Medium | `platform_wiring.py`, `nexus_loop.trace_store` | Typed trace resolution |

#### U-Arch — Integration & composition consistency

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Arch.1 | **Single lab integration preset** — `create_lab_interaction_adapter()` uses `lab_harness_preset()` (not `IntegrationProfile.lab()`) | **Done** | High | `integration_wiring.py` | `test_lab_harness_environment_wiring.py` |
| U-Arch.2 | **Typed lab wiring returns** — remove `# type: ignore` on trace/checkpoint/notification adapters; explicit bundle types | **Done** | Medium | `SQLiteIntegrationBundle`, `integration_wiring.py` | Typed sqlite facades |
| U-Arch.3 | **Rename runtime `tools_agent_*` fields** — `tools_agent_answer` → `tool_planner_answer` (or `catalog_tool_answer`); update trace diag types | **Done** | Low | `runtime_state.py`, tracing adapters | Gate green |

#### U-Leg — Legacy stack removal

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Leg.1 | **`ToolsAgent.run` deprecation freeze** — document; block new imports; optional redirect to `ToolRuntime` only path | **Done** | Medium | `tools_agent.py`, `check_tools_agent_run.py` | CI audit |
| U-Leg.2 | **`rag.answers` removal or archive** — migrate remaining `legacy_rag_answers` tests to `RetrievalService`; delete or move module under `intergrax/legacy/` | **Done** | Medium | `intergrax/legacy/rag_answers/` | `test_rag_answers_removed.py` |
| U-Leg.3 | **Legacy tool plan booleans** — document sunset for `from_legacy` / `uses_legacy_booleans_only`; gate new usage | **Done** | Low | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | Deprecation warnings |

#### U-Doc — Operator & architecture alignment

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-Doc.1 | **`HARNESS_ENVIRONMENT.md`** — security (auth, MCP), strict profile, policy bundle wiring truth | **Done** | High | `docs/HARNESS_ENVIRONMENT.md` | Phase U security section |
| U-Doc.2 | **Canon §52 / strategy** — lab vs production harness checklist references Phase U | **Won't fix** | Medium | — | Deferred; plan + HARNESS_ENVIRONMENT sufficient |
| U-Doc.3 | **Fix Phase K footer** in `HARNESS_ENVIRONMENT.md` (post-T, gated on U) | **Done** | Low | `HARNESS_ENVIRONMENT.md` | Gated on Phase U |

#### U-CI — Verification & smoke

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| U-CI.1 | **harness-smoke includes Phase U tests** — auth, strict harness, lab settings | **Done** | High | `.github/workflows/unit-tests.yml` | harness-smoke extended |
| U-CI.2 | **Acceptance: production harness path** — one gate test: strict lab + sqlite trace + policy bundle + skill-resolved tools | **Done** | **Critical** | `tests/acceptance/agent_os/`, unit strict harness | `pytest -m gate` **479 passed** |
| U-CI.3 | **Optional: strict harness job** — separate CI job with `LAB_STRICT_HARNESS=true` + API key | **Done** | Medium | `.github/workflows/unit-tests.yml` | `harness-strict` job |

#### Phase U — Definition of done

1. Lab **policy bundle** reaches `RuntimeConfig` for all reference agents (U-Pol.1); tool policy resolution exercised in test.
2. **Secured-by-configuration** lab/debug/MCP (U-Sec.1–U-Sec.2); **strict_harness** E2E exists (U-Sec.4, U-CI.2).
3. Reference agents use **HarnessReferenceAgent** or equivalent enforced UAEP (U-Con.1–U-Con.2).
4. **`ToolsAgentConfig` bug fixed**; Tier-1 planner config decoupled from `tools_agent` (U-Typ.1–U-Typ.2).
5. **Integration preset** consistent (U-Arch.1); docs accurate (U-Doc.*).
6. Gate: `uv run pytest -m gate -q` green; getattr + tools_agent audits pass.
7. §0.5 **Harness production hardening** row **Done** with date; Appendix G 100% **Done** or **Won't fix**.
8. **K.1/K.2 remain Deferred** until U Done.

#### Phase U — Recommended execution order

```text
Wave U0 (plan):     U.0.* (Done with this edit)
Wave U1 (security): U-Sec.1 → U-Sec.2 → U-Sec.4
Wave U2 (policy):   U-Pol.1 → U-Pol.2 → U-Con.3
Wave U3 (contracts): U-Con.1 → U-Con.2 → U-Typ.1
Wave U4 (typing):   U-Typ.2 → U-Typ.3 → U-Typ.4 → U-Typ.5
Wave U5 (arch):     U-Arch.1 → U-Arch.2 → U-Pol.3
Wave U6 (legacy):   U-Leg.2 → U-Leg.1 → U-Leg.3 → U-Arch.3
Wave U7 (close):    U-Doc.* → U-CI.* → Appendix G paydown log
```

**After Phase U Done (historical):** Production-grade harness baseline achieved. **Scheduling (2026-06-02):** start K.1/K.2 only via **§6.3** after explicit product decision — not by default.

---

### Phase V — Harness Architecture Hardening (post-U)

**Source:** Architecture hardening audit against `IDEAL_HARNESS_AI_ARCHITECTURE.md` (2026-06-02).  
**Status:** **Done** (2026-06-05) — Phase V-REM closed all runtime enforcement gaps. **Prerequisites:** Phase U **Done**.  
**Goal:** Close architecture-level gaps that increase long-term technical debt, reduce extensibility, or weaken governance in harness-only scope.

**Explicitly in scope for Phase V:**

- Capability dependency graph + compatibility gates
- Agent lifecycle governance (certification/promotion/deprecation/retirement/ownership)
- Context quality scoring + context regression discipline
- Prompt engineering architecture and governance
- Evaluation registry operations (offline/online/shadow/human)
- Architecture metrics and architecture debt governance
- Advanced security/data governance defenses (prompt/tool/retrieval attacks)
- Cost/resource governance (budgets, quotas, forecasting, optimization)
- Multi-agent coordination model catalog and selection matrix
- Knowledge-graph/Graph-RAG evolution path (harness capability, no product-domain rollout)

**Explicitly out of scope for Phase V:**

- K.1/K.2 business agent delivery
- New product-specific Tier-3 applications
- Domain skill packs not under `harness.*`

#### V-CG — Capability Graph Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CG.1 | Capability graph schema (nodes + edges for Integration/Tool/Skill/Policy/Agent/Application/Product) | **Done** | **Critical** | Typed schema + docs in canon |
| V-CG.2 | Graph lineage builder from registries | **Done** | High | Per-application agent→application edges via `capability_graph_applications.py` |
| V-CG.3 | Impact analysis report (blast radius) for changed capabilities | **Done** | High | Guard script green on corrected graph |
| V-CG.4 | Compatibility validation on dependency graph edges | **Done** | **Critical** | `phase_v_capability_graph_guard.py --enforce` green |

#### V-ALG — Agent Lifecycle Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-ALG.1 | Agent certification gate contract (quality/policy/security) | **Done** | **Critical** | Certification criteria codified + tested |
| V-ALG.2 | Promotion flow (dev -> staging -> production) with evidence | **Done** | High | Promotion requires evidence bundle |
| V-ALG.3 | Deprecation + retirement workflow and migration window policy | **Done** | High | `AgentRegistry` / `AgentRouter` filter retired/deprecated via `agent_routing_policy.py` |
| V-ALG.4 | Owner/on-call metadata required for production-eligible agents | **Done** | High | Production-mode ownership gate enforced at selection |

#### V-CE — Context Quality and Regression Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-CE.1 | Relevance/freshness/confidence scoring in context assembly | **Done** | High | Scores emitted in trace/runtime events |
| V-CE.2 | Duplicate suppression + context quality thresholds | **Done** | Medium | Threshold policy test coverage |
| V-CE.3 | Context regression benchmark suite | **Done** | High | CI regression baseline stored and compared |
| V-CE.4 | Retrieval effectiveness evaluation (precision/recall@k style) | **Done** | Medium | Bench report in evaluation registry |

#### V-PE — Prompt Engineering Architecture

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-PE.1 | Prompt registry governance contract (owner/version/risk metadata) | **Done** | High | `PromptMeta` extended; `harness_capability_summary` reference prompt; registry governance validation |
| V-PE.2 | Prompt composition model (system/task/policy/context layers) | **Done** | High | Canon + reference implementation path |
| V-PE.3 | Deterministic policy injection overlays | **Done** | High | Prompt build trace shows overlays |
| V-PE.4 | Prompt regression/adversarial test suite | **Done** | Medium | Gate includes prompt regression subset |

#### V-EVAL — Evaluation and Benchmarking Operations

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-EVAL.1 | Unified evaluation modes: offline/online/shadow/human | **Done** | **Critical** | Mode contracts documented + wired |
| V-EVAL.2 | Golden datasets + scenario libraries + regression suites | **Done** (typed asset bundle contracts) | High | Versioned benchmark assets |
| V-EVAL.3 | Automated evaluators (rule-based + LLM judge) | **Done** | High | Evaluator outputs persisted |
| V-EVAL.4 | Evaluation registry trend/comparison reports | **Done** | High | Report artifact required for major releases |

#### V-AM — Architecture Metrics & Debt Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-AM.1 | Architecture health metric spec (modularity/dependency/coverage/debt) | **Done** | **Critical** | Canon metrics section + thresholds |
| V-AM.2 | Metrics emission pipeline and dashboards | **Done** (pipeline + trend/gate contracts) | High | Dashboard + alert definitions |
| V-AM.3 | Governance coverage and observability coverage measurement | **Done** | High | Coverage reports generated in CI |
| V-AM.4 | Architecture debt index + periodic review process | **Done** | High | Debt report cadence defined and used |

#### V-SEC — Security & Data Governance Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-SEC.1 | Prompt injection defense profile + tests | **Done** | **Critical** | Adversarial tests in gate subset |
| V-SEC.2 | Tool injection defense (schema/argument/capability controls) | **Done** | High | `ToolInjectionDefenseMiddleware` on `BEFORE_TOOL_CALL` via `application_security_wiring.py` |
| V-SEC.3 | Retrieval poisoning defense (trust score/quarantine flow) | **Done** | High | `retrieval_security_wiring.py` filters chunks in `RagStep` when profile enabled |
| V-SEC.4 | Tenant isolation verification + security audit trail checks | **Done** | High | `TenantSecurityMiddleware` on `BEFORE_TASK_INTAKE` |

#### V-COST — Cost & Resource Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-COST.1 | Budget envelopes (tenant/app/agent/model/tool) | **Done** | High | Budget policy enforcement tests |
| V-COST.2 | Token/tool/resource quotas with deny/degrade behavior | **Done** | High | Quota exceedance behavior deterministic |
| V-COST.3 | Forecast + anomaly detection for spend and token drift | **Done** | Medium | Forecast/anomaly report available |
| V-COST.4 | Optimization recommendations with policy guardrails | **Done** | Medium | Recommendations recorded in ops reports |

#### V-MA — Multi-Agent Coordination Model Catalog

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-MA.1 | Coordination patterns catalog (hierarchical/orchestrator-worker/supervisor-worker/peer/swarm/evaluator-loop) | **Done** | High | Canon section + selection table |
| V-MA.2 | Pattern selection matrix (risk/latency/cost/complexity) | **Done** | High | Matrix used in planning docs |
| V-MA.3 | Pattern-specific acceptance tests | **Done** | Medium | Test suite covers selected patterns |

#### V-KG — Knowledge Graph Evolution Path (Harness)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-KG.1 | Graph-RAG architecture contract | **Done** | Medium | Canon section + terminology alignment |
| V-KG.2 | Hybrid retrieval reference path (vector + keyword + graph) | **Done** | Medium | Reference implementation notes |
| V-KG.3 | Graph-backed explainability trace fields | **Done** | Medium | Trace schema supports graph provenance |

#### V-V6 — Phase V Closeout (L3/L4 Evidence & CI)

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-V6.1 | Bounded adaptive governance contracts (policy-learning envelopes, human gates) | **Done** | High | `adaptive_governance.py` + unit tests |
| V-V6.2 | L3/L4 maturity gate evidence aggregator | **Done** | **Critical** | `maturity_gate_evidence.py` + `maturity_gate_evidence_report.json` |
| V-V6.3 | CI closeout gate (`phase_v_closeout_gate.py --enforce`) | **Done** | **Critical** | Regression workflow runs closeout after gate tests |

#### Phase V — Execution matrix (dependencies and order)

Phase V should be executed in dependency-aware waves:

```text
Wave V0 (planning):      V-CG.1 + V-AM.1 + ownership/cadence baseline
Wave V1 (foundations):   V-CG.2 -> V-CG.4 + V-ALG.1 + V-PE.1 + V-EVAL.1
Wave V2 (quality):       V-CE.1 -> V-CE.3 + V-PE.2 -> V-PE.4 + V-EVAL.2 -> V-EVAL.3
Wave V3 (governance):    V-ALG.2 -> V-ALG.4 + V-SEC.1 -> V-SEC.4 + V-COST.1 -> V-COST.2
Wave V4 (ops maturity):  V-AM.2 -> V-AM.4 + V-EVAL.4 + V-COST.3 -> V-COST.4
Wave V5 (advanced):      V-MA.1 -> V-MA.3 + V-KG.1 -> V-KG.3
Wave V6 (closeout):      L3/L4 gate evidence + docs sync + priority reset
```

Critical dependency rules:

- `V-CG.1` must precede `V-CG.2/V-CG.4` and dependency-health metrics in `V-AM`.
- `V-PE.1` and `V-EVAL.1` must precede prompt/eval regression gates.
- `V-ALG.1` must precede production promotion flow (`V-ALG.2`).
- `V-SEC.*` and `V-COST.*` deny/degrade behavior must be validated before L3 gate.

#### Phase V — KPI thresholds and acceptance metrics

Minimum quantitative targets for Phase V completion:

| Area | Metric | Target |
|------|--------|--------|
| Capability graph | Changed harness PRs with graph impact artifact | **>= 95%** |
| Compatibility | Graph-edge compatibility gate pass on default branch | **100% required** |
| Lifecycle governance | Production-eligible agents with owner + certification metadata | **100% required** |
| Context quality | Context regression suite pass rate | **>= 95%** |
| Prompt quality | Prompt regression/adversarial suite pass rate | **>= 95%** |
| Evaluation ops | Critical capabilities with baseline + post-change scores | **100% required** |
| Security hardening | Adversarial defense suite pass rate (prompt/tool/retrieval) | **100% required** |
| Cost governance | Budget/quota policy test pass rate | **100% required** |
| Architecture metrics | Modularity/dependency/governance/observability coverage reported | **100% runs** |
| Architecture debt | Critical debt items trending (rolling 30d) | **non-increasing** |

#### Phase V — Operating cadence and governance ceremonies

- **Weekly:** Architecture hardening triage (V-* progress, blockers, scope control).
- **Weekly:** Security/cost review for new deny/degrade paths and policy regressions.
- **Bi-weekly:** Architecture review board for high-impact V-* design changes.
- **Monthly:** Architecture debt review (index trend + mitigation decisions).
- **Per release candidate:** L3/L4 evidence review (gates below) before release approval.

#### Phase V — Stream ownership model

| Stream | Primary owner | Supporting owners |
|--------|----------------|-------------------|
| V-CG | Platform architecture | Runtime + DevEx |
| V-ALG | Runtime governance | Platform + QA |
| V-CE / V-PE | Runtime + Prompt systems | QA/Eval |
| V-EVAL | Evaluation engineering | Runtime + Product quality |
| V-AM | Platform observability | Runtime + DevEx |
| V-SEC | Security engineering | Runtime + Platform |
| V-COST | Runtime economics | Platform + FinOps |
| V-MA | Orchestration/runtime | QA |
| V-KG | Knowledge systems | Runtime + Eval |

Owner rules:

- Every V-* PR must include a single accountable owner.
- Cross-stream dependencies must list an explicit approver before merge.
- Ownership metadata for production-impacting components must be reflected in registries where applicable.

#### Phase V — L3/L4 gate evidence (architecture maturity)

L3 readiness requires:

1. `V-CG.*`, `V-ALG.*`, `V-EVAL.1-4`, `V-SEC.1-4`, `V-COST.1-2`, `V-AM.1-3` complete.
2. KPI thresholds marked **100% required** above are satisfied.
3. Security and compatibility gates are green for two consecutive release cycles.
4. Architecture governance artifacts updated (canon + plan + traceability appendices).

L4 readiness requires:

1. L3 criteria met and stable.
2. `V-COST.3-4`, `V-MA.*`, `V-KG.*`, and adaptive loops with bounded governance controls.
3. Closed-loop evaluation feedback demonstrates measurable quality/cost improvement over baseline.
4. Policy-learning/adaptive behavior remains human-governed and auditable.

#### Phase V — Definition of done

1. Capability graph compatibility validation is active in CI for harness-critical changes.
2. Agent lifecycle governance gates exist and are enforced for production-eligible agents.
3. Context/prompt/evaluation governance artifacts are versioned and regression-tested.
4. Architecture health metrics are measurable and reviewed on a recurring cadence.
5. Security/data/cost hardening controls are testable, observable, and documented.
6. All changes remain harness-only (no implicit K.1/K.2 scope creep).
7. Coverage matrix (Appendix H) has **no `Uncovered` rows** for harness-scope architecture domains.

#### Phase V — Paydown log

| Date | V ID | Summary |
|------|------|---------|
| 2026-06-02 | V-CG.1, V-AM.1, V-ALG.1 | Typed baseline contracts added (`intergrax/runtime/architecture/`) + report-only artifacts script (`scripts/phase_v_foundations_report.py`) + unit tests |
| 2026-06-02 | V-CG.2, V-CG.3, V-CG.4 | Lineage/impact/compatibility modules + capability graph guard script (`scripts/phase_v_capability_graph_guard.py`) + enforce switch + unit tests |
| 2026-06-02 | V-AM.2, V-ALG.2, V-EVAL.1 | Metrics pipeline contracts + promotion flow evaluator + unified evaluation mode contracts + governance artifacts script (`scripts/phase_v_governance_report.py`) + unit tests |
| 2026-06-02 | V-ALG.3, V-ALG.4, V-EVAL.2 | Lifecycle/deprecation governance contracts + production ownership guard + evaluation asset bundle contracts + governance report extensions + unit tests |
| 2026-06-02 | V-EVAL.3, V-AM.3 | Automated evaluators (`evaluation_automation.py`) + architecture coverage report (`architecture_coverage.py`) + governance report persistence + unit tests |
| 2026-06-02 | V-AM.4, V-EVAL.4 | Debt governance cadence/policy report (`debt_governance.py`) + release trend/comparison report (`evaluation_registry_trends.py`) + governance script artifacts + unit tests |
| 2026-06-02 | V-SEC.1, V-SEC.2 | Prompt injection defense profile (`prompt_security.py`) + tool injection defense controls (`tool_security.py`) + governance artifacts + adversarial unit tests |
| 2026-06-02 | V-SEC.3, V-SEC.4 | Retrieval poisoning defense (`retrieval_security.py`) + tenant isolation/audit verification (`tenant_security.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-COST.1, V-COST.2, V-COST.3, V-COST.4 | Budget envelopes + quota deny/degrade + cost forecast/anomaly + optimization guardrails (`cost_*.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.1, V-CE.2, V-PE.1, V-PE.2 | Context quality scoring/dedup (`context_engineering.py`) + prompt registry/composition (`prompt_registry_governance.py`, `prompt_composition.py`) + governance artifacts + unit tests |
| 2026-06-02 | V-CE.3, V-CE.4, V-PE.3, V-PE.4 | Context regression benchmark + retrieval effectiveness + policy overlays + prompt regression suite + governance artifacts + unit tests |
| 2026-06-02 | V-MA.1, V-MA.2, V-MA.3, V-KG.1, V-KG.2, V-KG.3 | Multi-agent coordination catalog/selection/acceptance + Graph-RAG/hybrid retrieval/provenance contracts + governance artifacts + unit tests |
| 2026-06-02 | V-V6.1, V-V6.2, V-V6.3 | Bounded adaptive governance + L3/L4 maturity evidence + `phase_v_closeout_gate.py` CI enforcement |
| 2026-06-03 | H-APP.* | Phase H-APP: ApplicationEnvironmentProfile, unified wiring, 43 tasks, gate 510 |
| 2026-06-05 | V-REM.0.* | Plan audit: 9 Phase V + 1 Phase A gaps reclassified Partial; Phase V-REM + Appendix J + §6.1z queue opened |
| — | — | *(append row per merged PR)* |

---

## Phase V-REM — Phase V Runtime Remediation (audit closeout)

**Source:** Plan/code audit (2026-06-05) — reconcile Phase V **Done** claims vs runtime evidence; aligned with [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) layers 5, 19, 21, 23, 25, 26.  
**Status:** **Done** (2026-06-05) — **10/10 Done**.  
**Prerequisites:** Phase V contracts **Done**; Phase H-APP **Done** (Tier-3 `ApplicationSecurityProfile` hooks exist).  
**Goal:** Close every **Partial** Phase V row and **A.4** EvalRunner gap — move from governance/evidence-only to **runtime-enforced** behavior. **Achieved 2026-06-05.**  
**Priority ladder:** **Band 2i** (§4.0) — closed.  
**Execution order:** [§6.2v](#62v-phase-v-rem-execution-order-band-2i--closed-2026-06-05).  
**Traceability:** [Appendix J](#appendix-j--phase-v-remediation-traceability-audit-gap--v-rem-id).

**Explicitly out of scope:** K.1/K.2, new product Tier-3 apps, full 32-layer re-audit (use [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) separately).

**Delivery rule:** One `V-REM.*` ID per PR → update master table + Appendix J + paydown log → `pytest -m gate` + relevant architecture scripts green.

### V-REM — Traceability (audit gap → task ID)

| Parent ID | Gap summary | V-REM ID |
|-----------|-------------|----------|
| V-CG.2–V-CG.4 | Incorrect system edge mapping agents→application breaks lineage/impact/CI | V-REM-CG.1, V-REM-CG.2 |
| V-ALG.3 | No runtime cutoff for retired/deprecated agents | V-REM-ALG.1 |
| V-ALG.4 | No production-eligible-only filter at selection | V-REM-ALG.2 |
| V-PE.1 | PromptMeta missing owner/risk; no YAML prompt assets | V-REM-PE.1, V-REM-PE.2 |
| V-SEC.2 | Tool injection defense not wired in execution path | V-REM-SEC.1 |
| V-SEC.3 | Retrieval poisoning defense not enforced per tenant/app | V-REM-SEC.2 |
| V-SEC.4 | Tenant isolation + audit trail hooks missing in main path | V-REM-SEC.3 |
| A.4 / A.4.1 | NexusEvalRunner missing integration tests + gate | V-REM-A.1 |

### V-REM — Master deliverables register

| ID | Stream | Deliverable | Status | Priority | Closes | Acceptance |
|----|--------|-------------|--------|----------|--------|------------|
| V-REM.0.1 | Governance | **Appendix J** — audit gap → V-REM ID matrix (100% mapped) | **Done** | Critical | — | Every Partial row has V-REM ID |
| V-REM.0.2 | Governance | Sync Phase V header, §0.5, §4.0 Band 2i, Appendix H, §6.1z | **Done** | High | — | No Phase V domain row marked **Done** while child Partial |
| V-REM-CG.1 | V-CG | **Fix system edge mapping** — per-application agents→application edges from manifest/roster (not global cross-product) | **Done** | **Critical** | V-CG.2–4 | Unit: lab/legal/poc graphs have correct agent-application edges |
| V-REM-CG.2 | V-CG | **Re-run graph guard** — lineage, impact, compatibility on corrected mapping; update `phase_v_capability_graph_guard.py` fixtures | **Done** | High | V-CG.2–4 | CI guard green; impact blast radius matches expected for sample change |
| V-REM-ALG.1 | V-ALG | **Runtime lifecycle filter** — AgentRegistry / NexusLoop reject or reroute retired/deprecated agents | **Done** | High | V-ALG.3 | Unit tests: deprecated agent not selected for new runs |
| V-REM-ALG.2 | V-ALG | **Production-eligible gate** — discovery/selection requires owner + certification metadata for production mode | **Done** | High | V-ALG.4 | Test: agent without owner blocked in strict/production profile |
| V-REM-PE.1 | V-PE | **Extend PromptMeta / YamlPromptRegistry** — add `owner`, `risk_tier`, version governance fields + validation | **Done** | High | V-PE.1 | Schema round-trip + registry validation tests |
| V-REM-PE.2 | V-PE | **Seed YAML prompt assets catalog** — minimal harness reference prompts under versioned assets path | **Done** | Medium | V-PE.1 | E2E governance validation passes on `harness_capability_summary` |
| V-REM-SEC.1 | V-SEC | **Wire tool injection defense** — `ApplicationSecurityProfile` → ToolRuntime / pre-tool hook on main execution path | **Done** | High | V-SEC.2 | Unit: dangerous payload denied via middleware |
| V-REM-SEC.2 | V-SEC | **Wire retrieval poisoning defense** — per-tenant/app middleware on RAG retrieval path | **Done** | High | V-SEC.3 | Unit: quarantine/trust score filters retrieval chunks |
| V-REM-SEC.3 | V-SEC | **Wire tenant isolation + audit trail** — enforcement + security audit events in UnifiedTaskRunner/NexusLoop | **Done** | High | V-SEC.4 | Unit: tenant boundary violation blocked at intake |
| V-REM-A.1 | Phase A | **NexusEvalRunner integration tests + gate** — NexusLoop→UnifiedTaskRunner→EvalRunner path | **Done** | Medium | A.4, A.4.1 | Gate tests in `tests/integration/eval/test_nexus_eval_runner.py` |

```text
Wave V-REM-0 (governance):  V-REM.0.1 -> V-REM.0.2  — Done (plan sync)
Wave V-REM-1 (graph):       V-REM-CG.1 -> V-REM-CG.2  — Done (2026-06-05)
Wave V-REM-2 (lifecycle):   V-REM-ALG.1 -> V-REM-ALG.2  — Done (2026-06-05)
Wave V-REM-3 (prompt):      V-REM-PE.1 -> V-REM-PE.2  — Done (2026-06-05)
Wave V-REM-4 (security):    V-REM-SEC.1 -> V-REM-SEC.2 -> V-REM-SEC.3  — Done (2026-06-05)
Wave V-REM-5 (eval):        V-REM-A.1  — Done (2026-06-05)
```

**Phase V-REM complete when:** All rows **Done**; parent V-CG.2–4, V-ALG.3–4, V-PE.1, V-SEC.2–4, A.4 marked **Done**; Appendix H rows updated; §6.1z queue closed. **Status: complete (2026-06-05).**

### V-REM — Paydown log

| Date | V-REM ID | Summary |
|------|----------|---------|
| 2026-06-05 | V-REM.0.1, V-REM.0.2 | Audit → plan: Phase V-REM register, Appendix J, §6.1z queue, status sync |
| 2026-06-05 | V-REM-CG.1–A.1 | Runtime remediation: capability graph edges, lifecycle routing, V-SEC wiring, prompt governance, NexusEvalRunner gate |

---

## Phase W-ML — Model & Modality Plane (Vision, Audio, Classical ML)

**Status:** **Done** (2026-06-02) — docs + implementation waves W-ML.0–W-ML.8.  
**Canon:** [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) §7.1.9, §53.13 · **Catalog:** [`MODALITY.md`](MODALITY.md) · **Ideal:** [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.5.1, §7.1, §17.

**Strategic fit:** Extends Harness AI at scale without MLOps scope creep. Same patterns as LLM adapters and Integration Library — registries, contracts, atomic tools, policy, trace, V-COST budgets.

**Explicitly in scope:**

- Three-plane modality model (generative LLM / ingest / dedicated inference).
- Extensible **vision inference engine** (YOLO/Ultralytics, ONNX Runtime, OpenVINO, TensorRT, remote Triton/TorchServe, cloud endpoints).
- `speech_provider` integrations (e.g. ElevenLabs) + TTS/STT tools.
- Classical ML registry (`ModelArtifact`, `ml.predict` tools).
- Hugging Face role separation (embeddings vs hosted inference vs hub governance).
- `ModalityProfile` for Tier-3/agent assembly.
- `modality_metrics` + cost envelope extensions.

**Explicitly out of scope:**

- Online training / AutoML / feature stores as platform products.
- LLM slugs in Integration Catalog (§44.10).
- CV or ML SDK imports in Tier-2 `agents/`.
- Monolithic “vision skills” without atomic tools.

**Dependency:** Documentation may land during Phase V; code waves SHOULD not block V closeout but SHOULD follow V-COST/V-SEC patterns.

#### W-ML — Deliverables

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| W-ML.0 | Canon §7.1.9 + §53.13 + `MODALITY.md` + IDEAL/LLM_ADAPTERS sync | **Done** | **Critical** | Docs merged; three planes documented |
| W-ML.1 | Multimodal LLM contract — `supports_vision` / audio flags; `AttachmentRef` → vendor parts | **Done** | High | Conformance tests in `tests/unit/llm_adapters/`; OpenAI + Gemini vision flags |
| W-ML.2 | `speech_provider` category + `elevenlabs` (or stub) + tools `speech.synthesize` / `speech.transcribe` | **Done** | Medium | `ElevenLabsSpeechBackend` when `ELEVENLABS_API_KEY` set; stub otherwise |
| W-ML.3 | `intergrax/model_inference/` scaffold — `VisionInferenceAdapter`, registry, `yolo_ultralytics` + `onnxruntime` slugs | **Done** | High | OpenCV contour adapter (default); optional Ultralytics; golden PNG fixture |
| W-ML.4 | Remote serving integrations — `vision_serving` / `huggingface_inference` (Triton HTTP + HF Inference API) | **Done** | Medium | `triton_vision.py`, `huggingface_inference_vision.py`; env `INTERGRAX_TRITON_URL`, `HUGGINGFACE_API_KEY` |
| W-ML.5 | `ModelInferenceAdapter` + `ml.predict` + `ModelArtifact` metadata contract | **Done** | Medium | `ml.predict` tool + stub sklearn classifier artifact |
| W-ML.6 | `ModalityProfile` + Tier-3 wiring + policy intersection with `ToolAccessPolicy` | **Done** | High | `runtime/modality/modality_profile.py` + `ToolAccessPolicy.apply_modality_profile` |
| W-ML.7 | `modality_metrics` export on `TASK_COMPLETED` + V-COST fields (`inference_ms`, `media_bytes`, `tts_characters`) | **Done** | Medium | `runtime/observability/modality_metrics.py` + metrics export |
| W-ML.8 | Capability graph nodes for modality tools + compatibility guard entries | **Done** | Low | Modality tools registered in default catalog (`register_default_tools`) |

#### W-ML — Execution waves

```text
Wave W0 (docs):       W-ML.0  — Done 2026-06-02
Wave W1 (LLM):        W-ML.1  — multimodal attachments (Plane A)
Wave W2 (speech):     W-ML.2  — speech_provider + tools
Wave W3 (vision CV):  W-ML.3  — YOLO + ONNX local inference + vision.* tools
Wave W4 (scale-out):  W-ML.4  — remote serving integrations
Wave W5 (classical):  W-ML.5  — ml.predict + ModelArtifact
Wave W6 (governance): W-ML.6 + W-ML.7 + W-ML.8 — profiles, metrics, capability graph
```

**Priority ladder placement:** Band 2 extension — run **after** critical Phase V streams (V-CG, V-SEC, V-COST) or **in parallel** with V-MA/V-KG when owners are separate. **Not** Band 3 product work.

#### W-ML — Existing assets (no rework required)

| Asset | Plane | Location |
|-------|-------|----------|
| Whisper / yt_dlp ingest | B | `integrations/providers/document_parser/` |
| Image/audio smart loaders | B | `intergrax/multimedia/`, `rag/document_loaders/` |
| HF embeddings | B | `rag/embedding/providers/hf_embedding_provider.py` |
| SPLADE sparse (optional) | B | `rag/vectorstore/sparse/splade_sparse_encoder.py` |
| LLM adapters (19 slugs) | A | `intergrax/llm_adapters/` |

#### W-ML — Paydown log

| Date | W-ML ID | Summary |
|------|---------|---------|
| 2026-06-02 | W-ML.0 | Canon §7.1.9, §53.13, `MODALITY.md`, IDEAL §3.5.1/§7.1/§17, `LLM_ADAPTERS.md` multimodal section, docs README |
| 2026-06-02 | W-ML.1–W-ML.8 | Multimodal LLM flags + attachment mapping, speech/vision/ml tools, model_inference scaffold, ModalityProfile, modality metrics, runtime governance bridge |
| 2026-06-02 | W-ML.2–W-ML.3, W-ML.6 | Lab harness modality tool wiring, OpenCV/ElevenLabs backends, golden vision fixture, `RuntimeConfig.modality_profile` |
| 2026-06-02 | W-ML.4+ | Triton/HF vision adapters, `vision.segment`/`vision.ocr_regions`/`ml.explain`, `harness.vision_qa`, extended `ModalityProfile`, legal `LEGAL_ENABLE_MODALITY_TOOLS` |
| 2026-06-02 | W-ML.workers | `ModalityExecutionProfile`, thread-pool executor, `ml.batch_predict`, `harness.modality_smoke`, `max_media_bytes` enforcement |
| 2026-06-02 | W-ML.celery | `CeleryModalityInferenceExecutor`, serialized modality jobs, trace `modality_metrics` on `tool_invocation_end`, aggregated export |
| 2026-06-02 | W-ML.metrics+ | Typed `ModalityInvocationCounters`, `media_bytes`/`tts_characters`/`ml_predictions` recording, message_bus Celery registration, capability graph modality `COMPATIBLE_WITH` edges |
| 2026-06-03 | W-ML.7b | `TASK_COMPLETED` payload includes aggregated `modality_metrics` via `NexusRuntimeEventPublisher` + `RunTraceReader` |

---

## Phase W-OPS — Operational Harness Maturity (IDEAL L3 ops)

**Status:** **Done** (2026-06-06) — W-OPS.1–W-OPS.15 delivered including W-OPS.10 lab stack health probes; **operational L3** sign-off still requires `W_OPS_RELEASE_CYCLES>=2` (or `build/architecture_hardening/release_cycles.json`) via `phase_w_ops_evidence.py --enforce`.  
**Source:** Harness maturity audit (2026-06-02; conversation) · [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) §12.3–§12.4 · [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md)  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, §4.1 **Done**.  
**Goal:** Close the gap between **L3 CI evidence** (`maturity_gate_evidence`, relaxed thresholds) and **L3 operational** (IDEAL critical areas Policy/Reliability/Observability ≥ 3 with release evidence).  
**Out of scope:** K.1, K.2, new product Tier-3 apps, domain/product skills (Band 3 · §6.3).

**Audit verdict (harness-only):** Intergrax is **L2+ scalable harness** with strong Tier-0 catalogs and Nexus §42; default implementation queue is **§6.1 + §6.2w**, not product agents.

#### W-OPS — Deliverables

| # | Deliverable | Status | Priority | Location / acceptance |
|---|-------------|--------|----------|------------------------|
| W-OPS.0 | Plan traceability from maturity audit | **Done** | — | This phase + §6.2w + doc model row |
| W-OPS.1 | **Side-effect idempotency** — `IdempotentToolInvoker` + `idempotency_key` on `ToolExecutionRequest` | **Done** | **Critical** | `runtime/tools/idempotent_invoker.py`; gate `test_idempotent_invoker.py` |
| W-OPS.2 | **Integration circuit breaker** — `IntegrationCircuitBreaker` in `integrations/_shared/` | **Done** | **Critical** | `IntegrationDependencyError`; `test_integration_circuit_breaker.py` |
| W-OPS.3 | **Reliability gate tests** — long-running scheduler / checkpoint in gate | **Done** | High | `test_long_running_scheduler_j4.py` (`pytest -m gate`) |
| W-OPS.4 | **SLO catalog + incident budget** — harness SLIs + runbook stubs | **Done** | **Critical** | `HARNESS_ENVIRONMENT.md` § Harness SLO catalog |
| W-OPS.5 | **L3-ops evidence artifact** — distinct from V-V6 CI gate | **Done** | **Critical** | `phase_w_ops_evidence.py`; `record_harness_release_cycle.py`; `release_cycles.json` |
| W-OPS.6 | **`tenant_id` on execution path** — required on `RuntimeRequest`; trace/events scoped | **Done** | High | `runtime/nexus/engine/runtime.py`; `RuntimeState.tenant_id` |
| W-OPS.7 | **Mandatory harness auth** — stage/prod/strict require `INTERGRAX_HARNESS_API_KEY` | **Done** | High | `LabApplicationSettings.requires_harness_api_key`; `test_lab_harness_api_key_required.py` |
| W-OPS.8 | **`harness.*` skill expansion** — `harness.reliability_smoke`, `harness.policy_smoke` | **Done** | Medium | `skills/providers/harness/manifests.py` |
| W-OPS.9 | **`requires_skills` adoption** — `harness.stack_demo` | **Done** | Medium | `test_harness_requires_skills_demo.py` |
| W-OPS.10 | **Harness lab stack health** — per-slug probes + circuit breaker | **Done** | Medium | `health_check_catalog_slugs`, `harness_lab_health.py`; `test_harness_lab_health.py` |
| W-OPS.11 | **Online evaluation path** — shadow observations → evaluation trends | **Done** | Medium | `online_evaluation_trend.py`, `export_harness_shadow_eval_trend.py`; file registry + RuntimeEngine hook |
| W-OPS.12 | **W-ML Celery scale-out (optional)** — env-driven via `wire_modality_extras` | **Done** | Low | `INTERGRAX_MODALITY_EXECUTION=celery`; documented in HARNESS_ENVIRONMENT |
| W-OPS.13 | **ToolsAgent removal roadmap** — CI blocks new imports; module frozen | **Done** | Low | `check_tools_agent_imports.py`, `check_tools_agent_run.py` |
| W-OPS.14 | **Typed Tier-3 wiring** — `load_callable` uses module namespace (no `getattr`) | **Done** | Low | `applications/_shared/wiring.py` |
| W-OPS.15 | **Architecture metrics enforcement (phased)** — tightened V-V6 thresholds | **Done** | Low | `maturity_gate_evidence.collect_harness_governance_signals` |

#### W-OPS — Execution waves (dependency order)

```text
Wave W-OPS-0 (governance):  W-OPS.0  — Done (audit → plan)
Wave W-OPS-P0 (critical):   W-OPS.1 → W-OPS.2 → W-OPS.3 → W-OPS.4 → W-OPS.5 → W-OPS.6 → W-OPS.7
Wave W-OPS-P1 (extend):     W-OPS.8 → W-OPS.9 → W-OPS.10 → W-OPS.11 → W-OPS.12 (optional)
Wave W-OPS-P2 (hygiene):    W-OPS.13 → W-OPS.14 → W-OPS.15
```

**IDEAL §12.3 gate:** Do not declare **operational L3** until W-OPS-P0 is **Done** and W-OPS.5 records **two consecutive release cycles** within SLO/incident budget (W-OPS.4).

**Delivery rule:** One **W-OPS.\*** ID per PR → update this table + paydown log → `pytest -m gate` + harness audit scripts (§6.1).

#### W-OPS — Paydown log

| Date | W-OPS ID | Summary |
|------|----------|---------|
| 2026-06-02 | W-OPS.0 | Maturity audit → Phase W-OPS + §6.2w execution order in implementation plan |
| 2026-06-06 | W-OPS.1–W-OPS.15 | Circuit breaker, idempotency gate, SLO docs, ops evidence script, staging API key, harness skills, online eval, wiring/metrics |
| 2026-06-02 | OPS-L3.1 | `phase_w_ops_evidence.py` Windows pytest argv + shadow trend probe; `--enforce` green |
| 2026-06-02 | REG / §6.1 | `doctor --ci` green: research `ToolEnablementProfile` protocol; lab factory via `bootstrap_lab_integration_wiring` |
| 2026-06-03 | W-OPS.10–W-OPS.11 | Lab stack health by catalog slug; shadow eval wired in `RuntimeEngine`; CI `phase_w_ops_evidence.py`; gate **470** |
| 2026-06-03 | W-OPS.5/11 | File-backed shadow eval registry; `record_harness_release_cycle.py`; extended ops evidence checks |
| 2026-06-03 | §6.1 / N.9 | Product scaffold `legal_product()` manifest + catalog bootstrap; gate **470** |
| 2026-06-03 | W-OPS.11 | Shadow eval trend export + `--verify-gate` on release cycle recorder |
| — | — | *(append row per merged PR)* |

---

## Phase H-APP — Tier-3 Application Environment (full configurability)

**Status:** **Done** (2026-06-03) — **43** deliverables; memory bridge via Phase MEM **Done**; source audit: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, **W-OPS**, §4.1 **Done**.  
**Goal:** Close every **Partial** / **Gap** topic from the harness application-layer audit — full Tier-3 configurability of agent workspaces via `ApplicationEnvironmentProfile` and unified wiring (IDEAL §17), **without** Band 3 product agents (K.1/K.2).
**Priority ladder:** **Band 2e** (§4.0) — default implementation queue after §6.1 maintenance.  
**Execution order:** [§6.2x](#62x-phase-h-app-execution-order-band-2e--active).

**Delivery rule:** One `H-APP.*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Out of scope (audit §7.7 — not counted in 43):** integration marketplace UI, catalog hot-reload, skill-as-LangGraph-pack, IDEAL L4 policy learning, new Tier-0 integration categories without §5.2.4 RFC, K.1/K.2 business agents.

```text
Wave H0 — Docs & hygiene (5 tasks)
Wave H1 — ApplicationEnvironmentProfile + unified wiring (8 tasks)
Wave H2 — Identity, policy DSL, execution modes, V-SEC app hooks (8 tasks)
Wave H3 — Orchestration factory: graph spec, shadow/sandbox, Nexus composition (6 tasks)
Wave H4 — Context/Memory/Reliability/Observability profiles (8 tasks)
Wave H5 — Migrate all Tier-3 hosts + scaffold (5 tasks)
Wave H6 — Operational L3 sign-off (3 tasks)
Total: 43
```

### H-APP — Traceability (audit section → task IDs)

| Audit § | Topic | Task IDs |
|---------|--------|----------|
| §1 | Terminology harness vs application vs agent | H-APP.0.1–H-APP.0.2 |
| §2.3.2 | Identity ABAC/RBAC per application | H-APP.2.1–H-APP.2.3 |
| §2.3.3, §3.4 | Policy DSL, execution modes, V-SEC per app | H-APP.2.4–H-APP.2.8 |
| §2.3.4, §3.5 | Orchestration graph spec, Nexus factory | H-APP.3.1–H-APP.3.6 |
| §2.3.5, §3.6 | LLMProfile on application manifest | H-APP.1.3, H-APP.1.6 |
| §2.3.7, §3.6 | ContextProfile, MemoryProfile | H-APP.4.1–H-APP.4.4 |
| §2.3.8, §3.8 | ReliabilityProfile | H-APP.4.5–H-APP.4.7 |
| §3.1 | Typed composition, no getattr in hosts | H-APP.0.3, H-APP.5.4 |
| §3.3 | Skill/tool permission consistency | H-APP.1.7, H-APP.0.4 |
| §3.5 | Shadow workspace + sandbox wiring | H-APP.3.4–H-APP.3.5 |
| §3.7 | Product observability profile (optional debug) | H-APP.4.8 |
| §4 | Operational L3 release evidence | H-APP.6.1–H-APP.6.2 |
| §5 | Registry bypass prevention | H-APP.0.4 |
| §6 | EnvironmentProfile recommendation | H-APP.1.1–H-APP.1.5 |
| §6 (follow-up) | Per-app migration checklist | H-APP.5.1–H-APP.5.3 |

### H-APP — Master deliverables register (all 43 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| H-APP.0.1 | H0 | **Harness terminology glossary** — Harness vs Tier-1 Nexus vs Tier-3 Application vs Tier-2 Agent vs Product; map to IDEAL §0.2 chain | **Done** | Medium | `intergrax_runtime_architecture.md` §5.3 + `IDEAL_HARNESS_AI_ARCHITECTURE.md` §26 cross-link |
| H-APP.0.2 | H0 | **Author guide: environment vs agent** — what belongs in `applications/` vs `agents/`; forbidden patterns | **Done** | Medium | `EXTENSION_AUTHOR_GUIDE.md` or `AGENT_CREATION_GUIDE.md` |
| H-APP.0.3 | H0 | Fix `poc_template_application/host/wiring.py` — `manifest.integration_profile` (no `getattr`) | **Done** | High | Typed access; gate test |
| H-APP.0.4 | H0 | **`check_agent_registry_bypass.py`** — CI fails if Tier-2 agents import integrations/tools directly | **Done** | High | `scripts/` + `pytest -m gate` |
| H-APP.0.5 | H0 | **Conformance test** — `ApplicationManifest` + `ApplicationBuildContext` round-trip (lab/legal/poc) | **Done** | High | `tests/unit/applications/test_manifest_conformance.py` |
| H-APP.1.1 | H1 | **`ApplicationEnvironmentProfile`** Pydantic model aggregating Tool/Skill/Modality/Policy/LLM/Context/Memory/Reliability/Observability/Orchestration/Identity profiles + `ApplicationFeatures` | **Done** | **Critical** | `intergrax/applications/contracts/environment_profile.py` |
| H-APP.1.2 | H1 | Extend **`ApplicationManifest`** with optional `environment` + `environment_defaults()` for `lab` / `product` | **Done** | **Critical** | `applications/contracts/manifest.py` |
| H-APP.1.3 | H1 | **`LLMProfile` slot** on environment — default adapter unless agent factory overrides | **Done** | High | Field + validation; no Tier-3 business logic |
| H-APP.1.4 | H1 | **`wire_application_environment(ctx, profile)`** — single Tier-3 entry for catalogs, modality, policy, tool/skill registries | **Done** | **Critical** | `applications/_shared/environment_wiring.py` |
| H-APP.1.5 | H1 | **`materialize_runtime_config(request, harness_ctx, env)`** — environment → `RuntimeConfig` | **Done** | **Critical** | `applications/_shared/runtime_config_bridge.py` |
| H-APP.1.6 | H1 | **`resolve_llm_adapter(env, agent_override)`** — precedence: agent factory > environment > platform default | **Done** | High | Typed resolver; unit tests |
| H-APP.1.7 | H1 | **`EnvironmentSkillToolConsistencyCheck`** — fail/warn if contract tools/skills not subset of environment | **Done** | High | `applications/_shared/conformance.py` |
| H-APP.1.8 | H1 | Gate tests: lab manifest + full `ApplicationEnvironmentProfile` | **Done** | High | `tests/unit/applications/test_environment_profile.py` |
| H-APP.2.1 | H2 | **`IdentityProfile`** — API key, tenant_required, role_claims_header, service_identities | **Done** | High | Part of `ApplicationEnvironmentProfile` |
| H-APP.2.2 | H2 | **`wire_application_identity(app, profile)`** — harness auth from profile | **Done** | High | `applications/_shared/identity_wiring.py` |
| H-APP.2.3 | H2 | **`ApplicationScopePolicy`** Protocol + static implementation — roles/scopes → tool_id / agent_id | **Done** | Medium | `applications/contracts/` or `runtime/identity/` |
| H-APP.2.4 | H2 | **`PolicyRulesProfile`** — declarative YAML/JSON rules + typed handler registry (no eval/getattr) | **Done** | **Critical** | `runtime/policy/rules/` + schema |
| H-APP.2.5 | H2 | **`ExecutionMode`** enum: STRICT \| BALANCED \| EXPLORATORY → RuntimePolicies defaults | **Done** | High | `applications/contracts/execution_mode.py` |
| H-APP.2.6 | H2 | **`wire_policy_bundle(env)`** merges rules + fragments + ExecutionMode | **Done** | High | Extend `policy_wiring.py` |
| H-APP.2.7 | H2 | **`ApplicationSecurityProfile`** — per-app V-SEC toggles (prompt/tool/retrieval/tenant) | **Done** | Medium | Bridge to `runtime/architecture` V-SEC |
| H-APP.2.8 | H2 | Lab reference: `policy/rules/harness_lab.yaml` | **Done** | Low | `applications/lab_application/policy/` + test |
| H-APP.3.1 | H3 | **`OrchestrationProfile`** — planner/classifier kinds, retry, long_running, max_delegation_depth | **Done** | High | Typed fields on environment |
| H-APP.3.2 | H3 | **`ApplicationGraphSpec`** — declarative multi-agent topology validated against roster | **Done** | High | `applications/contracts/graph_spec.py` |
| H-APP.3.3 | H3 | **`build_nexus_loop_from_environment(registry, integrations, env)`** | **Done** | **Critical** | `applications/_shared/nexus_factory.py` |
| H-APP.3.4 | H3 | **`wire_shadow_workspace(env)`** — ShadowWorkspaceManager paths, quotas, retention | **Done** | High | `applications/_shared/shadow_wiring.py` |
| H-APP.3.5 | H3 | **`wire_sandbox_sessions(env)`** — SandboxSessionManager + conditional `sandbox.exec` | **Done** | High | `applications/_shared/sandbox_wiring.py` |
| H-APP.3.6 | H3 | Integration test: lab graph spec echo → mock chain + trace | **Done** | Medium | `tests/integration/applications/test_lab_graph_spec.py` |
| H-APP.4.1 | H4 | **`ContextProfile`** — assembly options, budget presets, RAG/web toggles | **Done** | High | Pydantic model |
| H-APP.4.2 | H4 | **`MemoryProfile`** — user/org/long-term flags, retention, scope boundaries | **Done** | High | Pydantic model |
| H-APP.4.3 | H4 | Wire context/memory into `materialize_runtime_config` | **Done** | High | Phase MEM **MEM-1.*** — `memory_runtime_bridge.py`, `memory_wiring.py` |
| H-APP.4.4 | H4 | **`wire_task_memory_from_profile(env)`** — unify task memory under environment | **Done** | Medium | `_shared/task_memory_wiring.py` |
| H-APP.4.5 | H4 | **`ReliabilityProfile`** — idempotency, circuit breaker, checkpoint, scheduler | **Done** | High | Pydantic model |
| H-APP.4.6 | H4 | Apply reliability to `NexusLoop` + `RuntimeConfig` + integration circuit breaker | **Done** | High | `nexus_factory.py` |
| H-APP.4.7 | H4 | Gate test: long-running + idempotency via environment only | **Done** | Medium | `tests/unit/applications/test_reliability_profile.py` |
| H-APP.4.8 | H4 | **`ObservabilityProfile`** — trace, OTEL, metrics plugins, optional product debug surface | **Done** | Medium | Product hosts read-only debug option |
| H-APP.5.1 | H5 | **`lab_application`** — `build_lab_environment_profile` + refactor wiring/factory to unified environment | **Done** | **Critical** | No regression; gate + smoke |
| H-APP.5.2 | H5 | **`legal_application`** + **`research_application`** — product environment defaults + domain fragments | **Done** | High | Legal modality + skill bundles preserved |
| H-APP.5.3 | H5 | **`poc_template_application`** + **`docker_verify_application`** — environment template | **Done** | High | Scaffold emits profile stub |
| H-APP.5.4 | H5 | **Migration checklist** — per-file before/after (see table below) | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` §7.6 + this phase |
| H-APP.5.5 | H5 | **`intergrax scaffold new-application`** — `environment_profile.py`, `policy/rules/`, wired manifest | **Done** | Medium | CLI parity with H-APP.1 |
| H-APP.6.1 | H6 | Record **2 release cycles** via `record_harness_release_cycle.py --verify-gate` | **Done** | **Critical** | `build/architecture_hardening/release_cycles.json` |
| H-APP.6.2 | H6 | CI job: `phase_w_ops_evidence.py --enforce` on release tags | **Done** | High | `.github/workflows/` |
| H-APP.6.3 | H6 | Mark Operational L3 **Signed off** in audit §4 with dates | **Done** | Low | `HARNESS_APPLICATION_LAYER_AUDIT.md` after H-APP.6.1 |

### H-APP — Per-application migration checklist (H-APP.5.4)

| Application | Files to refactor | Must wire via environment |
|-------------|-------------------|---------------------------|
| `lab_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py`, `host/integration_wiring.py` | Full lab profile + harness tools + modality + plugins |
| `legal_application` | `host/wiring.py`, `host/factory.py`, `host/tool_wiring.py` | Product profile + legal skill bundle + optional modality |
| `research_application` | `host/wiring.py`, `host/factory.py` | Product profile + research agents roster |
| `poc_template_application` | `host/wiring.py`, `host/factory.py` | Minimal product/lab selectable template |
| `docker_verify_application` | `host/factory.py` | CI-oriented slim profile |

### H-APP — Explicitly deferred (not in the 43-task register)

| Topic | Reason |
|-------|--------|
| Integration marketplace UI | Out of P-Ext / audit §3.8 scope |
| Catalog hot-reload | Out of P-Ext scope |
| LangGraph skill packs | Separate initiative |
| IDEAL L4 adaptive / policy learning | IDEAL §25; not Band 2e |
| New Tier-0 integration categories | Requires canon §5.2.4 RFC (H-APP.0.2 documents process) |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |

### H-APP — Paydown log

| Date | H-APP ID | Summary |
|------|----------|---------|
| — | — | *(append row per merged PR)* |

**Suggested PR order:** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

---

## Phase DX — Developer Authoring Experience (fast environment + agent builds)

**Status:** **Done** (2026-06-02) — **47/47** deliverables **Done** in master table; gate **533+ passed**.  
**Prerequisites:** Phase **H-APP** **Done** (typed `ApplicationEnvironmentProfile`, `wire_application_environment`, `build_harness_host_runtime`). Phases **N**, **P-Ext**, **S** scaffold baseline **Done**.  
**Goal:** Make building **Tier-3 application environments** and **Tier-2 agents** trivial for Python developers — LangGraph-like mental model (state/steps → graph → run), **measurable** time-to-first-run (TTFRun), progressive disclosure (minimal → standard → production), and **UI-ready** serialized specs for Phase 2 (non-developer environment builder).  
**Priority ladder:** **Band 2f** (§4.0) — **closed for core path**; residual IDs are **infrastructure** follow-ups, not Band 3.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2y](#62y-phase-dx-execution-order-band-2f--mostly-done).

**Delivery rule:** One `DX-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Strategic split:**

| Phase | Audience | Outcome |
|-------|----------|---------|
| **DX (this phase)** | Python developers | Import contracts → define typed agents → configure environment → run HTTP/MCP in minutes |
| **Phase 2 (future — not DX)** | Business users via UI | Visual builder over same Pydantic/YAML specs (`DX-7.*` prepares artifacts only) |

**Target metrics (enforced by DX-3.5, DX-8.1):**

| Metric | Baseline (2026-06-03) | Target after DX |
|--------|----------------------|-----------------|
| **TTFRun** (scaffold → successful `POST …/run`) | ~45–90 min (docs + wiring) | **≤15 min** guided; **≤60 s** CI smoke |
| Author-edited files (hello world) | ~12–25 | **≤4** (`--minimal`) |
| Author LOC (excluding generated boilerplate) | ~200–400 | **≤120** |
| Commands to first run | 3+ | **1** (`intergrax run`) |
| Scaffold H-APP alignment | Partial (`factory.py` legacy path) | **100%** |

**LangGraph mapping (author mental model — implement in DX-0.2):**

| LangGraph | Intergrax (DX target) |
|-----------|------------------------|
| `State` fields | `AgentContract` + `RuntimeExecutionContext.metadata` |
| Node function | `@step` / `run_step` on `IntergraxAgent` |
| Conditional edges | `decide_after_step` → `AgentDecision` |
| `StateGraph.compile()` | `AgentGraph.build()` → `ApplicationGraphSpec` |
| `app.invoke()` | `HarnessApplication.serve()` / `POST /v1/…/run` |

**Out of scope (not counted in 47):** Band 3 product agents (K.1/K.2); visual graph editor UI; integration marketplace; catalog hot-reload; renaming `applications/` → `harness/` (canon §5.3.0 — **Application** = Tier-3 instance, **Harness** = platform); LangGraph skill pack import.

```text
Wave DX0 — Docs & traceability (4 tasks)
Wave DX1 — Scaffold/H-APP alignment fix (6 tasks) — P0 before facades
Wave DX2 — Authoring facades: HarnessApplication, AgentGraph, IntergraxAgent (6 tasks)
Wave DX3 — Minimal path + CLI + TTFRun gates (6 tasks)
Wave DX4 — Integration presets & picker (4 tasks)
Wave DX5 — Host hooks, YAML, observability/logging DX (8 tasks)
Wave DX6 — Tier hygiene + external projects (5 tasks)
Wave DX7 — UI engine prep: JSON Schema, spec versioning, catalog feed (5 tasks)
Wave DX8 — DX metrics & CI guards (3 tasks)
Total: 47
```

### DX — Traceability (audit gap → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| L1 | Scaffold generates legacy + H-APP wiring in parallel | DX-1.1–DX-1.2, DX-1.6, DX-8.3 |
| L2 | No minimal hello harness (1–3 files) | DX-3.1, DX-2.1–DX-2.3, DX-3.2 |
| L3 | No fluent graph API | DX-2.2, DX-7.3 |
| L4 | No `HarnessApplication` / single entry class | DX-2.1, DX-2.6, DX-5.1–DX-5.2 |
| L5 | Monorepo-only (`pythonpath`) | DX-6.3–DX-6.5 |
| L6 | Tier-2 agents import `applications/_shared` | DX-6.1–DX-6.2 |
| L7 | IntegrationProfile slot knowledge burden | DX-4.1–DX-4.3, DX-4.2 |
| L8 | No `intergrax run` / `intergrax doctor` / TTFRun metric | DX-3.2–DX-3.3, DX-3.5, DX-8.1–DX-8.2 |
| L9 | Documentation sprawl, no single 15-min path | DX-0.1–DX-0.4, DX-3.6 |
| L10 | No JSON Schema / stable spec for UI phase 2 | DX-7.1–DX-7.5 |
| H-APP.5.3 gap | `poc_template` / scaffold `factory.py` not on `build_nexus_loop_from_environment` | DX-1.1, DX-1.3 |
| §6 responsibility table | Agent vs environment concerns split | DX-0.3 |
| Progressive disclosure | minimal → standard → production | DX-0.4, DX-3.4 |
| Architecture audit rec. | Product observability preset, trace_id in logs, event catalog | DX-5.5–DX-5.7 |
| Architecture audit rec. | Policy handler plugins (extend without core PR) | DX-5.8 |
| Do not weaken tiers | Doctor/checks enforce boundaries | DX-0.3, DX-3.3, DX-6.2, DX-8.3 |

### DX — Master deliverables register (all 47 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| DX-0.1 | DX0 | **Phase DX register** in this plan + doc model row (§Documentation model) | **Done** | Low | This section + §6.2y |
| DX-0.2 | DX0 | **LangGraph ↔ Intergrax mapping** table (state, nodes, edges, compile, invoke) | **Done** | High | `EXTENSION_AUTHOR_GUIDE.md` §0 or `AGENT_CREATION_GUIDE.md` §1 |
| DX-0.3 | DX0 | **Responsibility matrix** — what belongs in agent vs environment (single canonical table) | **Done** | High | `EXTENSION_AUTHOR_GUIDE.md` §0 + cross-link canon §5.3.0 |
| DX-0.4 | DX0 | **Progressive disclosure** doc — minimal (`--minimal`) → standard scaffold → production (`expand`, Docker, MCP) | **Done** | Medium | `AGENT_CREATION_GUIDE.md` Step 4E § E.0 + `applications/USAGE.md` |
| DX-1.1 | DX1 | **Scaffold `factory.py`** — build `NexusLoop` only via `build_nexus_loop_from_environment(registry, env, …)` + integration bundle from `wire_application_environment` | **Done** | **Critical** | `intergrax/scaffold/new_application.py`, `new_application_product.py` |
| DX-1.2 | DX1 | **Scaffold default output** — remove generated `integration_wiring.py` + `tool_wiring.py`; retain via `--full` flag only | **Done** | **Critical** | Scaffold CLI + README in generated app |
| DX-1.3 | DX1 | **Migrate `poc_template_application/host/factory.py`** to H-APP factory pattern (no parallel legacy wiring) | **Done** | High | Parity with `lab_application`; gate smoke |
| DX-1.4 | DX1 | **Audit + fix** `legal_application` / `research_application` factories — single env path, no duplicate tool/integration wiring | **Done** | High | Host smoke tests green |
| DX-1.5 | DX1 | **Scaffold manifest** — embed `environment: ApplicationEnvironmentProfile…` at generation (not only lazy `environment_profile.py` fallback) | **Done** | High | Generated `manifest.py` |
| DX-1.6 | DX1 | **Gate test** — scaffold output: `factory.py` must not import `host.tool_wiring` / `host.integration_wiring` unless `--full` | **Done** | High | `tests/unit/scaffold/test_scaffold_harness_alignment.py` |
| DX-2.1 | DX2 | **`HarnessApplication` facade** — `.agents()`, `.integrations()`, `.graph()`, `.mode()`, `.llm()`, `.hooks()`, `.build()`, `.serve()` | **Done** | **Critical** | `intergrax/harness/app.py` |
| DX-2.2 | DX2 | **`AgentGraph` fluent builder** — nodes, edges, default agent, `on_error(retry=…)` → `ApplicationGraphSpec` | **Done** | **Critical** | `intergrax/applications/contracts/graph_builder.py` |
| DX-2.3 | DX2 | **`IntergraxAgent` base** + **`@step` decorator** — generates UAEP `get_steps` / `run_step` wiring | **Done** | **Critical** | `intergrax/agents/authoring/` |
| DX-2.4 | DX2 | **Decision helpers** — `continue_to()`, `complete()`, `delegate_to()` wrapping `AgentDecision` | **Done** | Medium | `intergrax/agents/authoring/decisions.py` |
| DX-2.5 | DX2 | **Unit test** — minimal `HarnessApplication` + `EchoAgent`/`IntergraxAgent` runs offline (no network) | **Done** | High | `tests/unit/harness/test_harness_application_minimal.py` |
| DX-2.6 | DX2 | **Public package** `intergrax.harness` — stable imports documented in author guide | **Done** | High | `intergrax/harness/__init__.py` |
| DX-3.1 | DX3 | **`new-stack --minimal`** — ≤4 author-facing files + smoke test (no Docker/MCP by default) | **Done** | **Critical** | `intergrax/scaffold/new_stack.py`, `new_application.py` `--minimal` |
| DX-3.2 | DX3 | **`intergrax run <module>:app`** — load `.env`, uvicorn, print route + sample curl | **Done** | High | `intergrax/cli/run.py` + `scaffold/cli.py` |
| DX-3.3 | DX3 | **`intergrax doctor`** — tier import violations, manifest/env conformance, scaffold freshness hint, TTFRun estimate | **Done** | High | `intergrax/cli/doctor.py` |
| DX-3.4 | DX3 | **`intergrax scaffold expand`** — promote minimal app → standard (Docker, MCP, debug, BUILD_AND_DEPLOY) | **Done** | Medium | `intergrax/scaffold/expand_application.py` |
| DX-3.5 | DX3 | **Acceptance test** `test_minimal_stack_ttf_run` — scaffold minimal → pytest → HTTP run **≤60s** in CI | **Done** | High | `tests/acceptance/dx/test_minimal_stack_ttf_run.py` |
| DX-3.6 | DX3 | **15-minute quickstart** — single numbered path: `new-stack --minimal` → edit agent → `intergrax run` → curl | **Done** | High | `AGENT_CREATION_GUIDE.md` Step 4E § E.0 |
| DX-4.1 | DX4 | **`IntegrationProfile` presets** — `.lab_stack()`, `.legal_stack()`, `.data_stack()`, `.observability_stack()` (typed, documented slugs) | **Done** | High | `intergrax/integrations/registry/presets.py` |
| DX-4.2 | DX4 | **`intergrax integrations pick`** CLI — emit profile fragment (postgres, redis, s3, prometheus, …) for `environment_profile.py` | **Done** | Medium | `intergrax/cli/integrations_pick.py` |
| DX-4.3 | DX4 | **Preset catalog table** in `INTEGRATIONS.md` + `EXTENSION_AUTHOR_GUIDE.md` | **Done** | Medium | `INTEGRATIONS.md` § Named integration presets |
| DX-4.4 | DX4 | **Gate tests** — each preset resolves with in-memory/sqlite stubs (no network) | **Done** | High | `tests/unit/integrations/test_integration_presets.py` |
| DX-5.1 | DX5 | **`ApplicationHost` Protocol/base** — override methods for environment control (intake, agent selection, finalize, error) | **Done** | High | `intergrax/harness/application_host.py` |
| DX-5.2 | DX5 | **Map host overrides → `HookPoint`** + optional `RuntimeEventBus` subscribe API on `HarnessApplication` | **Done** | High | Bridge in `intergrax/harness/hooks.py` |
| DX-5.3 | DX5 | **`HarnessApplication.from_yaml(path)`** — load `ApplicationEnvironmentProfile` + roster from `env.yaml` | **Done** | Medium | `intergrax/harness/yaml_loader.py` |
| DX-5.4 | DX5 | **Optional `agents.yaml`** — declarative `AgentBinding` list validated against importable classes | **Done** | Low | Same loader; schema test |
| DX-5.5 | DX5 | **Product scaffold observability preset** — `ObservabilityProfile` template (trace + optional read-only debug) | **Done** | Medium | `new_application_product.py` `environment_profile.py` (`otel_enabled`, debug override) |
| DX-5.6 | DX5 | **Structured log correlation** — inject `trace_id` / `run_id` in FastAPI middleware (lab + product factories) | **Done** | Medium | `intergrax/applications/_shared/logging_middleware.py` |
| DX-5.7 | DX5 | **Runtime event catalog table** — `RuntimeEventType` → emit phase → ops filter hints in canon §42 | **Done** | Low | `intergrax_runtime_architecture.md` §42.1.5; `phase_coverage.EVENT_OPS_FILTER_HINTS` |
| DX-5.8 | DX5 | **Policy rule handler plugins** — entry point group `intergrax.policy_rules` (mirror P-Ext pattern) | **Done** | Medium | `runtime/policy/rules/` + author guide § |
| DX-6.1 | DX6 | **`intergrax.agents.defaults`** — `harness_production_mode`, lab runtime config helpers (no Tier-3 import from agents) | **Done** | High | `intergrax/agents/defaults.py`; Tier-3 re-export in `runtime_defaults.py` |
| DX-6.2 | DX6 | **Fix reference agents** — `echo`, `research` (and scaffold template) must not import `applications/_shared` | **Done** | High | `agents/echo/`, `agents/research/` + `check_agent_registry_bypass` |
| DX-6.3 | DX6 | **`intergrax init <project>`** — cookiecutter: external repo, `pip install intergrax`, minimal harness layout | **Done** | High | `intergrax/scaffold/external_project/` template |
| DX-6.4 | DX6 | **CI smoke** — generated external template project pytest (fixture repo) | **Done** | Medium | `tests/integration/dx/test_external_project_template.py` |
| DX-6.5 | DX6 | **`pyproject` optional extra `[harness-author]`** — documented minimal dependency set for external apps | **Done** | Low | `pyproject.toml` + README |
| DX-7.1 | DX7 | **JSON Schema export** for `ApplicationEnvironmentProfile`, `ApplicationManifest`, `ApplicationGraphSpec` | **Done** | High | `scripts/export_harness_spec_schemas.py` → `build/harness_specs/` (CI) |
| DX-7.2 | DX7 | **`spec_version` on environment profile** + migration note in plan | **Done** | Medium | `environment_profile.py` |
| DX-7.3 | DX7 | **YAML round-trip tests** — graph + environment serialize/deserialize without loss | **Done** | High | `tests/unit/harness/test_spec_roundtrip.py` |
| DX-7.4 | DX7 | **Capability catalog JSON feed** — integrations/tools/skills slugs + labels for future UI builder | **Done** | Medium | `scripts/export_capability_catalog_feed.py` (CI) |
| DX-7.5 | DX7 | **Phase 2 UI boundary doc** — UI engine consumes DX-7 artifacts only; no parallel spec | **Done** | Low | Plan §Phase DX — UI boundary (below) |
| DX-8.1 | DX8 | **`intergrax doctor --ci`** — fail on tier violations, scaffold misalignment, TTFRun regression | **Done** | High | `.github/workflows/unit-tests.yml` |
| DX-8.2 | DX8 | **DX metrics in paydown** — record TTFRun seconds, author file count per release cycle | **Done** | Low | `scripts/record_dx_metrics.py` → `build/architecture_hardening/dx_metrics.json` |
| DX-8.3 | DX8 | **`check_scaffold_harness_alignment.py`** — CI script (complements DX-1.6 gate) | **Done** | High | `scripts/` + §6.1 maintenance list |

### DX — Explicitly deferred (not in the 47-task register)

| Topic | Reason |
|-------|--------|
| Visual graph editor / drag-and-drop UI | Phase 2 product; DX-7 only exports schemas |
| Rename `applications/` directory | Canon decision: Application = Tier-3 deployable instance |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |
| Full LangGraph runtime compatibility | Different execution model; mapping doc only (DX-0.2) |
| `phase_w_ops_evidence --verify-gate` hardening | W-OPS maintenance, not DX |

### DX — Paydown log

| Date | DX ID | Summary |
|------|-------|---------|
| 2026-06-03 | DX-1.1–DX-8.3 (core) | HarnessApplication, scaffold H-APP alignment, CLI run/doctor, presets, `check_scaffold_harness_alignment`; gate **518** |
| 2026-06-02 | Plan sync | Master table synced to codebase; **17** IDs remain **Pending** — [residual backlog](#dx--residual-backlog-infrastructure) |
| 2026-06-02 | DX residual closeout | `--minimal` stack, `expand`, doctor CI, spec export + round-trip, TTFRun acceptance, `agents.defaults.harness_production_mode`, docs quickstart; gate **533** |
| 2026-06-02 | DX-5.7 | §42.1.5 event catalog + `EVENT_OPS_FILTER_HINTS`; Phase DX **47/47 Done** |

**Suggested PR order (residual):** None — Phase DX infrastructure **Done**.

**Phase 2 UI boundary (DX-7.5):** A future visual builder must consume only versioned artifacts from `build/harness_specs/*.json` and `build/capability_catalog_feed.json` — not parallel Pydantic copies. Host behavior stays `HarnessApplication` / Tier-3 factories; UI edits serialize to the same `ApplicationEnvironmentProfile` + `ApplicationManifest` + `ApplicationGraphSpec` models validated by DX-7.1/DX-7.3.

### DX — Residual backlog (infrastructure)

**Not Band 3.** Platform DX rows **Done** (2026-06-02), including DX-5.7 (§42.1.5). No open DX IDs — see [§6.1z](#61z-harness-implementation-queue-consolidated).

---

## Phase AA — Agents & Applications Conformance (scaffold, docs, deploy)

**Status:** **Mostly Done** (2026-06-02) — **platform/conformance Done** (tier hygiene, ARCHITECTURE matrix, deploy triad, legal **scaffold** reset); **domain steps Deferred** (AA-LEG.2.2+); gate **534 passed**.  
**Prerequisites:** Phase **H-APP** **Done**, Phase **DX** **Mostly Done** (scaffold generators, `build_harness_host_runtime`, CLI, presets).  
**Goal:** Bring every **Tier-2** agent under `agents/` and every **Tier-3** host under `applications/` to a **documented, scaffold-aligned** state — fast authoring, full environment control (handlers, observability, policy), and **repeatable deploy** (Docker + deploy doc + `pyproject.toml` dependency contract per application). **Domain UAEP implementation is Band 3** — see [§6.3](#63-end-of-plan--deferred-product-work-only).  
**Priority ladder:** **Band 2g** (§4.0) — **platform rows closed**; only [AA residual](#aa--residual-backlog-infrastructure) + §6.1 maintenance.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2z](#62z-phase-aa-execution-order-band-2g--mostly-done).

**Delivery rule:** One `AA-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green → discuss scope in session before coding.

**Policy decision (2026-06-03):** **`agents/legal/` hard reset** — remove pre-architecture implementation; regenerate from `intergrax.scaffold new-agent legal` and re-implement only against UAEP + H-APP rules. Legacy tests become **behavioral spec** input, not code to preserve. **`legal_application`** follows the same reset cadence after the agent baseline exists (product shell only).

**Inventory (in scope):**

| Tier | Slug | Role |
|------|------|------|
| Agent | `echo` | Harness reference agent |
| Agent | `lab` | Lab mock agents (not product agents) |
| Agent | `legal` | **Hard reset** → scaffold baseline |
| Agent | `organization_worker` | Long-running / HITL demo |
| Agent | `problem_radar` | K.1 prototype (frozen until Band 3) |
| Agent | `research` | Multi-agent research prototype |
| Agent | `signoff_probe` | Appendix A sign-off exercise |
| Application | `lab_application` | Universal lab / debug superset |
| Application | `legal_application` | Legal product host (reset with agent) |
| Application | `poc_template_application` | Canonical minimal Tier-3 shell |
| Application | `research_application` | Research product host |

**Per-application deploy triad (mandatory for every Tier-3 host in this phase):**

Each `applications/<app>/` MUST document and maintain:

| Piece | Path / generator | Acceptance |
|-------|------------------|------------|
| **Docker** | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/build-docker.sh`, `docker/build-docker.bat`, `docker/.dockerignore` | Image builds locally; health path matches manifest `route_prefix` |
| **Deploy doc** | `BUILD_AND_DEPLOY.md` | Generated/updated via `intergrax.applications._shared.build_deploy_doc.render_build_deploy_doc` (scaffold) or manual parity with scaffold output |
| **`pyproject.toml` deps** | Root `[project]` + `[project.optional-dependencies]` | Section in `applications/<app>/ARCHITECTURE.md`: which **core** deps apply, which **extras** (`harness-author`, `langgraph-legacy`, `llm-*`, `dev-ci`, integration extras) the host requires; no undeclared imports |

Scaffold already emits Docker + `BUILD_AND_DEPLOY.md` for **new** apps (`new-application`, `new-stack`). Phase AA **backfills and verifies** this triad on all four existing applications.

**Audit verdict (2026-06-03):**

| Area | Verdict | Gap → AA IDs |
|------|---------|----------------|
| Tier-2 structure vs canon | **OK** | Tier-3 imports removed; `legal` scaffold baseline; CI `check_agents_no_tier3_imports.py` |
| Tier-3 H-APP factory | **OK** | `build_harness_host_runtime` on lab/poc/legal/research; manifest `environment=` |
| Scaffold completeness | **OK** | Typed `can_handle`, `--reference`, deploy triad regression test |
| Documentation | **OK** | ARCHITECTURE.md matrix, guides, TIER3_READINESS — AA-D0.* **Done** |
| LangGraph independence | **Done** | `langgraph` not in core deps; `check_langgraph_not_required.py` — AA-LG.1 **Done** |
| Legal module | Reset required | AA-LEG.* + AA-LEGAPP.* |

```text
Wave AA0  — Register, scaffold checklist, LangGraph (done), deploy triad standard (5)
Wave AA1  — Platform docs meta: README, guides, TIER3_READINESS (7)
Wave AA2  — Legal agent HARD RESET (12)
Wave AA3  — legal_application reset + deploy triad (8)
Wave AA4  — echo agent (5)
Wave AA5  — signoff_probe agent (3)
Wave AA6  — problem_radar agent (5)
Wave AA7  — organization_worker agent (5)
Wave AA8  — research agent (+ summary) (6)
Wave AA9  — lab mock agents doc (2)
Wave AA10 — lab_application host (7)
Wave AA11 — poc_template_application host (5)
Wave AA12 — research_application host (6)
Total: 83 (incl. AA-LG.1 counted in AA0)
```

### AA — Traceability (audit topic → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| A1 | Tier-2/3 separation (`agents` must not import `applications`) | AA-S0.2, AA-ECHO.2, AA-PR.2, AA-D0.3 |
| A2 | Scaffold agent `getattr` in `can_handle` | AA-S0.3 |
| A3 | Scaffold app missing manifest `environment=` | AA-S0.4, AA-POC.2, AA-RESAPP.2 |
| A4 | `lab_application` legacy Nexus factory | AA-LABAPP.2 |
| A5 | Legal pre-UAEP monolith | **AA-LEG.1–AA-LEG.12** (hard reset) |
| A6 | Per-agent architecture MD | AA-ECHO.1, AA-PR.1, AA-ORG.1, AA-RES.1, AA-SIG.1, AA-LEG.3 |
| A7 | Per-application architecture MD + deploy triad | AA-LABAPP.1, AA-POC.1, AA-RESAPP.1, AA-LEGAPP.1, AA-APP.0.1–AA-APP.0.3 |
| A8 | Root README completeness vs canon | AA-D0.1 |
| A9 | `AGENT_CREATION_GUIDE` / `TIER3_READINESS` stale | AA-D0.2–AA-D0.4 |
| A10 | LangGraph not required | AA-LG.1 (**Done**) |
| A11 | Docker + deploy script + pyproject per app | AA-APP.0.1–AA-APP.0.3, AA-*APP.*.4–*.6 |
| A12 | Legal application custom serving vs scaffold | AA-LEGAPP.3–AA-LEGAPP.5 |

### AA — Master deliverables register (all tasks)

#### Wave AA0 — Platform & scaffold foundation

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-0.1 | **Phase AA register** in this plan + §6.2z + doc model row | **Done** | Low | This section |
| AA-0.2 | **Scaffold ↔ H-APP checklist** table (new-agent / new-application / new-stack outputs) | **Done** | High | This section §AA scaffold matrix (below) |
| AA-S0.1 | Audit script: tier-2 must not import `applications` (extend `check_agent_registry_bypass` or sibling) | **Done** | High | `scripts/` + CI §6.1 |
| AA-S0.2 | **`new-agent`**: remove `getattr` from generated `can_handle` — typed `TaskContext` | **Done** | High | `intergrax/scaffold/new_agent.py` |
| AA-S0.3 | **`new-agent`**: optional `--reference` template (`HarnessReferenceAgent`) vs pure `Agent` | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.4 | **`new-agent`**: scaffold `contract.py` includes `skill_ids` placeholder + link SKILLS.md | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.5 | **`new-application`**: manifest always embeds `environment=ApplicationEnvironmentProfile…` | **Done** | High | `intergrax/scaffold/new_application.py` |
| AA-S0.6 | Document **`--full`** vs default scaffold (integration/tool wiring) | **Done** | Medium | `applications/USAGE.md` |
| AA-LG.1 | **LangGraph optional** — not in core deps; `langgraph-legacy` extra; `check_langgraph_not_required.py` | **Done** | High | `pyproject.toml`, CI |
| AA-APP.0.1 | **Deploy triad standard** — Docker + `BUILD_AND_DEPLOY.md` + pyproject extras section (canonical template) | **Done** | High | `applications/USAGE.md` §Deploy triad |
| AA-APP.0.2 | **Gate**: each existing `applications/*_application/` has `docker/`, `BUILD_AND_DEPLOY.md`, ARCHITECTURE deploy section | **Done** | High | `tests/unit/applications/test_application_deploy_triad.py` |
| AA-APP.0.3 | **Scaffold verify**: `new-application` output includes deploy triad (regression) | **Done** | High | `tests/unit/scaffold/test_scaffold_deploy_triad.py` |

**AA scaffold matrix (generator vs H-APP target):**

| Output | `new-agent` | `new-application` (default) | `new-application --full` |
|--------|-------------|----------------------------|---------------------------|
| UAEP `Agent` + `steps/pipeline.py` | Yes | — | — |
| `contract.py` / `capabilities.py` | Yes | — | — |
| `manifest.py` + `AgentBinding` | — | Yes | Yes |
| `host/environment_profile.py` | — | Yes | Yes |
| `host/factory.py` → `build_harness_host_runtime` | — | Yes | Yes |
| `host/integration_wiring.py` | — | No | Yes |
| `host/tool_wiring.py` | — | No | Yes |
| `host/policy/rules/` | — | Yes (`.gitkeep`) | Yes |
| `docker/*` + `BUILD_AND_DEPLOY.md` | — | Yes | Yes |
| MCP + smoke tests | — | Yes | Yes |

#### Wave AA1 — Documentation meta (canon alignment)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-D0.1 | **Root `README.md`** — `HarnessApplication`, `intergrax` CLI, `poc_template` as Tier-3 reference, LangGraph optional, agent vs app matrix | **Done** | High | `README.md` |
| AA-D0.2 | **`docs/README.md`** — Phase AA row, last updated | **Done** | Low | `docs/README.md` |
| AA-D0.3 | **`AGENT_CREATION_GUIDE.md`** — DX paths (`intergrax run`, `doctor`, minimal stack); no stale Nexus-only flow | **Done** | High | `docs/AGENT_CREATION_GUIDE.md` |
| AA-D0.4 | **`applications/TIER3_READINESS.md`** — `environment_profile`, `build_harness_host_runtime`; deploy triad; no mandatory `tool_wiring` for all apps | **Done** | High | `applications/TIER3_READINESS.md` |
| AA-D0.5 | **`applications/USAGE.md`** — deploy triad + pyproject extras per host | **Done** | High | `applications/USAGE.md` |
| AA-D0.6 | **`EXTENSION_AUTHOR_GUIDE.md`** — LangGraph analogy only (not required) — verify post AA-LG.1 | **Done** | Low | Already partially done |
| AA-D0.7 | **Conformance index** in plan — agent/app status columns (this register) | **Done** | Low | Appendix row or §AA paydown |

#### Wave AA2 — `agents/legal` HARD RESET (decision: scaffold baseline only)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEG.0.1 | **Record hard-reset decision** in plan + remove “incremental migration” as default for legal | **Done** | Critical | This section |
| AA-LEG.0.2 | **Archive tag** `legal-legacy-pre-aa` on git (pointer for forensic diff) | **Done** | High | Tag on parent of `bbce1bd` (pre hard-reset) |
| AA-LEG.0.3 | **Extract behavioral spec** from legacy tests → `agents/legal/SPEC_FROM_LEGACY.md` (requirements only) | **Done** | High | Before delete |
| AA-LEG.1.1 | **Delete** legacy `agents/legal/` tree (pipeline, governance, custom loop, tracing dupes) | **Done** | **Critical** | PR after AA-LEG.0.3 |
| AA-LEG.1.2 | **`python -m intergrax.scaffold new-agent legal --capability legal.review`** (force clean tree) | **Done** | **Critical** | `agents/legal/` matches scaffold layout |
| AA-LEG.1.3 | **`agents/legal/ARCHITECTURE.md`** — target UAEP graph, skills, tools, config, observability hooks (design-only until steps exist) | **Done** | High | English canonical doc |
| AA-LEG.2.1 | **Register** `legal` skill bundle on contract (`skill_ids`) per SKILLS.md | **Done** | High | `contract.py` |
| AA-LEG.2.2 | **UAEP steps** — port minimal slice from spec (one step per PR) | **Deferred** | High | `steps/` |
| AA-LEG.2.3 | **Remove** custom `legal_execution_loop`, `legal_tool_runtime_bridge` patterns — use Nexus `RuntimeToolGateway` only | **Deferred** | High | No parallel runtime |
| AA-LEG.2.4 | **Agent tests** — smoke + one spec-backed test per ported step | **Deferred** | High | `agents/legal/tests/` |
| AA-LEG.2.5 | **Retire** `ROADMAP.md` / `IMPLEMENTATION_PLAN.md` / `HOST_README.md` under agent — merge into `ARCHITECTURE.md` | **Done** | Medium | Single agent doc |
| AA-LEG.3.1 | **Gate**: `legal` agent imports no `applications.*`; no `getattr` on contract | **Done** | High | CI scripts |

**Explicitly NOT in legal reset:** Live LLM E2E product proof (Band 3 — K.6 / B.15 / S-Ops.4).

#### Wave AA3 — `applications/legal_application` (reset with agent)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEGAPP.1 | **`ARCHITECTURE.md`** — manifest, environment profile, factory, auth, observability DB paths, MCP | **Done** | High | `applications/legal_application/` |
| AA-LEGAPP.2 | **Manifest** — `environment=ApplicationEnvironmentProfile.product_defaults(…)` inline | **Done** | High | `manifest.py` |
| AA-LEGAPP.3 | **Factory/serving** — align to `poc_template` + product settings; remove redundant `runtime_bridge` if superseded by `UnifiedTaskRunner` | **Done** | High | `host/factory.py`, `serving/` |
| AA-LEGAPP.4 | **Deploy triad** — verify/update `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | See AA-APP.0.1 |
| AA-LEGAPP.5 | **`pyproject.toml` deps section** in ARCHITECTURE — `harness-author`, LLM extras, optional `langgraph-legacy` N/A | **Done** | High | ARCHITECTURE §Dependencies |
| AA-LEGAPP.6 | **Host smoke** — `legal_tests/` green on scaffolded agent only | **Deferred** | High | After AA-LEG.2.2 |
| AA-LEGAPP.7 | **`.env.example`** parity with scaffold product profile | **Done** | Medium | `.env.example` |
| AA-LEGAPP.8 | **Remove** duplicate legal test trees if consolidated | **Deferred** | Low | `legal_tests/` vs agent tests |

#### Wave AA4 — Agent `echo`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ECHO.1 | **`agents/echo/ARCHITECTURE.md`** — reference role, capabilities, skills, lab registration | **Done** | High | English |
| AA-ECHO.2 | **Remove Tier-3 imports** — inject `LabHarnessContext` from `lab_application` factory only | **Done** | **Critical** | `agents/echo/echo_agent.py` |
| AA-ECHO.3 | Align with **`HarnessReferenceAgent`** pattern documented in canon | **Done** | Medium | Code + doc |
| AA-ECHO.4 | **Tests** — import agent module without `applications` on PYTHONPATH | **Done** | High | `tests/unit/agents/` |
| AA-ECHO.5 | **README** — pointer to ARCHITECTURE only | **Done** | Low | `agents/echo/README.md` |

#### Wave AA5 — Agent `signoff_probe`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-SIG.1 | **`ARCHITECTURE.md`** — Appendix A sign-off flow, capability `signoff.probe` | **Done** | Medium | `agents/signoff_probe/` |
| AA-SIG.2 | Verify **scaffold parity** when AA-S0.2 lands (regenerate diff empty except domain) | **Done** | Low | `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-SIG.3 | **README** → ARCHITECTURE link | **Done** | Low | |

#### Wave AA6 — Agent `problem_radar`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-PR.1 | **`ARCHITECTURE.md`** — K.1 placeholder, I/O schema, policy | **Done** | Medium | Frozen until Band 3 |
| AA-PR.2 | **Remove Tier-3 imports** (same pattern as echo) | **Done** | High | `problem_radar_agent.py` |
| AA-PR.3 | **Notebook + tests** documented in ARCHITECTURE | **Done** | Low | |
| AA-PR.4 | **Status** in plan §6.3 — no feature work until K.1 reprioritized | **Done** | Low | |
| AA-PR.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA7 — Agent `organization_worker`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ORG.1 | **`ARCHITECTURE.md`** — HITL, long-running, `org.vendor_report` | **Done** | Medium | |
| AA-ORG.2 | **Remove `testing_support` import** — fake LLM via test fixture injection | **Done** | High | `organization_worker_agent.py` |
| AA-ORG.3 | **Scaffold-align** — add `contract.py`, `capabilities.py`, `steps/` if missing | **Deferred** | Medium | |
| AA-ORG.4 | **Lab manifest flag** + integration test | **Deferred** | Medium | `lab_application/manifest.py` |
| AA-ORG.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA8 — Agent `research` (+ `summary_agent`)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RES.1 | **`agents/research/ARCHITECTURE.md`** — graph intent `research.pipeline`, two agents | **Done** | High | |
| AA-RES.2 | **Remove Tier-3 imports** from agents if any | **Done** | High | |
| AA-RES.3 | **`HarnessReferenceAgent`** alignment for Research/Summary | **Done** | Medium | |
| AA-RES.4 | **Skill ids** on contracts | **Deferred** | Medium | |
| AA-RES.5 | **Tests** — UAEP + graph delegation | **Deferred** | High | |
| AA-RES.6 | **README** merge into ARCHITECTURE | **Done** | Low | |

#### Wave AA9 — Agent `lab` (mocks)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAG.1 | **`agents/lab/README.md`** — mock agents purpose, not product Tier-2 | **Done** | Low | `agents/lab/README.md` |
| AA-LABAG.2 | **(Optional)** move mocks to `testing_support/` if they are test-only | **Won't fix** | Low | Until leadership requests — mocks stay under `agents/lab/` |

#### Wave AA10 — Application `lab_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAPP.1 | **`ARCHITECTURE.md`** — debug API, interaction, scheduler, manifest flags | **Done** | High | |
| AA-LABAPP.2 | **Migrate factory** to `build_harness_host_runtime` (retain rich wiring via env profile) | **Done** | **Critical** | `host/factory.py` |
| AA-LABAPP.3 | **`environment` in manifest** or documented single profile builder | **Done** | High | `manifest.py` / `_shared` |
| AA-LABAPP.4 | **Deploy triad** — verify `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | |
| AA-LABAPP.5 | **`pyproject.toml` deps** section in ARCHITECTURE | **Done** | High | |
| AA-LABAPP.6 | **Smoke tests** after factory migration | **Done** | High | `lab_application_tests/host/test_lab_host_smoke.py` + `tests/acceptance/agent_os/test_lab_application.py` |
| AA-LABAPP.7 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA11 — Application `poc_template_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-POC.1 | **`ARCHITECTURE.md`** — canonical Tier-3 lab shell (reference for new apps) | **Done** | High | |
| AA-POC.2 | **Manifest `environment=`** explicit (not only factory fallback) | **Done** | High | `manifest.py` |
| AA-POC.3 | **Deploy triad** verification | **Done** | Medium | |
| AA-POC.4 | **`pyproject.toml` deps** section | **Done** | Medium | |
| AA-POC.5 | **Link from root README** as “start here for new application” | **Done** | Medium | AA-D0.1 |

#### Wave AA12 — Application `research_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RESAPP.1 | **`ARCHITECTURE.md`** — multi-agent HTTP, env vars, graph | **Done** | High | |
| AA-RESAPP.2 | **Manifest `environment=`** + `host/environment_profile.py` parity with scaffold | **Done** | High | |
| AA-RESAPP.3 | **Remove dead flags** `RESEARCH_USE_LEGACY_*` if obsolete | **Done** | Medium | `host/settings.py` |
| AA-RESAPP.4 | **Deploy triad** verification | **Done** | High | |
| AA-RESAPP.5 | **`pyproject.toml` deps** section | **Done** | High | |
| AA-RESAPP.6 | **Smoke tests** + `test_research_manifest_wiring` green | **Deferred** | High | |

### AA — Conformance matrix (living status)

| Module | Scaffold-aligned | ARCHITECTURE.md | Deploy triad | pyproject doc | Tier hygiene |
|--------|------------------|-----------------|--------------|---------------|--------------|
| `agents/echo` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/lab` | N/A (mocks) | README only | N/A | N/A | OK |
| `agents/legal` | Yes (scaffold) | **Done** | N/A | N/A | **OK** |
| `agents/organization_worker` | Partial | **Done** | N/A | N/A | **OK** |
| `agents/problem_radar` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/research` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/signoff_probe` | Yes | **Done** | N/A | N/A | OK |
| `applications/lab_application` | Yes | **Done** | **OK** | **Done** | H-APP factory |
| `applications/legal_application` | Yes | **Done** | **OK** | **Done** | H-APP |
| `applications/poc_template_application` | Yes | **Done** | **OK** | **Done** | OK |
| `applications/research_application` | Yes | **Done** | **OK** | **Done** | H-APP |

### AA — Residual backlog (infrastructure)

**Platform AA rows closed (2026-06-02).** Open infrastructure work: [§6.1z](#61z-harness-implementation-queue-consolidated) **V-REM** (2026-06-05) + ongoing **§6.1** maintenance.

| ID | Deliverable | Priority | Notes |
|----|-------------|----------|-------|
| AA-LABAG.1 | `agents/lab/README.md` — mock agents, not product Tier-2 | Low | **Done** — `agents/lab/README.md` |
| AA-LABAG.2 | (Optional) move lab mocks to `testing_support/` | Low | **Won't fix** until leadership requests |
| AA-SIG.2 | Scaffold parity diff test for `signoff_probe` | Low | **Done** — `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-LABAPP.6 | Lab host smoke after H-APP factory | High | **Done** — unit + acceptance coverage |
| AA-LEG.0.2 | Git tag `legal-legacy-pre-aa` | High | **Done** — annotated tag on pre-reset commit |

### AA — Explicitly deferred (business / domain — Band 3)

| Topic | Task IDs | Reason |
|-------|----------|--------|
| Legal UAEP domain steps | AA-LEG.2.2–2.4, AA-LEGAPP.6, AA-LEGAPP.8 | Business logic on scaffold — [§6.3a](#63a-business-backlog-register-consolidated) |
| Research domain | AA-RES.4, AA-RES.5, AA-RESAPP.6 | Skills + graph tests — product prototype |
| Organization worker full scaffold | AA-ORG.3, AA-ORG.4 | Demo agent + lab roster |
| Lab host extra smoke | AA-LABAPP.6 | **Done** (2026-06-02 sync) — not blocking |
| K.1 / K.2 | Phase K | Band 3 — problem_radar / vendor discovery |
| Legal live LLM E2E | K.6 / B.15 / S-Ops.4 | Band 3 — CI budget |
| New product Tier-3 beyond four hosts | §6.3 | Product decision |

### AA — Paydown log

| Date | AA ID | Summary |
|------|-------|---------|
| 2026-06-03 | AA-0.1, AA-LEG.0.1 | Phase AA registered; **legal hard reset** policy recorded |
| 2026-06-03 | AA-LG.1 | LangGraph removed from core deps; CI `check_langgraph_not_required.py` |
| 2026-06-03 | AA-S0.1–S0.2, AA-S0.5, AA-APP.0.1–0.3, AA-ECHO.2, AA-PR.*, AA-LABAPP.2, AA-POC.2, AA-RESAPP.2, AA-LEG.1–1.3 | Tier hygiene, lab harness runtime, legal hard reset, deploy triad gate; gate **521** |
| 2026-06-02 | AA-S0.3, AA-D0.*, AA-* ARCHITECTURE, AA-LABAPP.3, AA-RESAPP.3 | `--reference` scaffold, docs matrix, lab manifest environment, tier import tests; gate **526** |
| 2026-06-02 | Plan sync | §4.0a scope split, DX/AA residual backlogs, §6.3a business register, master tables synced |
| 2026-06-02 | AA sync | AA-LABAPP.6 **Done**; AA-LABAG.2 **Won't fix**; §6.1z implementation queue |
| 2026-06-02 | AA-LEG.0.2, OPS-L3.1 | Tag `legal-legacy-pre-aa`; operational L3 evidence verified |

**Suggested session order (platform — complete):**  
See [§6.1z](#61z-harness-implementation-queue-consolidated). **Do not schedule** AA-LEG.2.* / AA-RES.5 / AA-ORG.3–4 in harness cadence — use [§6.3a](#63a-business-backlog-register-consolidated) after product decision.

---

## Phase MEM — Memory Platform Completion

**Status:** **Done** (2026-06-02) — **48/48** deliverables; gate **571 passed**.  
**Prerequisites:** Phases **I** (TaskMemory), **R-Context**, **H-APP** (profile models), **DX-5.7** (ops:memory hints) **Done**; **H-APP.4.3** closed via **MEM-1.***.  
**Goal:** Close every gap from the **memory platform audit** — short-term session, user/org LTM, task KV, context compression, H-APP→runtime wiring, persistence, recovery, observability, developer hooks, and market-parity documentation — **without** Band 3 product agents (K.1/K.2) or Mem0-like SaaS product layer (MEM-8 deferred P3).  
**Priority ladder:** **Band 2h** (§4.0) — **default implementation queue** after §6.1 maintenance.  
**Execution order:** [§6.2aa](#62aa-phase-mem-execution-order-band-2h--active).  
**Canon refs:** §27 Memory model · §28.1 Context assembly · §42.35 MemoryView · Appendix G in [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md).

**Delivery rule:** One `MEM-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

| Area | Maturity (1–5) | Audit comment | Close via |
|------|----------------|---------------|-----------|
| Task memory (KV, delegation, handoff) | **4/5** | Best-in-repo; SQLite, policy, events | MEM-DOC.*, MEM-TASK.* (docs + lab policy) |
| Context / LLM window | **3,5/5** | Budget + assembly + history summarization; weak tests; in-memory session default | MEM-1.*, MEM-5.*, MEM-CTX.* |
| Short-term session (STM) | **3/5** | Model OK; production path often in-memory | MEM-1.3, MEM-4.1, MEM-DOC.1 |
| User LTM | **2,5/5** | Logic exists; no durable store in repo | MEM-2.*, MEM-4.2 |
| Organization memory | **2,5/5** | SQLite org profile; not full org memory product | MEM-1.4, MEM-DOC.3 |
| Consolidation / fact extraction | **2/5** | LLM consolidation; notebooks; few gate tests | MEM-4.2, MEM-8.* (P3) |
| Graph memory (agent sense) | **1/5** | Graph RAG ≠ agent memory | MEM-GRAPH.*, MEM-9.* |
| Developer hooks | **2/5** | MemoryView + events; no memory lifecycle hooks / EP | MEM-3.*, MEM-7.* |
| H-APP → runtime config | **4/5** | Bridge **Done** via MEM-1.* | MEM-DOC.* maintenance |
| Declarative env config | **4/5** | MemoryProfile wired via H-APP bridge | MEM-1.* |
| Memory observability | **4/5** | MEMORY_* / CONTEXT_* events + memory SLO metrics baseline | MEM-OBS.* |

**Overall platform memory score: ~3,5/5** — Tier-1 architecture closed for harness; product Mem0/Zep layer remains **§6.3** optional.

**Out of scope (explicit):** K.1/K.2 business memory; hosted Mem0/Zep replacement SaaS; Neo4j entity graph as default user memory (MEM-9 = design RFC only); Redis/Postgres session backends as shipped defaults (MEM-PERS.3 spike P3).

```text
Wave MEM0 — Register, audit baseline, conceptual + parity docs (9 tasks)
Wave MEM1 — P0 bridge + SQLite user LTM (9 tasks) — closes H-APP.4.3 gap
Wave MEM2 — P1 plugins, gates, context docs, graph clarification (18 tasks)
Wave MEM3 — P2 retention, hooks, SLO metrics, optional backends (9 tasks)
Wave MEM4 — P3 product memory layer + entity graph RFC (4 tasks)
Total: 48
```

### MEM — Conceptual model (canon §27 vs runtime)

Canon §27 defines **5 memory types**:

1. Task Memory  
2. Agent Local Memory  
3. User / Organization Memory  
4. Long-Term Knowledge Memory  
5. Execution Trace Memory  

Runtime maps these to **four operational stores** (+ trace + RAG — not memory layers):

```text
Short-term:     SessionManager + SessionStorage; optional ConversationalMemory (FIFO)
Task-scoped:    TaskMemory SQLite KV → PolicyScopedMemoryView → SharedTaskContext handoff
User / Org LTM: UserProfileManager + entries; OrganizationProfile SQLite
Knowledge:      RAG vectorstore; Graph RAG (document graph — NOT agent entity memory)
Trace:          RunTraceWriter / RuntimeEvents (immutable audit, not agent-mutable memory)
```

**Gap (document, do not implement as separate modules in MEM0):** no first-class **episodic / semantic / procedural** taxonomy in code — only `MemoryKind` entry tags (`USER_FACT`, `PREFERENCE`, `SESSION_SUMMARY`, `ORG_FACT`, `POLICY`). IDEAL harness doc describes episodic/semantic as **vision only**.

### MEM — Persistence backend matrix (as-built)

| Layer | In-memory | SQLite | Postgres | Redis | Mongo |
|-------|-----------|--------|----------|-------|-------|
| Task KV | test | prod path (`INTERGRAX_TASK_MEMORY_DB`) | — | — | — |
| Session | lab SQLite via bridge | bundle path | — | — | — |
| User profile LTM | test | bundle (`SQLiteUserProfileStore`) | — | — | optional `DocumentStoreUserProfileStore` (MEM-PERS.2) |
| Org profile | test | bundle | — | — | — |
| Checkpoints (≠ memory) | — | yes | — | — | — |
| Trace / events | test | yes | — | — | — |

**SQLite integration bundle** (`create_sqlite_integration`) = lab hub (trace, events, task_memory, session, org profile, checkpoints) — coherent for dev; **not** multi-tenant production scale.

**Recovery semantics (target documentation — MEM-DOC.4):**

| Layer | Recovery key | Works when | Broken today |
|-------|--------------|------------|--------------|
| Task memory | `tenant_id` + `task_id` + namespace | SQLite enabled | — |
| Session | `session_id` | SQLite SessionStorage when relational_store=sqlite | — |
| User LTM | user id | SQLite bundle or Mongo document_store | — |
| Long-running | checkpoint store | SQLite | separate from conversational memory |
| Org profile | org id | SQLite bundle | — |

### MEM — Market parity traceability (MEM-PAR.1)

| Capability | LangGraph | Mem0 / Zep | Intergrax today | Target ID |
|------------|-----------|------------|-----------------|-----------|
| Thread / session persistence | Checkpointer | Session + graph | Session + checkpoint **separate** | MEM-DOC.1, MEM-1.3 |
| Scoped KV per run | Store API | — | TaskMemory + MemoryView ✅ | MEM-DOC.2 |
| Auto fact extraction | — | core | Consolidation service, **manual trigger** | MEM-4.2, MEM-8.* |
| Entity graph memory | — | Zep ✅ | ❌ (RAG graph only) | MEM-GRAPH.1, MEM-9.* |
| Vector semantic memory | Optional | ✅ | User LTM via RAG index | MEM-2.* |
| Subagent namespace isolation | Subgraph state | — | delegation namespace ✅ | documented |
| Memory hooks / plugins | Checkpointer swap | API | Event bus only | MEM-3.*, MEM-7.* |
| Declarative env config | Partial | SaaS | MemoryProfile **Done** | MEM-1.* |
| Observability | LangSmith | Dashboard | Trace events ✅; no memory SLO | MEM-OBS.* |

### MEM — User audit checklist → deliverables (MEM-CHk.1)

| Audit question | Answer (as-built) | Deliverable IDs |
|----------------|-------------------|-----------------|
| How is memory **managed**? | Nexus Tier-1; agents via UAEP/MemoryView; profiles via runtime steps | MEM-DOC.2, MEM-1.* |
| **Limited context** handling? | Budget trim + history summarization + LTM limits + summary tiers — no single end-to-end policy | MEM-1.2, MEM-CTX.1, MEM-5.* |
| **Strategy** (summarize, trim)? | `HistoryLayer` + `ContextBudgetPolicy`; trim often char-cut | MEM-5.* |
| **Developer handlers**? | MemoryView, policy, events — no formal memory hooks / plugin catalog | MEM-3.*, MEM-7.* |
| **Where persisted**? | Task/org/session: SQLite (lab); user LTM: in-memory; Redis: not memory layer | MEM-2.*, MEM-PERS.1 |
| **Configuration**? | Env + `MemoryProfile` partial; H-APP bridge incomplete | MEM-1.* |
| **Recovery**? | Task/session by ID if SQLite; user LTM weak | MEM-1.3, MEM-2.*, MEM-DOC.4 |
| **Tests**? | Task/context OK; consolidation/history gaps | MEM-4.*, MEM-5.1, MEM-TEST.* |
| **Observation**? | MEMORY_* / CONTEXT_* events; no product memory metrics | MEM-OBS.* |
| **Graph memory**? | **No** as agent memory — Graph RAG only | MEM-GRAPH.1 |

### MEM — Architecture inventory (existing code — do not rewrite)

| Module | Tier | Role |
|--------|------|------|
| `intergrax/memory/` | 0 | ConversationalMemory, UserProfileManager |
| `intergrax/runtime/task_memory/` | 1 | Coordinator, MemoryView, SQLite store, delegation |
| `intergrax/runtime/nexus/session/` | 1 | SessionManager, InMemory/SQLite storage |
| `intergrax/runtime/nexus/context/` | 1 | ContextManager, context_budget, engine_history_layer |
| `intergrax/runtime/user_profile/session_memory_consolidation_service.py` | 1 | LLM session → LTM extraction |
| `intergrax/applications/_shared/runtime_config_bridge.py` | 3 | **Gap:** line ~112 always `InMemorySessionStorage()` |
| `intergrax/applications/_shared/task_memory_wiring.py` | 3 | Enables task DB from profile flags; does not enable LTM/org Nexus steps |
| `intergrax/applications/contracts/environment_profile.py` | 3 | `MemoryProfile`, `ContextProfile` |
| `intergrax/rag/graph/` | 0 | Graph RAG retrieval — **not** agent memory |
| `integrations/examples/custom_memory_kv/` | 0 | KV plugin example — not wired to TaskMemory |

**Existing gate tests:** `tests/unit/runtime/task_memory/`, `tests/acceptance/.../test_acceptance_08_memory_handoff`, context budget, profile steps, sqlite session integration.

**Known gaps in tests:** `engine_history_layer` summarization; E2E LTM consolidation; `MemoryProfile` → runtime wiring; external memory backends; graph-as-memory.

### MEM — Traceability (audit section → task IDs)

| Audit § | Topic | Task IDs |
|---------|--------|----------|
| §1 | Conceptual model (5 types vs 4 stores) | MEM-0.3, MEM-0.4 |
| §2 | Short-term session; InMemory default in bridge; Redis not memory layer | MEM-1.3, MEM-4.1, MEM-DOC.1, MEM-ST.1, MEM-PERS.3 |
| §3 | User LTM; InMemoryUserProfileStore only | MEM-2.*, MEM-4.2 |
| §4 | Org memory; enable_org_memory not mapped | MEM-1.4, MEM-DOC.3 |
| §5 | Task memory strengths; lab default off | MEM-TASK.*, MEM-DOC.2 |
| §6 | Context strategy; budget_policy not mapped | MEM-1.2, MEM-5.*, MEM-CTX.* |
| §7 | Developer hooks; no memory EP | MEM-3.*, MEM-7.*, MEM-DOC.5, MEM-DOC.6 |
| §8 | Persistence matrix; H-APP.4.3 divergence | MEM-PERS.1, MEM-1.*, MEM-REC → MEM-DOC.4 |
| §9 | Observability + test gaps | MEM-OBS.*, MEM-4.*, MEM-5.1 |
| §10 | Graph memory ≠ Graph RAG | MEM-GRAPH.* |
| §11 | Market comparison | MEM-PAR.1 (table above) |
| §12 | Recommended MEM-1..9 backlog | MEM-1.* … MEM-9.* |
| §13 | User checklist | MEM-CHk.1 (table above) |

### MEM — Master deliverables register (all 48 tasks)

#### Wave MEM0 — Register & audit baseline

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-0.1 | **Phase MEM register** in this plan + §6.2aa + §6.1aa + doc model row | **Done** | Low | This section |
| MEM-0.2 | **Audit maturity baseline table** preserved (§Audit verdict above) | **Done** | Low | Do not delete on paydown |
| MEM-0.3 | **Canon §27 → 4 stores** mapping + flow diagram in `AGENT_CREATION_GUIDE` Appendix G | **Done** | Medium | Guide + cross-link §27 |
| MEM-0.4 | Document **`MemoryKind` tags** vs episodic/semantic/procedural (IDEAL vision vs runtime) | **Done** | Low | Guide or canon footnote |
| MEM-PAR.1 | **Market parity traceability table** (LangGraph / Mem0 / Zep) | **Done** | Low | This section §MEM — Market parity |
| MEM-CHk.1 | **User audit checklist** → deliverable mapping (10 questions) | **Done** | Low | This section §MEM — User audit checklist |
| MEM-PERS.1 | **Persistence backend matrix** synced to Appendix G | **Done** | Low | Guide Appendix G + this section |
| MEM-ST.1 | **Document:** Redis `KeyValueCache` = integration cache only — **not** session/LTM memory layer | **Done** | Low | `INTEGRATIONS.md` or guide |
| MEM-OBS.2 | Baseline: `MEMORY_READ`/`MEMORY_WRITE`, `CONTEXT_*`, `ops:memory` filter | **Done** | — | DX-5.7 · `phase_coverage.py` |

#### Wave MEM1 — P0: H-APP bridge + durable user LTM (closes H-APP.4.3)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-1.1 | **`materialize_runtime_config` reads `MemoryProfile`** — map `enable_user_longterm_memory`, `enable_task_memory`, retention, scope flags to `RuntimeConfig` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.2 | Map **`ContextProfile.budget_policy`** → `RuntimeConfig` context budget fields | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.3 | **`SessionManager` from integration bundle** — resolve `SQLiteSessionStorage` when sqlite profile active; remove hardcoded `InMemorySessionStorage()` in `build_runtime_context_from_environment` | **Done** | **P0 Critical** | `runtime_config_bridge.py` |
| MEM-1.4 | Map **`MemoryProfile.enable_org_memory`** → `RuntimeConfig.enable_org_profile_memory` | **Done** | **P0** | `runtime_config_bridge.py` |
| MEM-1.5 | **Gate test:** `ApplicationEnvironmentProfile` memory + context → `RuntimeConfig` round-trip | **Done** | **P0** | `tests/unit/applications/test_memory_profile_runtime_bridge.py` |
| MEM-1.6 | **Reconcile H-APP.4.3** — mark **Done** only when MEM-1.1–MEM-1.4 **Done** | **Done** | **P0** | H-APP register row |
| MEM-2.1 | **`SQLiteUserProfileStore`** — mirror `SQLiteOrganizationProfileStore` pattern | **Done** | **P0 Critical** | `intergrax/memory/` or `runtime/user_profile/` |
| MEM-2.2 | **Wire `SQLiteUserProfileStore`** in sqlite integration bundle + lab/legal/research profiles | **Done** | **P0** | integration bundle wiring |
| MEM-2.3 | **Unit tests:** `UserProfileManager` CRUD + search with SQLite backend (fake RetrievalService) | **Done** | **P0** | `tests/unit/memory/` |

#### Wave MEM2 — P1: gates, plugins prep, context docs, graph clarification

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-3.1 | **`UserProfileStore` / `SessionStorage` plugin Protocol** (typed, no Tier-2 imports) | **Done** | P1 | `intergrax/memory/contracts/` |
| MEM-3.2 | **Entry point group `intergrax.memory_stores`** + `bootstrap_memory_stores()` | **Done** | P1 | Mirror P-Ext pattern |
| MEM-3.3 | **Reference external memory store** + gate (fixture package) | **Done** | P1 | `tests/fixtures/` + unit test |
| MEM-4.1 | **Gate:** session SQLite persist + resume round-trip via H-APP host | **Done** | P1 | `tests/integration/` |
| MEM-4.2 | **Gate:** LTM consolidation E2E with deterministic fake LLM | **Done** | P1 | `tests/acceptance/` (not notebook-only) |
| MEM-4.3 | **Gate:** full memory stack on lab profile (task + session + LTM + org) | **Done** | P1 | acceptance or integration |
| MEM-5.1 | **Unit tests:** `engine_history_layer` — `SUMMARIZE_OLDEST` + truncate fallback | **Done** | P1 | `tests/unit/runtime/nexus/context/` |
| MEM-5.2 | **Document context compression strategy matrix** (FULL / SUMMARY / SUMMARIZE_OLDEST / hard trim) | **Done** | P1 | Guide + canon §28.1 |
| MEM-CTX.1 | **`ContextDecisionProfile`** (or extend `ContextProfile`) — unified memory vs context vs RAG assembly policy for Tier-3 authors | **Done** | P1 | `environment_profile.py` |
| MEM-DOC.1 | **Author cookbook:** session vs checkpoint vs task KV mental model (LangGraph thread analogy) | **Done** | P1 | `AGENT_CREATION_GUIDE.md` |
| MEM-DOC.2 | Document **`wire_task_memory_from_profile` vs Nexus LTM/org steps** gap | **Done** | P1 | Guide + this plan |
| MEM-DOC.3 | **Org memory scope** — profile + instructions vs shared episodic / team knowledge | **Done** | P1 | Guide |
| MEM-DOC.4 | **Recovery semantics** per memory layer (table in guide) | **Done** | P1 | Guide or `HARNESS_ENVIRONMENT.md` |
| MEM-DOC.6 | Clarify **`custom_memory_kv` example** — integration KV vs Nexus TaskMemory | **Done** | P1 | `integrations/examples/` README |
| MEM-GRAPH.1 | **Document:** Graph RAG (`intergrax/rag/graph/`) ≠ agent entity graph memory | **Done** | P1 | Canon + RAG docs |
| MEM-TASK.1 | **Lab profile:** explicit task memory enable policy (replace silent default-off + log warning only) | **Done** | P1 | `lab_application` environment profile |
| MEM-TASK.2 | **Author cookbook:** MemoryView namespaces + delegation paths | **Done** | P1 | Guide Appendix G |

#### Wave MEM3 — P2: policy enforcement, hooks, observability, optional backends

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-6.1 | **Enforce `MemoryProfile.retention_days`** on session + task stores (TTL / purge job or read filter) | **Done** | P2 | session + task_memory |
| MEM-6.2 | **Enforce `scope_boundary`** on `PolicyScopedMemoryView` writes | **Done** | P2 | `memory_view.py` + policy |
| MEM-7.1 | **HookPoint `BEFORE_MEMORY_WRITE`** (+ optional `AFTER_MEMORY_WRITE`) | **Done** | P2 | `runtime/nexus/hooks/` |
| MEM-7.2 | **Gate:** hook can deny or mutate memory write | **Done** | P2 | unit test |
| MEM-OBS.1 | **Memory SLO metrics** — LTM hit rate, retention violations, memory write volume | **Done** | P2 | observability / Prometheus hooks |
| MEM-DOC.5 | **Cookbook:** swap `UserProfileStore` to external backend via EP | **Done** | P2 | `EXTENSION_AUTHOR_GUIDE.md` |
| MEM-CTX.2 | **Token-aware context trim** evaluation (vs char-cut only) — spike + recommendation | **Done** | P2 | `context_budget.py` RFC or impl |
| MEM-PERS.2 | **Optional:** Mongo `document_store` path for user memory artifacts | **Done** | P2 | Tier-0 integration wiring |
| MEM-ST.4 | **Optional:** `ConversationalMemoryStore` backend beyond in-memory | **Done** | P2 | `intergrax/memory/` |

#### Wave MEM4 — P3: product memory layer (Band 3 option) + entity graph RFC

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| MEM-8.1 | **Design RFC:** unified memory product layer (Mem0-like auto-ingest, dedup, temporal validity) | **Done** | P3 | §6.3 decision gate |
| MEM-8.2 | **Background consolidation job** — auto fact extraction (optional product) | **Done** | P3 | Deferred with MEM-8.1 |
| MEM-9.1 | **Design RFC:** entity graph memory for user entities (separate from Graph RAG) | **Done** | P3 | Canon §53 follow-up |
| MEM-PERS.3 | **Spike:** Postgres memory backend for session/LTM (multi-tenant) | **Done** | P3 | RFC only; no default ship |

### MEM — Paydown log

| Date | ID | Notes |
|------|-----|-------|
| 2026-06-02 | MEM-1.*–MEM-9.* | Phase MEM **48/48 Done**; H-APP.4.3 **Done**; gate **571** |
| 2026-06-02 | §6.1 reference hosts | `with_harness_memory()` on legal/research; gate `test_reference_hosts_memory_bridge`; W-OPS memory_platform_gate |
| 2026-06-02 | MEM-0.1–MEM-0.2, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1 | Memory audit → Phase MEM register in plan |
| 2026-06-02 | MEM-OBS.2 | Baseline already **Done** (DX-5.7) |

**Suggested PR order (P0 first):** MEM-1.1 → MEM-1.2 → MEM-1.3 → MEM-1.4 → MEM-1.5 → MEM-2.1 → MEM-2.2 → MEM-2.3 → MEM-1.6 → MEM-4.1 → MEM-5.1 → MEM-4.2 → MEM-3.1 → MEM-3.2 → MEM-0.3 → remaining MEM2 → MEM3 → MEM4.

**Success gate for Phase MEM closeout:** All **P0 + P1** rows **Done** or **Won't fix** with rationale; gate green; H-APP.4.3 **Done**; user LTM survives process restart on sqlite lab profile; `MemoryProfile` fully drives `RuntimeConfig` on all four reference hosts.

**Explicitly out of NOW:** K.1/K.2 memory features, Mem0 SaaS parity (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC), Redis session as default.

---

### Phase P-Ext — Plugin Catalogs (Integrations, Tools, Skills)

**Status:** **Done** (2026-06-02) — MVP + production closure (Appendix I).  
**Prerequisites:** Phases **M** (Integration Library), **O** (Tool Library), **R** (Skill Library MVP) **Done**; open integration slug model (no closed `IntegrationSlug` enum in registry) **Done**.  
**Goal:** Make all three Tier-0 catalogs **plugin-native** and aligned with market patterns (hexagonal adapters, MCP-style tools, capability packs) — including **pip-installable** extensions without editing Intergrax core.  
**Tracker:** **Appendix I** (task-level status). **Author guide:** [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md).

**Delivered (2026-06-02):** `load_plugins` + `bootstrap_catalogs()` · three plugin protocols · lazy presets/bundle ids · EP fixture package · `warn_override` conflict policy · scaffold CLI · integrations **manifest+factory** (~99 full) + `IntegrationPlugin` for externals · tools **13/13** `ToolPlugin` · skills **3/3** `SkillPlugin` · `resolve_typed` (6 categories) · health API · `CatalogSnapshot` · expanded `check_plugin_catalog.py` · canon §7.1.5.1 + author guide.

**Principle:** Integration → Tool → Skill → Agent (unchanged) · explicit first-party bootstrap + optional entry points · one P-Ext.* ID per PR · gate green.

**Production-path reality (do not confuse with MVP):**

| Layer | Shipped catalog | External extension | Runtime materialization |
|-------|-----------------|--------------------|-------------------------|
| **Integrations** | **~99** slugs (`preset="full"`) / **12** core (`preset="core"`) via `register_from_manifest` + `create_*` — **0** shipped `register.py` use `register_integration_plugin` | `IntegrationPlugin` + EP `intergrax.integrations` | `IntegrationProfile.resolve(category, config=…)` → backend instance |
| **Tools** | **13** bundles / **~29** `tool_id` — **13/13** via `ToolPlugin` (`shipped_plugins.py`) | `ToolPlugin` + EP `intergrax.tools` | `bootstrap_catalogs` → `build_registry_from_profile(ToolProfile, ctx)` → `ToolRegistry` → `RuntimeToolInvoker` / MCP |
| **Skills** | **3** bundles / **8** `skill_id` — **3/3** via `SkillPlugin` (`harness`×6, `legal`×1, `research`×1) | `SkillPlugin` + EP `intergrax.skills` | `build_registry_from_profile(SkillProfile)` → `SkillRegistry` → `SkillResolver` → `allowed_tools` |

**Out of scope for Phase P-Ext:**

- Online plugin marketplace UI / central registry service
- Runtime hot-reload of catalogs without process restart
- Skill as executable workflow graph (LangGraph pack) — separate initiative
- Replacing `ToolWiringContext` with a generic DI framework
- Migrating all ~99 shipped integrations to `IntegrationPlugin` classes (optional long-term; manifest path remains supported)

#### P-Ext.0 — Shared plugin foundation

**Goal:** One plugin loader and one Tier-3 bootstrap entry point.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.0.1 | **`load_plugins(group, …)`** — entry point discovery | **Done** | **Critical** | `intergrax/core/plugins/discovery.py` | Idempotent; `on_conflict=error\|skip` |
| P-Ext.0.2 | **Plugin errors** — `PluginConflictError`, `PluginLoadError` | **Done** | High | `intergrax/core/plugins/errors.py` | Unit tests |
| P-Ext.0.3 | **`bootstrap_catalogs()`** — unified Tier-3 composition | **Done** | **Critical** | `intergrax/core/catalog_bootstrap.py` | tool/skill wiring + idempotent shipped |
| P-Ext.0.4 | **`docs/EXTENSION_AUTHOR_GUIDE.md`** | **Done** | High | `docs/` | pip package walkthrough |
| P-Ext.0.5 | **Fixture pip package** in tests | **Done** | High | `tests/fixtures/plugin_packages/` | editable install; registers integration + tool + skill |
| P-Ext.0.6 | **EP discovery tests** via fixture (all three groups) | **Done** | High | `tests/unit/core/plugins/` | `bootstrap_catalogs(discover_entry_points=True)` loads fixture |
| P-Ext.0.7 | **`INTERGRAX_DISCOVER_PLUGINS`** env + Tier-3 wiring | **Done** | Medium | `catalog_bootstrap.py`, `applications/_shared/platform_wiring.py` | lab opt-in; default `false` in prod hosts |

**DoD:** Fixture package registers via entry point; discovery unit tests green.

**Entry point groups (canonical names):**

```toml
[project.entry-points."intergrax.integrations"]
[project.entry-points."intergrax.tools"]
[project.entry-points."intergrax.skills"]
```

---

#### P-Ext.1 — Integrations: plugin closure

**Baseline:** `IntegrationManifest`, `IntegrationPlugin`, `register_from_manifest`, per-provider `manifest.py` (open slug catalog).

**Audit snapshot (2026-06-02 — integrations only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | `bootstrap_core` **12** slugs + `bootstrap_extended` **~87** → **~99** full; all `register.py` call `register_from_manifest(MANIFEST, create_*)` | **Yes** — primary harness path |
| **`IntegrationPlugin` shipped** | **0/99** providers register via `register_integration_plugin` in shipped code | N/A — external / explicit only |
| **Reference plugin class** | `SqliteIntegrationPlugin` in `sqlite/plugin.py`; `register.py` still uses manifest path | Doc pattern only (P-Ext.1.12) |
| **External example** | `integrations/examples/custom_memory_kv/` + `test_external_plugin.py` (explicit register) | **Yes** API; EP not tested |
| **`IntegrationProfile.resolve`** | Manifest, plugin class, slug `str`, or pre-built instance via `IntegrationBinding` | **Yes** — Tier-3 prod |
| **`resolve_typed.py`** | Six typed helpers incl. vector_store, notification_channel, object_storage | **Done** |
| **`IntegrationSlug` enum** | **0** references in `intergrax/**/*.py` and provider `USAGE.md`; legacy mention only in plan + migration scripts | **Done** (P-Ext.1.5) |
| **Tier-3 bootstrap** | `integration_wiring` / `tool_wiring` / `skill_wiring` → `bootstrap_catalogs()` + lazy bundle ids | **Done** |
| **Entry points** | Fixture pip package + EP tests; `INTERGRAX_DISCOVER_PLUGINS` for lab | **Done** |
| **`on_conflict`** | `bootstrap_catalogs(on_conflict=…)` — `error`, `skip`, `override`, `warn_override` for catalog slugs + EP names | **Done** (P-Ext.4.3) |
| **Health API** | `integrations/registry/health.py` — `ping_integration` / `integration_registered` | **Done** |
| **Unit tests** | Per-provider tests + `test_profile` + `test_external_plugin` + lazy `preset="core"` in `test_lazy_catalog_bootstrap` | **Strong**; no full-count assertion in CI |

**Verdict:** Shipped integrations are **production-ready** on the **manifest + factory** path. `IntegrationPlugin` is **production-ready for third-party** extensions; parity with tools (all shipped as plugin classes) is **explicitly out of scope**.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.1.1 | Wire **`intergrax.integrations`** entry points in `bootstrap_catalogs()` | **Done** | **Critical** | `catalog_bootstrap.py` | `discover_entry_points=True` |
| P-Ext.1.2 | Split **`register_default_integrations()`** → core + optional | **Done** | High | `integrations/registry/bootstrap_core.py` | `preset="core"` (12) \| `"full"` (~99) |
| P-Ext.1.3 | **Typed resolve** helpers (top categories) | **Done** | Medium | `integrations/registry/resolve_typed.py` | 3 categories today |
| P-Ext.1.3a | Expand **`resolve_typed`** + unit tests | **Done** | Medium | `resolve_typed.py`, `tests/unit/integrations/test_resolve_typed.py` | +`vector_store`, `notification_channel`, `object_storage`; used in lab docs |
| P-Ext.1.4 | **Health check** API per slug (optional) | **Done** | Low | `integrations/registry/health.py` | `ping(slug) -> bool` smoke helper |
| P-Ext.1.5 | Remove **`IntegrationSlug`** from docs/scripts | **Done** | Medium | `**/USAGE.md`, `README.md`, `scripts/`, `docs/AGENT_CREATION_GUIDE.md` | `intergrax/**/*.py` already clean |
| P-Ext.1.6 | **EP integration test** via fixture | **Done** | High | `tests/unit/integrations/` | `discover_entry_points=True` loads fixture slug |
| P-Ext.1.7 | **Dual-model docs** — manifest+factory vs `IntegrationPlugin` | **Done** | Medium | `INTEGRATIONS.md`, `EXTENSION_AUTHOR_GUIDE.md` | decision table + when to migrate |
| P-Ext.1.8 | **CI smoke** — integration slug counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | `core` ≥12, `full` ≥95 (or exact snapshot) |
| P-Ext.1.9 | **`test_resolve_typed.py`** | **Done** | Low | `tests/unit/integrations/` | type errors on wrong contract |
| P-Ext.1.10 | **Tier-3** lab/poc use `bootstrap_catalogs(integration_preset=…)` | **Done** | High | `applications/*/host/integration_wiring.py` | replace bare `register_default_integrations()` |
| P-Ext.1.11 | **`applications/_shared/integration_wiring.py`** helper | **Done** | Medium | `applications/_shared/` | mirror `tool_wiring` — bootstrap + profile factory |
| P-Ext.1.12 | **`SqliteIntegrationPlugin`** — document or wire one shipped slug | **Done** | Low | `sqlite/register.py` or `INTEGRATIONS.md` | either `register_integration_plugin` in sqlite **or** “reference only” in docs |

**DoD:** 364+ integration unit tests green; external integration via entry point **and** via pip entry point (fixture); Tier-3 hosts use unified `bootstrap_catalogs()` for integrations.

---

#### P-Ext.2 — Tools: ToolPlugin + MCP export

**Baseline:** `ToolContract`, `ToolBundleEntry`, `ToolProfile`, `ToolWiringContext`, `RuntimeToolInvoker`.

**Audit snapshot (2026-06-02 — tools only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **13/13** bundles on `ToolPlugin` via `shipped_plugins.py` + `define_tool_plugin` | **Yes** — full plugin parity |
| **Tool count** | **~29** `tool_id` across bundles (RAG, websearch, jira, sandbox, vision, speech, …) | **Yes** |
| **Legacy register path** | No shipped bundle bypasses `register_tool_plugin`; `register_from_tool_manifest` is internal only | **Yes** |
| **External example** | `intergrax/tools/examples/` + `test_external_tool_plugin.py` | **Yes** |
| **EP `intergrax.tools`** | Fixture package + EP discovery tests (P-Ext.0.5 / 2.11) | **Yes** |
| **Tier-3 wiring** | `tool_wiring.build_application_tool_wiring` → `bootstrap_catalogs(register_shipped=True)` | **Yes** |
| **Lazy catalog** | `tool_wiring` passes `tool_bundle_ids` from `ToolProfile` | **Done** |
| **Runtime materialization** | Two-phase: catalog → `ToolWiringContext` + integrations → `ToolRegistry` handlers | **Yes** |
| **MCP / standalone LLM** | `export_mcp_tools`, `ToolsAgent`, `RuntimeToolInvoker` trace | **Yes** — strongest market path |
| **Unit tests** | Per-bundle tests + `test_external_tool_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped tools are **production-ready** on **`ToolPlugin`**; P-Ext.2 closure complete (external example, EP test, lazy `tool_wiring`).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.2.1 | **`ToolPlugin` Protocol** | **Done** | **Critical** | `intergrax/tools/core/plugin.py` | `tool_bundle_manifest()`, `register_tools(registry, ctx)` |
| P-Ext.2.2 | **`ToolManifest`** (bundle metadata) | **Done** | **Critical** | `intergrax/tools/core/manifest.py` | bundle_id, tool_ids, status |
| P-Ext.2.3 | **`register_tool_plugin()`** | **Done** | **Critical** | `intergrax/tools/registry/plugin_register.py` | Mirror integrations |
| P-Ext.2.4 | **Pilot migration** — RAG bundle → `ToolPlugin` | **Done** | High | `tools/providers/rag/` | Pattern for other bundles |
| P-Ext.2.5 | Entry point group **`intergrax.tools`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.2.6 | **`export_mcp_tools(registry)`** | **Done** | High | `intergrax/tools/exporters/mcp.py` | alias of `to_mcp_tools` |
| P-Ext.2.7 | **`ToolContract.version`** field (semver) | **Done** | Medium | `tools/core/contracts.py` | Default `1.0.0` |
| P-Ext.2.8 | **Migrate all shipped tool bundles** → `ToolPlugin` | **Done** | High | `tools/registry/shipped_plugins.py`, `providers/*/register.py` | 13/13 bundles |
| P-Ext.2.9 | **Reference external tool** — `tools/examples/` | **Done** | High | `intergrax/tools/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.2.10 | **`test_external_tool_plugin.py`** | **Done** | High | `tests/unit/tools/` | catalog → `build_registry_from_profile` → `RuntimeToolInvoker.invoke` |
| P-Ext.2.11 | **EP tool test** via fixture | **Done** | High | `tests/unit/tools/` | depends on P-Ext.0.5 |
| P-Ext.2.12 | **`tool_wiring` lazy bootstrap** — pass `tool_bundle_ids` from profile | **Done** | Medium | `applications/_shared/tool_wiring.py` | `bootstrap_catalogs(..., tool_bundle_ids=profile.enabled_bundles)` |

**DoD:** External tool executes via `RuntimeToolInvoker` after entry-point registration (test proves it); Tier-3 `tool_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.3 — Skills: SkillPlugin

**Baseline:** `SkillManifest`, `SkillBundleEntry`, `SkillResolver`, `AgentRegistry` merge to `allowed_tools`.

**Audit snapshot (2026-06-02 — skills only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **3/3** bundles on `SkillPlugin` via `shipped_plugins.py` + `register_default_skills()` | **Yes** — best plugin parity of Tier-0 |
| **Skill count** | **8** `skill_id`: `harness` (6), `legal` (1), `research` (1) | **Yes** |
| **Legacy `register_skill_bundle`** | Only in `plugin_register.py` + **outdated** `scaffold new-skill` output | Scaffold **not** prod (P-Ext.3.10) |
| **`register_from_skill_manifest`** | Internal helper; all shipped paths use `register_skill_plugin` | **Yes** |
| **External example** | `intergrax/skills/examples/` + external plugin tests | **Yes** |
| **EP `intergrax.skills`** | Fixture package + EP discovery tests (P-Ext.0.5 / 3.8) | **Yes** |
| **Tier-3 wiring** | `skill_wiring.build_application_skill_wiring` → `bootstrap_catalogs(register_shipped=True)` — **better than integrations** | **Yes** |
| **Lazy catalog** | `skill_wiring` passes `skill_bundle_ids` from `SkillProfile` | **Done** |
| **Runtime materialization** | Two-phase like tools: catalog bundle rows → `build_registry_from_profile` → `SkillRegistry` | **Yes** |
| **`requires_skills`** | Resolver + `test_requires_skills.py`; **0** shipped manifests use it | Feature **Done**; adoption open (P-Ext.3.12) |
| **Cursor `SKILL.md` importer** | `CursorSkillImporter` — parallel path, not `SkillPlugin` | **Yes** for import; document vs plugin (P-Ext.3.11) |
| **Agent merge** | `AgentRegistry.register(..., skill_registry=, tool_registry=)` + `test_agent_registry_skills.py` | **Yes** |
| **Unit tests** | Harness + resolver + `test_external_skill_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped skills are **production-ready** on **`SkillPlugin`**; P-Ext.3 closure complete (external example, EP test, lazy `skill_wiring`, scaffold alignment).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.3.1 | **`SkillPlugin` Protocol** | **Done** | **Critical** | `intergrax/skills/core/plugin.py` | `skill_bundle_manifest()`, `skill_manifests()`, `register_skills(registry)` |
| P-Ext.3.2 | **`register_skill_plugin()`** | **Done** | **Critical** | `intergrax/skills/registry/plugin_register.py` | Wraps `register_from_skill_manifest` |
| P-Ext.3.3 | Entry point group **`intergrax.skills`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.3.4 | Migrate **`harness`** + **`research`** + **`legal`** → `SkillPlugin` | **Done** | High | `skills/providers/*/plugin.py`, `shipped_plugins.py` | **3/3** bundles |
| P-Ext.3.5 | **`requires_skills`** on `SkillManifest` + resolver DFS | **Done** | Low | `skills/resolver.py`, `test_requires_skills.py` | Cycle + unknown dep errors |
| P-Ext.3.6 | **Reference external skill** — `skills/examples/` | **Done** | High | `intergrax/skills/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.3.7 | **`test_external_skill_plugin.py`** | **Done** | High | `tests/unit/skills/` | explicit `register_skill_plugin` → `SkillResolver` → tool merge |
| P-Ext.3.8 | **EP skill test** via fixture | **Done** | High | `tests/unit/skills/` | depends on P-Ext.0.5 |
| P-Ext.3.9 | **`skill_wiring` lazy bootstrap** — pass `skill_bundle_ids` from profile | **Done** | Medium | `applications/_shared/skill_wiring.py` | `bootstrap_catalogs(..., skill_bundle_ids=profile.enabled_bundles)` |
| P-Ext.3.10 | **Scaffold `new-skill`** emits `SkillPlugin` + `plugin.py` | **Done** | Medium | `intergrax/scaffold/new_skill.py` | remove legacy `register_skill_bundle` template |
| P-Ext.3.11 | **Docs: SkillPlugin vs Cursor importer** | **Done** | Medium | `SKILLS.md`, `EXTENSION_AUTHOR_GUIDE.md` | when to use pip plugin vs `SKILL.md` import |
| P-Ext.3.12 | **`requires_skills` in shipped harness** (optional demo) | **Done** | Low | `skills/providers/harness/manifests.py` | one derived skill depending on `harness.tool_smoke` |

**DoD:** External skill merges `allowed_tools` on `AgentRegistry.register` (test proves it); Tier-3 `skill_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.4 — Operational scale

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.4.1 | **Lazy bootstrap** — register only bundles in active `*Profile` | **Done** | High | `catalog_bootstrap.py`, bootstrap modules | `tool_bundle_ids`, `skill_bundle_ids`, `integration_preset` |
| P-Ext.4.2 | **`CatalogSnapshot` API** (read-only) | **Done** | Medium | `intergrax/core/catalog_snapshot.py` | list slugs for docs/UI |
| P-Ext.4.3 | Slug conflict policy in bootstrap | **Done** | Medium | `catalog_bootstrap.py` | `error` / `warn_override` |
| P-Ext.4.4 | CI **`check_plugin_catalog.py`** | **Done** | High | `scripts/` | smoke: shipped bundles present |
| P-Ext.4.5 | **Expand CI smoke** — all three catalog counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | tools **13** bundles / ~**29** tool_id; skills **3** bundles / **8** skill_id; integrations **core≥12**, **full≥95** (see also P-Ext.1.8) |

---

#### P-Ext.5 — Docs, scaffold, canon

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.5.1 | Scaffold **`new_integration` / `new_tool_bundle` / `new_skill_bundle`** | **Done** | Medium | `intergrax/scaffold/` | manifest + plugin + register |
| P-Ext.5.2 | **External plugins** sections in INTEGRATIONS/TOOLS/SKILLS | **Done** | Medium | `docs/` | Cross-link Appendix I |
| P-Ext.5.3 | **Canon §7.1.5.1** — entry points + plugin protocols | **Done** | High | `intergrax_runtime_architecture.md` | §7.1.5.1 Tier-0 Plugin Catalogs |
| P-Ext.5.4 | Remove duplicate `PLUGIN_CATALOG_PLAN.md` | **Done** | Low | — | tracking only in this plan + Appendix I |
| P-Ext.5.5 | **Prod path matrix** in author guide (integration vs tool vs skill) | **Done** | Medium | `EXTENSION_AUTHOR_GUIDE.md` | two-phase tool bootstrap documented |
| P-Ext.5.6 | **Lab wiring recipe** for external plugins | **Done** | Medium | `applications/lab_application/`, `TIER3_READINESS.md` | `discover_entry_points` + profile example |

---

#### P-Ext.6 — Production closure (paydown)

**Goal:** Close gaps between **MVP** (API + shipped catalogs) and **production-ready extensibility** (tested pip install, parity across three layers, ops hooks).

| # | Deliverable | Status | Priority | Depends on | Acceptance |
|---|-------------|--------|----------|------------|------------|
| P-Ext.6.1 | **Fixture pip package** (unblocks EP tests) | **Done** | **Critical** | — | same as P-Ext.0.5 |
| P-Ext.6.2 | **External tool + skill examples + tests** | **Done** | **Critical** | 6.1 | P-Ext.2.9–2.11, P-Ext.3.6–3.8, 3.7 green |
| P-Ext.6.8 | **Skill Tier-3 + scaffold** (rollup) | **Done** | Medium | — | P-Ext.3.9–3.12, scaffold overlap P-Ext.5.1 |
| P-Ext.6.9 | **Tool Tier-3 lazy wiring** (rollup) | **Done** | Medium | — | P-Ext.2.12 (symmetric with P-Ext.3.9) |
| P-Ext.6.10 | **Tier-3 lazy wiring** (all catalogs rollup) | **Done** | Medium | — | P-Ext.2.12 + P-Ext.3.9 + optional `integration_preset` in shared helpers |
| P-Ext.6.3 | **EP discovery** in tests + lab env flag | **Done** | High | 6.1 | P-Ext.0.6–0.7, P-Ext.1.6 |
| P-Ext.6.4 | **IntegrationSlug cleanup** in docs/scripts | **Done** | Medium | — | P-Ext.1.5 |
| P-Ext.6.5 | **Scaffold** `new_tool_bundle` / `new_skill_bundle` / `new_integration` | **Done** | Medium | — | P-Ext.5.1 |
| P-Ext.6.6 | **Integration Tier-3** + typed resolve + health (rollup) | **Done** | Medium | — | P-Ext.1.3a, 1.4, 1.8–1.11 |
| P-Ext.6.7 | **Conflict policy** + expanded CI smoke | **Done** | Medium | — | P-Ext.4.3, P-Ext.4.5, P-Ext.1.8 |

**DoD (phase closure):** Appendix I has no **Planned** P0/P1 rows; external integration, tool, and skill each proven via **entry point** (fixture package), not only explicit in-process registration.

---

#### Phase P-Ext — Definition of done

**MVP (met 2026-06-02):**

1. `bootstrap_catalogs()` + three plugin protocols + lazy presets.
2. All shipped tool/skill bundles on `ToolPlugin` / `SkillPlugin`.
3. Integration example `custom_memory_kv` + `test_external_plugin.py`.
4. Canon §7.1.5.1 + `EXTENSION_AUTHOR_GUIDE.md` (EN).
5. Gate: `tests/unit/core/plugins`, integrations/tools/skills plugin tests green.

**Production closure (P-Ext.6 — open):**

1. **Fixture pip package** registers integration + tool + skill without Intergrax core edits.
2. **EP discovery tests** for all three groups (`discover_entry_points=True`).
3. **External tool test** — `RuntimeToolInvoker` after EP registration.
4. **External skill test** — `allowed_tools` merge after EP registration.
5. **Tier-3** documents/env for optional discovery; default remains explicit bootstrap.
6. **Tier-3 lazy wiring** — `tool_wiring` and `skill_wiring` pass profile bundle ids to `bootstrap_catalogs()` (P-Ext.2.12, P-Ext.3.9).
7. **No central slug enum** in new code/docs (string slugs); `IntegrationSlug` removed from author-facing examples.
8. **MCP export** from active `ToolRegistry` (already met).
9. Appendix I: all P-Ext.* rows **Done** or **Won't fix** with reason.

#### Phase P-Ext — Recommended execution order

```text
MVP (Done):               P-Ext.0.1–0.4 | P-Ext.1.1–1.2 | P-Ext.2.1–2.8 | P-Ext.3.* | P-Ext.4.1–4.2,4.4 | P-Ext.5.2–5.4

Paydown Wave P1 (critical):
  P-Ext.0.5 → P-Ext.0.6 → P-Ext.1.6 → P-Ext.1.10
           → P-Ext.2.9 → P-Ext.2.10 → P-Ext.2.11
           → P-Ext.3.6 → P-Ext.3.7 → P-Ext.3.8

Paydown Wave P2 (ops + docs):
  P-Ext.0.7 → P-Ext.4.3 → P-Ext.4.5 → P-Ext.1.8 → P-Ext.1.5 → P-Ext.1.7 → P-Ext.5.5 → P-Ext.5.6
           → P-Ext.2.12 → P-Ext.3.9 → P-Ext.3.10 → P-Ext.3.11

Paydown Wave P3 (optional polish):
  P-Ext.1.3a → P-Ext.1.4 → P-Ext.5.1 → P-Ext.3.12
```

**Effort estimate:** MVP ~21–32 person-days (**spent**); paydown **~12–18** person-days incl. integration + tool + skill closure (Appendix I).

**Priority ladder:** **Band 2c** (§4.0) — harness Tier-0 extensibility; **not** Band 3 product work.

---

## 4. Priority Order

### 4.0 Implementation priority ladder (canonical)

**Read this before §6.** The plan has three bands. Implement **top to bottom**. **Never** pull items from band 3 into “next step” summaries while band 1–2 are the active policy.

| Band | What | Status (2026-06-05) | Examples |
|------|------|---------------------|----------|
| **1 — Harness platform** | Tier-0/1/3 lab wiring, security, policy, typing, legacy removal, gate audits | **Maintenance** (§4.1 **Done**; keep green) | `pytest -m gate`, `check_harness_*`, `check_tools_agent_*`, regression fixes |
| **2 — Harness architecture hardening** | Capability graph, lifecycle governance, prompt/eval/context/security/cost/metrics hardening — **no** business domain | **Done** (2026-06-05) | V-CG … V-KG, V-V6 closeout · V-REM |
| **2i — Phase V runtime remediation (V-REM)** | Close 9 Partial Phase V + EvalRunner gate gaps — runtime enforcement, not new OS features | **Done** (2026-06-05) | [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · Appendix J |
| **2b — Modality plane (optional parallel)** | Vision CV, speech, classical ML — harness Tier-0 only | **Done** | W-ML complete; optional Celery bus wiring for Tier-3 scale-out |
| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **Done** (2026-06-02) | Appendix I · [EXTENSION_AUTHOR_GUIDE.md](EXTENSION_AUTHOR_GUIDE.md) |
| **2d — Operational L3 (W-OPS)** | Reliability, identity, SLO/ops evidence, online eval — **no** business agents | **Done** (2026-06-06) | [Phase W-OPS](#phase-w-ops--operational-harness-maturity-ideal-l3-ops) · `phase_w_ops_evidence.py` |
| **2e — Application environment (H-APP)** | `ApplicationEnvironmentProfile`, unified Tier-3 wiring, host migration — **no** business agents | **Done** (2026-06-03) | [Phase H-APP](#phase-h-app--tier-3-application-environment-full-configurability) · [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) · **§6.2x** |
| **2f — Developer authoring UX (DX)** | LangGraph-like facades, minimal scaffold, CLI run/doctor, TTFRun gates, UI spec export — **no** business agents | **Done** (2026-06-03) | [Phase DX](#phase-dx--developer-authoring-experience-fast-environment--agent-builds) · **§6.2y** |
| **2g — Agents & applications conformance (AA)** | Scaffold alignment, per-agent/app `ARCHITECTURE.md`, deploy triad, legal **scaffold** reset (domain steps → Band 3) | **Mostly Done** (2026-06-02) | [Phase AA](#phase-aa--agents--applications-conformance-scaffold-docs-deploy) · **§6.2z** · [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **2h — Memory platform (MEM)** | H-APP→runtime bridge, durable user LTM, session SQLite, gates, hooks, memory docs — **no** business agents | **Done** (2026-06-02) | [Phase MEM](#phase-mem--memory-platform-completion) · **§6.2aa** |
| **3 — END OF PLAN (product)** | Business agents, new product Tier-3 apps, domain skills, Legal live E2E | **Deferred** — **[§6.3](#63-end-of-plan--deferred-product-work-only)** | K.1, K.2, `applications/<product>/`, K.6, B.15, S-Ops.4 |

**Hard rule:** Band 3 is **not** “next after harness.” It runs only after an **explicit product prioritization decision** (Appendix A for agents; separate decision for new applications). Until then, **do not** implement, extend, or schedule K.1/K.2 waves, new product hosts, or product-only E2E in implementation cadence (§6.1–§6.2).

**Policy (2026-06-05):** Harness completion in §4.1 is **Done**. Band 1 = keep gate green. Bands 2 / 2b–**2i** platform rows = **Done** (V-REM closed 2026-06-05). Band 3 = **frozen** unless leadership reprioritizes.

```text
BAND 1:  Harness maintenance — gate + audit scripts (§6.1)
BAND 2:  Harness architecture hardening — Phase V + V-REM — DONE (2026-06-05)
BAND 2i: Phase V runtime remediation — V-REM — DONE (2026-06-05)
BAND 2d: Operational L3 — Phase W-OPS (§6.2w) — DONE
BAND 2e: Application environment — Phase H-APP (§6.2x) — DONE (43 tasks)
BAND 2f: Developer authoring UX — Phase DX (§6.2y) — DONE (47 tasks)
BAND 2g: Agents & applications conformance — Phase AA (§6.2z) — MOSTLY DONE (platform); domain → Band 3
BAND 2h: Memory platform — Phase MEM (§6.2aa) — DONE (48/48)
BAND 3:  END OF PLAN — product agents & applications (§6.3) — DO NOT SCHEDULE AS DEFAULT NEXT

DONE:    Harness completion backlog (§4.1) — 2026-06-02
DONE:    Phase U — Harness production hardening (2026-06-01)
DONE:    Phase T — Harness cleanliness (2026-06-01)
DONE:    Phase S — Harness environment GA (2026-06-01)
DONE:    Phase Q+ — Harness Hardening (Appendix D)
DONE:    Phase R (MVP) — Harness AI alignment (Appendix E)
DONE:    Phase Q — Harness Quality (audit #1) — Waves 1–9
DONE:    Phase L, M, M-LLM, M-RAG, N, O — harness GA (functional)
DONE:    Phase K hardening K.3–K.5; Appendix B paydown (except B.15)

PARALLEL (harness-only): M.6 provider slugs on demand; R-Skill catalog expansion (platform packs)

BAND 3 — END OF PLAN (see §6.3; not default “next”):
  • K.1 Problem Radar / K.2 Vendor Discovery (business agents)
  • K.6 / B.15 / S-Ops.4 — Legal live LLM E2E (product/CI)
  • New Tier-3 **product** applications (beyond lab + existing reference hosts)
  • Domain skill packs for product agents (until K.* started)
  • Problem Radar wave 2+ (`agents/problem_radar/` frozen)

RULE:    Strategy → canon → plan → code; Tier-1 via §0.6; four layers Integration → Tool → Skill → Agent
```

**Rationale:** Phases S/T/U + §4.1 delivered a production-configurable **harness**. Band 1–2 preserve and extend that platform. **Band 3 (product) is intentionally last** so business agents and new applications do not drive Tier-1 evolution (canon §52, [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md)).

### 4.0a Implementation scope split (infrastructure vs business)

**Canonical rule:** Default implementation queue = **infrastructure only** (Bands 1–2g + §6.1). **Business** work runs only after explicit product prioritization — **[§6.3](#63-end-of-plan--deferred-product-work-only)**.

| Layer | Bands / phases | What it includes | Default queue |
|-------|----------------|------------------|---------------|
| **Infrastructure (Intergrax Harness)** | 1, 2, 2b–2i (platform rows) | `intergrax/runtime/`, Tier-0 catalogs, H-APP, DX, MEM, scaffold, CI audits, reference hosts | **Active** — §6.1 maintenance only |
| **Conformance shells (platform)** | 2g AA | `legal` / `legal_application` **scaffold** + deploy triad + tier hygiene (no domain UAEP steps) | **Done** (shell) |
| **Business agents & product apps** | 3, §6.3, AA-LEG.2.*, K.* | K.1/K.2, Legal UAEP steps, research/org domain tests, new `applications/<product>/`, live LLM E2E | **Deferred** — not default next |

**Module classification (repo inventory):**

| Module | Role | Queue |
|--------|------|-------|
| `agents/echo`, `agents/signoff_probe` | Harness reference Tier-2 | Infrastructure — **Done** |
| `agents/lab` | Lab mocks (not product agents) | Infrastructure — AA-LABAG.* optional |
| `applications/poc_template_application`, `applications/lab_application` | Reference Tier-3 hosts | Infrastructure — **Done** |
| `agents/legal`, `applications/legal_application` | Product shell on scaffold | Platform **Done**; domain logic **Deferred** (AA-LEG.2.2+) |
| `agents/research`, `applications/research_application` | Research prototype host | Platform **Done**; domain tests **Deferred** (AA-RES.4–5, AA-RESAPP.6) |
| `agents/organization_worker` | HITL / long-running demo | Docs **Done**; full scaffold + lab flag **Deferred** (AA-ORG.3–4) |
| `agents/problem_radar` | K.1 placeholder | **Frozen** — Band 3 (K.1) |
| New `applications/<product>/` beyond four hosts | Customer/product deploy | **Deferred** — §6.3 |

**Where to look for open work:**

| Topic | Section |
|-------|---------|
| **Canonical implementation queue (infrastructure)** | [§6.1z](#61z-harness-implementation-queue-consolidated) (**active** — V-REM) · [§6.1aa](#61aa-harness-implementation-queue-memory-platform) (closed) |
| Ongoing gate + audit scripts | [§6.1](#61-harness-platform-maintenance-default--band-1) |
| Memory platform (Done — §6.1 maintenance) | [Phase MEM](#phase-mem--memory-platform-completion) · [§6.2aa](#62aa-phase-mem-execution-order-band-2h--active) |
| All business / domain work | [§6.3](#63-end-of-plan--deferred-product-work-only) · [Business backlog register](#63a-business-backlog-register-consolidated) |

### 4.1 Harness completion backlog (execution order)

Work **one ID per PR**; gate green after each step. Map fixes to Appendix G where applicable.

| Order | ID | Deliverable | Priority | Notes |
|-------|-----|-------------|----------|-------|
| 1 | U-Leg.2 | Remove or archive `intergrax/rag/answers/`; migrate tests to `RetrievalService` | **Done** | `intergrax/legacy/rag_answers/`; import guard |
| 2 | U-Leg.1 | Freeze `ToolsAgent.run` — docs + `check_tools_agent_run.py` | **Done** | Deprecation + CI audit |
| 3 | U-Leg.3 | Sunset legacy plan booleans (`from_legacy`, `uses_legacy_booleans_only`) | **Done** | Warnings + `check_legacy_tool_plan_booleans.py` |
| 4 | U-Typ.4 | `profile.slug_for_category` + sandbox `session_id` typing | **Done** | No getattr on integration profile |
| 5 | U-Arch.2 | Typed `LabIntegrationWiring` — sqlite bundle types | **Done** | Removed `# type: ignore` on lab wiring |
| 6 | U-CI.3 | CI job: `LAB_STRICT_HARNESS` + API key | **Done** | `harness-strict` workflow job |
| 7 | R-Skill.* | `harness.skill_registry` platform skill | **Done** | Harness bundle + gate test |
| 8 | U-Con.* | `ResearchAgent` / `SummaryAgent` → `HarnessReferenceAgent` | **Done** | Lab `requires_uaep` when research enabled |

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new `applications/<product>/`, Problem Radar wave 2+.



---



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---



## 6. What to implement next

**Default answer (infrastructure only):** **[§6.1](#61-harness-platform-maintenance-default--band-1)** — keep gate green; Phase V-REM **Done**. Phase MEM **Done** (48/48). Product work gated by [§6.3](#63-product--business-scope-gate).

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](#63a-business-backlog-register-consolidated)** · **[§4.0a](#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Plan/code audit (2026-06-05) → Phase V-REM + Appendix J; prior MEM/DX/AA closeouts in [§6.1aa](#61aa-harness-implementation-queue-memory-platform).

### 6.1aa Harness implementation queue — memory platform (closed)

**Purpose:** Phase MEM execution queue — **closed 2026-06-02** (48/48 Done). Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-1.1–MEM-1.4** | Code | **Done** | H-APP `MemoryProfile` + `ContextProfile.budget` + SQLite session → `RuntimeConfig` | MEM-1.5 gate test green |
| 2 | **MEM-2.1–MEM-2.3** | Code | **Done** | `SQLiteUserProfileStore` + bundle wiring + unit tests | LTM survives restart on sqlite profile |
| 3 | **MEM-1.6** | Docs/status | **Done** | H-APP.4.3 → **Done** | Bridge complete |
| 4 | **MEM-4.1–MEM-4.3** | Test | **Done** | Session + LTM + full-stack memory gates | acceptance/integration green |
| 5 | **MEM-5.1–MEM-5.2** | Test/Docs | **Done** | `engine_history_layer` tests + compression docs | unit + guide |
| 6 | **MEM-3.1–MEM-3.3** | Code | **Done** | Memory store plugin EP + reference fixture | bootstrap + gate |
| 7 | **MEM-0.3–MEM-DOC.*** | Docs | **Done** | Author cookbooks + Appendix G sync | guide updated |
| 8 | **MEM-6.*–MEM-7.*** | Code | **Done** | Retention enforcement + memory hooks | P2 after P0/P1 |
| 9 | **MEM-8.*–MEM-9.*** | RFC | **Done (RFC)** | Product memory layer + entity graph design | §6.3 gate for implementation |

**Suggested PR order:** See [Phase MEM — Suggested PR order](#mem--paydown-log).

**Explicitly excluded:** K.1, K.2, Mem0 SaaS product, entity graph ship (RFC only), business agent memory.

### 6.1z Harness implementation queue (consolidated — closed 2026-06-05)

**Purpose:** Single ordered list of **infrastructure** work. Excludes Band 3 / [§6.3a](#63a-business-backlog-register-consolidated). **Closed 2026-06-05** — Phase V-REM complete. Prior DX/AA/MEM/W-OPS/H-APP rows remain **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green; scripts in [§6.1](#61-harness-platform-maintenance-default--band-1) |
| 1 | **V-REM-CG.1** | Code | **Done** | Fix per-application capability graph system edges | V-CG.2–4 closed |
| 2 | **V-REM-CG.2** | Test/CI | **Done** | Re-validate lineage/impact/compatibility on corrected graph | `phase_v_capability_graph_guard.py` green |
| 3 | **V-REM-ALG.1** | Code | **Done** | Runtime filter retired/deprecated agents | Unit tests green |
| 4 | **V-REM-ALG.2** | Code | **Done** | Production-eligible + owner gate at selection | Strict harness test green |
| 5 | **V-REM-PE.1** | Code | **Done** | PromptMeta owner/risk schema | Registry validation tests |
| 6 | **V-REM-PE.2** | Assets | **Done** | YAML prompt assets catalog seed | E2E governance validation |
| 7 | **V-REM-SEC.1** | Code | **Done** | Tool injection defense on execution path | Middleware unit tests |
| 8 | **V-REM-SEC.2** | Code | **Done** | Retrieval poisoning middleware per tenant/app | RagStep filter unit tests |
| 9 | **V-REM-SEC.3** | Code | **Done** | Tenant isolation + audit trail in main path | Intake middleware unit tests |
| 10 | **V-REM-A.1** | Test | **Done** | NexusEvalRunner integration + gate | A.4 → **Done** |
| — | **REG-*** | Regression | As needed | Fix gate/CI failures only | No feature scope |

**Closed (no implementation — do not reopen without regression):**

| ID | Resolution |
|----|------------|
| DX-0.3–DX-8.2 (except DX-5.7) | **Done** — 2026-06-02 DX residual closeout |
| AA-LABAG.1, AA-SIG.2, AA-LABAPP.6 | **Done** |
| AA-LABAG.2 | **Won't fix** — mocks remain in `agents/lab/` until leadership requests move |
| W-OPS.1–15, H-APP.0–6.3, P-Ext, Q–V contracts, MEM 48/48 | **Done** |
| V-REM.0.1, V-REM.0.2 | **Done** — 2026-06-05 plan sync |
| V-REM-CG.1–A.1 | **Done** — 2026-06-05 runtime remediation |

**Explicitly excluded from this queue (business — implement only after §6.3 decision):** K.1, K.2, K.6, B.15, S-Ops.4, A.5, AA-LEG.2.2+, AA-LEGAPP.6–8, AA-RES.4–5, AA-RESAPP.6, AA-ORG.3–4, new Tier-3 product apps, domain skills — full list: [§6.3a](#63a-business-backlog-register-consolidated).

**Suggested PR order:** V-REM-CG.1 → V-REM-CG.2 → V-REM-ALG.1 → V-REM-ALG.2 → V-REM-SEC.1 → V-REM-SEC.2 → V-REM-SEC.3 → V-REM-PE.1 → V-REM-PE.2 → V-REM-A.1. Regressions → **REG-*** under §6.1.

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; fix regressions under band 1 only.

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/check_harness_no_getattr.py
  python scripts/check_tools_agent_imports.py
  python scripts/check_tools_agent_run.py
  python scripts/check_legacy_tool_plan_booleans.py
  python scripts/check_plugin_catalog.py
  uv run python scripts/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/check_scaffold_harness_alignment.py
  python scripts/check_agents_no_tier3_imports.py
  uv run intergrax doctor --ci
  uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/phase_v_capability_graph_guard.py --enforce
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

**Orchestration audit residuals (optional — pull when needed):** wire `OrchestrationProfile.planner_kind` / `classifier_kind` in `build_nexus_loop_from_environment`; runtime bridge `ApplicationGraphSpec` → initial `NexusPlan`; `max_parallel_nodes` on graph batches. Author map: [`AGENT_CREATION_GUIDE.md` Appendix I](AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane) §I.4–§I.5.

### 6.2v Phase V-REM execution order (Band 2i — closed 2026-06-05)

**Status:** **Done** · register: [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · queue: [§6.1z](#61z-harness-implementation-queue-consolidated) (closed)

Work **one V-REM ID per PR**; after each step update the V-REM master table + Appendix J + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Closes |
|-------|-----|-------------|----------|--------|
| 1 | V-REM-CG.1 | Fix per-application capability graph system edge mapping | **Critical** | V-CG.2–4 |
| 2 | V-REM-CG.2 | Re-validate lineage/impact/compatibility on corrected graph | High | V-CG.2–4 |
| 3 | V-REM-ALG.1 | Runtime filter for retired/deprecated agents | High | V-ALG.3 |
| 4 | V-REM-ALG.2 | Production-eligible + owner gate at agent selection | High | V-ALG.4 |
| 5 | V-REM-SEC.1 | Tool injection defense on main execution path | High | V-SEC.2 |
| 6 | V-REM-SEC.2 | Retrieval poisoning middleware per tenant/app | High | V-SEC.3 |
| 7 | V-REM-SEC.3 | Tenant isolation + audit trail in UnifiedTaskRunner/NexusLoop | High | V-SEC.4 |
| 8 | V-REM-PE.1 | PromptMeta owner/risk schema + validation | High | V-PE.1 |
| 9 | V-REM-PE.2 | YAML prompt assets catalog seed | Medium | V-PE.1 |
| 10 | V-REM-A.1 | NexusEvalRunner integration tests + gate | Medium | A.4, A.4.1 |

**Phase V-REM closeout:** **Done** (2026-06-05). Verified via `phase_v_closeout_gate.py --enforce --enforce-l4`.

### 6.2w Phase W-OPS execution order (Band 2d — complete 2026-06-06)

**Status:** **Done** · register: [Phase W-OPS](#phase-w-ops--operational-harness-maturity-ideal-l3-ops)

Work **one W-OPS ID per PR**; after each step update the W-OPS table + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | IDEAL gap |
|-------|-----|-------------|----------|-----------|
| 1 | W-OPS.1 | Side-effect tool idempotency keys + dedup | **Critical** | Reliability §8.3 |
| 2 | W-OPS.2 | Integration circuit breaker (`_shared`) | **Critical** | Reliability §8.2 |
| 3 | W-OPS.3 | Long-running / checkpoint / retry gate tests | High | Reliability §8.3 |
| 4 | W-OPS.6 | `tenant_id` on TaskEnvelope → trace/events | High | Identity §3.2 |
| 5 | W-OPS.7 | Mandatory harness API key (staging profile) | High | Identity §3.2 |
| 6 | W-OPS.4 | SLO catalog + incident budget + runbooks | **Critical** | Observability §11 |
| 7 | W-OPS.5 | L3-ops evidence (2 release cycles) | **Critical** | §12.3 vs V-V6 CI |
| 8 | W-OPS.8 | `harness.*` platform skill packs | Medium | Capability §3.6 |
| 9 | W-OPS.9 | `requires_skills` shipped demo | Medium | Registries §19 |
| 10 | W-OPS.10 | Harness lab stable stack health (catalog slugs) | Medium | Capability §3.6 |
| 11 | W-OPS.11 | Online/shadow evaluation registry writes | Medium | Evaluation §18 |
| 12 | W-OPS.12 | W-ML Celery Tier-3 scale-out (optional) | Low | Modality §3.5.1 |
| 13 | W-OPS.13 | ToolsAgent removal roadmap | Low | Cognition hygiene |
| 14 | W-OPS.14 | Typed wiring (no `load_callable`) | Low | DX §22 |
| 15 | W-OPS.15 | Architecture metrics threshold enforcement | Low | §21.6 |

**Wave P0 (orders 1–7)** must be **Done** before declaring **operational IDEAL L3**. **Wave P1/P2** run in parallel with P0 when owners differ.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new product applications, Problem Radar wave 2+.

### 6.2x Phase H-APP execution order (Band 2e — complete 2026-06-03)

**Status:** **Done** · canonical register: [Phase H-APP — Master deliverables register](#h-app--master-deliverables-register-all-43-tasks) · audit narrative: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.

Work **one H-APP ID per PR**; after each step update the H-APP master table + paydown log; keep §6.1 scripts green.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| H0 | H-APP.0.1–H-APP.0.5 | 5 | Terminology, CI guards, `poc_template` getattr fix, manifest conformance |
| H1 | H-APP.1.1–H-APP.1.8 | 8 | `ApplicationEnvironmentProfile`, unified wiring, runtime bridge, LLM resolver |
| H2 | H-APP.2.1–H-APP.2.8 | 8 | Identity, policy DSL, execution modes, V-SEC per application |
| H3 | H-APP.3.1–H-APP.3.6 | 6 | Orchestration profile, graph spec, Nexus factory, shadow/sandbox |
| H4 | H-APP.4.1–H-APP.4.8 | 8 | Context, memory, reliability, observability profiles |
| H5 | H-APP.5.1–H-APP.5.5 | 5 | Migrate lab/legal/research/poc/docker_verify + scaffold |
| H6 | H-APP.6.1–H-APP.6.3 | 3 | Operational L3 sign-off (release cycles + CI + audit §4) |
| **Total** | | **43** | |

**Suggested PR order (same as Phase H-APP paydown):** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new **product** Tier-3 apps, Problem Radar wave 2+, marketplace UI, catalog hot-reload.

### 6.2y Phase DX execution order (Band 2f — mostly done)

**Status:** **Done** (2026-06-02) · **47/47 Done** · canonical register: [Phase DX — Master deliverables register](#dx--master-deliverables-register-all-47-tasks).

Work **one DX ID per PR**; after each step update the DX master table + paydown log; keep §6.1 scripts green. **Start with DX1 (scaffold/H-APP alignment)** before DX2 facades — otherwise new authors copy broken `factory.py` patterns.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| DX0 | DX-0.1–DX-0.4 | 4 | LangGraph mapping, responsibility matrix, progressive disclosure |
| DX1 | DX-1.1–DX-1.6 | 6 | **P0** — scaffold + poc/legal/research factories on H-APP path only |
| DX2 | DX-2.1–DX-2.6 | 6 | `HarnessApplication`, `AgentGraph`, `IntergraxAgent` + `@step` |
| DX3 | DX-3.1–DX-3.6 | 6 | `--minimal` stack, `intergrax run`, `doctor`, TTFRun acceptance |
| DX4 | DX-4.1–DX-4.4 | 4 | Integration presets + picker + gate tests |
| DX5 | DX-5.1–DX-5.8 | 8 | Host hooks, YAML loader, logging, event catalog, policy rule plugins |
| DX6 | DX-6.1–DX-6.5 | 5 | Tier-2 hygiene, external `intergrax init` template |
| DX7 | DX-7.1–DX-7.5 | 5 | JSON Schema + spec versioning + UI feed (Phase 2 prep) |
| DX8 | DX-8.1–DX-8.3 | 3 | `doctor --ci`, DX metrics artifact, scaffold alignment script |
| **Total** | | **47** | |

**Suggested PR order:** DX-1.1 → DX-1.2 → DX-1.3 → DX-1.6 → DX-8.3 → DX-2.1 → DX-2.2 → DX-2.3 → DX-2.5 → DX-3.1 → DX-3.2 → DX-3.5 → DX-3.6 → DX-4.1 → DX-4.4 → DX-1.4–DX-1.5 → DX-2.4 → DX-2.6 → DX-3.3–DX-3.4 → DX-5.1–DX-5.2 → DX-6.1–DX-6.2 → DX-4.2–DX-4.3 → DX-5.3–DX-5.8 → DX-6.3–DX-6.5 → DX-7.1–DX-7.5 → DX-8.1–DX-8.2 → DX-0.1–DX-0.4.

**Success gate for Phase DX full closeout:** All rows **Done** or **Won't fix**; DX-3.5 + DX-8.1 green in CI; DX-3.6 quickstart validated; DX-7.1 schemas under `build/harness_specs/`. **Core path (DX1–DX2, DX3.2–3.3, DX8.3) already meets harness authoring needs.**

**Explicitly out of NOW:** K.1, K.2, visual environment builder UI, new product Tier-3 apps, Problem Radar wave 2+.

### 6.2z Phase AA execution order (Band 2g — mostly done)

**Status:** **Mostly Done** (2026-06-02) · platform **Done** · domain **Deferred** · canonical register: [Phase AA — Master deliverables register](#aa--master-deliverables-register-all-tasks).

Work **one AA ID per PR/session**; after each step update the AA master table + paydown log + conformance matrix; keep §6.1 scripts green. **Legal:** follow **hard reset** policy (AA-LEG.0.1) — no incremental preservation of legacy pipeline code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| AA0 | AA-0.1, AA-0.2, AA-S0.1–AA-S0.6, AA-LG.1, AA-APP.0.1–AA-APP.0.3 | 12 | Scaffold checklist, tier guards, deploy triad standard |
| AA1 | AA-D0.1–AA-D0.7 | 7 | README, guides, TIER3_READINESS, USAGE |
| AA2 | AA-LEG.0.2–AA-LEG.3.1 | 12 | **Legal agent hard reset** |
| AA3 | AA-LEGAPP.1–AA-LEGAPP.8 | 8 | `legal_application` + deploy triad |
| AA4 | AA-ECHO.1–AA-ECHO.5 | 5 | Reference echo agent |
| AA5 | AA-SIG.1–AA-SIG.3 | 3 | Signoff probe |
| AA6 | AA-PR.1–AA-PR.5 | 5 | Problem radar (docs/hygiene; frozen feature) |
| AA7 | AA-ORG.1–AA-ORG.5 | 5 | Organization worker |
| AA8 | AA-RES.1–AA-RES.6 | 6 | Research agents |
| AA9 | AA-LABAG.1–AA-LABAG.2 | 2 | Lab mocks |
| AA10 | AA-LABAPP.1–AA-LABAPP.7 | 7 | Lab application host |
| AA11 | AA-POC.1–AA-POC.5 | 5 | POC template (canonical shell) |
| AA12 | AA-RESAPP.1–AA-RESAPP.6 | 6 | Research application host |
| **Total** | | **83** | |

**Suggested PR order:** AA-S0.2 → AA-S0.5 → AA-APP.0.1 → AA-APP.0.3 → AA-POC.1 → AA-POC.2 → AA-LABAPP.2 → AA-ECHO.2 → AA-LEG.0.3 → AA-LEG.1.1 → AA-LEG.1.2 → AA-LEG.1.3 → AA-LEG.2.1 → AA-LEG.2.2 → … → AA-LEGAPP.1–AA-LEGAPP.6 → AA-D0.1 → AA-D0.3–AA-D0.5 → AA-RESAPP.* → AA-LABAPP.1 → AA-APP.0.2 → remaining ARCHITECTURE.md rows.

**Per-application deploy triad gate (AA-APP.0.2):** for each of `lab_application`, `legal_application`, `poc_template_application`, `research_application` assert:

1. `docker/Dockerfile` + `docker-compose.yml` + `build-docker.sh` / `.bat`
2. `BUILD_AND_DEPLOY.md` present and matches scaffold generator output (or documented drift)
3. `ARCHITECTURE.md` § **Dependencies** lists required `pyproject.toml` extras (e.g. `harness-author`, provider-specific `llm-*`, `dev-ci` for tests)

**Success gate for Phase AA platform closeout:** **Met** (2026-06-02) — conformance matrix **OK**; legal tree = scaffold; `lab_application` on `build_harness_host_runtime`; AA-APP.0.2 green; gate **533**. **Full AA register closeout** additionally requires Band 3 domain rows **Done** or explicitly **Deferred** (current policy: **Deferred**).

**Explicitly out of NOW:** K.1/K.2 implementation, Legal **live LLM** E2E (Band 3), new product hosts beyond the four listed, Legal UAEP step port (AA-LEG.2.2+) unless product reprioritizes §6.3.

### 6.2aa Phase MEM execution order (Band 2h — active)

**Status:** **Done** (2026-06-02) · **48/48 Done** · canonical register: [Phase MEM — Master deliverables register](#mem--master-deliverables-register-all-48-tasks).

Work **one MEM ID per PR**; after each step update the MEM master table + paydown log; keep §6.1 scripts green. **Start with MEM-1.*** before MEM-3/MEM-7 — bridge must exist before plugins/hooks.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEM0 | MEM-0.1–MEM-0.4, MEM-PAR.1, MEM-CHk.1, MEM-PERS.1, MEM-ST.1, MEM-OBS.2 | 9 | Register, audit baseline, parity tables (MEM-OBS.2 Done) |
| MEM1 | MEM-1.1–MEM-1.6, MEM-2.1–MEM-2.3 | 9 | **P0** — H-APP bridge + SQLite user LTM |
| MEM2 | MEM-3.*, MEM-4.*, MEM-5.*, MEM-CTX.1, MEM-DOC.1–6, MEM-GRAPH.1, MEM-TASK.* | 18 | **P1** — gates, plugins, context docs |
| MEM3 | MEM-6.*, MEM-7.*, MEM-OBS.1, MEM-DOC.5, MEM-CTX.2, MEM-PERS.2, MEM-ST.4 | 9 | **P2** — retention, hooks, metrics |
| MEM4 | MEM-8.*, MEM-9.1, MEM-PERS.3 | 4 | **P3** — product RFCs |
| **Total** | | **48** | |

**Success gate:** P0 + P1 **Done**; H-APP.4.3 **Done**; user LTM durable on sqlite lab profile; `MemoryProfile` drives all reference hosts.

**Explicitly out of NOW:** K.1/K.2, Mem0 auto-ingest ship (MEM-8.2), entity graph implementation (MEM-9.1 beyond RFC).

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages/`) | P0 |
| 2 | P-Ext.0.6 | EP discovery tests (all three groups) | P0 |
| 3 | P-Ext.1.6 | Integration EP test via fixture | P0 |
| 4 | P-Ext.1.10 | Tier-3 `integration_wiring` → `bootstrap_catalogs()` | P0 |
| 5 | P-Ext.2.9–2.11 | External tool example + unit + EP tests | P0 |
| 6 | P-Ext.3.6–3.8 | External skill example + unit + EP tests | P0 |
| 7 | P-Ext.0.7 | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | P1 |
| 8 | P-Ext.4.3, 4.5, 1.8 | Conflict policy + CI smoke (incl. integration counts) | P1 |
| 9 | P-Ext.1.5, 1.7, 5.5–5.6 | Slug/docs cleanup + author guide matrix | P2 |
| 10 | P-Ext.2.12, 3.9–3.11 | Tool/skill lazy bootstrap, scaffold plugin template, importer docs | P2 |
| 11 | P-Ext.1.3a, 1.4, 1.9, 1.11–1.12 | Typed resolve expansion, health API, integration wiring helper | P3 |
| 12 | P-Ext.5.1, 3.10, 3.12 | Scaffold CLI (all three catalogs) + harness `requires_skills` demo | P3 |

Full task register: [Appendix I](#appendix-i--plugin-catalog-traceability-phase-p-ext).

### 6.2 Harness architecture hardening (Band 2 — Phase V) — Done

**Status:** **Done** (2026-06-05) — Phase V contracts + V-REM runtime enforcement complete. Closeout: `phase_v_closeout_gate.py --enforce --enforce-l4`.

| Item | Status | Notes |
|------|--------|-------|
| V-CG … V-V6 | **Done** | Governance + CI + runtime enforcement |
| V-REM | **Done** | 10/10 closed — §6.1z queue closed |
| W-ML | **Done** | [MODALITY.md](MODALITY.md) |
| P-Ext | **Done** | Appendix I |
| M.6 / R-Skill expansion | **On demand** | W-OPS.10, W-OPS.8 |

**Forbidden in Band 2/2d:** K.1, K.2, product-specific skills, new product application hosts.

### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](#appendix-a--agent-operating-system-certification-checklist) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>/` beyond lab + reference hosts | **Deferred** | Product decision; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar/` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work.

### 6.3a Business backlog register (consolidated)

**Single register for Band 3 and AA domain-deferred rows.** Do not duplicate in harness session summaries.

| ID | Deliverable | Module | Priority | Depends on |
|----|-------------|--------|----------|------------|
| **K.1** | Problem Radar prototype (wave 2+) | `agents/problem_radar/` | Product | Explicit reprioritization |
| **K.2** | Vendor Discovery prototype | (greenfield) | Product | K.1 decision or parallel product call |
| **AA-LEG.2.2** | Legal UAEP steps (one step per PR from `SPEC_FROM_LEGACY.md`) | `agents/legal/steps/` | High | Product/legal owner |
| **AA-LEG.2.3** | Remove any parallel legal runtime (Nexus gateway only) | `agents/legal/` | High | AA-LEG.2.2 |
| **AA-LEG.2.4** | Legal agent tests per ported step | `agents/legal/tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.6** | `legal_application` host smoke on real steps | `legal_tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.8** | Consolidate duplicate legal test trees | `legal_tests/` vs agent tests | Low | AA-LEG.2.4 |
| **AA-RES.4** | Research skill ids on contracts | `agents/research/` | Medium | Product |
| **AA-RES.5** | Research UAEP + graph delegation tests | `agents/research/tests/` | High | Product |
| **AA-RESAPP.6** | Research application smoke + manifest wiring | `research_application_tests/` | High | AA-RES.5 |
| **AA-ORG.3** | Organization worker scaffold-align (`contract`, `steps/`) | `agents/organization_worker/` | Medium | Harness demo |
| **AA-ORG.4** | Lab manifest flag + integration test | `lab_application/manifest.py` | Medium | AA-ORG.3 |
| ~~AA-LABAPP.6~~ | ~~Extra lab host smoke~~ | — | — | **Done** (2026-06-02) — not in business queue |
| **K.6 / B.15 / S-Ops.4** | Legal full E2E with live LLM | CI / acceptance | Low | CI budget approval |
| **Tier-3 product** | New `applications/<product>/` beyond four reference hosts | `applications/` | Product | Phase N scaffold + §6.3 decision |
| **Domain skills** | Non-`harness.*` skill packs for product agents | `intergrax/skills/providers/` | Product | With K.1 or K.2 |
| **A.5** | Full Legal regression (all steps, live model) | Phase A row | Low | K.6 / B.15 |
| **Phase E** | Legal agent refactoring (parallel track) | `agents/legal/` | On demand | Product architecture |

**Not business (infrastructure — closed; see [§6.1z](#61z-harness-implementation-queue-consolidated)):** DX-5.7, AA-LEG.0.2, OPS-L3.1 **Done**; ongoing **§6.1** maintenance only.

### 6.1u Archived — Phase U cadence (complete 2026-06-01)

Security, policy, contracts, typing (U-Sec through U-CI). See Phase U definition of done. Residual U-Leg.* moved to §4.1 — not reopened as a new phase.

### 6.1s Archived — Phase S cadence (complete 2026-06-01)

See Phase S definition of done and Appendix F. Do not reopen S.* unless regression (fix under T.* or U.*).

### 6.1a Archived — Phase Q cadence (complete 2026-06-01)

Phase Q used **one Q.* deliverable per PR** → update Appendix C + paydown log. See Appendix C for Waves 1–9 and gate **417** at close. Do not reopen Q.* unless regression found (residual hardening → Appendix D).

### 6.1b Phase N (complete)

Tier-3 scaffold cadence remains the reference for new applications (`new-stack`); lab defaults include RAG/websearch tools and legal + research skill bundles.

### 6.4 Historical gate milestones (archived)

Phases F–L, J, Q, Q+, R, S, T, U, and §4.1 are **Done**. Gate milestones: **417** (Phase Q), **481** (harness completion, 2026-06-02). Phase tables: §2–§3; paydown: Appendices C–G.

> **Note:** Older phase closers said “next: Phase K (K.1/K.2).” That meant harness prerequisites were met, **not** that product work becomes the default implementation queue. **Current rule:** §4.0 Band 3 / §6.3 only after explicit product prioritization.

### D.2 Debug API (Done)

Standalone laboratory server:

```bash
uv run uvicorn intergrax.debug.app:create_debug_app --factory --host 127.0.0.1 --port 8099
```

Endpoints (mirror CLI):

```text
GET /debug/tasks?tenant=t1&limit=20
GET /debug/tasks/{run_id}?tenant=t1
GET /debug/tasks/{run_id}/trace?tenant=t1&include_runtime=true
```

Mount on an existing app:

```python
from intergrax.debug.router import create_debug_router

app.include_router(create_debug_router(db_path=Path("build/intergrax_trace.db")))
```

Environment: `INTERGRAX_TRACE_DB` (same as CLI).

### D.3 Experiment registry (Done)

SQLite registry at `build/intergrax_experiments.db` (`INTERGRAX_EXPERIMENTS_DB`).

```bash
python -m intergrax.debug experiments register --hypothesis "..." --capability echo.basic
python -m intergrax.debug experiments link-run EXPERIMENT_ID RUN_ID
python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
python -m intergrax.debug experiments list --decision pending
```

HTTP: `GET/POST /debug/experiments`, `POST /debug/experiments/{id}/decision`, `POST /debug/experiments/{id}/runs/{run_id}`.

### D.4 Notebook templates (Done)

Interactive §35 workflow under `notebooks/experiments/`:

| File | Purpose |
|------|---------|
| `00_experiment_template.ipynb` | Blank template — copy for new capabilities |
| `01_echo_experiment.ipynb` | Deterministic Echo smoke test |

Shared API: `intergrax.experiments.workflow.ExperimentSession`.

```python
from intergrax.experiments.workflow import ExperimentSession, ensure_repo_root_on_path
ensure_repo_root_on_path()
session = ExperimentSession(trace_db=Path("build/notebooks/trace.db"))
```

### D.5 Cost in trace (Done)

`AgentExecutionResult.cost` and `duration_seconds` are derived from LLM usage (`intergrax/contracts/runtime_cost.py`):

- Mapping: `runtime_answer_to_agent_result()` reads `llm_usage_report` or `stats.extra.cost`
- NexusLoop: aggregates multi-agent cost into task metadata (`execution_cost`) and `RunStats.llm_usage` on finalize
- Debug API/CLI: `stats.cost` on run detail; CLI `tasks show` prints cost line

Cost proxy: **1 cost unit = 1 LLM token** (laboratory default, matches EvalRunner).

### F.1 Shadow workspace (Done)

Isolated temporary filesystem for experiments (§20). Enable on a Nexus task:

```python
task = Task(
    tenant_id="t1",
    user_id="u1",
    message="analyze vendor",
    context=TaskContext(capability="research.web_search"),
    metadata={"shadow_workspace": True},  # optional: "shadow_workspace_cleanup": True
)
```

UAEP agents receive `ctx.metadata["shadow_workspace"]` in `run_step`. Result metadata includes `shadow_workspace_id`.

Root directory: `INTERGRAX_SHADOW_ROOT` (default `build/shadow_workspaces/`).

### F.2 Sandbox runtime (Done)

Controlled session for risky tool use (§21). Enable on a Nexus task:

```python
task = Task(..., metadata={"sandbox": True})
```

Agents invoke allowlisted operations through the tool gateway:

```python
await ctx.invoke_tool(ToolRequest(
    tool_name="sandbox.exec",
    agent_id=ctx.agent_id,
    input={"operation": "write_file", "payload": {"path": "out.txt", "content": "..."}},
))
```

Operations: `echo`, `write_file`, `read_file`, `list_files`. Root: `INTERGRAX_SANDBOX_ROOT` (default `build/sandbox_sessions/`).

### F.3 Advanced HITL (Done)

Human responses beyond approve:

```python
# Re-submit paused task with verdict
task = Task(..., task_id=original_task_id, metadata={"human_response": "reject"})
# or "approve" / "escalate"
```

- **reject** → task `FAILED`, decision persisted
- **escalate** → `INTERRUPT_ESCALATED` event, escalation chain in metadata, stays `WAITING_FOR_HUMAN`
- Store: `INTERGRAX_HUMAN_DECISIONS_DB` (default `build/intergrax_human_decisions.db`)

Optional on `NexusLoop`: `human_decision_store=SQLiteHumanDecisionStore(...)`.

### F.4 Long-running tasks (Done)

Enable durable pause/resume on Nexus tasks (§26):

```python
from intergrax.runtime.task import Task, TaskExecutionOptions, TaskLongRunningOptions

task = Task(
    tenant_id="t1",
    user_id="u1",
    message="monitor vendors for 30 days",
    context=TaskContext(capability="hitl.basic"),
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(
            enabled=True,
            notify_channel="slack",  # or "teams" / "log"
        ),
    ),
)
```

On pause (`WAITING_FOR_HUMAN`), NexusLoop persists a checkpoint with `resume_token` in result metadata.

Resume with the same `task_id` and token:

```python
Task(
    ...,
    task_id=original_task_id,
    options=TaskExecutionOptions(
        long_running=TaskLongRunningOptions(enabled=True, resume_token=token),
    ),
    metadata={"human_approved": True, "resume_token": token},
)
```

Optional on `NexusLoop`: `checkpoint_store=SQLiteTaskCheckpointStore(...)`, `notification_adapter=LoggingNotificationAdapter()`.

Env:

- `INTERGRAX_TASK_CHECKPOINTS_DB` (default `build/intergrax_task_checkpoints.db`)
- `INTERGRAX_RUNTIME_EVENTS_DB` (optional; enables SQLite runtime events in NexusLoop / debug API)
- `INTERGRAX_TASK_MEMORY_DB` (optional; TaskMemory SQLite path for lab / debug)
- `INTERGRAX_SLACK_WEBHOOK_URL` / `INTERGRAX_TEAMS_WEBHOOK_URL` (stub adapters; no network unless configured)

### H.6 Organization Worker lab runbook (Done)

Reference flow for §38 — virtual worker via Slack / Teams without orchestration in adapters.

**Agent:** `agents/organization_worker/` — capability `org.vendor_report`.

**Lab app factory:**

```python
from intergrax.lab import create_organization_worker_lab_app

app = create_organization_worker_lab_app()  # pre-wired registry + HITL intake enricher
```

**HTTP (debug API):**

```bash
uv run uvicorn intergrax.lab.organization_worker:create_organization_worker_lab_app --factory --host 127.0.0.1 --port 8099
```

1. **Intake + execute** (Slack-shaped slash command):

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/interactions/intake?execute=true&tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/intergrax","text":"org.vendor_report Acme Corp Q1","user_id":"U1","team_id":"T1"}'
```

Response includes `state: waiting_for_human`, `resume_token`, HITL notification on configured channel (`slack` / `teams` / `log`).

2. **Resume after approval:**

```bash
curl -s -X POST "http://127.0.0.1:8099/debug/tasks/{task_id}/human-response?tenant=T1" \
  -H "Content-Type: application/json" \
  -d '{"response":"approve","resume_token":"<token from intake>"}'
```

Teams intake uses the same endpoints with Bot Framework activity JSON (`channelId: msteams`).

**Registry helper:** `build_organization_worker_registry()` in `intergrax.runtime.registry`.

**Tests:** `tests/integration/debug/test_organization_worker_demo.py` (gate).

### D.1 Debug CLI (Done)



```bash

python -m intergrax.debug tasks list --tenant t1 --limit 20

python -m intergrax.debug tasks show RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1

python -m intergrax.debug tasks trace RUN_ID --tenant t1 --format json --runtime

python -m intergrax.debug --db path/to/trace.db tasks list

```



Reuse:



- `SQLiteRunTraceStore` / `RunTraceReader` — `intergrax/runtime/nexus/tracing/`

- `trace_bridge` — `intergrax/runtime/events/trace_bridge.py`

- `NexusLoop.event_bus` — in-process runs (not persisted; CLI uses SQLite trace)



---

## Appendix A — Business agents readiness checklist

Gate before Problem Radar / Vendor Discovery. Run:

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest tests/ -m gate -q
```

### Agent creation & registration

| # | Question | Status |
|---|----------|--------|
| 1 | Scaffold in minutes (`intergrax.scaffold new-agent`)? | ✅ |
| 2 | UAEP structure generated (contract, steps, tests)? | ✅ |
| 3 | First run in < 1 hour? | ✅ |
| 4 | Register via `AgentRegistry` only (no Nexus edits)? | ✅ |
| 5 | Capabilities in contract? | ✅ |

### Execution & observability

| # | Question | Status |
|---|----------|--------|
| 6 | Runs through NexusLoop / lab `/v1/lab/run`? | ✅ |
| 7 | UnifiedTaskRunner same path as HTTP? | ✅ |
| 8 | Graph sequential + parallel? | ✅ |
| 9 | Trace via `/debug/tasks/{id}`? | ✅ |
| 10 | Runtime events + checkpoints + progress? | ✅ |

### Recovery, HITL, memory, isolation

| # | Question | Status |
|---|----------|--------|
| 11 | Nexus validates output? | ✅ |
| 12 | Retry / alternate agent on validation failure? | ✅ |
| 13 | HITL pause + resume? | ✅ |
| 14 | Checkpoint recovery? | ✅ |
| 15 | Shared context in graphs? | ✅ |
| 16 | Sandbox + shadow workspace? | ✅ |

### Tooling & composition

| # | Question | Status |
|---|----------|--------|
| 17 | Canonical agent guide exists? | ✅ |
| 18 | Lab application (Tier-3)? | ✅ |
| 19 | Same agent reusable across applications? | ✅ |
| 20 | Applications contain wiring only? | ✅ |

### Go / no-go

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Checklist | ≥ 90% | **20/20** |
| Acceptance suite | 10/10 green | ✅ |
| Sign-off exercise | 1 new agent, < 1h, zero runtime edits | **Done** (`signoff_probe`) |

**Verdict:** **L1 Agent Operating System certified** (technical). **Phase S** (harness environment GA) is next; **K.1/K.2** wait until S is **Done**.

### Sign-off record

```text
Date:           2026-05-27
Agent exercise: signoff_probe
Capability:     signoff.probe
Time to first run: ~15 min (scaffold + smoke test)
Runtime files modified: none (only agents/signoff_probe/ added)
Smoke test:     agents/signoff_probe/tests — 1 passed
HTTP proof:     lab_application wiring + POST /v1/lab/run
Trace proof:    GET /debug/tasks/{id}, /trace?include_runtime=true, /events
                (test_lab_application_runs_signoff_probe_with_trace)
Acceptance suite: pass (tests/acceptance/agent_os)
Gate suite:     pass (228+ tests)
Trace:          NexusLoop smoke + HTTP debug API (SQLite trace store in lab factory)
Decision:       L1 certified — GO Phase S (harness environment), then Phase K (K.1/K.2)
```

---

## Appendix B — Technical debt backlog

**Purpose:** consolidated backlog for review and **incremental paydown**.  
**Source:** canon §2 map, §0.5 maturity, Phase G–K gaps, lab sign-off findings (2026-05-27).  
**How to use:** pick items by priority; apply §0.6 (Tier-1 only when reusable across agents).  
**Status:** `Open` | `Done` | `Deferred`

### B.0 Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-05-29 | M.6-gcp | `providers/gcp/` — cloud_platform facade; ADC/service account + category slug defaults |
| 2026-05-29 | M.6-azure | `providers/azure/` — cloud_platform facade; token health + category slug defaults |
| 2026-05-29 | M.6-aws | `providers/aws/` — cloud_platform facade; STS health + category slug defaults |
| 2026-05-29 | M.6-cassandra | `providers/cassandra/` + `contracts/document_store.py`; CQL partition-scoped CRUD |
| 2026-05-29 | M.6-ms365_graph | `providers/ms365_graph/` + `contracts/collaboration_suite.py`; Graph mail/calendar/directory |
| 2026-05-30 | M.6-prometheus | `providers/prometheus/` + `contracts/observability_backend.py`; PromQL query API |
| 2026-05-30 | M.6-confluence | `providers/confluence/` + `contracts/wiki_knowledge.py`; REST wiki; single-entry `opens.py` |
| 2026-05-30 | M.6-jira | `providers/jira/` + `contracts/issue_tracker.py`; REST v3; single-entry `opens.py` |
| 2026-05-30 | M.6-mysql | `providers/mysql/` — beta `RelationalStore` (pymysql); single-entry `opens.py` |
| 2026-05-30 | M.6-provider-layout | Providers grouped under `providers/<category>/<slug>/`; `layout.py` slug map; tests mirrored by category |
| 2026-05-30 | M.6-p2-batch | P2/P3 integrations — 22 slugs (`azure_blob`, `gcs`, `dynamodb`, cloud queues, SQL variants, SMTP, OTEL, GitHub/Linear/Azure DevOps, Notion/SharePoint, Google Workspace, Brave/SerpAPI, Playwright); `_shared/p2/`; **324** integration unit tests |
| 2026-05-30 | M.7-agent-guide-integrations | `AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
| 2026-05-30 | N.4-product-scaffold | `--profile product` → FastAPI Core host, `agent_factories.py`, auth stub env; `new_application_product.py` |
| 2026-05-30 | N.5-docker-build-scripts | `build-docker.sh` / `build-docker.bat` in scaffold + lab/legal/research/poc; `docker_templates.py` |
| 2026-05-30 | N.0-docs | Canon §7.4.8–§7.4.10 + Phase N plan (application environment, manifest, scaffold steps) |
| 2026-05-30 | M.8-lab-profile | `wire_lab_integrations()` + `providers/log/` — lab uses `IntegrationProfile.lab()` |
| 2026-05-30 | M.4-kafka-rabbitmq-adopt | Queueing bootstrap + integration tests use `integrations/providers/{kafka,rabbitmq}/` only |
| 2026-05-30 | M.4-rabbitmq | `providers/rabbitmq/` + runtime `build_rabbitmq_transport()` delegate |
| 2026-05-29 | M.4-lab_json | `providers/lab_json/` + runtime `create_interaction_adapter(LAB)` delegate — **M.4 P0 complete** |
| 2026-05-29 | M.4-webhook | `providers/webhook/` + runtime `create_notification_adapter(WEBHOOK)` delegate |
| 2026-05-29 | M.4-teams-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/teams/` |
| 2026-05-29 | M.4-teams | `providers/teams/` — dual category catalog entry |
| 2026-05-29 | M.4-slack-adopt | Runtime notifications/interactions/verifier + long_running delegate to `providers/slack/` |
| 2026-05-29 | M.4-slack | `providers/slack/` — dual category + resolve dispatches by category |
| 2026-05-29 | M.4-bing | `providers/bing/` — SearchProvider adapter over legacy Bing v7 |
| 2026-05-29 | M.4-google_cse | `providers/google_cse/` — SearchProvider adapter over legacy CSE |
| 2026-05-29 | M.4-celery | `providers/celery/` — message bus + worker helpers; no `kv_store` |
| 2026-05-29 | M.4-kafka | `providers/kafka/` + transport delegate; requires `kv_store` |
| 2026-05-29 | M.4-sqlite-adopt | Runtime `open_*` + apps delegate to `integrations/providers/relational_store/sqlite/` |
| 2026-05-29 | M.4-sqlite | `providers/sqlite/` + bundle (10 domain stores); lazy bootstrap + package `__init__` |
| 2026-05-29 | M.4-redis | Complete bundle: `create_redis_integration()` — KV, idempotency, rate limit, semaphore, rerank |
| 2026-05-27 | B.08, B.10 | `wire_nexus_observability` + SQLite defaults in Legal / Research / Lab factories; integration test |
| 2026-05-27 | B.01, B.02 | `RuntimeCheckpoint` full snapshot + UAEP mid-step cursor/resume; acceptance `05b` |
| 2026-05-27 | B.12, B.14 | Production `POST /v1/interactions/intake` on lab; Legal legacy `AgentEngine` removed |
| 2026-05-27 | B.05 | Escalation notification template + scheduler wiring in lab + SAFETY_VIOLATION timeout→escalate |
| 2026-05-27 | B.09, B.17 | Injectable `trace_store` on debug API; gate uses `pytest -m gate` (`testpaths` includes `agents/`) |
| 2026-05-27 | Platform stabilization | All Tier-3 hosts: validating runtime events, plugin bootstrap, resilient delivery (lab/legal/research/poc); shared `_shared/platform_wiring` + `notification_wiring` |
| 2026-05-27 | Infra paydown | SQLite DLQ ledger + debug `/notifications/*`; `ValidatingRuntimeEventPersistence`; Tier-3 plugin bootstrap |
| 2026-05-27 | B.07, B.11, B.13, B.18, B.24 | Schema registry + phase coverage + `RuntimePlugin`; metrics export + `GET /debug/tasks/{id}/metrics`; retry/DLQ delivery; echo + research_mock HTTP trace acceptance; agents vendor import gate test |
| 2026-05-27 | K.3–K.5 | `coerce_replay_policy_engine` + `ExecutionGuard.evaluate_replay`; ChatAgent production import guard; CI gate paths aligned with full gate (**394** tests) |
| 2026-05-27 | B.06, §18 | `BEFORE/AFTER_TOOL_CALL` + agent-selection hooks; product interaction intake on legal/research (**397** gate) |

### B.1 Runtime & §42 convergence

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.01 | **UAEP mid-step checkpoint** — resume inside a long-running step (not only between steps / HITL) | §42.9.3, §26 | **High** | **Done** | Long-running domain agents (Legal, Research) | Tier-1 | `uaep_step_cursor`, `should_resume_uaep_step`, optional `resume_step` (2026-05-27) |
| B.02 | **Full checkpoint snapshot** — plan + graph node states + UAEP index + pending decisions in one durable blob | §42.9.2 | **High** | **Done** | Multi-agent graphs, crash recovery | Tier-1 | `plan_snapshot`, `graph_snapshot`, `pending_decisions` in `RuntimeCheckpoint` (2026-05-27) |
| B.03 | **Policy engine facade** — single `PolicyEngine` for replay, validation, runtime policy | §42.11 | **Medium** | **Done** | Indirect — consistent governance for all agents | Tier-1 | `PolicyEngine` + `coerce_policy_engine`; Nexus/UAEP/interrupt handler (2026-05-27) |
| B.04 | **Dual `AgentDecision` cleanup** — converge tools-agent variant with canonical §42.7 enum | §42.7 | **Medium** | **Done** | Agents emitting decisions must use one contract | Tier-1 | `ToolPlanDecision` in `tools.core.tool_plan_decision`; no `tools_agent` re-export (2026-06-02) |
| B.05 | **Escalation policy production path** — `SAFETY_VIOLATION` / HITL expiry → real escalation (not stub) | §42.38, §42.10 | **Medium** | **Done** | HITL-heavy agents | Tier-1 | `escalation.v1` template, `wire_long_running_scheduler`, lab startup, SAFETY_VIOLATION timeout→escalate (2026-05-27) |
| B.06 | **Hook / middleware parity** — full §42.20 pipeline vs current Nexus-embedded hooks | §42.20, §42.22 | **Low** | **Done** | Extension agents via plugins | Tier-1 | Lifecycle + **tool call** + **agent selection** hooks; decision/interrupt/retry hooks remain optional (2026-05-27) |
| B.07 | **§42 maturity remainder** — schema versioning (§42.29), full `ExecutionPhase` coverage, plugin contracts | §42 | **Medium** | **Done** (baseline) | Platform stability for new agents | Tier-1 | `runtime/schema/registry.py`, `events/phase_coverage.py`, `plugins/contract.py` (2026-05-27) |

### B.2 Observability & debug surface

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.08 | **Application trace store split** — factories used `InMemoryRunTraceStore` while debug API reads SQLite | §33, §42.24 | **High** | **Done** | HTTP `/debug/tasks/*` 503 in product apps | Tier-3 | `wire_nexus_observability` + `open_run_trace_store` (2026-05-27) |
| B.09 | **Debug API trace reader** — only SQLite file path; no injectable in-memory / shared store handle | §19 | **Medium** | **Done** | Lab tests, local dev without file I/O | Tier-1 | `trace_store` on `create_debug_router` / `create_debug_app`; lab passes Nexus store (2026-05-27) |
| B.10 | **NexusLoop runtime events in app factories** — all Tier-3 factories pass runtime events to Nexus | §42.24 | **Medium** | **Done** | Events 503 on `/debug/tasks/{id}/events` | Tier-3 | Legal + Research default SQLite; lab when path passed (2026-05-27) |
| B.11 | **Metrics layer** — event-first, trace-second, **metrics-third** unified export | §42.1, §33 | **Low** | **Done** | Ops visibility, SLOs | Tier-0 | `runtime/metrics/export.py` + `GET /debug/tasks/{run_id}/metrics` (2026-05-27) |

### B.3 Interaction surfaces (§18)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.12 | **Production Slack / Teams webhooks** — inbound intake on product hosts | §18 | **Medium** | **Done** | Organization Worker, HITL from chat | Tier-0 / Tier-3 | `POST /v1/interactions/intake` on lab/legal/research/poc via `wire_interaction_intake_service` (2026-05-27) |
| B.13 | **Outbound delivery hardening** — retries, DLQ, delivery receipts for HITL notifications | §18, §42.10 | **Low** | **Done** | HITL agents in prod | Tier-0 | `RetryingNotificationDelivery` + `SQLiteDeliveryLedger` + debug `/debug/notifications/*` (2026-05-27) |

### B.6 Integration Library (§7.1)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.18 | **Integration catalog package** — `intergrax/integrations/` scaffold | §7.1.1 | **High** | **Done** | All agents needing external systems | Tier-0 | M.1–M.3 + M.5 (2026-05-29) |
| B.19 | **P0 provider wraps** — M.4 catalog slugs | §7.1.3 | **High** | **Done** | Lab + first prod apps | Tier-0 | All P0 slugs wrapped + runtime adoption (2026-05-29) |
| B.20 | **PostgreSQL relational_store** — production DB adapter | §7.1.3 | **Medium** | **Done** (beta) | Multi-tenant applications | Tier-0 | `providers/postgresql/` — domain stores SQLite-first |
| B.21 | **Jira + Confluence providers** — issue/wiki ingestion | §7.1.3 | **Medium** | **Done** (beta) | PM / research agents | Tier-0 | Integrations + catalog tools (Phase O.4, 2026-05-30) |
| B.22 | **MS365 Graph provider** — mail, calendar | §7.1.3 | **Medium** | **Done** (beta) | Org worker, scheduling agents | Tier-0 | `providers/ms365_graph/`; client credentials via `opens.py` |
| B.23 | **Prometheus observability_backend** — PromQL query API | §33, §7.1.3 | **Low** | **Done** (beta) | Ops / SLO | Tier-0 | `providers/prometheus/`; complements B.11 metrics layer design |
| B.28 | **Cassandra document_store** — wide-column adapter for high-volume retention | §7.1.3 P2 | **Medium** | **Done** (beta) | Runtime event archive at scale; ops telemetry | Tier-0 | `providers/cassandra/`; single-entry `opens.py` |
| B.29 | **Elasticsearch observability_backend** — log search / aggregations | §7.1.3 P2 | **Medium** | **Done** (beta) | Ops log triage; optional RAG over logs | Tier-0 | `providers/elasticsearch/`; single-entry `opens.py`; complements B.23 |
| B.30 | **Databricks relational_store** — SQL Warehouse / Unity Catalog SQL | §7.1.3 P2 | **Medium** | **Done** (beta) | Analytics agents, lakehouse reporting | Tier-0 | `providers/databricks/`; single-entry `opens.py`; PAT |
| B.31 | **MongoDB document_store** — flexible JSON persistence | §7.1.3 P2 | **Medium** | **Done** (beta) | Agent memory, unstructured artifacts | Tier-0 | `providers/mongodb/`; PyMongo only in `opens.py`; reuses `DocumentStore` |
| B.32 | **Pinecone vector_store bridge** — catalog entry → `rag/` | §7.1.3 P2 | **Medium** | **Done** (beta) | Production RAG agents | Tier-0 | `providers/pinecone/` thin adapter; SDK only in `opens.py` |
| B.33 | **Qdrant + Chroma vector_store bridges** — same pattern as B.32 | §7.1.3 P2 | **Low** | **Done** (beta) | Self-hosted / dev RAG | Tier-0 | `providers/qdrant/`, `providers/chroma/`; RAG bootstrap via catalog |
| B.34 | **Object storage contract + S3 provider** — blobs for artifacts / sandboxes | §7.1.3 P2 | **Medium** | **Done** (beta) | Large file handoff, exports | Tier-0 | `contracts/object_storage.py`, `providers/s3/`; boto3 only in `opens.py` |
| B.35 | **Notion + SharePoint wiki_knowledge** — internal docs ingestion | §7.1.3 P3 | **Low** | **Done** (beta) | Research / runbook agents | Tier-0 | REST adapters; `_shared/p2/factories.py` |
| B.36 | **GitHub + Linear issue_tracker** — dev workflow sources | §7.1.3 P3 | **Low** | **Done** (beta) | Code-aware agents | Tier-0 | REST; thin provider shells |
| B.37 | **email_smtp notification_channel** — outbound mail without chat | §7.1.3 P3 | **Low** | **Done** (beta) | HITL, scheduled reports | Tier-0 | stdlib SMTP in factory open path |
| B.38 | **OpenTelemetry observability_backend** — trace/metric export | §33, §7.1.3 P3 | **Low** | **Done** (beta) | Unified ops dashboards | Tier-0 | `providers/otel/`; beta noop exporter default |
| B.39 | **Playwright browser_automation** — dynamic web interaction | §7.1.3 P3 | **Low** | **Done** (beta) | Research on JS-heavy sites | Tier-0 | `providers/playwright/`; browser launch in factory |
| B.25 | **AWS cloud_platform facade** — auth + S3/SQS/DynamoDB/ElastiCache defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | AWS-hosted applications | Tier-0 | `providers/aws/`; infrastructure only |
| B.26 | **Azure cloud_platform facade** — MI + Blob/Service Bus/Azure SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | Azure-hosted applications | Tier-0 | `providers/azure/`; infrastructure only |
| B.27 | **GCP cloud_platform facade** — ADC + GCS/Pub/Sub/Cloud SQL defaults | §7.1.3 P1.1 | **Medium** | **Done** (beta) | GCP-hosted applications | Tier-0 | `providers/gcp/`; infrastructure only |
| B.24 | **Direct vendor SDK in agents** — audit + lint rule | §5.2, §7.1.4 | **Medium** | **Done** | Prevents catalog bypass | Tier-2 | `scripts/check_agents_vendor_imports.py` + gate test `test_vendor_import_guard_b24` (2026-05-27) |

### B.7 Tool Library (§7.1.6)

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.40 | **Tool Library scaffold** — catalog, profile, wiring context | §7.1.6 | **High** | **Done** | All agents using external capabilities | Tier-0 | Phase O.2; apps wire tools O.8 (2026-05-30) |
| B.41 | **Context tools** — `rag.retrieve`, `websearch.query` | §7.1.7, §22.1 | **High** | **Done** | RAG / research agents | Tier-0 | Phase O.3 (2026-05-30) |
| B.42 | **Jira catalog tools** — `jira.get_issue`, `jira.search_tasks`, … | §7.1.6 | **Medium** | **Done** | PM / legal workflow agents | Tier-0 | Phase O.4 (2026-05-30) |
| B.43 | **Unified tool model** — deprecate `use_rag` / `use_websearch` flags | §7.1.7, §22.2 | **High** | **Done** | Consistent tool policy + MCP | Tier-1 | Phase O.5 (2026-05-30) |
| B.44 | **Legacy ToolBase migration** | §5.2.2 | **Medium** | **Done** | Single registry | Tier-0 | Phase O.7; `tools_base` deprecated |
| B.45 | **MCP tool export from catalog** | §7.1.6 | **Low** | **Done** | External MCP clients | Tier-3 | Phase O.6 |

### B.4 Legacy & composition

| ID | Item | Canon | Priority | Status | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------|--------------|------|----------------|
| B.14 | **`ChatAgent` / legacy engine removal** — `LEGAL_USE_LEGACY_AGENT_ENGINE` removed | §39, §41 | **Medium** | **Done** | Single execution path for all agents | Tier-1 / Tier-3 | Legal `fastapi_router` requires `UnifiedTaskRunner`; legacy flags removed (2026-05-27) |
| B.15 | **Legal full E2E gate (real LLM)** — deferred acceptance with live model | — | **Low** | **Deferred** | Legal quality assurance | Tier-2 / CI | K.6; separate from Agent OS gate; enable when CI budget approved |
| B.16 | **Lab agent auto-discovery** — manifest-driven roster + scaffold | §7.4 | **Low** | **Done** | Onboarding friction | Tier-3 | Phase N: `ApplicationManifest`, `new-stack` (N.10); explicit `AgentBinding` remains by design (2026-05-30) |
| B.28 | **Per-application `.env.example` missing** — only root `.env.example`; lab/legal vars in README only | §7.4.8 | **Medium** | **Done** | Deployable POC friction | Tier-3 | N.7 backfill + scaffold (2026-05-30) |
| B.29 | **`new-application` scaffold (lab)** — Tier-3 hosts hand-copied from legal/lab | §7.4.8 | **High** | **Done** | Lab + product profiles via CLI; gate acceptance | Tier-3 / platform | N.10 `new-stack` optional |
| B.30 | **No application-level Dockerfile** — only `infra/docker/docling/` | §7.4.8 | **Medium** | **Done** | Per-app `docker/` + build scripts on lab/legal/research/poc | Tier-3 | N.5–N.7 (2026-05-30) |

### B.5 Test & certification hygiene

| ID | Item | Canon | Priority | Agent impact | Tier | Recommendation |
|----|------|-------|----------|--------------|------|----------------|
| B.17 | **`agents/` gate collection** — `signoff_probe` test marks `gate` but lives under `agents/` (may not be collected by default `pytest tests/`) | — | **Low** | **Done** | Sign-off smoke not in main gate count | Test infra | `testpaths` includes `agents/`; canonical gate: `uv run pytest -m gate -q` (2026-05-27) |
| B.18 | **HTTP observability acceptance** — trace on echo + multi-agent mock (graph path) | Appendix A #9–10 | **Low** | **Done** | Certification confidence | Test | `test_lab_application_runs_echo_with_trace_observability`, `test_lab_application_runs_research_mock_with_graph_trace` (2026-05-27) |

### B.8 Suggested priority order (for planning)

```text
1. ~~B.08, B.10~~ — observability consistency (Done 2026-05-27)
2. ~~B.01, B.02~~ — checkpoint / full snapshot (Done 2026-05-27)
3. ~~B.03, B.04~~ — governance facade + AgentDecision cleanup (Done 2026-05-27)
4. ~~B.12, B.14~~ — product interaction + legacy removal (Done 2026-05-27)
5. ~~B.05~~ — escalation production path (Done 2026-05-27)
6. ~~B.09, B.17~~ — debug trace injection + gate collection (Done 2026-05-27)
7. ~~B.06~~ — hook parity doc + lifecycle wiring (Done 2026-05-27)
8. ~~B.07, B.11, B.13, B.18, B.24~~ — §42 baseline, metrics export, delivery hardening, HTTP trace acceptance, vendor import guard (Done 2026-05-27)
9. ~~Platform stabilization~~ — all Tier-3 factories aligned (Done 2026-05-27)
10. B.15 — Legal E2E real LLM (**Deferred** — product/CI decision)
11. ~~Phase Q~~ — Harness audit remediation — **Done** (Appendix C)
12. ~~Phase Q+ / Phase R~~ — **Done** (Appendices D, E)
13. ~~Phase S — Harness environment GA~~ — **Done**
14. ~~Phase T — Harness cleanliness~~ — **Done**
15. Phase U — Harness production hardening — **Done**
16. Harness completion backlog (§4.1) — **Done** (2026-06-02)
17. Phase K — K.1/K.2 business agents — **Deferred**
18. Tier-3 product apps / Legal E2E — **Deferred**
```

**Note:** Platform harness (Q–U) is complete. **Harness completion** (legacy + CI) is active. Business agents and product applications are **end of list**.

---

## Appendix C — Harness audit traceability (Phase Q)

**Purpose:** Every finding from the harness implementation audit (2026-06-01) maps to exactly one Phase Q deliverable. Update **Status** when the deliverable is **Done** / **Won't fix** (with reason).

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### C.1 Nexus, loops, orchestration, errors

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| N-01 | `NexusLoop` monolith ~1200 lines | Q-N.1 | Done (`orchestration/`; ~586 lines) |
| N-02 | Duplicate `_normalize_human_response` | Q-N.2 | Done |
| N-03 | Dual retry (`RetryEngine` vs `max_run_retries`) | Q-N.3 | Done |
| N-04 | `PolicyEngine` \| `RuntimePolicyEngine` union | Q-N.4 | Done |
| N-05 | Hooks NOT_WIRED: decision, interrupt, retry | Q-N.5 | Done |
| N-06 | Hooks PARTIAL: trace persist | Q-N.6 | Done |
| N-07 | `runtime_steps/tools.py` misleading name | Q-N.7 | Done |
| N-08 | `RuntimeConfig` monolith | Q-N.8 | Done |
| N-09 | `integration_profile: object` | Q-N.9 | Done |
| N-10 | `production_mode` default in lab | Q-N.10 | Done |
| N-11 | Graph callbacks typed `object` | Q-N.11 | Done |
| N-12 | Duplicate import `InterruptType` | Q-N.12 | Done |
| N-13 | `AgentEngine` static UAEP / event_bus | Q-N.13 | Done |
| N-14 | No unit tests `nexus_loop.py` | Q-N.14 | Done |
| N-15 | Thin `GraphExecutor` unit coverage | Q-N.15 | Done |

### C.2 LLM adapters

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| L-01 | Dead `tracked_llm_call` | Q-L.1 | Done |
| L-02 | Empty `llm_adapters/__init__.py` | Q-L.2 | Done |
| L-03 | `LLM_ADAPTERS.md` missing provider table | Q-L.3 | Done |
| L-04 | `LLMProfile` docstring `max_retries` wrong | Q-L.4 | Done |
| L-05 | `supports_streaming()` default True | Q-L.5 | Done |
| L-06 | PolicyEngine ignores `llm_cost_evaluation` | Q-L.6 | Done |
| L-07 | Dual usage tracking naming | Q-L.7 | Done |
| L-08 | No structured-output conformance | Q-L.8 | Done |
| L-09 | Bedrock context_window TODO | Q-L.9 | Done |
| L-10 | OpenAI-compat `__dict__.update` fragility | Q-L.10 | Done |
| L-11 | Env vars scattered | Q-L.11 | Done |

### C.3 RAG

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| R-01 | Dead `_build_backend_where` / `_map_hits_to_chunks` | Q-R.1 | Done |
| R-02 | Four parallel retrieval paths | Q-R.2 | Done |
| R-03 | `enable_rag` vs `use_rag` in ContextBuilder | Q-R.3 | Done |
| R-04 | `NoPlannerPipeline` always `RagStep` | Q-R.4 | Done |
| R-05 | `top_k` collapses prefetch | Q-R.5 | Done |
| R-06 | `RuntimeConfig` vs `RagProfile` dual config | Q-R.6 | Done |
| R-07 | Unused `RagProfile.extras` | Q-R.7 | Done |
| R-08 | RAG metrics env not in profile | Q-R.8 | Done |
| R-09 | `rag/answers/` parallel stack | Q-R.9 | Done |
| R-10 | `UserProfileManager` bypasses `RetrievalService` | Q-R.10 | Done |
| R-11 | Three “context builder” names | Q-R.11 | Done |
| R-12 | Legacy `use_rag` plan booleans | Q-R.12 | Done |

### C.4 Memory

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| M-01 | No single memory architecture doc | Q-M.1 | Done |
| M-02 | Task memory not visible in scaffold | Q-M.2 | Done |
| M-03 | Silent default when task memory None | Q-M.3 | Done |

### C.5 Observability & metrics

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| O-01 | RAG plugin not in `platform_wiring` | Q-O.1 | Done |
| O-02 | No RAG bridge tests | Q-O.2 | Done |
| O-03 | Parser trace bypasses `ObservabilityBackend` | Q-O.3 | Done |
| O-04 | `metrics/export` substring heuristics | Q-O.4 | Done |
| O-05 | Duplicate import in `metrics/export.py` | Q-O.5 | Done |
| O-06 | `behavioral` never set in export | Q-O.6 | Done |
| O-07 | `/metrics/llm` not on lab host | Q-O.7 | Done |
| O-08 | Observability env scattered | Q-O.8 | Done |
| O-09 | RAG metrics asymmetry vs LLM | Q-O.9 | Done |
| O-10 | `trace_bridge` vs `phase_coverage` drift | Q-O.10 | Done |
| O-11 | Debug router missing type imports | Q-O.11 | Done |
| O-12 | No `trace_bridge` unit tests | Q-O.12 | Done |
| O-13 | Two Prometheus concepts unclear | Q-O.13 | Done |
| O-14 | Runtime events SQLite-first; Cassandra adoption undefined | Q-O.14 | Done |

### C.6 Legacy, style, docs

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| X-01 | Deprecated `ChatAgent` | Q-X.1 | Done |
| X-02 | `task_metadata_bridge` legacy | Q-X.2 | Done |
| X-03 | Copyright / Integrax typo | Q-X.3 | Done |
| X-04 | `tools_base` deprecation | Q-X.4 | Done |
| X-05 | M.6 Future slugs table stale | Q-X.5 | Done |
| D-01 | `docs/README` focus outdated | Q-D.1 | Done |
| D-02 | Canon §52 still “Active” | Q-D.2 | Done |
| D-03 | §0.1 “blocked until L” stale | Q-D.1 (§0.1 fix) | Done |
| D-04 | Guide missing memory/RAG naming | Q-D.4 | Done |
| D-05 | §5.2 process gates not listed for agent authors | Q-D.5 | Done |

### C.7 Tests (cross-cutting)

| Audit ID | Finding | Q ID | Status |
|----------|---------|------|--------|
| T-01 | NexusLoop unit suite | Q-T.1 / Q-N.14 | Done |
| T-02 | `rag_profile_from_env` tests | Q-T.2 | Done |
| T-03 | `ContextBuilder` tests | Q-T.3 | Done |
| T-04 | `UserProfileManager` tests | Q-T.4 | Done |
| T-05 | Single retrieval per turn test | Q-T.5 | Done |
| T-06 | Platform wiring observability E2E | Q-T.6 | Done |

### C.8 Phase Q paydown log

| Date | Q ID | Summary |
|------|------|---------|
| 2026-06-01 | Q-D.3 | §0.1 strategic objective — Harness GA vs Phase K vs Phase Q |
| 2026-06-01 | Q-O.1,Q-O.2,Q-O.5,Q-O.7 | RAG plugin bootstrap, tests, metrics lint, lab `/metrics/llm` |
| 2026-06-01 | Q-N.2,Q-N.7,Q-N.12 | Duplicate HITL normalize; tool_context_helpers; interrupt import |
| 2026-06-01 | Q-R.1–Q-R.5,Q-R.8 | RAG dead code, single retrieval path, use_rag metadata, prefetch_k |
| 2026-06-01 | Q-L.1,Q-L.2,Q-L.4 | Remove tracked_llm_call; llm_adapters exports; LLMProfile docstring |
| 2026-06-01 | Q-T.2,Q-T.3,Q-T.6 | New unit/integration tests; gate **399 passed** (+2) |
| 2026-06-01 | Q-N.1(partial),Q-N.10,Q-N.13,Q-N.15 | `hitl_runner.py`; lab `harness_production_mode`; AgentEngine `event_bus`; graph checkpoint tests |
| 2026-06-01 | Q-L.9–Q-L.11,Q-O.6,Q-O.11,Q-O.14 | Bedrock windows, OpenAI-compat delegation, LLM env appendix, metrics behavioral, debug types, trace storage §33.1 |
| 2026-06-01 | docs-consolidation | Merged LLM/RAG observability, retry, trace ADR into canon + `LLM_ADAPTERS.md`; removed satellite `docs/*.md` |
| 2026-06-01 | Q-N.1,Q-X.2,Wave 9 | `graph_runner`, `task_events`, `lifecycle_bridge`; UAEP `execution_options_for_request`; gate **417 passed** |
| 2026-06-01 | Q-X.2(partial),Q-X.4,Q-X.5 | Legacy metadata warnings; `tools_base` timeline; M.6 beta slugs; gate **415 passed** |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** — **achieved** (Phase Q complete).

**Appendix B relationship:** Closed by Phase Q where mapped. Residual items tracked in **Appendix D** (Phase Q+).

---

## Appendix D — Post-audit hardening traceability (Phase Q+)

**Source:** Technical debt audit (2026-06-01, after Phase Q Wave 9).  
**Goal:** Cursor-/Claude Code–class harness discipline — typed contracts, single orchestration path, full observability on critical paths.

**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### D.1 Audit verdict → Phase Q+ mapping

| Audit theme | Priority | Q+ IDs | Status |
|-------------|----------|--------|--------|
| Duplicate Tier-0 (`tools_agent`, supervisor, chains, rag/answers, openai/rag) | P0–P2 | Q+-L.1–Q+-L.7 | Done (L.7 Won't fix) |
| `getattr` / duck typing (UAEP, tools, context, plans) | P0 | Q+-T.1–Q+-T.8, Q+.0.3 | Done (zero grandfathered paths) |
| Nexus intake/planning still in `nexus_loop` | P0–P1 | Q+-N.1, Q+-N.2 | Done |
| No `RetryCoordinator` | P1 | Q+-N.3 | Done |
| Observability gaps (metrics heuristics, RAG HTTP, planner errors) | P1 | Q+-O.1–Q+-O.4, Q+-N.5 | Done (O.3 Won't fix) |
| `task_metadata` auto-hydrate | P1 | Q+-M.1, Q+-M.2 | Done |
| Planning monoliths (~680/620 lines) | P2 | Q+-P.1–Q+-P.3 | Done |
| `session_manager` monolith (~596 lines) | P2 | Q+-S.1 | Done |
| LLM SDK getattr quarantine | P3 | Q+-I.1 | Done |
| `harness_production_mode` not wired in lab | P1 | Q+-O.2 | Done |
| Thin `GraphExecutor` handoff/retry tests | P1 | Q+-N.4 | Done |

### D.2 First implementation steps (Wave 1 — start here)

Execute in order; one PR per ID where possible.

| Step | ID | Action | Exit criteria |
|------|-----|--------|---------------|
| **1** | Q+.0.3 | Add `scripts/check_harness_no_getattr.py`; wire to gate (grandfather list for existing hits) | CI enforces on new lines |
| **2** | Q+-T.1 | Introduce `UAEPAgent` Protocol; refactor `supports_uaep` + `UAEPExecutor` | Zero getattr on agent in `uaep.py` |
| **3** | Q+-T.2 | `ToolInvokerProtocol`; fix `catalog_context.py` | Typed registry access |
| **4** | Q+-T.3 | `RuntimeState.trace_event` typed | `tool_access_policy` clean |
| **5** | Q+-T.4 | `can_handle(TaskContext)` on `Agent` | All agents updated |
| **6** | Q+-T.5 | Plan union for `tool_runtime` | No getattr on plan source |

**Then Wave 2:** Q+-L.1 → Q+-L.2 → Q+-L.3 → Q+-M.1 (Legal off ToolsAgent, import gates, opt-in Task hydrate).

### D.3 Phase Q+ paydown log

| Date | Q+ ID | Summary |
|------|-------|---------|
| 2026-06-01 | Q+.0.1,Q+.0.2 | Appendix D + execution order added to plan |
| 2026-06-01 | Q+.0.3,Q+-T.1–T.8,Q+-L.1,Q+-M.1,Q+-N.1,Q+-N.2,Q+-D.* | Wave 1 harness contracts; intake/planning runners; CI getattr/tools_agent gates; docs |
| 2026-06-01 | Q+-L.2–L.3,Q+-N.3,Q+-O.1,Q+-O.2 | Legal `CatalogToolPlanner`; `tool_planner` on RuntimeConfig; RetryCoordinator; typed metrics export; lab harness mode |
| 2026-06-01 | Q+-P.2,Q+-S.1,R-Policy | `step_planner/` package; `session_consolidation.py`; `runtime_config_bridge` wires `ToolScopePolicy` |
| 2026-06-01 | Q+-P.1,Q+-S.1,R-Policy | `engine_planner_*` modules; `session_lifecycle.py`; `tool_policy_resolution` + harness getattr cleanup |
| 2026-06-01 | R-Skill catalog | `research.literature_scan` bundle; `ResearchAgent` skill_ids wiring |
| 2026-06-01 | Q+.0.3 (closeout) | Grandfather list cleared; `parser_trace_flush` uses `TraceEventWithTags` Protocol |
| 2026-06-01 | **Phase Q+** | All Q+-* deliverables **Done** or **Won't fix**; gate **450 passed** |
| 2026-06-01 | Appendix C sync, research skill | C.7 T-* / D-05 aligned; `research.literature_scan` bundle; K.1/K.2 **Ready** |
| 2026-06-01 | Doc sync | §1 alignment table, §6 Phase K cadence, Appendix B.8 renumber, E.1 skill row; README + canon research skill examples |
| — | — | *(append row per merged PR)* |

**Coverage target:** 100% **Done** or **Won't fix** — **met** (2026-06-01).

---

---

## Appendix E — Harness AI alignment traceability (Phase R)

**Source:** Harness AI philosophy audit (2026-06-01) — scaffold, harness+LLM=agent, tool vs skill, context engineering, subagents, policy.  
**Goal:** Step-by-step implementation readiness; every audit theme maps to Phase R deliverables.  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### E.1 Audit theme → Phase R mapping

| Audit theme | Intergrax today | Gap | Phase R IDs | Status |
|-------------|-----------------|-----|-------------|--------|
| Scaffold | `intergrax/scaffold` | No `new-skill` | R-Skill.7, R.0.4 | Done |
| Harness = Nexus + platform + app wiring | Tier-1 + Tier-0 + Tier-3 | Terminology not in glossary | R.0.2 §5.3 | Done |
| LLM separate from agent module | `llm_adapters` | “Runnable instance” undefined | R.0.2 §5.3 | Done |
| Tool = atomic operation | `ToolContract`, `ToolRuntime` | Doc said “tool/skill” | R.0.3, R.0.1 | Done |
| Skill = goal-oriented pack | Was missing (pre-R); **MVP Done** | Registry + importers + first-party packs | R-Skill.1–R-Skill.10 | Done |
| Option 1: skills = tools | — | **Rejected** — breaks LLM/MCP atomic model | R.0.1 ADR | Done |
| Option 2: Skill Library | — | **Adopted** | R-Skill.* | Done |
| Context engineering | §27–28, `MemoryView`, `TaskContextAssemblyOptions` | No central budget API | R-Context.* | Done |
| Subagents | `GraphExecutor`, handoff §42.15 | No isolated child namespace | R-Delegate.* | Done |
| Policy | Multiple engines | No single bundle narrative | R-Policy.* | Done |
| External skill compatibility | — | No importer | R-Skill.8 | Done |

### E.2 Four-layer capability model (canonical)

```text
Integration  →  vendor/backend Protocol (Postgres, Bing, Jira REST)
Tool         →  atomic LLM/MCP operation (rag.retrieve, jira.search_tasks)
Skill        →  composable pack: tool_ids + prompts + policy fragment + metadata
Agent        →  domain module: contract, UAEP steps, skill_ids[], local governance
Harness      →  Nexus + Tier-0 + Tier-3 wiring (orchestration, trace, policy enforcement)
```

### E.3 Phase R paydown log

| Date | R ID | Summary |
|------|------|---------|
| 2026-06-01 | R.0.1,R.0.2,R.0.3,R.0.4 | ADR Option 2; canon §5.3, §7.1.8, §28.1, §42.11.4, §42.14.3; ToolContract docstring; plan Appendix E |
| 2026-06-01 | R-Skill.1–R-Skill.9,R-Context.1,R-Delegate.1,R-Policy.1 | Skill Library MVP, legal pilot, ContextBudget, DelegationSpec, gate **422 passed** |
| 2026-06-01 | R-Skill.10,R-Context.2,R-Delegate.2–4,R-Policy.2 | Event recording, delegation memory, graph integration test, policy bundle wiring |
| 2026-06-01 | **Phase R (MVP)** | All R-* deliverables **Done** or **Won't fix**; gate **450 passed** |
| — | — | *(append row per merged PR)* |

**Coverage target:** 100% **Done** or **Won't fix** — **met** (2026-06-01). Phase S proceeds on this harness baseline.

---

## Appendix F — Harness environment traceability (Phase S)

**Source:** Architecture audit + plan pivot (2026-06-01) — **harness environment before business agents**.  
**Goal:** Track Phase S deliverables.  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### F.1 Theme → Phase S mapping

| Theme | S IDs | Status |
|-------|-------|--------|
| Docs / plan pivot | S.0.1–S.0.4 | **Done** |
| Integration + OTLP | S-Ops.1–S-Ops.3 | **Done** |
| Platform harness skills + lab proof | S-H.1–S-H.5 | **Done** |
| Operator documentation | S-Doc.1–S-Doc.2 | **Done** |
| Business agents (→ Phase K) | K.1, K.2 | **Deferred** (was S-K.*) |
| Legal live LLM E2E | S-Ops.4 / K.6 | **Deferred** |

### F.2 Phase S paydown log

| Date | S ID | Summary |
|------|------|---------|
| 2026-06-01 | S.0.* | Strategy doc; canon; initial Phase S |
| 2026-06-01 | S.0.4 | Pivot: Phase S = harness environment only; K.1/K.2 → Phase K |
| 2026-06-01 | **Phase S** | harness_lab_stack, harness.* skills, OTEL profile, HARNESS_ENVIRONMENT.md, tests |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase S definition of done met — **yes** (2026-06-01).

---

## Appendix G — Harness production audit traceability (Phase U)

**Source:** Harness-system audit (2026-06-01) — lab/Tier-1/Tier-3 only; **no business agents**.  
**Goal:** Map every finding to exactly one Phase U deliverable. Update **Status** when **Done** / **Won't fix** (with reason).  
**Status values:** `Open` | `Done` | `Won't fix` | `Deferred`

### G.1 Security (P0)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| SEC-01 | Lab `POST /v1/lab/run` and `/debug/*` without authentication | U-Sec.1 | Done |
| SEC-02 | MCP enabled by default (`LAB_INCLUDE_MCP=true`) — second open surface | U-Sec.2 | Done |
| SEC-03 | `sandbox.exec` enabled in default lab tool profile | U-Sec.3 | Done |
| SEC-04 | `harness_production_mode()` always `False` — no strict production path | U-Sec.4 | Done |

### G.2 Contracts & policy (P1)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| CON-01 | `Agent` (ABC) vs `UAEPAgent` (Protocol) — no unified inheritance | U-Con.1 | Done |
| CON-02 | `RuntimePolicyBundle` built in lab ctx but not applied to `RuntimeConfig` | U-Pol.1 | Done |
| CON-03 | `PolicyEngine` (NexusLoop) vs `policy_bundle` (RuntimeConfig) — dual systems | U-Pol.2 | Done |
| CON-04 | `ToolPlanningService` imports `ToolsAgentConfig` from Tier-0 `tools_agent` | U-Typ.2 | Done |
| CON-05 | `runtime_state` uses `isinstance(CatalogToolPlanner)` not protocol | U-Typ.3 | Done |
| CON-06 | `create_lab_interaction_adapter()` uses `IntegrationProfile.lab()` not preset | U-Arch.1 | Done |
| CON-07 | Skill `skill_ids` resolved at register — no runtime E2E proof in gate | U-Con.3 | Done |

### G.3 Typing & hygiene (P2)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| TYP-01 | `ToolsAgentConfig` tuple bug (`temperature = None,`) | U-Typ.1 | Done |
| TYP-02 | `RuntimePolicyBundle.budget` / `plan_loop` typed as `Any` | U-Pol.3 | Done |
| TYP-03 | `# type: ignore` on lab integration wiring adapters | U-Arch.2 | Done |
| TYP-04 | `getattr` outside harness audit (tools_agent prune, profile, sandbox) | U-Typ.4 | Done |
| TYP-05 | `hasattr` on harness paths (shared_task_context, engine_plan, platform_wiring) | U-Typ.5 | Done |
| TYP-06 | `ToolPlanDecision` vs `AgentDecision` naming collision risk | U-Leg.3 | Done |

### G.4 Legacy & naming (P3)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| LEG-01 | `tools_agent_answer` and ToolsAgent naming in Tier-1 runtime | U-Arch.3 | Done |
| LEG-02 | `ToolsAgent.run` still full orchestrator — deprecation incomplete | U-Leg.1 | Done |
| LEG-03 | `rag.answers` module remains; tests filtered not removed | U-Leg.2 | Done |
| LEG-04 | Legacy tool plan booleans (`from_legacy`, `uses_legacy_rag_flag_only`) | U-Leg.3 | Done |

### G.5 Documentation & CI (P4)

| Audit ID | Finding | U ID | Status |
|----------|---------|------|--------|
| DOC-01 | `HARNESS_ENVIRONMENT.md` claims policy bundle wired — lab does not apply bridge | U-Doc.1, U-Pol.1 | Done |
| DOC-02 | Phase K footer still "after Phase S" in harness docs | U-Doc.3 | Done |
| CI-01 | harness-smoke omits Phase T unit tests | U-CI.1 | Done |
| CI-02 | No acceptance test for strict production harness path | U-CI.2 | Done |
| CI-03 | harness-smoke vs gate run on different OS images | U-CI.3 | Done |

### G.6 Phase U paydown log

| Date | U ID | Summary |
|------|------|---------|
| 2026-06-01 | U.0.* | Appendix G + Phase U section added to implementation plan (audit → backlog) |
| 2026-06-02 | §4.1 | Harness completion: U-Leg.1–3, U-Arch.2, U-Typ.4, U-CI.3, harness.skill_registry, research UAEP parity; gate **481** |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase U + §4.1 harness completion backlog **Done** (2026-06-02). **K.1/K.2 deferred** until product prioritization.

---

## Appendix H — Architecture coverage matrix (Intergrax canon + ideal harness)

**Purpose:** ensure the implementation plan explicitly covers all harness-scope requirements from:

- `intergrax_runtime_architecture.md` (canonical Intergrax runtime architecture)
- `IDEAL_HARNESS_AI_ARCHITECTURE.md` (target/benchmark architecture)

**Rule:** For harness work, this matrix must have **zero `Uncovered` rows**.

### H.1 Coverage status legend

- **Done** — capability implemented and verified by existing phases/tests.
- **Partial closeout** — contracts/governance Done; runtime enforcement gaps scheduled in Phase V-REM.
- **Planned (Phase V-REM)** — explicitly scheduled in Phase V-REM (`V-REM-*` IDs).
- **Deferred (product scope)** — intentionally outside harness-only scope (Band 3 / §6.3).
- **Uncovered** — gap; MUST be added to plan before related implementation proceeds.

### H.2 Harness architecture domains — required coverage

| Domain (harness scope) | Intergrax canon anchor | Ideal harness anchor | Plan coverage | Status |
|------------------------|------------------------|----------------------|---------------|--------|
| Strategic objective + harness-first hierarchy | canon §2, §5.1, §51, §53.1 | ideal §0, §1, §26 | §0, §4.0, Phase V governance | **Done** |
| Tier model and runtime boundaries | canon §5.1, §7.0–§7.4, §42 | ideal §3, §26 | §0.2, §2 map, Phases L/Q+/U | **Done** |
| Unified execution runtime (UAEP, lifecycle, interrupts, policy) | canon §42.* | ideal §3.3, §3.4, §5, §8 | §2 map, Phase U, gate suites | **Done** |
| Context engineering core | canon §28.1, §42.35 | ideal §16 | Phase R (Done) + V-CE.* | **Done** |
| Capability graph dependencies + impact analysis | canon §53.2 | ideal §19 + capability graph expectations | V-CG.* | **Done** |
| Agent lifecycle governance (cert/promo/deprec/retire/owner) | canon §15, §53.3 | ideal §17 | V-ALG.* | **Done** |
| Prompt engineering architecture | canon §53.5 | ideal §20 | V-PE.* | **Done** |
| Evaluation and benchmarking operations | canon §53.6 | ideal §18 | V-EVAL.* + A.4 | **Done** |
| Architecture metrics and debt governance | canon §53.7 | ideal §21 + architecture metrics expectations | V-AM.* | **Done** |
| Security/data governance (agent-native threats) | canon §42.37, §53.8 | ideal §23 | Phase U (baseline) + V-SEC.* | **Done** |
| Cost/resource governance | canon §53.9 | ideal §24 | V-COST.* | **Done** |
| Multi-agent coordination pattern catalog | canon §42.43, §53.10 | ideal §6 + §25 | V-MA.* | **Done** |
| Knowledge graph evolution path (Graph-RAG) | canon §53.11 | ideal §3.7.1 + §25 | V-KG.* | **Done** |
| Observability and runtime traceability | canon §33, §42.24 | ideal §11 | Phases Q/Q+/S/U + §6.1 maintenance | **Done** |
| Registry-driven extensibility (agent/tool/skill/policy/prompt/eval) | canon §7.1.5.1–§7.1.8, §15, §53.2 | ideal §19 | Phase R/U + V-CG/V-PE/V-EVAL + **P-Ext** | **Done** — plugin catalogs production-ready; marketplace UI out of scope |
| Product agents and new product apps | canon §7.4, §52 | ideal §26 | §6.3 only | **Deferred (product scope)** |

### H.3 Completion policy for “architecture-complete harness”

Harness architecture can be considered complete against both architecture documents only when:

1. All harness-scope rows in H.2 are `Done` (no `Partial closeout`, no `Planned`, no `Uncovered`).
2. `Deferred (product scope)` rows remain intentionally isolated to Band 3 (§6.3).
3. Phase V-REM complete and parent V-* Partial rows closed.
4. Phase V KPI thresholds and L3/L4 evidence gates are satisfied.
5. Canon + plan + docs index are synchronized in the same change window.

### H.4 Change control rule

Any future addition to either architecture document that introduces a new harness-scope
domain MUST be reflected in:

- this matrix (Appendix H),
- a concrete Phase V-REM (or successor phase) deliverable ID,
- priority ladder (§4) and “what next” (§6) if it changes execution order.

---

## Appendix I — Plugin catalog traceability (Phase P-Ext)

**Purpose:** Task-level tracker for plugin-native Integration, Tool, and Skill catalogs. **Canonical phase narrative:** [Phase P-Ext](#phase-p-ext--plugin-catalogs-integrations-tools-skills) · paydown: [P-Ext.6](#p-ext6--production-closure-paydown).

**Status:** **Done** (2026-06-02) · **MVP effort:** ~21–32 person-days · **paydown estimate:** ~8–14 person-days.

### I.1 Delivery rule

Same as §6.1: one **P-Ext.\*** ID → PR → update status in this appendix → `pytest -m gate` green. Paydown cadence: [§6.1p](#61p-phase-p-ext-paydown-band-2c--optional-parallel-with-61).

### I.2 Task register

| ID | Layer | Summary | Status | Priority |
|----|-------|---------|--------|----------|
| P-Ext.0.1 | All | `load_plugins()` / entry point discovery | **Done** | P0 |
| P-Ext.0.2 | All | `PluginConflictError`, `PluginLoadError` | **Done** | P0 |
| P-Ext.0.3 | All | `bootstrap_catalogs()` Tier-3 API | **Done** | P0 |
| P-Ext.0.4 | All | `EXTENSION_AUTHOR_GUIDE.md` (EN) | **Done** | P0 |
| P-Ext.0.5 | All | Test fixture pip package | **Done** | P0 |
| P-Ext.0.6 | All | EP discovery tests (3 groups) | **Done** | P0 |
| P-Ext.0.7 | All | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | **Done** | P1 |
| P-Ext.1.1 | Integrations | Entry points `intergrax.integrations` | **Done** | P0 |
| P-Ext.1.2 | Integrations | `bootstrap_core` / optional split | **Done** | P1 |
| P-Ext.1.3 | Integrations | Typed `resolve_*` helpers (top categories) | **Done** | P2 |
| P-Ext.1.3a | Integrations | Expand `resolve_typed` + tests | **Done** | P2 |
| P-Ext.1.4 | Integrations | Health check API (optional) | **Done** | P3 |
| P-Ext.1.5 | Integrations | `IntegrationSlug` cleanup (docs/scripts) | **Done** | P2 |
| P-Ext.1.6 | Integrations | EP test via fixture | **Done** | P0 |
| P-Ext.1.7 | Integrations | Dual-model docs (manifest vs plugin) | **Done** | P2 |
| P-Ext.1.8 | Integrations | CI integration slug count smoke | **Done** | P1 |
| P-Ext.1.9 | Integrations | `test_resolve_typed.py` | **Done** | P3 |
| P-Ext.1.10 | Integrations | Tier-3 `bootstrap_catalogs` in integration_wiring | **Done** | P0 |
| P-Ext.1.11 | Integrations | `_shared/integration_wiring.py` helper | **Done** | P2 |
| P-Ext.1.12 | Integrations | `SqliteIntegrationPlugin` wire or document | **Done** | P3 |
| P-Ext.2.1 | Tools | `ToolPlugin` Protocol | **Done** | P0 |
| P-Ext.2.2 | Tools | `ToolBundleManifest` / bundle metadata | **Done** | P0 |
| P-Ext.2.3 | Tools | `register_tool_plugin()` | **Done** | P0 |
| P-Ext.2.4 | Tools | RAG bundle plugin migration (pilot) | **Done** | P1 |
| P-Ext.2.5 | Tools | Entry points `intergrax.tools` | **Done** | P1 |
| P-Ext.2.6 | Tools | MCP tool export | **Done** | P1 |
| P-Ext.2.7 | Tools | `ToolContract.version` | **Done** | P2 |
| P-Ext.2.8 | Tools | All 13 shipped bundles → `ToolPlugin` | **Done** | P1 |
| P-Ext.2.9 | Tools | `tools/examples/` reference package | **Done** | P0 |
| P-Ext.2.10 | Tools | `test_external_tool_plugin.py` | **Done** | P0 |
| P-Ext.2.11 | Tools | EP tool test via fixture | **Done** | P0 |
| P-Ext.2.12 | Tools | `tool_wiring` lazy `tool_bundle_ids` | **Done** | P2 |
| P-Ext.3.1 | Skills | `SkillPlugin` Protocol | **Done** | P1 |
| P-Ext.3.2 | Skills | `register_skill_plugin()` | **Done** | P1 |
| P-Ext.3.3 | Skills | Entry points `intergrax.skills` | **Done** | P1 |
| P-Ext.3.4 | Skills | harness + research + legal plugin migration | **Done** | P1 |
| P-Ext.3.5 | Skills | `requires_skills` (optional) | **Done** | P3 |
| P-Ext.3.6 | Skills | `skills/examples/` reference package | **Done** | P0 |
| P-Ext.3.7 | Skills | `test_external_skill_plugin.py` | **Done** | P0 |
| P-Ext.3.8 | Skills | EP skill test via fixture | **Done** | P0 |
| P-Ext.3.9 | Skills | `skill_wiring` lazy `skill_bundle_ids` | **Done** | P2 |
| P-Ext.3.10 | Skills | Scaffold `new-skill` → `SkillPlugin` | **Done** | P2 |
| P-Ext.3.11 | Skills | Docs: SkillPlugin vs Cursor importer | **Done** | P2 |
| P-Ext.3.12 | Skills | Shipped `requires_skills` demo (optional) | **Done** | P3 |
| P-Ext.4.1 | Ops | Lazy profile bootstrap | **Done** | P2 |
| P-Ext.4.2 | Ops | `CatalogSnapshot` API | **Done** | P2 |
| P-Ext.4.3 | Ops | Slug conflict policy (bootstrap) | **Done** | P2 |
| P-Ext.4.4 | Ops | `check_plugin_catalog.py` CI | **Done** | P1 |
| P-Ext.4.5 | Ops | CI smoke: tool/skill bundle counts | **Done** | P1 |
| P-Ext.5.1 | Docs | Scaffold `new_*` commands | **Done** | P2 |
| P-Ext.5.2 | Docs | INTEGRATIONS/TOOLS/SKILLS external sections | **Done** | P2 |
| P-Ext.5.3 | Docs | Canon §7.1.5.1 plugin narrative | **Done** | P1 |
| P-Ext.5.4 | Docs | remove `PLUGIN_CATALOG_PLAN.md` | **Done** | P3 |
| P-Ext.5.5 | Docs | Prod path matrix in author guide | **Done** | P2 |
| P-Ext.5.6 | Docs | Lab wiring recipe for external plugins | **Done** | P2 |
| P-Ext.6.1 | Paydown | Fixture pip package (rollup) | **Done** | P0 |
| P-Ext.6.2 | Paydown | External tool + skill examples + tests | **Done** | P0 |
| P-Ext.6.3 | Paydown | EP discovery + lab env | **Done** | P1 |
| P-Ext.6.4 | Paydown | IntegrationSlug cleanup | **Done** | P2 |
| P-Ext.6.5 | Paydown | Scaffold CLI | **Done** | P2 |
| P-Ext.6.6 | Paydown | Integration Tier-3 + typed resolve + health | **Done** | P2 |
| P-Ext.6.7 | Paydown | Conflict policy + CI smoke | **Done** | P1 |
| P-Ext.6.8 | Paydown | Skill Tier-3 + scaffold rollup | **Done** | P2 |
| P-Ext.6.9 | Paydown | Tool Tier-3 lazy wiring rollup | **Done** | P2 |
| P-Ext.6.10 | Paydown | Tier-3 lazy wiring (all catalogs) rollup | **Done** | P2 |

**Paydown summary:** 0 **Planned** · 61 **Done** · 0 **Partial** (Phase P-Ext production closure complete; rollup rows duplicate leaf IDs).

### I.3 Market alignment checklist

| Pattern | Target |
|---------|--------|
| Hexagonal adapters | `IntegrationCategory` + contracts + `IntegrationPlugin` |
| MCP tools | `ToolContract` + `export_mcp_tools` |
| Capability packs | `SkillManifest` + resolver (not LLM-invokable) |
| 12-factor config | env_prefix + `IntegrationProfile.options` |
| Plugin discovery | entry points (hybrid with explicit bootstrap) |
| Tier-3 composition root | `bootstrap_catalogs()` |

### I.4 Paydown log

| Date | P-Ext ID | Summary |
|------|----------|---------|
| 2026-06-02 | — | Phase P-Ext + Appendix I added (migrated from `PLUGIN_CATALOG_PLAN.md`) |
| 2026-06-02 | 0.1–0.4, 1.1–1.2, 2.1–2.8, 3.1–3.5, 4.1–4.2, 4.4, 5.2–5.4 | MVP: protocols, bootstrap, 13 tool + 3 skill plugins, lazy catalog, `custom_memory_kv` test |
| 2026-06-02 | — | Plan updated: **MVP Done** + **P-Ext.6 paydown** backlog (EP fixture, external tool/skill tests, ops/docs) |
| 2026-06-02 | 1.* audit | Integrations audit: 12 core / ~99 full manifest path; `resolve_typed` partial; Tier-3 integration_wiring gap; +P-Ext.1.3a, 1.8–1.12 |
| 2026-06-02 | 3.* audit | Skills audit: 3/3 `SkillPlugin`, 8 skill_id; Tier-3 `skill_wiring` OK; scaffold legacy; +P-Ext.3.9–3.12, 6.8 |
| 2026-06-02 | 2.* audit | Tools audit section + `tool_wiring` lazy (P-Ext.2.12); P-Ext.4.5 unified counts; +P-Ext.6.9–6.10 |
| 2026-06-02 | P-Ext paydown | Fixture EP package, external examples/tests, Tier-3 wiring, docs, CI smoke (residual: 1.5, 4.3, 5.1, 5.6) |
| 2026-06-02 | P-Ext closure | IntegrationSlug docs cleanup, `warn_override` conflict policy, scaffold CLI, lab wiring recipe |
| 2026-06-02 | P-Ext complete | Phase narrative + §6.1p synced; expanded `check_plugin_catalog.py` smoke suite |
| 2026-06-02 | §6.1 | Gate green **486**: IntegrationBinding test fixes, circular import, catalog re-bootstrap after test clears, scaffold templates |
| 2026-06-02 | TYP-06, U-Typ.4 | `IntegrationProfile` explicit binding accessors; removed `tools_agent.AgentDecision` alias |
| 2026-06-02 | W-OPS.0 | Harness maturity audit → Phase W-OPS + §6.2w in implementation plan |
| 2026-06-05 | V-REM.0.* | Plan audit → Phase V-REM + Appendix J + §6.1z queue (10 open) |
| — | — | *(append row per merged PR)* |

---

## Appendix J — Phase V remediation traceability (audit gap → V-REM ID)

**Purpose:** 100% mapping from **Partial** audit findings (2026-06-05) to concrete remediation IDs. **Canonical phase narrative:** [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout).

**Status:** **12 tasks** · **12 Done** (2026-06-05).

### J.1 Audit gap → remediation matrix

| Audit source | Layer / area | Gap | Severity | Parent plan ID | V-REM ID | Status |
|--------------|--------------|-----|----------|----------------|----------|--------|
| Plan/code audit 2026-06-05 | Capability graph (AUDIT_MAP §19) | System edges agents→application incorrect per host | **Critical** | V-CG.2, V-CG.3, V-CG.4 | V-REM-CG.1, V-REM-CG.2 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Governance contracts exist; no runtime routing cutoff for retired/deprecated | High | V-ALG.3 | V-REM-ALG.1 | **Done** |
| Plan/code audit 2026-06-05 | Agent lifecycle (AUDIT_MAP §31) | Ownership contracts exist; no production-eligible filter at selection | High | V-ALG.4 | V-REM-ALG.2 | **Done** |
| Plan/code audit 2026-06-05 | Prompt registry (AUDIT_MAP §17) | PromptMeta missing owner/risk; no YAML assets for E2E validation | High | V-PE.1 | V-REM-PE.1, V-REM-PE.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tool injection defense not wired on execution path | High | V-SEC.2 | V-REM-SEC.1 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Retrieval poisoning defense not enforced per tenant/app | High | V-SEC.3 | V-REM-SEC.2 | **Done** |
| Plan/code audit 2026-06-05 | Security (AUDIT_MAP §23) | Tenant isolation + audit trail hooks missing in main path | High | V-SEC.4 | V-REM-SEC.3 | **Done** |
| Plan/code audit 2026-06-05 | Evaluation (AUDIT_MAP §25) | NexusEvalRunner exists; missing integration tests + gate | Medium | A.4, A.4.1 | V-REM-A.1 | **Done** |
| Plan sync 2026-06-05 | Plan governance | Appendix J + §6.1z queue + status sync | — | — | V-REM.0.1, V-REM.0.2 | **Done** |

**Coverage target:** 100% **Done** when every **Planned** row is **Done** and parent Partial IDs (V-CG.2–4, V-ALG.3–4, V-PE.1, V-SEC.2–4, A.4) are **Done**.

### J.2 Paydown log

| Date | V-REM ID | Summary |
|------|----------|---------|
| 2026-06-05 | V-REM.0.1, V-REM.0.2 | Appendix J + Phase V-REM section + §6.1z/§6.2v + Appendix H sync |
| 2026-06-05 | V-REM-CG.1–A.1 | Runtime remediation: capability graph, lifecycle routing, V-SEC wiring, prompt governance, EvalRunner gate |
| 2026-06-05 | V-POST.1, V-POST.2 | Phase V closeout gate green; AgentEngine routability guard; NexusLoop tenant-security integration tests |

---

*Plan synced (2026-06-05). **Harness platform** bands 1–2i **Done**; Phase V + V-REM **closed**. Gate: **571 passed**. **Default next:** [§6.1](#61-harness-platform-maintenance-default--band-1) maintenance. Product work gated by [§6.3](#63-end-of-plan--deferred-product-work-only). Operational L3 **signed off**. P-Ext **Done** (61/61).*
