# PLATFORM_FOUNDATION — embedded detail

**Parent hub:** [`PLATFORM_FOUNDATION.md`](../PLATFORM_FOUNDATION.md)

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §0, §21 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2az** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-1.1 | §1 Strategic | Operationalize quarterly strategy review process | P2 | **Done** |
| AUDIT-IDEAL-1.2 | §1 Strategic | Architecture health metrics as live signals | P2 | **Done** |
| AUDIT-IDEAL-2.1 | §2 Tiers | Continuous tier-boundary gate maintenance | P3 | **Done** (gates exist) |
| AUDIT-IDEAL-32.1 | §32 Doc gov | Living architecture debt burn-down tied to milestones | P2 | **Done** |
| AUDIT-IDEAL-32.2 | §32 Doc gov | Scorecard auto-sync on plan row change | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

## 0. Architecture at a glance

Condensed from the canon. For full contracts and forbidden patterns, read `intergrax_runtime_architecture.md`.

### 0.1 Strategic objective

Intergrax is an **Agent Operating System / Harness AI runtime** — not a collection of business agents. **Priority 1:** production-grade Harness AI (see [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)).

Current optimization targets:

- **harness environment GA** (Phase S) · **developer authoring UX** (Phase DX) · experimentation speed · agent creation speed · runtime stability
- orchestration quality · observability · composability · skill/platform packs (Integration → Tool → Skill)

**Harness GA (Phase L):** Agent OS certified — Appendix A **20/20**. New agents ship via scaffold without Nexus edits.

**Harness environment (Phase S):** **Done** (2026-06-01) — stable stack, OTLP profile, `harness.*` skills, `guides/HARNESS_ENVIRONMENT.md`, CI smoke. **Did not include** business agents (K.1/K.2).

**Harness cleanliness (Phase T):** **Done** (2026-06-01) — unified `lab_harness_preset()`, typed reference agents, native `CatalogToolPlanner`, expanded stable stack. See Phase T.

**Harness production hardening (Phase U):** **Done** (2026-06-01) — auth surfaces, strict harness profile, `HarnessReferenceAgent`, typed policy bundle, planner decoupling, sandbox opt-in. **U-Leg** legacy removal remains tracked in Appendix G. See Phase U + **[Appendix G](plan/plan/PLATFORM_FOUNDATION_appendices.md)**.

**Product agents (Phase K):** Problem Radar (K.1), Vendor Discovery (K.2) — **end of plan** (§4.0 Band 3, §6.3); not default next. K.3–K.5 platform hardening **Done**.

**Platform quality (Phase Q):** Done (2026-06-01) — first harness audit remediation; gate was **417 passed** at close (see Appendix C).

**Harness hardening (Phase Q+):** **Done** (2026-06-01) — Protocols (zero grandfathered `getattr` in harness paths), legacy stack removal, Nexus decomposition, monolith splits. See Appendix D.

**Harness AI alignment (Phase R):** **Done (MVP)** (2026-06-01) — **Skill Library**, context-engineering API, graph-native delegation, unified policy bundle. See Appendix E. **Phase S** hardens the **environment** agents run in; product agents follow in **Phase K**.

**Skill layer decision (ADR R.0.1):** **Do not** collapse skills into tools. Tools remain **atomic LLM-invokable operations**; skills are **composable capability packs** (tools + prompts + policy + metadata) with **import adapters** for external skill formats (e.g. Cursor `SKILL.md`). See architecture §7.1.8.

**Plugin catalogs (Phase P-Ext):** **Done** (2026-06-02) — protocols, `bootstrap_catalogs()`, EP fixture, conflict policy, scaffold CLI, 13/13 `ToolPlugin`, 3/3 `SkillPlugin`. See Appendix I · [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md).

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

**Detailed narrative** (sequence/state diagrams, decision matrix, edge cases, Phase FLOW paydown): [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) — **Done** 18/18 harness (FLOW-8 harness **Done**; product host **Deferred** §6.3).

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
| **Application environment profile (Phase H-APP)** | **Done** (2026-06-03) | No (harness-only) | [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7 — 43 tasks; memory bridge gap → [Phase MEM](plan/MEMORY.md) |
| **Memory platform (Phase MEM)** | **Done** (~3,5/5 post-closeout) | No (harness-only) | Memory platform **48/48** — gate **581** |
| **Memory intelligence depth (Phase MEM-DEPTH)** | **Done** (26/26) | No (harness-only) | [`architecture/MEMORY.md`](architecture/MEMORY.md) · Band **2am** · **§6.2ab** |
| **Governance audit closeout (GOV-AUDIT)** | **Done** (docs) | No | GOV-DOC.1–2; code via V-REM/H-APP/DX-5.8 |
| **Orchestration closeout (Phase ORCH)** | **Done** (Band 2j) | No (harness-only) | ORCH-1–4 — [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) |
| **Tools/skills closeout (Phase TS)** | **Done** (Band 2k) | No (harness-only) | TS-1–3 — [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) |
| **Integration closeout (Phase INT)** | **Done** (Band 2l) | No (harness-only) | INT-1–2 — [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) |
| **RAG closeout (Phase RAG)** | **Done** (Band 2m) | No (harness-only) | RAG-1 — [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) |
| **Context engineering closeout (Phase CTX)** | **Done** (Band 2n) | No (harness-only) | CTX-1–2 — [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed) |
| **Prompt registry closeout (Phase PE)** | **Done** (Band 2p) | No (harness-only) | PE-1–3 — [§6.1i](#61i-harness-implementation-queue--prompt-registry-closeout-closed) |
| **Legacy module closeout (Phase CLEAN)** | **Done** | No (harness-only) | CLEAN-1–4 — [§6.1j](#61j-harness-implementation-queue--legacy-module-closeout-closed) |
| **Agent assembly closeout (Phase AS)** | **Done** (Band 2q) | No (harness-only) | AS-1–3 — [§6.1k](#61k-harness-implementation-queue--agent-assembly-closeout-closed) |
| **Registry architecture closeout (Phase REG)** | **Done** (Band 2r) | No (harness-only) | REG-1–3 — [§6.1l](#61l-harness-implementation-queue--registry-architecture-closeout-closed) |
| **Capability graph closeout (Phase CG)** | **Done** (Band 2s) | No (harness-only) | CG-1–3 — [§6.1m](#61m-harness-implementation-queue--capability-graph-closeout-closed) |
| **Observability closeout (Phase OBS)** | **Done** (Band 2t) | No (harness-only) | OBS-1–3 — [§6.1n](#61n-harness-implementation-queue--observability-closeout-closed) |
| **Reliability closeout (Phase REL)** | **Done** (Band 2u) | No (harness-only) | REL-1–3 — [§6.1o](#61o-harness-implementation-queue--reliability-closeout-closed) |
| **Security closeout (Phase SEC)** | **Done** (Band 2v) | No (harness-only) | SEC-1–3 — [§6.1q](#61q-harness-implementation-queue--security-closeout-closed) |
| **Cost governance closeout (Phase COST)** | **Done** (Band 2w) | No (harness-only) | COST-1–3 — [§6.1r](#61r-harness-implementation-queue--cost-governance-closeout-closed) |
| **Evaluation closeout (Phase EVAL)** | **Done** (Band 2x) | No (harness-only) | EVAL-1–3 — [§6.1s](#61s-harness-implementation-queue--evaluation-closeout-closed) |
| **Adaptive Harness Intelligence (Phase W-ADAPT)** | **Done** (Band 2y) | No (harness-only) | Wave 0–7 **Done** (70/70) · [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed) · AHIA |
| **LLM completion envelope (Phase M-LLM-R)** | **Done** (Band 2z) | No (harness-only) | Audit 2026-06-06 — typed `LLMAdapterResponse`; **39/39** — [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed) · **Appendix L** |
| Regression gate | **1498 passed** | No | Must stay green after each harness PR (gate snapshot 2026-06-19; was 906 at FLOW closeout 2026-06-07) |
| **Full architecture audit (FAUDIT-32)** | **Done** (2026-06-06) | No (harness-only) | 32-layer audit + **23/23 remediation** → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) |
| **Nexus execution depth (Phase FLOW)** | **Done** (18/18 harness) | No (harness-only) | Band **2aj** — [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · FLOW-8 harness **Done**; product host **Deferred** §6.3 · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) |
| **Critic & Verification Layer (Phase CRIT-V)** | **Done** (24/24) | No (harness-only) | Band **2ak** — [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) |
| **Unified Observability Spine (Phase OBS-BUS)** | **Done** (8/8) | No (harness-only) | Band **2al** — [§6.1al](#61al-harness-implementation-queue--unified-observability-spine-closed) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md) |

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

| §7.1.8 Skill Library | Composable capability packs, importers | **MVP Done** | `intergrax/skills/` · `docs/architecture/SKILLS.md` |
| §7.1.9 Model & Modality Plane | Vision (YOLO/ONNX/…), speech, classical ML, HF roles | **Done** | `docs/architecture/MODALITY.md` · Phase W-ML |

| §5.3 Harness AI alignment | Scaffold, skill/tool split, context, delegation | **Done** | `scaffold/new_skill.py`, `intergrax/skills/`, Appendix E |

| §23 Task lifecycle | States + trace + typed contract | **Done** | `task/`, `task_contract.py`, `TaskContextAssemblyOptions`, `task_metadata_bridge.py` |

| §24–25 Execution graph | Multi-agent | **Done** | `execution/`, `GraphExecutor` |

| §29 Validation | Nexus + agent | **Done** | `NexusValidationEngine` |

| §31 Retry | Runtime-managed | **Done** | `RetryEngine` |

| §33 Observability | Trace + events | **Done** (lab scope) | Trace + runtime events + metrics export (`B.08`–`B.11`); OTel provider beta |

| §42 Execution runtime | UAEP, hooks, governance, tool gateway | **Done (~99%)** | UAEP Protocol, planner events, tool gateway, harness getattr audit |
| §19 Debug / experiments | CLI, API, registry, cost | **Done** | D.1–D.5 ✅ |

| §7.4 Repo split | agents / applications | **Done** | `agents/legal`, `applications/legal_application` |
| §7.1 Integration Library | Catalog + contracts + providers | **Done** · **167 slugs** | Phase M + M.6 P4/P5/P6 **Done** |

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

Historical phase registers: [`plan/plan/PLATFORM_FOUNDATION_phase_closeout.md`](plan/plan/PLATFORM_FOUNDATION_phase_closeout.md).

| Domain | File |
|--------|------|
| Historical A–V | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) |
| Core runtime | [`plan/ORCHESTRATION.md`](plan/ORCHESTRATION.md) |
| Integrations | [`plan/INTEGRATIONS.md`](plan/INTEGRATIONS.md) |
| Tools & skills | [`plan/TOOLS.md`](plan/TOOLS.md) |
| LLM & modality | [`plan/LLM_ADAPTERS.md`](plan/LLM_ADAPTERS.md) |
| RAG engine | [`plan/RAG.md`](plan/RAG.md) |
| Context, memory | [`plan/MEMORY.md`](plan/MEMORY.md) |
| Governance & security | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](plan/UNIFIED_EXECUTION_RUNTIME.md) |
| Observability & reliability | [`plan/OBSERVABILITY.md`](plan/OBSERVABILITY.md) |
| Registry & capability graph | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Evaluation, AHI, critic | [`plan/CRITIC_VERIFICATION.md`](plan/CRITIC_VERIFICATION.md) |
| Tier-3, DX, conformance | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) |
| Platform quality | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) |
Appendices: [`plan/`](plan/)

## 4. Priority Order

### 4.0 Implementation priority ladder (canonical)

**Read this before §6.** The plan has three bands. Implement **top to bottom**. **Never** pull items from band 3 into “next step” summaries while band 1–2 are the active policy.

| Band | What | Status (2026-06-05) | Examples |
|------|------|---------------------|----------|
| **1 — Harness platform** | Tier-0/1/3 lab wiring, security, policy, typing, legacy removal, gate audits | **Maintenance** (§4.1 **Done**; keep green) | `pytest -m gate`, `check_harness_*`, `check_legacy_modules_removed.py`, regression fixes |
| **2 — Harness architecture hardening** | Capability graph, lifecycle governance, prompt/eval/context/security/cost/metrics hardening — **no** business domain | **Done** (2026-06-05) | V-CG … V-KG, V-V6 closeout · V-REM |
| **2i — Phase V runtime remediation (V-REM)** | Close 9 Partial Phase V + EvalRunner gate gaps — runtime enforcement, not new OS features | **Done** (2026-06-05) | [Phase V-REM](plan/ORCHESTRATION.md) · Appendix J |
| **2b — Modality plane (optional parallel)** | Vision CV, speech, classical ML — harness Tier-0 only | **Done** | W-ML complete; optional Celery bus wiring for Tier-3 scale-out |
| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **Done** (2026-06-02) | Appendix I · [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md) |
| **2d — Operational L3 (W-OPS)** | Reliability, identity, SLO/ops evidence, online eval — **no** business agents | **Done** (2026-06-06) | [Phase W-OPS](plan/PLATFORM_FOUNDATION.md) · `phase_w_ops_evidence.py` |
| **2e — Application environment (H-APP)** | `ApplicationEnvironmentProfile`, unified Tier-3 wiring, host migration — **no** business agents | **Done** (2026-06-03) | [Phase H-APP](plan/TIER3_APPLICATION_ENVIRONMENT.md) · [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) · **§6.2x** |
| **2f — Developer authoring UX (DX)** | LangGraph-like facades, minimal scaffold, CLI run/doctor, TTFRun gates, UI spec export — **no** business agents | **Done** (2026-06-03) | [Phase DX](plan/TIER3_APPLICATION_ENVIRONMENT.md) · **§6.2y** |
| **2g — Agents & applications conformance (AA)** | Scaffold alignment, per-agent/app `ARCHITECTURE.md`, deploy triad, legal **scaffold** reset (domain steps → Band 3) | **Mostly Done** (2026-06-02) | [Phase AA](plan/TIER3_APPLICATION_ENVIRONMENT.md) · **§6.2z** · [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **2h — Memory platform (MEM)** | H-APP→runtime bridge, durable user LTM, session SQLite, gates, hooks, memory docs — **no** business agents | **Done** (2026-06-02) | [Phase MEM](plan/MEMORY.md) · **§6.2aa** |
| **2j — Orchestration closeout (ORCH)** | Wire `planner_kind`/`classifier_kind`, `ApplicationGraphSpec`→plan, graph concurrency cap — **no** business agents | **Done** (2026-06-05) | [Phase ORCH](plan/ORCHESTRATION.md) · **§6.1b** · **§6.2bb** |
| **2k — Tools/skills closeout (TS)** | Catalog→`RuntimeConfig` bridge, harness LLM wiring, `SkillResolverProtocol`, Appendix J — **no** business agents | **Done** (2026-06-02) | [Phase TS](plan/TOOLS.md) · **§6.1c** · **§6.2bc** |
| **2l — Integration closeout (INT)** | `integration_runtime_bridge`, bootstrap health probes, Appendix K — **no** business agents | **Done** (2026-06-02) | [Phase INT](plan/INTEGRATIONS.md) · **§6.1d** · **§6.2bd** |
| **2m — RAG closeout (RAG)** | `rag_runtime_bridge`, RAG stack on environment wire — **no** business agents | **Done** (2026-06-02) | [Phase RAG](plan/RAG.md) · **§6.1e** · **§6.2be** |
| **2n — Context engineering closeout (CTX)** | `context_runtime_bridge`, `context_wiring`, Nexus `ContextManager` wire — **no** business agents | **Done** (2026-06-02) | [Phase CTX](plan/MEMORY.md) · **§6.1f** · **§6.2bf** |
| **2o — Legacy tool plan closeout (LEG)** | `tool_ids` canonical path; gateway/engine planner migration — **no** business agents | **Done** (2026-06-02) | [Phase LEG](plan/TOOLS.md) · **§6.1h** |
| **2p — Prompt registry closeout (PE)** | `PromptProfile`, `prompt_runtime_bridge`, `prompt_wiring`, Appendix M — **no** business agents | **Done** (2026-06-02) | [Phase PE](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1i** |
| **2q — Agent assembly closeout (AS)** | Agent contract conformance, capability/skill resolution, lifecycle state — **no** business agents | **Done** (2026-06-02) | [Phase AS](plan/ORCHESTRATION.md) · **§6.1k** · **Appendix N** |
| **2r — Registry architecture closeout (REG)** | Registry snapshot, assembly resolver, host resolution CI — **no** business agents | **Done** (2026-06-02) | [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1l** · **Appendix O** |
| **2s — Capability graph closeout (CG)** | Environment graph slice, wire-time validation, CI audit — **no** business agents | **Done** (2026-06-02) | [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1m** · **Appendix P** |
| **2t — Observability closeout (OBS)** | Profile bridge, assembly resolver, host wiring CI — **no** business agents | **Done** (2026-06-02) | [Phase OBS](plan/OBSERVABILITY.md) · **§6.1n** · **Appendix Q** |
| **2u — Reliability closeout (REL)** | Idempotency bridge, circuit breaker wire, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase REL](plan/OBSERVABILITY.md) · **§6.1o** · **Appendix R** |
| **2v — Security closeout (SEC)** | V-SEC bridge, middleware assembly resolver, host CI — **no** business agents | **Done** (2026-06-02) | [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1q** · **Appendix S** |
| **2w — Cost governance closeout (COST)** | Budget bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1r** · **Appendix T** |
| **2x — Evaluation closeout (EVAL)** | Registry bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase EVAL](plan/CRITIC_VERIFICATION.md) · **§6.1s** · **Appendix U** |
| **2y — Adaptive Harness Intelligence (W-ADAPT)** | L4 **runtime** closed loop — SignalCollector, AdaptationEngine, ProfileVersionStore, verify/rollback — **no** business agents | **Done** (2026-06-02) — **70/70 Done** | [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · **§6.1t** · **§6.2ac** · **Appendix K** |
| **2z — LLM completion envelope (M-LLM-R)** | Typed `LLMAdapterResponse` replaces `str`/`dict` adapter returns; full consumer refactor — **no** business agents | **Done** (2026-06-06) — **39/39** | [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · **§6.1v** · **§6.2ad** · **Appendix L** |
| **2aa — Integration expansion (M.6 P4)** | 28 harness-ROI provider slugs (secrets, observability stack, OLAP, feature flags, prod deploy) — **no** business agents | **Done** (2026-06-02) — **28/28** | [M.6 P4 register](#m6-p4--harness-platform-expansion-done) · **§6.1w** · **§6.2ae** |
| **2ab — Integration depth (M.6 P5)** | Harden 25 beta + 8 greenfield harness slugs (metrics, CI/CD, eval, async, data plane) — **no** business agents | **Done** (2026-06-02) — **33/34** | [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) · **§6.1x** · **§6.2af** |
| **2ac — Integration expansion (M.6 P6)** | 32 harness slugs + post-catalog wiring (tools, bridges, promote gate, infra `p6`) — **no** business agents | **Done** (2026-06-02) — **32/32 + M-P6-WIRE** | [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · **§6.1y** · **§6.2ag** |
| **2ad — FAUDIT-32 remediation** | Close 32-layer audit residuals (tier gate, intake, observability taxonomy, registry depth, eval release gate) — **no** business agents | **Done** (2026-06-06) — **23/23 + §6.1ai follow-up** | [Phase FAUDIT-32](plan/plan/PLATFORM_FOUNDATION_phase_closeout.md) · **§6.1ah** · **§6.1ai** · **[Appendix M](plan/plan/PLATFORM_FOUNDATION_appendices.md)** |
| **2aj — Nexus execution depth (FLOW)** | Close `FLOW-GAP.*` (01–16) — delegation, SubtaskContract, backpressure profile, LLM planner, merge, eval, graph hardening — **no** K.1/K.2 | **Done** (2026-06-07) — **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3) | [Phase FLOW](plan/ORCHESTRATION.md) · **§6.1aj** · **§6.2aj** · **Appendix N (FLOW)** |
| **2ak — Critic & Verification Layer (CRIT-V)** | PEV verify depth — `CriticOrchestrator`, `eval.judge`, `eval.trajectory`, evaluator-loop, semantic offline runner — **no** business agents | **Done** | [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · **§6.1ak** · **§6.2ak** · canon §55 · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md) |
| **2al — Unified Observability Spine (OBS-BUS)** | Full HOS — typed payloads, `ObservabilityEmitter`, emission coverage, extension SDK, L4 §21 — **no** business agents | **Done** | [Phase OBS-BUS](plan/OBSERVABILITY.md) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **§6.1al** · [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md) |
| **2am — Memory intelligence depth (MEM-DEPTH)** | Context Compiler, never-overflow invariant, lifecycle automation, explore delegation, entity memory — **no** business agents | **Done** (2026-06-08) — **26/26** | [Phase MEM-DEPTH](plan/MEMORY.md) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · **§6.2ab** |
| **2an — Elastic capacity domain pair (ECP-DOC)** | ECP canon + ADR — docs only | **Done** (2026-06-08) | [Phase ECP-DOC](plan/ELASTIC_CAPACITY_AND_SCALING.md) · Band **2an** |
| **2ao — Elastic capacity runtime (ECP-DEPTH)** | ScalingProfile, signal collector, evaluator, K8s provisioner, policy gates — **no** business agents | **Done** (2026-06-09) — **28/28** | [Phase ECP-DEPTH](plan/ELASTIC_CAPACITY_AND_SCALING.md) · [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) |
| **2ap — Orchestration strategy canon (ORCH-STRAT)** | §50–§54 coordination/resilience docs | **Done** (2026-06-09) | [Phase ORCH-STRAT](plan/ORCHESTRATION.md) |
| **2ar — Platform interaction config (ORCH-CONFIG)** | CFG-* harness simulation + reference host presets — **no** business agents | **Done** (2026-06-09) — **11/11** | [Phase ORCH-CONFIG](plan/ORCHESTRATION.md) · [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §56–§59 |
| **2as — Reasoning layer depth (COG-DEPTH)** | Planner unification, Prompt Registry on planners, DecisionRecord, failure taxonomy — **no** business agents | **Done** (2026-06-09) — **22/22** | [Phase COG-DEPTH](plan/REASONING_AND_COGNITION.md) · [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) |
| **2aw — Tier-3 execution surface parity (H-APP-WIRING)** | Close FLOW-GAP-17–20 / ORCH §59 Tier-3 wiring debt — task control API, async exposure, reference host adoption — **no** Nexus fork | **Done** (2026-06-09) — **6/6** | [Phase H-APP-WIRING](plan/TIER3_APPLICATION_ENVIRONMENT.md) · [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §59 |
| **2ay — LLM guardrail integrations (M.12 / GR-INT)** | `llm_guardrail` catalog + `LlmGuardrailMiddleware` + assembly/CI + E2E gate + `GUARDRAIL_BLOCKED` observability — **no** business agents | **Done** (2026-06-09) — **14/14 + M-P12.HARD** | [Phase M.12](plan/INTEGRATIONS.md) · [GR-DOC](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1an** · [ADR-GR-001](adr/entries/2026-06-09/ADR-GR-001.md) |
| **2ax — Ideal Harness L3 depth (IDEAL-L3)** | L2→L3 uplift per 32-layer audit — identity, reliability, security, cost, prompts, gates — **no** business agents | **W2 Done** (2026-06-09) — **32/32 L3** | [Phase IDEAL-L3](plan/IDEAL_HARNESS_L3.md) · **§6.1at** · Band **2ax** |
| **2az — Ideal architecture gap (AUDIT-IDEAL)** | Post-L3 audit → full IDEAL architecture — memory org, ECP sync, registry durable, L4 evidence, DX HTTP — **no** business agents unless §6.3 | **Done** (2026-06-18) — **90/90 Done** · **0 Planned** | [Phase AUDIT-IDEAL](plan/AUDIT_IDEAL_2026.md) · **§6.1au** · Band **2az** |
| **2ba — LLM developer excellence (M-LLM-X)** | ModelCatalog, routing, tokenizer preflight, ACP StepLLMRouter DX — **no** business agents | **Partial** — LC baseline Done; P2+ backlog | [Phase M-LLM-X](plan/LLM_ADAPTERS.md) · **§6.1ax** · Band **2ba** |
| **2bb — Security & Trust Planes (SEC-PLANES)** | Modular S1/S2/S3 planes, `security_defenses` EP, shipped defense bundles, encryption bridge — **no** standalone Security tier | **Done** (2026-06-19) — **17/17** | [Phase SEC-PLANES](plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes--security--trust-planes-closed) · **§6.1aw** · canon §42.45 · [ADR-SEC-001](adr/entries/2026-06-19/ADR-SEC-001.md) |
| **2bc — Security Planes enterprise hardening (SEC-PLANES-EVOL)** | Catalog bootstrap wiring, EP lab fixture, security spine signals, encrypt-via-adapter, defense inspection budget — **no** new Security tier | **Done** (2026-06-19) — **7/7** | [Phase SEC-PLANES-EVOL](plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-planes-evol--enterprise-hardening-closed) · **§6.1bc** · canon §42.45.10 |
| **2bd — Security enterprise production (SEC-ENT)** | Live SecretsStore encryptor, typed spine payloads, tenant-scope defense guard, ops counters — **no** new Security tier | **Done** (2026-06-19) — **6/6** | [Phase SEC-ENT](plan/UNIFIED_EXECUTION_RUNTIME.md#phase-sec-ent--enterprise-production-closed) · **§6.1bd** · canon §42.45.11 |
| **3 — END OF PLAN (product)** | Business agents, new product Tier-3 apps, domain skills, Legal live E2E | **Deferred** — **[§6.3](#63-end-of-plan--deferred-product-work-only)** | K.1, K.2, `applications/<product>/`, K.6, B.15, S-Ops.4 · FLOW-8 |

**Hard rule:** Band 3 is **not** “next after harness.” It runs only after an **explicit product prioritization decision** (Appendix A for agents; separate decision for new applications). Until then, **do not** implement, extend, or schedule K.1/K.2 waves, new product hosts, or product-only E2E in implementation cadence (§6.1–§6.2).

**Policy (2026-06-07):** Harness completion in §4.1 is **Done**. Band 1 = keep gate green on every PR. Bands **2j–2ad** platform closeouts = **Done**. **Band 2aj (Phase FLOW)** = **Done** (18/18 harness; FLOW-8 product **Deferred** §6.3). **Band 2ak (Phase CRIT-V)** = **Done** (24/24). Band 3 = **frozen** unless leadership reprioritizes.

```text
BAND 1:  Harness maintenance — gate + audit scripts (§6.1) — every PR
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (2026-06-06)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (2026-06-05)
BAND 2:  Harness architecture hardening — Phase V + V-REM — DONE (2026-06-05)
BAND 2i: Phase V runtime remediation — V-REM — DONE (2026-06-05)
BAND 2d: Operational L3 — Phase W-OPS (§6.2w) — DONE
BAND 2e: Application environment — Phase H-APP (§6.2x) — DONE (43 tasks)
BAND 2f: Developer authoring UX — Phase DX (§6.2y) — DONE (47 tasks)
BAND 2g: Agents & applications conformance — Phase AA (§6.2z) — MOSTLY DONE (platform); domain → Band 3
BAND 2h: Memory platform — Phase MEM (§6.2aa) — DONE (48/48)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (ORCH-1 → ORCH-4)
BAND 2k: Tools/skills closeout — Phase TS (§6.1c) — DONE (TS-1 → TS-3)
BAND 2l: Integration closeout — Phase INT (§6.1d) — DONE (INT-1 → INT-2)
BAND 2m: RAG closeout — Phase RAG (§6.1e) — DONE (RAG-1)
BAND 2n: Context engineering closeout — Phase CTX (§6.1f) — DONE (CTX-1 → CTX-2)
BAND 2o: Legacy tool plan closeout — Phase LEG (§6.1h) — DONE (LEG-1 → LEG-3)
BAND 2p: Prompt registry closeout — Phase PE (§6.1i) — DONE (PE-1 → PE-3)
BAND 2q: Agent assembly closeout — Phase AS (§6.1k) — DONE (AS-1 → AS-3)
BAND 2r: Registry architecture closeout — Phase REG (§6.1l) — DONE (REG-1 → REG-3)
BAND 2s: Capability graph closeout — Phase CG (§6.1m) — DONE (CG-1 → CG-3)
BAND 2t: Observability closeout — Phase OBS (§6.1n) — DONE (OBS-1 → OBS-3)
BAND 2u: Reliability closeout — Phase REL (§6.1o) — DONE (REL-1 → REL-3)
BAND 2v: Security closeout — Phase SEC (§6.1q) — DONE (SEC-1 → SEC-3)
BAND 2w: Cost governance closeout — Phase COST (§6.1r) — DONE (COST-1 → COST-3)
BAND 2x: Evaluation closeout — Phase EVAL (§6.1s) — DONE (EVAL-1 → EVAL-3)
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70, Wave 0–7 Done)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (39/39)
BAND 2aa: Integration expansion — Phase M.6 P4 (§6.1w) — DONE (28/28)
BAND 2ab: Integration depth — Phase M.6 P5 (§6.1x) — DONE (33/34)
BAND 2ac: Integration expansion — Phase M.6 P6 (§6.1y) — DONE (32/32 + M-P6-WIRE)
BAND 2ad: FAUDIT-32 remediation — DONE (2026-06-06)
BAND 2aj: Nexus execution depth — Phase FLOW (§6.1aj) — DONE (18/18 harness; FLOW-8 product Deferred §6.3)
BAND 2ak: Critic & Verification Layer — Phase CRIT-V (§6.1ak) — **Done** (incl. CRIT-V-FOLLOWUP)
BAND 2al: Unified Observability Spine — Phase OBS-BUS (§6.1al) — **Done**
DONE:    Phase CLEAN — legacy module closeout (§6.1j) — 2026-06-02
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

PARALLEL (harness-only): M.6 P6 integration expansion (§6.1y, **32 Done — closed 2026-06-02**); M.6 P5 residual `trivy` absorbed into P6 M-P6.1; legacy M.6 on-demand slugs; R-Skill catalog expansion (platform packs)

BAND 3 — END OF PLAN (see §6.3; not default “next”):
  • K.1 Problem Radar / K.2 Vendor Discovery (business agents)
  • K.6 / B.15 / S-Ops.4 — Legal live LLM E2E (product/CI)
  • New Tier-3 **product** applications (beyond lab + existing reference hosts)
  • Domain skill packs for product agents (until K.* started)
  • Problem Radar wave 2+ (`agents/problem_radar/` frozen)

RULE:    Strategy → canon → plan → code; Tier-1 via §0.6; four layers Integration → Tool → Skill → Agent
```

**Rationale:** Phases S/T/U + §4.1 delivered a production-configurable **harness**. Band 1–2 preserve and extend that platform. **Band 3 (product) is intentionally last** so business agents and new applications do not drive Tier-1 evolution (canon §52, [INTERGRAX_DEVELOPMENT_STRATEGY.md](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)).

### 4.0a Implementation scope split (infrastructure vs business)

**Canonical rule:** Default implementation queue = **infrastructure only** (Bands 1–2g + §6.1). **Business** work runs only after explicit product prioritization — **[§6.3](#63-end-of-plan--deferred-product-work-only)**.

**Documentation rule:** This plan and [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) document **platform** delivery (Harness / Agent OS). They do **not** subsume `applications/<product>/IMPLEMENTATION_PLAN.md` or `agents/<name>/` product roadmaps — each business environment and business agent owns its architecture and deployment narrative.

| Layer | Bands / phases | What it includes | Default queue |
|-------|----------------|------------------|---------------|
| **Infrastructure (Intergrax Harness)** | 1, 2, 2b–2j (platform rows) | `intergrax/runtime/`, Tier-0 catalogs, H-APP, DX, MEM, ORCH, scaffold, CI audits, reference hosts | **Active** — §6.1 maintenance only |
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
| **Canonical implementation queue (infrastructure)** | [§6.1](#61-harness-platform-maintenance-default--band-1) (**active** — maintenance) · [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) · [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) · [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) (all closed) · [§6.1z](#61z-harness-implementation-queue-consolidated) (closed) |
| Integration catalog expansion (Done) | [M.6 P4](#m6-p4--harness-platform-expansion-done) · [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed) — **28/28 Done** |
| Integration harness depth (Done) | [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done) — **33/34 Done** |
| Integration harness expansion | [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done) — **Done** (32/32 + wiring) |
| LLM guardrail integrations (Done) | [M.12](plan/INTEGRATIONS.md) · [§6.1an](#61an-harness-implementation-queue--llm-guardrail-integrations-closed) — **Done** (14/14 + hardening) |
| Ongoing gate + audit scripts | [§6.1](#61-harness-platform-maintenance-default--band-1) |
| Memory platform wiring (Done) | [Phase MEM](plan/MEMORY.md) · [§6.2aa](#62aa-phase-mem-execution-order-band-2h--closed) |
| **Memory intelligence depth (closed)** | [Phase MEM-DEPTH](plan/MEMORY.md) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · [§6.1am](#61am-harness-implementation-queue--memory-intelligence-depth-closed) · [§6.2ab](#62ab-phase-mem-depth-execution-order-band-2am--closed) |
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
