# Platform Foundation — Implementation Plan

**Architecture (1:1):** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

---

## Documentation model

This file is the **implementation plan hub**. Detailed phase registers live under [`plan/`](plan/). Appendices: [`plan/`](plan/).

### Documentation boundary

| Covers | Does **not** cover |
|--------|---------------------|
| Harness AI platform, Nexus Agent OS, Tier-0 catalogs, reference hosts, §6.1 maintenance, T-EXPAND tool waves | Architecture, roadmap, or deployment plan of a **specific business environment** (`applications/<product>/`) |
| How Tier-2 agents and Tier-3 apps **plug into** the Harness | Architecture, roadmap, or deployment plan of a **specific business agent** (`agents/<name>/`) |

Each **business environment** and each **business agent** maintains its own `ARCHITECTURE.md`, local implementation plan, and product roadmap. See [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) and [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) §1.1.

| Topic | Where |
|-------|--------|
| Strategic goal, decision hierarchy, work cycle | [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md) |
| Full architecture specification | [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) (hub) · [`architecture/`](architecture/) |
| Phase status, gaps, priority | **This file** — **§4** ladder; **§4.0a** infrastructure vs business; **§6.3** / **§6.3a** = product work only |
| Tier-0 integration catalog (what / where) | [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) + [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) |
| Tier-0 integration implementation (how) | **This file** Phase M |
| Tier-0 tool catalog (what / where) | [`architecture/TOOLS.md`](architecture/TOOLS.md) + [`architecture/TOOLS.md`](architecture/TOOLS.md) |
| Tier-0 tool implementation (how) | **This file** Phase O |
| Agent creation workflow | `guides/AGENT_CREATION_GUIDE.md` |
| Governance / policy / observability control plane (authoring) | `guides/AGENT_CREATION_GUIDE.md` **Appendix H** · [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11 · `guides/EXTENSION_AUTHOR_GUIDE.md` §10 (`intergrax.policy_rules`) |
| Orchestration / graph / delegation control plane (authoring) | `guides/AGENT_CREATION_GUIDE.md` **Appendix I** · [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) · R-Delegate **Done** · closeout [Phase ORCH](plan/ORCHESTRATION.md) |
| **Nexus execution flow (runtime narrative, diagrams, gap → plan rows)** | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) · [Phase FLOW](plan/ORCHESTRATION.md) · **§6.1aj** · Band **2aj** · **Appendix N (FLOW)** · [ADR-FLOW-001](adr/ADR-FLOW-001.md) |
| Governance audit closeout (docs + residuals register) | [Phase GOV-AUDIT](plan/UNIFIED_EXECUTION_RUNTIME.md) · **GOV-DOC.\*** **Done** |
| Orchestration audit closeout (runtime wiring) | [Phase ORCH](plan/ORCHESTRATION.md) · **§6.1b** · Band **2j** |
| Tools / skills audit closeout (runtime bridge) | [Phase TS](plan/TOOLS.md) · **§6.1c** · Band **2k** · `guides/AGENT_CREATION_GUIDE.md` **Appendix J** |
| Integration audit closeout (runtime bridge + health) | [Phase INT](plan/INTEGRATIONS.md) · **§6.1d** · Band **2l** · **Appendix K** |
| RAG audit closeout (runtime bridge) | [Phase RAG](plan/MEMORY.md) · **§6.1e** · Band **2m** · **Appendix K** §K.5 |
| Context engineering closeout (runtime + Nexus wiring) | [Phase CTX](plan/MEMORY.md) · **§6.1f** · Band **2n** · **Appendix L** |
| Prompt registry closeout (runtime + environment wiring) | [Phase PE](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1i** · Band **2p** · **Appendix M** |
| Legacy module closeout (chat_router, tools_agent, chains) | [Phase CLEAN](plan/ORCHESTRATION.md) · **§6.1j** |
| Agent assembly closeout (contracts, capabilities, lifecycle) | [Phase AS](plan/ORCHESTRATION.md) · **§6.1k** · Band **2q** · **Appendix N** |
| Registry architecture closeout (snapshots, conformance, CI) | [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1l** · Band **2r** · **Appendix O** |
| Capability graph closeout (environment slice, blast-radius wire) | [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1m** · Band **2s** · **Appendix P** |
| Observability closeout (profile bridge, assembly resolver, CI) | [Phase OBS](plan/OBSERVABILITY.md) · **§6.1n** · Band **2t** · **Appendix Q** |
| **Unified Observability Spine (full mechanism)** | [Phase OBS-BUS](plan/OBSERVABILITY.md) · **§6.1al** · Band **2al** · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · [ADR-OBS-001](adr/ADR-OBS-001.md) |
| Reliability closeout (idempotency bridge, circuit breaker, CI) | [Phase REL](plan/OBSERVABILITY.md) · **§6.1o** · Band **2u** · **Appendix R** |
| Security closeout (V-SEC bridge, middleware assembly, CI) | [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1q** · Band **2v** · **Appendix S** |
| Cost governance closeout (budget bridge, policy bundle, CI) | [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1r** · Band **2w** · **Appendix T** |
| Evaluation closeout (registry bridge, policy bundle, CI) | [Phase EVAL](plan/CRITIC_VERIFICATION.md) · **§6.1s** · Band **2x** · **Appendix U** |
| **Critic & Verification Layer (PEV verify depth)** | [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · **§6.1ak** · Band **2ak** · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/ADR-CRITIC-001.md) |
| **Adaptive Harness Intelligence (AHI / L4 runtime)** | [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · **§6.1t** · Band **2y** · [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · **Appendix K** |
| **LLM response envelope (typed completion contract)** | [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · **§6.1v** · Band **2z** · [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) · **Appendix L** |
| **Integration catalog expansion (harness ROI slugs)** | [M.6 P4 register](#m6-p4--harness-platform-expansion-done) · **§6.1w** · Band **2aa** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **Integration harness depth (audit 2026-06-02)** | [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) · **§6.1x** · Band **2ab** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **Integration harness expansion (audit 2026-06-02)** | [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · **§6.1y** · Band **2ac** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **LLM guardrail integrations (M.12 / GR-INT)** | [Phase M.12](plan/INTEGRATIONS.md) · **§6.1an** · Band **2ay** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) §47 · [plan/UNIFIED_EXECUTION_RUNTIME.md](plan/UNIFIED_EXECUTION_RUNTIME.md) GR-DOC |
| Tier-3 application environment (self-contained deploy) | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) |
| Tier-3 composition engine (manifest, wiring API) | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| Tier-3 application hosts (`applications/<app>/`) | [`applications/USAGE.md`](../applications/USAGE.md) |
| Application scaffold & deploy plan | **This file** Phase N |
| Business-agent go/no-go checklist | **[Appendix A](plan/PLATFORM_FOUNDATION.md)** |
| Technical debt backlog (analysis only) | **[Appendix B](plan/PLATFORM_FOUNDATION.md)** |
| Harness quality audit (2026-06-01) → Phase Q tracker | **This file** Phase Q + **Appendix C** |
| Post-audit hardening (typing, legacy, monoliths) | **This file** Phase Q+ + **Appendix D** |
| Harness GA / consolidation (no new OS features) | **This file** Phase Q / Q+ |
| Harness AI alignment audit (2026-06-01) → Phase R | **This file** Phase R + **Appendix E** + canon [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) §5.3 |
| Harness environment GA (2026-06-01) → Phase S | **This file** Phase S + **Appendix F** (K.1/K.2 → §6.3 end-of-plan) |
| Harness production hardening (2026-06-01 audit) → Phase U | **This file** Phase U + **Appendix G** (**Done**; does **not** schedule K.1/K.2 — see §6.3) |
| Skill / Tool / Integration layering (canon) | [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) §5.3 · [`architecture/TOOLS.md`](architecture/TOOLS.md) · [`architecture/SKILLS.md`](architecture/SKILLS.md) |
| Skill catalog | `architecture/SKILLS.md` |
| Model & modality plane (vision, audio, ML) | [`architecture/MODALITY.md`](architecture/MODALITY.md) · [`architecture/MODALITY.md`](architecture/MODALITY.md) · **Phase W-ML** (below) |
| Plugin catalogs (integrations, tools, skills) | **This file** Phase P-Ext + **Appendix I** · [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md) |
| Harness maturity audit (2026-06-02) → operational L3 | **Phase W-OPS** (below) · **§6.2w** · source: maturity audit 2026-06-02 (conversation) |
| Tier-3 application environment audit → full configurability | [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) → **Phase H-APP** · **§6.2x** |
| Developer authoring UX audit (LangGraph-like entry, measurable TTFRun) | **Phase DX** (below) · **§6.2y** · source: harness DX audit 2026-06-03 (conversation + H-APP gap analysis) |
| Agents & applications conformance audit (structure, scaffold, per-agent/app docs, deploy) | **Phase AA** (below) · **§6.2z** · source: Tier-2/Tier-3 audit 2026-06-03 (conversation) |
| Memory platform audit (STM/LTM/org/task/context/hooks/persistence) | **Phase MEM** (below) · **§6.2aa** · **§6.1aa** · source: memory audit 2026-06-02 (conversation) |
| **Memory intelligence depth (context compiler, lifecycle, explore)** | [`architecture/MEMORY.md`](architecture/MEMORY.md) · [Phase MEM-DEPTH](plan/MEMORY.md) · **§6.2ab** · **§6.1am** · source: memory audit 2026-06-08 |
| Phase V runtime remediation (2026-06-05 audit) → close Partial gaps | **Phase V-REM** (below) · **Appendix J** · **§6.1z** · **§6.2v** · source: plan/code audit vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` |
| Phase V remediation traceability (audit gap → V-REM ID) | **[Appendix J](plan/PLATFORM_FOUNDATION.md)** |
| Full architecture audit procedure (32 layers) | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · prompt: [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) |
| **Full architecture audit closeout (32 layers, scope C)** | [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md) · **§6.1ah** · Band **2ad** · **Appendix M** · source: audit 2026-06-06 (`scope: C`, `audit-and-fix`) |
| Infrastructure vs business scope split | **§4.0a** · [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) (closed) · [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) (closed) · [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) (closed) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) (closed) · [§6.1g](#61g-harness-implementation-queue--governance-audit-closed) (closed) · [§6.3a](#63a-business-backlog-register-consolidated) |

**Note on audit source documents:** Some historical audit narratives (e.g. `HARNESS_APPLICATION_LAYER_AUDIT.md`) may live outside the repo. **Task traceability in this plan is canonical** — H-APP (43 tasks), W-OPS, MEM, DX, AA registers below; do not re-derive scope from missing files.

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

**Harness production hardening (Phase U):** **Done** (2026-06-01) — auth surfaces, strict harness profile, `HarnessReferenceAgent`, typed policy bundle, planner decoupling, sandbox opt-in. **U-Leg** legacy removal remains tracked in Appendix G. See Phase U + **Appendix G**.

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
| Regression gate | **906 passed** | No | Must stay green after each harness PR (Phase FLOW closeout 2026-06-07) |
| **Full architecture audit (FAUDIT-32)** | **Done** (2026-06-06) | No (harness-only) | 32-layer audit + **23/23 remediation** → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) |
| **Nexus execution depth (Phase FLOW)** | **Done** (18/18 harness) | No (harness-only) | Band **2aj** — [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · FLOW-8 harness **Done**; product host **Deferred** §6.3 · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) |
| **Critic & Verification Layer (Phase CRIT-V)** | **Done** (24/24) | No (harness-only) | Band **2ak** — [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) |
| **Unified Observability Spine (Phase OBS-BUS)** | **Done** (8/8) | No (harness-only) | Band **2al** — [§6.1al](#61al-harness-implementation-queue--unified-observability-spine-closed) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · [ADR-OBS-001](adr/ADR-OBS-001.md) |

---



## 1. Plan Objective



Transform Intergrax into an **internal agent experimentation laboratory** (§2, §35) aligned with the canonical architecture:



```text

hypothesis → capability → contract → registration → Nexus → trace → evaluation → decision

```



**Success metric:** time from idea to first running experiment **< 1 hour**.

**Capability model:** Integration → Tool → **Skill** → Agent (Harness AI alignment). Skill Library **MVP Done** — see §0, Phase R, Appendix E, architecture §7.1.8, [architecture/SKILLS.md](architecture/SKILLS.md).



**Current alignment** (synced with §0.5, 2026-06-05):

| Scope | Score | Notes |
|-------|-------|-------|
| Architecture §1–41 (tiers, Nexus, graph, repo split) | **~98%** | Phases A–O, N, Q+ complete |
| §42 Unified Execution Runtime | **~99%** | UAEP Protocol, hook parity, planner split (`step_planner/`), harness getattr audit |
| Laboratory workflow (inspect, decide) | **~99%** | D.1–D.5, metrics parity, debug API |
| Harness quality (Phase Q) | **Done** | Appendix C — gate **417** at Phase Q close |
| Harness hardening (Phase Q+) | **Done** | Appendix D — typing, monolith splits, zero grandfathered `getattr` |
| Harness AI alignment (Phase R MVP) | **Done** | Appendix E — Skill Library, context, delegation, policy bundle |
| Regression gate | **608 passed** | `pytest -m gate`; also `scripts/check_harness_no_getattr.py` |
| Harness environment GA (Phase S) | **Done** (2026-06-01) | S-H.* + S-Ops + S-Doc |
| Harness cleanliness (Phase T) | **Done** (2026-06-01) | T-Ops + T-H |
| Harness production hardening (Phase U) | **Done** | Appendix G audit → U.* (U-Leg residual) |
| Harness architecture hardening (Phase V) | **Done** | Phase V-REM runtime enforcement complete (2026-06-05) |
| Phase V runtime remediation (V-REM) | **Done** | 10/10 tasks — capability graph, lifecycle routing, prompt governance, V-SEC wiring, EvalRunner gate |
| Model & modality plane (Phase W-ML) | **Done** (2026-06-02) | Vision/speech profiles, tools, remote adapters, `harness.vision_qa` — canon §7.1.9 |
| **Harness completion backlog** | **Done** (2026-06-02) | §4.1 — U-Leg, typing/CI, platform skills, research UAEP parity |
| **Plugin catalogs (Phase P-Ext)** | **Done** (2026-06-02) | [Phase P-Ext](#phase-p-ext--plugin-catalogs-integrations-tools-skills) · [P-Ext.6 paydown](#p-ext6--production-closure-paydown) · Appendix I |
| **Application environment (Phase H-APP)** | **Done** (2026-06-03) | [Phase H-APP](plan/TIER3_APPLICATION_ENVIRONMENT.md) · 43 tasks from application-layer audit |
| **Developer authoring UX (Phase DX)** | **Done** (2026-06-02) | [Phase DX](plan/TIER3_APPLICATION_ENVIRONMENT.md) · **47/47 Done** — [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **Agents & applications conformance (Phase AA)** | **Platform Done** (2026-06-02) | [Phase AA](plan/TIER3_APPLICATION_ENVIRONMENT.md) · platform **Done**; domain **Deferred** — [§6.3a](#63a-business-backlog-register-consolidated) |
| **Memory platform (Phase MEM)** | **Done** (2026-06-02) | [Phase MEM](plan/MEMORY.md) · **48/48** |
| **Governance audit (GOV-AUDIT)** | **Done** (docs) | [Phase GOV-AUDIT](plan/UNIFIED_EXECUTION_RUNTIME.md) |
| **Orchestration closeout (Phase ORCH)** | **Done** (2026-06-05) | [Phase ORCH](plan/ORCHESTRATION.md) · [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) |
| **Tools/skills closeout (Phase TS)** | **Done** (2026-06-02) | [Phase TS](plan/TOOLS.md) · [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) |
| **Integration closeout (Phase INT)** | **Done** (2026-06-02) | [Phase INT](plan/INTEGRATIONS.md) · [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) |
| **RAG closeout (Phase RAG)** | **Done** (2026-06-02) | [Phase RAG](plan/MEMORY.md) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) |
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

Historical phase registers (A–V) and closeout phases are decomposed under [`plan/`](plan/).

| Domain | File |
|--------|------|
| Historical A–V | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) |
| Core runtime | [`plan/ORCHESTRATION.md`](plan/ORCHESTRATION.md) |
| Integrations | [`plan/INTEGRATIONS.md`](plan/INTEGRATIONS.md) |
| Tools & skills | [`plan/TOOLS.md`](plan/TOOLS.md) |
| LLM & modality | [`plan/LLM_ADAPTERS.md`](plan/LLM_ADAPTERS.md) |
| RAG, context, memory | [`plan/MEMORY.md`](plan/MEMORY.md) |
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
| **2m — RAG closeout (RAG)** | `rag_runtime_bridge`, RAG stack on environment wire — **no** business agents | **Done** (2026-06-02) | [Phase RAG](plan/MEMORY.md) · **§6.1e** · **§6.2be** |
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
| **2ad — FAUDIT-32 remediation** | Close 32-layer audit residuals (tier gate, intake, observability taxonomy, registry depth, eval release gate) — **no** business agents | **Done** (2026-06-06) — **23/23 + §6.1ai follow-up** | [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md) · **§6.1ah** · **§6.1ai** · **Appendix M** |
| **2aj — Nexus execution depth (FLOW)** | Close `FLOW-GAP.*` (01–16) — delegation, SubtaskContract, backpressure profile, LLM planner, merge, eval, graph hardening — **no** K.1/K.2 | **Done** (2026-06-07) — **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3) | [Phase FLOW](plan/ORCHESTRATION.md) · **§6.1aj** · **§6.2aj** · **Appendix N (FLOW)** |
| **2ak — Critic & Verification Layer (CRIT-V)** | PEV verify depth — `CriticOrchestrator`, `eval.judge`, `eval.trajectory`, evaluator-loop, semantic offline runner — **no** business agents | **Done** | [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · **§6.1ak** · **§6.2ak** · canon §55 · [ADR-CRITIC-001](adr/ADR-CRITIC-001.md) |
| **2al — Unified Observability Spine (OBS-BUS)** | Full HOS — typed payloads, `ObservabilityEmitter`, emission coverage, extension SDK, L4 §21 — **no** business agents | **Done** | [Phase OBS-BUS](plan/OBSERVABILITY.md) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **§6.1al** · [ADR-OBS-001](adr/ADR-OBS-001.md) |
| **2am — Memory intelligence depth (MEM-DEPTH)** | Context Compiler, never-overflow invariant, lifecycle automation, explore delegation, entity memory — **no** business agents | **Done** (2026-06-08) — **26/26** | [Phase MEM-DEPTH](plan/MEMORY.md) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · **§6.2ab** |
| **2an — Elastic capacity domain pair (ECP-DOC)** | ECP canon + ADR — docs only | **Done** (2026-06-08) | [Phase ECP-DOC](plan/ELASTIC_CAPACITY_AND_SCALING.md) · Band **2an** |
| **2ao — Elastic capacity runtime (ECP-DEPTH)** | ScalingProfile, signal collector, evaluator, K8s provisioner, policy gates — **no** business agents | **Done** (2026-06-09) — **28/28** | [Phase ECP-DEPTH](plan/ELASTIC_CAPACITY_AND_SCALING.md) · [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) |
| **2ap — Orchestration strategy canon (ORCH-STRAT)** | §50–§54 coordination/resilience docs | **Done** (2026-06-09) | [Phase ORCH-STRAT](plan/ORCHESTRATION.md) |
| **2ar — Platform interaction config (ORCH-CONFIG)** | CFG-* harness simulation + reference host presets — **no** business agents | **Done** (2026-06-09) — **11/11** | [Phase ORCH-CONFIG](plan/ORCHESTRATION.md) · [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §56–§59 |
| **2as — Reasoning layer depth (COG-DEPTH)** | Planner unification, Prompt Registry on planners, DecisionRecord, failure taxonomy — **no** business agents | **Done** (2026-06-09) — **22/22** | [Phase COG-DEPTH](plan/REASONING_AND_COGNITION.md) · [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) |
| **2aw — Tier-3 execution surface parity (H-APP-WIRING)** | Close FLOW-GAP-17–20 / ORCH §59 Tier-3 wiring debt — task control API, async exposure, reference host adoption — **no** Nexus fork | **Done** (2026-06-09) — **6/6** | [Phase H-APP-WIRING](plan/TIER3_APPLICATION_ENVIRONMENT.md) · [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §59 |
| **2ay — LLM guardrail integrations (M.12 / GR-INT)** | `llm_guardrail` catalog + `LlmGuardrailMiddleware` + assembly/CI + E2E gate + `GUARDRAIL_BLOCKED` observability — **no** business agents | **Done** (2026-06-09) — **14/14 + M-P12.HARD** | [Phase M.12](plan/INTEGRATIONS.md) · [GR-DOC](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1an** · [ADR-GR-001](adr/ADR-GR-001.md) |
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

PARALLEL (harness-only): M.6 P6 integration expansion (§6.1y, **32 planned**); M.6 P5 residual `trivy` absorbed into P6 M-P6.1; legacy M.6 on-demand slugs; R-Skill catalog expansion (platform packs)

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
| Integration harness expansion | [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned) — **Done** (32/32 + wiring) |
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



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---



**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `guides/HARNESS_ENVIRONMENT.md`, [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`guides/EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision only — confirmed 2026-06-09; harness path = `observability_backend` + OBS-BUS |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---



**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done.

**Post depth bands (2026-06-09):** MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG closeout complete — Appendix M scorecard refreshed; remaining L2 rows are incremental depth only.

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L3 | 0 | 0 | Yes |
| 3 | Interface and Task Intake | L3 | 0 | 1 | Partial |
| 4 | Identity, Trust and Tenancy | L2 | 0 | 2 | Partial |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L3 | 0 | 0 | Yes |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 0 | Yes |
| 10 | Subagents and Multi-Agent Coordination | L2 | 0 | 2 | Partial |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L3 | 0 | 0 | Yes |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L2 | 0 | 1 | **No** |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L2 | 0 | 2 | **No** |
| 21 | Observability and Telemetry | L2 | 0 | 2 | **No** |
| 22 | Error Handling and Reliability | L2 | 0 | 1 | **No** |
| 23 | Security and Data Governance | L2 | 0 | 2 | **No** |
| 24 | Cost and Resource Governance | L2 | 0 | 1 | **No** |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 1 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 1 | Partial |
| 31 | Agent Lifecycle Governance | L2 | 0 | 2 | Partial |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax/` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax/` → `applications/` imports | High | `scripts/check_intergrax_no_applications_imports.py` in §6.1 |
| FAUDIT-INTAKE.1 | §3 | No canonical `TaskEnvelope`; `Task` + `RuntimeRequest` split | High | Typed envelope alias or consolidation; plan W-OPS.6 naming sync |
| FAUDIT-INTAKE.2 | §3 | Worker≡HTTP intake parity test matrix incomplete | High | Acceptance test: CLI/worker/HTTP same `Task` shape |
| FAUDIT-ID.1 | §4 | No user/service/agent identity distinction | High | Identity contracts + propagation to delegation |
| FAUDIT-ID.2 | §4 | `DelegationSpec` lacks permission scope audit | High | Scope field + trace on child runs |
| FAUDIT-POL.1 | §5 | No pre-LLM / pre-output policy hooks in runtime | High | PolicyEngine extension points documented + wired |
| FAUDIT-LLM.1 | §6 | No policy-driven model routing / fallback chain | High | Router module or AdaptiveProfile integration |
| FAUDIT-COG.1 | §7 | No universal `DecisionRecord` per UAEP step | High | Typed decision artifact + trace event |
| FAUDIT-ORCH.1 | §9 | No graph backpressure beyond parallel cap | High | Queue depth / shed policy in `GraphExecutor` |
| FAUDIT-SUB.1 | §10 | No formal `SubtaskContract`; `inherit_tool_policy=True` default | High | Contract type + safer delegation defaults |
| FAUDIT-MEM.1 | §15 | Entity graph memory absent; STM retention partial | High | Route to MEM-9.* / new MEM row if scoped |
| FAUDIT-PE.1 | §17 | No golden prompt content regression in CI | High | Golden YAML fixtures + gate test |
| FAUDIT-REG.1 | §19 | `HarnessRegistrySnapshot` omits agents + eval registry | High | Extend snapshot + assembly resolver |
| FAUDIT-CG.1 | §20 | Capability graph seed skips `prompt:*` nodes | High | `_seed_node_ids()` parity |
| FAUDIT-CG.2 | §20 | Blast-radius not enforced at release | High | `phase_v_capability_graph_guard.py` impact check |
| FAUDIT-OBS.1 | §21 | `RuntimeEventType` missing `LLM_CALL` / `POLICY_DECISION` | High | Canon event catalog + bridge from trace |
| FAUDIT-REL.1 | §22 | Shallow error taxonomy (`VALIDATION_ERROR` only + 2) | High | Expand classifier per AUDIT_MAP §22 |
| FAUDIT-SEC.1 | §23 | No `DataClassification` model | High | Security profile + enforcement hooks |
| FAUDIT-COST.1 | §24 | Per-tenant cost attribution not mandatory in NexusLoop | High | Budget gate on main path |
| FAUDIT-EVAL.1 | §25 | `require_baseline_for_release` not CI-enforced | High | `phase_v_closeout_gate.py` eval baseline check |
| FAUDIT-ALG.1 | §31 | Lifecycle states ≠ AUDIT_MAP catalog; weak agent adoption | High | Align or document ADR; scaffold defaults |
| FAUDIT-OPS.1 | §30 | `release_cycles.json` not in repo; W-OPS.5 artifact policy unclear | High | Document committed vs generated artifact |

**Delivery rule:** One **FAUDIT.\*** ID per PR → update §6.1ah + Appendix M paydown log → gate green.

**Explicitly out of scope (audit-and-fix):** source code or test changes during this audit pass; K.1/K.2; new product Tier-3 apps.

---



**Status:** **Done** (2026-06-05) — **6/6** deliverables Done (ORCH-DOC.* + ORCH-1–4); gate **581 passed**  
**Prerequisites:** R-Delegate **Done**, Q+-N.* runners **Done**, H-APP.3.1–3.2 **Done**, V-MA.* **Done**  
**Goal:** Close orchestration audit residuals (AUDIT_MAP §7–§10) — wire declared Tier-3 profile fields to runtime; bridge declarative graph spec to execution plan; cap graph batch concurrency.  
**Priority ladder:** **Band 2j** (§4.0) — **default implementation queue** after §6.1 gate on each PR.  
**Execution order:** [§6.2bb](#62bb-phase-orch-execution-order-band-2j--active) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-active)  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane)

**Delivery rule:** One **ORCH-*** ID per PR → update master table + §6.1b + paydown log below → `pytest -m gate` + §6.1 scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

| Area | Maturity (L0–L4) | Residual before ORCH | Close via |
|------|------------------|----------------------|-----------|
| Nexus stack (§8) | **L3–L4** | — | ORCH-DOC.* (documented) |
| Planning strategies (§7) | **L3–L4** | — | ORCH-1 **Done** |
| Declarative graph (§9) | **L3–L4** | — | ORCH-2 **Done** |
| Graph concurrency (§9) | **L3** | — | ORCH-3 **Done** |
| Subagent delegation (§10) | **L3–L4** | — | R-Delegate (Done) |

### ORCH — Master register

| ID | Wave | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-------------|--------|----------|---------------|------------|
| ORCH-DOC.1 | ORCH0 | **Appendix I** — orchestration control plane map (§I.1–I.10) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| ORCH-DOC.2 | ORCH0 | **Cross-ref sync** — plan, README, strategy, AUDIT_MAP §7–§10, audit prompt ref #6, canon §42.43 | **Done** | Medium | `docs/*` | Links resolve |
| ORCH-1 | ORCH1 | **Wire `planner_kind` / `classifier_kind`** — registry maps kinds → `TaskPlanner` / `ClassifyingTaskClassifier`; `build_nexus_loop_from_environment` passes resolved instances to `NexusLoop` | **Done** | **Critical** | `orchestration_wiring.py`, `nexus_factory.py` | `test_orchestration_wiring.py` |
| ORCH-2 | ORCH2 | **`ApplicationGraphSpec` → `NexusPlan` seed** — `graph_spec_to_plan.py` + `GraphSpecSeedingPlanner` when task has no plan id | **Done** | **High** | `graph_spec_to_plan.py`, `PlanStep.delegation` | `test_graph_spec_to_plan.py`, `test_lab_graph_spec.py` |
| ORCH-3 | ORCH3 | **`max_parallel_nodes` on `OrchestrationProfile`** — cap concurrent nodes per graph batch in `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `graph_executor.py` | `test_graph_executor_parallel_cap.py` |
| ORCH-4 | ORCH4 | **Docs closeout** — Appendix I + plan sync | **Done** | Low | `docs/*` | No “planned wiring” residuals |

**Supported `planner_kind` values (ORCH-1 contract):**

| Kind | Implementation | Notes |
|------|----------------|-------|
| `null` / `default` | `TaskPlanner()` | Current harness default |
| `engine` | `EnginePlanner` adapter implementing plan contract | Requires `RuntimeConfig` on build context — lab/legal hosts only in v1 |
| Unknown kind | — | **Fail fast** at Nexus bootstrap with typed error (no silent fallback) |

**Supported `classifier_kind` values (ORCH-1 contract):**

| Kind | Implementation |
|------|----------------|
| `null` / `default` | `ClassifyingTaskClassifier(registry)` |

**Explicitly out of scope:** Nested full harness per child (use R-Delegate); new graph node types (Tier-1 canon change); product-specific orchestration in `agents/`.

### ORCH — Paydown log

| Date | ORCH ID | Summary |
|------|---------|---------|
| 2026-06-05 | ORCH-DOC.1, ORCH-DOC.2 | Governance + orchestration audit docs; Appendix H/I; AUDIT_MAP cross-refs |
| 2026-06-05 | ORCH-1, ORCH-2, ORCH-3 | Orchestration wiring, graph spec plan seed, parallel cap; gate **581** |
| 2026-06-05 | ORCH-4 | Plan + author guide closeout |

**Phase ORCH complete when:** ORCH-1–4 **Done**; §6.1b queue closed; Appendix I has no “planned wiring” gaps; gate **581** green. **Status: complete (2026-06-05).**

---



**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
**Execution order:** [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--active) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)  
**Traceability:** **Appendix N (FLOW)** — [`§Appendix N`](#appendix-n--nexus-execution-flow-traceability-phase-flow)

**Delivery rule:** One **FLOW-*** ID per PR → update master table + §6.1aj + Appendix N paydown → `pytest -m gate` + §6.1 scripts green.

**Maturity target (phase complete):**

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target after FLOW |
|-------------|----------------------|-------------------|
| §5 Policy (pre-plan hooks) | L2 partial | **L3** (FLOW-11) |
| §7 Reasoning / planning | L2 | **L3+** (FLOW-1, FLOW-12, COG-DEPTH) |
| §8 Execution runtime | L3 | **L3** (FLOW-10, maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** (FLOW-4–7, FLOW-6, FLOW-13, FLOW-16) |
| §10 Subagents | L2 | **L3** (FLOW-2, FLOW-3, FLOW-14, FLOW-15) |
| §25 Evaluation | L2 | **L3** (FLOW-9) |

**Explicitly out of scope:** Nested full harness per child; new graph node **types** (Tier-1 canon change); K.1/K.2 business agents (FLOW-8 → §6.3 unless reprioritized).

### FLOW — Master register

| ID | Wave | Gap | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-----|-------------|--------|----------|---------------|------------|
| FLOW-DOC.1 | FLOW0 | — | **Flow reference sync** — paydown §23 gaps in `architecture/NEXUS_EXECUTION_FLOW.md` after each FLOW PR | **Done** | Low | `docs/architecture/NEXUS_EXECUTION_FLOW.md` | No stale `FLOW-GAP` rows for Done IDs |
| FLOW-2 | FLOW1 | FLOW-GAP-02 | **ADR-FLOW-001 implementation** — expand `DELEGATES_TO` to child `PlanStep` + `ExecutionNode`; `DelegationSpec` on **child**; `GraphExecutor` routes `child_agent_id` | **Done** | **Critical** | `graph_spec_to_plan.py`, `graph_builder.py`, `graph_executor.py` | `test_graph_spec_to_plan.py` + integration delegation path; canon §42.14.3 note updated |
| FLOW-3 | FLOW1 | FLOW-GAP-03 | **`max_delegation_depth` enforcement** — count expanded delegation chain in `GraphExecutor`; fail with trace | **Done** | High | `graph_executor.py`, `environment_profile.py` | Unit test depth exceeded |
| FLOW-1 | FLOW2 | FLOW-GAP-01 | **Real `EngineBackedNexusPlanner`** — bridge `engine_planner_orchestrator` → `NexusTaskPlannerProtocol`; typed `NexusPlan` from LLM parse | **Done** | High | `orchestration_wiring.py`, `planning/engine_planner_*.py` | `test_orchestration_wiring.py` + planner integration tests |
| FLOW-6 | FLOW2 | FLOW-GAP-06 | **Strict cycle detection** — `ExecutionGraph.batches()` raises on cycle; no unsafe fallback | **Done** | High | `execution_graph.py` | Unit test cyclic graph → error |
| FLOW-4 | FLOW3 | FLOW-GAP-04 | **Opt-in run-level retry** — `OrchestrationProfile.max_run_retries`; wire `RetryCoordinator` in `NexusGraphRunner` | **Done** | Medium | `environment_profile.py`, `graph_runner.py`, `nexus_factory.py` | Integration test graph retry once |
| FLOW-7 | FLOW3 | FLOW-GAP-07 | **`MergePolicy` / `FinalResponseComposerProfile`** — deterministic + structured merge; optional LLM merge hook (policy-gated) | **Done** | Medium | `final_response_composer.py`, `environment_profile.py` | Multi-agent merge unit tests |
| FLOW-9 | FLOW3 | FLOW-GAP-11 | **Evaluation hooks on multi-agent fan-in** — post-graph eval observation; evaluator-node cookbook; registry write on multi-node runs | **Done** | Medium | `nexus_loop.py`, `evaluation_wiring.py`, docs §18 | `EvaluationProfile` observation recorded; guide §18 |
| FLOW-11 | FLOW3 | FLOW-GAP-09 | **Pre-plan / pre-LLM policy extension points** — document + wire hooks at planning boundary | **Done** | Medium | `planning_runner.py`, `policy_engine.py` | Hook tests + Appendix H cross-ref |
| FLOW-5 | FLOW4 | FLOW-GAP-05 | **`AgentGraph.on_error(retry)`** — wire to `RetryPolicy` / graph executor | **Done** | Low | `graph_builder.py`, `orchestration_wiring.py` | Integration test declared retry |
| FLOW-10 | FLOW4 | FLOW-GAP-08 | **Reserved lifecycle states** — ADR: implement `WAITING_FOR_RESOURCES`/`EXPIRED` **or** trim enum + canon sync | **Done** | Low | `task_lifecycle.py`, `adr/ADR-FLOW-002.md` | [ADR-FLOW-002](adr/ADR-FLOW-002.md) accepted; reserved v1 semantics |
| FLOW-12 | FLOW4 | §24 / FAUDIT-COG | **`DecisionRecord` regression gate** — verify FAUDIT-COG.1 emit on every UAEP decision path; gate test; sync flow §24 | **Done** | Medium | `uaep.py`, `tests/integration/agents/` | `DECISION_EMITTED` + `decision_record` on each step decision |
| FLOW-13 | FLOW4 | FLOW-GAP-12 | **`max_inflight_nodes` profile + wire** — field on `OrchestrationProfile`; `resolve_max_inflight_nodes()`; `nexus_factory` → `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `orchestration_wiring.py`, `nexus_factory.py` | `GRAPH_BACKPRESSURE` event when cap hit; profile round-trip test |
| FLOW-14 | FLOW4 | FLOW-GAP-13 | **`SubtaskContract` in delegation expansion** — `graph_spec_to_plan` / ADR-FLOW-001 child node uses `SubtaskContract.to_delegation_spec()` (`objective`, `permission_scopes`, `inherit_tool_policy=False`) | **Done** | Medium | `graph_spec_to_plan.py`, `subtask_contract.py` | Unit test scopes + objective on child `DelegationSpec` |
| FLOW-15 | FLOW4 | FLOW-GAP-14 | **Subagent budget envelope** — optional `budget_envelope` on `SubtaskContract` / `DelegationSpec`; enforce in child `GraphExecutor` run via existing budget bridge | **Done** | Medium | `subtask_contract.py`, `delegation.py`, `graph_executor.py` | Child run exceeds envelope → fail with trace |
| FLOW-16 | FLOW4 | FLOW-GAP-15 | **`MODIFY_PLAN` ADR** — [ADR-FLOW-003](adr/ADR-FLOW-003.md): document reserved semantics (policy-gated replan hook) **or** trim `AgentDecision` enum | **Done** | Low | `adr/ADR-FLOW-003.md`, `interrupts/handler.py` | ADR accepted; `MODIFY_PLAN_NOT_SUPPORTED` when no handoff |
| FLOW-17 | FLOW4 | FLOW-GAP-16 | **`MULTI_AGENT` ordering policy** — `OrchestrationProfile.multi_agent_order` (`registry` \| `priority` \| `stable_alpha`); deterministic step order in `TaskPlanner` | **Done** | Low | `environment_profile.py`, `task_planner.py` | Gate test: two agents same capability → stable declared order |
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
| FLOW-DOC.2 | FLOW5 | — | **Phase closeout** — Appendix N (FLOW), flow reference §23 paydown (all gaps), maturity dashboard §0.5 | **Done** | Low | `docs/*` | All non-deferred FLOW rows **Done**; zero open `FLOW-GAP` in §23 |

### FLOW — Suggested PR order

```text
FLOW-2 → FLOW-14 → FLOW-3 → FLOW-15 → FLOW-6 → FLOW-1 → FLOW-4 → FLOW-13 → FLOW-7 → FLOW-9 → FLOW-11 → FLOW-5 → FLOW-10 → FLOW-12 → FLOW-16 → FLOW-17 → FLOW-DOC.*
```

**Parallel OK after FLOW-2:** FLOW-1, FLOW-6, FLOW-13 (disjoint modules). **FLOW-14** same PR as FLOW-2 or immediately after.

**FLOW-8:** Schedule only after explicit product decision ([§6.3](#63-end-of-plan--deferred-product-work-only)).

### FLOW — Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled from `architecture/NEXUS_EXECUTION_FLOW.md` §25; queue §6.1aj; Appendix N (FLOW) |
| 2026-06-07 | — | Audit gap closeout: FLOW-13–17 + FLOW-GAP-12–16 added; FLOW-12 narrowed to regression gate; **0/18** |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3) |

**Phase FLOW complete when:** FLOW-1–7, FLOW-9, FLOW-11–17, FLOW-DOC.* **Done**; FLOW-8 **Deferred** or Done per product; §6.1aj closed; **zero open `FLOW-GAP-*`** in flow reference §23; AUDIT_MAP §5/§7/§9/§10/§25 at target maturity; gate green.

---



**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (INT-DOC.* + INT-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §13; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix K**.

**Priority ladder:** **Band 2l** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bd](#62bd-phase-int-execution-order-band-2l--closed) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

### INT — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| INT-DOC.1 | INT0 | **Appendix K** — integration control plane (§K.1–K.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| INT-DOC.2 | INT0 | **Cross-ref sync** — plan, README, AUDIT_MAP §13, audit prompt ref #8 | **Done** | Medium | `docs/*` | Links resolve |
| INT-1 | INT1 | **`integration_runtime_bridge.py`** — explicit `integration_profile` on `RuntimeConfig` | **Done** | **Critical** | `integration_runtime_bridge.py`, `runtime_config_bridge.py` | `test_integration_runtime_bridge.py` |
| INT-2 | INT2 | **`integration_health_wiring.py`** — bootstrap health probes on `wire_application_environment` | **Done** | High | `integration_health_wiring.py`, `environment_wiring.py` | `test_integration_health_wiring.py` |

### INT — Paydown log

| Date | INT ID | Summary |
|------|--------|---------|
| 2026-06-02 | INT-DOC.1, INT-DOC.2 | Appendix K + cross-refs; AUDIT_MAP §13 |
| 2026-06-02 | INT-1, INT-2 | Integration runtime bridge + health wiring |

**Phase INT complete when:** INT-1–2 + INT-DOC.* **Done**; §6.1d queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2be](#62be-phase-rag-execution-order-band-2m--closed) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

### RAG — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| RAG-DOC.1 | RAG0 | **Appendix K** §K.5 + AUDIT_MAP §14 cross-ref | **Done** | High | `docs/*` | RAG bridge documented |
| RAG-1 | RAG1 | **`rag_runtime_bridge.py`** + RAG stack on `wire_application_environment` | **Done** | **Critical** | `rag_runtime_bridge.py`, `environment_wiring.py`, `runtime_config_bridge.py` | `test_rag_runtime_bridge.py` |

### RAG — Paydown log

| Date | RAG ID | Summary |
|------|--------|---------|
| 2026-06-02 | RAG-DOC.1 | Appendix K §K.5 + plan sync |
| 2026-06-02 | RAG-1 | RAG runtime bridge + environment wire; gate **600** |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CTX-DOC.* + CTX-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §16; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix L**.

**Priority ladder:** **Band 2n** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bf](#62bf-phase-ctx-execution-order-band-2n--closed) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

### CTX — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| CTX-DOC.1 | CTX0 | **Appendix L** — context engineering control plane (§L.1–L.6) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CTX-DOC.2 | CTX0 | **Cross-ref sync** — plan, README, AUDIT_MAP §16, audit prompt ref #9 | **Done** | Medium | `docs/*` | Links resolve |
| CTX-1 | CTX1 | **`context_runtime_bridge.py`** — dedicated context profile → `RuntimeConfig` | **Done** | **Critical** | `context_runtime_bridge.py`, `runtime_config_bridge.py` | `test_context_runtime_bridge.py` |
| CTX-2 | CTX2 | **`context_wiring.py`** — `ContextManager` + task options from environment; `nexus_factory` wire | **Done** | High | `context_wiring.py`, `nexus_factory.py`, `harness_host_runtime.py` | `test_context_wiring.py` |

### CTX — Paydown log

| Date | CTX ID | Summary |
|------|--------|---------|
| 2026-06-02 | CTX-DOC.1, CTX-DOC.2 | Appendix L + cross-refs; AUDIT_MAP §16 |
| 2026-06-02 | CTX-1, CTX-2 | Context runtime bridge + Nexus ContextManager wiring; gate **608** |

**Phase CTX complete when:** CTX-1–2 + CTX-DOC.* **Done**; §6.1f queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `engine_planner_messages.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (PE-DOC.* + PE-1–3); gate **623 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §17; V-REM-PE.1/PE.2 governance schema (**Done**); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix M**.

**Priority ladder:** **Band 2p** (§4.0) — closed; default queue = **§6.1** maintenance.

### PE — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| PE-1 | PE1 | **`PromptProfile`** + `prompt_runtime_bridge` — `catalog_path` → `RuntimeConfig.prompt_catalog_path` | **Done** | `environment_profile.py`, `prompt_runtime_bridge.py`, `config.py` | `test_prompt_runtime_bridge.py` |
| PE-2 | PE2 | **`prompt_wiring`** — `resolve_prompt_registry()`, `PromptRegistryProtocol` | **Done** | `prompt_wiring.py`, `prompt_registry_protocol.py` | `test_prompt_wiring.py` |
| PE-3 | PE3 | **Environment wire** — `materialize_runtime_config`, `build_runtime_context_from_environment`, `ApplicationBuildContext.prompt_registry` | **Done** | `runtime_config_bridge.py`, `environment_wiring.py`, `runtime_context.py` | wiring tests + gate |
| PE-4 | PE4 | **Nexus injection** — `prompt_registry_resolver`; `tools_step`, `tool_planning_prompts`, `engine_plan_models`, `engine_planner_messages` use `RuntimeContext.prompt_registry` | **Done** | `prompt_registry_resolver.py`, Nexus steps/planner | `test_tools_step_prompt_registry.py` |
| PE-DOC.1 | PE0 | **Appendix M** — prompt registry control plane (§M.1–M.6) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |

**Residual:** none on Tier-3 host build path. Legacy YAML prompt assets (`chat_router*`, `tools_agent_*`) remain as catalog files only.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](plan/ORCHESTRATION.md).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent/` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts/`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers/` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (AS-DOC.1 + AS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix N**.

**Priority ladder:** **Band 2q** (§4.0) — closed; default queue = **§6.1** maintenance.

### AS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| AS-DOC.1 | AS0 | **Appendix N** — agent assembly control plane (contract, capabilities, skills, lifecycle) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| AS-1 | AS1 | **`agent_assembly_resolver`** — contract metadata validation at register time | **Done** | `runtime/registry/agent_assembly_resolver.py`, `agent_registry.py` | `test_agent_assembly_resolver.py` |
| AS-2 | AS2 | **Lifecycle metadata enforcement** — `production_eligible` owner/runbook requirements | **Done** | `agent_assembly_resolver.py`, `agent_routing_policy.py` | resolver + routing tests |
| AS-3 | AS3 | **`skill_ids` → `allowed_tools` resolution audit** — CI script + docs cross-ref | **Done** | `scripts/check_agent_skill_resolution.py`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), Legal domain steps, product-only contract variants — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REG-DOC.1 + REG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix O**.

**Priority ladder:** **Band 2r** (§4.0) — closed; default queue = **§6.1** maintenance.

### REG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REG-DOC.1 | REG0 | **Appendix O** — registry architecture control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REG-1 | REG1 | **`HarnessRegistrySnapshot`** + `registry_wiring` + `RegistrySnapshotProtocol` | **Done** | `registry_snapshot.py`, `registry_wiring.py` | `test_registry_wiring.py` |
| REG-2 | REG2 | **`registry_assembly_resolver`** — profile ↔ registry conformance at wire time | **Done** | `registry_assembly_resolver.py`, `environment_wiring.py` | `test_registry_wiring.py` |
| REG-3 | REG3 | **Host registry resolution CI** — `check_harness_registry_resolution.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), marketplace UI, Band 3 product hosts — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CG-DOC.1 + CG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix P**.

**Priority ladder:** **Band 2s** (§4.0) — closed; default queue = **§6.1** maintenance.

### CG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CG-DOC.1 | CG0 | **Appendix P** — capability graph control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CG-1 | CG1 | **`capability_graph_wiring`** — environment subgraph from catalog + registry snapshot | **Done** | `capability_graph_wiring.py`, `capability_graph_protocol.py` | `test_capability_graph_wiring.py` |
| CG-2 | CG2 | **`capability_graph_assembly_resolver`** — wire-time catalog node validation | **Done** | `capability_graph_assembly_resolver.py`, `environment_wiring.py` | `test_capability_graph_wiring.py` |
| CG-3 | CG3 | **Host capability graph CI** — `check_harness_capability_graph_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only graph nodes — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (OBS-DOC.1 + OBS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21; complements GOV-AUDIT Appendix H; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix Q**.

**Priority ladder:** **Band 2t** (§4.0) — closed; default queue = **§6.1** maintenance.

### OBS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| OBS-DOC.1 | OBS0 | **Appendix Q** — observability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| OBS-1 | OBS1 | **`observability_runtime_bridge`** + **`observability_wiring`** | **Done** | `observability_runtime_bridge.py`, `observability_wiring.py`, `runtime_config_bridge.py` | `test_harness_observability_wiring.py` |
| OBS-2 | OBS2 | **`observability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `observability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| OBS-3 | OBS3 | **Host observability CI** — `check_harness_observability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only observability dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **8/8** deliverables · OBS-BUS-0–7 **Done**

**Purpose:** Implement the full **Harness Observability Spine (HOS)** — one bus for Harness, applications, and agents; typed extension; causal trees; complete catalog emission; L4 audit §21.

**Architecture:** [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **ADR:** [ADR-OBS-001](adr/ADR-OBS-001.md)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21 · complements Phase OBS (wiring closeout) · supersedes residual “live bus emit for all LLM paths” row when OBS-BUS-2 ships.

**Priority ladder:** **Band 2al** (§4.0) — runs **after** Phase CRIT-V (Band 2ak) or in parallel §6.1 maintenance slices; **one OBS-BUS ID per PR**.

**Depends on:** Phase OBS (wiring) **Done** · OBS-DEPTH.1/2 **Done** · FAUDIT-OBS.1 **Done**

### OBS-BUS — Master register

| ID | Area | Deliverable | Status | Modules / artifacts | Acceptance |
|----|------|-------------|--------|---------------------|------------|
| OBS-BUS-0 | OBS0 | **Architecture canon** — `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README links | **Done** | `docs/architecture/OBSERVABILITY.md`, `docs/adr/ADR-OBS-001.md` | Doc review; links from §33 |
| OBS-BUS-1 | OBS1 | **`RuntimeEventPayload` registry** — typed canonical payloads per `RuntimeEventType` (§42.23.1 families) | **Done** | `intergrax/runtime/events/payload_registry.py`, `payloads/`, `schema_guard.py`, `trace_bridge.py`, `context_skill_recording.py` | Gate: `test_runtime_event_payload_registry.py` |
| OBS-BUS-2 | OBS2 | **`ObservabilityEmitter` + `TraceScope`** — single emit API; `parent_event_id` causal tree | **Done** | `intergrax/runtime/observability/emitter.py`, `trace_scope.py`, `runtime_state.py` | `RuntimeState.trace_event` delegates; `test_observability_emitter.py` |
| OBS-BUS-3 | OBS3 | **Emission coverage** — `AGENT_SELECTED`, `STEP_FAILED`, graph typed payloads, critic `evaluator_loop` bridge | **Done** | `agent_router.py`, `graph_trace_callbacks.py`, `task_trace.py`, `trace_bridge.py`, `graph_node_diag.py` | `check_observability_emission_coverage.py` |
| OBS-BUS-4 | OBS4 | **Extension SDK** — agent/app `DiagnosticPayload` scaffold, namespace rules, `PayloadSchemaRegistry` | **Done** | `extension_sdk.py`, `tracing_templates.py`, `new_agent.py`, `new_application.py` | `check_payload_schema_registry.py` |
| OBS-BUS-5 | OBS5 | **Persistence conformance** — Cassandra/ES adapters implement same protocols; profile docs | **Done** | `document_backed_runtime_event_store.py`, `persistence_conformance.py`, profile wiring | `check_observability_persistence_conformance.py` |
| OBS-BUS-6 | OBS6 | **Export sinks** — OTLP dual-write from unified journal; parser trace link | **Done** | `journal_export.py`, `export_bridge.py`, `task_events.py`, `platform_wiring.py` | `TASK_COMPLETED` carries `journal_ref`; export plugin dual-writes OTLP JSON + parser trace |
| OBS-BUS-7 | OBS7 | **CI gates** — emission coverage + schema registry + L4 §21 evidence | **Done** | `scripts/check_observability_gates.py`, emission/schema/persistence audits, CI workflow | Gate suite green; audit map §21 → **L4** |

### OBS-BUS — Execution order (recommended)

```text
OBS-BUS-0 (docs) → OBS-BUS-1 (typed payloads)
  → OBS-BUS-2 (emitter + TraceScope)
  → OBS-BUS-3 (coverage gaps)
  → OBS-BUS-4 (extension SDK)
  → OBS-BUS-5 (persistence)
  → OBS-BUS-6 (sinks)
  → OBS-BUS-7 (gates / L4 closeout)
```

**DoD:** All OBS-BUS rows **Done**; `build_unified_run_journal` reproduces full Nexus+AgentEngine path without reading source; every `RuntimeEventType` in §42.1.2 has ≥1 production emitter; `parent_event_id` populated for tool/LLM/delegation; extension scaffold documented; gate green.

**Explicitly excluded:** product-specific dashboards (§6.3a); replacing external APM as mandatory deployment.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (SEC-DOC.1 + SEC-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §23; V-SEC / V-REM-SEC **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix S**.

**Priority ladder:** **Band 2v** (§4.0) — closed; default queue = **§6.1** maintenance.

### SEC — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| SEC-DOC.1 | SEC0 | **Appendix S** — security control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| SEC-1 | SEC1 | **`security_runtime_bridge`** + **`security_wiring`** | **Done** | `security_runtime_bridge.py`, `security_wiring.py`, `runtime_config_bridge.py` | `test_harness_security_wiring.py` |
| SEC-2 | SEC2 | **`security_assembly_resolver`** — profile ↔ middleware conformance | **Done** | `security_assembly_resolver.py`, `harness_host_runtime.py`, `nexus_factory.py` | assembly validation tests |
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (EVAL-DOC.1 + EVAL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25; V-EVAL **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix U**.

**Priority ladder:** **Band 2x** (§4.0) — closed; default queue = **§6.1** maintenance.

### EVAL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| EVAL-DOC.1 | EVAL0 | **Appendix U** — evaluation control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| EVAL-1 | EVAL1 | **`EvaluationProfile`** + **`evaluation_runtime_bridge`** + **`evaluation_wiring`** | **Done** | `environment_profile.py`, `evaluation_runtime_bridge.py`, `evaluation_wiring.py`, `policy_wiring.py` | `test_harness_evaluation_wiring.py` |
| EVAL-2 | EVAL2 | **`evaluation_assembly_resolver`** — profile ↔ registry conformance | **Done** | `evaluation_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py`, `runtime.py` | assembly validation tests |
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **24/24** deliverables Done (CRIT-V-0 through CRIT-V-7)  
**Prerequisites:** Phase EVAL **Done** (registry wiring), Phase FLOW **Done** (graph hooks), Phase M-LLM-R **Done** (typed LLM envelope)  
**Goal:** Deliver production-grade PEV **Verify** infrastructure — L0/L1/L2 critic stack with tier-separated competencies; uplift Evaluation audit layer L2→L3.  
**Priority ladder:** **Band 2ak** (§4.0) — **Done** (2026-06-08). Default queue reverts to §6.1 gate maintenance.  
**Architecture:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · canon [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/ADR-CRITIC-001.md)  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25 (Evaluation), §7 (Reasoning), §10 (Multi-agent); closes **FAUDIT-EVAL.1** residual  
**Execution order:** [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

**Delivery rule:** One **CRIT-V-*** ID per PR → update master table + §6.1ak + gate green.

### CRIT-V — Master register

| ID | Wave | Deliverable | Status | Modules / docs | Acceptance |
|----|------|-------------|--------|----------------|------------|
| CRIT-V-0.1 | 0 | **Architecture RFC** — CVL full spec | **Done** | `architecture/CRITIC_VERIFICATION.md` | Linked from canon §55, README |
| CRIT-V-0.2 | 0 | **ADR-CRITIC-001** — tier-separated PEV verify | **Done** | `docs/adr/ADR-CRITIC-001.md` | Status Accepted; adr index |
| CRIT-V-0.3 | 0 | **Canon §55** addendum | **Done** | `intergrax_runtime_architecture.md` §55 | Cross-links resolve |
| CRIT-V-0.4 | 0 | **README** sections (root + docs) | **Done** | `README.md`, `docs/README.md` | Navigation table |
| CRIT-V-1.1 | 1 | **`CriticProfile`** on `ApplicationEnvironmentProfile` | **Done** | `contracts/environment_profile.py`, `critic_runtime_bridge.py`, `RuntimeConfig` | Unit: `test_harness_critic_wiring.py` |
| CRIT-V-1.2 | 1 | **CVL contracts** — `CriticRequest`, `CriticVerdict`, `LayerVerdict`, `RubricSpec` | **Done** | `runtime/critic/contracts.py` | Unit: `test_critic_contracts.py` |
| CRIT-V-1.3 | 1 | **`EvaluatorLoopSpec`** — max iterations, revise routing | **Done** | `runtime/critic/evaluator_loop_spec.py` | Unit: `test_evaluator_loop_spec.py` |
| CRIT-V-2.1 | 2 | **`eval.judge` tool** — semantic scoring via separate LLM profile | **Done** | `tools/providers/eval/judge.py`, bundle | `test_eval_critic_tools.py` |
| CRIT-V-2.2 | 2 | **`eval.trajectory` tool** — process scoring from replay slice | **Done** | `tools/providers/eval/trajectory.py` | Uses `trace_reader` |
| CRIT-V-2.3 | 2 | **Registry hook** — judge/trajectory → `OnlineEvaluationObservation` | **Done** | `service.py` `_append_critic_observation` | Observation appended when registry bound |
| CRIT-V-3.1 | 3 | **`CriticOrchestrator`** — L0→L1→L2 pipeline | **Done** | `runtime/critic/critic_orchestrator.py` | Unit: short-circuit, layer order |
| CRIT-V-3.2 | 3 | **`L0Gateway`** — wraps `NexusValidationEngine` + schema | **Done** | `runtime/critic/l0_gateway.py` | Reuses existing validators |
| CRIT-V-3.3 | 3 | **`L1Gateway`** — invokes eval tools via `CriticEvalToolClient` | **Done** | `runtime/critic/l1_gateway.py` | No direct LLM in Tier-1 |
| CRIT-V-3.4 | 3 | **Graph partial hook** — `GraphExecutor` → `verify_partial` | **Done** | `graph_executor.py`, `critic_wiring.py` | Integration test: L0 fail → retry |
| CRIT-V-3.5 | 3 | **Graph final hook** — `GraphRunner` → `verify_final` | **Done** | `graph_runner.py` | Terminal state respects verdict |
| CRIT-V-3.6 | 3 | **Critic trace events** — `critic.*` trace steps | **Done** | `runtime/critic/trace.py`, `trace_bridge.py` | Visible in lab trace API |
| CRIT-V-4.1 | 4 | **`EvaluatorLoopExecutor`** — critique→revise routing | **Done** | `runtime/critic/evaluator_loop_executor.py` | Unit: budget exhaustion → FAIL/HITL |
| CRIT-V-4.2 | 4 | **Graph integration** — `EVALUATOR_LOOP` pattern wired | **Done** | `graph_executor.py`, `evaluator_loop_metadata.py` | Acceptance: 2-iteration loop |
| CRIT-V-5.1 | 5 | **`NexusEvalRunner` semantic mode** — optional L1 via `eval.judge` | **Done** | `eval/nexus_eval_runner.py` | Integration: non-exact pass |
| CRIT-V-5.2 | 5 | **`EvalCase` rubric field** — rubric_ref + semantic_threshold | **Done** | `eval/eval_case.py` | Backward compatible |
| CRIT-V-6.1 | 6 | **`wire_application_critic()`** — Tier-3 wiring | **Done** | `applications/_shared/critic_wiring.py` | Mirror EVAL pattern |
| CRIT-V-6.2 | 6 | **`critic_assembly_resolver`** — wire-time validation | **Done** | `critic_assembly_resolver.py`, `check_harness_critic_wiring.py` | CI script |
| CRIT-V-6.3 | 6 | **Policy bundle** — `critic_governance` fragment | **Done** | `policy_wiring.py` | Merged at host build |
| CRIT-V-6.4 | 6 | **Appendix W** — critic control plane author map | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CRIT-V-7.1 | 7 | **FAUDIT-EVAL.1** — `require_baseline_for_release` CI gate | **Done** | `phase_v_closeout_gate.py`, `check_harness_critic_wiring.py` | Closeout gate green |
| CRIT-V-7.2 | 7 | **Flow reference §18 sync** — CVL hook table | **Done** | `architecture/NEXUS_EXECUTION_FLOW.md` | Hooks documented |
| CRIT-V-7.3 | 7 | **Lab harness demo** — L0+L1 on sample agent (not FLOW-8) | **Done** | `test_harness_critic_wiring.py`, lab host | Trace shows critic steps |
| CRIT-V-F.1 | F | **`ToolRegistryCriticEvalClient`** — L1 bridge to Tier-0 eval tools | **Done** | `runtime/critic/tool_registry_client.py`, `critic_tool_wiring.py` | `test_critic_closeout.py` |
| CRIT-V-F.2 | F | **`critic_llm_profile`** — separate judge LLM adapter | **Done** | `critic_llm_resolver.py`, `environment_profile.py` | Assembly + wiring tests |
| CRIT-V-F.3 | F | **L2 `L2Gateway`** + HITL escalation path | **Done** | `l2_gateway.py`, `critic_orchestrator.py` | `test_critic_l2_gateway.py` |
| CRIT-V-F.4 | F | **UAEP step hook** | **Done** | `uaep.py`, `validate_uaep_step_with_critic_detail` | UAEP critic path |
| CRIT-V-F.5 | F | **`CriticPolicyBridge`** + policy engine | **Done** | `policy_bridge.py`, `runtime_policy_engine.py` | `test_critic_closeout.py` |
| CRIT-V-F.6 | F | **Assembly gate** — require L1 client when semantic/trajectory enabled | **Done** | `critic_assembly_resolver.py` | `test_critic_assembly_resolver.py` |

**Explicitly excluded:** FLOW-8 §42.43 product reference app ([§6.3](#63-end-of-plan--deferred-product-work-only)); domain rubric packs in Tier-0; mandatory universal LLM-judge on all runs.

**Phase CRIT-V complete when:** CRIT-V-1 through CRIT-V-7 **Done**; Evaluation audit layer ≥ **L3**; gate green; FAUDIT-EVAL.1 closed.

---

---

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](#63a-business-backlog-register-consolidated)** · **[§4.0a](#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](#61z-harness-implementation-queue-consolidated) / [§6.1aa](#61aa-harness-implementation-queue-memory-platform).

### 6.1i Harness implementation queue — prompt registry closeout (closed)

**Purpose:** Single ordered list for **Phase PE** (Band 2p). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **PE-DOC.1** | Docs | **Done** | Appendix M + cross-refs | Author map complete |
| 2 | **PE-1** | Code | **Done** | `prompt_runtime_bridge` + `PromptProfile` | `test_prompt_runtime_bridge.py` |
| 3 | **PE-2** | Code | **Done** | `prompt_wiring` + `PromptRegistryProtocol` | `test_prompt_wiring.py` |
| 4 | **PE-3** | Code | **Done** | environment + runtime context wire | gate green |
| 5 | **PE-4** | Code | **Done** | Nexus prompt registry injection | `test_tools_step_prompt_registry.py` |

### 6.1j Harness implementation queue — legacy module closeout (closed)

**Purpose:** Single ordered list for **Phase CLEAN** (post-2p closeout). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CLEAN-1** | Code | **Done** | Remove `chat_router.py`; YAML-only tests | `test_chat_agent_prompts_yaml.py` |
| 2 | **CLEAN-2** | Code | **Done** | Remove `tools_agent.py`; planner tests | `test_catalog_tool_planner.py` |
| 3 | **CLEAN-3** | CI | **Done** | `check_legacy_modules_removed.py` in CI | workflow green |
| 4 | **CLEAN-4** | Docs | **Done** | Plan + harness docs sync | no stale production refs |

**Suggested PR order (complete):** CLEAN-1 → CLEAN-2 → CLEAN-3 → CLEAN-4.

### 6.1k Harness implementation queue — agent assembly closeout (closed)

**Purpose:** Single ordered list for **Phase AS** (Band 2q). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **AS-DOC.1** | Docs | **Done** | Appendix N + cross-refs | Author map complete |
| 2 | **AS-1** | Code | **Done** | `agent_assembly_resolver` | `test_agent_assembly_resolver.py` |
| 3 | **AS-2** | Code | **Done** | Lifecycle metadata enforcement | resolver + routing tests |
| 4 | **AS-3** | CI | **Done** | `check_agent_skill_resolution.py` | CI green |

**Suggested PR order (complete):** AS-DOC.1 → AS-1 → AS-2 → AS-3.

**Explicitly excluded:** K.1, K.2, new product agents, domain-only contract packs — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1l Harness implementation queue — registry architecture closeout (closed)

**Purpose:** Single ordered list for **Phase REG** (Band 2r). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REG-DOC.1** | Docs | **Done** | Appendix O + cross-refs | Author map complete |
| 2 | **REG-1** | Code | **Done** | `HarnessRegistrySnapshot` + `registry_wiring` | `test_registry_wiring.py` |
| 3 | **REG-2** | Code | **Done** | `registry_assembly_resolver` wire | `test_registry_wiring.py` |
| 4 | **REG-3** | CI | **Done** | `check_harness_registry_resolution.py` | CI green |

**Suggested PR order (complete):** REG-DOC.1 → REG-1 → REG-2 → REG-3.

### 6.1m Harness implementation queue — capability graph closeout (closed)

**Purpose:** Single ordered list for **Phase CG** (Band 2s). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CG-DOC.1** | Docs | **Done** | Appendix P + cross-refs | Author map complete |
| 2 | **CG-1** | Code | **Done** | `capability_graph_wiring` | `test_capability_graph_wiring.py` |
| 3 | **CG-2** | Code | **Done** | `capability_graph_assembly_resolver` | wire-time validation tests |
| 4 | **CG-3** | CI | **Done** | `check_harness_capability_graph_wiring.py` | CI green |

**Suggested PR order (complete):** CG-DOC.1 → CG-1 → CG-2 → CG-3.

### 6.1n Harness implementation queue — observability closeout (closed)

**Purpose:** Single ordered list for **Phase OBS** (Band 2t). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **OBS-DOC.1** | Docs | **Done** | Appendix Q + cross-refs | Author map complete |
| 2 | **OBS-1** | Code | **Done** | `observability_runtime_bridge` + `observability_wiring` | `test_harness_observability_wiring.py` |
| 3 | **OBS-2** | Code | **Done** | `observability_assembly_resolver` | wire-time validation tests |
| 4 | **OBS-3** | CI | **Done** | `check_harness_observability_wiring.py` | CI green |

**Suggested PR order (complete):** OBS-DOC.1 → OBS-1 → OBS-2 → OBS-3.

### 6.1o Harness implementation queue — reliability closeout (closed)

**Purpose:** Single ordered list for **Phase REL** (Band 2u). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **REL-DOC.1** | Docs | **Done** | Appendix R + cross-refs | Author map complete |
| 2 | **REL-1** | Code | **Done** | `reliability_runtime_bridge` + `reliability_wiring` | `test_harness_reliability_wiring.py` |
| 3 | **REL-2** | Code | **Done** | `reliability_assembly_resolver` | wire-time validation tests |
| 4 | **REL-3** | CI | **Done** | `check_harness_reliability_wiring.py` | CI green |

**Suggested PR order (complete):** REL-DOC.1 → REL-1 → REL-2 → REL-3.

### 6.1q Harness implementation queue — security closeout (closed)

**Purpose:** Single ordered list for **Phase SEC** (Band 2v). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **SEC-DOC.1** | Docs | **Done** | Appendix S + cross-refs | Author map complete |
| 2 | **SEC-1** | Code | **Done** | `security_runtime_bridge` + `security_wiring` | `test_harness_security_wiring.py` |
| 3 | **SEC-2** | Code | **Done** | `security_assembly_resolver` | wire-time validation tests |
| 4 | **SEC-3** | CI | **Done** | `check_harness_security_wiring.py` | CI green |

**Suggested PR order (complete):** SEC-DOC.1 → SEC-1 → SEC-2 → SEC-3.

### 6.1r Harness implementation queue — cost governance closeout (closed)

**Purpose:** Single ordered list for **Phase COST** (Band 2w). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **COST-DOC.1** | Docs | **Done** | Appendix T + cross-refs | Author map complete |
| 2 | **COST-1** | Code | **Done** | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | `test_harness_cost_wiring.py` |
| 3 | **COST-2** | Code | **Done** | `cost_assembly_resolver` | wire-time validation tests |
| 4 | **COST-3** | CI | **Done** | `check_harness_cost_wiring.py` | CI green |

**Suggested PR order (complete):** COST-DOC.1 → COST-1 → COST-2 → COST-3.

### 6.1s Harness implementation queue — evaluation closeout (closed)

**Purpose:** Single ordered list for **Phase EVAL** (Band 2x). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **EVAL-DOC.1** | Docs | **Done** | Appendix U + cross-refs | Author map complete |
| 2 | **EVAL-1** | Code | **Done** | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | `test_harness_evaluation_wiring.py` |
| 3 | **EVAL-2** | Code | **Done** | `evaluation_assembly_resolver` | wire-time validation tests |
| 4 | **EVAL-3** | CI | **Done** | `check_harness_evaluation_wiring.py` | CI green |

**Suggested PR order (complete):** EVAL-DOC.1 → EVAL-1 → EVAL-2 → EVAL-3.

### 6.1f Harness implementation queue — context engineering closeout (closed)

**Purpose:** Single ordered list for **Phase CTX** (Band 2n). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CTX-DOC.1–2** | Docs | **Done** | Appendix L + cross-refs | Author map complete |
| 2 | **CTX-1** | Code | **Done** | `context_runtime_bridge` | `test_context_runtime_bridge.py` |
| 3 | **CTX-2** | Code | **Done** | `context_wiring` + `nexus_factory` | `test_context_wiring.py` |

### 6.1e Harness implementation queue — RAG closeout (closed)

**Purpose:** Single ordered list for **Phase RAG** (Band 2m). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **RAG-DOC.1** | Docs | **Done** | Appendix K §K.5 + AUDIT_MAP §14 | Author map complete |
| 2 | **RAG-1** | Code | **Done** | `rag_runtime_bridge` + environment wire | `test_rag_runtime_bridge.py` |

### 6.1d Harness implementation queue — integration closeout (closed)

**Purpose:** Single ordered list for **Phase INT** (Band 2l). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **INT-DOC.1–2** | Docs | **Done** | Appendix K + cross-refs | Author map complete |
| 2 | **INT-1** | Code | **Done** | `integration_runtime_bridge` | `test_integration_runtime_bridge.py` |
| 3 | **INT-2** | Code | **Done** | `integration_health_wiring` | `test_integration_health_wiring.py` |

### 6.1c Harness implementation queue — tools/skills closeout (closed)

**Purpose:** Single ordered list for **Phase TS** (Band 2k). **Closed 2026-06-02** — all TS rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **TS-DOC.1–2** | Docs | **Done** | Appendix J + cross-refs | Author map complete |
| 2 | **TS-1** | Code | **Done** | `catalog_runtime_bridge` + `RuntimeConfig.skill_profile` | `test_catalog_runtime_bridge.py` |
| 3 | **TS-2** | Code | **Done** | Harness host `resolve_llm_adapter` wiring | `test_harness_host_runtime_llm.py` |
| 4 | **TS-3** | Code | **Done** | `SkillResolverProtocol` | skill resolver tests green |

**Suggested PR order (complete):** TS-1 → TS-2 → TS-3 → TS-DOC.*.

**Explicitly excluded:** K.1, K.2, new product tools/skills, business agent packs — [§6.3a](#63a-business-backlog-register-consolidated).

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

### 6.1aj Harness implementation queue — Nexus execution depth (closed)

**Purpose:** Single ordered list for **Phase FLOW** (Band 2aj). **Closed 2026-06-09** — **18/18 harness Done** (FLOW-8 harness ORCH-CONFIG.5); product host **Deferred** §6.3. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **FLOW-2** | Code | **Done** | ADR-FLOW-001 — `DELEGATES_TO` → child node | Delegation integration tests |
| 2 | **FLOW-14** | Code | **Done** | `SubtaskContract` in delegation expansion | Scopes on child `DelegationSpec` |
| 3 | **FLOW-3** | Code | **Done** | `max_delegation_depth` enforcement | Depth limit test |
| 4 | **FLOW-15** | Code | **Done** | Subagent budget envelope | Child budget exceeded → fail |
| 5 | **FLOW-6** | Code | **Done** | Graph cycle detection | Cyclic graph fails fast |
| 6 | **FLOW-1** | Code | **Done** | Real `EngineBackedNexusPlanner` | LLM plan parse tests |
| 7 | **FLOW-4** | Code | **Done** | Run-level retry profile field | Graph retry integration |
| 8 | **FLOW-13** | Code | **Done** | `max_inflight_nodes` profile + wire | Backpressure event test |
| 9 | **FLOW-7** | Code | **Done** | `MergePolicy` / composer profile | Multi-agent merge tests |
| 10 | **FLOW-9** | Code | **Done** | Multi-agent eval hooks | Registry observation |
| 11 | **FLOW-11** | Code | **Done** | Pre-plan policy hooks | Planning boundary tests |
| 12 | **FLOW-5** | Code | **Done** | `AgentGraph.on_error` wire | Integration test |
| 13 | **FLOW-10** | Code/Docs | **Done** | Reserved lifecycle ADR ([ADR-FLOW-002](adr/ADR-FLOW-002.md)) | Lifecycle doc |
| 14 | **FLOW-12** | Code | **Done** | `DecisionRecord` regression gate | Gate test per step |
| 15 | **FLOW-16** | Docs | **Done** | `MODIFY_PLAN` ADR (ADR-FLOW-003) | ADR accepted |
| 16 | **FLOW-17** | Code | **Done** | `MULTI_AGENT` ordering policy | Stable order gate test |
| 17 | **FLOW-DOC.*** | Docs | **Done** | Flow reference + Appendix N paydown | Zero open FLOW-GAP |
| — | **FLOW-8** | Harness + Product | **Partial** | Harness: `test_orchestration_cfg_simulation.py` · Product §42.43: **§6.3** |

**Suggested PR order:** See [Phase FLOW — Suggested PR order](#flow--suggested-pr-order).

**Explicitly excluded:** K.1, K.2 (unless FLOW-8 activated), nested harness per child.

### 6.1ak Harness implementation queue — Critic & Verification Layer (closed)

**Purpose:** Single ordered list for **Phase CRIT-V** (Band 2ak). **Closed 2026-06-08** — CRIT-V-0…7 + **CRIT-V-FOLLOWUP** closeout **Done**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **CRIT-V-0.*** | Docs | **Done** | Architecture RFC + ADR + canon §55 + README | Cross-links resolve |
| 2 | **CRIT-V-1.*** | Code | **Done** | `CriticProfile` + CVL contracts | Unit tests |
| 3 | **CRIT-V-2.*** | Code | **Done** | `eval.judge` + `eval.trajectory` tools | Tool gate tests |
| 4 | **CRIT-V-3.1–3.3** | Code | **Done** | `CriticOrchestrator` + L0/L1 gateways | `test_critic_orchestrator.py` |
| 5 | **CRIT-V-3.4–3.5** | Code | **Done** | Graph partial + final hooks | Integration tests |
| 5 | **CRIT-V-4.*** | Code | **Done** | `EvaluatorLoopExecutor` | Loop budget tests |
| 6 | **CRIT-V-5.*** | Code | **Done** | Semantic `NexusEvalRunner` | Eval integration test |
| 7 | **CRIT-V-6.*** | Code/Docs | **Done** | Tier-3 wiring + Appendix W | CI assembly script |
| 8 | **CRIT-V-7.*** | Code/Docs | **Done** | FAUDIT-EVAL.1 + flow reference sync | Closeout gate green |
| 9 | **CRIT-V-FOLLOWUP** | Code | **Done** | L1 client, L2 HITL, UAEP hook, policy bridge | `test_critic_closeout.py`, gate green |

**Suggested PR order:** See [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed).

**Explicitly excluded:** FLOW-8 product app; domain rubric packs in Tier-0; mandatory universal LLM-judge.

### 6.1al Harness implementation queue — Unified Observability Spine (closed)

**Purpose:** Single ordered list for **Phase OBS-BUS** (Band 2al). **Closed 2026-06-08** — all OBS-BUS rows **Done**; audit map §21 → **L4**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **OBS-BUS-0** | Docs | **Done** | `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README | Links resolve |
| 2 | **OBS-BUS-1** | Code | **Done** | `RuntimeEventPayload` registry | Payload registry gate |
| 3 | **OBS-BUS-2** | Code | **Done** | `ObservabilityEmitter` + `TraceScope` | Causal tree tests |
| 4 | **OBS-BUS-3** | Code | **Done** | Emission coverage gaps | `check_observability_emission_coverage.py` |
| 5 | **OBS-BUS-4** | Code/Docs | **Done** | Extension SDK + scaffold | Agent tracing template |
| 6 | **OBS-BUS-5** | Code | **Done** | Persistence conformance | Integration tests |
| 7 | **OBS-BUS-6** | Code | **Done** | OTLP/journal dual-write | `test_journal_export.py`, `test_export_bridge.py` |
| 8 | **OBS-BUS-7** | CI | **Done** | L4 §21 gates | `check_observability_gates.py` in CI; audit map §21 → L4 |

**Suggested PR order:** See [Phase OBS-BUS — Execution order](#obs-bus--execution-order-recommended).

**Explicitly excluded:** Product dashboards (§6.3a); vendor-only APM as sole store.

### 6.1am Harness implementation queue — Memory intelligence depth (closed)

**Purpose:** Single ordered list for **Phase MEM-DEPTH** (Band 2am). **Closed 2026-06-08** — **26/26 Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **MEM-DEPTH-0.1–0.4** | Docs | **Done** | `architecture/MEMORY.md` + plan + ADR + cross-links | ADR-MEM-001 |
| 2 | **MEM-DEPTH-1.1–1.6** | Code/Test | **Done** | Context Compiler + never-overflow | Long-session gate green |
| 3 | **MEM-DEPTH-2.1–2.2** | Code/Test | **Done** | Mongo session persistence | Integration round-trip |
| 4 | **MEM-DEPTH-3.1–3.5** | Code | **Done** | Lifecycle automation | Auto consolidate on lab profile |
| 5 | **MEM-DEPTH-4.1–4.3** | Code | **Done** | Explore delegation | Delegation gate + synthesis return |
| 6 | **MEM-DEPTH-5.1–5.6** | Code/RFC | **Done** | Entity intelligence | FAUDIT Memory L3+ |

**Suggested PR order:** See [§6.2ab](#62ab-phase-mem-depth-execution-order-band-2am--closed).

**Explicitly excluded:** K.1/K.2, Mem0 SaaS, Redis session default — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1b Harness implementation queue — orchestration closeout (closed)

**Purpose:** Single ordered list for **Phase ORCH** (Band 2j). **Closed 2026-06-05** — all ORCH rows **Done**. Ongoing: **§6.1** maintenance only.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **ORCH-DOC.1–2** | Docs | **Done** | Appendix I + cross-refs | Author map complete |
| 2 | **ORCH-1** | Code | **Done** | `planner_kind` / `classifier_kind` wiring | `test_orchestration_wiring.py` |
| 3 | **ORCH-2** | Code | **Done** | `ApplicationGraphSpec` → `NexusPlan` | `test_graph_spec_to_plan.py` |
| 4 | **ORCH-3** | Code | **Done** | `max_parallel_nodes` cap | `test_graph_executor_parallel_cap.py` |
| 5 | **ORCH-4** | Docs | **Done** | Closeout sync | Plan + Appendix I updated |

**Suggested PR order (complete):** ORCH-1 → ORCH-2 → ORCH-3 → ORCH-4.

**Explicitly excluded:** K.1, K.2, new graph node types, nested harness per child — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1g Harness implementation queue — governance audit (closed)

**Purpose:** Phase GOV-AUDIT documentation closeout — **closed 2026-06-05**.

| Order | ID | Status | Deliverable |
|-------|-----|--------|-------------|
| 1 | GOV-DOC.1 | **Done** | Appendix H control plane |
| 2 | GOV-DOC.2 | **Done** | Cross-ref sync |
| 3 | GOV-DOC.3 | **Done** | EXTENSION_AUTHOR §10 |
| — | GOV-PROD.1 | **Deferred** | Product dashboard → §6.3 |

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

**Explicitly excluded:** K.1, K.2, new product eval modes requiring business datasets — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1t Harness implementation queue — Adaptive Harness Intelligence (closed)

**Purpose:** Single ordered list for **Phase W-ADAPT** (Band 2y). **Closed 2026-06-02** — **70/70 Done** (Wave W-ADAPT-0 through Wave W-ADAPT-7 **Done**). Maintenance-only; see [§6.1](#61-harness-platform-maintenance-default--band-1).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **W-ADAPT-0.2–0.5** | Docs/Code | **Done** | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold | Import + gate stub |
| 2 | **W-ADAPT-1.1–1.12** | Code | **Done** | Observe (L4-O): signals + utility + report | `phase_w_adapt_report.py` |
| 3 | **W-ADAPT-2.1–2.12** | Code | **Done** | Recommend (L4-R): engines + proposals (no apply) | Proposals in report |
| 4 | **W-ADAPT-3.1–3.7** | Code | **Done** | Shadow (L4-S): ProfileVersionStore + executor.shadow | Integration test green |
| 5 | **W-ADAPT-4.1–4.10** | Code | **Done** | Apply (L4-A): canary, apply, rollback, events | Policy learning HITL enforced |
| 6 | **W-ADAPT-5.1–5.12** | Code/Docs | **Done** | Verify (L4-V): VerificationLoop + runtime L4 closeout | `--enforce-l4-runtime` |
| 7 | **W-ADAPT-6.1–6.5** | Code | **Done** | ProcessPatternMiner + daily scheduler | pattern report |
| 8 | **W-ADAPT-7.1–7.7** | Code/Docs | **Done** | Tier-3 AdaptiveProfile + Appendix V + acceptance | E2E observe→recommend |

**Suggested PR order:** See [Phase W-ADAPT — Suggested PR order](#w-adapt--suggested-pr-order).

**Explicitly excluded:** K.1, K.2, deep RL, foundation model training, autonomous prompt edits — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1v Harness implementation queue — LLM completion response envelope (closed)

**Purpose:** Single ordered list for **Phase M-LLM-R** (Band 2z). **Closed 2026-06-06** — **39/39 Done**. Runs **in parallel** with W-ADAPT waves 5–7 (Tier-0 LLM contract; independent of L4 runtime loop).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **M-LLM-R.0.2–0.3** | Docs | **Done** | ADR-LLM-001 + canon §5.2.2 addendum | Linked from plan |
| 2 | **M-LLM-R.1.1–1.8** | Code | **Done** | Contract types + builders + public exports | Import smoke; no dict returns |
| 3 | **M-LLM-R.2.1–2.6** | Code | **Done** | `LLMAdapter` ABC typed signatures | ABC compiles; stubs updated |
| 4 | **M-LLM-R.3.1–3.7** | Code | **Done** | All provider adapters return envelope | Conformance per provider family |
| 5 | **M-LLM-R.4.1–4.6** | Code | **Done** | Nexus runtime consumers | `test_core_llm_step` + tool planner |
| 6 | **M-LLM-R.5.1–5.3** | Code | **Done** | RAG + websearch + legacy | RAG unit tests green |
| 7 | **M-LLM-R.6.1–6.4** | Code | **Done** | Agents + scaffold + CI lint | `check_llm_adapter_typed_returns.py` + `check_agents_llm_adapter_response.py` |
| 8 | **M-LLM-R.7.1–7.5** | Code | **Done** | Usage alignment + replay/trace bridge | `test_replay_engine` + diagnostics |
| 9 | **M-LLM-R.8.1–8.4** | Docs/CI | **Done** | Docs + conformance + closeout | M-LLM.14 Done; Appendix L complete |

**Suggested PR order:** See [Phase M-LLM-R — Suggested PR order](plan/LLM_ADAPTERS.md).

**Explicitly excluded:** K.1, K.2, product HTTP API DTOs, provider SDK rewrites — [§6.3a](#63a-business-backlog-register-consolidated).

### 6.1w Harness implementation queue — Integration expansion (M.6 P4 closed)

**Purpose:** Ordered backlog for **Phase M.6 P4** (Band 2aa). **Status:** **Done** (2026-06-02) — **28/28 Done** · catalog **127**.  
**Register:** [M.6 P4 — Master register](#m6-p4--master-register-28-slugs) · **Execution order:** [§6.2ae](#62ae-phase-m6-p4-execution-order--done)  
**Policy:** One slug per PR; runs **in parallel** with §6.1 maintenance — pull only when harness ops/adaptive/INT health needs the slug.

| Order | Wave | IDs | Slugs | Priority | Status |
|-------|------|-----|-------|----------|--------|
| 0 | CAT | M-P4-CAT.1, M-P4-CAT.2 | *(categories)* | **P0** | **Done** (beta) |
| 1 | H-INT-1 | M-P4.1–M-P4.4 | `pgvector`, `duckdb`, `influxdb`, `timescaledb` | P0/P1 | **Done** |
| 2 | H-INT-2 | M-P4.5–M-P4.7 | `grafana`, `loki`, `tempo` | **P0** | **Done** |
| 3 | H-INT-3 | M-P4.8–M-P4.11 | `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`, `doppler` | P0/P1 | **Done** |
| 4 | H-INT-4 | M-P4.12–M-P4.16 | `unleash`, `launchdarkly`, `github_actions`, `redpanda`, `cloudflare_r2` | P0/P1 | **Done** |
| 5 | H-INT-5 | M-P4.17–M-P4.28 | `memgraph`, `falkordb`, `incident_io`, `kubernetes`, `servicenow`, `bitbucket`, `asana`, `sendgrid`, `mailgun`, `mlflow`, `huggingface_hub`, `ollama` | P1/P2 | **Done** |

**Per-slug checklist (M.4):** contract → `providers/<category>/<slug>/` → unit tests → `USAGE.md` → `layout.py` → `architecture/INTEGRATIONS.md` → canon §7.1.3 row → gate green → paydown log row.

**Explicitly excluded:** CRM, payments, blockchain, duplicate vector SaaS, LLM vendor APIs — see [M.6 P4 register](#m6-p4--harness-platform-expansion-planned).

### 6.1x Harness implementation queue — Integration depth (M.6 P5 done)

**Purpose:** Closeout record for **Phase M.6 P5** (Band 2ab). **Status:** **Done** (2026-06-02) — **33/34**.  
**Register:** [M.6 P5 — Master register](#m6-p5--master-register-34-slugs) · **Execution order:** [§6.2af](#62af-phase-m6-p5-execution-order-band-2ab--planned)  
**Policy:** One slug per PR (or one harden wave ≤4 slugs); runs **in parallel** with §6.1 maintenance — pull when W-OPS / W-ADAPT / EVAL / prod stack needs the slug.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P5-CAT.1–3 | `ci_cd` extend, `security_scanner`, category mapping | **P0** | **Done** (CAT.2 deferred: `trivy`) |
| 1 | H-INT-6 | M-P5.1–M-P5.10 | Ops/metrics/CI/local cloud: prometheus, clickhouse, vault, pagerduty, github, gitlab_ci, circleci, azure_pipelines, mailpit, localstack | **P0** | **Done** |
| 2 | H-INT-7 | M-P5.11–M-P5.20 | Eval/async/artifacts: langfuse, phoenix, braintrust, mlflow, influxdb, timescaledb, temporal, redpanda, minio, s3 | **P0/P1** | **Done** |
| 3 | H-INT-8 | M-P5.21–M-P5.28 | Data plane lab: neo4j, mongodb, elasticsearch, nats, chroma, weaviate, launchdarkly, signoz | **P1/P2** | **Done** |
| 4 | H-INT-9 | M-P5.29–M-P5.34 | P2 reserve: codecov, trivy, grafana_oncall, opentelemetry_collector, snowflake, supabase | **P2** | **Done** |
| 5 | PRE | M-P5-PRE.1 | Tier-3 presets: `harness_metrics_stack`, `harness_eval_stack`, `harness_async_stack`, `harness_ci_stack` | **P0** | **Done** |

**Explicitly excluded:** Band 3 product agents; see [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334).

### 6.1y Harness implementation queue — Integration expansion (M.6 P6 planned)

**Purpose:** Ordered backlog for **Phase M.6 P6** (Band 2ac). **Status:** **Done** (2026-06-02) — **32/32**.  
**Register:** [M.6 P6 — Master register](#m6-p6--master-register-32-slugs) · **Execution order:** [§6.2ag](#62ag-phase-m6-p6-execution-order-band-2ac--planned)  
**Policy:** One slug per PR (or one CAT wave before first slug in a new category); runs **in parallel** with §6.1 maintenance — pull when security/sandbox/identity/GitOps/speech harness gaps block ops.

| Order | Wave | IDs | Slugs (summary) | Priority | Status |
|-------|------|-----|-----------------|----------|--------|
| 0 | CAT | M-P6-CAT.1–9 | New categories: `security_scanner`, `sandbox_host`, `identity_provider`, `speech_provider`, `workflow_orchestrator`, `vision_serving`, `ml_inference_host`, `billing_meter`, `crm` | **P0** | **Done** |
| 1 | H-INT-10 | M-P6.1–M-P6.4 | Security + secrets: `trivy`, `snyk`, `semgrep`, `infisical` | **P0** | **Done** |
| 2 | H-INT-11 | M-P6.5–M-P6.7 | Cloud sandbox: `e2b`, `modal`, `daytona` | **P0/P1** | **Done** |
| 3 | H-INT-12 | M-P6.8–M-P6.10 | Identity: `auth0`, `keycloak`, `workos` | **P0/P1** | **Done** |
| 4 | H-INT-13 | M-P6.11–M-P6.13 | GitOps CI: `argocd`, `buildkite`, `jenkins` | **P0/P1** | **Done** |
| 5 | H-INT-14 | M-P6.14–M-P6.15 | Speech catalog: `elevenlabs`, `deepgram` | **P0** | **Done** |
| 6 | H-INT-15 | M-P6.16–M-P6.19 | Enterprise ops: `newrelic`, `splunk`, `zendesk`, `statsig` | **P1** | **Done** |
| 7 | H-INT-16 | M-P6.20–M-P6.24 | Data/workflow: `prefect`, `airflow`, `typesense`, `neon`, `pulsar` | **P1** | **Done** |
| 8 | H-INT-17 | M-P6.25–M-P6.32 | Reserve: `algolia`, `confluent`, `backblaze_b2`, `triton`, `replicate`, `stripe`, `salesforce`, `hubspot` | **P2** | **Done** |
| 9 | PRE | M-P6-PRE.1 | Tier-3 presets: `harness_security_stack`, `harness_sandbox_stack`, `harness_identity_stack`, `harness_gitops_stack` | **P0** | **Done** |
| 10 | WIRE | M-P6-WIRE.1–7 | Tool surface + sandbox/speech/identity bridges + promote gate + infra `p6` | **P0** | **Done** |

**Per-slug checklist:** see [M.6 P6 register](#m6-p6--harness-integration-expansion-planned).

**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS`; four Tier-3 presets; gate green.

### 6.1an Harness implementation queue — LLM guardrail integrations (closed)

**Purpose:** Close **Phase M.12 / GR-INT** (Band 2ay) — vendor `llm_guardrail` catalog, Tier-3 middleware bridge, CI, Nexus E2E gate, `GUARDRAIL_BLOCKED` runtime events. **Status:** **Done** (2026-06-09) — **14/14 + M-P12.HARD**.

**Register:** [M.12 — Master register](plan/INTEGRATIONS.md#phase-m12--llm-guardrail-integrations-planned) · **Canon:** [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) §47 · [ADR-GR-001](adr/ADR-GR-001.md)

| ID | Deliverable | Status |
|----|-------------|--------|
| M-P12.HARD.1 | Nexus E2E integration test (`test_nexus_loop_guardrail_wiring.py`) | **Done** |
| M-P12.HARD.2 | `RuntimeEventType.GUARDRAIL_BLOCKED` + middleware emission | **Done** |
| M-P12.HARD.3 | `UAEPBlockedError` → `AgentExecutionStatus.FAILED` bridge | **Done** |
| M-P12.HARD.4 | PLATFORM register row (Band 2ay) | **Done** |

**Deferred (product):** full NeMo Colang, `llama_guard` via inference host, heavy optional deps group — see [M.12 paydown](plan/INTEGRATIONS.md).

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned)**. **Band 2ay M.12** = **Done** — see **[§6.1an](#61an-harness-implementation-queue--llm-guardrail-integrations-closed)**. **Next product work** = [§6.3](#63-end-of-plan--deferred-product-work-only) (product prioritization only).

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/check_harness_no_getattr.py
  python scripts/check_legacy_modules_removed.py
  python scripts/check_agent_skill_resolution.py
  python scripts/check_harness_registry_resolution.py
  python scripts/check_harness_capability_graph_wiring.py
  python scripts/check_legacy_tool_plan_booleans.py
  python scripts/check_trace_bridge_event_catalog.py
  python scripts/check_plugin_catalog.py
  python scripts/check_llm_adapter_typed_returns.py
  python scripts/check_agents_llm_adapter_response.py
  uv run python scripts/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/check_scaffold_harness_alignment.py
  python scripts/check_agents_no_tier3_imports.py
  python scripts/check_intergrax_no_applications_imports.py
  uv run python scripts/check_harness_prompt_golden_catalog.py
  uv run python scripts/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/phase_v_capability_graph_guard.py --enforce
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

**Maintenance depth (2026-06-07):** **OBS-DEPTH.1 Done** — unified run journal. **T10-DEPTH.1 Done** — broker task index + PagerDuty acknowledge adapter. **T-EXPAND T11 Done** — 160 tools. **LEG-DEPTH.1–3 + O.5 depth Done** — planner schema uses `tool_ids`; legacy booleans accepted with deprecation trace; `from_legacy()` gated by `check_legacy_tool_plan_booleans.py`. **OBS-DEPTH.2 Done** — `check_trace_bridge_event_catalog.py` + gate test. **OBS live emit Done** — `RuntimeState.trace_event` → `runtime_event_bus`. **Celery purge_completed Done** — optional KV task index. **notify.dispatch_due Done** — Tier-0 dispatcher tool. **T-EXPAND T12 Done** — 170 tools (health slot probes + notify dispatcher). **T-EXPAND T13 Done** — 172 tools (`eval.judge`, `eval.trajectory` / CRIT-V). **L2→L3 §21 Done** — `test_observability_layer_depth_gate.py` regression gate.

### 6.1ah Harness implementation queue — FAUDIT-32 remediation (closed)

**Status:** **Done** (2026-06-06) — **23/23 Done**  
**Source:** [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md) · **Appendix M**  
**Priority ladder:** **Band 2ad** (§4.0) — runs **after** FAUDIT-TIER.1 on every harness PR that touches `intergrax/runtime/architecture/`

**Execution order (recommended):**

```text
Wave P0 (architecture integrity):
  FAUDIT-TIER.1 → FAUDIT-TIER.2

Wave P1 (identity + intake + observability):
  FAUDIT-INTAKE.1 → FAUDIT-ID.1 → FAUDIT-OBS.1 → FAUDIT-EVAL.1

Wave P2 (control-plane depth):
  FAUDIT-PE.1 → FAUDIT-REG.1 → FAUDIT-CG.1 → FAUDIT-CG.2
  → FAUDIT-SEC.1 → FAUDIT-REL.1 → FAUDIT-COST.1

Wave P3 (orchestration + cognition + memory):
  FAUDIT-ORCH.1 → FAUDIT-SUB.1 → FAUDIT-COG.1 → FAUDIT-LLM.1
  → FAUDIT-POL.1 → FAUDIT-MEM.1 → FAUDIT-ALG.1 → FAUDIT-OPS.1
  → FAUDIT-INTAKE.2 → FAUDIT-ID.2
```

| ID | Status | Priority | Blocks |
|----|--------|----------|--------|
| FAUDIT-TIER.1 | **Done** | **Critical** | `intergrax/applications/reference/harness_manifest_catalog.py` |
| FAUDIT-TIER.2 | **Done** | High | `scripts/check_intergrax_no_applications_imports.py` |
| FAUDIT-INTAKE.1 | **Done** | High | `intergrax/contracts/task_envelope.py` |
| FAUDIT-INTAKE.2 | **Done** | Medium | `tests/unit/runtime/architecture/test_faudit_remediation.py` |
| FAUDIT-ID.1 | **Done** | High | `intergrax/contracts/actor_identity.py` |
| FAUDIT-ID.2 | **Done** | Medium | `DelegationSpec.permission_scopes` |
| FAUDIT-POL.1 | **Done** | High | `PolicyEngine.evaluate_pre_llm/pre_output` |
| FAUDIT-LLM.1 | **Done** | High | `intergrax/llm_adapters/registry/model_router.py` |
| FAUDIT-COG.1 | **Done** | High | `intergrax/contracts/decision_record.py` + UAEP emit |
| FAUDIT-ORCH.1 | **Done** | Medium | `GraphExecutor` inflight backpressure |
| FAUDIT-SUB.1 | **Done** | High | `SubtaskContract` + safer defaults |
| FAUDIT-MEM.1 | **Done** | High | `retention_enforcement.py` + `PolicyScopedMemoryView` STM purge |
| FAUDIT-PE.1 | **Done** | High | `prompt_golden_catalog.py` + `tests/fixtures/prompt_golden/` + CI script |
| FAUDIT-ALG.1 | **Done** | High | lifecycle states + reference agent `owner_team` adoption + CI script |
| FAUDIT-REG.1 | **Done** | High | `HarnessRegistrySnapshot` agent/eval fields |
| FAUDIT-CG.1 | **Done** | High | prompt seeds in `capability_graph_wiring.py` |
| FAUDIT-CG.2 | **Done** | Medium | `phase_v_capability_graph_guard.py` impact log |
| FAUDIT-OBS.1 | **Done** | High | `RuntimeEventType.LLM_CALL/POLICY_DECISION` |
| FAUDIT-REL.1 | **Done** | High | expanded `RuntimeErrorCode` + classifier |
| FAUDIT-SEC.1 | **Done** | High | `intergrax/contracts/data_classification.py` |
| FAUDIT-COST.1 | **Done** | High | `run_budget` wired in `nexus_factory` |
| FAUDIT-EVAL.1 | **Done** | High | `phase_v_closeout_gate.py` eval baseline |
| FAUDIT-OPS.1 | **Done** | Medium | `build/architecture_hardening/release_cycles.json` |

**DoD (§6.1ah queue closure):** All **Planned** rows **Done**; Appendix M scorecard shows **0 Critical**, **≤5 High** (documented deferrals only); tier gate green.

### 6.1ai Harness implementation queue — FAUDIT-32 follow-up (closed)

**Status:** **Done** (2026-06-06) — post-remediation depth for PE/ALG/MEM adoption  
**Priority ladder:** **Band 2ad** (§4.0) — runs after §6.1ah closure

| ID | Status | Deliverable |
|----|--------|-------------|
| FAUDIT-PE.1+ | **Done** | Real `prompts/` golden hashes in `tests/fixtures/prompt_golden/expectations.json`; `scripts/check_harness_prompt_golden_catalog.py`; gate test |
| FAUDIT-ALG.1+ | **Done** | `lifecycle_state` + `owner_team` on reference Tier-2 agents; `scripts/check_agents_lifecycle_metadata.py` |
| FAUDIT-MEM.1+ | **Done** | `should_forget_stm_record` wired in `PolicyScopedMemoryView.read` |

**Explicitly deferred (Band 3 / product):** MEM-9 entity graph memory implementation (RFC only); K.1/K.2 business agents.

### 6.2bo Phase EVAL execution order (Band 2x — closed 2026-06-02)

**Status:** **Done** · register: [Phase EVAL](plan/CRITIC_VERIFICATION.md) · queue: [§6.1s](#61s-harness-implementation-queue--evaluation-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | EVAL-DOC.1 | Appendix U + plan sync | High |
| 2 | EVAL-1 | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | Critical |
| 3 | EVAL-2 | `evaluation_assembly_resolver` | High |
| 4 | EVAL-3 | `check_harness_evaluation_wiring.py` | Medium |

### 6.2bn Phase COST execution order (Band 2w — closed 2026-06-02)

**Status:** **Done** · register: [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) · queue: [§6.1r](#61r-harness-implementation-queue--cost-governance-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | COST-DOC.1 | Appendix T + plan sync | High |
| 2 | COST-1 | `CostProfile` + `cost_runtime_bridge` + `cost_wiring` | Critical |
| 3 | COST-2 | `cost_assembly_resolver` | High |
| 4 | COST-3 | `check_harness_cost_wiring.py` | Medium |

### 6.2bm Phase SEC execution order (Band 2v — closed 2026-06-02)

**Status:** **Done** · register: [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) · queue: [§6.1q](#61q-harness-implementation-queue--security-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | SEC-DOC.1 | Appendix S + plan sync | High |
| 2 | SEC-1 | `security_runtime_bridge` + `security_wiring` | Critical |
| 3 | SEC-2 | `security_assembly_resolver` | High |
| 4 | SEC-3 | `check_harness_security_wiring.py` | Medium |

### 6.2bl Phase REL execution order (Band 2u — closed 2026-06-02)

**Status:** **Done** · register: [Phase REL](plan/OBSERVABILITY.md) · queue: [§6.1o](#61o-harness-implementation-queue--reliability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REL-DOC.1 | Appendix R + plan sync | High |
| 2 | REL-1 | `reliability_runtime_bridge` + `reliability_wiring` | Critical |
| 3 | REL-2 | `reliability_assembly_resolver` | High |
| 4 | REL-3 | `check_harness_reliability_wiring.py` | Medium |

### 6.2bk Phase OBS execution order (Band 2t — closed 2026-06-02)

**Status:** **Done** · register: [Phase OBS](plan/OBSERVABILITY.md) · queue: [§6.1n](#61n-harness-implementation-queue--observability-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | OBS-DOC.1 | Appendix Q + plan sync | High |
| 2 | OBS-1 | `observability_runtime_bridge` + `observability_wiring` | Critical |
| 3 | OBS-2 | `observability_assembly_resolver` | High |
| 4 | OBS-3 | `check_harness_observability_wiring.py` | Medium |

### 6.2bj Phase CG execution order (Band 2s — closed 2026-06-02)

**Status:** **Done** · register: [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1m](#61m-harness-implementation-queue--capability-graph-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CG-DOC.1 | Appendix P + plan sync | High |
| 2 | CG-1 | `capability_graph_wiring` | Critical |
| 3 | CG-2 | `capability_graph_assembly_resolver` | High |
| 4 | CG-3 | `check_harness_capability_graph_wiring.py` | Medium |

### 6.2bi Phase REG execution order (Band 2r — closed 2026-06-02)

**Status:** **Done** · register: [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · queue: [§6.1l](#61l-harness-implementation-queue--registry-architecture-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | REG-DOC.1 | Appendix O + plan sync | High |
| 2 | REG-1 | `HarnessRegistrySnapshot` + `registry_wiring` | Critical |
| 3 | REG-2 | `registry_assembly_resolver` | High |
| 4 | REG-3 | `check_harness_registry_resolution.py` | Medium |

### 6.2bg Phase AS execution order (Band 2q — closed 2026-06-02)

**Status:** **Done** · register: [Phase AS](plan/ORCHESTRATION.md) · queue: [§6.1k](#61k-harness-implementation-queue--agent-assembly-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | AS-DOC.1 | Appendix N + plan sync | High |
| 2 | AS-1 | `agent_assembly_resolver` | Critical |
| 3 | AS-2 | Lifecycle state on `AgentContract` | High |
| 4 | AS-3 | `skill_ids` resolution audit script | Medium |

### 6.2bh Phase CLEAN execution order (closed 2026-06-02)

**Status:** **Done** · register: [Phase CLEAN](plan/ORCHESTRATION.md) · queue: [§6.1j](#61j-harness-implementation-queue--legacy-module-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CLEAN-1 | Remove `chat_router.py` | Critical |
| 2 | CLEAN-2 | Remove `tools_agent.py` | Critical |
| 3 | CLEAN-3 | `check_legacy_modules_removed.py` in CI | High |
| 4 | CLEAN-4 | Docs sync | Low |

### 6.2bf Phase CTX execution order (Band 2n — closed 2026-06-02)

**Status:** **Done** · register: [Phase CTX](plan/MEMORY.md) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | CTX-1 | `context_runtime_bridge` | Critical |
| 2 | CTX-2 | `context_wiring` + Nexus factory wire | High |
| 3 | CTX-DOC.1–2 | Appendix L + plan sync | Low |

### 6.2be Phase RAG execution order (Band 2m — closed 2026-06-02)

**Status:** **Done** · register: [Phase RAG](plan/MEMORY.md) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | RAG-1 | `rag_runtime_bridge` + environment wire | Critical |
| 2 | RAG-DOC.1 | Appendix K §K.5 + plan sync | Low |

### 6.2bd Phase INT execution order (Band 2l — closed 2026-06-02)

**Status:** **Done** · register: [Phase INT](plan/INTEGRATIONS.md) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | INT-1 | `integration_runtime_bridge` | Critical |
| 2 | INT-2 | `integration_health_wiring` | High |
| 3 | INT-DOC.1–2 | Appendix K + plan sync | Low |

### 6.2bc Phase TS execution order (Band 2k — closed 2026-06-02)

**Status:** **Done** · register: [Phase TS](plan/TOOLS.md) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

Work **one TS ID per PR**; after each step update the TS master table + §6.1c + paydown log; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | TS-1 | `catalog_runtime_bridge` + `materialize_runtime_config` | Critical | TS-DOC.* (parallel OK) |
| 2 | TS-2 | Harness host LLM adapter wiring | High | — |
| 3 | TS-3 | `SkillResolverProtocol` | Medium | — |
| 4 | TS-DOC.1–2 | Appendix J + plan sync | Low | TS-1–3 |

### 6.2aj Phase FLOW execution order (Band 2aj — closed 2026-06-07)

**Status:** **Done** · register: [Phase FLOW](plan/ORCHESTRATION.md) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)

Work **one FLOW ID per PR**; after each step update FLOW master table + §6.1aj + Appendix N; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | FLOW-2 | Delegation graph expansion (ADR-FLOW-001) | **Critical** | — |
| 2 | FLOW-14 | `SubtaskContract` on expanded child node | High | FLOW-2 |
| 3 | FLOW-3 | `max_delegation_depth` enforcement | High | FLOW-2 |
| 4 | FLOW-15 | Subagent budget envelope | Medium | FLOW-14 |
| 5 | FLOW-6 | Strict graph cycle detection | High | — |
| 6 | FLOW-1 | LLM-backed Nexus planner | High | — (parallel with 5–8 after step 1) |
| 7 | FLOW-4 | Run-level retry profile | Medium | FLOW-2 |
| 8 | FLOW-13 | `max_inflight_nodes` profile wire | Medium | — |
| 9 | FLOW-7 | Merge policy / composer profile | Medium | — |
| 10 | FLOW-9 | Multi-agent evaluation hooks | Medium | FLOW-7 optional |
| 11 | FLOW-11 | Pre-plan policy hooks | Medium | — |
| 12 | FLOW-5 | `AgentGraph.on_error` wire | Low | FLOW-4 optional |
| 13 | FLOW-10 | Reserved lifecycle states ADR | Low | — |
| 14 | FLOW-12 | `DecisionRecord` regression gate | Medium | — |
| 15 | FLOW-16 | `MODIFY_PLAN` ADR (ADR-FLOW-003) | Low | — |
| 16 | FLOW-17 | `MULTI_AGENT` ordering policy | Low | — |
| 17 | FLOW-DOC.* | Docs closeout | Low | FLOW-1–17 (except deferred FLOW-8) |

### 6.2ak Phase CRIT-V execution order (Band 2ak — closed)

**Status:** **Done** (2026-06-08) · register: [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

Work **one CRIT-V ID per PR**; after each step update CRIT-V master table + §6.1ak; keep §6.1 scripts green.

| Step | ID | Deliverable | Priority | Depends on |
|------|-----|-------------|----------|------------|
| 1 | CRIT-V-0.* | Architecture + ADR + canon + README | High | — |
| 2 | CRIT-V-1.1 | `CriticProfile` on environment profile | **Critical** | CRIT-V-0 |
| 3 | CRIT-V-1.2 | CVL contracts (`CriticRequest`, `CriticVerdict`, …) | **Critical** | CRIT-V-1.1 |
| 4 | CRIT-V-1.3 | `EvaluatorLoopSpec` | High | CRIT-V-1.2 |
| 5 | CRIT-V-2.1 | `eval.judge` tool | **Critical** | CRIT-V-1.2, M-LLM-R |
| 6 | CRIT-V-2.2 | `eval.trajectory` tool | High | CRIT-V-2.1 |
| 7 | CRIT-V-2.3 | Registry observation hook for judge/trajectory | Medium | CRIT-V-2.1 |
| 8 | CRIT-V-3.1 | `CriticOrchestrator` | **Critical** | CRIT-V-2.1 |
| 9 | CRIT-V-3.2–3.3 | L0/L1 gateways | High | CRIT-V-3.1 |
| 10 | CRIT-V-3.4–3.5 | Graph partial + final hooks | High | CRIT-V-3.1 |
| 11 | CRIT-V-3.6 | Critic trace events | Medium | CRIT-V-3.4 |
| 12 | CRIT-V-4.1–4.2 | Evaluator-loop executor + graph wire | High | CRIT-V-3.4 |
| 13 | CRIT-V-5.1–5.2 | Semantic offline eval runner | Medium | CRIT-V-2.1 |
| 14 | CRIT-V-6.1–6.3 | Tier-3 critic wiring + policy + CI | High | CRIT-V-3.1 |
| 15 | CRIT-V-6.4 | Appendix W author map | Medium | CRIT-V-6.1 |
| 16 | CRIT-V-7.1 | FAUDIT-EVAL.1 baseline CI gate | High | CRIT-V-6.3 |
| 17 | CRIT-V-7.2–7.3 | Flow reference sync + lab demo | Medium | CRIT-V-3.6 |

### 6.2bb Phase ORCH execution order (Band 2j — closed 2026-06-05)

**Status:** **Done** · register: [Phase ORCH](plan/ORCHESTRATION.md) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed)

Work **one ORCH ID per PR**; after each step update the ORCH master table + §6.1b + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | Depends on |
|-------|-----|-------------|----------|------------|
| 1 | ORCH-1 | Planner/classifier kind registry + `nexus_factory` wiring | **Critical** | ORCH-DOC.* |
| 2 | ORCH-2 | `graph_spec_to_plan` + planning runner integration | High | ORCH-1 (shared factory path) |
| 3 | ORCH-3 | `max_parallel_nodes` on `OrchestrationProfile` + `GraphExecutor` | Medium | — (parallel OK after ORCH-1) |
| 4 | ORCH-4 | Docs closeout — Appendix I + plan §0.5 | Low | ORCH-1–3 |

### 6.2v Phase V-REM execution order (Band 2i — closed 2026-06-05)

**Status:** **Done** · register: [Phase V-REM](plan/ORCHESTRATION.md) · queue: [§6.1z](#61z-harness-implementation-queue-consolidated) (closed)

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

**Status:** **Done** · register: [Phase W-OPS](plan/PLATFORM_FOUNDATION.md)

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

**Per-application deploy triad gate (AA-APP.0.2):** for each of `lab_application`, `legal_application`, `local_workspace_application`, `poc_template_application`, `research_application` assert:

1. `docker/Dockerfile` + `docker-compose.yml` + `build-docker.sh` / `.bat`
2. `BUILD_AND_DEPLOY.md` present and matches scaffold generator output (or documented drift)
3. `ARCHITECTURE.md` § **Dependencies** lists required `pyproject.toml` extras (e.g. `harness-author`, provider-specific `llm-*`, `dev-ci` for tests)

**Doc pair gate (AA-D0.6):** for each listed Tier-2 agent and Tier-3 application assert `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md` exist and cross-link. Gate: `tests/unit/applications/test_agent_app_doc_pair.py`.

**Success gate for Phase AA platform closeout:** **Met** (2026-06-02) — conformance matrix **OK**; legal tree = scaffold; `lab_application` on `build_harness_host_runtime`; AA-APP.0.2 green; gate **533**. **Full AA register closeout** additionally requires Band 3 domain rows **Done** or explicitly **Deferred** (current policy: **Deferred**).

**Explicitly out of NOW:** K.1/K.2 implementation, Legal **live LLM** E2E (Band 3), new product hosts beyond the four listed, Legal UAEP step port (AA-LEG.2.2+) unless product reprioritizes §6.3.

### 6.2aa Phase MEM execution order (Band 2h — closed)

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

### 6.2ab Phase MEM-DEPTH execution order (Band 2am — closed)

**Status:** **Done** (2026-06-08) · **26/26 Done** · canonical register: [Phase MEM-DEPTH — Master deliverables register](plan/MEMORY.md#mem-depth--master-deliverables-register-all-26-tasks).

Work **one MEM-DEPTH ID per PR**; after each step update the MEM-DEPTH master table + paydown log; keep §6.1 scripts green. **Start with MEM-DEPTH-0.3 (ADR) then MEM-DEPTH-1.*** — architecture decision before Context Compiler code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| MEMD0 | MEM-DEPTH-0.1–0.4 | 4 | Canon doc, plan register, ADR, cross-links |
| MEMD1 | MEM-DEPTH-1.1–1.6 | 6 | **P0** — Context Compiler + never-overflow |
| MEMD2 | MEM-DEPTH-2.1–2.2 | 2 | **P1** — Mongo session persistence parity |
| MEMD3 | MEM-DEPTH-3.1–3.5 | 5 | **P1** — Lifecycle automation |
| MEMD4 | MEM-DEPTH-4.1–4.3 | 3 | **P1** — Explore / discovery |
| MEMD5 | MEM-DEPTH-5.1–5.6 | 6 | **P2/P3** — Entity graph, temporal validity, quality gates |
| **Total** | | **26** | |

**Success gate:** P0 + P1 **Done**; never-overflow acceptance green; Memory Layer **L3+** on FAUDIT re-run.

**Explicitly out of NOW:** K.1/K.2, Mem0 SaaS, entity graph ship without §6.3 decision (MEM-DEPTH-5.1).

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

Full task register: [Appendix I](plan/PLATFORM_FOUNDATION.md).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (planned).

### 6.2ag Phase M.6 P6 execution order (Band 2ac — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P6](#m6-p6--harness-integration-expansion-planned) · queue: [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned)

```text
Wave H-INT-0 (categories):  M-P6-CAT.1 → M-P6-CAT.2 → M-P6-CAT.3 → M-P6-CAT.4 → M-P6-CAT.5 → M-P6-CAT.6 → M-P6-CAT.7 → M-P6-CAT.8 → M-P6-CAT.9
Wave H-INT-10 (security):   M-P6.1 → M-P6.2 → M-P6.3 → M-P6.4
Wave H-INT-11 (sandbox):    M-P6.5 → M-P6.6 → M-P6.7
Wave H-INT-12 (identity):   M-P6.8 → M-P6.9 → M-P6.10
Wave H-INT-13 (gitops CI):  M-P6.11 → M-P6.12 → M-P6.13
Wave H-INT-14 (speech):     M-P6.14 → M-P6.15
Wave H-INT-15 (enterprise): M-P6.16 → M-P6.17 → M-P6.18 → M-P6.19
Wave H-INT-16 (data/wf):    M-P6.20 → M-P6.21 → M-P6.22 → M-P6.23 → M-P6.24
Wave H-INT-17 (reserve):    M-P6.25 → M-P6.26 → M-P6.27 → M-P6.28 → M-P6.29 → M-P6.30 → M-P6.31 → M-P6.32
Wave PRE (presets):         M-P6-PRE.1  (after H-INT-10 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P5 **Done**; M-P5.FU wiring **Done**; Phase SEC closeout **Done** (V-SEC patterns for `security_scanner`).  
**Parallelism:** H-INT-10 unblocks STABLE promote gate; H-INT-11 unblocks cloud `sandbox.exec`; H-INT-12 unblocks multi-tenant hosts; H-INT-14 unifies speech catalog.  
**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.  
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.2ae Phase M.6 P4 execution order (Band 2aa — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P4](#m6-p4--harness-platform-expansion-done) · queue: [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed)

```text
Wave H-INT-0 (categories):  M-P4-CAT.1 → M-P4-CAT.2  (before first slug in new category)
Wave H-INT-1 (storage):     M-P4.1 → M-P4.2 → M-P4.3 → M-P4.4
Wave H-INT-2 (obs stack):   M-P4.5 → M-P4.6 → M-P4.7
Wave H-INT-3 (secrets):     M-P4.8 → M-P4.9 → M-P4.10 → M-P4.11
Wave H-INT-4 (control):     M-P4.12 → M-P4.13 → M-P4.14 → M-P4.15 → M-P4.16
Wave H-INT-5 (enterprise):  M-P4.17 → M-P4.18 → M-P4.19 → M-P4.20 → M-P4.21 → M-P4.22 → M-P4.23 → M-P4.24 → M-P4.25 → M-P4.26 → M-P4.27 → M-P4.28
```

**Prerequisites:** Phase M core + M.6 P1/P2/P3 **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** Any wave after H-INT-0 may start when a slug is needed — prefer H-INT-1 → H-INT-2 → H-INT-3 order for W-OPS/adaptive unblock.  
**Closeout:** **Done** — catalog **127** in `layout.py`; `tests/unit/integrations/providers/test_p5_m6_p4_providers.py` (42 tests).

### 6.2ad Phase M-LLM-R execution order (Band 2z — closed 2026-06-06)

**Status:** **Done** · register: [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · queue: [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

```text
Wave M-LLM-R-0 (planning):     M-LLM-R.0.2 → 0.3  (0.1 **Done**)
Wave M-LLM-R-1 (contracts):    M-LLM-R.1.1 → 1.8
Wave M-LLM-R-2 (ABC):          M-LLM-R.2.6 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5
Wave M-LLM-R-3 (providers):    M-LLM-R.3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7
Wave M-LLM-R-4 (Nexus):        M-LLM-R.4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
Wave M-LLM-R-5 (RAG/web):      M-LLM-R.5.1 → 5.2 → 5.3
Wave M-LLM-R-6 (agents):       M-LLM-R.6.1 → 6.2 → 6.3 → 6.4
Wave M-LLM-R-7 (obs/replay):   M-LLM-R.7.1 → 7.2 → 7.3 → 7.4 → 7.5
Wave M-LLM-R-8 (closeout):     M-LLM-R.8.1 → 8.2 → 8.3 → 8.4
```

**Prerequisites:** Phase M-LLM **Done** (M-LLM.1–13); no dependency on W-ADAPT runtime L4 gate.

**Parallelism:** May run alongside W-ADAPT-5+; coordinate M-LLM-R.7.5 with W-ADAPT signal work if both touch `signal_collector.py`.

**Closeout gate:** `scripts/check_llm_adapter_typed_returns.py` + `scripts/check_agents_llm_adapter_response.py` + full `tests/unit/llm_adapters/` gate green (M-LLM-R.8.3, M-LLM-R.6.4).

### 6.2ac Phase W-ADAPT execution order (Band 2y — closed)

**Status:** **Done** (2026-06-02) · register: [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · queue: [§6.1t](#61t-harness-implementation-queue--adaptive-harness-intelligence-closed)

```text
Wave W-ADAPT-0 (planning):        W-ADAPT-0.2 → 0.3 → 0.4 → 0.5  (**Done**)
Wave W-ADAPT-1 (observe L4-O):    W-ADAPT-1.1 → 1.12  (**Done**)
Wave W-ADAPT-2 (recommend L4-R):  W-ADAPT-2.1 → 2.12  (**Done**)
Wave W-ADAPT-3 (shadow L4-S):      W-ADAPT-3.1 → 3.2 → 3.3 → 3.4 → 3.6 → 3.7 → 3.5  (**Done**)
Wave W-ADAPT-4 (apply L4-A):       W-ADAPT-4.1 → 4.10  (**Done**)
Wave W-ADAPT-5 (verify L4-V):      W-ADAPT-5.1 → 5.3 → 5.4 → 5.5 → 5.2 → 5.11 → 5.6 → 5.7 → 5.8 → 5.9 → 5.10 → 5.12  (**Done**)
Wave W-ADAPT-6 (patterns):         W-ADAPT-6.2 → 6.1 → 6.3 → 6.5 → 6.4  (**Done**)
Wave W-ADAPT-7 (Tier-3 + docs):    W-ADAPT-7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 7.6 → 7.7  (**Done**)
```

**Prerequisites:** Phase V + V-REM + W-OPS + EVAL + COST + CG closeouts **Done**.

**Runtime L4 gate:** `uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime` (added in W-ADAPT-5.6).

### 6.2 Harness architecture hardening (Band 2 — Phase V) — Done

**Status:** **Done** (2026-06-05) — Phase V contracts + V-REM runtime enforcement complete. Closeout: `phase_v_closeout_gate.py --enforce --enforce-l4`.

| Item | Status | Notes |
|------|--------|-------|
| V-CG … V-V6 | **Done** | Governance + CI + runtime enforcement |
| V-REM | **Done** | 10/10 closed — §6.1z queue closed |
| W-ML | **Done** | [architecture/MODALITY.md](architecture/MODALITY.md) |
| P-Ext | **Done** | Appendix I |
| M.6 P5 / M.6 P6 / R-Skill expansion | **On demand** | W-OPS.10, W-OPS.8, §6.1x, §6.1y |

**Forbidden in Band 2/2d:** K.1, K.2, product-specific skills, new product application hosts.

---

### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](plan/PLATFORM_FOUNDATION.md) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>/` beyond lab + reference hosts | **Deferred** | Product decision only — confirmed 2026-06-09; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar/` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work. **Product:** [`local_workspace_application`](../applications/local_workspace_application/) — Local Knowledge Workspace (LKW) — first business environment after harness GA; see [ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md).

### 6.3a Business backlog register (consolidated)

**Single register for Band 3 and AA domain-deferred rows.** Do not duplicate in harness session summaries.

| ID | Deliverable | Module | Priority | Depends on |
|----|-------------|--------|----------|------------|
| **LKW.0** | Local Knowledge Workspace — scaffold + architecture baseline | `agents/local_{indexer,search,synthesizer}/`, `applications/local_workspace_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **LKW.1** | Wave 1 — ingest + search smoke on explicit paths | `agents/local_*/steps/` | **High** | LKW.0 |
| **LKW.2** | Multi-agent pipeline (`local.workspace.pipeline` graph) | `local_workspace_application/` + Nexus graph | High | LKW.1 |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | `intergrax/tools/providers/filesystem/` | Medium | LKW.1 |
| **LKW.4** | Background ingest queue + incremental index | Tier-0 queue + Tier-3 worker | Medium | LKW.2 |
| **LKW.5** | `LKW_DATA_HOME` + Chroma persistent local index | `local_workspace_application/host/settings.py` | High | LKW.1 |
| **LKW.6** | Local OS daemon (Win/Linux/macOS) + interaction intake on host | `local_workspace_application/` | High | LKW.1 |
| **LKW.6b** | Slack Socket Mode + slash command → Nexus (interaction surface) | Tier-3 + `slack` integration | Medium | LKW.6 |
| **LKW.7** | Background file watcher + incremental index + optional Slack notify | Tier-0 queue + Tier-3 worker | Medium | LKW.3 |
| **LKW.8** | Tray / file-picker UI (localhost HTTP/MCP client) | Product (out of harness) | Low | LKW.6 |
| **DSW.0** | Dispute Simulation Workspace — scaffold + architecture baseline | `agents/dispute_{intake,analyst,strategist,scenario}/`, `applications/dispute_sim_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **DSW.1** | Wave 1 — case intake + RAG ingest + timeline artifact | `agents/dispute_intake/steps/` | **High** | DSW.0 |
| **DSW.2** | Multi-agent pipeline (`dispute.pipeline` graph) | `dispute_sim_application/` + Nexus graph | High | DSW.1 · **Harness:** CFG-06 proven in `test_orchestration_cfg_simulation.py`; product wiring §6.3 |
| **DSW.3** | Analyst matrix + strategist brief domain steps | `agents/dispute_analyst/`, `agents/dispute_strategist/` | High | DSW.1 |
| **DSW.4** | Scenario variants + correspondence review + HITL | `agents/dispute_scenario/` | High | DSW.3 |
| **DSW.5** | Optional subgraph to `legal.review` for clause drill-down | Nexus graph | Medium | DSW.3 |
| **DSW.6** | Case persistence + retention policy | `dispute_sim_application/host/settings.py` | Medium | DSW.1 |
| **DSW.7** | Polish dispute eval fixtures + regression | `tests/` / agent eval | Medium | DSW.4 |
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

---

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
| 2026-05-30 | M.7-agent-guide-integrations | `guides/AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `guides/AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
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
| L-03 | `architecture/LLM_ADAPTERS.md` missing provider table | Q-L.3 | Done |
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
| 2026-06-01 | docs-consolidation | Merged LLM/RAG observability, retry, trace ADR into canon + `architecture/LLM_ADAPTERS.md`; removed satellite `docs/*.md` |
| 2026-06-01 | Q-N.1,Q-X.2,Wave 9 | `graph_runner`, `task_events`, `lifecycle_bridge`; UAEP `execution_options_for_request`; gate **417 passed** |
| 2026-06-01 | Q-X.2(partial),Q-X.4,Q-X.5 | Legacy metadata warnings; `tools_base` timeline; M.6 beta slugs; gate **415 passed** |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** — **achieved** (Phase Q complete).

**Appendix B relationship:** Closed by Phase Q where mapped. Residual items tracked in **Appendix D** (Phase Q+).

---



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
| 2026-06-01 | **Phase S** | harness_lab_stack, harness.* skills, OTEL profile, guides/HARNESS_ENVIRONMENT.md, tests |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase S definition of done met — **yes** (2026-06-01).

---



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
| DOC-01 | `guides/HARNESS_ENVIRONMENT.md` claims policy bundle wired — lab does not apply bridge | U-Doc.1, U-Pol.1 | Done |
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
| Tier model and runtime boundaries | canon §5.1, §7.0–§7.4, §42 | ideal §3, §26 | §0.2, §2 map, Phases L/Q+/U, **FAUDIT-TIER.\*** | **Done** — reference manifest catalog in `intergrax/applications/reference/` + CI gate |
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
| **Adaptive Harness Intelligence (L4 runtime closed loop)** | canon §54 | ideal §25 | **Phase W-ADAPT** · AHIA | **Done** (Band 2y, 70/70) — L4 runtime closed; observe/recommend/apply/verify per AHIA |
| Observability and runtime traceability | canon §33, §42.24 · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | ideal §11 | Phases OBS + OBS-DEPTH.* + **Phase OBS-BUS** | **L4 Done** — spine, typed payloads, emitter, emission coverage, journal export; gate: `check_observability_gates.py` |
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
| P-Ext.0.4 | All | `guides/EXTENSION_AUTHOR_GUIDE.md` (EN) | **Done** | P0 |
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
| 2026-06-02 | M.6 P5 closeout | Catalog **135** full (`12` core); timeline 99→127→135; P-Ext integration counts synced |
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



**Purpose:** 100% mapping from **Partial** audit findings (2026-06-05) to concrete remediation IDs. **Canonical phase narrative:** [Phase V-REM](plan/ORCHESTRATION.md).

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



**Purpose:** 100% mapping from [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) (AHIA) to concrete **W-ADAPT.\*** IDs. **Canonical phase narrative:** [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md).

**Status:** **70/70 Done** (Band 2y closed 2026-06-05) — Waves W-ADAPT-0 through W-ADAPT-7 complete.

### K.1 AHIA component → W-ADAPT ID matrix

| AHIA component (§9) | Existing module to reuse | W-ADAPT ID |
|---------------------|--------------------------|------------|
| SignalCollector | `metrics/export.py`, `execution_guard.py`, `online_evaluation_registry.py` | W-ADAPT-1.4–1.11 |
| HarnessOutcomeSignal + utility | — (new) | W-ADAPT-1.1, W-ADAPT-1.8 |
| SignalStore | — (new SQLite) | W-ADAPT-1.3 |
| BanditStateStore | — (new) | W-ADAPT-2.1 |
| RoutingTuningEngine | `rag/routing/query_router.py`, LLM profiles | W-ADAPT-2.2, W-ADAPT-3.7, W-ADAPT-4.10 |
| ExecutionStrategyEngine | `history_evaluator.py`, `nexus_factory.py` | W-ADAPT-2.3, W-ADAPT-4.10 |
| PolicyLearningEngine | `adaptive_governance.py`, `tool_security.py` | W-ADAPT-2.4, W-ADAPT-4.6, W-ADAPT-4.9 |
| EvaluationFeedbackEngine | `evaluation_registry_trends.py` | W-ADAPT-2.5, W-ADAPT-5.3 |
| ProposalBuilder | `adaptive_governance.py` (`AdaptiveLoopProposal`) | W-ADAPT-2.6 |
| AdaptationEngine facade | — (new) | W-ADAPT-2.7 |
| Governance gate | `adaptive_governance.py`, `capability_graph_compatibility.py` | W-ADAPT-2.8–2.9 |
| ProfileVersionStore | — (new; pattern from `agent_promotion.py`) | W-ADAPT-3.1–3.2, W-ADAPT-3.5 |
| AdaptationExecutor | `runtime_governance_bridge.py` (extend) | W-ADAPT-3.3–3.4, W-ADAPT-4.4–4.5, W-ADAPT-4.8 |
| VerificationLoop | `evaluation_registry_trends.py`, `execution_guard.py` | W-ADAPT-5.1–5.5 |
| ProcessPatternMiner | trace persistence | W-ADAPT-6.* |
| AdaptationScheduler | Celery/message bus pattern from W-ML | W-ADAPT-2.12, W-ADAPT-5.12, W-ADAPT-6.5 |
| AdaptiveProfile (Tier-3) | `environment_profile.py` | W-ADAPT-4.1, W-ADAPT-7.1–7.2 |
| Ops reports / CI | `phase_v_governance_report.py` pattern | W-ADAPT-1.12, W-ADAPT-2.11, W-ADAPT-5.6–5.8 |
| Runtime L4 evidence | `maturity_gate_evidence.py` | W-ADAPT-5.7, W-ADAPT-5.11 |
| Author docs | AGENT_CREATION_GUIDE appendices | W-ADAPT-7.3–7.4 |

### K.2 Adaptive loop kind → implementation wave

| `AdaptiveLoopKind` | Engine | Apply wave | Authority default |
|--------------------|--------|------------|-------------------|
| `ROUTING_TUNING` | W-ADAPT-2.2 | W-ADAPT-4.10 | RECOMMEND |
| `EXECUTION_STRATEGY_TUNING` | W-ADAPT-2.3 | W-ADAPT-4.10 | RECOMMEND |
| `POLICY_LEARNING` | W-ADAPT-2.4 | W-ADAPT-4.6, W-ADAPT-4.9 | AUTO_WITH_HUMAN_GATE |
| `EVALUATION_FEEDBACK` | W-ADAPT-2.5 | observe only (W-ADAPT-5.3) | OBSERVE_ONLY |

### K.3 Lifecycle mode → task coverage

| Mode | Code | Primary tasks |
|------|------|---------------|
| Observe | L4-O | W-ADAPT-1.* |
| Recommend | L4-R | W-ADAPT-2.* |
| Shadow | L4-S | W-ADAPT-3.* |
| Canary | L4-C | W-ADAPT-4.3 |
| Apply | L4-A | W-ADAPT-4.4–4.10 |
| Verify | L4-V | W-ADAPT-5.* |

### K.4 Paydown log

| Date | W-ADAPT ID | Summary |
|------|------------|---------|
| 2026-06-05 | W-ADAPT-1.1–1.12 | Observe (L4-O): contracts, SignalStore, SignalCollector, Nexus/Runtime hooks, `phase_w_adapt_report.py` |
| 2026-06-05 | W-ADAPT-0.2–0.5 | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold + gate import tests |
| 2026-06-05 | W-ADAPT-0.1 | Phase W-ADAPT register + §6.1t + §6.2ac + Appendix K + Band 2y |
| 2026-06-02 | W-ADAPT-2.1–2.12 | Recommend (L4-R): AdaptationEngine, ProposalBuilder, bandit store, proposal report |
| 2026-06-02 | W-ADAPT-3.1–3.7 | Shadow (L4-S): ProfileVersionStore, shadow executor, integration tests |
| 2026-06-02 | W-ADAPT-4.1–4.10 | Apply (L4-A): canary, apply, rollback, policy-learning HITL |
| 2026-06-02 | W-ADAPT-5.1–5.12 | Verify (L4-V): VerificationLoop, auto-rollback, L4 runtime closeout gate, runbooks |
| 2026-06-02 | W-ADAPT-6.1–6.5 | ProcessPatternMiner, trace sequence reader, pattern report export |
| 2026-06-02 | W-ADAPT-7.1–7.7 | Tier-3 AdaptiveProfile wiring, debug routes, business outcome webhook, acceptance E2E |
| 2026-06-02 | W-ADAPT-OPS | Lab L4-O observe default (`LAB_ADAPTIVE_OBSERVE`); CI/release `--enforce-l4-runtime`; canon §54 + AHIA sync |

---



**Source:** Tier-0 LLM adapter audit (2026-06-06) — plain `str` / `Dict[str, Any]` returns insufficient for production observability, replay, cost attribution, and L4 adaptive signals.

**Phase register:** [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · **Band 2z** · queue [§6.1v](#61v-harness-implementation-queue--llm-completion-response-envelope-closed)

### L.1 Audit finding → remediation map

| # | Audit finding | Remediation | Task IDs |
|---|---------------|-------------|----------|
| 1 | `generate_messages` returns bare `str` | `LLMAdapterResponse` with `content: str` | M-LLM-R.1.1, M-LLM-R.2.1, M-LLM-R.3.*, M-LLM-R.4–6.* |
| 2 | `generate_with_tools` returns `Dict[str, Any]` | Same envelope; `tool_calls: tuple[LLMToolCall, ...]` | M-LLM-R.1.3, M-LLM-R.1.7, M-LLM-R.2.2, M-LLM-R.4.2 |
| 3 | Streaming yields `str` / dict chunks | `LLMStreamEvent` partial/final | M-LLM-R.1.5, M-LLM-R.2.3–2.4, M-LLM-R.3.6 |
| 4 | `generate_structured` return untyped | `LLMStructuredResult[T]` | M-LLM-R.1.6, M-LLM-R.2.5, M-LLM-R.3.7 |
| 5 | SDK `finish_reason` / stop metadata lost | `LLMFinishReason` on response | M-LLM-R.1.1, M-LLM-R.3.1–3.4 |
| 6 | Provider `response_id` / request correlation lost | `response_id: str \| None` on response | M-LLM-R.1.1, M-LLM-R.3.1 |
| 7 | Cached / reasoning tokens discarded | `LLMTokenUsage.cached_input_tokens`, `reasoning_tokens` | M-LLM-R.1.2, M-LLM-R.3.1 |
| 8 | Refusal / content-filter signals lost | `refusal: str \| None` + finish_reason enum | M-LLM-R.1.1, M-LLM-R.3.1–3.2 |
| 9 | Usage only via side-channel (`LLMAdapterUsageLog`) | Per-call `usage` on response + aligned `end_call` | M-LLM-R.1.2, M-LLM-R.2.6, M-LLM-R.7.1 |
| 10 | Inconsistent token counting (estimate vs SDK) | Prefer SDK counts; flag estimate in `LLMProviderExtensions` | M-LLM-R.3.5, M-LLM-R.1.4 |
| 11 | No extensibility without dict bags | `LLMProviderExtensions` tagged union | M-LLM-R.1.4 |
| 12 | Replay `LLMCallInfo` not populated from adapter | Trace bridge from `LLMAdapterResponse` | M-LLM-R.7.2, M-LLM-R.7.3 |
| 13 | `CoreLLMAdapterReturnedDiagV1` tracks `adapter_return_type="str"` | Diagnostics carry finish_reason + tokens | M-LLM-R.7.4 |
| 14 | Conformance enforces `isinstance(text, str)` | Typed conformance helpers | M-LLM-R.8.2 |
| 15 | ~50 call sites assume `str` | Full consumer refactor (Nexus, RAG, agents, websearch) | M-LLM-R.4.*, M-LLM-R.5.*, M-LLM-R.6.* |
| 16 | `make_tool_result` dict factory | Delete; typed `build_adapter_response` | M-LLM-R.1.7 |
| 17 | Public API missing response types | Re-export from `llm_adapters/__init__.py` | M-LLM-R.1.8 |
| 18 | Docs describe two-layer usage but not response envelope | `architecture/LLM_ADAPTERS.md` envelope section | M-LLM-R.8.1 |
| 19 | No CI guard against regression to `str` returns | `check_llm_adapter_typed_returns.py` | M-LLM-R.8.3 |

### L.2 Consumer inventory (must migrate)

| Area | Modules | Task |
|------|---------|------|
| Nexus core LLM | `core_llm_step.py` | M-LLM-R.4.1 |
| Tool planning | `tool_planning_service.py` | M-LLM-R.4.2 |
| Planning / history | `plan_sources.py`, `engine_history_layer.py` | M-LLM-R.4.3 |
| Profile services | `user_profile/*`, `organization/*`, `session_memory_consolidation_service.py` | M-LLM-R.4.4 |
| Supervisor | `supervisor.py` | M-LLM-R.4.5 |
| RAG | `query_refiner.py`, `query_expander.py`, `chunk_enricher.py`, `llm_graph_indexer.py` | M-LLM-R.5.1 |
| Websearch | `websearch_context_generator.py`, `websearch_answerer.py` | M-LLM-R.5.2 |
| Legacy RAG | `legacy/rag_answers/pipeline/answer_pipeline.py` | M-LLM-R.5.3 |
| Agents (Tier-2) | `agents/*/steps/pipeline.py`, `mock_agents.py` | M-LLM-R.6.1 |
| Scaffold / tests | `scaffold/new_agent.py`, `testing_support/builder.py` | M-LLM-R.6.2–6.3 |
| All providers | `llm_adapters/providers/*` | M-LLM-R.3.* |

### L.3 Paydown log

| Date | M-LLM-R ID | Summary |
|------|------------|---------|
| 2026-06-06 | M-LLM-R.0.1 | Phase M-LLM-R register + §6.1v + §6.2ad + Appendix L + Band 2z |
| 2026-06-06 | M-LLM-R.* | Typed `LLMAdapterResponse` envelope; providers + consumers migrated; gate **755** passed |
| — | — | *(append row per merged PR)* |

---



**Purpose:** 100% mapping from 32-layer [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8 audit to concrete **FAUDIT.\*** remediation IDs. **Canonical phase narrative:** [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md).

**Status:** **Done** (2026-06-06) · **23/23 remediation Done** + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed) follow-up · gate **901**

### M.1 Layer → FAUDIT ID matrix (High + Critical only)

| Layer | AUDIT_MAP § | Gap summary | Severity | FAUDIT ID |
|-------|-------------|-------------|----------|-----------|
| Tier boundaries | §2 | `intergrax/runtime/architecture/capability_graph_applications.py` imports `applications.*` | **Critical** | FAUDIT-TIER.1, FAUDIT-TIER.2 |
| Task intake | §3 | No `TaskEnvelope`; worker≡HTTP parity incomplete | High | FAUDIT-INTAKE.1, FAUDIT-INTAKE.2 |
| Identity | §4 | No service/agent identity; delegation scope | High | FAUDIT-ID.1, FAUDIT-ID.2 |
| Policy | §5 | Pre-LLM/pre-output hooks absent | High | FAUDIT-POL.1 |
| LLM adapters | §6 | No policy-driven routing | High | FAUDIT-LLM.1 |
| Cognition | §7 | No `DecisionRecord` per step | High | FAUDIT-COG.1 |
| Orchestration | §9 | No backpressure | High | FAUDIT-ORCH.1 |
| Subagents | §10 | No `SubtaskContract` | High | FAUDIT-SUB.1 |
| Memory | §15 | Entity graph memory; STM retention | High | FAUDIT-MEM.1 |
| Prompts | §17 | No golden prompt CI | High | FAUDIT-PE.1 |
| Registry | §19 | Snapshot omits agents/eval | High | FAUDIT-REG.1 |
| Capability graph | §20 | Missing prompt nodes; no release impact gate | High | FAUDIT-CG.1, FAUDIT-CG.2 |
| Observability | §21 | Missing `LLM_CALL`/`POLICY_DECISION` events | High | FAUDIT-OBS.1 |
| Reliability | §22 | Shallow error taxonomy | High | FAUDIT-REL.1 |
| Security | §23 | No `DataClassification` | High | FAUDIT-SEC.1 |
| Cost | §24 | Tenant attribution not mandatory | High | FAUDIT-COST.1 |
| Evaluation | §25 | Release baseline not CI-enforced | High | FAUDIT-EVAL.1 |
| Lifecycle | §31 | State catalog mismatch; weak adoption | High | FAUDIT-ALG.1 |
| Ops / SLOs | §30 | `release_cycles.json` artifact policy | High | FAUDIT-OPS.1 |

### M.2 Cross-layer themes

| Theme | Layers affected | Risk |
|-------|-----------------|------|
| **Closeout vs maturity** | §17–§25, §31 | Plan **Done** on wiring; AUDIT_MAP **L2** on depth — do not conflate |
| **Dual-path telemetry** | §21, §6 | **L4 Done:** [Phase OBS-BUS](plan/OBSERVABILITY.md) — unified journal, `ObservabilityEmitter`, typed payloads, emission coverage, journal export |
| **Tier boundary drift** | §2, §28 | Single Critical violation undermines canon §7.4.4 |
| **Identity / intake naming** | §3, §4 | Resolved — `TaskEnvelope` in `intergrax/contracts/task_envelope.py`; parity tests in `test_faudit_remediation.py` |

### M.3 Paydown log

| Date | FAUDIT ID | Summary |
|------|-----------|---------|
| 2026-06-06 | FAUDIT-32.0 | Full 32-layer audit (`scope: C`, `audit-and-fix`); scorecard + §6.1ah queue + Appendix M; gate **893**; boundary scripts OK |
| 2026-06-06 | FAUDIT-TIER.1–OPS.1 | **23/23** remediation implemented; tier gate + intake + observability + registry depth |
| 2026-06-06 | FAUDIT-PE.1+/ALG.1+/MEM.1+ | Golden prompt CI, reference agent lifecycle metadata, STM retention wiring; gate **901** |
| 2026-06-09 | PLAN-SYNC | Hub closeout: MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG, CRIT-V, ORCH-5; Appendix M refresh |
| 2026-06-09 | CFG-HOST | Reference host presets (`with_reference_host_platform_defaults`); dispute_sim interactions; scheduler defaults |
| 2026-06-07 | OBS-DEPTH.* + T12 + LEG depth | Unified journal + trace bridge gate + live bus emit + 170-tool catalog + §21 L3 depth gate; gate **967** |
| 2026-06-07 | T13 + CRIT-V-2.* | `eval.judge` + `eval.trajectory`; catalog **172**; doc sync; gate **990** |
| 2026-06-07 | CRIT-V-3.1–3.3 | `CriticOrchestrator`, `L0Gateway`, `L1Gateway`, `CriticEvalToolClient` | gate **996** |

---



**Source:** [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25 · [ADR-FLOW-001](adr/ADR-FLOW-001.md)

**Phase register:** [Phase FLOW](plan/ORCHESTRATION.md) · **Band 2aj** · queue [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed) · execution [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--closed-2026-06-07)

**Status:** **Done** (2026-06-07) · **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3)

> **Note:** Distinct from `guides/AGENT_CREATION_GUIDE.md` Appendix N (agent assembly). This appendix maps **orchestration runtime depth** gaps only.

### N.1 FLOW-GAP → FLOW ID matrix (complete)

| Gap ID | Category | Severity | FLOW ID | Deliverable | AUDIT_MAP § |
|--------|----------|----------|---------|-------------|-------------|
| FLOW-GAP-01 | Runtime-core | High | FLOW-1 | Real `EngineBackedNexusPlanner` | §7 |
| FLOW-GAP-02 | Runtime-core | **Critical** | FLOW-2 | ADR-FLOW-001 delegation expansion | §10 |
| FLOW-GAP-03 | Runtime-core | Medium | FLOW-3 | `max_delegation_depth` enforcement | §10 |
| FLOW-GAP-04 | Runtime-core | Medium | FLOW-4 | Opt-in run-level retry | §9, §22 |
| FLOW-GAP-05 | DX | Low | FLOW-5 | `AgentGraph.on_error` wire | §9 |
| FLOW-GAP-06 | Runtime-core | Medium | FLOW-6 | Strict cycle detection | §9 |
| FLOW-GAP-07 | Production-hardening | Medium | FLOW-7 | `MergePolicy` / composer profile | §9 |
| FLOW-GAP-08 | DX / lifecycle | Low | FLOW-10 | Reserved lifecycle states ADR | §8 |
| FLOW-GAP-09 | Production-hardening | Medium | FLOW-11 | Pre-plan policy hooks | §5 |
| FLOW-GAP-10 | Product-proof | Harness + Product | FLOW-8 | Harness sim **Partial**; product §42.43 **Deferred** §6.3 | §28 |
| FLOW-GAP-11 | Production-hardening | Medium | FLOW-9 | Multi-agent eval hooks | §25 |
| FLOW-GAP-12 | Runtime-core | Medium | FLOW-13 | `max_inflight_nodes` profile + factory wire | §9 |
| FLOW-GAP-13 | Runtime-core | Medium | FLOW-14 | `SubtaskContract` in delegation expansion | §10 |
| FLOW-GAP-14 | Production-hardening | Medium | FLOW-15 | Subagent budget envelope enforcement | §10 |
| FLOW-GAP-15 | DX | Low | FLOW-16 | `MODIFY_PLAN` reserved semantics ADR | §9 |
| FLOW-GAP-16 | DX | Low | FLOW-17 | `MULTI_AGENT` deterministic ordering policy | §9 |
| §24 / FAUDIT-COG-1 | Cognition | Medium | FLOW-12 | `DecisionRecord` regression gate | §7 |
| — | Docs | Low | FLOW-DOC.* | Flow reference + plan sync | — |

### N.2 Maturity uplift targets

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target | Closing FLOW IDs |
|-------------|----------------------|--------|------------------|
| §5 Policy | L2 partial | **L3** | FLOW-11 |
| §7 Reasoning / planning | L2 | **L3** | FLOW-1, FLOW-12 |
| §8 Execution runtime | L3 | **L3** | FLOW-10 (maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** | FLOW-4–7, FLOW-6, FLOW-13, FLOW-16, FLOW-17 |
| §10 Subagents | L2 | **L3** | FLOW-2, FLOW-3, FLOW-14, FLOW-15 |
| §25 Evaluation | L2 | **L3** | FLOW-9 |

### N.3 Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled; Appendix N (FLOW) created; §6.1aj + §6.2aj active |
| 2026-06-07 | — | FLOW-GAP-12–16 + FLOW-13–17 added; orchestration plan complete vs flow reference |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Full Phase FLOW closeout; ADR-FLOW-001/002/003 accepted; gate green |

---

*Plan synced (2026-06-07). **Harness platform** bands 1–2aj **Done** (FAUDIT-32 **23/23** + Phase FLOW **18/18 harness**). **Default active queue:** [§6.1](#61-harness-implementation-queue--continuous-gate) maintenance. Product: [§6.3](#63-end-of-plan--deferred-product-work-only) incl. **FLOW-8**. **Every PR:** §6.1 gate green.*

---

### 6.1s Harness implementation queue — evaluation closeout (closed)

**Purpose:** Single ordered list for **Phase EVAL** (Band 2x). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **EVAL-DOC.1** | Docs | **Done** | Appendix U + cross-refs | Author map complete |
| 2 | **EVAL-1** | Code | **Done** | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | `test_harness_evaluation_wiring.py` |
| 3 | **EVAL-2** | Code | **Done** | `evaluation_assembly_resolver` | wire-time validation tests |
| 4 | **EVAL-3** | CI | **Done** | `check_harness_evaluation_wiring.py` | CI green |

**Suggested PR order (complete):** EVAL-DOC.1 → EVAL-1 → EVAL-2 → EVAL-3.### 6.1f Harness implementation queue — context engineering closeout (closed)

**Purpose:** Single ordered list for **Phase CTX** (Band 2n). **Closed 2026-06-02**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts | `pytest -m gate` green |
| 1 | **CTX-DOC.1–2** | Docs | **Done** | Appendix L + cross-refs | Author map complete |
| 2 | **CTX-1** | Code | **Done** | `context_runtime_bridge` | `test_context_runtime_bridge.py` |
| 3 | **CTX-2** | Code | **Done** | `context_wiring` + `nexus_factory` | `test_context_wiring.py` |

---

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

**Explicitly excluded:** K.1, K.2, new product eval modes requiring business datasets — [§6.3a](#63a-business-backlog-register-consolidated).### 6.1t Harness implementation queue — Adaptive Harness Intelligence (closed)

**Purpose:** Single ordered list for **Phase W-ADAPT** (Band 2y). **Closed 2026-06-02** — **70/70 Done** (Wave W-ADAPT-0 through Wave W-ADAPT-7 **Done**). Maintenance-only; see [§6.1](#61-harness-platform-maintenance-default--band-1).

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 0 | **§6.1** | Continuous | **Active** | Gate + audit scripts on every harness PR | `pytest -m gate` green |
| 1 | **W-ADAPT-0.2–0.5** | Docs/Code | **Done** | ADR-ADAPT-001 + `intergrax/runtime/adaptive/` scaffold | Import + gate stub |
| 2 | **W-ADAPT-1.1–1.12** | Code | **Done** | Observe (L4-O): signals + utility + report | `phase_w_adapt_report.py` |
| 3 | **W-ADAPT-2.1–2.12** | Code | **Done** | Recommend (L4-R): engines + proposals (no apply) | Proposals in report |
| 4 | **W-ADAPT-3.1–3.7** | Code | **Done** | Shadow (L4-S): ProfileVersionStore + executor.shadow | Integration test green |
| 5 | **W-ADAPT-4.1–4.10** | Code | **Done** | Apply (L4-A): canary, apply, rollback, events | Policy learning HITL enforced |
| 6 | **W-ADAPT-5.1–5.12** | Code/Docs | **Done** | Verify (L4-V): VerificationLoop + runtime L4 closeout | `--enforce-l4-runtime` |
| 7 | **W-ADAPT-6.1–6.5** | Code | **Done** | ProcessPatternMiner + daily scheduler | pattern report |
| 8 | **W-ADAPT-7.1–7.7** | Code/Docs | **Done** | Tier-3 AdaptiveProfile + Appendix V + acceptance | E2E observe→recommend |

**Suggested PR order:** See [Phase W-ADAPT — Suggested PR order](#w-adapt--suggested-pr-order).

**Explicitly excluded:** K.1, K.2, deep RL, foundation model training, autonomous prompt edits — [§6.3a](#63a-business-backlog-register-consolidated).

---

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-planned)**. **Next product work** = [§6.3](#63-end-of-plan--deferred-product-work-only) (product prioritization only).

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/check_harness_no_getattr.py
  python scripts/check_legacy_modules_removed.py
  python scripts/check_agent_skill_resolution.py
  python scripts/check_harness_registry_resolution.py
  python scripts/check_harness_capability_graph_wiring.py
  python scripts/check_legacy_tool_plan_booleans.py
  python scripts/check_trace_bridge_event_catalog.py
  python scripts/check_plugin_catalog.py
  python scripts/check_llm_adapter_typed_returns.py
  python scripts/check_agents_llm_adapter_response.py
  uv run python scripts/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/check_scaffold_harness_alignment.py
  python scripts/check_agents_no_tier3_imports.py
  python scripts/check_intergrax_no_applications_imports.py
  uv run python scripts/check_harness_prompt_golden_catalog.py
  uv run python scripts/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/phase_v_capability_graph_guard.py --enforce
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

**Maintenance depth (2026-06-07):** **OBS-DEPTH.1 Done** — unified run journal. **T10-DEPTH.1 Done** — broker task index + PagerDuty acknowledge adapter. **T-EXPAND T11 Done** — 160 tools. **LEG-DEPTH.1–3 + O.5 depth Done** — planner schema uses `tool_ids`; legacy booleans accepted with deprecation trace; `from_legacy()` gated by `check_legacy_tool_plan_booleans.py`. **OBS-DEPTH.2 Done** — `check_trace_bridge_event_catalog.py` + gate test. **OBS live emit Done** — `RuntimeState.trace_event` → `runtime_event_bus`. **Celery purge_completed Done** — optional KV task index. **notify.dispatch_due Done** — Tier-0 dispatcher tool. **T-EXPAND T12 Done** — 170 tools (health slot probes + notify dispatcher). **T-EXPAND T13 Done** — 172 tools (`eval.judge`, `eval.trajectory` / CRIT-V). **L2→L3 §21 Done** — `test_observability_layer_depth_gate.py` regression gate.### 6.1ah Harness implementation queue — FAUDIT-32 remediation (closed)

**Status:** **Done** (2026-06-06) — **23/23 Done**  
**Source:** [Phase FAUDIT-32](plan/PLATFORM_FOUNDATION.md) · **Appendix M**  
**Priority ladder:** **Band 2ad** (§4.0) — runs **after** FAUDIT-TIER.1 on every harness PR that touches `intergrax/runtime/architecture/`

**Execution order (recommended):**

```text
Wave P0 (architecture integrity):
  FAUDIT-TIER.1 → FAUDIT-TIER.2

Wave P1 (identity + intake + observability):
  FAUDIT-INTAKE.1 → FAUDIT-ID.1 → FAUDIT-OBS.1 → FAUDIT-EVAL.1

Wave P2 (control-plane depth):
  FAUDIT-PE.1 → FAUDIT-REG.1 → FAUDIT-CG.1 → FAUDIT-CG.2
  → FAUDIT-SEC.1 → FAUDIT-REL.1 → FAUDIT-COST.1

Wave P3 (orchestration + cognition + memory):
  FAUDIT-ORCH.1 → FAUDIT-SUB.1 → FAUDIT-COG.1 → FAUDIT-LLM.1
  → FAUDIT-POL.1 → FAUDIT-MEM.1 → FAUDIT-ALG.1 → FAUDIT-OPS.1
  → FAUDIT-INTAKE.2 → FAUDIT-ID.2
```

| ID | Status | Priority | Blocks |
|----|--------|----------|--------|
| FAUDIT-TIER.1 | **Done** | **Critical** | `intergrax/applications/reference/harness_manifest_catalog.py` |
| FAUDIT-TIER.2 | **Done** | High | `scripts/check_intergrax_no_applications_imports.py` |
| FAUDIT-INTAKE.1 | **Done** | High | `intergrax/contracts/task_envelope.py` |
| FAUDIT-INTAKE.2 | **Done** | Medium | `tests/unit/runtime/architecture/test_faudit_remediation.py` |
| FAUDIT-ID.1 | **Done** | High | `intergrax/contracts/actor_identity.py` |
| FAUDIT-ID.2 | **Done** | Medium | `DelegationSpec.permission_scopes` |
| FAUDIT-POL.1 | **Done** | High | `PolicyEngine.evaluate_pre_llm/pre_output` |
| FAUDIT-LLM.1 | **Done** | High | `intergrax/llm_adapters/registry/model_router.py` |
| FAUDIT-COG.1 | **Done** | High | `intergrax/contracts/decision_record.py` + UAEP emit |
| FAUDIT-ORCH.1 | **Done** | Medium | `GraphExecutor` inflight backpressure |
| FAUDIT-SUB.1 | **Done** | High | `SubtaskContract` + safer defaults |
| FAUDIT-MEM.1 | **Done** | High | `retention_enforcement.py` + `PolicyScopedMemoryView` STM purge |
| FAUDIT-PE.1 | **Done** | High | `prompt_golden_catalog.py` + `tests/fixtures/prompt_golden/` + CI script |
| FAUDIT-ALG.1 | **Done** | High | lifecycle states + reference agent `owner_team` adoption + CI script |
| FAUDIT-REG.1 | **Done** | High | `HarnessRegistrySnapshot` agent/eval fields |
| FAUDIT-CG.1 | **Done** | High | prompt seeds in `capability_graph_wiring.py` |
| FAUDIT-CG.2 | **Done** | Medium | `phase_v_capability_graph_guard.py` impact log |
| FAUDIT-OBS.1 | **Done** | High | `RuntimeEventType.LLM_CALL/POLICY_DECISION` |
| FAUDIT-REL.1 | **Done** | High | expanded `RuntimeErrorCode` + classifier |
| FAUDIT-SEC.1 | **Done** | High | `intergrax/contracts/data_classification.py` |
| FAUDIT-COST.1 | **Done** | High | `run_budget` wired in `nexus_factory` |
| FAUDIT-EVAL.1 | **Done** | High | `phase_v_closeout_gate.py` eval baseline |
| FAUDIT-OPS.1 | **Done** | Medium | `build/architecture_hardening/release_cycles.json` |

**DoD (§6.1ah queue closure):** All **Planned** rows **Done**; Appendix M scorecard shows **0 Critical**, **≤5 High** (documented deferrals only); tier gate green.

---

### 6.1ai Harness implementation queue — FAUDIT-32 follow-up (closed)

**Status:** **Done** (2026-06-06) — post-remediation depth for PE/ALG/MEM adoption  
**Priority ladder:** **Band 2ad** (§4.0) — runs after §6.1ah closure

| ID | Status | Deliverable |
|----|--------|-------------|
| FAUDIT-PE.1+ | **Done** | Real `prompts/` golden hashes in `tests/fixtures/prompt_golden/expectations.json`; `scripts/check_harness_prompt_golden_catalog.py`; gate test |
| FAUDIT-ALG.1+ | **Done** | `lifecycle_state` + `owner_team` on reference Tier-2 agents; `scripts/check_agents_lifecycle_metadata.py` |
| FAUDIT-MEM.1+ | **Done** | `should_forget_stm_record` wired in `PolicyScopedMemoryView.read` |

**Explicitly deferred (Band 3 / product):** MEM-9 entity graph memory implementation (RFC only); K.1/K.2 business agents.### 6.2bo Phase EVAL execution order (Band 2x — closed 2026-06-02)

**Status:** **Done** · register: [Phase EVAL](plan/CRITIC_VERIFICATION.md) · queue: [§6.1s](#61s-harness-implementation-queue--evaluation-closeout-closed)

| Step | ID | Deliverable | Priority |
|------|-----|-------------|----------|
| 1 | EVAL-DOC.1 | Appendix U + plan sync | High |
| 2 | EVAL-1 | `EvaluationProfile` + `evaluation_runtime_bridge` + `evaluation_wiring` | Critical |
| 3 | EVAL-2 | `evaluation_assembly_resolver` | High |
| 4 | EVAL-3 | `check_harness_evaluation_wiring.py` | Medium |

---

### 6.2 Harness architecture hardening (Band 2 — Phase V) — Done

**Status:** **Done** (2026-06-05) — Phase V contracts + V-REM runtime enforcement complete. Closeout: `phase_v_closeout_gate.py --enforce --enforce-l4`.

| Item | Status | Notes |
|------|--------|-------|
| V-CG … V-V6 | **Done** | Governance + CI + runtime enforcement |
| V-REM | **Done** | 10/10 closed — §6.1z queue closed |
| W-ML | **Done** | [architecture/MODALITY.md](architecture/MODALITY.md) |
| P-Ext | **Done** | Appendix I |
| M.6 P5 / M.6 P6 / R-Skill expansion | **On demand** | W-OPS.10, W-OPS.8, §6.1x, §6.1y |

**Forbidden in Band 2/2d:** K.1, K.2, product-specific skills, new product application hosts.### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](plan/PLATFORM_FOUNDATION.md) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>/` beyond lab + reference hosts | **Deferred** | Product decision only — confirmed 2026-06-09; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar/` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work. **Product:** [`local_workspace_application`](../applications/local_workspace_application/) — Local Knowledge Workspace (LKW) — first business environment after harness GA; see [ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md).

---

### 6.1s Archived — Phase S cadence (complete 2026-06-01)

See Phase S definition of done and Appendix F. Do not reopen S.* unless regression (fix under T.* or U.*).### 6.1a Archived — Phase Q cadence (complete 2026-06-01)

Phase Q used **one Q.* deliverable per PR** → update Appendix C + paydown log. See Appendix C for Waves 1–9 and gate **417** at close. Do not reopen Q.* unless regression found (residual hardening → Appendix D).

---

(Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---



**Status:** **Done** (2026-06-05) — runtime governance via V-REM, H-APP, DX-5.8; documentation via GOV-DOC.*  
**Prerequisites:** Phase V-REM **Done**, H-APP.2.4–2.8 **Done**, DX-5.8 **Done**  
**Goal:** Close governance/policy/observability audit (AUDIT_MAP §5, §21) with a single authoring map and traceability — **no** new OS features.  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix H](guides/AGENT_CREATION_GUIDE.md#appendix-h--governance-policy--observability-control-plane)

**Delivery rule:** GOV-DOC.* = docs-only PRs; no code unless regression found → route to **REG-*** under §6.1.

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| GOV-DOC.1 | **Appendix H** — control plane map (profiles, bundles, hooks, EP groups, mandatory vs optional observability) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + §H.1–H.8 present |
| GOV-DOC.2 | **Cross-ref sync** — plan Documentation model, README, `guides/HARNESS_ENVIRONMENT.md`, [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.5, AUDIT_MAP §5/§21, audit prompt ref #5 | **Done** | Medium | `docs/*` | Links resolve; no orphan audit layer |
| GOV-DOC.3 | **`guides/EXTENSION_AUTHOR_GUIDE.md` §10** — `intergrax.policy_rules` author surface | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | DX-5.8 traceability |
| GOV-PROD.1 | Unified product observability dashboard (beyond lab debug APIs) | **Deferred** | — | — | **§6.3** product decision only — confirmed 2026-06-09; harness path = `observability_backend` + OBS-BUS |

**Explicitly out of scope:** K.1/K.2 policy; product-specific legal/org policy fragments beyond lab reference YAML.

---



**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done.

**Post depth bands (2026-06-09):** MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG closeout complete — Appendix M scorecard refreshed; remaining L2 rows are incremental depth only.

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L3 | 0 | 0 | Yes |
| 3 | Interface and Task Intake | L3 | 0 | 1 | Partial |
| 4 | Identity, Trust and Tenancy | L2 | 0 | 2 | Partial |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L3 | 0 | 0 | Yes |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 0 | Yes |
| 10 | Subagents and Multi-Agent Coordination | L2 | 0 | 2 | Partial |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L3 | 0 | 0 | Yes |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L2 | 0 | 1 | **No** |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L2 | 0 | 2 | **No** |
| 21 | Observability and Telemetry | L2 | 0 | 2 | **No** |
| 22 | Error Handling and Reliability | L2 | 0 | 1 | **No** |
| 23 | Security and Data Governance | L2 | 0 | 2 | **No** |
| 24 | Cost and Resource Governance | L2 | 0 | 1 | **No** |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 1 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 1 | Partial |
| 31 | Agent Lifecycle Governance | L2 | 0 | 2 | Partial |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax/` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax/` → `applications/` imports | High | `scripts/check_intergrax_no_applications_imports.py` in §6.1 |
| FAUDIT-INTAKE.1 | §3 | No canonical `TaskEnvelope`; `Task` + `RuntimeRequest` split | High | Typed envelope alias or consolidation; plan W-OPS.6 naming sync |
| FAUDIT-INTAKE.2 | §3 | Worker≡HTTP intake parity test matrix incomplete | High | Acceptance test: CLI/worker/HTTP same `Task` shape |
| FAUDIT-ID.1 | §4 | No user/service/agent identity distinction | High | Identity contracts + propagation to delegation |
| FAUDIT-ID.2 | §4 | `DelegationSpec` lacks permission scope audit | High | Scope field + trace on child runs |
| FAUDIT-POL.1 | §5 | No pre-LLM / pre-output policy hooks in runtime | High | PolicyEngine extension points documented + wired |
| FAUDIT-LLM.1 | §6 | No policy-driven model routing / fallback chain | High | Router module or AdaptiveProfile integration |
| FAUDIT-COG.1 | §7 | No universal `DecisionRecord` per UAEP step | High | Typed decision artifact + trace event |
| FAUDIT-ORCH.1 | §9 | No graph backpressure beyond parallel cap | High | Queue depth / shed policy in `GraphExecutor` |
| FAUDIT-SUB.1 | §10 | No formal `SubtaskContract`; `inherit_tool_policy=True` default | High | Contract type + safer delegation defaults |
| FAUDIT-MEM.1 | §15 | Entity graph memory absent; STM retention partial | High | Route to MEM-9.* / new MEM row if scoped |
| FAUDIT-PE.1 | §17 | No golden prompt content regression in CI | High | Golden YAML fixtures + gate test |
| FAUDIT-REG.1 | §19 | `HarnessRegistrySnapshot` omits agents + eval registry | High | Extend snapshot + assembly resolver |
| FAUDIT-CG.1 | §20 | Capability graph seed skips `prompt:*` nodes | High | `_seed_node_ids()` parity |
| FAUDIT-CG.2 | §20 | Blast-radius not enforced at release | High | `phase_v_capability_graph_guard.py` impact check |
| FAUDIT-OBS.1 | §21 | `RuntimeEventType` missing `LLM_CALL` / `POLICY_DECISION` | High | Canon event catalog + bridge from trace |
| FAUDIT-REL.1 | §22 | Shallow error taxonomy (`VALIDATION_ERROR` only + 2) | High | Expand classifier per AUDIT_MAP §22 |
| FAUDIT-SEC.1 | §23 | No `DataClassification` model | High | Security profile + enforcement hooks |
| FAUDIT-COST.1 | §24 | Per-tenant cost attribution not mandatory in NexusLoop | High | Budget gate on main path |
| FAUDIT-EVAL.1 | §25 | `require_baseline_for_release` not CI-enforced | High | `phase_v_closeout_gate.py` eval baseline check |
| FAUDIT-ALG.1 | §31 | Lifecycle states ≠ AUDIT_MAP catalog; weak agent adoption | High | Align or document ADR; scaffold defaults |
| FAUDIT-OPS.1 | §30 | `release_cycles.json` not in repo; W-OPS.5 artifact policy unclear | High | Document committed vs generated artifact |

**Delivery rule:** One **FAUDIT.\*** ID per PR → update §6.1ah + Appendix M paydown log → gate green.

**Explicitly out of scope (audit-and-fix):** source code or test changes during this audit pass; K.1/K.2; new product Tier-3 apps.

---



**Status:** **Done** (2026-06-05) — **6/6** deliverables Done (ORCH-DOC.* + ORCH-1–4); gate **581 passed**  
**Prerequisites:** R-Delegate **Done**, Q+-N.* runners **Done**, H-APP.3.1–3.2 **Done**, V-MA.* **Done**  
**Goal:** Close orchestration audit residuals (AUDIT_MAP §7–§10) — wire declared Tier-3 profile fields to runtime; bridge declarative graph spec to execution plan; cap graph batch concurrency.  
**Priority ladder:** **Band 2j** (§4.0) — **default implementation queue** after §6.1 gate on each PR.  
**Execution order:** [§6.2bb](#62bb-phase-orch-execution-order-band-2j--active) · queue: [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-active)  
**Author map:** [`guides/AGENT_CREATION_GUIDE.md` Appendix I](guides/AGENT_CREATION_GUIDE.md#appendix-i--orchestration-control-plane)

**Delivery rule:** One **ORCH-*** ID per PR → update master table + §6.1b + paydown log below → `pytest -m gate` + §6.1 scripts green.

**Audit verdict (baseline — preserve as acceptance context):**

| Area | Maturity (L0–L4) | Residual before ORCH | Close via |
|------|------------------|----------------------|-----------|
| Nexus stack (§8) | **L3–L4** | — | ORCH-DOC.* (documented) |
| Planning strategies (§7) | **L3–L4** | — | ORCH-1 **Done** |
| Declarative graph (§9) | **L3–L4** | — | ORCH-2 **Done** |
| Graph concurrency (§9) | **L3** | — | ORCH-3 **Done** |
| Subagent delegation (§10) | **L3–L4** | — | R-Delegate (Done) |

### ORCH — Master register

| ID | Wave | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-------------|--------|----------|---------------|------------|
| ORCH-DOC.1 | ORCH0 | **Appendix I** — orchestration control plane map (§I.1–I.10) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| ORCH-DOC.2 | ORCH0 | **Cross-ref sync** — plan, README, strategy, AUDIT_MAP §7–§10, audit prompt ref #6, canon §42.43 | **Done** | Medium | `docs/*` | Links resolve |
| ORCH-1 | ORCH1 | **Wire `planner_kind` / `classifier_kind`** — registry maps kinds → `TaskPlanner` / `ClassifyingTaskClassifier`; `build_nexus_loop_from_environment` passes resolved instances to `NexusLoop` | **Done** | **Critical** | `orchestration_wiring.py`, `nexus_factory.py` | `test_orchestration_wiring.py` |
| ORCH-2 | ORCH2 | **`ApplicationGraphSpec` → `NexusPlan` seed** — `graph_spec_to_plan.py` + `GraphSpecSeedingPlanner` when task has no plan id | **Done** | **High** | `graph_spec_to_plan.py`, `PlanStep.delegation` | `test_graph_spec_to_plan.py`, `test_lab_graph_spec.py` |
| ORCH-3 | ORCH3 | **`max_parallel_nodes` on `OrchestrationProfile`** — cap concurrent nodes per graph batch in `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `graph_executor.py` | `test_graph_executor_parallel_cap.py` |
| ORCH-4 | ORCH4 | **Docs closeout** — Appendix I + plan sync | **Done** | Low | `docs/*` | No “planned wiring” residuals |

**Supported `planner_kind` values (ORCH-1 contract):**

| Kind | Implementation | Notes |
|------|----------------|-------|
| `null` / `default` | `TaskPlanner()` | Current harness default |
| `engine` | `EnginePlanner` adapter implementing plan contract | Requires `RuntimeConfig` on build context — lab/legal hosts only in v1 |
| Unknown kind | — | **Fail fast** at Nexus bootstrap with typed error (no silent fallback) |

**Supported `classifier_kind` values (ORCH-1 contract):**

| Kind | Implementation |
|------|----------------|
| `null` / `default` | `ClassifyingTaskClassifier(registry)` |

**Explicitly out of scope:** Nested full harness per child (use R-Delegate); new graph node types (Tier-1 canon change); product-specific orchestration in `agents/`.

### ORCH — Paydown log

| Date | ORCH ID | Summary |
|------|---------|---------|
| 2026-06-05 | ORCH-DOC.1, ORCH-DOC.2 | Governance + orchestration audit docs; Appendix H/I; AUDIT_MAP cross-refs |
| 2026-06-05 | ORCH-1, ORCH-2, ORCH-3 | Orchestration wiring, graph spec plan seed, parallel cap; gate **581** |
| 2026-06-05 | ORCH-4 | Plan + author guide closeout |

**Phase ORCH complete when:** ORCH-1–4 **Done**; §6.1b queue closed; Appendix I has no “planned wiring” gaps; gate **581** green. **Status: complete (2026-06-05).**

---



**Status:** **Done** (2026-06-07) — **18/18 harness** deliverables Done (FLOW-8 harness **Done**; product host **Deferred** §6.3 §6.3) · source: [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23–§25  
**Prerequisites:** Phase ORCH **Done**; [ADR-FLOW-001](adr/ADR-FLOW-001.md) **Accepted** (delegation target semantics)  
**Goal:** Close **all** orchestration depth gaps (`FLOW-GAP-01`…`16`) from flow reference — uplift AUDIT_MAP §5, §7, §8, §9, §10, §25 from L2/L3-partial to **L3+** operational maturity  
**Priority ladder:** **Band 2aj** (§4.0) — **maintenance only** — §6.1 gate (Band 3 §6.3 frozen)  
**Execution order:** [§6.2aj](#62aj-phase-flow-execution-order-band-2aj--active) · queue: [§6.1aj](#61aj-harness-implementation-queue--nexus-execution-depth-closed)  
**Traceability:** **Appendix N (FLOW)** — [`§Appendix N`](#appendix-n--nexus-execution-flow-traceability-phase-flow)

**Delivery rule:** One **FLOW-*** ID per PR → update master table + §6.1aj + Appendix N paydown → `pytest -m gate` + §6.1 scripts green.

**Maturity target (phase complete):**

| AUDIT_MAP § | Baseline (FAUDIT-32) | Target after FLOW |
|-------------|----------------------|-------------------|
| §5 Policy (pre-plan hooks) | L2 partial | **L3** (FLOW-11) |
| §7 Reasoning / planning | L2 | **L3+** (FLOW-1, FLOW-12, COG-DEPTH) |
| §8 Execution runtime | L3 | **L3** (FLOW-10, maintain) |
| §9 Orchestration / graph | L3 partial | **L3+** (FLOW-4–7, FLOW-6, FLOW-13, FLOW-16) |
| §10 Subagents | L2 | **L3** (FLOW-2, FLOW-3, FLOW-14, FLOW-15) |
| §25 Evaluation | L2 | **L3** (FLOW-9) |

**Explicitly out of scope:** Nested full harness per child; new graph node **types** (Tier-1 canon change); K.1/K.2 business agents (FLOW-8 → §6.3 unless reprioritized).

### FLOW — Master register

| ID | Wave | Gap | Deliverable | Status | Priority | Module / test | Acceptance |
|----|------|-----|-------------|--------|----------|---------------|------------|
| FLOW-DOC.1 | FLOW0 | — | **Flow reference sync** — paydown §23 gaps in `architecture/NEXUS_EXECUTION_FLOW.md` after each FLOW PR | **Done** | Low | `docs/architecture/NEXUS_EXECUTION_FLOW.md` | No stale `FLOW-GAP` rows for Done IDs |
| FLOW-2 | FLOW1 | FLOW-GAP-02 | **ADR-FLOW-001 implementation** — expand `DELEGATES_TO` to child `PlanStep` + `ExecutionNode`; `DelegationSpec` on **child**; `GraphExecutor` routes `child_agent_id` | **Done** | **Critical** | `graph_spec_to_plan.py`, `graph_builder.py`, `graph_executor.py` | `test_graph_spec_to_plan.py` + integration delegation path; canon §42.14.3 note updated |
| FLOW-3 | FLOW1 | FLOW-GAP-03 | **`max_delegation_depth` enforcement** — count expanded delegation chain in `GraphExecutor`; fail with trace | **Done** | High | `graph_executor.py`, `environment_profile.py` | Unit test depth exceeded |
| FLOW-1 | FLOW2 | FLOW-GAP-01 | **Real `EngineBackedNexusPlanner`** — bridge `engine_planner_orchestrator` → `NexusTaskPlannerProtocol`; typed `NexusPlan` from LLM parse | **Done** | High | `orchestration_wiring.py`, `planning/engine_planner_*.py` | `test_orchestration_wiring.py` + planner integration tests |
| FLOW-6 | FLOW2 | FLOW-GAP-06 | **Strict cycle detection** — `ExecutionGraph.batches()` raises on cycle; no unsafe fallback | **Done** | High | `execution_graph.py` | Unit test cyclic graph → error |
| FLOW-4 | FLOW3 | FLOW-GAP-04 | **Opt-in run-level retry** — `OrchestrationProfile.max_run_retries`; wire `RetryCoordinator` in `NexusGraphRunner` | **Done** | Medium | `environment_profile.py`, `graph_runner.py`, `nexus_factory.py` | Integration test graph retry once |
| FLOW-7 | FLOW3 | FLOW-GAP-07 | **`MergePolicy` / `FinalResponseComposerProfile`** — deterministic + structured merge; optional LLM merge hook (policy-gated) | **Done** | Medium | `final_response_composer.py`, `environment_profile.py` | Multi-agent merge unit tests |
| FLOW-9 | FLOW3 | FLOW-GAP-11 | **Evaluation hooks on multi-agent fan-in** — post-graph eval observation; evaluator-node cookbook; registry write on multi-node runs | **Done** | Medium | `nexus_loop.py`, `evaluation_wiring.py`, docs §18 | `EvaluationProfile` observation recorded; guide §18 |
| FLOW-11 | FLOW3 | FLOW-GAP-09 | **Pre-plan / pre-LLM policy extension points** — document + wire hooks at planning boundary | **Done** | Medium | `planning_runner.py`, `policy_engine.py` | Hook tests + Appendix H cross-ref |
| FLOW-5 | FLOW4 | FLOW-GAP-05 | **`AgentGraph.on_error(retry)`** — wire to `RetryPolicy` / graph executor | **Done** | Low | `graph_builder.py`, `orchestration_wiring.py` | Integration test declared retry |
| FLOW-10 | FLOW4 | FLOW-GAP-08 | **Reserved lifecycle states** — ADR: implement `WAITING_FOR_RESOURCES`/`EXPIRED` **or** trim enum + canon sync | **Done** | Low | `task_lifecycle.py`, `adr/ADR-FLOW-002.md` | [ADR-FLOW-002](adr/ADR-FLOW-002.md) accepted; reserved v1 semantics |
| FLOW-12 | FLOW4 | §24 / FAUDIT-COG | **`DecisionRecord` regression gate** — verify FAUDIT-COG.1 emit on every UAEP decision path; gate test; sync flow §24 | **Done** | Medium | `uaep.py`, `tests/integration/agents/` | `DECISION_EMITTED` + `decision_record` on each step decision |
| FLOW-13 | FLOW4 | FLOW-GAP-12 | **`max_inflight_nodes` profile + wire** — field on `OrchestrationProfile`; `resolve_max_inflight_nodes()`; `nexus_factory` → `GraphExecutor` | **Done** | Medium | `environment_profile.py`, `orchestration_wiring.py`, `nexus_factory.py` | `GRAPH_BACKPRESSURE` event when cap hit; profile round-trip test |
| FLOW-14 | FLOW4 | FLOW-GAP-13 | **`SubtaskContract` in delegation expansion** — `graph_spec_to_plan` / ADR-FLOW-001 child node uses `SubtaskContract.to_delegation_spec()` (`objective`, `permission_scopes`, `inherit_tool_policy=False`) | **Done** | Medium | `graph_spec_to_plan.py`, `subtask_contract.py` | Unit test scopes + objective on child `DelegationSpec` |
| FLOW-15 | FLOW4 | FLOW-GAP-14 | **Subagent budget envelope** — optional `budget_envelope` on `SubtaskContract` / `DelegationSpec`; enforce in child `GraphExecutor` run via existing budget bridge | **Done** | Medium | `subtask_contract.py`, `delegation.py`, `graph_executor.py` | Child run exceeds envelope → fail with trace |
| FLOW-16 | FLOW4 | FLOW-GAP-15 | **`MODIFY_PLAN` ADR** — [ADR-FLOW-003](adr/ADR-FLOW-003.md): document reserved semantics (policy-gated replan hook) **or** trim `AgentDecision` enum | **Done** | Low | `adr/ADR-FLOW-003.md`, `interrupts/handler.py` | ADR accepted; `MODIFY_PLAN_NOT_SUPPORTED` when no handoff |
| FLOW-17 | FLOW4 | FLOW-GAP-16 | **`MULTI_AGENT` ordering policy** — `OrchestrationProfile.multi_agent_order` (`registry` \| `priority` \| `stable_alpha`); deterministic step order in `TaskPlanner` | **Done** | Low | `environment_profile.py`, `task_planner.py` | Gate test: two agents same capability → stable declared order |
| FLOW-8 | FLOW5 | FLOW-GAP-10 | **Harness CFG simulation** (ORCH-CONFIG.5) + optional Tier-3 §42.43 product host | **Partial** | Harness + Product | `tests/integration/runtime/test_orchestration_cfg_simulation.py` · product §6.3 gate |
| FLOW-DOC.2 | FLOW5 | — | **Phase closeout** — Appendix N (FLOW), flow reference §23 paydown (all gaps), maturity dashboard §0.5 | **Done** | Low | `docs/*` | All non-deferred FLOW rows **Done**; zero open `FLOW-GAP` in §23 |

### FLOW — Suggested PR order

```text
FLOW-2 → FLOW-14 → FLOW-3 → FLOW-15 → FLOW-6 → FLOW-1 → FLOW-4 → FLOW-13 → FLOW-7 → FLOW-9 → FLOW-11 → FLOW-5 → FLOW-10 → FLOW-12 → FLOW-16 → FLOW-17 → FLOW-DOC.*
```

**Parallel OK after FLOW-2:** FLOW-1, FLOW-6, FLOW-13 (disjoint modules). **FLOW-14** same PR as FLOW-2 or immediately after.

**FLOW-8:** Schedule only after explicit product decision ([§6.3](#63-end-of-plan--deferred-product-work-only)).

### FLOW — Paydown log

| Date | FLOW ID | Summary |
|------|---------|---------|
| 2026-06-07 | — | Phase FLOW scheduled from `architecture/NEXUS_EXECUTION_FLOW.md` §25; queue §6.1aj; Appendix N (FLOW) |
| 2026-06-07 | — | Audit gap closeout: FLOW-13–17 + FLOW-GAP-12–16 added; FLOW-12 narrowed to regression gate; **0/18** |
| 2026-06-07 | FLOW-1–17, FLOW-DOC.* | Phase FLOW implementation complete: delegation expansion, graph hardening, profile wiring, ADR-FLOW-002/003; gate **906 passed**; **18/18 harness** (FLOW-8 harness **Done**; product host **Deferred** §6.3) |

**Phase FLOW complete when:** FLOW-1–7, FLOW-9, FLOW-11–17, FLOW-DOC.* **Done**; FLOW-8 **Deferred** or Done per product; §6.1aj closed; **zero open `FLOW-GAP-*`** in flow reference §23; AUDIT_MAP §5/§7/§9/§10/§25 at target maturity; gate green.

---



**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (INT-DOC.* + INT-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §13; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix K**.

**Priority ladder:** **Band 2l** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bd](#62bd-phase-int-execution-order-band-2l--closed) · queue: [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed)

### INT — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| INT-DOC.1 | INT0 | **Appendix K** — integration control plane (§K.1–K.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| INT-DOC.2 | INT0 | **Cross-ref sync** — plan, README, AUDIT_MAP §13, audit prompt ref #8 | **Done** | Medium | `docs/*` | Links resolve |
| INT-1 | INT1 | **`integration_runtime_bridge.py`** — explicit `integration_profile` on `RuntimeConfig` | **Done** | **Critical** | `integration_runtime_bridge.py`, `runtime_config_bridge.py` | `test_integration_runtime_bridge.py` |
| INT-2 | INT2 | **`integration_health_wiring.py`** — bootstrap health probes on `wire_application_environment` | **Done** | High | `integration_health_wiring.py`, `environment_wiring.py` | `test_integration_health_wiring.py` |

### INT — Paydown log

| Date | INT ID | Summary |
|------|--------|---------|
| 2026-06-02 | INT-DOC.1, INT-DOC.2 | Appendix K + cross-refs; AUDIT_MAP §13 |
| 2026-06-02 | INT-1, INT-2 | Integration runtime bridge + health wiring |

**Phase INT complete when:** INT-1–2 + INT-DOC.* **Done**; §6.1d queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (RAG-DOC.* + RAG-1); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §14; author map: **Appendix K** §K.5.

**Priority ladder:** **Band 2m** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2be](#62be-phase-rag-execution-order-band-2m--closed) · queue: [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed)

### RAG — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| RAG-DOC.1 | RAG0 | **Appendix K** §K.5 + AUDIT_MAP §14 cross-ref | **Done** | High | `docs/*` | RAG bridge documented |
| RAG-1 | RAG1 | **`rag_runtime_bridge.py`** + RAG stack on `wire_application_environment` | **Done** | **Critical** | `rag_runtime_bridge.py`, `environment_wiring.py`, `runtime_config_bridge.py` | `test_rag_runtime_bridge.py` |

### RAG — Paydown log

| Date | RAG ID | Summary |
|------|--------|---------|
| 2026-06-02 | RAG-DOC.1 | Appendix K §K.5 + plan sync |
| 2026-06-02 | RAG-1 | RAG runtime bridge + environment wire; gate **600** |

**Phase RAG complete when:** RAG-1 + RAG-DOC.* **Done**; §6.1e queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CTX-DOC.* + CTX-1–2); gate **612 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §16; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix L**.

**Priority ladder:** **Band 2n** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bf](#62bf-phase-ctx-execution-order-band-2n--closed) · queue: [§6.1f](#61f-harness-implementation-queue--context-engineering-closeout-closed)

### CTX — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| CTX-DOC.1 | CTX0 | **Appendix L** — context engineering control plane (§L.1–L.6) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CTX-DOC.2 | CTX0 | **Cross-ref sync** — plan, README, AUDIT_MAP §16, audit prompt ref #9 | **Done** | Medium | `docs/*` | Links resolve |
| CTX-1 | CTX1 | **`context_runtime_bridge.py`** — dedicated context profile → `RuntimeConfig` | **Done** | **Critical** | `context_runtime_bridge.py`, `runtime_config_bridge.py` | `test_context_runtime_bridge.py` |
| CTX-2 | CTX2 | **`context_wiring.py`** — `ContextManager` + task options from environment; `nexus_factory` wire | **Done** | High | `context_wiring.py`, `nexus_factory.py`, `harness_host_runtime.py` | `test_context_wiring.py` |

### CTX — Paydown log

| Date | CTX ID | Summary |
|------|--------|---------|
| 2026-06-02 | CTX-DOC.1, CTX-DOC.2 | Appendix L + cross-refs; AUDIT_MAP §16 |
| 2026-06-02 | CTX-1, CTX-2 | Context runtime bridge + Nexus ContextManager wiring; gate **608** |

**Phase CTX complete when:** CTX-1–2 + CTX-DOC.* **Done**; §6.1f queue closed. **Status: complete (2026-06-02).**

---



**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `engine_planner_messages.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (PE-DOC.* + PE-1–3); gate **623 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §17; V-REM-PE.1/PE.2 governance schema (**Done**); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix M**.

**Priority ladder:** **Band 2p** (§4.0) — closed; default queue = **§6.1** maintenance.

### PE — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| PE-1 | PE1 | **`PromptProfile`** + `prompt_runtime_bridge` — `catalog_path` → `RuntimeConfig.prompt_catalog_path` | **Done** | `environment_profile.py`, `prompt_runtime_bridge.py`, `config.py` | `test_prompt_runtime_bridge.py` |
| PE-2 | PE2 | **`prompt_wiring`** — `resolve_prompt_registry()`, `PromptRegistryProtocol` | **Done** | `prompt_wiring.py`, `prompt_registry_protocol.py` | `test_prompt_wiring.py` |
| PE-3 | PE3 | **Environment wire** — `materialize_runtime_config`, `build_runtime_context_from_environment`, `ApplicationBuildContext.prompt_registry` | **Done** | `runtime_config_bridge.py`, `environment_wiring.py`, `runtime_context.py` | wiring tests + gate |
| PE-4 | PE4 | **Nexus injection** — `prompt_registry_resolver`; `tools_step`, `tool_planning_prompts`, `engine_plan_models`, `engine_planner_messages` use `RuntimeContext.prompt_registry` | **Done** | `prompt_registry_resolver.py`, Nexus steps/planner | `test_tools_step_prompt_registry.py` |
| PE-DOC.1 | PE0 | **Appendix M** — prompt registry control plane (§M.1–M.6) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |

**Residual:** none on Tier-3 host build path. Legacy YAML prompt assets (`chat_router*`, `tools_agent_*`) remain as catalog files only.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CLEAN-1–4)

**Audit basis:** Phase U-Leg residual; `scripts/check_legacy_modules_removed.py`; prior `check_tools_agent_*` audits merged.

**Priority ladder:** closeout between Band 2p and 2q; default queue = **Band 2q** [Phase AS](plan/ORCHESTRATION.md).

### CLEAN — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CLEAN-1 | CLEAN1 | **Remove `legacy/chat_router.py`** — YAML assets tested without runtime module | **Done** | `tests/unit/chat_agent/` | prompt YAML tests green |
| CLEAN-2 | CLEAN2 | **Remove `tools/tools_agent.py`** — `CatalogToolPlanner` + `ToolPlanningService` canonical | **Done** | `catalog_tool_planner.py`, `tool_planning_service.py` | `test_catalog_tool_planner.py` |
| CLEAN-3 | CLEAN3 | **Unified CI audit** — `check_legacy_modules_removed.py` replaces `check_tools_agent_*` | **Done** | `scripts/`, `.github/workflows/unit-tests.yml` | audit script green in CI |
| CLEAN-4 | CLEAN4 | **Docs sync** — plan, HARNESS_ENVIRONMENT, AGENT_CREATION_GUIDE, README, TOOLS | **Done** | `docs/*` | no stale `ToolsAgent` production paths |

**Retained (not CLEAN scope):** `ToolInvocationPlan.from_legacy()` + deprecation tests; `EnginePlan.use_rag`/`use_websearch` on LLM schema; `intergrax/legacy/rag_answers/` archive with import guard; diagnostic type names (`CoreLLMUsedToolsAgentAnswerDiagV1`).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (AS-DOC.1 + AS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §18; ideal model §17 in [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md); author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix N**.

**Priority ladder:** **Band 2q** (§4.0) — closed; default queue = **§6.1** maintenance.

### AS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| AS-DOC.1 | AS0 | **Appendix N** — agent assembly control plane (contract, capabilities, skills, lifecycle) | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| AS-1 | AS1 | **`agent_assembly_resolver`** — contract metadata validation at register time | **Done** | `runtime/registry/agent_assembly_resolver.py`, `agent_registry.py` | `test_agent_assembly_resolver.py` |
| AS-2 | AS2 | **Lifecycle metadata enforcement** — `production_eligible` owner/runbook requirements | **Done** | `agent_assembly_resolver.py`, `agent_routing_policy.py` | resolver + routing tests |
| AS-3 | AS3 | **`skill_ids` → `allowed_tools` resolution audit** — CI script + docs cross-ref | **Done** | `scripts/check_agent_skill_resolution.py`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), Legal domain steps, product-only contract variants — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REG-DOC.1 + REG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §19; capability graph V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix O**.

**Priority ladder:** **Band 2r** (§4.0) — closed; default queue = **§6.1** maintenance.

### REG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REG-DOC.1 | REG0 | **Appendix O** — registry architecture control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REG-1 | REG1 | **`HarnessRegistrySnapshot`** + `registry_wiring` + `RegistrySnapshotProtocol` | **Done** | `registry_snapshot.py`, `registry_wiring.py` | `test_registry_wiring.py` |
| REG-2 | REG2 | **`registry_assembly_resolver`** — profile ↔ registry conformance at wire time | **Done** | `registry_assembly_resolver.py`, `environment_wiring.py` | `test_registry_wiring.py` |
| REG-3 | REG3 | **Host registry resolution CI** — `check_harness_registry_resolution.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), marketplace UI, Band 3 product hosts — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (CG-DOC.1 + CG-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §20; Phase V-CG **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix P**.

**Priority ladder:** **Band 2s** (§4.0) — closed; default queue = **§6.1** maintenance.

### CG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| CG-DOC.1 | CG0 | **Appendix P** — capability graph control plane | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CG-1 | CG1 | **`capability_graph_wiring`** — environment subgraph from catalog + registry snapshot | **Done** | `capability_graph_wiring.py`, `capability_graph_protocol.py` | `test_capability_graph_wiring.py` |
| CG-2 | CG2 | **`capability_graph_assembly_resolver`** — wire-time catalog node validation | **Done** | `capability_graph_assembly_resolver.py`, `environment_wiring.py` | `test_capability_graph_wiring.py` |
| CG-3 | CG3 | **Host capability graph CI** — `check_harness_capability_graph_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only graph nodes — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (OBS-DOC.1 + OBS-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21; complements GOV-AUDIT Appendix H; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix Q**.

**Priority ladder:** **Band 2t** (§4.0) — closed; default queue = **§6.1** maintenance.

### OBS — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| OBS-DOC.1 | OBS0 | **Appendix Q** — observability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| OBS-1 | OBS1 | **`observability_runtime_bridge`** + **`observability_wiring`** | **Done** | `observability_runtime_bridge.py`, `observability_wiring.py`, `runtime_config_bridge.py` | `test_harness_observability_wiring.py` |
| OBS-2 | OBS2 | **`observability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `observability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| OBS-3 | OBS3 | **Host observability CI** — `check_harness_observability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only observability dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **8/8** deliverables · OBS-BUS-0–7 **Done**

**Purpose:** Implement the full **Harness Observability Spine (HOS)** — one bus for Harness, applications, and agents; typed extension; causal trees; complete catalog emission; L4 audit §21.

**Architecture:** [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **ADR:** [ADR-OBS-001](adr/ADR-OBS-001.md)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §21 · complements Phase OBS (wiring closeout) · supersedes residual “live bus emit for all LLM paths” row when OBS-BUS-2 ships.

**Priority ladder:** **Band 2al** (§4.0) — runs **after** Phase CRIT-V (Band 2ak) or in parallel §6.1 maintenance slices; **one OBS-BUS ID per PR**.

**Depends on:** Phase OBS (wiring) **Done** · OBS-DEPTH.1/2 **Done** · FAUDIT-OBS.1 **Done**

### OBS-BUS — Master register

| ID | Area | Deliverable | Status | Modules / artifacts | Acceptance |
|----|------|-------------|--------|---------------------|------------|
| OBS-BUS-0 | OBS0 | **Architecture canon** — `architecture/OBSERVABILITY.md` + ADR-OBS-001 + canon/README links | **Done** | `docs/architecture/OBSERVABILITY.md`, `docs/adr/ADR-OBS-001.md` | Doc review; links from §33 |
| OBS-BUS-1 | OBS1 | **`RuntimeEventPayload` registry** — typed canonical payloads per `RuntimeEventType` (§42.23.1 families) | **Done** | `intergrax/runtime/events/payload_registry.py`, `payloads/`, `schema_guard.py`, `trace_bridge.py`, `context_skill_recording.py` | Gate: `test_runtime_event_payload_registry.py` |
| OBS-BUS-2 | OBS2 | **`ObservabilityEmitter` + `TraceScope`** — single emit API; `parent_event_id` causal tree | **Done** | `intergrax/runtime/observability/emitter.py`, `trace_scope.py`, `runtime_state.py` | `RuntimeState.trace_event` delegates; `test_observability_emitter.py` |
| OBS-BUS-3 | OBS3 | **Emission coverage** — `AGENT_SELECTED`, `STEP_FAILED`, graph typed payloads, critic `evaluator_loop` bridge | **Done** | `agent_router.py`, `graph_trace_callbacks.py`, `task_trace.py`, `trace_bridge.py`, `graph_node_diag.py` | `check_observability_emission_coverage.py` |
| OBS-BUS-4 | OBS4 | **Extension SDK** — agent/app `DiagnosticPayload` scaffold, namespace rules, `PayloadSchemaRegistry` | **Done** | `extension_sdk.py`, `tracing_templates.py`, `new_agent.py`, `new_application.py` | `check_payload_schema_registry.py` |
| OBS-BUS-5 | OBS5 | **Persistence conformance** — Cassandra/ES adapters implement same protocols; profile docs | **Done** | `document_backed_runtime_event_store.py`, `persistence_conformance.py`, profile wiring | `check_observability_persistence_conformance.py` |
| OBS-BUS-6 | OBS6 | **Export sinks** — OTLP dual-write from unified journal; parser trace link | **Done** | `journal_export.py`, `export_bridge.py`, `task_events.py`, `platform_wiring.py` | `TASK_COMPLETED` carries `journal_ref`; export plugin dual-writes OTLP JSON + parser trace |
| OBS-BUS-7 | OBS7 | **CI gates** — emission coverage + schema registry + L4 §21 evidence | **Done** | `scripts/check_observability_gates.py`, emission/schema/persistence audits, CI workflow | Gate suite green; audit map §21 → **L4** |

### OBS-BUS — Execution order (recommended)

```text
OBS-BUS-0 (docs) → OBS-BUS-1 (typed payloads)
  → OBS-BUS-2 (emitter + TraceScope)
  → OBS-BUS-3 (coverage gaps)
  → OBS-BUS-4 (extension SDK)
  → OBS-BUS-5 (persistence)
  → OBS-BUS-6 (sinks)
  → OBS-BUS-7 (gates / L4 closeout)
```

**DoD:** All OBS-BUS rows **Done**; `build_unified_run_journal` reproduces full Nexus+AgentEngine path without reading source; every `RuntimeEventType` in §42.1.2 has ≥1 production emitter; `parent_event_id` populated for tool/LLM/delegation; extension scaffold documented; gate green.

**Explicitly excluded:** product-specific dashboards (§6.3a); replacing external APM as mandatory deployment.

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (REL-DOC.1 + REL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §22; H-APP `ReliabilityProfile` **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix R**.

**Priority ladder:** **Band 2u** (§4.0) — closed; default queue = **§6.1** maintenance.

### REL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| REL-DOC.1 | REL0 | **Appendix R** — reliability control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| REL-1 | REL1 | **`reliability_runtime_bridge`** + **`reliability_wiring`** | **Done** | `reliability_runtime_bridge.py`, `reliability_wiring.py`, `runtime_config_bridge.py` | `test_harness_reliability_wiring.py` |
| REL-2 | REL2 | **`reliability_assembly_resolver`** — profile ↔ stores conformance | **Done** | `reliability_assembly_resolver.py`, `harness_host_runtime.py` | assembly validation tests |
| REL-3 | REL3 | **Host reliability CI** — `check_harness_reliability_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only retry/fallback policies — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (SEC-DOC.1 + SEC-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §23; V-SEC / V-REM-SEC **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix S**.

**Priority ladder:** **Band 2v** (§4.0) — closed; default queue = **§6.1** maintenance.

### SEC — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| SEC-DOC.1 | SEC0 | **Appendix S** — security control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| SEC-1 | SEC1 | **`security_runtime_bridge`** + **`security_wiring`** | **Done** | `security_runtime_bridge.py`, `security_wiring.py`, `runtime_config_bridge.py` | `test_harness_security_wiring.py` |
| SEC-2 | SEC2 | **`security_assembly_resolver`** — profile ↔ middleware conformance | **Done** | `security_assembly_resolver.py`, `harness_host_runtime.py`, `nexus_factory.py` | assembly validation tests |
| SEC-3 | SEC3 | **Host security CI** — `check_harness_security_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product-only security dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (COST-DOC.1 + COST-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §24; V-COST **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix T**.

**Priority ladder:** **Band 2w** (§4.0) — closed; default queue = **§6.1** maintenance.

### COST — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| COST-DOC.1 | COST0 | **Appendix T** — cost governance control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| COST-1 | COST1 | **`CostProfile`** + **`cost_runtime_bridge`** + **`cost_wiring`** | **Done** | `environment_profile.py`, `cost_runtime_bridge.py`, `cost_wiring.py`, `policy_wiring.py` | `test_harness_cost_wiring.py` |
| COST-2 | COST2 | **`cost_assembly_resolver`** — profile ↔ budget conformance | **Done** | `cost_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py` | assembly validation tests |
| COST-3 | COST3 | **Host cost CI** — `check_harness_cost_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product FinOps dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (EVAL-DOC.1 + EVAL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25; V-EVAL **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix U**.

**Priority ladder:** **Band 2x** (§4.0) — closed; default queue = **§6.1** maintenance.

### EVAL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| EVAL-DOC.1 | EVAL0 | **Appendix U** — evaluation control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| EVAL-1 | EVAL1 | **`EvaluationProfile`** + **`evaluation_runtime_bridge`** + **`evaluation_wiring`** | **Done** | `environment_profile.py`, `evaluation_runtime_bridge.py`, `evaluation_wiring.py`, `policy_wiring.py` | `test_harness_evaluation_wiring.py` |
| EVAL-2 | EVAL2 | **`evaluation_assembly_resolver`** — profile ↔ registry conformance | **Done** | `evaluation_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py`, `runtime.py` | assembly validation tests |
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---



**Status:** **Done** (2026-06-08) — **24/24** deliverables Done (CRIT-V-0 through CRIT-V-7)  
**Prerequisites:** Phase EVAL **Done** (registry wiring), Phase FLOW **Done** (graph hooks), Phase M-LLM-R **Done** (typed LLM envelope)  
**Goal:** Deliver production-grade PEV **Verify** infrastructure — L0/L1/L2 critic stack with tier-separated competencies; uplift Evaluation audit layer L2→L3.  
**Priority ladder:** **Band 2ak** (§4.0) — **Done** (2026-06-08). Default queue reverts to §6.1 gate maintenance.  
**Architecture:** [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · canon [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/ADR-CRITIC-001.md)  
**Audit alignment:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25 (Evaluation), §7 (Reasoning), §10 (Multi-agent); closes **FAUDIT-EVAL.1** residual  
**Execution order:** [§6.2ak](#62ak-phase-crit-v-execution-order-band-2ak--closed) · queue: [§6.1ak](#61ak-harness-implementation-queue--critic-verification-layer-closed)

**Delivery rule:** One **CRIT-V-*** ID per PR → update master table + §6.1ak + gate green.

### CRIT-V — Master register

| ID | Wave | Deliverable | Status | Modules / docs | Acceptance |
|----|------|-------------|--------|----------------|------------|
| CRIT-V-0.1 | 0 | **Architecture RFC** — CVL full spec | **Done** | `architecture/CRITIC_VERIFICATION.md` | Linked from canon §55, README |
| CRIT-V-0.2 | 0 | **ADR-CRITIC-001** — tier-separated PEV verify | **Done** | `docs/adr/ADR-CRITIC-001.md` | Status Accepted; adr index |
| CRIT-V-0.3 | 0 | **Canon §55** addendum | **Done** | `intergrax_runtime_architecture.md` §55 | Cross-links resolve |
| CRIT-V-0.4 | 0 | **README** sections (root + docs) | **Done** | `README.md`, `docs/README.md` | Navigation table |
| CRIT-V-1.1 | 1 | **`CriticProfile`** on `ApplicationEnvironmentProfile` | **Done** | `contracts/environment_profile.py`, `critic_runtime_bridge.py`, `RuntimeConfig` | Unit: `test_harness_critic_wiring.py` |
| CRIT-V-1.2 | 1 | **CVL contracts** — `CriticRequest`, `CriticVerdict`, `LayerVerdict`, `RubricSpec` | **Done** | `runtime/critic/contracts.py` | Unit: `test_critic_contracts.py` |
| CRIT-V-1.3 | 1 | **`EvaluatorLoopSpec`** — max iterations, revise routing | **Done** | `runtime/critic/evaluator_loop_spec.py` | Unit: `test_evaluator_loop_spec.py` |
| CRIT-V-2.1 | 2 | **`eval.judge` tool** — semantic scoring via separate LLM profile | **Done** | `tools/providers/eval/judge.py`, bundle | `test_eval_critic_tools.py` |
| CRIT-V-2.2 | 2 | **`eval.trajectory` tool** — process scoring from replay slice | **Done** | `tools/providers/eval/trajectory.py` | Uses `trace_reader` |
| CRIT-V-2.3 | 2 | **Registry hook** — judge/trajectory → `OnlineEvaluationObservation` | **Done** | `service.py` `_append_critic_observation` | Observation appended when registry bound |
| CRIT-V-3.1 | 3 | **`CriticOrchestrator`** — L0→L1→L2 pipeline | **Done** | `runtime/critic/critic_orchestrator.py` | Unit: short-circuit, layer order |
| CRIT-V-3.2 | 3 | **`L0Gateway`** — wraps `NexusValidationEngine` + schema | **Done** | `runtime/critic/l0_gateway.py` | Reuses existing validators |
| CRIT-V-3.3 | 3 | **`L1Gateway`** — invokes eval tools via `CriticEvalToolClient` | **Done** | `runtime/critic/l1_gateway.py` | No direct LLM in Tier-1 |
| CRIT-V-3.4 | 3 | **Graph partial hook** — `GraphExecutor` → `verify_partial` | **Done** | `graph_executor.py`, `critic_wiring.py` | Integration test: L0 fail → retry |
| CRIT-V-3.5 | 3 | **Graph final hook** — `GraphRunner` → `verify_final` | **Done** | `graph_runner.py` | Terminal state respects verdict |
| CRIT-V-3.6 | 3 | **Critic trace events** — `critic.*` trace steps | **Done** | `runtime/critic/trace.py`, `trace_bridge.py` | Visible in lab trace API |
| CRIT-V-4.1 | 4 | **`EvaluatorLoopExecutor`** — critique→revise routing | **Done** | `runtime/critic/evaluator_loop_executor.py` | Unit: budget exhaustion → FAIL/HITL |
| CRIT-V-4.2 | 4 | **Graph integration** — `EVALUATOR_LOOP` pattern wired | **Done** | `graph_executor.py`, `evaluator_loop_metadata.py` | Acceptance: 2-iteration loop |
| CRIT-V-5.1 | 5 | **`NexusEvalRunner` semantic mode** — optional L1 via `eval.judge` | **Done** | `eval/nexus_eval_runner.py` | Integration: non-exact pass |
| CRIT-V-5.2 | 5 | **`EvalCase` rubric field** — rubric_ref + semantic_threshold | **Done** | `eval/eval_case.py` | Backward compatible |
| CRIT-V-6.1 | 6 | **`wire_application_critic()`** — Tier-3 wiring | **Done** | `applications/_shared/critic_wiring.py` | Mirror EVAL pattern |
| CRIT-V-6.2 | 6 | **`critic_assembly_resolver`** — wire-time validation | **Done** | `critic_assembly_resolver.py`, `check_harness_critic_wiring.py` | CI script |
| CRIT-V-6.3 | 6 | **Policy bundle** — `critic_governance` fragment | **Done** | `policy_wiring.py` | Merged at host build |
| CRIT-V-6.4 | 6 | **Appendix W** — critic control plane author map | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| CRIT-V-7.1 | 7 | **FAUDIT-EVAL.1** — `require_baseline_for_release` CI gate | **Done** | `phase_v_closeout_gate.py`, `check_harness_critic_wiring.py` | Closeout gate green |
| CRIT-V-7.2 | 7 | **Flow reference §18 sync** — CVL hook table | **Done** | `architecture/NEXUS_EXECUTION_FLOW.md` | Hooks documented |
| CRIT-V-7.3 | 7 | **Lab harness demo** — L0+L1 on sample agent (not FLOW-8) | **Done** | `test_harness_critic_wiring.py`, lab host | Trace shows critic steps |
| CRIT-V-F.1 | F | **`ToolRegistryCriticEvalClient`** — L1 bridge to Tier-0 eval tools | **Done** | `runtime/critic/tool_registry_client.py`, `critic_tool_wiring.py` | `test_critic_closeout.py` |
| CRIT-V-F.2 | F | **`critic_llm_profile`** — separate judge LLM adapter | **Done** | `critic_llm_resolver.py`, `environment_profile.py` | Assembly + wiring tests |
| CRIT-V-F.3 | F | **L2 `L2Gateway`** + HITL escalation path | **Done** | `l2_gateway.py`, `critic_orchestrator.py` | `test_critic_l2_gateway.py` |
| CRIT-V-F.4 | F | **UAEP step hook** | **Done** | `uaep.py`, `validate_uaep_step_with_critic_detail` | UAEP critic path |
| CRIT-V-F.5 | F | **`CriticPolicyBridge`** + policy engine | **Done** | `policy_bridge.py`, `runtime_policy_engine.py` | `test_critic_closeout.py` |
| CRIT-V-F.6 | F | **Assembly gate** — require L1 client when semantic/trajectory enabled | **Done** | `critic_assembly_resolver.py` | `test_critic_assembly_resolver.py` |

**Explicitly excluded:** FLOW-8 §42.43 product reference app ([§6.3](#63-end-of-plan--deferred-product-work-only)); domain rubric packs in Tier-0; mandatory universal LLM-judge on all runs.

**Phase CRIT-V complete when:** CRIT-V-1 through CRIT-V-7 **Done**; Evaluation audit layer ≥ **L3**; gate green; FAUDIT-EVAL.1 closed.

---

---

## Phase V-REM — Phase V Runtime Remediation (audit closeout)

**Source:** Plan/code audit (2026-06-05) — reconcile Phase V **Done** claims vs runtime evidence; aligned with [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) layers 5, 19, 21, 23, 25, 26.  
**Status:** **Done** (2026-06-05) — **10/10 Done**.  
**Prerequisites:** Phase V contracts **Done**; Phase H-APP **Done** (Tier-3 `ApplicationSecurityProfile` hooks exist).  
**Goal:** Close every **Partial** Phase V row and **A.4** EvalRunner gap — move from governance/evidence-only to **runtime-enforced** behavior. **Achieved 2026-06-05.**  
**Priority ladder:** **Band 2i** (§4.0) — closed.  
**Execution order:** [§6.2v](#62v-phase-v-rem-execution-order-band-2i--closed-2026-06-05).  
**Traceability:** [Appendix J](#appendix-j--phase-v-remediation-traceability-audit-gap--v-rem-id).

**Explicitly out of scope:** K.1/K.2, new product Tier-3 apps, full 32-layer re-audit (use [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) separately).

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

---

## Phase FAUDIT-32 — Full architecture audit closeout

**Status:** **Done** (2026-06-06) — 32-layer audit (`scope: C`) + **23/23 FAUDIT remediation** implemented → [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed)  
**Source:** [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8  
**Traceability:** **Appendix M** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done.

**Post depth bands (2026-06-09):** MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG closeout complete — Appendix M scorecard refreshed; remaining L2 rows are incremental depth only.

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L3 | 0 | 0 | Yes |
| 3 | Interface and Task Intake | L3 | 0 | 1 | Partial |
| 4 | Identity, Trust and Tenancy | L2 | 0 | 2 | Partial |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L3 | 0 | 0 | Yes |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 0 | Yes |
| 10 | Subagents and Multi-Agent Coordination | L2 | 0 | 2 | Partial |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L3 | 0 | 0 | Yes |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L2 | 0 | 1 | **No** |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L2 | 0 | 2 | **No** |
| 21 | Observability and Telemetry | L2 | 0 | 2 | **No** |
| 22 | Error Handling and Reliability | L2 | 0 | 1 | **No** |
| 23 | Security and Data Governance | L2 | 0 | 2 | **No** |
| 24 | Cost and Resource Governance | L2 | 0 | 1 | **No** |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 1 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 1 | Partial |
| 31 | Agent Lifecycle Governance | L2 | 0 | 2 | Partial |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

### FAUDIT-32 — Remediation register (implementation queue → §6.1ah)

| ID | Layer | Gap | Severity | Module / acceptance |
|----|-------|-----|----------|-------------------|
| FAUDIT-TIER.1 | §2 | Tier-0 imports `applications/*` in `capability_graph_applications.py` | **Critical** | Move manifest catalog to Tier-3 injection or static metadata; zero `from applications` under `intergrax/` |
| FAUDIT-TIER.2 | §2 | No CI gate for `intergrax/` → `applications/` imports | High | `scripts/check_intergrax_no_applications_imports.py` in §6.1 |
| FAUDIT-INTAKE.1 | §3 | No canonical `TaskEnvelope`; `Task` + `RuntimeRequest` split | High | Typed envelope alias or consolidation; plan W-OPS.6 naming sync |
| FAUDIT-INTAKE.2 | §3 | Worker≡HTTP intake parity test matrix incomplete | High | Acceptance test: CLI/worker/HTTP same `Task` shape |
| FAUDIT-ID.1 | §4 | No user/service/agent identity distinction | High | Identity contracts + propagation to delegation |
| FAUDIT-ID.2 | §4 | `DelegationSpec` lacks permission scope audit | High | Scope field + trace on child runs |
| FAUDIT-POL.1 | §5 | No pre-LLM / pre-output policy hooks in runtime | High | PolicyEngine extension points documented + wired |
| FAUDIT-LLM.1 | §6 | No policy-driven model routing / fallback chain | High | Router module or AdaptiveProfile integration |
| FAUDIT-COG.1 | §7 | No universal `DecisionRecord` per UAEP step | High | Typed decision artifact + trace event |
| FAUDIT-ORCH.1 | §9 | No graph backpressure beyond parallel cap | High | Queue depth / shed policy in `GraphExecutor` |
| FAUDIT-SUB.1 | §10 | No formal `SubtaskContract`; `inherit_tool_policy=True` default | High | Contract type + safer delegation defaults |
| FAUDIT-MEM.1 | §15 | Entity graph memory absent; STM retention partial | High | Route to MEM-9.* / new MEM row if scoped |
| FAUDIT-PE.1 | §17 | No golden prompt content regression in CI | High | Golden YAML fixtures + gate test |
| FAUDIT-REG.1 | §19 | `HarnessRegistrySnapshot` omits agents + eval registry | High | Extend snapshot + assembly resolver |
| FAUDIT-CG.1 | §20 | Capability graph seed skips `prompt:*` nodes | High | `_seed_node_ids()` parity |
| FAUDIT-CG.2 | §20 | Blast-radius not enforced at release | High | `phase_v_capability_graph_guard.py` impact check |
| FAUDIT-OBS.1 | §21 | `RuntimeEventType` missing `LLM_CALL` / `POLICY_DECISION` | High | Canon event catalog + bridge from trace |
| FAUDIT-REL.1 | §22 | Shallow error taxonomy (`VALIDATION_ERROR` only + 2) | High | Expand classifier per AUDIT_MAP §22 |
| FAUDIT-SEC.1 | §23 | No `DataClassification` model | High | Security profile + enforcement hooks |
| FAUDIT-COST.1 | §24 | Per-tenant cost attribution not mandatory in NexusLoop | High | Budget gate on main path |
| FAUDIT-EVAL.1 | §25 | `require_baseline_for_release` not CI-enforced | High | `phase_v_closeout_gate.py` eval baseline check |
| FAUDIT-ALG.1 | §31 | Lifecycle states ≠ AUDIT_MAP catalog; weak agent adoption | High | Align or document ADR; scaffold defaults |
| FAUDIT-OPS.1 | §30 | `release_cycles.json` not in repo; W-OPS.5 artifact policy unclear | High | Document committed vs generated artifact |

**Delivery rule:** One **FAUDIT.\*** ID per PR → update §6.1ah + Appendix M paydown log → gate green.

**Explicitly out of scope (audit-and-fix):** source code or test changes during this audit pass; K.1/K.2; new product Tier-3 apps.

---

---

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
| Q-N.10 | **`production_mode` lab default** — `lab_application` / scaffold sets `production_mode=False`; document in Step 4E | **Done** | Low | Tier-3 factories, `guides/AGENT_CREATION_GUIDE.md` | `harness_production_mode()` in `applications/_shared/runtime_defaults.py` |
| Q-N.11 | **Graph callback typing** — `ExecutionNode` instead of `object` in `GraphExecutor` / NexusLoop node callbacks | **Done** | Low | `execution/graph_executor.py`, `nexus_loop.py` | Mypy/ruff on execution package |
| Q-N.12 | **Interrupt handler hygiene** — remove duplicate `InterruptType` import; add unit test for interrupt → policy path | **Done** | Low | `interrupts/handler.py` | Duplicate import removed (2026-06-01) |
| Q-N.13 | **`AgentEngine` static UAEP** — document or inject `event_bus` for `AgentEngine.run` static path; no silent missing events | **Done** | Low | `agents/agent_engine.py` | `_resolve_static_executor`; `tests/unit/agents/test_agent_engine_event_bus.py` |
| Q-N.14 | **Unit tests for `NexusLoop` helpers** — `_finish_task`, lifecycle transitions, HITL branch stubs (mock deps) | **Done** | High | `tests/unit/runtime/nexus/test_nexus_loop.py` | New file; ≥15 focused tests; marker `gate` |
| Q-N.15 | **`GraphExecutor` unit coverage** — failure recovery, skip completed, handoff edge (beyond stub integration) | **Done** | Medium | `tests/unit/runtime/execution/` | `test_graph_executor_coverage.py` + checkpoint skip in `test_runtime_checkpoint.py` |

---

#### Phase Q-L — LLM adapters

**Components:** `intergrax/llm_adapters/`, `docs/architecture/LLM_ADAPTERS.md`, governance plugin.

| # | Deliverable | Status | Priority | Location / notes | Acceptance |
|---|-------------|--------|----------|------------------|------------|
| Q-L.1 | **Remove or complete `tracked_llm_call`** — if kept: `finally` calls `usage.end_call`; if removed: delete `tracked_call.py` + references | **Done** | Medium | `_shared/tracked_call.py` | File removed (unused) (2026-06-01) |
| Q-L.2 | **Public API surface** — re-export `LLMAdapter`, `LLMProvider`, `LLMAdapterRegistry`, `LLMProfile` from `llm_adapters/__init__.py` | **Done** | Low | `llm_adapters/__init__.py` | Public re-exports (2026-06-01) |
| Q-L.3 | **Provider catalog table in docs** — 19 rows: slug, adapter class, env vars, tools/stream/structured, native vs compat | **Done** | High | `docs/architecture/LLM_ADAPTERS.md` | Table matches `LLMProvider` enum + conformance list |
| Q-L.4 | **Fix `LLMProfile` docstring** — `max_retries` only via `options={}`; align examples in guide | **Done** | Low | `registry/profile.py`, tests | Example fixed (2026-06-01) |
| Q-L.5 | **Per-provider `supports_streaming()` / `supports_structured_output()`** — override defaults (`False` base default for streaming); table in Q-L.3 | **Done** | Medium | Each `providers/*.py`, ABC defaults | Conformance reads flags; no false positives |
| Q-L.6 | **`PolicyEngine` + `llm_cost_evaluation`** — rule hook on `TASK_COMPLETED` or policy replay; or remove “next step” from docs until done | **Done** | Medium | `governance/`, `observability_bridge.py`, `policy_engine.py` | Test: over-quota/warn triggers policy decision or structured log contract |
| Q-L.7 | **Usage tracking doc** — distinguish adapter `LLMAdapterUsageLog` vs runtime `LLMUsageTracker` | **Done** | Low | `docs/architecture/LLM_ADAPTERS.md` § Observability | Two-layer table |
| Q-L.8 | **Conformance: structured output** — parametrize providers with `supports_structured_output`; mock SDK | **Done** | Medium | `tests/unit/llm_adapters/` | Added to gate subset in `llm-adapters-guard.yml` |
| Q-L.9 | **Bedrock `context_window_tokens`** — lookup table or model metadata for common `model_id` | **Done** | Low | `providers/aws_bedrock_adapter.py` | `_CONTEXT_WINDOWS` + prefix fallback; `test_bedrock_context_window.py` |
| Q-L.10 | **OpenAI-compat adapter init** — replace `__dict__.update` with explicit delegation or composition wrapper | **Done** | Low | `openai_compat_providers.py`, factory | `_delegate` + `__getattr__` composition |
| Q-L.11 | **Central env appendix** — single table: `INTERGRAX_LLM_*`, secrets map, per-provider overrides | **Done** | Medium | `architecture/LLM_ADAPTERS.md` appendix | Cross-links from each `providers/*/USAGE.md` |

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
| Q-O.13 | **Clarify dual Prometheus** — in-process scrape vs `integrations` PromQL backend | **Done** | Low | `docs/architecture/LLM_ADAPTERS.md` § Observability | Prevents operator confusion |
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
| Q+-L.5 | **`chains/langchain_qa_chain`** — removed from harness (package deleted) | **Done** | Medium | — | No `intergrax.chains` imports |
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
- `stable` on all **135** integration slugs — only the **lab harness stack** (see S-Ops.1)

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
| S-Ops.1 | **Integration stable track** — lab harness stack (`sqlite`, `redis`, `qdrant`, `slack`, `sentry`, …) marked `stable` in catalog | **Done** | **Critical** | `harness_lab_stack.py`, `architecture/INTEGRATIONS.md` | `test_harness_lab_stable_stack.py` |
| S-Ops.2 | **OTLP / observability** — lab profile wires `otel` when `LAB_OTEL_ENABLED`; document noop vs export | **Done** | High | `IntegrationProfile.harness_environment()`, `.env.example` | `test_lab_harness_environment_wiring.py` |
| S-Ops.3 | **Harness-smoke CI** — expand M.12+ coverage for stable stack (network optional) | **Done** | Medium | `.github/workflows/unit-tests.yml` | harness-smoke includes S unit tests |
| S-Ops.4 | **Legal live LLM E2E** | **Deferred** | Low | K.6 / B.15 | Not blocking harness environment |

#### S-H — Platform harness capabilities (no business agents)

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-H.1 | **Platform skill bundle `harness`** — ≥3 skills (e.g. `harness.tool_smoke`, `harness.context_demo`, `harness.trace_read`) | **Done** | **Critical** | `intergrax/skills/providers/harness/`, `architecture/SKILLS.md`, bootstrap | `test_harness_skill_bundle.py` |
| S-H.2 | **Lab wiring** — `SkillProfile` + `ToolProfile` + policy bundle documented as canonical harness preset | **Done** | High | `skill_wiring.py`, `guides/HARNESS_ENVIRONMENT.md` | lab enables `harness` bundle |
| S-H.3 | **Cursor SKILL.md importer** in gate | **Done** | Medium | `tests/unit/skills/importers/test_cursor_skill_md.py` | `pytest.mark.gate` |
| S-H.4 | **`rag.answers` test migration** — no deprecation warnings in gate | **Done** | Low | `tests/integration/rag/answers/` | `RetrievalService` only |
| S-H.5 | **Echo/signoff path** — lab run proves skills + trace + policy bundle (existing agents) | **Done** | High | `tests/acceptance/agent_os/test_lab_application.py` | gate + harness wiring tests |

#### S-Doc — Operator & author surfaces

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| S-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — lab stack, env vars, stable integrations, OTLP, policy bundle read order | **Done** | **Critical** | `docs/guides/HARNESS_ENVIRONMENT.md`, `docs/README.md` index | Linked from plan §6 |
| S-Doc.2 | **Context / trace operator section** — `CONTEXT_*` events, debug API, metrics routes | **Done** | Medium | `guides/HARNESS_ENVIRONMENT.md` | Pointers to canon §28.1 |

#### Phase S — Definition of done

1. **Stable** integration list for lab harness stack published and tested (S-Ops.1).
2. **OTLP path** documented and wired for lab when env configured (S-Ops.2).
3. **≥ 3** `harness.*` platform skills + legal/research bundles registered (S-H.1).
4. **`guides/HARNESS_ENVIRONMENT.md`** complete; lab wiring matches doc (S-H.2, S-Doc.1).
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
| U-Sec.2 | **MCP default opt-in** — `LAB_INCLUDE_MCP=false` default for strict profile; document in `guides/HARNESS_ENVIRONMENT.md` | **Done** | High | `LabApplicationSettings`, `.env.example` | `test_lab_application_settings_phase_u.py` |
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
| U-Doc.1 | **`guides/HARNESS_ENVIRONMENT.md`** — security (auth, MCP), strict profile, policy bundle wiring truth | **Done** | High | `docs/guides/HARNESS_ENVIRONMENT.md` | Phase U security section |
| U-Doc.2 | **Canon §52 / strategy** — lab vs production harness checklist references Phase U | **Won't fix** | Medium | — | Deferred; plan + HARNESS_ENVIRONMENT sufficient |
| U-Doc.3 | **Fix Phase K footer** in `guides/HARNESS_ENVIRONMENT.md` (post-T, gated on U) | **Done** | Low | `guides/HARNESS_ENVIRONMENT.md` | Gated on Phase U |

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

---

#### V-AM — Architecture Metrics & Debt Governance

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-AM.1 | Architecture health metric spec (modularity/dependency/coverage/debt) | **Done** | **Critical** | Canon metrics section + thresholds |
| V-AM.2 | Metrics emission pipeline and dashboards | **Done** (pipeline + trend/gate contracts) | High | Dashboard + alert definitions |
| V-AM.3 | Governance coverage and observability coverage measurement | **Done** | High | Coverage reports generated in CI |
| V-AM.4 | Architecture debt index + periodic review process | **Done** | High | Debt report cadence defined and used |#### V-SEC — Security & Data Governance Hardening

| # | Deliverable | Status | Priority | Acceptance |
|---|-------------|--------|----------|------------|
| V-SEC.1 | Prompt injection defense profile + tests | **Done** | **Critical** | Adversarial tests in gate subset |
| V-SEC.2 | Tool injection defense (schema/argument/capability controls) | **Done** | High | `ToolInjectionDefenseMiddleware` on `BEFORE_TOOL_CALL` via `application_security_wiring.py` |
| V-SEC.3 | Retrieval poisoning defense (trust score/quarantine flow) | **Done** | High | `retrieval_security_wiring.py` filters chunks in `RagStep` when profile enabled |
| V-SEC.4 | Tenant isolation verification + security audit trail checks | **Done** | High | `TenantSecurityMiddleware` on `BEFORE_TASK_INTAKE` |

---

## Appendix B


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
| 2026-05-30 | M.7-agent-guide-integrations | `guides/AGENT_CREATION_GUIDE.md` Appendix E — agents vs Tier-3 wiring |
| 2026-05-30 | N.2.1-unified-wiring | `ApplicationBuildContext`, `builder_key`/`factory_path`, lab+legal on `build_application_registry` |
| 2026-05-30 | N.2-conformance | `build_registry_from_manifest`, `load_agent_from_binding` + unit tests |
| 2026-05-30 | N.1-manifest | `ApplicationManifest`, `AgentBinding`, `ApplicationFeatures` + unit tests |
| 2026-05-30 | N.10-new-stack | `scaffold new-stack` — agent + application; `TIER3_READINESS.md` |
| 2026-05-30 | N.9-scaffold-acceptance | `test_scaffold_acceptance.py` — lab/product runtime E2E; fix product `agent_factories.py` indent |
| 2026-05-30 | N.8-agent-guide-4e | `guides/AGENT_CREATION_GUIDE.md` Step 4E — `new-application`, Docker scripts, §7.4.8 links |
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

---

## Appendix C


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
| L-03 | `architecture/LLM_ADAPTERS.md` missing provider table | Q-L.3 | Done |
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
| 2026-06-01 | docs-consolidation | Merged LLM/RAG observability, retry, trace ADR into canon + `architecture/LLM_ADAPTERS.md`; removed satellite `docs/*.md` |
| 2026-06-01 | Q-N.1,Q-X.2,Wave 9 | `graph_runner`, `task_events`, `lifecycle_bridge`; UAEP `execution_options_for_request`; gate **417 passed** |
| 2026-06-01 | Q-X.2(partial),Q-X.4,Q-X.5 | Legacy metadata warnings; `tools_base` timeline; M.6 beta slugs; gate **415 passed** |
| — | — | *(append row per merged PR)* |

**Coverage:** 58 audit rows → 49 unique Q deliverables (some Q IDs satisfy multiple rows). **Target:** 100% **Done** or **Won't fix** — **achieved** (Phase Q complete).

**Appendix B relationship:** Closed by Phase Q where mapped. Residual items tracked in **Appendix D** (Phase Q+).

---

---

## Appendix D


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

---

## Appendix E


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

---

## Appendix F


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
| 2026-06-01 | **Phase S** | harness_lab_stack, harness.* skills, OTEL profile, guides/HARNESS_ENVIRONMENT.md, tests |
| — | — | *(append row per merged PR)* |

**Coverage target:** Phase S definition of done met — **yes** (2026-06-01).

---

---

## Appendix G


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
| DOC-01 | `guides/HARNESS_ENVIRONMENT.md` claims policy bundle wired — lab does not apply bridge | U-Doc.1, U-Pol.1 | Done |
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

---

## Appendix M


---

## Appendix M — Full architecture audit traceability (Phase FAUDIT-32)

**Purpose:** 100% mapping from 32-layer [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8 audit to concrete **FAUDIT.\*** remediation IDs. **Canonical phase narrative:** [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout).

**Status:** **Done** (2026-06-06) · **23/23 remediation Done** + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed) follow-up · gate **901**

### M.1 Layer → FAUDIT ID matrix (High + Critical only)

| Layer | AUDIT_MAP § | Gap summary | Severity | FAUDIT ID |
|-------|-------------|-------------|----------|-----------|
| Tier boundaries | §2 | `intergrax/runtime/architecture/capability_graph_applications.py` imports `applications.*` | **Critical** | FAUDIT-TIER.1, FAUDIT-TIER.2 |
| Task intake | §3 | No `TaskEnvelope`; worker≡HTTP parity incomplete | High | FAUDIT-INTAKE.1, FAUDIT-INTAKE.2 |
| Identity | §4 | No service/agent identity; delegation scope | High | FAUDIT-ID.1, FAUDIT-ID.2 |
| Policy | §5 | Pre-LLM/pre-output hooks absent | High | FAUDIT-POL.1 |
| LLM adapters | §6 | No policy-driven routing | High | FAUDIT-LLM.1 |
| Cognition | §7 | No `DecisionRecord` per step | High | FAUDIT-COG.1 |
| Orchestration | §9 | No backpressure | High | FAUDIT-ORCH.1 |
| Subagents | §10 | No `SubtaskContract` | High | FAUDIT-SUB.1 |
| Memory | §15 | Entity graph memory; STM retention | High | FAUDIT-MEM.1 |
| Prompts | §17 | No golden prompt CI | High | FAUDIT-PE.1 |
| Registry | §19 | Snapshot omits agents/eval | High | FAUDIT-REG.1 |
| Capability graph | §20 | Missing prompt nodes; no release impact gate | High | FAUDIT-CG.1, FAUDIT-CG.2 |
| Observability | §21 | Missing `LLM_CALL`/`POLICY_DECISION` events | High | FAUDIT-OBS.1 |
| Reliability | §22 | Shallow error taxonomy | High | FAUDIT-REL.1 |
| Security | §23 | No `DataClassification` | High | FAUDIT-SEC.1 |
| Cost | §24 | Tenant attribution not mandatory | High | FAUDIT-COST.1 |
| Evaluation | §25 | Release baseline not CI-enforced | High | FAUDIT-EVAL.1 |
| Lifecycle | §31 | State catalog mismatch; weak adoption | High | FAUDIT-ALG.1 |
| Ops / SLOs | §30 | `release_cycles.json` artifact policy | High | FAUDIT-OPS.1 |

### M.2 Cross-layer themes

| Theme | Layers affected | Risk |
|-------|-----------------|------|
| **Closeout vs maturity** | §17–§25, §31 | Plan **Done** on wiring; AUDIT_MAP **L2** on depth — do not conflate |
| **Dual-path telemetry** | §21, §6 | **L4 Done:** [Phase OBS-BUS](#phase-obs-bus--unified-observability-spine) — unified journal, `ObservabilityEmitter`, typed payloads, emission coverage, journal export |
| **Tier boundary drift** | §2, §28 | Single Critical violation undermines canon §7.4.4 |
| **Identity / intake naming** | §3, §4 | Resolved — `TaskEnvelope` in `intergrax/contracts/task_envelope.py`; parity tests in `test_faudit_remediation.py` |

### M.3 Paydown log

| Date | FAUDIT ID | Summary |
|------|-----------|---------|
| 2026-06-06 | FAUDIT-32.0 | Full 32-layer audit (`scope: C`, `audit-and-fix`); scorecard + §6.1ah queue + Appendix M; gate **893**; boundary scripts OK |
| 2026-06-06 | FAUDIT-TIER.1–OPS.1 | **23/23** remediation implemented; tier gate + intake + observability + registry depth |
| 2026-06-06 | FAUDIT-PE.1+/ALG.1+/MEM.1+ | Golden prompt CI, reference agent lifecycle metadata, STM retention wiring; gate **901** |
| 2026-06-09 | PLAN-SYNC | Hub closeout: MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG, CRIT-V, ORCH-5; Appendix M refresh |
| 2026-06-09 | CFG-HOST | Reference host presets (`with_reference_host_platform_defaults`); dispute_sim interactions; scheduler defaults |
| 2026-06-07 | OBS-DEPTH.* + T12 + LEG depth | Unified journal + trace bridge gate + live bus emit + 170-tool catalog + §21 L3 depth gate; gate **967** |
| 2026-06-07 | T13 + CRIT-V-2.* | `eval.judge` + `eval.trajectory`; catalog **172**; doc sync; gate **990** |
| 2026-06-07 | CRIT-V-3.1–3.3 | `CriticOrchestrator`, `L0Gateway`, `L1Gateway`, `CriticEvalToolClient` | gate **996** |

---
