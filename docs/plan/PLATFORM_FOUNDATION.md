# Platform Foundation — Implementation Plan

**Architecture (1:1):** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (satellites under [`plan/plan/`](plan/plan/) on demand).

---

## Documentation model

This file is the **implementation plan hub**. Detailed registers and appendices: [`plan/plan/`](plan/plan/) — **load on demand**.

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
| **Nexus execution flow (runtime narrative, diagrams, gap → plan rows)** | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) · [Phase FLOW](plan/ORCHESTRATION.md) · **§6.1aj** · Band **2aj** · **Appendix N (FLOW)** · [ADR-FLOW-001](adr/entries/2026-06-07/ADR-FLOW-001.md) |
| Governance audit closeout (docs + residuals register) | [Phase GOV-AUDIT](plan/UNIFIED_EXECUTION_RUNTIME.md) · **GOV-DOC.\*** **Done** |
| Orchestration audit closeout (runtime wiring) | [Phase ORCH](plan/ORCHESTRATION.md) · **§6.1b** · Band **2j** |
| Tools / skills audit closeout (runtime bridge) | [Phase TS](plan/TOOLS.md) · **§6.1c** · Band **2k** · `guides/AGENT_CREATION_GUIDE.md` **Appendix J** |
| Integration audit closeout (runtime bridge + health) | [Phase INT](plan/INTEGRATIONS.md) · **§6.1d** · Band **2l** · **Appendix K** |
| RAG audit closeout (runtime bridge) | [Phase RAG](plan/RAG.md) · **§6.1e** · Band **2m** · **Appendix K** §K.5 |
| Context engineering closeout (runtime + Nexus wiring) | [Phase CTX](plan/MEMORY.md) · **§6.1f** · Band **2n** · **Appendix L** |
| Prompt registry closeout (runtime + environment wiring) | [Phase PE](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1i** · Band **2p** · **[Appendix M](plan/plan/PLATFORM_FOUNDATION_appendices.md)** |
| Legacy module closeout (chat_router, tools_agent, chains) | [Phase CLEAN](plan/ORCHESTRATION.md) · **§6.1j** |
| Agent assembly closeout (contracts, capabilities, lifecycle) | [Phase AS](plan/ORCHESTRATION.md) · **§6.1k** · Band **2q** · **Appendix N** |
| Registry architecture closeout (snapshots, conformance, CI) | [Phase REG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1l** · Band **2r** · **Appendix O** |
| Capability graph closeout (environment slice, blast-radius wire) | [Phase CG](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) · **§6.1m** · Band **2s** · **Appendix P** |
| Observability closeout (profile bridge, assembly resolver, CI) | [Phase OBS](plan/OBSERVABILITY.md) · **§6.1n** · Band **2t** · **Appendix Q** |
| **Unified Observability Spine (full mechanism)** | [Phase OBS-BUS](plan/OBSERVABILITY.md) · **§6.1al** · Band **2al** · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md) |
| Reliability closeout (idempotency bridge, circuit breaker, CI) | [Phase REL](plan/OBSERVABILITY.md) · **§6.1o** · Band **2u** · **Appendix R** |
| Security closeout (V-SEC bridge, middleware assembly, CI) | [Phase SEC](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1q** · Band **2v** · **Appendix S** |
| Cost governance closeout (budget bridge, policy bundle, CI) | [Phase COST](plan/UNIFIED_EXECUTION_RUNTIME.md) · **§6.1r** · Band **2w** · **Appendix T** |
| Evaluation closeout (registry bridge, policy bundle, CI) | [Phase EVAL](plan/CRITIC_VERIFICATION.md) · **§6.1s** · Band **2x** · **Appendix U** |
| **Critic & Verification Layer (PEV verify depth)** | [Phase CRIT-V](plan/CRITIC_VERIFICATION.md) · **§6.1ak** · Band **2ak** · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md) |
| **Adaptive Harness Intelligence (AHI / L4 runtime)** | [Phase W-ADAPT](plan/CRITIC_VERIFICATION.md) · **§6.1t** · Band **2y** · [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · **Appendix K** |
| **LLM response envelope (typed completion contract)** | [Phase M-LLM-R](plan/LLM_ADAPTERS.md) · **§6.1v** · Band **2z** · [architecture/LLM_ADAPTERS.md](architecture/LLM_ADAPTERS.md) · **Appendix L** |
| **LLM developer excellence (post-audit 2026-06-14)** | [Phase M-LLM-X](plan/LLM_ADAPTERS.md) · **§6.1ax** · Band **2ba** · ModelCatalog · routing · DX |
| **Integration catalog expansion (harness ROI slugs)** | [M.6 P4 register](#m6-p4--harness-platform-expansion-done) · **§6.1w** · Band **2aa** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **Integration harness depth (audit 2026-06-02)** | [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) · **§6.1x** · Band **2ab** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **Integration harness expansion (audit 2026-06-02)** | [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · **§6.1y** · Band **2ac** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) |
| **LLM guardrail integrations (M.12 / GR-INT)** | [Phase M.12](plan/INTEGRATIONS.md) · **§6.1an** · Band **2ay** · [architecture/INTEGRATIONS.md](architecture/INTEGRATIONS.md) §47 · [plan/UNIFIED_EXECUTION_RUNTIME.md](plan/UNIFIED_EXECUTION_RUNTIME.md) GR-DOC |
| Tier-3 application environment (self-contained deploy) | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) |
| Tier-3 composition engine (manifest, wiring API) | [`intergrax/applications/USAGE.md`](../intergrax/applications/USAGE.md) |
| Tier-3 application hosts (`applications/<app>/`) | [`applications/USAGE.md`](../applications/USAGE.md) |
| Application scaffold & deploy plan | **This file** Phase N |
| Business-agent go/no-go checklist | **[Appendix A](plan/PLATFORM_FOUNDATION.md)** |
| Technical debt backlog (analysis only) | **[Appendix B](plan/plan/PLATFORM_FOUNDATION_appendices.md)** |
| Harness quality audit (2026-06-01) → Phase Q tracker | **This file** Phase Q + **[Appendix C](plan/plan/PLATFORM_FOUNDATION_appendices.md)** |
| Post-audit hardening (typing, legacy, monoliths) | **This file** Phase Q+ + **[Appendix D](plan/plan/PLATFORM_FOUNDATION_appendices.md)** |
| Harness GA / consolidation (no new OS features) | **This file** Phase Q / Q+ |
| Harness AI alignment audit (2026-06-01) → Phase R | **This file** Phase R + **[Appendix E](plan/plan/PLATFORM_FOUNDATION_appendices.md)** + canon [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) §5.3 |
| Harness environment GA (2026-06-01) → Phase S | **This file** Phase S + **[Appendix F](plan/plan/PLATFORM_FOUNDATION_appendices.md)** (K.1/K.2 → §6.3 end-of-plan) |
| Harness production hardening (2026-06-01 audit) → Phase U | **This file** Phase U + **[Appendix G](plan/plan/PLATFORM_FOUNDATION_appendices.md)** (**Done**; does **not** schedule K.1/K.2 — see §6.3) |
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
| Phase V remediation traceability (audit gap → V-REM ID) | **[Appendix J](plan/plan/PLATFORM_FOUNDATION_06_phase_detail.md)** |
| Full architecture audit procedure (32 layers) | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · prompt: [`guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](guides/HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) |
| **Full architecture audit closeout (32 layers, scope C)** | [Phase FAUDIT-32](plan/plan/PLATFORM_FOUNDATION_phase_closeout.md) · **§6.1ah** · Band **2ad** · **[Appendix M](plan/plan/PLATFORM_FOUNDATION_appendices.md)** · source: audit 2026-06-06 (`scope: C`, `audit-and-fix`) |
| **Ideal Harness L3 depth (32-layer uplift)** | [Phase IDEAL-L3](plan/IDEAL_HARNESS_L3.md) · **§6.1at** · Band **2ax** |
| **Ideal architecture gap closeout (post-L3 audit 2026-06-09)** | [Phase AUDIT-IDEAL](plan/AUDIT_IDEAL_2026.md) · **§6.1au** · Band **2az** · [`ARCHITECTURE_DEBT_REGISTER.md`](guides/ARCHITECTURE_DEBT_REGISTER.md) |
| Infrastructure vs business scope split | **§4.0a** · closed §6.1b–g queues → [`plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md) · [§6.3a](#63a-business-backlog-register-consolidated) |

**Note on audit source documents:** Some historical audit narratives (e.g. `HARNESS_APPLICATION_LAYER_AUDIT.md`) may live outside the repo. **Task traceability in this plan is canonical** — H-APP (43 tasks), W-OPS, MEM, DX, AA registers below; do not re-derive scope from missing files.


## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor token use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md) | Closed §6.1/§6.2 implementation queues |
| [`plan/plan/PLATFORM_FOUNDATION_06_phase_detail.md`](plan/plan/PLATFORM_FOUNDATION_06_phase_detail.md) | §6 embedded phase/appendix detail (L/M/N, …) |
| [`plan/plan/PLATFORM_FOUNDATION_appendices.md`](plan/plan/PLATFORM_FOUNDATION_appendices.md) | Appendices B–M |
| [`plan/plan/PLATFORM_FOUNDATION_master_registers.md`](plan/plan/PLATFORM_FOUNDATION_master_registers.md) | §5 domain master registers + paydown logs |
| [`plan/plan/PLATFORM_FOUNDATION_phase_closeout.md`](plan/plan/PLATFORM_FOUNDATION_phase_closeout.md) | Phase V-REM, FAUDIT-32 closeout |

> **Cursor context budget:** read this hub + **at most one** satellite per session.
> Closed queues and appendices are **audit-on-demand** only.


## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/plan/PLATFORM_FOUNDATION_embedded_detail.md`](plan/plan/PLATFORM_FOUNDATION_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read this hub + **at most one** satellite per session.

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
| **RAG closeout (Phase RAG)** | **Done** (2026-06-02) | [Phase RAG](plan/RAG.md) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) |
| Product agents (Phase K) | **Deferred** | K.1/K.2 — end of priority list |
| Tier-3 product applications | **Deferred** | New apps / product routes — after harness backlog |



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
**Traceability:** **[Appendix M](plan/plan/PLATFORM_FOUNDATION_appendices.md)** (layer scorecard + gap → FAUDIT ID matrix)

**Audit verdict (2026-06-06, pre-remediation snapshot):** Harness **control-plane wiring closeouts** (ORCH, TS, INT, RAG, CTX, PE, AS, REG, CG, OBS, REL, SEC, COST, EVAL, W-ADAPT, M-LLM-R) are **Done** as documented — but **closeout ≠ full layer maturity**. Per-layer inspection at audit time showed **12/32 layers at L3+**, **19/32 at L2**, **1 Critical** tier-boundary violation, **~20 High** residuals — all routed to **FAUDIT.\*** and **closed** via [§6.1ah](#61ah-harness-implementation-queue--faudit-32-remediation-closed) + [§6.1ai](#61ai-harness-implementation-queue--faudit-32-follow-up-closed).

**Post-remediation (2026-06-06):** **0 Critical** open; tier CI gate green; **23/23 FAUDIT** + follow-up Done.

**Post depth bands (2026-06-09):** MEM-DEPTH, COG-DEPTH, ECP-DEPTH, ORCH-CONFIG closeout complete — Appendix M scorecard refreshed. **IDEAL-L3 W2 (2026-06-09):** P0+P1 depth uplift — **32/32 layers L3** (see [Phase IDEAL-L3](plan/IDEAL_HARNESS_L3.md)).

**Gate evidence (verify step):** `uv run pytest -m gate -q` → **901 passed**; `check_harness_no_getattr.py`, `check_intergrax_no_applications_imports.py`, `check_harness_prompt_golden_catalog.py`, `check_agents_lifecycle_metadata.py` → **OK**.

### FAUDIT-32 — Layer scorecard (summary)

| # | Layer | Score | Crit | High | Plan accurate? |
|---|-------|-------|------|------|----------------|
| 1 | Strategic Harness Model | L3 | 0 | 0 | Yes |
| 2 | Tier Model and Dependency Boundaries | L3 | 0 | 0 | Yes |
| 3 | Interface and Task Intake | L3 | 0 | 1 | Partial |
| 4 | Identity, Trust and Tenancy | L3 | 0 | 0 | Yes |
| 5 | Policy and Governance | L3 | 0 | 2 | Partial |
| 6 | LLM and Model Adapter Layer | L3 | 0 | 1 | Yes |
| 7 | Reasoning, Planning and Cognition | L3 | 0 | 0 | Yes |
| 8 | Execution Runtime and Agent OS | L3 | 0 | 0 | Yes |
| 9 | Orchestration, Scheduler and Execution Graph | L3 | 0 | 0 | Yes |
| 10 | Subagents and Multi-Agent Coordination | L3 | 0 | 0 | Yes |
| 11 | Tool Layer | L3 | 0 | 1 | Yes |
| 12 | Skill Layer | L3 | 0 | 0 | Yes |
| 13 | Integration Layer | L3 | 0 | 0 | Yes |
| 14 | RAG and Retrieval Layer | L3 | 0 | 0 | Yes |
| 15 | Memory Layer | L3 | 0 | 0 | Yes |
| 16 | Context Engineering Layer | L3 | 0 | 0 | Yes |
| 17 | Prompt Engineering and Prompt Registry | L3 | 0 | 0 | Yes |
| 18 | Agent Assembly and Agent Contracts | L2 | 0 | 1 | Yes |
| 19 | Registry Architecture | L2 | 0 | 2 | **No** |
| 20 | Capability Graph Architecture | L3 | 0 | 0 | Yes |
| 21 | Observability and Telemetry | L3 | 0 | 0 | Yes |
| 22 | Error Handling and Reliability | L3 | 0 | 0 | Yes |
| 23 | Security and Data Governance | L3 | 0 | 0 | Yes |
| 24 | Cost and Resource Governance | L3 | 0 | 0 | Yes |
| 25 | Evaluation and Benchmarking | L2 | 0 | 1 | **No** |
| 26 | Testing, CI and Architecture Gates | L3 | 0 | 0 | Yes |
| 27 | Developer Experience, Scaffold and Lab | L3 | 0 | 1 | Yes |
| 28 | Product Environment and Tier-3 Applications | L3 | 0 | 1 | Partial |
| 29 | Modality, Vision, Audio and Dedicated ML | L3 | 0 | 1 | Yes |
| 30 | Operational Excellence and SLOs | L3 | 0 | 1 | Partial |
| 31 | Agent Lifecycle Governance | L3 | 0 | 0 | Yes |
| 32 | Architecture Governance and Documentation Loop | L3 | 0 | 1 | Yes |

**Plan accuracy note:** Rows marked **No** or **Partial** mean the phase closeout register claims **Done** for **wiring/bridge** work, but FAUDIT found **High** gaps vs `IDEAL_HARNESS_AI_ARCHITECTURE.md` / `INTEGRAX_HARNESS_AUDIT_MAP.md` §8 — tracked as **FAUDIT.\*** residuals, not reopening closed closeout phases.

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

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](#63a-business-backlog-register-consolidated)** · **[§4.0a](#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](#61z-harness-implementation-queue-consolidated) / [§6.1aa](#61aa-harness-implementation-queue-memory-platform).

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Band 2ay M.12** = **Done** — see **[§6.1an](#61an-harness-implementation-queue--llm-guardrail-integrations-closed)**. **Next product work** = [§6.3](#63-end-of-plan--deferred-product-work-only) (product prioritization only).

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
  python scripts/check_agents_no_inline_prompts.py
  python scripts/check_agents_no_vendor_sdk_imports.py
  uv run python scripts/check_ideal_harness_l3_gates.py
  uv run python scripts/harness_maturity_report.py --enforce-l3-critical
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

### 6.1av Harness implementation queue — Platform Foundation audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `PLATFORM_FOUNDATION` layers 1, 2, 32 · [`../audit_results/2026-06-19/PLATFORM_FOUNDATION.md`](../audit_results/2026-06-19/PLATFORM_FOUNDATION.md) · prior: [`../audit_results/2026-06-18/PLATFORM_FOUNDATION.md`](../audit_results/2026-06-18/PLATFORM_FOUNDATION.md)  
**Priority ladder:** **Band 1** (§6.1) — doc hygiene + optional legacy cleanup; runs **in parallel** with gate maintenance

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **PF-MAINT-DOC-01** | Docs | P2 | **Done** | Remove stale M.6 P6 from audit prompt known-gaps; sync audit result file | Audit prompt + result match plan §6.1y (**Done** 32/32) |
| 2 | **PF-MAINT-DOC-02** | Docs | P2 | **Done** | Sync §6.1au + §4.0 Band 2az counter with `AUDIT_IDEAL_2026.md` | Plan shows **90/90 Done** · **0 Planned** |
| 3 | **PF-MAINT-DX-01** | Docs | P3 | **Done** | Implementer quick-start in `intergrax_runtime_architecture.md` hub | §4.0 ladder + scaffold flow linked |
| 4 | **PF-MAINT-LEG-01** | Code | P3 | **Done** | Remove `use_rag`/`use_websearch` from LLM planner schema (`EnginePlan`) | `check_legacy_tool_plan_booleans.py` green; `tool_ids` only |
| 5 | **PF-MAINT-DOC-03** | Docs | P3 | **Done** | Sync §0.5 regression gate counter with live `pytest -m gate` snapshot | Plan §0.5 shows **1498 passed** (2026-06-19) |
| 6 | **PF-MAINT-LEG-02** | Code | P3 | **Done** | Remove legacy `use_rag`/`use_websearch` shims from `ToolInvocationPlan` (`tool_runtime.py`) | Zero DeprecationWarning in gate; `tool_ids` only at runtime bridge |
| 7 | **PF-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19/` | `PLATFORM_FOUNDATION.md` + `progress.json` present |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** Phase K, §50 marketplace, new Tier-0 mechanisms — [§6.3](#63-end-of-plan--deferred-product-work-only).

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

Full task register: [Appendix I](plan/plan/PLATFORM_FOUNDATION_appendices.md).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

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
