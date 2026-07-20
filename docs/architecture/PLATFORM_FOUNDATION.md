# Platform Foundation

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Audit layers:** 1–2, 32  
**Audit instruction:** [`audit/PLATFORM_FOUNDATION.md`](../audit/PLATFORM_FOUNDATION.md)  
**Architecture governance:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) — platform evolution rules; Platform Foundation owns implementation gates and spine verification, not capability-ownership policy.
---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (PLATFORM_FOUNDATION canon).

- **Implement / audit default:** §1–§6 platform spine. Extended §7+: [`satellites/PLATFORM_FOUNDATION_extended_depth.md`](satellites/PLATFORM_FOUNDATION_extended_depth.md). §43+: [`satellites/PLATFORM_FOUNDATION_production_gates.md`](satellites/PLATFORM_FOUNDATION_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/PLATFORM_FOUNDATION.md`](../guides/audit_slices/PLATFORM_FOUNDATION.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/PLATFORM_FOUNDATION_extended_depth.md`](satellites/PLATFORM_FOUNDATION_extended_depth.md) | extended depth |
| [`satellites/PLATFORM_FOUNDATION_production_gates.md`](satellites/PLATFORM_FOUNDATION_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.
## Tier-1 — Nexus Runtime (Agent Operating System)

**Role:** the **Agent OS** — orchestrates agents the way an operating system orchestrates applications.

Nexus uses Tier-0 components to create a **controlled execution environment** for agents: lifecycle, routing, policy, shared services, and observability.

**Includes:**

- Global Nexus loop (`NexusLoop`, task intake, classification, planning); implementation split under `runtime/nexus/orchestration/` (`intake_runner`, `planning_runner`, `graph_runner`, `hitl_runner`, `task_events`, `lifecycle_bridge`, …) — loop file orchestrates only
- Agent registry and capability routing (`AgentRegistry`, `AgentRouter`)
- Task lifecycle and state machine (`Task`, `TaskLifecycle`)
- Execution graph and multi-agent coordination (`ExecutionGraph`, `GraphExecutor`)
- Context management and memory coordination policy
- Tool runtime gateway and adapter access policy
- Validation, retry, failure handling, human-in-the-loop gates
- Contracts: `AgentContract`, `AgentExecutionResult`, `ValidationResult`
- Trace system integration at task level
- `AgentEngine` bridge (Nexus → agent local loop)

**Analogy:** Nexus is to agents what an OS kernel + scheduler is to applications. Agents **run inside** Nexus; they do not replace it.

**Rules:**

- Tier-1 MUST remain **domain-agnostic** (no Legal logic, no UX logic inside Nexus).
- Tier-1 MUST NOT implement concrete agent business workflows.
- Tier-1 owns **global** orchestration; agents own **local** bounded execution.

**Repository:** `intergrax/runtime/`, `intergrax/contracts/`, `intergrax/agents/` (framework ABC only — **not** concrete agents).

---

## Tier-2 — Agents (Specialized Capability Modules)

**Role:** fully functional, domain-specialized modules that perform **concrete business or technical work**.

Each agent is a bounded capability: researcher, UX designer, PM, tester, marketer, legal reviewer, vendor discovery, etc.

**Includes per agent (`agents/<name>/`):**

- Agent class implementing `Agent` + `AgentContract`
- Declared capabilities (e.g. `research.web_search`, `legal.contract_review`)
- Local processing loop: pipeline, steps, prompts, domain models
- Agent-local governance and validation rules
- Agent-local tracing helpers
- Optional local tool bridge (via Tier-1 `ToolRuntime`, not raw Tier-0 bypass)

**Rules:**

- Agents MUST implement shared contracts (`get_contract()`, `can_handle()`, `build_context()`, `validate()`).
- Agents MAY have **bounded local loops** (multi-step domain execution).
- Agents MUST NOT own global orchestration, global routing, or HTTP host wiring.
- Agents consume Tier-0 **through** Tier-1 policies (not uncontrolled direct access in production).
- Agents MUST be runnable via Nexus without starting an HTTP server.

**Repository:** `agents/` at repository root (`agents/legal/`, `agents/research/`, `agents/echo/`, …).

---

## Tier-3 — Applications (Ready-Made Environments)

**Role:** **isolated, configured environments** that compose Nexus + a selected set of agents + rules + integrations for a specific context.

An application is not an agent. It is the **product shell** — the “Cursor AI for X” pattern: a ready environment for a defined industry, company type, or use case.

**Includes per application (`applications/<name>/`):**

- Host entrypoint (`main.py`, `factory.py`, `settings.py`, `wiring.py`)
- HTTP/CLI serving layer (routes, auth, tenant config)
- **Self-contained operational configuration** — own `.env` and `.env.example` (application-prefixed variables; see §7.4.8)
- Environment profiles (dev/staging/prod), SKU rules, feature flags
- Agent registry wiring: which agents are registered, with which IDs and policies
- `IntegrationProfile` composition — which Tier-0 backends this environment uses
- Orchestration config: default capabilities, routing hints, multi-agent topologies
- **Deployment package** — `docker/` (Dockerfile, optional `docker-compose.yml`) sufficient to build an image and push to production (see §7.4.8)

**Self-sufficiency rule:** A Tier-3 application is a **runnable, deployable environment** on its own. A developer MUST be able to start the host and build a container using **only** files under `applications/<name>/` plus the monorepo Python dependencies (`pyproject.toml` / `uv` at repository root). Application-specific secrets and toggles MUST NOT live only in the repository-root `.env.example`.

**Example environments:**

- `legal_application` — legal review for law firms (Legal agent + compliance rules)
- `research_application` — research → summarize pipeline for analysts
- `intergrax_assistant_application` — harness-native conversational lab (hub agent + swappable LLM + optional specialist delegation) — see §7.4.11
- `local_workspace_application` — Local Knowledge Workspace (LKW)
- `dispute_sim_application` — Dispute Simulation Workspace (DSW)
- Future: `agency_application`, `saas_pm_application`, `ecommerce_ux_application`

**Rules:**

- Applications MUST NOT contain agent domain logic (pipeline steps, prompts).
- Applications compose Tier-2 agents; they do not reimplement them.
- Multiple applications MAY reuse the same Tier-2 agent with different config.
- Applications are the only layer that binds **product-specific** env vars and deployment.

**Repository:** `applications/` at repository root.

---

## Tier Mapping Summary

| Tier | Name | Role | Analogy | Repository |
|------|------|------|---------|------------|
| **0** | Platform | Universal components & adapters | Drivers, network stack, libc | `intergrax/` (non-orchestration packages) |
| **1** | Nexus Runtime | Agent OS — orchestration & policy | Operating system | `intergrax/runtime/`, `intergrax/contracts/`, `intergrax/agents/` (ABC) |
| **2** | Agents | Specialized capability modules | Applications / programs | `agents/<name>/` |
| **3** | Applications | Configured environments | IDE product, industry workspace | `applications/<name>/` |

## Dependency Direction (Strict)

```text
Tier-3 Applications  →  Tier-2 Agents  →  Tier-1 Nexus  →  Tier-0 Platform
```

- Higher tiers import lower tiers only.
- Tier-0 MUST NOT import agents or applications.
- Tier-1 MUST NOT import concrete agents or applications.
- Tier-2 MUST NOT import applications.

**Cross-layer invariants (canonical):** [`guides/SYSTEM_INVARIANTS.md`](../guides/SYSTEM_INVARIANTS.md#cross-layer-system-invariants) (P2-ARCH-01) — MUST/MUST NOT rules across all tiers; `SYS-INV-*` index links to this §5 and domain pairs.

**Enforcement (FAUDIT-TIER, 2026-06-06 · extended 2026-06-27):** Lower layers (`intergrax/agents/`, `intergrax/runtime/`, `intergrax/contracts/`, `agents/`, …) MUST NOT import `intergrax.applications` or `applications`. Tier-3 manifest metadata for harness capability-graph seeding lives in `intergrax/applications/reference/harness_manifest_catalog.py`; runtime uses neutral `ApplicationCapabilityCatalogEntry` (`intergrax/contracts/capability_graph_catalog.py`) via `intergrax/runtime/architecture/harness_capability_catalog.py`. Application hosts map Tier-3 profiles/bindings to neutral contracts in `intergrax/applications/_shared/runtime_boundary_adapters.py`.

CI guards (no grandfather exceptions):

- `scripts/check_no_upward_application_imports.py` — full lower-layer scan
- `scripts/maintenance/check_intergrax_no_applications_imports.py`
- `scripts/maintenance/check_agents_no_tier3_imports.py`

## Relationship To “Layer 1 / 2 / 3” Naming

Earlier sections and diagrams may refer to **Layer 1 / 2 / 3**. Mapping:

| Legacy layer name | Canonical tier |
|-------------------|----------------|
| Layer 1 — Components / Adapters | **Tier-0** Platform |
| Layer 2 — Nexus Runtime | **Tier-1** Nexus (Agent OS) |
| Layer 3 — Agents | **Tier-2** Agents |
| *(not in old 3-layer model)* | **Tier-3** Applications |

**Always prefer Tier-0..3 terminology** in new code and documentation.

## Code Labels (`DeploymentTier`)

The package `intergrax.agent_kit.tiers` exposes `DeploymentTier` enum labels aligned with this model:

- `PLATFORM` (0) → Tier-0
- `FRAMEWORK` (1) → Tier-1
- `AGENT` (2) → Tier-2
- `APPLICATION` (3) → Tier-3

(`PRODUCT` is a deprecated alias for `AGENT` in legacy metadata.)

---

## 5.3 Harness AI Alignment (Conceptual Model)

Intergrax is a **Harness AI environment** (Agent OS). Industry harness literature uses vocabulary that maps to Intergrax as follows.

### 5.3.0 Terminology (Harness vs Application vs Agent)

| Term | Tier | Role |
|------|------|------|
| **Platform / Tier-0** | 0 | Catalogs: integrations, tools, skills, LLM adapters, modality inference |
| **Nexus / Runtime** | 1 | Orchestration loop, policy engine, trace, context, graph execution |
| **Agent** | 2 | Autonomous business logic: UAEP steps, `AgentContract`, prompts (`agents/`) |
| **Application** | 3 | Deployable **environment**: manifest, `ApplicationEnvironmentProfile`, host wiring (`applications/`) |
| **Harness (practical)** | 1+3+0 | Nexus + application wiring + platform catalogs — not a single Python package |
| **Product** | — | Business offering composed of Tier-3 app + selected Tier-2 agents |

IDEAL chain: `Harness → Runtime → Agents → Applications → Products`. Intergrax **Application** = Tier-3 host (IDEAL “environment”), not the Tier-2 agent module.

### 5.3.1 Core mapping

| Harness AI term | Intergrax implementation |
|-----------------|---------------------------|
| **Scaffold** | `python -m intergrax.scaffold` (`new-agent`, `new-application`, `new-stack`, `new-skill`) |
| **Harness** | Tier-1 **Nexus** + Tier-0 platform + Tier-3 **Application** wiring (policy, tools, integrations, trace) |
| **LLM** | Tier-0 `intergrax/llm_adapters/` — invoked per step/plan; not embedded inside Tier-2 agent class |
| **Agent** | Tier-2 module (`agents/<name>/`) implementing `Agent` + `AgentContract` + UAEP |
| **Runnable agent instance** | Harness + selected agent + `LLMProfile` + resolved `skill_ids` / `allowed_tools` + `RuntimePolicyBundle` for one run |
| **Tool** | Tier-0 atomic `ToolContract` — LLM/MCP/FastAPI invocable (§7.1.6) |
| **Skill** | Tier-0 composable **`SkillManifest`** — tools + prompts + policy fragment (§7.1.8) |
| **Context engineering** | Tier-1 `ContextManager` + `TaskContextAssemblyOptions` + `MemoryView` + `ContextBudgetPolicy` (§28.1) |
| **Subagent** | **Graph delegation** — Nexus `ExecutionGraph` child node, not nested OS (§42.14.3) |
| **Policy** | `PolicyEngine`, `ToolAccessPolicy`, budgets, HITL, org profiles — composed as `RuntimePolicyBundle` (§42.11.4); meaningful external side effects via `evaluate_meaningful_side_effect` / `MeaningfulSideEffectRequest` (GEC-5 · ADR-POLICY-SIDE-EFFECT-001) |
| **Guardrails** | Cross-cutting enforcement vector of Policy & Governance — not a separate tier. Hook-time checks (prompt, tool, output, cost, time) mapped in UAEP §42.11.6; optional vendor engines via Integration category `llm_guardrail` ([`INTEGRATIONS.md`](INTEGRATIONS.md) §47) |
| **Modality / ML** | Planes B+C via **tools** + optional **`ModalityProfile`** (§7.1.9); generative vision/audio via **`LLMProfile`** (Plane A); never vendor SDKs in agents |

### 5.3.2 Agent composition (not harness + LLM only)

```text
Harness (Nexus + app wiring)
    → runs Tier-2 Agent
        → composes SkillManifest(s)  →  resolves tool_ids, prompts, policy fragments
        → AgentEngine / UAEP steps
        → ToolRuntime.invoke(tool_id)  →  Integration adapters
        → LLM adapters (per step / planner)
        → Modality tools (vision.detect, speech.*, ml.predict)  →  Plane C registry / speech_provider
```

Agents MUST NOT call integrations directly. Agents MUST NOT import CV/ML SDKs (`ultralytics`, `torch`, `onnxruntime`, …) when a catalog tool or adapter exists (§7.1.9). Skills MUST NOT replace `ToolRuntime` or appear as fake `ToolContract` entries.

### 5.3.3 Architectural decision: Skill layer (ADR)

| Option | Description | Verdict |
|--------|-------------|---------|
| **1 — Skills = tools** | Encode instructions + multi-tool workflows as oversized tools | **Rejected** — breaks atomic LLM function schema, MCP export, risk/idempotency per operation, and external tool ecosystems |
| **2 — Skill Library** | Fourth layer: Integration → Tool → **Skill** → Agent | **Adopted** — **MVP Done**; importers for external formats (e.g. Cursor `SKILL.md`) after manifest validation |

Implementation tracker: [`plan/PLATFORM_FOUNDATION.md) Appendix E · catalog [`architecture/SKILLS.md`](architecture/SKILLS.md).

---


---

# 6. High Level Architecture

Intergrax consists of **four platform tiers** (see §5.1). The diagram below shows Tier-0 through Tier-3.

```text
+--------------------------------------------------------------+
|                      TIER-3 — APPLICATIONS                   |
|              Ready-made configured environments              |
|--------------------------------------------------------------|
| legal_application          research_application              |
| agency_workspace           saas_pm_environment   (future)  |
|  • host + serving + env config + agent registry wiring       |
|  • industry rules, roles, interaction topology               |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                        TIER-2 — AGENTS                       |
|           Specialized capability modules (domain)            |
|--------------------------------------------------------------|
| LegalAgent    ResearchAgent    UXAgent       PMAgent         |
| TesterAgent   MarketerAgent    VendorDiscoveryAgent (future K) ... |
|  • contracts, pipelines, steps, local loops                  |
|  • business logic; runs inside Nexus                         |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                   TIER-1 — NEXUS RUNTIME                     |
|                    Agent Operating System                    |
|--------------------------------------------------------------|
| NexusLoop          AgentRegistry       TaskLifecycle         |
| ExecutionGraph     AgentRouter         ContextManager        |
| ValidationEngine   RetryEngine         ToolRuntime           |
| AgentEngine        Trace coordination  Human approval        |
|  • global orchestration; domain-agnostic                     |
+--------------------------------------------------------------+

+--------------------------------------------------------------+
|                    TIER-0 — PLATFORM                         |
|              Universal components & adapters                 |
|--------------------------------------------------------------|
| LLM Providers    Memory / History    RAG / Vector Store      |
| PostgreSQL       Redis               Queue / Kafka           |
| Web Search       File Storage        Logging / Errors        |
| Slack / Teams    Browser             Sandbox executor        |
|  • no orchestration; no agent business logic                 |
+--------------------------------------------------------------+
```

**Execution flow:**

```text
User / API (Tier-3)
    → Nexus intake (Tier-1)
    → select & run agents (Tier-2)
    → agents call platform services (Tier-0) under Nexus policy
    → Nexus validates, traces, composes response
    → Application returns result (Tier-3)
```

---


---
