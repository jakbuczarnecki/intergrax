# Intergrax Runtime Architecture

**Hub only** — domain architecture and implementation are paired 1:1 under `architecture/` and `plan/`; multi-layer features are paired 1:1 under `features/architecture/` and `features/plan/`.
**Architecture principles:** [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) — canonical rules for platform capability ownership, domain creation, application adoption, and proof order (meta-architecture governance; not a domain pair).
**Target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Features:** [`features/README.md`](features/README.md) — cross-layer capability docs that coordinate multiple domain pairs without replacing domain ownership.
**Invariants:** [`guides/SYSTEM_INVARIANTS.md`](guides/SYSTEM_INVARIANTS.md) — cross-layer MUST/MUST NOT rules + `SYS-INV-*` index (P2-ARCH-01)
**Maturity:** [`guides/MATURITY_TAXONOMY.md`](guides/MATURITY_TAXONOMY.md) — four-axis A/I/P/E vocabulary; legacy L3/L4/L5 mapping (P2-ARCH-02). Maturity labels elsewhere in this hub are summaries only; authoritative production readiness claims require four-axis A/I/P/E statements in the owning architecture/plan pair.
**Layer completion:** [`guides/LAYER_COMPLETION_MODE.md`](guides/LAYER_COMPLETION_MODE.md) — deep domain layer closeout workflow
**Doc boundaries (Experimentation/DX):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#architecture-vs-implementation-rules-boundary) — architecture vs Cursor/workflow rules placement (P2-ARCH-13)
**Audit:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · **Idea intake (Mode I):** [`bootstrap/idea_audit.txt`](bootstrap/idea_audit.txt) · **Cursor bootstrap:** [`bootstrap/`](bootstrap/) · **Domain audit prompts:** [`audit/`](audit/) · **Architecture audit results:** [`audit_results/`](audit_results/README.md) · **Implementation journal:** [`implementation-journal/`](implementation-journal/README.md)
**Authoring:** [`guides/`](guides/)

---

## Documentation topology

```text
docs/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md   — architecture governance (meta; no plan pair)
docs/intergrax_runtime_architecture.md                   — runtime architecture hub (this file)
docs/architecture/<DOMAIN>.md       ↔ docs/plan/<DOMAIN>.md
docs/features/architecture/<FEATURE>.md ↔ docs/features/plan/<FEATURE>.md
```

**Architecture governance** ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)) defines how capabilities are owned, classified, adopted, and proved. **Runtime architecture** (this hub) indexes the platform topology. **Domain architecture/plan pairs** own one reusable capability each. **Feature coordination** pairs cut across domains without replacing domain ownership.

Domain pairs own layer architecture and implementation truth. Feature pairs coordinate capabilities that cut across multiple domain pairs. Feature implementation still lands in the owning domain plan rows.

Current feature pairs:

| Feature | Architecture | Plan |
|---------|--------------|------|
| `TOKEN_OPTIMIZATION` | [`features/architecture/TOKEN_OPTIMIZATION.md`](features/architecture/TOKEN_OPTIMIZATION.md) | [`features/plan/TOKEN_OPTIMIZATION.md`](features/plan/TOKEN_OPTIMIZATION.md) |
| `LANGCHAIN_INDEPENDENCE` | [`features/architecture/LANGCHAIN_INDEPENDENCE.md`](features/architecture/LANGCHAIN_INDEPENDENCE.md) | [`features/plan/LANGCHAIN_INDEPENDENCE.md`](features/plan/LANGCHAIN_INDEPENDENCE.md) |

---

## Domain pair index (24)

| # | Domain | Architecture | Plan |
|---|--------|--------------|------|
| 1 | `PLATFORM_FOUNDATION` | [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) |
| 2 | `UNIFIED_EXECUTION_RUNTIME` | [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](plan/UNIFIED_EXECUTION_RUNTIME.md) |
| 3 | `ORCHESTRATION` | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) | [`plan/ORCHESTRATION.md`](plan/ORCHESTRATION.md) |
| 4 | `NEXUS_EXECUTION_FLOW` | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) | [`plan/NEXUS_EXECUTION_FLOW.md`](plan/NEXUS_EXECUTION_FLOW.md) |
| 5 | `REASONING_AND_COGNITION` | [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) | [`plan/REASONING_AND_COGNITION.md`](plan/REASONING_AND_COGNITION.md) |
| 6 | `AGENT_CONTRACTS_AND_ASSEMBLY` | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| 7 | `LLM_ADAPTERS` | [`architecture/LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md) | [`plan/LLM_ADAPTERS.md`](plan/LLM_ADAPTERS.md) |
| 8 | `TOOLS` | [`architecture/TOOLS.md`](architecture/TOOLS.md) | [`plan/TOOLS.md`](plan/TOOLS.md) |
| 9 | `CODE_CRAFT` | [`architecture/CODE_CRAFT.md`](architecture/CODE_CRAFT.md) | [`plan/CODE_CRAFT.md`](plan/CODE_CRAFT.md) |
| 10 | `SKILLS` | [`architecture/SKILLS.md`](architecture/SKILLS.md) | [`plan/SKILLS.md`](plan/SKILLS.md) |
| 11 | `INTEGRATIONS` | [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) | [`plan/INTEGRATIONS.md`](plan/INTEGRATIONS.md) |
| 12 | `RAG` | [`architecture/RAG.md`](architecture/RAG.md) | [`plan/RAG.md`](plan/RAG.md) |
| 13 | `MEMORY` | [`architecture/MEMORY.md`](architecture/MEMORY.md) | [`plan/MEMORY.md`](plan/MEMORY.md) |
| 14 | `CONTEXT_ENGINEERING` | [`architecture/CONTEXT_ENGINEERING.md`](architecture/CONTEXT_ENGINEERING.md) | [`plan/CONTEXT_ENGINEERING.md`](plan/CONTEXT_ENGINEERING.md) |
| 15 | `MODALITY` | [`architecture/MODALITY.md`](architecture/MODALITY.md) | [`plan/MODALITY.md`](plan/MODALITY.md) |
| 16 | `OBSERVABILITY` | [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | [`plan/OBSERVABILITY.md`](plan/OBSERVABILITY.md) |
| 17 | `RELIABILITY_FAILURE_AND_HITL` | [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) | [`plan/RELIABILITY_FAILURE_AND_HITL.md`](plan/RELIABILITY_FAILURE_AND_HITL.md) |
| 18 | `CRITIC_VERIFICATION` | [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) | [`plan/CRITIC_VERIFICATION.md`](plan/CRITIC_VERIFICATION.md) |
| 19 | `ADAPTIVE_HARNESS_INTELLIGENCE` | [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| 20 | `ELASTIC_CAPACITY_AND_SCALING` | [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) | [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](plan/ELASTIC_CAPACITY_AND_SCALING.md) |
| 21 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| 22 | `TIER3_APPLICATION_ENVIRONMENT` | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) |
| 23 | `APPLICATION_HOSTING` | [`architecture/APPLICATION_HOSTING.md`](architecture/APPLICATION_HOSTING.md) | [`plan/APPLICATION_HOSTING.md`](plan/APPLICATION_HOSTING.md) |
| 24 | `UNIFIED_CONTEXT_LIFECYCLE` | [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](architecture/UNIFIED_CONTEXT_LIFECYCLE.md) | [`plan/UNIFIED_CONTEXT_LIFECYCLE.md`](plan/UNIFIED_CONTEXT_LIFECYCLE.md) |

**Plan-only hubs (no 1:1 architecture basename):** [`plan/HARNESS_EVIDENCE_PACK.md`](plan/HARNESS_EVIDENCE_PACK.md) · [`plan/IDEAL_HARNESS_L3.md`](plan/IDEAL_HARNESS_L3.md) · [`plan/AUDIT_IDEAL_2026.md`](plan/AUDIT_IDEAL_2026.md) · [`plan/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](plan/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) (satellite architecture under `architecture/satellites/`).

---

## Four tiers

```text
Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory · codecraft
Tier-1  intergrax/runtime/    Nexus · AgentEngine · UAEP · policy
Tier-2  agents/             domain capabilities
Tier-3  applications/       deployable hosts
```

Stack: Integration → Tool → Skill → Agent
Execution: [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)

---

## Implementer quick start

**Default queue:** [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) **§4.0** priority ladder — Band 1 gate maintenance on every PR; Band 3 product work is **frozen** unless leadership reprioritizes (§6.3).

| Goal | Read first | Command |
|------|------------|---------|
| New agent | [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-agent <name> --capability <cap>.<action>` |
| New application host | [`guides/APPLICATION_CREATION_GUIDE.md`](guides/APPLICATION_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-application <name>_application` |
| Agent + app bundle | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) | `python -m intergrax.scaffold new-stack <name>` |
| Extension / plugin | [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md) | `bootstrap_catalogs()` + entry points `intergrax.tools` / `intergrax.skills` / `intergrax.integrations` |
| Multi-layer feature | [`features/README.md`](features/README.md) | feature architecture → feature plan → affected domain pairs |
| Harness health | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) §6.1 | `uv run intergrax doctor --ci` · `uv run pytest -m gate -q` |

**Work cycle:** strategy → architecture pair or feature pair → smallest domain-owned plan item → implement → gate green → update paired docs + journal if significant.

---

## Agent in the harness environment

**Hub summary for architects, researchers, and AI crawlers** — full canon in [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 · plan [Phase ACP](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md).

Intergrax is **not** “one Python class that is also the OS.” The **agent** is a **domain decision unit** inside a **typed, governed environment**. Responsibility is split by design:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application + NexusLoop.handle_task()                                 │
│     Environment: profiles, AgentBinding, RequestIdentity, org envelope  │
│     Orchestration: Task graph, capability routing, HITL, Plane A log    │
│     DOES NOT: plan inside one agent's cognitive loop                      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node → one Agent.run() per role
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L3  Agent.run() — session decision loop (many steps, one user-facing run) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ each iteration
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L2  Agent.on_next_step() — author domain hook                             │
│     READ typed state · UPDATE state_delta · DECIDE StepOutcome §32.0      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ StepOutcome
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L1  HarnessKernel.execute_step() — deterministic harness primitive        │
│     policy · gateways · trace · budgets · state merge · checkpoint hook   │
│     DOES NOT: domain replan · choose next graph agent                     │
└─────────────────────────────────────────────────────────────────────────┘
```

| Question | Owner | Canon |
|----------|-------|-------|
| Who acts (tenant, user, org agent)? | Application intake → `RequestIdentity` | §30.9 |
| Which agents run on this Task? | **NexusLoop** + capability registry | §37.6 |
| What is the next domain move? | **`on_next_step`** → `StepOutcome` | §32 · §32.0 |
| Is policy/trace/state safe? | **`HarnessKernel`** | §38 |
| Lab vs prod same agent code? | `merge_environment` + `AgentBinding` | §30 |
| Can this agent ship to production? | Production Readiness Scoreboard | §40.15 · ACP-PROD-12 |

**Strategic invariants (ADR-AGENT-001..003):**

- **Nexus is not the agent** — it orchestrates; it does not replace `on_next_step`.
- **HarnessKernel does not plan** — it executes one harness cycle per step.
- **AgentRuntime.advance_step is glue only** — `on_next_step` then kernel; no policy logic in runtime.
- **Agents are replaceable; the harness is the product.**

**Author entry points:** [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) Appendix AC · roster [`agents/README.md`](../agents/README.md).

**Implementation:** architecture **decision-complete**; code delivery [ACP waves](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md#61aw-acp-detailed-implementation-waves) (typed contracts → step loop → fleet migration Wave 8 → prod gates → **ACP-CLOSE-LEG-5** pipeline retirement). Product agents control the loop via **`on_next_step`** only; Tier-1 `RuntimeEngine` pipeline stack removed ([ADR-FLOW-005](adr/entries/2026-06-12/ADR-FLOW-005.md)).

---

## Application in the harness environment

**Hub summary** — full canon in [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) §24–§51 (APP-CON / APP-EVOL / APP-OPS) · **freeze audit:** [`guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](guides/GOVERNANCE_CONSISTENCY_AUDIT.md) · plan [H-APP-CON](plan/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-con--application-environment-architecture-canon-app-con) · [H-APP-FREEZE](plan/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-freeze--cross-document-governance-consistency-audit).

The **application** is a **deployable composition shell** — not a cognitive agent. It normalizes intake → `Task`, declares roster and harness profiles, and returns product output. Tier-3 authors control environment through **three modes** (§30): declarative profile, rules envelope, imperative `ApplicationHost` hooks.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ L4  Application host (Tier-3)                                           │
│     ApplicationManifest · ApplicationEnvironmentProfile · surfaces      │
│     ApplicationHost.on_hook (optional) · ApplicationRunSummary (Plane A) │
│     DOES NOT: on_next_step · domain tool loops · private Nexus fork     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ UnifiedTaskRunner.run_task()
┌───────────────────────────────▼─────────────────────────────────────────┐
│ L3  NexusLoop.handle_task() — Agent OS (Tier-1)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ graph node → Agent.run() — see Agent section above
```

| Question | Owner | Canon |
|----------|-------|-------|
| What agents are active in this product? | **`ApplicationManifest`** roster | §24 · §27 |
| What harness slices are enabled? | **`ApplicationEnvironmentProfile`** (§22.1 flat · §22.6 bundles) | §22 · [ADR-APP-003](adr/entries/2026-06-17/ADR-APP-003.md) |
| Reactive vs daemon vs batch? | Posture + host factory | §23 |
| Always-on deployment lifecycle? | **`APPLICATION_HOSTING`** (wraps Tier-3 factory) | [`APPLICATION_HOSTING.md`](architecture/APPLICATION_HOSTING.md) |
| Who sets routing capability? | L1–L4 matrix | §23.3 |
| Virtual org / simulation rules? | **`OrganizationalPolicyEnvelope`** | §39 |
| Dynamic block at intake / selection? | **`ApplicationHost`** + `HookPoint` | §32 |
| Multi-agent orchestration summary? | **`ApplicationRunSummary`** | §26 · §33 |

**Strategic invariants (APP-CON §28.1):**

- **Applications compose; they do not cognate** — business logic stays in Tier-2 agents.
- **One Task lifecycle** — all surfaces converge on `UnifiedTaskRunner` → `NexusLoop`.
- **Tier-3 defines the application** — manifest, profile, surfaces, and Task/Nexus integration.
- **Application Hosting provides deployment lifecycle models** around that application — process lifecycle, readiness, instance ownership, signals, graceful shutdown, restart supervision, and OS adapters ([`APPLICATION_HOSTING.md`](architecture/APPLICATION_HOSTING.md)). Deployment posture does not alter Task semantics, Nexus execution, agent behavior, or product results.

### Product proof — Local Knowledge Workspace (LKW.6)

| ID | Scope | Status |
|----|-------|--------|
| **LKW.6** | Unified interaction intake, Application Hosting adoption, first Windows PowerShell product interaction adapter, live reviewer proof | **Closed** |
| **LKW.6C** | Windows PowerShell product interaction adapter + live reviewer proof | **Closed** |
| **LKW.6b** | Slack Socket Mode (optional) | Planned / optional |
| **LKW.7** | File watcher + incremental index | **Closed** (LKW.7B Closed; LKW.7C Closed; LKW.7C1 Done; LKW.7C2 Done) |
| **LKW.7A** | Incremental file-change contract and idempotent batches | **Done** |
| **LKW.7B1** | Runtime state machine, bounded debounce and existing enqueue boundary | **Done** |
| **LKW.7B2A** | Durable checkpoint and restart recovery | **Done** |
| **LKW.7B2B** | Sidecar settings, process loop, signals and automatic checkpoint lifecycle | **Done** |
| **LKW.7C** | Persistent incremental-index live proof | **Closed** (LKW.7C1 Done; LKW.7C2 Done) |
| **LKW.7C1** | Watcher-triggered persistent search E2E workload | **Done** |
| **LKW.7C2** | ProofReceipt, reviewer runner and final closeout | **Done** |


LKW.7 is **Closed**: LKW.7A Done; LKW.7B1 Done; LKW.7B2A Done; LKW.7B2B Done; LKW.7B Closed; LKW.7C Closed; LKW.7C1 Done; LKW.7C2 Done.
LKW.6 closed narrowly as: unified interaction intake; Application Hosting adoption; first Windows PowerShell product interaction adapter; live reviewer proof. Product docs: [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md).

### LKW Hybrid Knowledge Workspace — active product roadmap

Canonical execution order: [applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md](../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md). Architecture: [KNOWLEDGE_ACCESS_ARCHITECTURE.md](../applications/local_workspace_application/docs/KNOWLEDGE_ACCESS_ARCHITECTURE.md).

| Block | Scope | Status |
|-------|-------|--------|
| **1B-5-2** | End-to-end WEB_URL Knowledge Intake | **ACCEPTED** (including C1 and C2) |
| **LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1** | Hybrid knowledge access vocabulary and roadmap | **ACCEPTED** |
| **LKW-MODEL-RUNTIME-1** | Ollama / vLLM end-to-end portability | **ACCEPTED** |
| **LKW-KNOWLEDGE-ACCESS-1** | Connections, Indexed Sources, Live Access Bindings | **NEXT** |
| **LKW-CONVERSATION-CONTEXT-ARCH-1** | Provider-neutral conversation context with observed-audience validation, binding identity, thread memory isolation and deterministic guards | **READY_FOR_REVIEW** |
| **LKW-HYBRID-ASK-1** | RAG + live with unified provenance | **PLANNED** |
| **LKW-CONVERSATIONAL-FRONTEND-1** | Natural-language execution + Slack cutover | **PLANNED** |
| **LKW-VENDOR-ACCESS-COLLABORATION-1** | MS365, Jira, Confluence | **PLANNED** |
| **LKW-VENDOR-ACCESS-DATA-1** | Databricks, Power BI, Atlan | **PLANNED** |
| **LKW-KNOWLEDGE-LIFECYCLE-1** | Sync, freshness, permissions, removal | **PLANNED** |
| **LKW-LIVE-PLATFORM-PROOF-1** | Complete demonstrable Slack platform proof | **PLANNED** |
### Vendor Knowledge platform backlog

Canonical plan: [plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md](plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md). Architecture: [architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md](architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md).

| Task | Scope | Status |
|------|-------|--------|
| **SLACK-KNOWLEDGE-THREE-MODE-ARCH-1** | Freeze Slack three-mode reuse on SlackConversationChannelIntegration | **DONE** |
| **SLACK-KNOWLEDGE-FOUNDATION-1** | Platform Slack knowledge read surface (bot token + bot-membership inventory on same WebClient), Vendor Knowledge adapter, durable sync proof | **DONE** |
| **SLACK-LIVE-CAPABILITY-1** | Bounded Slack live reads via same integration | **PLANNED** |
| **MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR** | Microsoft Graph Calendar Vendor Knowledge adapter | **PLANNED** (after complete Slack user vertical) |

LKW application tasks (`LKW-SLACK-CONNECTED-SOURCE-1` **DONE**; `LKW-CONVERSATION-CONTEXT-1` **NEXT**; LKW-CONVERSATION-CONTEXT-ARCH-1, LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1, LKW-SLACK-KNOWLEDGE-PROOF-1; final proof joins LKW-HYBRID-ASK-1) are tracked in the [LKW Implementation Plan](../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) and [Conversation Context Architecture](../applications/local_workspace_application/docs/CONVERSATION_CONTEXT_ARCHITECTURE.md), not as platform adapter tasks.


Former 1B-6 / 1C–1E slices are **mapped into** the blocks above; see Implementation Plan §3.4.
