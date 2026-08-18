# Intergrax Runtime Architecture

> **Technical index** — complete 24-domain architecture ↔ plan register and cross-layer feature pairs. For the public project-level mental model, see [Architecture Overview](ARCHITECTURE_OVERVIEW.md). First contact: [README](../../../README.md).

**Hub only** — domain architecture and implementation are paired 1:1 under `.` and `../maintainers/plans`; multi-layer features are paired 1:1 under `../capabilities/architecture` and `../capabilities/plan`.
**Architecture principles:** [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md) — canonical rules for platform capability ownership, domain creation, application adoption, and proof order (meta-architecture governance; not a domain pair).
**Target:** [`../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Strategy:** [`../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Features:** [`../capabilities/README.md`](../capabilities/README.md) — cross-layer capability docs that coordinate multiple domain pairs without replacing domain ownership.
**Invariants:** [`../technical/guides/SYSTEM_INVARIANTS.md`](../technical/guides/SYSTEM_INVARIANTS.md) — cross-layer MUST/MUST NOT rules + `SYS-INV-*` index (P2-ARCH-01)
**Maturity:** [`../technical/guides/MATURITY_TAXONOMY.md`](../technical/guides/MATURITY_TAXONOMY.md) — four-axis A/I/P/E vocabulary; legacy L3/L4/L5 mapping (P2-ARCH-02). Maturity labels elsewhere in this hub are summaries only; authoritative production readiness claims require four-axis A/I/P/E statements in the owning architecture/plan pair.
**Layer completion:** [`../technical/guides/LAYER_COMPLETION_MODE.md`](../technical/guides/LAYER_COMPLETION_MODE.md) — deep domain layer closeout workflow
**Doc boundaries (Experimentation/DX):** [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md#architecture-vs-implementation-rules-boundary) — architecture vs Cursor/workflow rules placement (P2-ARCH-13)
**Audit:** [`../technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](../technical/guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · **Idea intake (Mode I):** [`../maintainers/bootstrap/idea_audit.txt`](../maintainers/bootstrap/idea_audit.txt) · **Cursor bootstrap:** [`../maintainers/bootstrap`](../maintainers/bootstrap) · **Domain audit prompts:** [`../maintainers/audit`](../maintainers/audit) · **Architecture audit results:** [`audit_results/`](../../audit_results/README.md) · **Implementation journal:** [`../maintainers/implementation-journal`](../maintainers/implementation-journal/README.md)
**Authoring:** [`../technical/guides`](../technical/guides)

---

## Documentation topology

```text
docs/project/architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md   — architecture governance (meta; no plan pair)
docs/project/architecture/intergrax_runtime_architecture.md                   — runtime architecture hub (this file)
docs/project/architecture/<DOMAIN>.md       ↔ docs/project/maintainers/plans/<DOMAIN>.md
docs/project/capabilities/architecture/<FEATURE>.md ↔ docs/project/capabilities/plan/<FEATURE>.md
```

**Architecture governance** ([`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](INTERGRAX_ARCHITECTURE_PRINCIPLES.md)) defines how capabilities are owned, classified, adopted, and proved. **Runtime architecture** (this hub) indexes the platform topology. **Domain architecture/plan pairs** own one reusable capability each. **Feature coordination** pairs cut across domains without replacing domain ownership.

Domain pairs own layer architecture and implementation truth. Feature pairs coordinate capabilities that cut across multiple domain pairs. Feature implementation still lands in the owning domain plan rows.

Current feature pairs:

| Feature | Architecture | Plan |
|---------|--------------|------|
| `TOKEN_OPTIMIZATION` | [`../capabilities/architecture/TOKEN_OPTIMIZATION.md`](../capabilities/architecture/TOKEN_OPTIMIZATION.md) | [`../capabilities/plan/TOKEN_OPTIMIZATION.md`](../capabilities/plan/TOKEN_OPTIMIZATION.md) |
| `LANGCHAIN_INDEPENDENCE` | [`../capabilities/architecture/LANGCHAIN_INDEPENDENCE.md`](../capabilities/architecture/LANGCHAIN_INDEPENDENCE.md) | [`../capabilities/plan/LANGCHAIN_INDEPENDENCE.md`](../capabilities/plan/LANGCHAIN_INDEPENDENCE.md) |

---

## Domain pair index (24)

| # | Domain | Architecture | Plan |
|---|--------|--------------|------|
| 1 | `PLATFORM_FOUNDATION` | [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) | [`../maintainers/plans/PLATFORM_FOUNDATION.md`](../maintainers/plans/PLATFORM_FOUNDATION.md) |
| 2 | `UNIFIED_EXECUTION_RUNTIME` | [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | [`../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`](../maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md) |
| 3 | `ORCHESTRATION` | [`ORCHESTRATION.md`](ORCHESTRATION.md) | [`../maintainers/plans/ORCHESTRATION.md`](../maintainers/plans/ORCHESTRATION.md) |
| 4 | `NEXUS_EXECUTION_FLOW` | [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | [`../maintainers/plans/NEXUS_EXECUTION_FLOW.md`](../maintainers/plans/NEXUS_EXECUTION_FLOW.md) |
| 5 | `REASONING_AND_COGNITION` | [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | [`../maintainers/plans/REASONING_AND_COGNITION.md`](../maintainers/plans/REASONING_AND_COGNITION.md) |
| 6 | `AGENT_CONTRACTS_AND_ASSEMBLY` | [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) | [`../maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| 7 | `LLM_ADAPTERS` | [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) | [`../maintainers/plans/LLM_ADAPTERS.md`](../maintainers/plans/LLM_ADAPTERS.md) |
| 8 | `TOOLS` | [`TOOLS.md`](TOOLS.md) | [`../maintainers/plans/TOOLS.md`](../maintainers/plans/TOOLS.md) |
| 9 | `CODE_CRAFT` | [`CODE_CRAFT.md`](CODE_CRAFT.md) | [`../maintainers/plans/CODE_CRAFT.md`](../maintainers/plans/CODE_CRAFT.md) |
| 10 | `SKILLS` | [`SKILLS.md`](SKILLS.md) | [`../maintainers/plans/SKILLS.md`](../maintainers/plans/SKILLS.md) |
| 11 | `INTEGRATIONS` | [`INTEGRATIONS.md`](INTEGRATIONS.md) | [`../maintainers/plans/INTEGRATIONS.md`](../maintainers/plans/INTEGRATIONS.md) |
| 12 | `RAG` | [`RAG.md`](RAG.md) | [`../maintainers/plans/RAG.md`](../maintainers/plans/RAG.md) |
| 13 | `MEMORY` | [`MEMORY.md`](MEMORY.md) | [`../maintainers/plans/MEMORY.md`](../maintainers/plans/MEMORY.md) |
| 14 | `CONTEXT_ENGINEERING` | [`CONTEXT_ENGINEERING.md`](CONTEXT_ENGINEERING.md) | [`../maintainers/plans/CONTEXT_ENGINEERING.md`](../maintainers/plans/CONTEXT_ENGINEERING.md) |
| 15 | `MODALITY` | [`MODALITY.md`](MODALITY.md) | [`../maintainers/plans/MODALITY.md`](../maintainers/plans/MODALITY.md) |
| 16 | `OBSERVABILITY` | [`OBSERVABILITY.md`](OBSERVABILITY.md) | [`../maintainers/plans/OBSERVABILITY.md`](../maintainers/plans/OBSERVABILITY.md) |
| 17 | `RELIABILITY_FAILURE_AND_HITL` | [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | [`../maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`](../maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md) |
| 18 | `CRITIC_VERIFICATION` | [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) | [`../maintainers/plans/CRITIC_VERIFICATION.md`](../maintainers/plans/CRITIC_VERIFICATION.md) |
| 19 | `ADAPTIVE_HARNESS_INTELLIGENCE` | [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) | [`../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| 20 | `ELASTIC_CAPACITY_AND_SCALING` | [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | [`../maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md`](../maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md) |
| 21 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | [`../maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| 22 | `TIER3_APPLICATION_ENVIRONMENT` | [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | [`../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`](../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md) |
| 23 | `APPLICATION_HOSTING` | [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md) | [`../maintainers/plans/APPLICATION_HOSTING.md`](../maintainers/plans/APPLICATION_HOSTING.md) |
| 24 | `UNIFIED_CONTEXT_LIFECYCLE` | [`UNIFIED_CONTEXT_LIFECYCLE.md`](UNIFIED_CONTEXT_LIFECYCLE.md) | [`../maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md`](../maintainers/plans/UNIFIED_CONTEXT_LIFECYCLE.md) — lifecycle owner for conversation context optimization; [`ADR-UCL-001`](../technical/adr/entries/2026-08-01/ADR-UCL-001.md) |

**Plan-only hubs (no 1:1 architecture basename):** [`../maintainers/plans/HARNESS_EVIDENCE_PACK.md`](../maintainers/plans/HARNESS_EVIDENCE_PACK.md) · [`../maintainers/plans/IDEAL_HARNESS_L3.md`](../maintainers/plans/IDEAL_HARNESS_L3.md) · [`../maintainers/plans/AUDIT_IDEAL_2026.md`](../maintainers/plans/AUDIT_IDEAL_2026.md) · [`../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../maintainers/plans/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md) (satellite architecture under `satellites`).

---

## Four tiers

```text
Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory · codecraft
Tier-1  intergrax/runtime/    Nexus · AgentEngine · UAEP · policy
Tier-2  agents/             domain capabilities
Tier-3  applications/       deployable hosts
```

Stack: Integration → Tool → Skill → Agent
Execution: [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md)

---

## Implementer quick start

**Default queue:** [`../maintainers/plans/PLATFORM_FOUNDATION.md`](../maintainers/plans/PLATFORM_FOUNDATION.md) **§4.0** priority ladder — Band 1 gate maintenance on every PR; Band 3 product work is **frozen** unless leadership reprioritizes (§6.3).

| Goal | Read first | Command |
|------|------------|---------|
| New agent | [`../technical/guides/AGENT_CREATION_GUIDE.md`](../technical/guides/AGENT_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-agent <name> --capability <cap>.<action>` |
| New application host | [`../technical/guides/APPLICATION_CREATION_GUIDE.md`](../technical/guides/APPLICATION_CREATION_GUIDE.md) | `python -m intergrax.scaffold new-application <name>_application` |
| Agent + app bundle | [`../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`](../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md) | `python -m intergrax.scaffold new-stack <name>` |
| Extension / plugin | [`../technical/guides/EXTENSION_AUTHOR_GUIDE.md`](../technical/guides/EXTENSION_AUTHOR_GUIDE.md) | `bootstrap_catalogs()` + entry points `intergrax.tools` / `intergrax.skills` / `intergrax.integrations` |
| Multi-layer feature | [`../capabilities/README.md`](../capabilities/README.md) | feature architecture → feature plan → affected domain pairs |
| Harness health | [`../maintainers/plans/PLATFORM_FOUNDATION.md`](../maintainers/plans/PLATFORM_FOUNDATION.md) §6.1 | `uv run intergrax doctor --ci` · `uv run pytest -m gate -q` |

**Work cycle:** strategy → architecture pair or feature pair → smallest domain-owned plan item → implement → gate green → update paired docs + journal if significant.

---

## Agent in the harness environment

**Hub summary for architects, researchers, and AI crawlers** — full canon in [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 · plan [Phase ACP](../maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md).

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

**Author entry points:** [`../technical/guides/AGENT_CREATION_GUIDE.md`](../technical/guides/AGENT_CREATION_GUIDE.md) Appendix AC · roster [`agents/README.md`](../../../agents/README.md).

**Implementation:** architecture **decision-complete**; code delivery [ACP waves](../maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md#61aw-acp-detailed-implementation-waves) (typed contracts → step loop → fleet migration Wave 8 → prod gates → **ACP-CLOSE-LEG-5** pipeline retirement). Product agents control the loop via **`on_next_step`** only; Tier-1 `RuntimeEngine` pipeline stack removed ([ADR-FLOW-005](../technical/adr/entries/2026-06-12/ADR-FLOW-005.md)).

---

## Application in the harness environment

**Hub summary** — full canon in [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) §24–§51 (APP-CON / APP-EVOL / APP-OPS) · **freeze audit:** [`../technical/guides/GOVERNANCE_CONSISTENCY_AUDIT.md`](../technical/guides/GOVERNANCE_CONSISTENCY_AUDIT.md) · plan [H-APP-CON](../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-con--application-environment-architecture-canon-app-con) · [H-APP-FREEZE](../maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md#phase-h-app-freeze--cross-document-governance-consistency-audit).

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
| What harness slices are enabled? | **`ApplicationEnvironmentProfile`** (§22.1 flat · §22.6 bundles) | §22 · [ADR-APP-003](../technical/adr/entries/2026-06-17/ADR-APP-003.md) |
| Reactive vs daemon vs batch? | Posture + host factory | §23 |
| Always-on deployment lifecycle? | **`APPLICATION_HOSTING`** (wraps Tier-3 factory) | [`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md) |
| Who sets routing capability? | L1–L4 matrix | §23.3 |
| Virtual org / simulation rules? | **`OrganizationalPolicyEnvelope`** | §39 |
| Dynamic block at intake / selection? | **`ApplicationHost`** + `HookPoint` | §32 |
| Multi-agent orchestration summary? | **`ApplicationRunSummary`** | §26 · §33 |

**Strategic invariants (APP-CON §28.1):**

- **Applications compose; they do not cognate** — business logic stays in Tier-2 agents.
- **One Task lifecycle** — all surfaces converge on `UnifiedTaskRunner` → `NexusLoop`.
- **Tier-3 defines the application** — manifest, profile, surfaces, and Task/Nexus integration.
- **Application Hosting provides deployment lifecycle models** around that application — process lifecycle, readiness, instance ownership, signals, graceful shutdown, restart supervision, and OS adapters ([`APPLICATION_HOSTING.md`](APPLICATION_HOSTING.md)). Deployment posture does not alter Task semantics, Nexus execution, agent behavior, or product results.

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
LKW.6 closed narrowly as: unified interaction intake; Application Hosting adoption; first Windows PowerShell product interaction adapter; live reviewer proof. Product docs: [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

### LKW Hybrid Knowledge Workspace — active product roadmap

Canonical execution order: [applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md). Architecture: [KNOWLEDGE_ACCESS_ARCHITECTURE.md](../KNOWLEDGE_ACCESS_ARCHITECTURE.md).

| Block | Scope | Status |
|-------|-------|--------|
| **1B-5-2** | End-to-end WEB_URL Knowledge Intake | **ACCEPTED** (including C1 and C2) |
| **LKW-KNOWLEDGE-ACCESS-ARCHITECTURE-1** | Hybrid knowledge access vocabulary and roadmap | **ACCEPTED** |
| **LKW-MODEL-RUNTIME-1** | Ollama / vLLM end-to-end portability | **ACCEPTED** |
| **LKW-KNOWLEDGE-ACCESS-1** | Connections, Indexed Sources, Live Access Bindings | **NEXT** |
| **LKW-CONVERSATION-CONTEXT-ARCH-1** | Provider-neutral conversation context with observed-audience validation, binding identity, thread memory isolation and deterministic guards | **ACCEPTED** |
| **LKW-HYBRID-ASK-1** | RAG + live with unified provenance | **PLANNED** |
| **LKW-CONVERSATIONAL-FRONTEND-1** | Natural-language execution + Slack cutover | **PLANNED** |
| **LKW-VENDOR-ACCESS-COLLABORATION-1** | MS365, Google Workspace, Jira, Confluence | **PLANNED** |
| **LKW-VENDOR-ACCESS-DATA-1** | Databricks, Power BI, Atlan | **PLANNED** |
| **LKW-KNOWLEDGE-LIFECYCLE-1** | Sync, freshness, permissions, removal | **PLANNED** |
| **LKW-LIVE-PLATFORM-PROOF-1** | Complete demonstrable platform proof (Slack, Google when implemented, MS365, local files, Web URLs) | **PLANNED** |
### Vendor Knowledge platform backlog

Canonical plan: [plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md](../maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md). Architecture: [architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md](KNOWLEDGE_SOURCE_INTEGRATIONS.md).

| Task | Scope | Status |
|------|-------|--------|
| **SLACK-KNOWLEDGE-THREE-MODE-ARCH-1** | Freeze Slack three-mode reuse on SlackConversationChannelIntegration | **DONE** |
| **SLACK-KNOWLEDGE-FOUNDATION-1** | Platform Slack knowledge read surface (bot token + bot-membership inventory on same WebClient), Vendor Knowledge adapter, durable sync proof | **DONE** |
| **SLACK-LIVE-CAPABILITY-1** | Bounded Slack live reads via same integration | **PLANNED** |
| **GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1** | Freeze Google Workspace knowledge architecture on GoogleWorkspaceCollaborationSuiteIntegration | **READY_FOR_REVIEW** |
| **GOOGLE-WORKSPACE-KNOWLEDGE-FOUNDATION-1** | Shared Google client, credentials, error mapping (after complete Slack vertical) | **PLANNED** |
| **LKW-GOOGLE-WORKSPACE-CONNECTED-SOURCE-1** | LKW Connected Source for Google Workspace | **PLANNED** |
| **LKW-GOOGLE-WORKSPACE-PROOF-1** | First user-oriented Google Doc/Sheet/Calendar/Drive proof | **PLANNED** |
| **MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR** | Microsoft Graph Calendar Vendor Knowledge adapter | **PLANNED** (after first accepted Google LKW proof) |

LKW application tasks (`LKW-SLACK-CONNECTED-SOURCE-1` **IN_PROGRESS / CHANGES_REQUIRED**; `LKW-CONVERSATION-CONTEXT-1` **NEXT**; `LKW-CONVERSATION-CONTEXT-ARCH-1` **ACCEPTED**; `LKW-SLACK-SHARED-CONVERSATION-ADAPTER-1`, `LKW-SLACK-KNOWLEDGE-PROOF-1` **PLANNED**; final proof joins `LKW-HYBRID-ASK-1`; Google Workspace runtime starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED**) are tracked in the [LKW Implementation Plan](../IMPLEMENTATION_PLAN.md) and [Conversation Context Architecture](../CONVERSATION_CONTEXT_ARCHITECTURE.md), not as platform adapter tasks.


Former 1B-6 / 1C–1E slices are **mapped into** the blocks above; see Implementation Plan §3.4.
