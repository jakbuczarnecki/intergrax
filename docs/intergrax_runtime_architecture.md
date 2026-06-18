# Intergrax Runtime Architecture

**Hub only** — domain architecture and implementation are paired 1:1 under `architecture/` and `plan/`.
**Target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Invariants:** [`guides/SYSTEM_INVARIANTS.md`](guides/SYSTEM_INVARIANTS.md) — cross-domain “never violate” index (P2-ARCH-01)
**Layer completion:** [`guides/LAYER_COMPLETION_MODE.md`](guides/LAYER_COMPLETION_MODE.md) — deep domain layer closeout workflow
**Audit:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) · **Domain audit prompts:** [`guides/audit/`](guides/audit/) · **Implementation journal:** [`guides/implementation-journal/`](guides/implementation-journal/README.md)
**Authoring:** [`guides/`](guides/)

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
| Who sets routing capability? | L1–L4 matrix | §23.3 |
| Virtual org / simulation rules? | **`OrganizationalPolicyEnvelope`** | §39 |
| Dynamic block at intake / selection? | **`ApplicationHost`** + `HookPoint` | §32 |
| Multi-agent orchestration summary? | **`ApplicationRunSummary`** | §26 · §33 |

**Strategic invariants (APP-CON §28.1):**

- **Applications compose; they do not cognate** — business logic stays in Tier-2 agents.
- **One Task lifecycle** — all surfaces converge on `UnifiedTaskRunner` → `NexusLoop`.
- **Profile is the composition root** — no ad-hoc `getattr` wiring in hosts; nested bundles (§22.6) group slices under the same root.
- **Hooks are boundaries, not step loops** — no `Application.on_next_orchestration_step()`.

**Author entry points:** [`applications/USAGE.md`](../applications/USAGE.md) · `HarnessApplication` (`intergrax/harness/app.py`) · scaffold `new-application` · [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md) §0.

**Implementation:** H-APP profile/wiring **Done**; APP-CON-1 host pipeline mount **Done**; budget reactions **Done** (ACP-TOK-1..3 · ACP-TOK-CI) + APP-PROD-1..9 gates; APP-EVOL-1..7 evolution **Done**; **APP-EVOL-8** hierarchical bundles **M1–M3 Done** ([ADR-APP-003](adr/entries/2026-06-17/ADR-APP-003.md)); APP-OPS-1..4 platform ops **Done** — [TIER3 plan](plan/TIER3_APPLICATION_ENVIRONMENT.md#master-implementation-backlog-app-unified). **Maturity:** Architecturally Mature for reference hosts; enterprise marketplace/distribution **P4**.

**Observability spine evolution:** **OBS-EVOL-9** layered `event_kind` catalog **Done** (2026-06-17; OBS-EVOL-9.9 deferred post-publication) — [ADR-OBS-003](adr/entries/2026-06-17/ADR-OBS-003.md) · [OBS plan](plan/OBSERVABILITY.md#phase-obs-evol-9--layered-event-catalog-p1-arch-02).

---

## Domain documents (architecture ↔ implementation 1:1)

| Architecture | Implementation plan |
|--------------|---------------------|
| [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) |
| [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) | [`plan/UNIFIED_EXECUTION_RUNTIME.md`](plan/UNIFIED_EXECUTION_RUNTIME.md) |
| [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) | [`plan/ORCHESTRATION.md`](plan/ORCHESTRATION.md) |
| [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) | [`plan/NEXUS_EXECUTION_FLOW.md`](plan/NEXUS_EXECUTION_FLOW.md) |
| [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) | [`plan/INTEGRATIONS.md`](plan/INTEGRATIONS.md) |
| [`architecture/RAG.md`](architecture/RAG.md) | [`plan/RAG.md`](plan/RAG.md) |
| [`architecture/TOOLS.md`](architecture/TOOLS.md) | [`plan/TOOLS.md`](plan/TOOLS.md) |
| [`architecture/CODE_CRAFT.md`](architecture/CODE_CRAFT.md) | [`plan/CODE_CRAFT.md`](plan/CODE_CRAFT.md) |
| [`architecture/SKILLS.md`](architecture/SKILLS.md) | [`plan/SKILLS.md`](plan/SKILLS.md) |
| [`architecture/LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md) | [`plan/LLM_ADAPTERS.md`](plan/LLM_ADAPTERS.md) |
| [`architecture/MEMORY.md`](architecture/MEMORY.md) | [`plan/MEMORY.md`](plan/MEMORY.md) |
| [`architecture/CONTEXT_ENGINEERING.md`](architecture/CONTEXT_ENGINEERING.md) | [`plan/CONTEXT_ENGINEERING.md`](plan/CONTEXT_ENGINEERING.md) |
| [`architecture/MODALITY.md`](architecture/MODALITY.md) | [`plan/MODALITY.md`](plan/MODALITY.md) |
| [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | [`plan/OBSERVABILITY.md`](plan/OBSERVABILITY.md) |
| [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) | [`plan/RELIABILITY_FAILURE_AND_HITL.md`](plan/RELIABILITY_FAILURE_AND_HITL.md) |
| [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) |
| [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) | [`plan/CRITIC_VERIFICATION.md`](plan/CRITIC_VERIFICATION.md) |
| [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) | [`plan/REASONING_AND_COGNITION.md`](plan/REASONING_AND_COGNITION.md) |
| [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) | [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](plan/ELASTIC_CAPACITY_AND_SCALING.md) |

---

## Reading order

1. This hub → [Agent in the harness environment](#agent-in-the-harness-environment) (above)
2. [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) + [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) — **agent model & ACP**
3. [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) + [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md)
4. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) + matching plan
5. [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) + matching plan
6. Your other domain pair from the table below
7. [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when building agents

**Per-iteration rule:** pick one domain — read only its architecture + plan pair; do not load unrelated domains.

**Audit map routing (32 layers → domain pair):**

| Layers | Domain pair |
|--------|-------------|
| 1–2 Strategic + tiers | `PLATFORM_FOUNDATION` |
| 3 Intake | `ORCHESTRATION` + `TIER3_APPLICATION_ENVIRONMENT` |
| 4 Identity / trust | `UNIFIED_EXECUTION_RUNTIME` §42.44 |
| 5 Policy | `UNIFIED_EXECUTION_RUNTIME` §42.11 |
| 6 LLM | `LLM_ADAPTERS` |
| 7 Reasoning / planning / cognition | `REASONING_AND_COGNITION` |
| 8 Execution runtime / Agent OS | `UNIFIED_EXECUTION_RUNTIME` + `NEXUS_EXECUTION_FLOW` (narrative) |
| 9 Orchestration / graph / scheduler | `ORCHESTRATION` + `NEXUS_EXECUTION_FLOW` |
| 10 Subagents | `NEXUS_EXECUTION_FLOW` §27 |
| 11 Tools (catalog) | `TOOLS` |
| 11b Ephemeral Code Craft | `CODE_CRAFT` |
| 12–13 Skills / integrations | `SKILLS` · `INTEGRATIONS` |
| 14 RAG | `RAG` (+ `MEMORY` for Knowledge vs LTM boundary) |
| 15 Memory | `MEMORY` |
| 16 Context engineering | `CONTEXT_ENGINEERING` |
| 17–20 Prompt / assembly / registry / capability graph | `AGENT_CONTRACTS_AND_ASSEMBLY` |
| 21 Observability | `OBSERVABILITY` |
| 22 Reliability / HITL | `RELIABILITY_FAILURE_AND_HITL` + UAEP §42.10 |
| 23–24 Security / cost | `UNIFIED_EXECUTION_RUNTIME` §42.45–47 |
| 25–27 Eval / CI / DX | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` + `CRITIC_VERIFICATION` |
| 28 Tier-3 hosts | `TIER3_APPLICATION_ENVIRONMENT` |
| 29 Modality | `MODALITY` |
| 30 Ops / SLO / elastic capacity | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` §43 + `OBSERVABILITY` + `ELASTIC_CAPACITY_AND_SCALING` |
| 31 Agent lifecycle | `AGENT_CONTRACTS_AND_ASSEMBLY` §20 |
| 32 Doc governance loop | `PLATFORM_FOUNDATION` + `guides/` |

Full audit procedure: [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md). **Domain audit prompts:** [`guides/audit/`](guides/audit/). **Completed implementation episodes:** [`guides/implementation-journal/INDEX.md`](guides/implementation-journal/INDEX.md).

---

## Platform runtime capabilities (cross-domain index)

Essential platform behaviours span multiple domain pairs — use this index before opening unrelated docs.

| Capability | Primary architecture | Plan phase |
|------------|---------------------|------------|
| **Agent session loop** (`run`, `on_next_step`, `HarnessKernel`, typed state) | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §13–§40 | **ACP** (active) |
| **Agent production readiness scoreboard** | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) §40.15 | ACP-PROD-12 |
| **Fleet migration** (roster → typed runtime) | [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](plan/AGENT_CONTRACTS_AND_ASSEMBLY.md) Wave 8 | ACP-MIG |
| Resilience policies (retry, reboot, circuit breaker) | [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) §34 | REL-ADV |
| Orchestration strategies (parallel, sequence, cooperation, scale, redundancy) | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §50–§53, §58 | ORCH-5, ORCH-6 |
| MVP → product evolution (eval, KPI, simulation, promotion) | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §44 | MVP-EVOL |
| Autonomy slider (manual / ask / autonomous) | REL §35 + [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.10.2 | REL-ADV |
| Sync / async execution postures | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §57 | ORCH-6 |
| Interrupt anywhere / resume from checkpoint | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §28 + UAEP §42.8–§42.9 | FLOW-CTL |
| Guardrails / policy enforcement (catalog) | [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.6 · §42.37 · vendor backends [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) §47 | GR-DOC · M.12 |
| RAG / retrieval engine | [`architecture/RAG.md`](architecture/RAG.md) · integration slugs [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) | M-RAG · M-RAG-DEPTH |
| Context engineering engine | [`architecture/CONTEXT_ENGINEERING.md`](architecture/CONTEXT_ENGINEERING.md) | CE-EXT Done · CE-ALIGN Done · **CE-PROV-WIRE Planned** |
| Ephemeral Code Craft (dynamic codegen loop) | [`architecture/CODE_CRAFT.md`](architecture/CODE_CRAFT.md) · substrate [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) | ECC-0…ECC-6 |
| LLM adapters (typed envelope + ModelCatalog) | [`architecture/LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md) · [`intergrax/llm_adapters/USAGE.md`](../intergrax/llm_adapters/USAGE.md) | M-LLM-R **Done** · **M-LLM-X** (active) |

Platform docs do not replace `agents/*/ARCHITECTURE.md` or `applications/*/ARCHITECTURE.md`.

### Platform execution audit (2026-06-09, synced)

**Verdict:** Tier-1 Nexus supports all documented launch/interaction scenarios (FLOW §3.1 S1–S7). Harness closeouts **Done**: ORCH-CONFIG, ORCH-5, H-APP-WIRING, MEM/COG/ECP-DEPTH, reference host CFG presets. **Remaining:** product-only items (§6.3) — FLOW-8 Tier-3 demo host, CFG-14 LKW daemon, GOV-PROD.1 dashboard.

| Topic | Canonical register | Status |
|-------|-------------------|--------|
| CFG-* configuration cases | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §56.7, §59.3 | **18/19 harness Done**; CFG-14 product **Deferred** §6.3 |
| Host surface parity | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §59.2 · [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) §23.7 | Reference hosts **Done**; LKW opt-in flags |
| FLOW runtime gaps | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §23.2 | FLOW-GAP-01–19 **Closed**; FLOW-GAP-20 **Deferred** §6.3 |
| Depth bands | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) §4.0 | MEM/COG/ECP/ORCH-CONFIG **Done** |
| Default queue | [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md) §6.1 | Gate maintenance + [Phase AUDIT-IDEAL](plan/AUDIT_IDEAL_2026.md) incremental |
| Ideal L3 depth (Band 2ax) | [`plan/IDEAL_HARNESS_L3.md`](plan/IDEAL_HARNESS_L3.md) · §6.1at | **W2 Done** (2026-06-09) — **32/32 L3** |
| Ideal architecture gaps (Band 2az) | [`plan/AUDIT_IDEAL_2026.md`](plan/AUDIT_IDEAL_2026.md) · §6.1au | **W1 in progress** (2026-06-09) — **15/78 Done** · **4 Deferred §6.3** |

---

## ADRs (harness — selected)

| ADR | Topic |
|-----|-------|
| [`adr/entries/2026-06-07/ADR-FLOW-001.md`](adr/entries/2026-06-07/ADR-FLOW-001.md) | Declarative delegation (`DELEGATES_TO`) |
| [`adr/entries/2026-06-07/ADR-FLOW-002.md`](adr/entries/2026-06-07/ADR-FLOW-002.md) | Reserved lifecycle states |
| [`adr/entries/2026-06-07/ADR-FLOW-003.md`](adr/entries/2026-06-07/ADR-FLOW-003.md) | `MODIFY_PLAN` semantics |
| [`adr/entries/2026-06-09/ADR-FLOW-004.md`](adr/entries/2026-06-09/ADR-FLOW-004.md) | Graph spec seed guard (`trigger_capabilities`) |
| [`adr/entries/2026-06-10/ADR-CODECRAFT-001.md`](adr/entries/2026-06-10/ADR-CODECRAFT-001.md) | Ephemeral Code Craft as separate Harness domain |
| [`adr/entries/2026-06-11/ADR-AGENT-001.md`](adr/entries/2026-06-11/ADR-AGENT-001.md) | Agent cognitive patterns (ACP) — Tier-2 library, Nexus stays Agent OS |
| [`adr/entries/2026-06-11/ADR-AGENT-002.md`](adr/entries/2026-06-11/ADR-AGENT-002.md) | Author `Agent.run()` facade + per-agent environment binding |
| [`adr/entries/2026-06-06/ADR-LLM-001.md`](adr/entries/2026-06-06/ADR-LLM-001.md) | Typed LLM adapter response envelope |
| [`adr/entries/2026-06-14/ADR-LLM-002.md`](adr/entries/2026-06-14/ADR-LLM-002.md) | ModelCatalog + context window resolution |
