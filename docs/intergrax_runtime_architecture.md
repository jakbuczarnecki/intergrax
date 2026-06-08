# Intergrax Runtime Architecture

**Status:** Canonical architecture hub (platform Harness / Agent OS)  
**Target reference:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Implementation status:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md)  
**Strategy:** [`INTERGRAX_DEVELOPMENT_STRATEGY.md`](INTERGRAX_DEVELOPMENT_STRATEGY.md)  
**Audit methodology:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md)

Audience: Humans, LLMs, coding agents, maintainers  
Purpose: Single entry point for the **current** Intergrax Harness AI architecture. Detailed contracts live in decomposed domain documents under [`architecture/`](architecture/).

**Monolith backup (pre-decomposition):** [`_archive_intergrax_runtime_architecture_monolith.md`](_archive_intergrax_runtime_architecture_monolith.md)

---

## Documentation model

| Layer | Document | Role |
|-------|----------|------|
| **Target** | [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md) | North-star Harness AI benchmark |
| **Current canon (hub)** | **This file** | Navigation, concepts, tier model, reading order |
| **Current canon (detail)** | [`architecture/`](architecture/) + specialized docs below | As-built contracts per domain |
| **Implementation** | [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) + [`plan/phases/`](plan/phases/) | Phase status, queues, evidence |
| **Authoring** | [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) | Create agents without Nexus edits |
| **Audit** | [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md) | 32-layer maturity audit |

### Documentation boundary (platform vs product)

**In scope:** Tier-0 platform, Tier-1 Nexus, reference wiring patterns, infrastructure to run agent environments.

**Out of scope:** Each `applications/<product>/` and `agents/<name>/` owns its own `ARCHITECTURE.md` and local plan.

---

## Executive summary

Intergrax is a **four-tier Harness AI / Agent OS**:

```text
Tier-0  intergrax/          Platform — integrations, tools, skills, LLM, RAG, memory
Tier-1  intergrax/runtime/   Nexus — orchestration, policy, graphs, HITL
Tier-2  agents/             Specialized capability modules
Tier-3  applications/       Deployable product environments
```

**Core asset:** the runtime that lets teams create, run, observe, and validate agents quickly — not any single agent.

**Operating modes (same codebase):**

| Mode | Goal |
|------|------|
| **Laboratory** | Fast hypothesis validation — idea → traced run in under one hour |
| **Production harness** | Governed, observable, policy-enforced agent execution |

**Capability stack:** Integration → Tool → Skill → Agent  
**Execution contract:** Unified Agent Execution Protocol (UAEP) — [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)

**Critical rule:** Reuse existing Tier-0 mechanisms. Do not duplicate universal platform components without explicit approval. See [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md).

---

## Four-tier model (summary)

| Tier | Location | Owns |
|------|----------|------|
| **Tier-0** | `intergrax/` | Integrations, tools, skills, LLM adapters, RAG, memory, modality |
| **Tier-1** | `intergrax/runtime/nexus/` | Intake, planning, graphs, policy, HITL, `AgentEngine`, `ToolRuntime` gateway |
| **Tier-2** | `agents/<name>/` | Domain pipelines, prompts, local validation — **no** vendor SDKs |
| **Tier-3** | `applications/<name>/` | Host wiring, manifests, deployment — **no** agent business logic |

**Dependency direction (strict):**

```text
applications/  →  agents/  →  intergrax/
intergrax/  MUST NOT import from agents/ or applications/
```

Full tier contracts: [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md)

---

## High-level execution stack

```text
Task intake (API / CLI / worker)
        ↓
   NexusLoop (Tier-1) — plan · schedule · policy · graph · HITL
        ↓
   AgentEngine (Tier-1) — UAEP · middleware · steps · validation
        ↓
   Domain Agent (Tier-2) — pipeline · prompts · domain validation
        ↓
   ToolRuntime → Tier-0 tools / integrations / skills
```

**Flow narrative and diagrams:** [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](NEXUS_EXECUTION_FLOW_REFERENCE.md)

---

## Architecture domain index

### Platform foundation

| Topic | Document |
|-------|----------|
| Tiers, principles, reuse, anti-patterns, naming | [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) |
| Task intake, `TaskEnvelope` | [`architecture/INTERFACE_AND_INTAKE.md`](architecture/INTERFACE_AND_INTAKE.md) |
| Tier-3 hosts, sandbox, shadow workspace | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) |
| Experimentation workflow, DX rules | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| CI gates, architecture boundary tests | [`architecture/TESTING_CI_AND_ARCHITECTURE_GATES.md`](architecture/TESTING_CI_AND_ARCHITECTURE_GATES.md) |

### Runtime core (Tier-1)

| Topic | Document |
|-------|----------|
| **UAEP, events, hooks, lifecycle, policy runtime** | [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) |
| Nexus, dual loop, graphs, task lifecycle | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) |
| Planning, cognition, decision records | [`architecture/REASONING_AND_PLANNING.md`](architecture/REASONING_AND_PLANNING.md) |
| Subagents, delegation, merge policies | [`architecture/SUBAGENTS_AND_COORDINATION.md`](architecture/SUBAGENTS_AND_COORDINATION.md) |
| Execution flow (narrative) | [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](NEXUS_EXECUTION_FLOW_REFERENCE.md) |

### Agents and registries

| Topic | Document |
|-------|----------|
| Agent contract, registry, capabilities | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| Registry snapshots, conformance | [`architecture/REGISTRY_ARCHITECTURE.md`](architecture/REGISTRY_ARCHITECTURE.md) |
| Capability graph, blast radius | [`architecture/CAPABILITY_GRAPH.md`](architecture/CAPABILITY_GRAPH.md) |

### Tier-0 capability stack

| Topic | Document |
|-------|----------|
| Integrations and adapters | [`INTEGRATIONS.md`](INTEGRATIONS.md) · [`architecture/INTEGRATIONS_ARCHITECTURE.md`](architecture/INTEGRATIONS_ARCHITECTURE.md) |
| Tools and ToolRuntime | [`TOOLS.md`](TOOLS.md) · [`architecture/TOOLS_RUNTIME.md`](architecture/TOOLS_RUNTIME.md) |
| Skills (composable packs) | [`SKILLS.md`](SKILLS.md) |
| LLM adapters and profiles | [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) |
| RAG and retrieval | [`architecture/RAG_AND_RETRIEVAL.md`](architecture/RAG_AND_RETRIEVAL.md) |
| Memory (STM/LTM/org/task) | [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) |
| Context engineering | [`architecture/CONTEXT_ENGINEERING.md`](architecture/CONTEXT_ENGINEERING.md) |
| Prompt registry | [`architecture/PROMPT_REGISTRY.md`](architecture/PROMPT_REGISTRY.md) |
| Modality plane (vision, audio, ML) | [`MODALITY.md`](MODALITY.md) |
| Plugin / extension authoring | [`EXTENSION_AUTHOR_GUIDE.md`](EXTENSION_AUTHOR_GUIDE.md) |

### Governance, security, operations

| Topic | Document |
|-------|----------|
| Policy and governance | [`architecture/POLICY_AND_GOVERNANCE.md`](architecture/POLICY_AND_GOVERNANCE.md) |
| Identity, trust, tenancy | [`architecture/IDENTITY_TRUST_AND_TENANCY.md`](architecture/IDENTITY_TRUST_AND_TENANCY.md) |
| Reliability, failure, retry, HITL | [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) |
| Security and data governance | [`architecture/SECURITY_AND_DATA_GOVERNANCE.md`](architecture/SECURITY_AND_DATA_GOVERNANCE.md) |
| Cost and resource governance | [`architecture/COST_AND_RESOURCE_GOVERNANCE.md`](architecture/COST_AND_RESOURCE_GOVERNANCE.md) |
| Evaluation and benchmarking | [`architecture/EVALUATION_AND_BENCHMARKING.md`](architecture/EVALUATION_AND_BENCHMARKING.md) |
| Observability and tracing | [`OBSERVABILITY_ARCHITECTURE.md`](OBSERVABILITY_ARCHITECTURE.md) |

### Advanced harness mechanisms

| Topic | Document |
|-------|----------|
| Adaptive Harness Intelligence (L4) | [`ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md`](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) |
| Critic & Verification Layer (PEV) | [`CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md`](CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md) |

---

## Design principles (canonical)

1. **Runtime first** — Nexus owns global reasoning; agents own local domain execution.
2. **Experimentation first** — fast path from idea to traced run.
3. **Integrations are adapters** — agents use `ToolRuntime`, not vendor SDKs.
4. **Observability is mandatory** — no important step without trace.
5. **Reuse Tier-0** — extend and wire; do not duplicate platform mechanisms.
6. **Policy-first** — nothing executes without appropriate policy checks.

Details and forbidden patterns: [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) (§8, §43)

---

## Harness AI alignment (conceptual)

Harness AI terms and mapping to Intergrax tiers are defined in [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) (§5.3). **Do not** duplicate that glossary elsewhere.

```text
Harness → Runtime (Nexus + Tier-0) → Agents → Applications → Products
```

---

## Reading order for implementers

1. This hub — tier model and index  
2. [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) — boundaries and reuse rules  
3. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) — UAEP contracts  
4. [`NEXUS_EXECUTION_FLOW_REFERENCE.md`](NEXUS_EXECUTION_FLOW_REFERENCE.md) — runtime path  
5. Domain doc for your task (tools, memory, orchestration, …)  
6. [`AGENT_CREATION_GUIDE.md`](AGENT_CREATION_GUIDE.md) — agent authoring  
7. [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) — what is Done vs Planned  

---

## Audit layer map (quick reference)

Full 32-layer audit: [`INTEGRAX_HARNESS_AUDIT_MAP.md`](INTEGRAX_HARNESS_AUDIT_MAP.md)

| Audit layer | Architecture document |
|-------------|----------------------|
| 1–2 Strategic / tiers | `PLATFORM_FOUNDATION` |
| 3 Interface | `INTERFACE_AND_INTAKE` |
| 4 Identity | `IDENTITY_TRUST_AND_TENANCY` |
| 5 Policy | `POLICY_AND_GOVERNANCE` |
| 6 LLM | `LLM_ADAPTERS` |
| 7 Reasoning | `REASONING_AND_PLANNING` |
| 8 Execution runtime | `UNIFIED_EXECUTION_RUNTIME` |
| 9 Orchestration | `ORCHESTRATION` |
| 10 Subagents | `SUBAGENTS_AND_COORDINATION` |
| 11–13 Tools / skills / integrations | `TOOLS`, `SKILLS`, `INTEGRATIONS` |
| 14 RAG | `RAG_AND_RETRIEVAL` |
| 15 Memory | `MEMORY_ARCHITECTURE` |
| 16 Context | `CONTEXT_ENGINEERING` |
| 17 Prompts | `PROMPT_REGISTRY` |
| 18–20 Agent / registry / graph | `AGENT_CONTRACTS`, `REGISTRY`, `CAPABILITY_GRAPH` |
| 21 Observability | `OBSERVABILITY_ARCHITECTURE` |
| 22 Reliability | `RELIABILITY_FAILURE_AND_HITL` |
| 23–25 Security / cost / eval | respective `architecture/*` docs |
| 26 Testing / CI | `TESTING_CI_AND_ARCHITECTURE_GATES` |
| 27 DX | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` |
| 28 Tier-3 | `TIER3_APPLICATION_ENVIRONMENT` |
| 29 Modality | `MODALITY` |
| 30–32 Ops / lifecycle / doc loop | plan + `PLATFORM_FOUNDATION` |

---

## Final canonical statement

Intergrax is an **Agent Operating System**. The Harness is the product. Agents are replaceable. All execution flows through Nexus, `AgentEngine`, and UAEP. All Tier-0 access flows through policy and `ToolRuntime`.

When in doubt, prefer the contracts in [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) and the boundaries in [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md).

---

## Legacy section index (redirects)

Former monolith section numbers map to decomposed documents:

| Former § | Document |
|----------|----------|
| §1–§8, §43–§51, §53, §5.3 | [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) |
| §9–§10, §23–§26, §47 | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) |
| §11–§16, §45 | [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) |
| §17–§18, §46 | [`architecture/INTEGRATIONS_ARCHITECTURE.md`](architecture/INTEGRATIONS_ARCHITECTURE.md) + [`INTEGRATIONS.md`](INTEGRATIONS.md) |
| §19–§21 | [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) |
| §22 | [`architecture/TOOLS_RUNTIME.md`](architecture/TOOLS_RUNTIME.md) + [`TOOLS.md`](TOOLS.md) |
| §27 | [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) |
| §28 | [`architecture/CONTEXT_ENGINEERING.md`](architecture/CONTEXT_ENGINEERING.md) |
| §29–§32 | [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) |
| §33 | [`OBSERVABILITY_ARCHITECTURE.md`](OBSERVABILITY_ARCHITECTURE.md) |
| §34 | [`architecture/EVALUATION_AND_BENCHMARKING.md`](architecture/EVALUATION_AND_BENCHMARKING.md) |
| §35, §39–§41 | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| §42 | [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) |
| §54 | [`ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md`](ADAPTIVE_HARNESS_INTELLIGENCE_ARCHITECTURE.md) |
| §55 | [`CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md`](CRITIC_VERIFICATION_LAYER_ARCHITECTURE.md) |
| §7.1.6–§7.1.8 | [`TOOLS.md`](TOOLS.md) · [`SKILLS.md`](SKILLS.md) |
| §7.1.9 | [`MODALITY.md`](MODALITY.md) |
