# Intergrax Runtime Architecture

**Hub only** — all domain canon: [`architecture/`](architecture/)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](IDEAL_HARNESS_AI_ARCHITECTURE.md)  
**Implementation:** [`INTERGRAX_IMPLEMENTATION_PLAN.md`](INTERGRAX_IMPLEMENTATION_PLAN.md) · [`plan/phases/`](plan/phases/)  
**Authoring:** [`guides/`](guides/)

---

## Four tiers

```text
Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory
Tier-1  intergrax/runtime/  Nexus · AgentEngine · UAEP · policy
Tier-2  agents/             domain capabilities
Tier-3  applications/       deployable hosts
```

Stack: Integration → Tool → Skill → Agent  
Execution: [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)

---

## Architecture documents (complete list)

| Document | Topic |
|----------|-------|
| [`PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) | Tiers, principles, anti-patterns |
| [`UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) | UAEP, policy engine, events |
| [`ORCHESTRATION.md`](architecture/ORCHESTRATION.md) | Nexus, graphs, reasoning/planning |
| [`NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) | Flow diagrams, subagents, delegation |
| [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) | Agents, registry, capability graph |
| [`INTEGRATIONS.md`](architecture/INTEGRATIONS.md) | Integrations + RAG backends catalog |
| [`TOOLS.md`](architecture/TOOLS.md) | ToolRuntime + catalog |
| [`SKILLS.md`](architecture/SKILLS.md) | Skills |
| [`LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md) | LLM adapters |
| [`MEMORY.md`](architecture/MEMORY.md) | Memory + context engineering |
| [`MODALITY.md`](architecture/MODALITY.md) | Vision, audio, ML |
| [`OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | Observability spine |
| [`RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) | Failure, retry, HITL |
| [`TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) | Tier-3 hosts |
| [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | Lab, DX, evaluation, gates |
| [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | L4 AHI |
| [`CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) | CVL / PEV |

---

## Reading order

1. This hub → [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md)  
2. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)  
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md)  
4. Your domain from the table above  
5. [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when building agents  

Platform docs do not replace `agents/*/ARCHITECTURE.md` or `applications/*/ARCHITECTURE.md`.
