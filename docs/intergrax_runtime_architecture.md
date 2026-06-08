# Intergrax Runtime Architecture

**Hub only** — domain architecture and implementation are paired 1:1 under `architecture/` and `plan/`.
**Target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Audit:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md)
**Authoring:** [`guides/`](guides/)

---

## Four tiers

```text
Tier-0  intergrax/          integrations · tools · skills · LLM · RAG · memory
Tier-1  intergrax/runtime/    Nexus · AgentEngine · UAEP · policy
Tier-2  agents/             domain capabilities
Tier-3  applications/       deployable hosts
```

Stack: Integration → Tool → Skill → Agent
Execution: [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md)

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
| [`architecture/TOOLS.md`](architecture/TOOLS.md) | [`plan/TOOLS.md`](plan/TOOLS.md) |
| [`architecture/SKILLS.md`](architecture/SKILLS.md) | [`plan/SKILLS.md`](plan/SKILLS.md) |
| [`architecture/LLM_ADAPTERS.md`](architecture/LLM_ADAPTERS.md) | [`plan/LLM_ADAPTERS.md`](plan/LLM_ADAPTERS.md) |
| [`architecture/MEMORY.md`](architecture/MEMORY.md) | [`plan/MEMORY.md`](plan/MEMORY.md) |
| [`architecture/MODALITY.md`](architecture/MODALITY.md) | [`plan/MODALITY.md`](plan/MODALITY.md) |
| [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) | [`plan/OBSERVABILITY.md`](plan/OBSERVABILITY.md) |
| [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) | [`plan/RELIABILITY_FAILURE_AND_HITL.md`](plan/RELIABILITY_FAILURE_AND_HITL.md) |
| [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](architecture/TIER3_APPLICATION_ENVIRONMENT.md) | [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](plan/TIER3_APPLICATION_ENVIRONMENT.md) |
| [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) |
| [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) | [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](plan/ADAPTIVE_HARNESS_INTELLIGENCE.md) |
| [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) | [`plan/CRITIC_VERIFICATION.md`](plan/CRITIC_VERIFICATION.md) |

---

## Reading order

1. This hub → [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) + [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md)
2. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) + matching plan
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) + matching plan
4. Your domain pair from the table above
5. [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when building agents

**Per-iteration rule:** pick one domain — read only its architecture + plan pair; do not load unrelated domains.

Platform docs do not replace `agents/*/ARCHITECTURE.md` or `applications/*/ARCHITECTURE.md`.
