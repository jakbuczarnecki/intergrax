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
| [`architecture/RAG.md`](architecture/RAG.md) | [`plan/RAG.md`](plan/RAG.md) |
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
| [`architecture/REASONING_AND_COGNITION.md`](architecture/REASONING_AND_COGNITION.md) | [`plan/REASONING_AND_COGNITION.md`](plan/REASONING_AND_COGNITION.md) |
| [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](architecture/ELASTIC_CAPACITY_AND_SCALING.md) | [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](plan/ELASTIC_CAPACITY_AND_SCALING.md) |

---

## Reading order

1. This hub → [`architecture/PLATFORM_FOUNDATION.md`](architecture/PLATFORM_FOUNDATION.md) + [`plan/PLATFORM_FOUNDATION.md`](plan/PLATFORM_FOUNDATION.md)
2. [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) + matching plan
3. [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) + matching plan
4. Your domain pair from the table above
5. [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when building agents

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
| 11–13 Tools / skills / integrations | `TOOLS` · `SKILLS` · `INTEGRATIONS` |
| 14 RAG | `RAG` (+ `MEMORY` for Knowledge vs LTM boundary) |
| 15–16 Memory / context | `MEMORY` |
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

Full audit procedure: [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md).

---

## Platform runtime capabilities (cross-domain index)

Essential platform behaviours span multiple domain pairs — use this index before opening unrelated docs.

| Capability | Primary architecture | Plan phase |
|------------|---------------------|------------|
| Resilience policies (retry, reboot, circuit breaker) | [`architecture/RELIABILITY_FAILURE_AND_HITL.md`](architecture/RELIABILITY_FAILURE_AND_HITL.md) §34 | REL-ADV |
| Orchestration strategies (parallel, sequence, cooperation, scale, redundancy) | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §50–§53, §58 | ORCH-5, ORCH-6 |
| MVP → product evolution (eval, KPI, simulation, promotion) | [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) §44 | MVP-EVOL |
| Autonomy slider (manual / ask / autonomous) | REL §35 + [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.10.2 | REL-ADV |
| Sync / async execution postures | [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §57 | ORCH-6 |
| Interrupt anywhere / resume from checkpoint | [`architecture/NEXUS_EXECUTION_FLOW.md`](architecture/NEXUS_EXECUTION_FLOW.md) §28 + UAEP §42.8–§42.9 | FLOW-CTL |
| Guardrails / policy enforcement (catalog) | [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.11.6 · §42.37 · vendor backends [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) §47 | GR-DOC · M.12 |
| RAG / retrieval engine | [`architecture/RAG.md`](architecture/RAG.md) · integration slugs [`architecture/INTEGRATIONS.md`](architecture/INTEGRATIONS.md) | M-RAG · M-RAG-DEPTH |

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
| [`adr/ADR-FLOW-001.md`](adr/ADR-FLOW-001.md) | Declarative delegation (`DELEGATES_TO`) |
| [`adr/ADR-FLOW-002.md`](adr/ADR-FLOW-002.md) | Reserved lifecycle states |
| [`adr/ADR-FLOW-003.md`](adr/ADR-FLOW-003.md) | `MODIFY_PLAN` semantics |
| [`adr/ADR-FLOW-004.md`](adr/ADR-FLOW-004.md) | Graph spec seed guard (`trigger_capabilities`) |

**Platform configuration canon (CFG-*):** [`architecture/ORCHESTRATION.md`](architecture/ORCHESTRATION.md) §56 · implementation [`plan/ORCHESTRATION.md`](plan/ORCHESTRATION.md) Phase **ORCH-CONFIG**.
