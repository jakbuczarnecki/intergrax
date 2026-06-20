# Tier-3 Application Environment, Sandbox, and Shadow Workspace

**Status:** Canonical architecture (domain pair 1:1) · **Application authoring gate:** §24–§50 + APP-CON-* / APP-EVOL-* / APP-OPS-* (host environments)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §26  
**Audit layers:** 3, 28  
**Audit instruction:** [`audit/TIER3_APPLICATION_ENVIRONMENT.md`](../audit/TIER3_APPLICATION_ENVIRONMENT.md)  
**Agent cooperation:** [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) §30 · §35–§39 · [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix H · AC  
**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates H-APP + APP-CON/EVOL/OPS)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (TIER3_APPLICATION_ENVIRONMENT canon).

- **Implement / audit default:** §20–§25 host profile + manifest wiring. Extended §26–§39: [`arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md`](arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md). §40+: [`arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md`](arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/TIER3_APPLICATION_ENVIRONMENT.md`](../guides/audit_slices/TIER3_APPLICATION_ENVIRONMENT.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md`](arch/TIER3_APPLICATION_ENVIRONMENT_extended_depth.md) | extended depth |
| [`arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md`](arch/TIER3_APPLICATION_ENVIRONMENT_production_gates.md) | production gates |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


## Table of contents

| § | Topic |
|---|--------|
| [§20](#20-shadow-workspace-model) | Shadow workspace model |
| [§21](#21-sandbox-model) | Sandbox model |
| [§22](#22-application-environment-profile-canonical) | **ApplicationEnvironmentProfile** (composition root) |
| [§22.6](#226-hierarchical-profile-bundles) | **Hierarchical profile bundles** (P1-ARCH-01 · ADR-APP-003) |
| [§23](#23-application-interaction-postures-canonical) | Interaction postures, routing, scenarios |
| [§24](#24-application-contract) | **Application contract** (`ApplicationManifest`) |
| [§25](#25-application-interface-run_task-facade-harnessapplication-and-applicationhost) | **Application interface:** `run_task()`, `HarnessApplication`, `ApplicationHost` |
| [§26](#26-application-execution-result) | **Application execution result** (Plane A) |
| [§27](#27-application-roster-and-registry-assembly) | Roster and registry assembly |
| [§28](#28-application-environment-architecture-app) | **Application Environment Architecture (APP)** |
| [§29](#29-tier-and-terminology-canon-application) | Tier and terminology canon (application) |
| [§30](#30-three-environment-control-modes) | Three environment control modes |
| [§31](#31-author-facing-harnessapplication-facade) | Author-facing `HarnessApplication` facade |
| [§32](#32-applicationhost-hook-surface) | **ApplicationHost hook surface** |
| [§33](#33-dual-observability-application-and-agent-planes) | Dual observability planes |
| [§34](#34-per-agent-binding-from-the-application) | Per-agent binding from application |
| [§35](#35-use-case-catalog-application--environment) | Use-case catalog |
| [§36](#36-final-architecture-application--agent--harness-cooperation) | Final architecture synthesis |
| [§37](#37-pre-implementation-operational-contracts-app-con) | Pre-implementation operational contracts |
| [§38](#38-execution-responsibility-stack-l4-application) | Execution stack: L4 application |
| [§39](#39-organizational-policy-envelope--virtual-workforce) | Organizational policy envelope |
| [§40](#40-production-reliability-safety-and-release-gates-tier-3) | Production reliability and release gates |
| [§41](#41-composition-primitives-separation-matrix) | Composition primitives separation |
| [§42](#42-applicationenvironmentstate-typed-host-state) | **ApplicationEnvironmentState** |
| [§43](#43-budget-reactions-and-token-governance) | Budget reactions and token governance |
| [§44](#44-scenario-test-matrix-tier-3) | Scenario test matrix |
| [§45](#45-checklist-for-new-application-implementation) | New application checklist |
| [§46](#46-production-readiness-acceptance-criteria) | Production readiness acceptance criteria |
| [§47](#47-developer-mental-model) | **Developer mental model** (recipes) |
| [§48](#48-application-artifacts) | **Application artifacts** |
| [§49](#49-runtime-evolution-and-governance) | **Runtime evolution and governance** |
| [§50](#50-platform-operations-canon) | **Platform operations canon** (capability graph, registry, ownership, health) |
| [§51](#51-cross-document-consistency-freeze) | **Cross-document consistency** (freeze audit) |

---

---

# 45. Checklist For New Application Implementation

Before implementing a new Tier-3 environment, answer:

```text
 1. What product hypothesis does this environment test?
 2. What is app_id and deployment posture (§23.1)?
 3. Which agents are on the roster — AgentBinding.mount for each?
 4. What capabilities route tasks — explicit L1 or classifier L3?
 5. Single-agent or multi-agent — graph_spec vs pipeline token (§23.4)?
 6. Full ApplicationEnvironmentProfile declared — no orphan slices?
 7. wire_application_environment() — no getattr on manifest?
 8. build_harness_host_runtime() — not ad-hoc NexusLoop?
 9. All surfaces → UnifiedTaskRunner.run_task()?
10. IdentityProfile matches auth story (tenant/user)?
11. ExecutionMode STRICT for prod?
12. ObservabilityProfile + ApplicationRunSummary on Task completion?
13. Business logic only in Tier-2 agents?
14. Org simulation needed — OrganizationalPolicyEnvelope (§39)?
15. Dynamic reactions — ApplicationHost hooks (§32) vs profile-only?
16. Deploy triad present (Docker, BUILD_AND_DEPLOY, .env.example)?
17. pytest smoke for manifest + host factory?
18. Cross-ref product ARCHITECTURE.md — not duplicated in platform plan?
```

If these questions cannot be answered, do not ship the host. **Guides:** [`guides/APPLICATION_CREATION_GUIDE.md`](../guides/APPLICATION_CREATION_GUIDE.md) · [`applications/USAGE.md`](../../applications/USAGE.md) · [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Step 4E · Appendix H.

---
