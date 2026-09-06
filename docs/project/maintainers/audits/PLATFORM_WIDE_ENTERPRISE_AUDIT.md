# Platform-Wide Enterprise Architecture / Security / Modularity Audit

**Audit ID:** PLATFORM-ENTERPRISE-AUDIT-1  
**Date:** 2026-09-03  
**Baseline ancestor:** `13092546caaf76a209161f747b7fce6ec6fa9897` (contained in HEAD — verified)  
**HEAD audited:** `f66137de9bda3b82b8c2653de6a476c1760afe90`  
**Branch:** `development`  
**Classification:** **IMPLEMENTATION_PENDING** (ADR-PLATFORM-SE-CONVERGENCE decided; P1 open until PLATFORM-SE-FAIL-CLOSED-1)

**Concurrent work note:** Untracked WIP under `tests/system/functional_diagnostics_durability/` (D1-R1 durability gates) was present at audit time and is evaluated as in-progress qualification infrastructure, not production platform code.

---

## 1. Executive verdict

Intergrax is **one platform with many mature domain slices**, not many unrelated local systems — but it is **not yet a single converged enterprise surface**. Canonical ownership is identifiable for Integrations, execution identity, Collaborative Work, provider qualification, platform plugins, diagnostics contracts, and governed continuation. The dominant residual risk is **parallel authority models for meaningful side effects** (Collaborative Work boundary vs Declarative Tool Authorization at the `RuntimeToolInvoker` boundary) without a documented, fail-closed convergence rule for all production mutation paths. Secondary risks are **intentional but unresolved provider-boundary splits** (LLM/embedding/RAG vs Integrations catalog), **partial Autonomous Work implementation** ahead of product claims, and **host-adoption gaps** (LKW durable CW, diagnostics emission at some adoption boundaries).

No P0 security defect was proven in audited production paths. Bounded P1/P2 findings are concrete and actionable. Several items require explicit architecture decisions before safe unification.

**Master question answer:** Intergrax is **one platform** reusing shared contracts and mechanisms, with **domain-local registries where semantics differ** — not a second platform hidden in every domain. Exceptions requiring decision: LLM provider plane, functional qualification runner, vendor_knowledge sync registry, and generic tool execution authorization vs External Work collaborative boundary.

---

## 2. Product truth

### Implemented + production-composable today

| Capability | What users/operators can do | Evidence |
|------------|---------------------------|----------|
| **Hosted applications** | Run Tier-3 apps (LKW, legal, governed contractor, assistant, research, attestation demo) via `wire_application_environment()` / host factories | `applications/*/host/factory.py`, `intergrax/applications/_shared/environment_wiring.py` |
| **Agent execution (Nexus/UAEP)** | Submit tasks, run agent graphs, tool invocation, checkpoints, HITL pause/resume | `runtime/nexus/`, `agents/uaep.py`, execution identity contracts |
| **Integrations** | Select providers per category via `IntegrationProfile`; 186 runtime-cutover slugs | `integrations/registry/catalog.py`, gate tests |
| **Provider qualification** | Record durable provider/capability qualification facts (PG/Mongo when env available) | `core/qualification/`, integration tests |
| **Collaborative Work (MP-1)** | Membership, delegation, authority, policy profiles, enforcement gate, PG/SQLite repos | `collaborative_work/`, prior CW audit ENTERPRISE_READY |
| **External Work** | Governed contractor demo with `MeaningfulSideEffectAuthorizationBoundary` | `agents/external_contractor_adapter/`, `governed_contractor_application/` |
| **Tools / Skills plugins** | External EP registration, runtime enablement via profile | `tools/registry/`, `skills/registry/`, plugin contract tests |
| **Diagnostics (functional evidence)** | Platform functional evidence model, document-store persistence, analyzer specs Q1–Q4 | `runtime/diagnostics/` |
| **Public proofs** | Scenario library with evidence/artifact verification | `platform_proofs/scenarios/` |
| **Agent distribution** | Package identity, roster, runtime revision, materialization | `agent_distribution/` |

### Implemented library capability (not full product)

| Capability | State |
|------------|-------|
| **Autonomous Work (AW-2B)** | `WorkerLifecycleService`, repository ports, in-memory repo — **no production host consumer** |
| **Functional qualification (Q5)** | Runner + `QualificationPluginRegistry` — distinct from provider qualification; proof suites in `tests/system/functional_diagnostics_*` |
| **Activity / WorkItem (CW MP-2+)** | Contracts planned; not implemented as business objects |
| **INTEGRATIONS-3B** | Registry-backed runtime binding — planned, not shipped |

### Tested proof capability

- Collaborative Work PG qualification (when PG available)
- Mongo provider qualification discovery/validity (7 passed in audit env)
- Tools side-effect safety (declarative policy + idempotency)
- Platform plugin cross-flow integration proofs
- Functional diagnostics durability D1-R1 (WIP, subprocess IPC design reviewed)

### Planned / not implemented

- Virtual Worker product (full AW runtime, recovery controller, durable worker repos in production)
- Universal LLM inside Integrations catalog
- Activity projection as product feature
- MP-2+ collaborative artifacts

---

## 3. Architecture map

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Tier-3 Application Host (factory.py, environment_wiring, profiles)      │
│  compose: IntegrationProfile, ToolProfile, SkillProfile, PolicyBundle   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Platform      │     │ Agent / Nexus   │     │ Governance       │
│ Plugins EP    │     │ Execution       │     │ Policy / HITL    │
│ discovery     │     │ Task lifecycle  │     │ Side-effect auth │
└───────┬───────┘     └────────┬────────┘     └────────┬─────────┘
        │                      │                         │
        ▼                      ▼                         ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Integrations Catalog (authoritative registration)                      │
│  → registry_v2 (derived projection) → IntegrationProfile selection  │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Provider adapters (relational, document, vector, KV, observability) │
└───────────────────────────────────────────────────────────────────────┘

Orthogonal validation layers:
  Provider Qualification (core/qualification)
  Functional Qualification (core/qualification + runtime/diagnostics evidence)
  Public Proofs (platform_proofs)
```

**Dependency direction:** Verified — `intergrax/` does not import `applications/` (except CLI demo import in `cli/external_work.py`). `agents/` does not import `applications/`. Tier boundaries respected.

---

## 4. Capability map

| Capability | Purpose | Authoritative module | Public contract | Production consumers | Persistence owner | Runtime owner | Pluginability | Diagnostics | Evidence | Maturity | Duplicate/legacy |
|------------|---------|---------------------|-----------------|---------------------|-------------------|---------------|---------------|-------------|----------|----------|-------------------|
| **Core contracts** | Cross-domain types | `intergrax/contracts/` | Per-domain pydantic models | All tiers | N/A | N/A | N/A | Via runtime | Trace/receipt contracts | STRONG | None |
| **Execution** | Run/attempt/execution identity | `runtime/execution/` | `contracts/execution_identity.py` | Nexus, UAEP, tasks | Host checkpoint stores | `ExecutionRuntime` | N/A | Trace bridge | Execution evidence | STRONG | Governance bridge only |
| **Nexus / Task** | Agent loop, task lifecycle | `runtime/nexus/`, `runtime/task/` | `contracts/task_envelope.py` | All harness apps | Task checkpoints | `NexusLoop`, `TaskLifecycle` | Tool/skill plugins | Trace events | Agent run trace | STRONG | `queueing/worker/registry` separate concern |
| **Autonomous Work** | Durable worker lifecycle | `autonomous_work/` | `contracts/autonomous_work/` | **None (library only)** | In-memory repo | `WorkerLifecycleService` | N/A | Not wired | Not wired | PARTIAL | No duplicate runtime |
| **Agents** | Agent definitions, UAEP | `agents/`, `runtime/registry/` | `contracts/agent_run.py` | Apps via projection | Agent persistence optional | UAEP | Agent distribution | Partial | Run results | ADEQUATE | Registry projection at host |
| **Agent distribution** | Package install/roster | `agent_distribution/` | `agent_distribution/identity.py` | LKW certification wiring | In-memory stores | Materialization services | EP packages | Limited | Lock packages | ADEQUATE | Not authority source |
| **Applications** | Host composition | `applications/`, `applications/_shared/` | `applications/contracts/` | Production hosts | Per-app | `wire_application_environment` | App manifests | Host wiring gates | App evidence | ADEQUATE | No local policy engine duplication |
| **Collaborative Work** | Shared authority | `collaborative_work/` | `contracts/collaborative_work.py` | Governed contractor, E2E harness | PG/SQLite via Integrations | Enforcement gate | Provider plugins | audit_payload | Policy decisions | STRONG | None |
| **Integrations** | Provider binding | `integrations/` | `integrations/contracts/` | All data paths | Provider adapters | Catalog + profile | `IntegrationPlugin` EP | Provider health | Qualification | STRONG | `registry_v2` projection only |
| **Plugins (coordination)** | Package discovery | `core/plugins/` | `package_contract.py` | All plugin kinds | N/A | Domain catalogs | setuptools EP | Qualification vocab | Plugin evidence | ADEQUATE | Domain-owned registries by design |
| **Provider qualification** | Vendor/capability facts | `core/qualification/` | `provider.py`, `execution.py` | Integrations materialization | PG/Mongo stores | Qualification runner | Suite plugins | `observability.py` | `ProviderQualificationRun` | STRONG | Distinct from functional Q |
| **Functional qualification** | Behavior quality eval | `core/qualification/functional_*` | `functional_qualification_*.py` | Test suites Q1–Q5 | Via diagnostics persistence | `QualificationPluginRegistry` | `FunctionalQualificationPlugin` | Functional specs | Normalized case results | PARTIAL | Justified separate engine |
| **Governance** | Control plane mutations | `runtime/governance/` | `contracts/control_plane_mutation.py` | Host wiring | Varies | `execution_guard.py` | Policy plugins | Partial | Audit payloads | ADEQUATE | Composes with policy |
| **Runtime policy** | Tool/side-effect rules | `runtime/policy/` | `contracts/runtime_policy.py` | All wired hosts | N/A | `RuntimePolicyEngine` | Policy rule EP | Trace | Policy decisions | STRONG | `policy_engine.py` layered |
| **HITL** | Human pause/continuation | `runtime/human/` | `contracts/governed_continuation.py` | Nexus + External Work | Human decision stores | Grant coordinator | N/A | Problems | Grants consumed | STRONG | No second approval framework |
| **Memory** | Context/personalization | `memory/` | `memory/contracts/` | LKW, apps via wiring | Store plugins | Resolver | Memory store EP | N/A | N/A | ADEQUATE | Not authority |
| **Evidence / trace** | Audit truth | `runtime/evidence/`, `contracts/execution_evidence/` | Receipt contracts | Proofs, execution | Store-backed | Collectors | N/A | Overlap with diagnostics | Authoritative | ADEQUATE | vs Activity (not impl) |
| **Observability** | Telemetry export | `runtime/observability/` | Span attributes | Wired hosts | OTLP backends | Emitter/export bridge | Observability vendor integrations | Problems optional | Metrics/traces | ADEQUATE | LLM tracking separate |
| **Diagnostics** | Operator problems | `runtime/diagnostics/` | `functional_diagnostic_bounds.py` | LKW hardening proofs | Document store PG/Mongo | Analyzer + persistence | N/A | Self | Functional evidence | ADEQUATE | Not domain DB silos |
| **Persistence topology** | Deployment modes | `contracts/persistence_topology.py` | Topology enum | Qualification, hosts | Provider-owned | N/A | Provider plugins | Errors w/o secrets | N/A | STRONG | CW/diagnostics domain SQL OK |
| **RAG** | Retrieval pipeline | `rag/` | Vector/retriever contracts | LKW, proofs | Via integrations | `RetrievalService` | Handler registry | RAG spans | Retrieval evidence | ADEQUATE | `legacy/rag_answers/` |
| **Tools** | Tool plugins | `tools/` | `tools/core/contracts.py` | Nexus invoker | Idempotency store | `ToolRegistry` | Tool EP | Tool trace | Execution results | STRONG | None |
| **Skills** | Skill plugins | `skills/` | `skills/core/contracts.py` | Agent registry | N/A | `SkillRegistry` | Skill EP | N/A | N/A | ADEQUATE | Does not grant authority |
| **External Work** | External mutations | `agents/external_contractor_adapter/` | Adapter contracts | Governed contractor | Host store | Adapter + boundary | N/A | Partial | Work receipts | ADEQUATE | Uses CW boundary |
| **Vendor knowledge** | Live resource sync | `runtime/vendor_knowledge/` | Sync models | LKW connected sources | Document store | Sync worker | VK registry | Sync events | Sync state | PARTIAL | Parallel to integrations for discovery |
| **Public proofs** | Product evidence | `platform_proofs/` | Scenario descriptors | CI/manual | Artifact dirs | Scenario runners | N/A | Proof logs | Public artifacts | ADEQUATE | Must use real platform paths |
| **LLM** | Model inference | `llm_adapters/` | `llm_provider.py` | All LLM hosts | N/A | `llm_provider_registry.py` | Builtin + dynamic | LLM tracking | Usage metadata | ADEQUATE | **Outside Integrations** |

---

## 5. Ownership matrix

| Concern | Authoritative owner | Public contract | Consumers | Forbidden duplication |
|---------|---------------------|-----------------|-----------|----------------------|
| Provider registration | `integrations/registry/catalog.py` | `IntegrationEntry`, manifests | All domains | Independent runtime provider map |
| Contract projection | `runtime/integrations/registry_v2.py` | `PlatformIntegrationContract` snapshot | Compatibility gates | Qualification dispatch via projection |
| Integration runtime selection | Host `IntegrationProfile` | `integrations/contracts/base.py` | Apps, CW, diagnostics | Domain env DSN parsing |
| Execution identity | `contracts/execution_identity.py` + lifecycle | `mint_*`, validators | Nexus, trace, qualification | Helpers minting run_id outside lifecycle |
| Nexus Task | `runtime/task/task_lifecycle.py` | `Task`, `TaskState` | Harness apps | Conflation with WorkItem |
| WorkItem (CW) | Planned MP-2+ | `contracts/collaborative_work.py` (future) | Not implemented | Nexus task_id reuse |
| Human approval | `runtime/human/governed_continuation_*` | `GovernedContinuationRequest` | Boundary, declarative tool HITL | Domain approval stores |
| Meaningful side effects (CW) | `collaborative_work/enforcement_gate.py` | `CollaborativeWorkEnforcementRequest` | External Work boundary | Second generic approval framework |
| Meaningful side effects (tools) | `runtime/policy/declarative_enforcer.py` + `RuntimeToolInvoker` | Declarative policy rules | Generic tool execution path | Silent bypass when unwired |
| Collaborative authority | `collaborative_work/authority.py` | Principal, grants, delegation | CW gate | Memory as authority |
| Diagnostics problems | `runtime/diagnostics/problem_lifecycle.py` | Problem contracts | Host export | Per-domain diagnostics DB |
| Observability | `runtime/observability/emitter.py` | Span attributes | Wired hosts | Production Recording* silos |
| Provider qualification | `core/qualification/execution.py` | `ProviderQualificationRequest` | Materialization gates | Fake qualification |
| Functional qualification | `core/qualification/functional_qualification_runner.py` | Plugin descriptor | Q1–Q5 suites | Substituting provider qualification |
| Tool plugins | `tools/registry/catalog.py` | `ToolPlugin` | Nexus | Core edit per tool |
| Skill plugins | `skills/registry/catalog.py` | `SkillPlugin` | Agent registry | Skill grants permissions |
| Agent distribution identity | `agent_distribution/identity.py` | `AgentPackageIdentity` | LKW roster | Distribution as authority |
| Principal | `contracts/collaborative_work.py` | `PrincipalKind` | CW | AgentDefinition conflation |
| Memory | `memory/` | Store plugins | Context only | Authorization decisions |

---

## 6. Dependency direction

**Expected:** contracts → platform mechanisms → domains/adapters → applications/hosts

**Verified:**
- No `agents/` → `applications/` imports
- No `intergrax/` → `applications/` except CLI demo path (`cli/external_work.py` — acceptable composition root)
- Provider SDK imports confined to `integrations/providers/**`, `llm_adapters/providers/**`, `distributed/providers/**`, `rag/embedding/providers/**` (see vendor matrix)

**Reverse dependency risks:** None material in Tier-0/2. Tier-3 apps correctly import platform.

---

## 7. Platform reuse

Domains **reuse** platform mechanics:

| Domain | Reuses | Local recreation |
|--------|--------|------------------|
| Collaborative Work | Integrations PG/SQLite, qualification runner, runtime policy, HITL | Domain SQL/schema only |
| External Work | CW boundary, execution, task lifecycle | Adapter semantics |
| Diagnostics | Document store integrations, secret_safety | Domain diagnostic specs |
| Provider qualification | Integrations materialization, persistence ports | Suite semantics |
| LKW | Full host wiring stack | Workspace domain repos (acceptable) |
| Vendor knowledge | Document store, tenant connections | Sync registry (domain-specific) |
| Autonomous Work | Contracts pattern, repository ports | No parallel execution engine |

**Second platform hidden in domain?** **No** for CW, diagnostics, qualification. **Partial** for vendor_knowledge (sync + connection registry). **Partial** for LLM plane (separate from Integrations by design).

---

## 8. Duplication matrix

| Mechanism | Canonical owner | Other implementations | Duplicate? | Risk | Action |
|-----------|-----------------|----------------------|------------|------|--------|
| Integration registration | `integrations/registry/catalog.py` | `registry_v2` projection | No (derived) | Low | Keep |
| Tool registry | `tools/registry/runtime.py` | `catalog.py`, nexus executor | No (layers) | Low | Document |
| Skill registry | `skills/registry/runtime.py` | `catalog.py` | No | Low | Keep |
| Agent registry | `runtime/registry/agent_registry.py` | Host projection snapshots | No | Low | Keep |
| LLM providers | `llm_adapters/llm_provider_registry.py` | Embedding registry in RAG | **Yes (intentional)** | Medium | ADR: Integrations vs LLM |
| Policy evaluation | `runtime/policy/runtime_policy_engine.py` | `policy_engine.py`, declarative enforcer | No (layers) | Low | Document precedence |
| Side-effect authorization | CW `enforcement_gate` + `meaningful_side_effect_authorization` | Declarative Tool Authorization + idempotency (`RuntimeToolInvoker`) | **Parallel** | **High** | ADR: convergence |
| Qualification engines | Provider: `core/qualification/execution.py` | Functional: `functional_qualification_runner.py` | No (semantic) | Low | Guard scope |
| Task systems | Nexus `TaskLifecycle` | Queue jobs, qualification runs, BG tasks | No (semantic) | Medium | Preserve ID namespaces |
| Retry | `llm_adapters/_shared/retry.py` | Execution retry, vendor transport | No (layer) | Low | Keep separated |
| Idempotency | `contracts/idempotency_store.py` | CW idempotency table, tool coordinator | No (scope) | Low | Keep |
| Diagnostics vs observability | Diagnostics problems + functional evidence | OTLP traces/metrics | No | Low | Host wires both |
| Registry (vendor knowledge) | `runtime/vendor_knowledge/registry.py` | Integration catalog | Partial overlap | Medium | Reuse provider open where possible |
| Plugin discovery | `core/plugins/discovery.py` | Per-domain `plugin_register.py` | No (by design) | Low | Keep |
| Memory store resolution | `memory/resolver/` | Tool memory providers | No | Low | Keep |
| RAG vector store | `rag/vectorstore/` + integrations | `legacy/rag_answers/` | Legacy | Low | Migrate/remove legacy |

---

## 9. Legacy matrix

| Legacy path | Current consumers | Reason | Risk | Action | Priority |
|-------------|-------------------|--------|------|--------|----------|
| `intergrax/legacy/rag_answers/` | Unknown/minimal | Pre-canonical RAG | Medium drift | MIGRATE | P3 |
| `registry_v2` alias `build_contract_registry_snapshot` | Compatibility tests | Projection API stability | Low | KEEP | P3 |
| Nexus `tool_runtime.py` LEG-1 boolean flags | Planning bridge | Manifest migration | Low | KEEP until cutover | P3 |
| `catalog_context.py` legacy step fallback | Nexus steps | Phase O.5b | Low | KEEP | P3 |
| `compat/langchain/` | RAG loaders | Interop | Low | KEEP | P3 |
| Deferred `llm_guardrail` slugs (9) | None production | Layout deferred | Low | KEEP | P2 |
| In-memory CW/AW repos | Tests, demos | Reference impl | Low (if not prod) | KEEP reference | P3 |
| `contract_capture` reflection | Built-in integration registration | One-time metadata | Medium | MIGRATE to explicit specs | P2 |

---

## 10. Pluginability matrix

| Surface | Public contract | Registration | External plugin? | Core edit required? | Runtime enablement | Status |
|---------|-----------------|--------------|------------------|---------------------|-------------------|--------|
| Integrations | `IntegrationPlugin`, manifests | `register_from_manifest` + EP | Yes | No | `IntegrationProfile` | STRONG |
| Tools | `ToolPlugin` | `register_tool_plugin` + EP | Yes | No | `ToolProfile` bundles | STRONG |
| Skills | `SkillPlugin` | `register_skill_plugin` + EP | Yes | No | Skill profile / agent registry | ADEQUATE |
| Policy rules | Policy rule EP | `runtime/policy/rules/registry.py` | Yes | No | Policy bundle wiring | ADEQUATE |
| Security defenses | Defense EP | `defense_registry.py` | Yes | No | Host enablement | ADEQUATE |
| Memory stores | `MemoryStorePlugin` | EP discovery | Yes | No | Resolver materialization | ADEQUATE |
| Functional qualification | `FunctionalQualificationPlugin` | `QualificationPluginRegistry` | Yes (explicit register) | No | Test runner plans | PARTIAL |
| LLM providers | `LLMAdapter` | Registry builtin table + dynamic | Partial | Add row for builtin | Host LLM profile | ADEQUATE |
| Provider qualification suites | Suite protocols | Integrations binders | Via provider packages | No | Qualification execution | STRONG |
| Agent distribution packages | Package identity | Install/materialize | Yes (packages) | No | Roster activation | ADEQUATE |

---

## 11. Integrations / provider abstraction

**Confirmed canonical architecture:**
- Catalog = single registration authority (`integrations/registry/catalog.py`)
- `registry_v2` = derived projection only (`runtime/integrations/registry_v2.py`)
- `IntegrationProfile` = runtime selection
- External plugins register once via manifest/EP
- No independent runtime provider map for qualification dispatch

**Vendor SDK classification (production Tier-0):**

| Class | Paths |
|-------|-------|
| **A — Provider adapter** | `integrations/providers/**`, `llm_adapters/providers/**`, `distributed/providers/**` |
| **B — Composition root** | `integrations/_shared/p3/factories.py` (lazy import) |
| **C — Test / proof** | `tests/**`, `proof_infrastructure/**`, `platform_proofs/**/qdrant/*` |
| **D — Violation** | `tools/providers/openai_vector_store/service.py` (OpenAI SDK in tool service); `rag/embedding/providers/*` (embedding SDK outside integrations adapters) |

**P2 carry:** `contract_capture.py` registration-time reflection — acceptable at registration, not at runtime dispatch.

**No generic core vendor switch** — confirmed.

---

## 12. Execution / task architecture

**Authoritative path:**
```text
mint_run_id / mint_attempt_id / mint_execution_id (contracts/execution_identity.py)
  → ExecutionRuntime / UAEP / NexusLoop
  → TaskLifecycle (runtime/task/)
  → Tool invoker / agent steps
  → Checkpoints (host-owned persistence)
```

**Semantic distinctions:**

| Concept | Owner | ID type | Notes |
|---------|-------|---------|-------|
| Nexus Task | `runtime/task/` | `task_id` | Harness work unit |
| WorkItem | CW (planned) | Not implemented | Must not alias task_id |
| Provider qualification run | `core/qualification/` | `qualification_run_id` | Platform-global facts |
| Functional qualification run | `functional_qualification_runner` | Plan-scoped | Behavior eval |
| Background job | `background_tasks/` | Job-specific | Separate scheduler |
| External work operation | External adapter | `operation_id` | Uses CW boundary |
| Agent run | `contracts/agent_run.py` | `run_id` | Execution correlation |

**Competing runtimes?** No — single Nexus/UAEP execution stack. Queue workers dispatch into it.

**Recent tightening:** `trace_bridge.py` requires active execution identity or explicit IDs — correct fail-closed for trace correlation; breaks tests that omit identity (see test findings).

---

## 13. Autonomous Work

**Code present:** `autonomous_work/lifecycle.py` (`WorkerLifecycleService`), `repository.py` ports, `in_memory_repository.py`.

**Platform reuse:** Uses contract validation, CAS/revision patterns — does **not** recreate execution engine, policy, or diagnostics.

**Gaps:**
- No production host wires AW repositories
- No durable backend qualification
- No side-effect integration (no worker mutations in production)
- Docs (`AUTONOMOUS_WORK.md`) state "runtime not implemented" while AW-2B lifecycle code exists — **documentation truth gap (P2)**

**Bypass risk:** None in production (not wired).

---

## 14. Agent distribution

**Frozen distinctions preserved:**
- `AgentPackageIdentity` / `AgentPackageCandidate` ≠ `AgentRun` ≠ `Principal`
- `agent_distribution/effective_roster_authority.py` — roster does not amplify authority
- Runtime lock prevents forged identity persistence (unit tested)

**Maturity:** ADEQUATE library; LKW uses certification/roster wiring.

---

## 15. Applications / composition

**Production hosts:** 9 `factory.py` roots — all use `applications/_shared/*_wiring.py` pattern.

**Apps do NOT implement locally:** policy engine, provider registry authority, generic approval store, execution runtime.

**App-local domain OK:** LKW workspace repos, legal serving, connected source sync.

**Adoption gaps (P2):**
- LKW: no durable CW persistence adoption
- Diagnostics RuntimeEvent emission incomplete at some CW adoption boundaries (per prior CW audit)

---

## 16. Collaborative Work

**Status:** ENTERPRISE_READY (MP-1) per `COLLABORATIVE_WORK_FINAL_ENTERPRISE_AUDIT.md` — **reconfirmed**.

Invariants closed:
- Tenant ≠ User; Workspace = collaboration boundary
- Principal ≠ AgentDefinition ≠ AgentRun
- Delegation never amplifies
- Memory not authority
- WorkItem ≠ Nexus Task (WorkItem not implemented)

**P2 residual:** `side_effect_scope_id` correlation with resource scope not enforced in gate.

---

## 17. Identity / authority

**Trust boundaries:**
```text
Authentication (API key, host identity)
  → Principal membership (CW repository)
  → Authority grants + optional delegation
  → Policy profiles + runtime policy
  → Enforcement gate / declarative enforcer
  → Governed continuation grant (one-time, scoped)
  → Side effect execution
```

**Forbidden shortcuts — audit result:**
| Shortcut | Found? |
|----------|--------|
| Provider qualified → operation allowed | No — qualification gates materialization, not business auth |
| Agent assigned → authority granted | No — assignment ≠ grant |
| Plugin installed → runtime enabled | No — profile enablement separate |
| Human response → unrestricted authority | No — grant consumed, scoped |
| Memory value → security decision | No production path found |

---

## 18. Policy / governance

**Inventory:**
- `RuntimePolicyEngine` — meaningful side effect evaluation
- `policy_engine.py` — bundle composition
- `declarative_enforcer.py` — tool invocation rules
- `runtime_policy_bundle_evaluator.py` — Protocol surface
- `governance/execution_guard.py` — control plane
- `collaborative_work/policy_composition.py` — CW layer composition

**Precedence:** Explicit in CW (`DENY` < `REQUIRE_HUMAN` < `ALLOW`). Declarative tool rules fail closed on `should_block_execution`.

**Gap (resolved by ADR):** Strategy selection is host/DI-bound per execution context; same mutation must not require both strategies simultaneously. See ADR-PLATFORM-SE-CONVERGENCE §11.

---

## 19. HITL

**Canonical components:**
- `HumanPauseCoordinator` / pause bridge
- `GovernedContinuationGrantCoordinator`
- `DeclarativeHitlApprovalGrant` for tool loop
- `compose_governed_continuation_from_enforcement`

No second generic approval framework found. Domain decision records in External Work are semantically scoped.

---

## 20. Meaningful side effects

**Production mutation paths audited:**

| Path | Authorization | Bypass? |
|------|---------------|---------|
| External Work adapter | `MeaningfulSideEffectAuthorizationBoundary` | No when wired |
| Generic tool execution (`RuntimeToolInvoker`, side_effects=True) | Optional scope policy + optional declarative enforcer + idempotency | **Yes when host omits policy wiring** |
| Control plane mutations | `control_plane_mutation_authorization.py` | Gate tested |
| CW repository writes | Authority via API caller | Domain port |

### P1 — FINDING-PLATFORM-SE-001 — **IMPLEMENTED_PENDING_INDEPENDENT_AUDIT**

- **Status:** Phase 1 fail-closed gate implemented in `RuntimeToolInvoker` (PLATFORM-SE-FAIL-CLOSED-1). Awaiting independent SHA security audit.
- **Files:** `intergrax/runtime/nexus/tools/invoker.py`, `intergrax/runtime/policy/declarative_tool_authorization_gate.py`, `intergrax/runtime/policy/side_effect_authorization_errors.py`
- **Behavior (after fix):** `side_effects=True` requires `DeclarativePolicyEnforcer` in `ENFORCE` mode; absent or `AUDIT_ONLY` runtime ⇒ typed denial before executor/idempotency claim.
- **ADR:** [`ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md`](../architecture/ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md)

---

## 21. Fresh authorization

**Verified in:**
- `meaningful_side_effect_authorization.py` — fresh `evaluate()` before execute; grant match + consume
- `test_g5c2b2b_governed_side_effect_reauthorization.py` — stale grant rejected
- Declarative HITL grant scoped to invocation

**P2:** Proposal digest / resource change detection not uniform across tool path.

---

## 22. Persistence

**Pattern confirmed:** Domain owns schema/semantics; Integrations owns connection/session/driver.

**Violations:** None material. CW PG/SQLite adapters delegate to `PostgreSQLConnectionProvider`.

**Autonomous Work:** In-memory only — no false durability claim.

---

## 23. CAS / concurrency

**Authoritative mutable state with CAS:** CW repos, AW repos, idempotency store, qualification persistence, functional evidence append.

**Distributed claims:** PG CW qualification proves multi-connection; Mongo diagnostics proofs exist when env available.

**Local locks:** In-memory repos use threading locks — honest single-process only.

---

## 24. Idempotency

**Implementations:**
- Tool: `IdempotencyPreEffectCoordinator`, `InMemoryIdempotencyStore`, distributed store
- CW: `collaborative_idempotency` table
- LKW workspace: domain idempotency keys

**Duplication:** Acceptable — different semantic scopes.

**Conflict semantics:** Same key + different command fails explicitly (tested in tools suite).

---

## 25. Memory boundary

Memory informs context via `memory/resolver/` and tool providers. **No production path** uses memory for authorization, membership, or approval decisions.

**P2:** Ensure future AW continuity docs keep memory non-authoritative (documented in AUTONOMOUS_WORK.md).

---

## 26. Observability

| Signal | Owner | Production |
|--------|-------|------------|
| Runtime events | `runtime/events/` | Wired hosts |
| Traces | `observability/trace_scope.py` | OTLP when configured |
| Metrics | Emitter + vendor integrations | Partial |
| PlatformProblemSignal | Diagnostics | PG/Mongo durable proofs |
| Activity | Not implemented | N/A |

**Blind spot (P2):** CW adoption — `audit_payload` only; platform signal emission at boundary incomplete.

---

## 27. Diagnostics

**Map:**
- Problem creation: `problem_lifecycle.py`
- Persistence: `document_store_problem_persistence.py`
- Functional evidence: `functional_evidence.py` + document store persistence
- Host export: `hosted_application_diagnostic_wiring.py`
- Correlation: execution identity fields on evidence

Operator can answer what ran / who / tenant / run / failure when host wires diagnostics store.

---

## 28. Identifiers / traceability

**Relationship map (simplified):**
```text
tenant_id + workspace_id
  → principal_id (CW)
  → task_id (Nexus) ≠ work_item_id (future)
  → run_id → attempt_id → execution_id
  → qualification_run_id (provider)
  → trace_id / event_id (observability)
  → side_effect_scope_id (governed continuation)
  → approval_grant_id (HITL)
```

**P2:** `side_effect_scope_id` not correlated in CW gate with resource scope.

**No dangerous stringly ID interchange** found in authoritative repos (typed validators used).

---

## 29. Evidence

| Kind | Purpose | Secret-safe |
|------|---------|-------------|
| `ProofReceipt` | Public proof | Validated |
| Provider qualification evidence | Vendor facts | `secret_safety` on persistence |
| Functional evidence | Diagnostic facts | Cursor secret for HMAC |
| Policy `audit_payload` | Decision audit | Designed safe |
| Trace | Execution truth | Should not contain secrets |

No universal untyped evidence dump found.

---

## 30. Qualification systems

| System | Question answered | Engine |
|--------|-------------------|--------|
| Provider qualification | Is provider X capable in env Y? | `core/qualification/execution.py` |
| Functional qualification | Does behavior meet spec? | `functional_qualification_runner.py` |
| Public proof | Does product scenario work? | `platform_proofs/` |
| CW repository qualification | Does CW repo meet semantics? | `repository_qualification_suite.py` |

**No silent substitution** between systems.

---

## 31. Functional qualification (HIGH PRIORITY)

**Assessment:** Justified separate system — evaluates cross-domain functional behavior using `runtime/diagnostics` evidence, not provider materialization.

**Does not duplicate:**
- Provider qualification (different evidence kinds)
- Public proofs (product scenarios vs diagnostic specs)

**WIP:** `tests/system/functional_diagnostics_durability/` — D1-R1 subprocess durability gates (A–G), `DurableBackendProbe` protocol, Mongo backend probe. **Well-designed** — uses platform persistence, process-boundary proof, fail-closed conflict append.

**Risk (P2):** If functional qualification registry grows generic dispatch beyond diagnostic domains → scope creep. Current `QualificationPluginRegistry` is explicit DI — acceptable.

---

## 32. RAG / LLM / search

**RAG:** Canonical in `intergrax/rag/`; vector backends via Integrations. Embedding providers in `rag/embedding/providers/` — **parallel to LLM adapters (P2)**.

**LLM:** `llm_adapters/llm_provider_registry.py` — intentional separate plane per `LLM_ADAPTERS.md`. Builtin table requires core edit for new builtin provider.

**Silent model fallback:** `llm_routing_wiring.py` has explicit fallback profile — host-configured, not silent vendor switch.

---

## 33. Tools / Skills

**Tools:** Typed `ToolContract`, `side_effects` flag, scope policy, declarative policy, idempotency coordinator. **P1 gap** when policy unwired (see FINDING-PLATFORM-SE-001).

**Skills:** Registration only — no permission grants. Invokes tools through agent/tool runtime.

---

## 34. External Work

Uses canonical CW boundary. Identity and idempotency tested. No special bypass because "integration."

---

## 35. Vendor knowledge

Separate `runtime/vendor_knowledge/registry.py` and sync worker — domain-specific live resource discovery. Reuses document store + tenant connections. **Does not duplicate** integration catalog authority for static providers.

**P2:** Overlap potential when VK discovers providers — document boundary.

---

## 36. Public proofs

Scenarios consume real platform wiring (`platform_proofs/scenarios/`). Architecture conformance tested (`test_scenario_architecture_conformance.py`).

**P2:** `verified_product_identification` has scenario-local Qdrant adapter outside `integrations/providers/` — proof-scoped, acceptable if not claimed as platform provider.

---

## 37. Configuration / secrets

**Layers:** env → `ApplicationEnvironmentProfile` → provider options → plugin config.

**Secret safety:** `core/security/secret_safety.py` used in qualification persistence, VK tenant connections, agent distribution config, plugin package contract.

**P2:** Not all diagnostic/error paths audited exhaustively for secret leakage — spot checks clean; recommend continued `validate_secret_safe_value` adoption.

---

## 38. Error / failure taxonomy

Typed errors exist for: authority denial, policy denial, HITL required, idempotency conflict, revision conflict, qualification conflicts, integration configuration.

**Concern:** Broad `except Exception` in observability export paths (`export_policy.py`, `export_routing.py`) — diagnostics failure must not alter business truth; appears isolated to telemetry (P2).

---

## 39. Retry / resource lifecycle

Retry at correct layers: LLM transport, execution recovery, vendor worker. Tool idempotency prevents blind retry on uncertain external effects.

Provider clients: opened via Integrations materialization; shutdown hooks in plugin bootstrap.

---

## 40. Multi-tenancy

CW repos enforce `tenant_id` + `workspace_id` on all operations. Functional evidence persistence tests tenant isolation (D1-R1-D).

**P2:** Global qualification runs may be platform-global by design — documented.

---

## 41. Security boundary map

```text
UNTRUSTED: user input, vendor responses, plugin packages
  → validation (contracts, pydantic)
  → authentication (host API key / identity)
  → principal resolution (CW) [trusted after auth]
  → authority grants (repository reload, fail-closed)
  → policy evaluation (runtime + CW)
  → HITL grant (one-time)
  → side effect execution
  → evidence/diagnostics (secret-safe serialization)
```

**Weakest link:** generic tool execution path (`RuntimeToolInvoker`) when policy layers omitted (P1).

---

## 42. Data authority map

| Data | Category |
|------|----------|
| Membership, grants, delegation | AUTHORITATIVE (CW repos) |
| Provider catalog | AUTHORITATIVE (integration catalog) |
| registry_v2 snapshot | DERIVED |
| Trace / functional evidence | EVIDENCE |
| Memory / user profile | CONTEXTUAL |
| Activity | NOT IMPLEMENTED (would be PROJECTION) |
| Agent roster | AUTHORITATIVE for availability, not permissions |

---

## 43. Test architecture

| Class | Quality |
|-------|---------|
| Unit gates | STRONG — architecture gates for CW, integrations, plugins |
| Integration | ADEQUATE — skips when backend unavailable (honest) |
| System qualification | ADEQUATE — functional diagnostics suites |
| E2E CW | STRONG with harness |

**False proof risks:**
- In-memory repos labeled reference only — OK
- D1-R1 durability uses real subprocess + optional Mongo — GOOD
- `test_external_tool_plugin` failure — test not updated for execution identity requirement (P2 test drift)
- `test_harness_registry_authority_ac3` failures — missing `ollama` optional dep in env, not architecture defect (P3 env)

---

## 44. Test proof matrix

| Capability | Unit | Integration | Real backend | E2E | Concurrency | Restart | Security neg | Qualification |
|------------|------|-------------|--------------|-----|-------------|---------|--------------|---------------|
| Integration catalog | ✓ | ✓ | — | — | — | — | ✓ | — |
| registry_v2 projection | ✓ | — | — | — | — | — | — | — |
| Provider qualification | ✓ | ✓ | PG skip/Mongo ✓ | — | ✓ | ✓ reopen | ✓ | ✓ |
| CW enforcement | ✓ | PG skip | PG skip | ✓ | ✓ | — | ✓ | ✓ |
| Side-effect auth | ✓ | — | — | partial | — | — | ✓ | — |
| Tool side-effect safety | ✓ | — | system suite | — | ✓ | — | ✓ | — |
| Execution identity | ✓ | ✓ | — | — | — | — | ✓ | — |
| HITL / governed continuation | ✓ | — | — | — | — | — | ✓ | — |
| Functional qualification | ✓ | — | diagnostics suites | — | D1-R1 proc | ✓ | — | ✓ |
| Plugin external EP | ✓ | ✓ | — | ✓ | — | — | — | ✓ |
| Agent distribution | ✓ | — | — | — | lock | — | ✓ | — |
| Autonomous work | ✓ | — | in-mem only | — | ✓ | — | — | — |

---

## 45. Documentation truth

| Doc claim | Code truth | Gap |
|-----------|------------|-----|
| Integrations 186 cutover | Gate-tested | OK |
| CW MP-1 enterprise ready | Matches code | OK |
| AW "runtime not implemented" | AW-2B lifecycle + repos exist | **P2 — partial implementation** |
| Platform plugins roadmap complete | Cross-flow tests exist | OK |
| Activity not implemented | Confirmed | OK |
| LLM outside Integrations | Confirmed intentional | Needs ADR clarity |

---

## 46. Enterprise readiness scorecard

| Dimension | Score | Evidence |
|-----------|-------|----------|
| Architecture coherence | ADEQUATE | Clear tiers; dual side-effect paths |
| Modularity | STRONG | Tier boundaries enforced |
| Pluginability | ADEQUATE | Tools/skills/integrations strong; LLM partial |
| Provider abstraction | STRONG | Integrations canonical |
| Governance | ADEQUATE | CW strong; tool path gap |
| Security | ADEQUATE | No P0; P1 side-effect convergence |
| Persistence | STRONG | Domain/platform split clean |
| Concurrency | ADEQUATE | CAS where claimed |
| Observability | PARTIAL | Host adoption varies |
| Diagnostics | ADEQUATE | Durable proofs when env available |
| Evidence | ADEQUATE | Multiple distinct models |
| Test quality | ADEQUATE | Some env/drift failures |
| Real-backend proof | PARTIAL | PG unavailable in audit env |
| Documentation truth | PARTIAL | AW maturity statement |
| Application adoption | ADEQUATE | LKW flagship; CW durable not in LKW |
| Operational maturity | PARTIAL | Env-dependent proofs |

---

## 47. P0 findings

**None identified** in audited production code paths.

---

## 48. P1 findings

### P1 — FINDING-PLATFORM-SE-001: Generic tool execution side effects without mandatory authorization gate

| Field | Detail |
|-------|--------|
| **Files** | `intergrax/runtime/nexus/tools/invoker.py` |
| **Symbols** | `RuntimeToolInvoker._prepare_invocation`, `resolve_declarative_policy_enforcer` |
| **Behavior** | Side-effecting tools execute when declarative enforcer and scope policy are both absent |
| **Impact** | Unbounded external/DB mutation if host omits policy wiring |
| **Scenario** | Misconfigured host enables tool bundles with `side_effects=True` |
| **Reuse** | `MeaningfulSideEffectAuthorizationBoundary` or mandatory enforcer |
| **Fix** | Fail closed for `side_effects=True` without enforcer; or ADR-unified boundary |
| **ADR** | **DECIDED** — implementation pending ([`ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md`](../architecture/ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md)) |

---

## 49. P2 findings

| ID | Finding | Files | Recommendation |
|----|---------|-------|----------------|
| P2-001 | Dual side-effect models without documented precedence | policy + CW docs | **Closed by ADR** — multi-strategy model documented; Phase 1 fail-closed pending |
| P2-002 | LLM/embedding outside Integrations catalog | `llm_adapters/`, `rag/embedding/` | **IMPLEMENTATION_IN_PROGRESS** ([ADR_PLATFORM_LLM_EMBEDDING_INTEGRATION_BOUNDARY.md](../architecture/ADR_PLATFORM_LLM_EMBEDDING_INTEGRATION_BOUNDARY.md)): Option C hybrid — LLM Adapters remain dedicated domain; embedding `embedding_provider` category B1–B3 complete; **B4 legacy registry removal complete** (bound `EmbeddingProvider` only); P2-002-C (LLM registry decentralization) and final audit pending. |
| P2-003 | Reflective provider discovery at registration | `integrations/registry/` (removed `contract_capture.py`) | **CLOSED — final independent audit PASS** (P2-003-D @ `cb547a0fe4e1d09a2da7e32f5422558629486f49`): 200/200 explicit provider/category keys; zero production `contract_capture`/reflective contract discovery; Catalog authority + provider-owned `IntegrationContractSpec` + `registry_v2` derived projection only; typed-category fail-closed derives from `PROVIDER_CATEGORY_CONTRACT_REGISTRY`; P2-003 suite 457 passed. |
| P2-004 | AW docs say "not implemented" but AW-2B code exists | `autonomous_work/`, `AUTONOMOUS_WORK.md` | Align maturity statement |
| P2-005 | LKW lacks durable CW persistence | LKW host | Planned adoption |
| P2-006 | CW gate missing `side_effect_scope_id` correlation | `enforcement_gate.py` | Contract evolution |
| P2-007 | OpenAI SDK in tool service layer | `tools/providers/openai_vector_store/service.py` | **CLOSED — final independent audit PASS** (audited state @ `74f59733df0f0ae38ba88e0a656fd3b4c0640844`): `ManagedRetrievalBackend` canonical in Integrations catalog (`IntegrationCategory.MANAGED_RETRIEVAL`); OpenAI SDK isolated in provider adapter; special `materialization.py` provider selector removed; Integration Catalog authority; provider-owned explicit `IntegrationContractSpec`; `registry_v2` derived projection only; pluginability confirmed (second provider test with explicit `contract_specs`); focused Managed Retrieval tests 24 passed; P2-003 regression 457 passed. |
| P2-008 | Functional qualification scope creep risk | `functional_qualification_registry.py` | Keep explicit DI |
| P2-009 | Execution identity tightening broke unit test | `trace_bridge.py`, `test_external_tool_plugin.py` | Update test helper |
| P2-010 | RAG legacy `legacy/rag_answers/` | `intergrax/legacy/` | Migration plan |
| P2-011 | VK registry overlap with integrations | `vendor_knowledge/registry.py` | Document boundary |

---

## 50. P3 findings

- Harness registry authority tests fail without optional `ollama` dep (environment)
- Nexus LEG-1 legacy boolean tool plan flags
- Documentation verbosity / dual path naming cleanup
- `registry_v2` backward-compatible alias retention

---

## 51. ARCHITECTURAL_DECISION_REQUIRED

### ADR-PLATFORM-SE-CONVERGENCE — **DECIDED** (implementation pending)

**Decision record:** [`docs/project/maintainers/architecture/ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md`](../architecture/ADR_PLATFORM_MEANINGFUL_SIDE_EFFECT_AUTHORIZATION.md)

**Decision summary:** Multi-strategy, fail-closed model. Canonical invariant: **no authorization path ⇒ no meaningful side effect.** Strategy A (`DECLARATIVE_TOOL_AUTHORIZATION`) for generic tool-executing hosts/runtime; Strategy B (`COLLABORATIVE_WORK_AUTHORIZATION`) for workspace/resource-scoped mutations and External Work. Shared coordination contract concept only — no third policy engine. Phase 1: PLATFORM-SE-FAIL-CLOSED-1 closes the generic tool execution authorization gap at `RuntimeToolInvoker`.

**Rejected:** (a) CW-only for all effects; (b) declarative-only replacing CW; (c) optional policy behavior; (d) universal policy engine.

**P1 FINDING-PLATFORM-SE-001: IMPLEMENTED_PENDING_INDEPENDENT_AUDIT** (PLATFORM-SE-FAIL-CLOSED-1 shipped; independent SHA audit pending).

### ADR-PLATFORM-LLM-INTEGRATIONS

**Question:** Should LLM/embedding providers migrate into Integrations catalog?

**Recommendation:** Document intentional separation (`LLM_ADAPTERS.md` authority) OR plan phased migration for embedding only.

---

## 52. Required roadmap before release

1. ~~**Decide** side-effect convergence (ADR-PLATFORM-SE-CONVERGENCE)~~ — **DONE** (ADR accepted 2026-09-03)
2. **Implement** PLATFORM-SE-FAIL-CLOSED-1 — fail-closed for `side_effects=True` without recognized authorization strategy
3. **Align** AUTONOMOUS_WORK.md with AW-2B code maturity
4. **Adopt** durable CW in LKW (when MP-2 scheduled)
5. **Complete** D1-R1 durability qualification commit + Mongo env CI
6. **Migrate** OpenAI vector store tool to integration adapter boundary
7. **Update** broken unit tests for execution identity contract
8. **Explicit** contract specs to reduce `contract_capture` reflection

---

## 53. Tests executed

```text
# Focused unit (audit session)
uv run pytest \
  tests/unit/runtime/integrations/test_canonical_registry_projection.py \
  tests/unit/runtime/integrations/test_contract_registry_v2.py \
  tests/unit/integrations/test_registry.py \
  tests/unit/core/qualification/test_functional_qualification_registry.py \
  tests/unit/core/qualification/test_functional_qualification_attempts.py \
  tests/unit/runtime/policy/test_meaningful_side_effect_authorization.py \
  tests/unit/runtime/policy/test_g5c2b2b_governed_side_effect_reauthorization.py \
  tests/unit/runtime/tools/test_fresh_side_effect_authorization.py \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py \
  tests/unit/autonomous_work/test_repository_architecture_gates.py \
  tests/unit/autonomous_work/test_lifecycle_service.py \
  tests/unit/runtime/execution/test_agentic_tool_execution_identity.py \
  tests/unit/runtime/human/test_declarative_hitl_grant.py \
  tests/unit/runtime/plugins/test_plugin_bootstrap.py \
  tests/unit/tools/test_external_tool_plugin.py \
  tests/unit/skills/test_external_skill_plugin.py
# Result: 156 passed, 1 failed (test_external_tool_plugin — execution identity)

uv run pytest tests/unit/collaborative_work/ tests/unit/agent_distribution/ \
  tests/unit/core/qualification/ tests/unit/runtime/policy/test_meaningful_side_effect_policy.py \
  tests/unit/runtime/policy/test_pg_fix_b_side_effect_policy_precedence.py \
  tests/unit/applications/test_production_registry_authority_guards_ac3.py \
  tests/unit/applications/test_harness_registry_authority_ac3.py
# Result: 1053 passed, 11 failed (harness_registry — missing ollama optional dep)

uv run pytest tests/integration/core/qualification/test_provider_qualification_execution_postgresql.py \
  tests/integration/core/qualification/test_provider_qualification_discovery_mongo.py \
  tests/integration/collaborative_work/test_postgresql_repository.py \
  tests/unit/collaborative_work/test_postgresql_platform_reuse.py
# Result: 7 passed, 15 skipped (PostgreSQL unavailable in audit environment)
```

**Log artifacts:** `.tmp/session/platform-enterprise-audit-1/`

---

## 54. Files touched

| Kind | Path |
|------|------|
| **Docs** | `docs/project/maintainers/audits/PLATFORM_WIDE_ENTERPRISE_AUDIT.md` |
| **Production** | None (audit-only) |
| **Tests** | None (audit-only) |

---

## 55. Acceptance gate

| Criterion | Status |
|-----------|--------|
| Platform capability map | ✓ |
| Canonical ownership | ✓ |
| Duplicate mechanisms | ✓ |
| Pluginability | ✓ |
| Provider boundaries | ✓ |
| Execution/governance | ✓ |
| Meaningful side-effect paths | ✓ |
| Persistence boundaries | ✓ |
| Diagnostics/observability | ✓ |
| Qualification compared | ✓ |
| Functional qualification | ✓ |
| Identity/traceability | ✓ |
| Test proof quality | ✓ |
| Docs claims checked | ✓ |
| P0/P1/P2 concrete | ✓ |
| Product truth | ✓ |
| Release roadmap | ✓ |

**Status:** **READY_FOR_REVIEW**

---

## 56. Result classification

**IMPLEMENTATION_PENDING**

Side-effect authorization convergence is **decided** (ADR-PLATFORM-SE-CONVERGENCE). Primary remaining platform blocker for enterprise side-effect safety is **PLATFORM-SE-FAIL-CLOSED-1** implementation. LLM/Integrations boundary ADR still required separately.

---

*End of PLATFORM-ENTERPRISE-AUDIT-1*
