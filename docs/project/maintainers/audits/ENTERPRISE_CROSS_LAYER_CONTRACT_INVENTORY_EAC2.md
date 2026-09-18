# EAC-2 — Enterprise Cross-Layer Contract Inventory

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-2 — Cross-Layer Contract Inventory  
**Type:** Read-only contract / boundary audit (no remediation)  
**Authority:** EAC-0 R1 + EAC-1R3 (`ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md`)

### Provenance

| Field | SHA | Meaning |
|-------|-----|---------|
| **EAC1_CLOSE_COMMIT** (independent EAC-1 close @ operator pin) | `588180053c908c5960d5bca099b14e7b217849bb` | Last EAC-1 matrix close referenced for drift gate |
| **EAC2_SESSION_START_HEAD** | `e1cdaa77e0b1bf7608573001b23a42ebf870507f` | Session open (`HEAD == origin/development`) |
| **EAC2_EVIDENCE_HEAD** | `e1cdaa77e0b1bf7608573001b23a42ebf870507f` | Repository state reconciled immediately before EAC-2 documentation commit |

| Gate | Value |
|------|-------|
| **Branch** | `development` |
| **HEAD == origin/development @ session** | **YES** (`e1cdaa77e0b1bf7608573001b23a42ebf870507f`) |
| **EAC-1 ancestor contained** | **YES** (`588180053…` ⊆ HEAD) |
| **Upstream** | [`ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md`](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md), [`ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md`](ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md) |

**Drift since `EAC1_CLOSE_COMMIT` (contract-relevant commits inspected):**

| Commit | Area | EAC-2 effect |
|--------|------|----------------|
| `4403f1b5d` | UCL `UclReferenceReadPort` | New **STATE_REFERENCE_CONTRACT** row (EAC-CON-028) |
| `162ba778f` | UCL workspace ownership hardening | Scope baseline for CE↔UCL (no new public family) |
| `0ba1a514c` | Governance pre-model policy | GE admission path evidence refresh |
| `e1cdaa77e` | Memory provider qualification docs | Qualification only; contracts unchanged |

**Out of scope:** uncommitted working-tree deltas (`intergrax/context/policy/budget_allocator.py`, CE-02 tests) — **not** evidence for this artifact.

**Method:** grep/import sampling @ `EAC2_EVIDENCE_HEAD`, `intergrax/contracts/` module census (**601** Python modules), EAC-1 §11 contract baseline, targeted reads of execution/orchestration, agent surface, memory/RAG/UCL reference ports, Tier-3/hosting manifests.

---

## 1. Scope

EAC-2 inventories **legal cross-layer contracts** (and explicit **BYPASS** / **LEGACY** paths) between the **34** canonical DOMAIN rows from EAC-0/EAC-1. For each contract family: semantic owner, physical location, consumers, replaceability, quality rating, and finding class.

**In scope:** contract taxonomy, ownership vs EAC-1 authorities, public/internal classification, duplication/versioning/pluginability baselines, CL-EAC1-001…005 contract mapping, EAC-3 dependency-direction inputs.

**Out of scope:** remediation, contract moves, import refactors, ADR resolution, new production contracts.

---

## 2. Contract taxonomy (Phase 1)

| Type | Definition |
|------|------------|
| **DOMAIN_CONTRACT** | Canonical public semantic contract owned by one domain |
| **PORT** | Consumer/provider boundary for replaceable implementation |
| **STRATEGY_SPI** | Replaceable strategy behind platform contract |
| **PROVIDER_SPI** | Vendor/provider abstraction boundary |
| **EVENT_CONTRACT** | Cross-layer immutable event/fact |
| **COMMAND_CONTRACT** | Typed request to authority owner |
| **RESULT_CONTRACT** | Typed response/result DTO |
| **STATE_REFERENCE_CONTRACT** | Typed reference to authoritative state without copying ownership |
| **COMPOSITION_CONTRACT** | Host/composition wiring contract |
| **INTERNAL_CONTRACT** | Legal only inside one domain or subordinate implementation |
| **LEGACY_CONTRACT** | Historical API still present; not canonical |
| **BYPASS** | Cross-layer interaction without canonical contract |

---

## 3. Contract quality model (Phase 2)

Dimensions (each rated **PASS** / **PARTIAL** / **FAIL** / **N/A** in inventory notes):

`OWNER_DEFINED` · `TYPED` · `VERSIONABLE` · `STABLE` · `IMPLEMENTATION_NEUTRAL` · `VENDOR_NEUTRAL` · `PLUGINABLE` · `SCOPE_AWARE` · `IDENTITY_AWARE` · `FAIL_CLOSED` (where required) · `NO_INTERNAL_TYPE_LEAKAGE` · `NO_CONCRETE_IMPLEMENTATION_LEAKAGE` · `NO_FOREIGN_DOMAIN_OWNERSHIP_LEAKAGE` · `NO_HIDDEN_GLOBAL_STATE`

---

## 4. Master contract inventory (Phase 3)

**Stable IDs** (`EAC-CON-###`) group **one semantic role** per family. Physical modules may span multiple files; IDs are not per-file.

| Contract ID | Contract / Family | Type | Semantic Owner | Defined In | Consumers | Providers / Implementers | Public/Internal | Pluginable | Versioned | Status | Findings |
|-------------|-------------------|------|----------------|------------|-----------|--------------------------|-----------------|------------|-----------|--------|----------|
| EAC-CON-001 | Execution identity (`RunId`, `AttemptId`, `ExecutionId`, UEA spine) | DOMAIN_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `intergrax/contracts/execution_identity*.py` | All runtime domains | EE mint authority | Public | N/A | PARTIAL (frozen IDs) | CANONICAL | — |
| EAC-CON-002 | Root execution admission | PORT | UNIFIED_EXECUTION_RUNTIME | `runtime_execution_admission.py` | T3, BG, AW, Hosting | EE admission service | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-003 | Execution intake / launch | PORT | UNIFIED_EXECUTION_RUNTIME | `execution_intake.py`, `root_execution_launch.py` | Strategies, T3 | EE runtime | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-004 | Execution continuation & checkpoint | PORT + COMMAND | UNIFIED_EXECUTION_RUNTIME | `execution_continuation.py`, NPSC-5E ports | Reliability, ERL, Recovery | Checkpoint providers | Public | Yes | PARTIAL | CANONICAL | ERL must not mutate tree (**PASS** @ canon) |
| EAC-CON-005 | Execution terminal persistence | PORT | UNIFIED_EXECUTION_RUNTIME | `execution_terminal.py` | Observability, Diagnostics | EE terminal store | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-006 | Orchestration topology submission/continuation | PORT | ORCHESTRATION | `orchestration_topology.py` | EE strategies | Orchestration backends | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-007 | Nexus orchestration backend (`NexusOrchestrationPort`) | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `runtime/execution/orchestration.py` | EE only | `NexusLoop` | **Internal** | No | N/A | CANONICAL_INTERNAL | Consumers must use EE-CON-003/006 |
| EAC-CON-008 | Nexus graph engine surface | BYPASS risk | NEXUS_EXECUTION_FLOW | `runtime/nexus/nexus_loop.py`, `GraphExecutor` | **Illegal:** Tier-2 agents, reference harness | Nexus | **Internal** | No | N/A | **BYPASS** | **CL-EAC1-001**, EAC2-F-001 |
| EAC-CON-009 | Decision lifecycle & events | DOMAIN_CONTRACT | DECISION_SYSTEM | `intergrax/contracts/decision*` | EE, CW (read) | Decision engine | Public | Strategy SPI | PARTIAL | CANONICAL | LEGACY critic path EAC-CON-082 |
| EAC-CON-010 | Decision authorization evaluator | PORT | DECISION_SYSTEM | `decision_authorization.py` | EE, GE (read) | Decision strategies | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-011 | Governance admission / policy | DOMAIN_CONTRACT | GOVERNED_EXECUTION | `runtime_execution_policy_admission.py`, `agent_runtime_governance.py` | EE, Tools, Agents | Policy engine | Public | Policy plugins | PARTIAL | CANONICAL | **CL-EAC1-004** duplicate WHETHER |
| EAC-CON-012 | Governance evidence persistence | PORT | GOVERNED_EXECUTION | `governed_execution_governance_evidence.py` | Observability | GE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-013 | CW workspace / ContextView | DOMAIN_CONTRACT | COLLABORATIVE_WORK | `collaborative_work.py`, CW apps contracts | CE, Agents | CW stores | Public | PARTIAL | PARTIAL | CANONICAL | **ADR-GOV-01** overlap with EAC-CON-011 |
| EAC-CON-014 | Multi-agent / physical delegation governance | DOMAIN_CONTRACT | GOVERNED_EXECUTION + CW | `multi_agent_coordination_governance.py`, `physical_delegation_governance.py` | EE, CW | Evaluators | Public | Yes | PARTIAL | CANONICAL | **CONTRACT-DUPLICATE** candidate vs EAC-CON-011 |
| EAC-CON-015 | Tool invocation spine | DOMAIN_CONTRACT | TOOLS | `intergrax/contracts/tools*` | Agents, Nexus (internal), Integrations | Tool drivers | Public | Provider SPI | PARTIAL | CANONICAL | — |
| EAC-CON-016 | Provider invocation dispatch / reliability | PORT | TOOLS + ERL | `provider_invocation_dispatch.py`, ERL runtime ports | ERL, Integrations | Providers | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-017 | Agent contract & step loop | DOMAIN_CONTRACT | AGENT_CONTRACTS_AND_ASSEMBLY | `agent_contract.py`, `contracts/agent_*` | EE, Nexus | Agent impls | Public | Yes | PARTIAL | CANONICAL | **INTERNAL-LEAK** via Nexus types EAC2-F-002 |
| EAC-CON-018 | UAEP protocol | DOMAIN_CONTRACT | AGENT_CONTRACTS_AND_ASSEMBLY | `uaep_protocol.py`, `contracts/uaep*` | EE, harness | Agents | Public | Yes | PARTIAL | CANONICAL | Imports `RuntimeContext` in executor |
| EAC-CON-019 | Runtime execution context (contract) | DOMAIN_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `runtime_execution_context.py` | Agents, UAEP | EE | Public | N/A | PARTIAL | CANONICAL | Preferred over Nexus `RuntimeContext` |
| EAC-CON-020 | Memory provider stores | PROVIDER_SPI | MEMORY | `intergrax/memory/contracts/*` | T3 wiring, RAG | Store plugins | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-021 | Memory reference read | STATE_REFERENCE_CONTRACT | MEMORY | `memory/contracts/memory_reference_read.py` | CE, ContextView adapters | Memory readers | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-022 | Memory runtime read (inspection) | PORT | MEMORY | `memory_runtime_read.py` | Observability inspection | Memory | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-023 | Knowledge reference read | STATE_REFERENCE_CONTRACT | RAG | `knowledge/contracts/knowledge_reference_read.py` | CE adapters | RAG stack | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-024 | ContextView composition | DOMAIN_CONTRACT | CONTEXT_ENGINEERING | `context_view_composition.py`, `intergrax/context/` | Agent, Nexus consumer | CE composers | Public | Strategy SPI | PARTIAL | CANONICAL | **CL-EAC1-003** impl locus in Nexus |
| EAC-CON-025 | Context source ports (memory/knowledge/UCL/CW) | PORT | CONTEXT_ENGINEERING | `context_view_source_ports.py` | CE assembly | Domain readers | Public | Yes | PARTIAL | CANONICAL | Aligns MP-5F-B1/B2/B3 |
| EAC-CON-026 | ContextView visibility policy | DOMAIN_CONTRACT | CONTEXT_ENGINEERING | `context_view_visibility_policy.py` | CE, CW | Policy plugins | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-027 | UCL lifecycle contracts | DOMAIN_CONTRACT | UNIFIED_CONTEXT_LIFECYCLE | `intergrax/ucl/contracts/*` | CE, Nexus (consumer) | UCL optimizer | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-028 | UCL reference read | STATE_REFERENCE_CONTRACT | UNIFIED_CONTEXT_LIFECYCLE | `ucl/contracts/ucl_reference_read.py` | CE adapters | UCL store | Public | Yes | PARTIAL | CANONICAL | Workspace scope **PARTIAL** @ B3 docstring |
| EAC-CON-029 | RuntimeEvent / evidence | EVENT_CONTRACT | OBSERVABILITY | `runtime_event.py`, `contracts/observability*` | Diagnostics, export | EE emitters | Public | Sink SPI | PARTIAL | CANONICAL | — |
| EAC-CON-030 | Event delivery / export | PORT | OBSERVABILITY | `event_delivery.py`, `observability_export.py` | T3 wiring, OTLP | Transports | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-031 | Diagnostic Problem | DOMAIN_CONTRACT | DIAGNOSTICS | `contracts/diagnostics/problem_record.py` | Operators | DIAG jobs | Public | Detector SPI | PARTIAL | CANONICAL | — |
| EAC-CON-032 | Diagnostic evidence contributors | PORT | DIAGNOSTICS | `diagnostic_evidence_contributor.py` | DIAG | Domains | Public | Yes | PARTIAL | CANONICAL | Must not mint RuntimeEvent |
| EAC-CON-033 | Terminal execution diagnostic port | PORT | DIAGNOSTICS | `diagnostics/terminal_execution_diagnostic_port.py` | EE (read) | DIAG | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-034 | Reliability HITL / human decision | DOMAIN_CONTRACT | RELIABILITY_FAILURE_AND_HITL | HITL contracts, `human_decision*` | EE | Reliability | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-035 | Recovery admission | PORT | RELIABILITY_FAILURE_AND_HITL | `recovery_admission.py` | EE | Reliability | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-036 | ERL effect / reconciliation | DOMAIN_CONTRACT | ENTERPRISE_RELIABILITY_LAYER | `contracts/enterprise_reliability/*` | EE, GE, Tools | ERL orchestrator | Public | GovernanceStrategy SPI | PARTIAL | EMERGING | Not lifecycle owner |
| EAC-CON-037 | ERL plugin SPI | STRATEGY_SPI | ENTERPRISE_RELIABILITY_LAYER | `enterprise_reliability/plugin_spi.py` | ERL host | Plugins | Public | Yes | PARTIAL | CANONICAL | ≠ platform GE authority |
| EAC-CON-038 | Capability catalog read | DOMAIN_CONTRACT | CAPABILITY_CATALOG_AND_DISCOVERY | `contracts/capability_catalog/*` | Marketplace, T3 | Domain sources | Public | Yes | PARTIAL | CANONICAL | Read-only |
| EAC-CON-039 | Agent distribution / RuntimeRevision | DOMAIN_CONTRACT | AGENT_DISTRIBUTION | `contracts/agent_distribution/*` | Catalog, EE | Installer | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-040 | Marketplace listing / acquisition | DOMAIN_CONTRACT | CAPABILITY_CATALOG (discovery) + product | `contracts/marketplace/*` | Distribution handoff | Marketplace | Public | PARTIAL | PARTIAL | CANONICAL | ADR-MKT-01 aligned |
| EAC-CON-041 | Tool/skill marketplace handoff ports | PORT | TOOLS / SKILLS | `tools/marketplace_lifecycle_handoff.py`, `skills/...` | Distribution | Marketplace | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-042 | Platform plugin lifecycle | DOMAIN_CONTRACT | PLATFORM_PLUGINS | `core/plugins/*` | All domains | EP loader | Public | Yes | PARTIAL | CANONICAL | EP not semantic owner |
| EAC-CON-043 | Tier-3 application manifest | COMPOSITION_CONTRACT | TIER3_APPLICATION_ENVIRONMENT | `applications/contracts/manifest.py` | Hosting, EE wiring | T3 apps | Public | PARTIAL | PARTIAL | CANONICAL | **CL-EAC1-005** |
| EAC-CON-044 | Application environment profile | COMPOSITION_CONTRACT | TIER3_APPLICATION_ENVIRONMENT | `applications/contracts/environment_profile/*` | Hosting | T3 | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-045 | Hosted application lifecycle | DOMAIN_CONTRACT | APPLICATION_HOSTING | `hosting/contracts/lifecycle.py` | T3 host | Hosting runtime | Public | PARTIAL | PARTIAL | CANONICAL | **CL-EAC1-005** |
| EAC-CON-046 | Hosted application context ports | PORT | APPLICATION_HOSTING | `hosting/contracts/context.py` | T3 adapters | Host | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-047 | LLM provider adapter | PROVIDER_SPI | LLM_ADAPTERS | `llm_adapters/*` | Reasoning, Agents | Vendors | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-048 | Integration provider contracts | PROVIDER_SPI | INTEGRATIONS | `integrations/contracts/*` | Tools, Memory, T3 | Vendors | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-049 | Skill composition | DOMAIN_CONTRACT | SKILLS | `contracts/skills/*` | Agents | Skill hosts | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-050 | Autonomous work execution dispatch | COMMAND_CONTRACT | AUTONOMOUS_WORK | `contracts/autonomous_work/*` | EE | AW workers | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-051 | Background task queue | DOMAIN_CONTRACT | BACKGROUND_TASKS | BG contracts | EE | Brokers | Public | Provider SPI | PARTIAL | CANONICAL | — |
| EAC-CON-052 | Elastic capacity admission | PORT | ELASTIC_CAPACITY_AND_SCALING | `execution_capacity_admission.py` | EE | Capacity service | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-053 | Modality ingestion | PROVIDER_SPI | MODALITY | modality contracts | CE | Adapters | Public | Yes | UNKNOWN | PARTIAL | — |
| EAC-CON-054 | Reasoning material (non-decision) | DOMAIN_CONTRACT | REASONING_AND_COGNITION | reasoning contracts | Decision | Strategies | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-055 | CodeCraft workspace | DOMAIN_CONTRACT | CODE_CRAFT | codecraft contracts | Agents | CC services | Internal/Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-056 | Proof receipts | DOMAIN_CONTRACT | PROOF_RECEIPTS | proof contracts | CI | Gates | Public | N/A | PARTIAL | CANONICAL | DX only |
| EAC-CON-057 | AHI research hooks | LEGACY_CONTRACT | ADAPTIVE_HARNESS_INTELLIGENCE | immature AHI modules | Research | — | Internal | No | N/A | LEGACY | Non-prod |
| EAC-CON-058 | Self-healing / preventive (platform extension) | DOMAIN_CONTRACT | RELIABILITY + GE adjunct | `contracts/self_healing/*`, `preventive/*` | EE | Plugins | Public | Yes | PARTIAL | CANONICAL | Scoped to healing, not GE root |
| EAC-CON-059 | Runtime inspection read ports | PORT | OBSERVABILITY + domains | `tool_runtime_read.py`, `model_runtime_read.py` | DIAG, operators | Runtime | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-060 | Causal / platform evidence | EVENT_CONTRACT | OBSERVABILITY | `platform_causal_evidence.py` | DIAG, ERL | EE | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-061 | Execution failure evidence | EVENT_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `execution_failure_evidence.py` | Observability | EE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-062 | Sandbox / isolation authority | DOMAIN_CONTRACT | GOVERNED_EXECUTION + EE | `runtime_sandbox_isolation_authority.py` | UAEP, Tools | EE | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-063 | Session persistence (Nexus) | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `runtime/nexus/session/*` | Nexus, **T3 memory_wiring** | SessionStorage impls | **Internal** | Yes | N/A | **BYPASS** | **CL-EAC1-002** |
| EAC-CON-064 | Nexus runtime request/answer | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `responses/response_schema.py` | Agents (**leak**) | Nexus | **Internal** | No | N/A | **BYPASS** | **CL-EAC1-001** |
| EAC-CON-065 | Nexus runtime context | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `engine/runtime_context.py` | Agents (**leak**) | Nexus | **Internal** | No | N/A | **BYPASS** | **CL-EAC1-001** |
| EAC-CON-066 | Diagnostic payload (trace) | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `tracing/trace_models.py` | Agents diagnostics | Nexus | **Internal** | No | N/A | LEGACY_SHIM | Prefer contract diagnostics |
| EAC-CON-067 | Tier-3 memory wiring composition | COMPOSITION_CONTRACT | TIER3_APPLICATION_ENVIRONMENT | `applications/_shared/memory_wiring.py` | T3 hosts | Memory plugins + **SessionManager** | Public wiring | PARTIAL | N/A | **PARTIAL** | **CONCRETE-COUPLING** EAC2-F-003 |
| EAC-CON-068 | Distributed KV (supporting) | PROVIDER_SPI | PLATFORM_FOUNDATION adjunct | `distributed/contracts/kv_store.py` | Multiple | Redis/SQLite | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-069 | External work / operations | COMMAND_CONTRACT | INTEGRATIONS + TOOLS | `external_work.py`, `external_operations/*` | ERL, Tools | Providers | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-070 | Delegated execution family | PORT | UNIFIED_EXECUTION_RUNTIME | `delegated_execution_*.py` | AW, providers | EE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-071 | Step LLM router port | PORT | AGENT_CONTRACTS_AND_ASSEMBLY | `step_llm_router_port.py` | ACP authoring | LLM adapters | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-072 | Budget / agent budget slice | DOMAIN_CONTRACT | CONTEXT_ENGINEERING + EE | `agent_budget.py`, CE budget | Agents, T3 | EE ledger | Public | Hook SPI | PARTIAL | CANONICAL | CE-02 qualification |
| EAC-CON-073 | Runtime invariant rules | STRATEGY_SPI | PLATFORM_FOUNDATION | `runtime_invariants.py` | EE | Rule packs | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-074 | Forecast / predictive analyzers | STRATEGY_SPI | DIAGNOSTICS adjunct | `predictive*.py`, `statistical_forecast_analyzer.py` | DIAG | Plugins | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-075 | Semantic verification judges | STRATEGY_SPI | DECISION_SYSTEM adjunct | `semantic_verification.py` | Decision qual | Judges | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-076 | Execution environment isolation view | STATE_REFERENCE_CONTRACT | TIER3 + EE | `execution_environment_isolation.py` | EE, sandbox | T3 profiles | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-077 | Compensation side-effect execution | COMMAND_CONTRACT | GOVERNED_EXECUTION | `compensation_side_effect_execution.py` | ERL, Tools | Tools | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-078 | Control plane mutation policy | DOMAIN_CONTRACT | GOVERNED_EXECUTION | `control_plane_mutation.py` | Ops automation | GE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-079 | Marketplace query context | RESULT_CONTRACT | CAPABILITY_CATALOG | `marketplace/query_context.py` | Marketplace UI | Catalog | Public | N/A | PARTIAL | CANONICAL | — |
| EAC-CON-080 | Agent marketplace lifecycle handoff | PORT | AGENT_DISTRIBUTION | `agent_distribution/marketplace_lifecycle_handoff.py` | Marketplace | Distribution | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-081 | Runtime event history buffer | PORT | OBSERVABILITY | `runtime_event_history.py` | EE strategies | Obs | Public | Strategy SPI | PARTIAL | CANONICAL | — |
| EAC-CON-082 | Legacy decision critic runtime | LEGACY_CONTRACT | DECISION_SYSTEM | `runtime/critic/*` | EE (legacy) | Critic | Internal | No | N/A | LEGACY | LEGACY-EAC1-001 |
| EAC-CON-083 | Harness reference / ACP patterns | COMPOSITION_CONTRACT | AGENT_CONTRACTS_AND_ASSEMBLY | `reference_harness.py`, `authoring/patterns/*` | DX, qual | **SessionManager** | Public DX | No | N/A | **BYPASS** | EAC2-F-004 NOT_REPLACEABLE for external agents |
| EAC-CON-084 | Catalog declarative invoker | INTERNAL_ADAPTER | AGENT_CONTRACTS_AND_ASSEMBLY | `catalog_declarative_invoker.py` | Qual | Nexus tools | Internal | No | N/A | CONCRETE-COUPLING | Test/qual path |
| EAC-CON-085 | Nexus context engine / compiler | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `runtime/nexus/context/*` | Nexus | CE algorithms (impl) | **Internal** | No | N/A | INTERNAL | **CL-EAC1-003** |
| EAC-CON-086 | Observability → Diagnostics adapter slice | PORT | DIAGNOSTICS | diagnostic read models consuming Obs | DIAG | Obs store adapters | Public | Yes | PARTIAL | CANONICAL | DOC ADR-REG-003 plan topology |
| EAC-CON-087 | ERL governance decision material | RESULT_CONTRACT | ENTERPRISE_RELIABILITY_LAYER | `enterprise_reliability/governance_decision.py` | GE (read) | ERL | Public | PARTIAL | PARTIAL | CANONICAL | Not GE authority transfer |
| EAC-CON-088 | Governed continuation | COMMAND_CONTRACT | GOVERNED_EXECUTION | `governed_continuation.py` | EE | GE | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-089 | Retry budget port | PORT | RELIABILITY_FAILURE_AND_HITL | `retry_budget.py` | Tools, providers | Reliability | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-090 | Execution reconstruction reader | PORT | OBSERVABILITY | `execution_reconstruction.py` | DIAG, operators | Obs | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-091 | Plugin memory store discovery | PORT | PLATFORM_PLUGINS → MEMORY | `core/plugins`, `memory/resolver` | T3 | Plugins | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-092 | Hosting service registry | COMPOSITION_CONTRACT | APPLICATION_HOSTING | `hosting/services.py` | T3 host apps | Hosting | Public | PARTIAL | PARTIAL | CANONICAL | **CL-EAC1-005** |

**Inventory metrics (@ grouped families):**

| Metric | Count |
|--------|------:|
| Total contract families (EAC-CON rows) | **92** |
| Public cross-layer contracts | **68** |
| Internal-only / subordinate contracts | **14** |
| Strategy SPI families | **12** |
| Provider SPI families | **18** |
| Event contract families | **6** |
| Legacy contract families | **3** |
| Documented bypass / leak paths | **8** |

---

## 5. Per-domain owned / consumed contracts (Phase 4)

Compact register — all **34** domains audited. Empty **Owned** is acceptable.

| Domain | Public contracts owned (IDs) | Key consumed contracts | Ports provided | Ports consumed | Known bypasses |
|--------|------------------------------|------------------------|----------------|----------------|----------------|
| PLATFORM_FOUNDATION | EAC-CON-073 | All (meta) | — | — | — |
| UNIFIED_EXECUTION_RUNTIME | EAC-CON-001…005, 019, 061, 070 | EAC-CON-011, 006, 029 | Admission, continuation | Governance, Obs | — |
| ORCHESTRATION | EAC-CON-006 | EAC-CON-003, 007 | Topology ports | EE admission | — |
| NEXUS_EXECUTION_FLOW | EAC-CON-007, 008, 063…066, 085 | EE context, tool/agent contracts | **None public** | Tools, Agents (internal) | **008, 063–066** |
| DECISION_SYSTEM | EAC-CON-009, 010, 075 | EAC-CON-029, 054 | Decision SPI | EE host | EAC-CON-082 legacy |
| GOVERNED_EXECUTION | EAC-CON-011, 012, 014, 062, 077, 078, 088 | EAC-CON-001, 015 | Policy evaluators | EE, Tools | **CL-EAC1-004** |
| REASONING_AND_COGNITION | EAC-CON-054 | EAC-CON-024, 047 | Reasoning SPI | LLM | — |
| AGENT_CONTRACTS_AND_ASSEMBLY | EAC-CON-017, 018, 071, 083 | EAC-CON-011, 024, 015, **064–065** | Agent SPI | **Nexus leak** | **001, 083** |
| AGENT_DISTRIBUTION | EAC-CON-039, 080 | EAC-CON-038, 002 | Install ports | Catalog | — |
| LLM_ADAPTERS | EAC-CON-047 | — | LLM backends | Vendor APIs | — |
| TOOLS | EAC-CON-015, 041 | EAC-CON-011, 016 | Tool drivers | GE, Integrations | — |
| SKILLS | EAC-CON-049, 041 | EAC-CON-015 | Skill hosts | Tools | — |
| INTEGRATIONS | EAC-CON-048, 069 | EAC-CON-015 | Provider backends | Credentials | — |
| RAG | EAC-CON-023 | EAC-CON-021, 048 | Retrieval | Memory refs | — |
| MEMORY | EAC-CON-020, 021, 022 | EAC-CON-091 | Store plugins | Obs (audit) | **067 wiring** |
| CONTEXT_ENGINEERING | EAC-CON-024, 025, 026, 072 | EAC-CON-021, 023, 028, 013 | Composers | Memory/RAG/UCL/CW | **085 impl locus** |
| UNIFIED_CONTEXT_LIFECYCLE | EAC-CON-027, 028 | EAC-CON-024 | Optimizer | CE bundles | Scope gap B3 |
| MODALITY | EAC-CON-053 | EAC-CON-024 | Adapters | CE | — |
| OBSERVABILITY | EAC-CON-029, 030, 059, 060, 081, 090 | EAC-CON-061 | Export sinks | EE events | — |
| DIAGNOSTICS | EAC-CON-031, 032, 033, 074, 086 | EAC-CON-029, 090 | Detectors | Obs evidence | — |
| RELIABILITY_FAILURE_AND_HITL | EAC-CON-034, 035, 058, 089 | EAC-CON-004, 011 | Recovery/HITL | EE lifecycle | — |
| ADAPTIVE_HARNESS_INTELLIGENCE | EAC-CON-057 | Research telemetry | — | — | — |
| ELASTIC_CAPACITY_AND_SCALING | EAC-CON-052 | EAC-CON-002 | Capacity | EE | — |
| EXPERIMENTATION_AND_DX | — (guides) | Qual contracts | — | — | — |
| TIER3_APPLICATION_ENVIRONMENT | EAC-CON-043, 044, 067, 076 | EAC-CON-002, 045, 091 | App manifests | Hosting, EE | **067, 005** |
| APPLICATION_HOSTING | EAC-CON-045, 046, 092 | EAC-CON-043, 002 | Host lifecycle | T3 manifest | **005** |
| CODE_CRAFT | EAC-CON-055 | CW, Tools | — | — | — |
| AUTONOMOUS_WORK | EAC-CON-050 | EAC-CON-002, 070 | Worker dispatch | EE | — |
| COLLABORATIVE_WORK | EAC-CON-013 | EAC-CON-011, 009 | CW stores | GE (**overlap**) | **004** |
| BACKGROUND_TASKS | EAC-CON-051 | EAC-CON-002 | Queues | EE | — |
| CAPABILITY_CATALOG_AND_DISCOVERY | EAC-CON-038, 079 | Domain descriptors | Catalog read | Distribution | — |
| PROOF_RECEIPTS | EAC-CON-056 | — | — | — | — |
| PLATFORM_PLUGINS | EAC-CON-042, 091 | Domain validation | Plugin registry | Domains | Registry ≠ owner |
| ENTERPRISE_RELIABILITY_LAYER | EAC-CON-036, 037, 087 | EAC-CON-011, 004, 029 | Reconciliation SPI | GE, EE | — |

---

## 6. Public vs internal contract boundaries (Phase 4 / 16)

| Class | Count (families) | Examples |
|-------|------------------|----------|
| **CORRECT** | 52 | `MemoryReferenceReadPort`, `execution_continuation`, `runtime_event` |
| **NEUTRAL_OK** | 24 | `intergrax/contracts/*` shared package with explicit semantic owner column |
| **MISPLACED** | 4 | Nexus session/request types consumed as agent public API |
| **INTERNAL_LEAK** | 6 | `RuntimeContext`, `RuntimeRequest`, `SessionManager`, `NexusLoop` |
| **LEGACY** | 3 | `runtime/critic/*`, trace `DiagnosticPayload` shim, AHI immature |

---

## 7. Strategy / provider / plugin interfaces (Phases 13, 19)

| Mechanism | Contract ID | Owner | Replaceable? | Finding |
|-----------|-------------|-------|--------------|---------|
| DecisionStrategy | EAC-CON-009 | DECISION_SYSTEM | Yes | PASS |
| Memory store plugin | EAC-CON-020, 091 | MEMORY | Yes | PASS |
| Tool driver | EAC-CON-015 | TOOLS | Yes | PASS |
| GE policy plugin | EAC-CON-011 | GOVERNED_EXECUTION | Partial | QUAL-EAC1-001 |
| ERL GovernanceStrategy | EAC-CON-037 | ERL | Yes | ≠ GE authority |
| Problem detector | EAC-CON-031 | DIAGNOSTICS | Yes | PASS |
| Orchestration backend | EAC-CON-007 | NEXUS (internal) | EE-injected only | PASS internal |
| Agent public surface | EAC-CON-017 | AGENT_CONTRACTS | **No** (Nexus types) | **NOT_REPLACEABLE** EAC2-F-002 |
| EP platform plugin | EAC-CON-042 | PLATFORM_PLUGINS | Yes | PASS |
| CE ordering/identity strategies | EAC-CON-024 | CE | Yes | PASS |
| UCL optimizer | EAC-CON-027 | UCL | Partial | PASS |
| Integration provider | EAC-CON-048 | INTEGRATIONS | Yes | PASS |
| LLM provider | EAC-CON-047 | LLM_ADAPTERS | Yes | PASS |

---

## 8. Contract ownership mismatches (Phase 5)

| Finding ID | Class | Severity | Contract | Expected owner (EAC-1) | Observed | Verdict |
|------------|-------|----------|----------|------------------------|----------|---------|
| EAC2-F-001 | INTERNAL-LEAK | **HIGH** | EAC-CON-064, 065 | NEXUS internal | Tier-2 `agents/*`, `intergrax/agents/agent_contract.py` import Nexus types | **MISPLACED** |
| EAC2-F-002 | CONTRACT-NOT-PLUGINABLE | **HIGH** | EAC-CON-017, 018 | AGENT_CONTRACTS | External agent cannot implement without Nexus imports | **FAIL** replaceability |
| EAC2-F-003 | CONCRETE-COUPLING | **MEDIUM** | EAC-CON-067 | MEMORY + T3 composition | `memory_wiring.py` constructs `SessionManager` + Nexus storage | **BYPASS** vs EAC-CON-021 |
| EAC2-F-004 | CONTRACT-DUPLICATE | **HIGH** | EAC-CON-011 vs 013 vs 014 | GOVERNED_EXECUTION | Parallel WHETHER for tools/workspace/delegation | **ADR** CL-EAC1-004 |
| EAC2-F-005 | CONTRACT-SCOPE-GAP | **MEDIUM** | EAC-CON-028 | UCL | Workspace-bound UCL reads fail closed; not fully modeled | **PARTIAL** |
| EAC2-F-006 | CONTRACT-VERSIONING-GAP | **LOW** | Multiple `contracts/*` | Domain owners | Few explicit schema versions; frozen IDs compensate | **PARTIAL** |
| EAC2-F-007 | CONTRACT-LEGACY | **MEDIUM** | EAC-CON-082 | DECISION_SYSTEM | Critic runtime still wired | LEGACY-EAC1-001 |
| EAC2-F-008 | CONTRACT-MISSING | **MEDIUM** | Agent ↔ EE step context | AGENT_CONTRACTS / EE | No single owner-neutral substitute for `RuntimeRequest`/`RuntimeContext` exported to Tier-2 | **CL-EAC1-001** |

**Semantic owner rule verified:** file location in `intergrax/contracts/` does **not** override EAC-1 authority column (recorded per row).

---

## 9. Internal implementation leaks (Phase 6–7)

### Execution ↔ Nexus ↔ Orchestration

| Boundary | Canonical contract | Consumer imports | Leak? |
|----------|-------------------|------------------|-------|
| EE → Orchestration | EAC-CON-006 | `orchestration_topology.py` via EE | **No** |
| Orchestration → Nexus | EAC-CON-007 (`NexusOrchestrationPort`) | `orchestration.py` → `NexusLoop` | **Legal internal** |
| EE → Nexus (forbidden public) | — | Tier-2/Agents → `RuntimeContext`, `RuntimeRequest`, `SessionManager` | **Yes** EAC2-F-001 |
| Nexus public root | **None** @ EAC-1 | `NexusLoop` in notebooks/tests only | Internal OK |

**CL-EAC1-001 contract replacement needed (design only):** owner-neutral **AgentRuntimeInteractionPort** (or expand EAC-CON-019) carrying admission-scoped execution context + typed step envelope — **without** `intergrax.runtime.nexus.*` types. Until then, EAC-CON-064/065 function as **undeclared public API**.

### Agents ↔ Execution / Nexus

| Surface | Nexus import? | Public contract? | External agent viable? |
|---------|---------------|------------------|------------------------|
| `intergrax/agents/agent_contract.py` | `RuntimeContext`, `RuntimeRequest` | **Yes (de facto)** | **No** |
| `intergrax/agents/uaep.py` | `RuntimeContext`, Nexus tool gateway | **Yes** | **No** |
| `intergrax/contracts/runtime_execution_context.py` | No | **Yes** | **Yes** (partial) |
| `reference_harness.py` | `SessionManager`, full Nexus stack | DX / qual | **No** |
| Tier-2 fleet agents | Widespread `RuntimeContext`/`RuntimeRequest` | Agent impl | **No** |

---

## 10. Concrete coupling findings

| ID | Consumer | Concrete type | Owner domain | Classification |
|----|----------|---------------|--------------|----------------|
| EAC2-CC-001 | `agents/*`, `intergrax/agents/*` | `RuntimeContext` | NEXUS | **INTERNAL-LEAK** |
| EAC2-CC-002 | Same | `RuntimeRequest` / `RuntimeAnswer` | NEXUS | **INTERNAL-LEAK** |
| EAC2-CC-003 | `memory_wiring.py` | `SessionManager`, `DocumentStoreSessionStorage` | NEXUS + MEMORY stores | **CONCRETE-COUPLING** |
| EAC2-CC-004 | `catalog_declarative_invoker.py` | `RuntimeToolInvoker`, `invoke_catalog_tool_request` | NEXUS tools | Qual-only adapter |
| EAC2-CC-005 | `uaep.py` | `BoundToolGateway` (Nexus) | TOOLS invocation locus | **PARTIAL** — should be EAC-CON-015 port |
| EAC2-CC-006 | Multiple agents | `DiagnosticPayload` | NEXUS trace | **LEGACY_SHIM** |

---

## 11. Duplicate / legacy contracts (Phases 17–18)

| Role | Families | Verdict |
|------|----------|---------|
| Authorization WHETHER | EAC-CON-011 (GE), EAC-CON-013 (CW policy), EAC-CON-014 (delegation) | **Duplicate semantic** — ADR-GOV-01 |
| Execution request envelope | EAC-CON-019 vs EAC-CON-064 | **Version evolution missing** — internal type acts as public |
| Context source | EAC-CON-025 ports vs ad-hoc Nexus context assembly | **Adapter** — CL-EAC1-003 |
| Decision outcome | EAC-CON-009 vs EAC-CON-082 critic | **Legacy** |
| Recovery vs reconciliation | EAC-CON-035 vs EAC-CON-036 | **Legitimate specialization** (Reliability vs ERL) |
| Catalog vs distribution lifecycle | EAC-CON-038 vs EAC-CON-039 | **Handoff** — no duplicate owner |

**Versioning baseline:** Execution/decision/governance IDs and pydantic models use implicit compatibility (frozen field names, qualification gates). External/plugin contracts (**EAC-CON-042**, **020**, **015**) rely on plugin manifests + qual tests — **credible but mostly implicit** (EAC2-F-006).

---

## 12. Scope / identity baseline (Phase 20)

| Contract | Tenant/workspace/execution identity | Rating |
|----------|--------------------------------------|--------|
| EAC-CON-021 Memory ref read | `RequestIdentity`, scoped resource refs | **PASS** |
| EAC-CON-023 Knowledge ref read | Scoped query + identity | **PASS** |
| EAC-CON-028 UCL ref read | `context_scope_id`; workspace auth documented gap | **PARTIAL** |
| EAC-CON-002 Admission | Execution scope | **PASS** |
| EAC-CON-011 Governance | Policy context, grants | **PARTIAL** (CW overlap) |
| EAC-CON-043 T3 manifest | App id, memory scope | **PASS** |

---

## 13. Plugin replaceability baseline (Phase 19)

| Status | Count |
|--------|------:|
| Replaceable via declared port/SPI | 38 families |
| Partial (default concrete in consumer) | 12 |
| **NOT_REPLACEABLE** | 3 (agent Nexus coupling, harness, qual invoker) |

---

## 14. Known EAC-1 finding — contract analysis (Phase 27)

### CL-EAC1-001 — Tier-2/Agents → Nexus internals (**OPEN / HIGH**)

**Evidence @ EAC2_EVIDENCE_HEAD:** 40+ production agent modules under `agents/` and `intergrax/agents/` import `intergrax.runtime.nexus.*` (`RuntimeContext`, `RuntimeRequest`, `SessionManager`, `RuntimeConfig`, trace models). **Contract gap:** EAC-CON-008/064/065 used as public API. **Required contract (not designed here):** EE/Agent-owned step execution envelope + session port abstracting persistence — consumers depend on **contract package only**.

### CL-EAC1-002 — Memory/session wiring (**NARROWED / MEDIUM**)

**Read boundary closed:** EAC-CON-021. **Remaining gap:** EAC-CON-067 wires `SessionManager` (EAC-CON-063) instead of a Memory-owned session contract or CE-neutral session port. **CONTRACT-MISSING:** durable session turn index vs Nexus session storage boundary.

### CL-EAC1-003 — CE/UCL/Nexus context (**QUALIFICATION ONLY / LOW**)

Nexus `runtime/nexus/context/*` (EAC-CON-085) is **implementation locus** for CE assembly algorithms — **not** a second CONTEXT_ASSEMBLY authority. **Leak risk:** if Nexus types appear in CE public imports (monitor EAC-3). **No separate public ContextView owner conflict.**

### CL-EAC1-004 — GE ↔ CW authority (**ADR REQUIRED / HIGH**)

**Mapped contracts:** EAC-CON-011 (`AgentRuntimePolicyProvider`, `CapabilityGrantResolverPort`), EAC-CON-013 (workspace/tool policy declarations), EAC-CON-014 (delegation evaluators). **Duplicate WHETHER** for tool side-effects — **do not resolve in EAC-2**.

### CL-EAC1-005 — Tier-3 ↔ Hosting (**OPEN / MEDIUM**)

**Handoff contracts:** EAC-CON-043 (manifest, `ApplicationProfile`) → EAC-CON-045/046 (hosted lifecycle, context ports). **Overlap:** both interpret profile/digest and operational ownership; no single **activation** contract bridging install (Distribution) vs deploy (Hosting). **Catalog does not mutate Distribution** — EAC-CON-038 read-only **PASS**.

---

## 15. EAC-3 dependency inputs (Phase 21)

| Pattern | Example | EAC-3 action |
|---------|---------|--------------|
| Owner imports consumer contract | GE reads EE execution context (expected) | Verify direction |
| Low-level → high-level impl | Agents → Nexus | **Forbidden** — graph proof |
| Nexus → Tools/Agents | Internal step dispatch | Legal if via contracts only |
| T3 → Nexus session | `memory_wiring.py` | Flag upward dependency |
| Diagnostics → Observability store | Adapter in runtime | Confirm port-only |

---

## 16. ADR-required findings (unchanged)

| ADR | Contract evidence | Status |
|-----|-------------------|--------|
| ADR-GOV-01 | EAC-CON-011 vs 013/014 | **Open** |
| ADR-CTX-01 | EAC-CON-024/027/025 | **Open** |
| ADR-MKT-01 | EAC-CON-038/040/080 | Canon aligned |

---

## 17. Internal-domain handoffs (**ID** — not remediated)

Inherited from EAC-1 §14; no new **ID** from EAC-2 (contract defects are **CL** / **EAC2-F-*** ).

---

## 18. Cross-layer boundary matrix (Phase 25)

| Consumer Domain | Provider/Owner Domain | Semantic | Canonical Contract | Contract Owner | Consumer Imports | Implementation Leak? | Pluginable? | Finding |
|-----------------|----------------------|----------|-------------------|----------------|------------------|---------------------|-------------|---------|
| Execution | Orchestration | Strategy dispatch | EAC-CON-006 | ORCHESTRATION | `orchestration_topology` | No | Yes | — |
| Orchestration | Nexus | Graph run | EAC-CON-007 | NEXUS (internal) | `NexusLoop` (EE only) | Internal OK | Injected | — |
| Agents | Execution | Lifecycle/admission | EAC-CON-002, 019 | EE | Mixed — often skips to Nexus | **Yes** | Partial | EAC2-F-001 |
| Agents | Nexus | Step/session | EAC-CON-064, 065 (**illegal public**) | NEXUS | `runtime.nexus.*` | **Yes** | **No** | CL-EAC1-001 |
| Decision | Execution | Host run | EAC-CON-009 | DECISION_SYSTEM | `decision_*` contracts | No | Yes | — |
| Execution | Governance | WHETHER | EAC-CON-011 | GE | `agent_runtime_governance` | No | Partial | CL-EAC1-004 |
| Governance | Tools | Invocation policy | EAC-CON-015 + 011 | TOOLS + GE | tool contracts | No | Yes | — |
| CW | Governance | Tool/workspace auth | EAC-CON-013 + 011 | CW + GE | `collaborative_work` | Doc conflict | Partial | CL-EAC1-004 |
| CE | Memory | Memory refs | EAC-CON-021 | MEMORY | `memory_reference_read` | No | Yes | — |
| CE | RAG | Knowledge refs | EAC-CON-023 | RAG | `knowledge_reference_read` | No | Yes | — |
| CE | UCL | Lifecycle refs | EAC-CON-028 | UCL | `ucl_reference_read` | No | Yes | EAC2-F-005 |
| Diagnostics | Observability | Evidence | EAC-CON-029, 090 | OBSERVABILITY | diagnostic ports | Adapter only | Yes | — |
| Reliability | Execution | Continue/pause signals | EAC-CON-004, 035 | EE + Reliability | continuation ports | No | Yes | — |
| ERL | Execution | Lifecycle intent (not mutate) | EAC-CON-036, 004 | ERL + EE | ERL contracts | No* | Yes | *canon |
| ERL | Governance | Consequential continue | EAC-CON-087, 011 | ERL + GE | `governance_decision` | No | Yes | — |
| ERL | Observability | Reliability facts | EAC-CON-029 | OBSERVABILITY | events | No | Yes | — |
| Catalog | Distribution | Discovery handoff | EAC-CON-038, 080 | Catalog + AD | read/handoff ports | No | Yes | — |
| Marketplace | Catalog | Listings | EAC-CON-040, 079 | Catalog | marketplace contracts | No | Yes | — |
| Marketplace | Distribution | Acquire | EAC-CON-080 | AD | handoff | No | Yes | — |
| Tier3 | Hosting | Deploy/host | EAC-CON-043 → 045 | T3 + Hosting | manifest + lifecycle | Partial | Partial | CL-EAC1-005 |
| Plugins | Domain owners | Semantic validation | EAC-CON-042 + domain ports | DOMAIN | plugin SPI | No | Yes | — |

---

## 19. Contract owner matrix (Phase 26)

| Contract Family | Semantic Owner | Physical Package | Consumers | Implementers | Ownership Verdict |
|-----------------|----------------|------------------|-----------|--------------|-------------------|
| `execution_*` / UEA | UNIFIED_EXECUTION_RUNTIME | `intergrax/contracts/` | All strategies | EE runtime | **CORRECT** |
| `orchestration_topology` | ORCHESTRATION | `intergrax/contracts/` | EE | Strategies, Nexus | **CORRECT** |
| Nexus loop/session/response | NEXUS_EXECUTION_FLOW | `intergrax/runtime/nexus/` | EE internal, **Agents (violation)** | Nexus | **INTERNAL_LEAK** at agent boundary |
| `decision_*` | DECISION_SYSTEM | `intergrax/contracts/` | EE, CW | Decision engine | **CORRECT** |
| Governance / agent runtime policy | GOVERNED_EXECUTION | `intergrax/contracts/` | EE, Agents, Tools | GE engine | **CORRECT** |
| `collaborative_work` | COLLABORATIVE_WORK | `intergrax/contracts/` | CE, Agents | CW | **CONFLICT** with GE doc (ADR) |
| `tools_*` | TOOLS | `intergrax/contracts/` | Agents, Nexus | Drivers | **CORRECT** |
| Memory reference read | MEMORY | `intergrax/memory/contracts/` | CE | Memory readers | **CORRECT** |
| Knowledge reference read | RAG | `intergrax/knowledge/contracts/` | CE | RAG | **CORRECT** |
| UCL reference read | UNIFIED_CONTEXT_LIFECYCLE | `intergrax/ucl/contracts/` | CE | UCL | **CORRECT** |
| ContextView composition | CONTEXT_ENGINEERING | `intergrax/contracts/` + `context/` | Agent, Nexus | CE/Nexus impl | **NEUTRAL_OK** (impl in Nexus) |
| `runtime_event` / obs export | OBSERVABILITY | `intergrax/contracts/` | DIAG, T3 | Obs runtime | **CORRECT** |
| `diagnostics/*` | DIAGNOSTICS | `intergrax/contracts/` | Ops | DIAG | **CORRECT** |
| `enterprise_reliability/*` | ENTERPRISE_RELIABILITY_LAYER | `intergrax/contracts/` | EE, GE | ERL | **CORRECT** |
| Agent UAEP / agent_contract | AGENT_CONTRACTS_AND_ASSEMBLY | `intergrax/agents/` + contracts | EE | Agents | **MISPLACED** Nexus types in agent package |
| Application manifest | TIER3_APPLICATION_ENVIRONMENT | `intergrax/applications/contracts/` | Hosting | T3 apps | **CORRECT** |
| Hosting lifecycle | APPLICATION_HOSTING | `intergrax/hosting/contracts/` | T3 | Host runtime | **CORRECT** |
| Platform plugins | PLATFORM_PLUGINS | `intergrax/core/plugins/` | Domains | Plugins | **CORRECT** |
| Capability catalog | CAPABILITY_CATALOG_AND_DISCOVERY | `intergrax/contracts/capability_catalog/` | Marketplace | Federator | **CORRECT** |
| Agent distribution | AGENT_DISTRIBUTION | `intergrax/contracts/agent_distribution/` | Catalog | Installer | **CORRECT** |
| Legacy critic | DECISION_SYSTEM | `intergrax/runtime/critic/` | EE legacy | Critic | **LEGACY** |

---

## 20. Quality metrics summary

| Verdict / finding class | Count |
|-------------------------|------:|
| CORRECT | 52 |
| NEUTRAL_OK | 24 |
| MISPLACED | 4 |
| INTERNAL_LEAK (families) | 6 |
| CONCRETE_COUPLING | 6 |
| CONTRACT_MISSING | 2 |
| CONTRACT_DUPLICATE | 3 |
| CONTRACT_NOT_PLUGINABLE | 3 |
| CONTRACT_VENDOR_LEAK | 0 (core boundaries — integrations isolated behind EAC-CON-048) |
| CONTRACT_SCOPE_GAP | 2 |
| CONTRACT_VERSIONING_GAP | 1 (broad baseline) |
| BYPASS paths documented | 8 |

---

## 21. Critical boundary verdicts

| Boundary | Verdict | Notes |
|----------|---------|-------|
| Execution ↔ Nexus | **ALIGNED (canon) + LEAK (callers)** | EE owns lifecycle; Nexus internal; **agents bypass** |
| Agents ↔ Nexus | **FAIL (public coupling)** | EAC2-F-001/002 |
| GE ↔ CW ↔ Tools | **DOCUMENTATION + CONTRACT DUPLICATE** | CL-EAC1-004 |
| CE ↔ Memory | **ALIGNED** | EAC-CON-021 |
| CE ↔ RAG | **ALIGNED** | EAC-CON-023 |
| CE ↔ UCL | **ALIGNED + scope partial** | EAC-CON-028 B3 |
| Observability ↔ Diagnostics | **ALIGNED** | Evidence vs Problem |
| Reliability ↔ ERL ↔ Execution | **ALIGNED (canon)** | Disjoint authorities |
| Plugins ↔ domain owners | **ALIGNED** | EP lifecycle only |
| Catalog ↔ Distribution | **ALIGNED** | Read + handoff |
| Tier3 ↔ Hosting | **PARTIAL** | CL-EAC1-005 |

---

## 22. Validation checklist (Phase 29)

| Gate | Result |
|------|--------|
| V1 — All 34 domains audited | **PASS** (§5) |
| V2 — Public cross-layer contract has semantic owner | **PASS** (§4, §19) |
| V3 — Owners map to EAC-1 authorities | **PASS** |
| V4 — Major boundaries have contract or CONTRACT-MISSING | **PASS** (§18) |
| V5 — External private types recorded | **PASS** (§9–10) |
| V6 — CL-EAC1-001…005 mapped | **PASS** (§14) |
| V7 — Plugin mechanisms inventoried | **PASS** (§7) |
| V8 — Vendor leakage inspected | **PASS** (integrations walled) |
| V9 — Versioning baseline | **PASS** (§11) |
| V10 — No production code changed in EAC-2 session | **PASS** (documentation only) |
| V11 — No ADR resolved | **PASS** |
| V12 — Sufficient for EAC-3 | **PASS** (§15) |

---

*EAC-2 artifact reconciled @ **EAC2_EVIDENCE_HEAD** `e1cdaa77e0b1bf7608573001b23a42ebf870507f`. Upstream: **EAC1_CLOSE_COMMIT** `588180053c908c5960d5bca099b14e7b217849bb`.*

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
