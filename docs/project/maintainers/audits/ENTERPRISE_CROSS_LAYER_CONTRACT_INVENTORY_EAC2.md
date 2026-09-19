# EAC-2 — Enterprise Cross-Layer Contract Inventory

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-2 — Cross-Layer Contract Inventory · **EAC-2R1** — Contract Ownership Normalization & Family Split  
**Type:** Read-only contract / boundary audit (no remediation)  
**Authority:** EAC-0 R1 + EAC-1R3 (`ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md`)

**EAC-2R1 rule:** one **EAC-CON** family → exactly one **Semantic Owner** (domain ID) or explicit `OWNER UNRESOLVED — ADR-*`. Consumers, implementers, and handoff participants are **not** co-owners.

### Provenance

| Field | SHA | Meaning |
|-------|-----|---------|
| **EAC1_CLOSE_COMMIT** (independent EAC-1 close @ operator pin) | `588180053c908c5960d5bca099b14e7b217849bb` | Last EAC-1 matrix close referenced for drift gate |
| **EAC2_INDEPENDENT_AUDIT_HEAD** (last EAC-2 @ operator pin) | `cffd71062d9145a7b52c00dff1f02dcdfec2e262` | Pre–EAC-2R1 inventory baseline |
| **EAC2_SESSION_START_HEAD** | `e1cdaa77e0b1bf7608573001b23a42ebf870507f` | Original EAC-2 documentation session |
| **EAC2R1_SESSION_START_HEAD** | `8552029bdd07a747157639d6e13b51c3856123d9` | EAC-2R1 session open (`HEAD == origin/development`) |
| **EAC2_EVIDENCE_HEAD** | `e1cdaa77e0b1bf7608573001b23a42ebf870507f` | Repository state @ original EAC-2 commit |
| **EAC2R1_EVIDENCE_HEAD** | `8552029bdd07a747157639d6e13b51c3856123d9` | Repository state reconciled immediately before EAC-2R1 commit (code evidence; doc commit follows) |

| Gate | Value |
|------|-------|
| **Branch** | `development` |
| **HEAD == origin/development @ EAC-2R1 session** | **YES** (`8552029bdd07a747157639d6e13b51c3856123d9`) |
| **EAC-1 ancestor contained** | **YES** (`588180053…` ⊆ HEAD) |
| **Upstream** | [`ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md`](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md), [`ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md`](ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md) |

**Drift since `EAC1_CLOSE_COMMIT` (contract-relevant commits inspected):**

| Commit | Area | EAC-2 effect |
|--------|------|----------------|
| `4403f1b5d` | UCL `UclReferenceReadPort` | New **STATE_REFERENCE_CONTRACT** row (EAC-CON-028) |
| `162ba778f` | UCL workspace ownership hardening | Scope baseline for CE↔UCL (no new public family) |
| `0ba1a514c` | Governance pre-model policy | GE admission path evidence refresh |
| `e1cdaa77e` | Memory provider qualification docs | Qualification only; contracts unchanged |

**Drift since `EAC2_INDEPENDENT_AUDIT_HEAD` (`cffd71062…`) — inspected @ EAC2R1_SESSION_START_HEAD (ownership-relevant only):**

| Commit | Area | EAC-2R1 effect |
|--------|------|----------------|
| `38b243d18` | CE-02 mandatory budget accounting | Supports **EAC-CON-072** vs **099** budget split (CE vs EE) |
| `470804d5c` | Governance pre-model identity | GE admission evidence unchanged; no new public family |
| `d2b138a0c` | Memory production provider admission | MEMORY provider SPI unchanged |
| `29bdecd15` | CE degradation ownership | CE contracts unchanged |
| `8552029bd` | UCL artifact workspace ownership | UCL scope note only |

**Out of scope:** uncommitted working-tree deltas — **not** evidence for this artifact.

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
| EAC-CON-014 | Physical delegation / workspace delegation governance | DOMAIN_CONTRACT | COLLABORATIVE_WORK | `physical_delegation_governance.py` | EE, GE (read) | CW evaluators | Public | Yes | PARTIAL | CANONICAL | **ADR-GOV-01** overlap with EAC-CON-011 (facts vs WHETHER) |
| EAC-CON-015 | Tool invocation spine | DOMAIN_CONTRACT | TOOLS | `intergrax/contracts/tools*` | Agents, Nexus (internal), Integrations | Tool drivers | Public | Provider SPI | PARTIAL | CANONICAL | — |
| EAC-CON-016 | Provider invocation dispatch | PORT | TOOLS | `provider_invocation_dispatch.py` | ERL, Integrations | Providers | Public | Yes | PARTIAL | CANONICAL | ERL consumes; does not own dispatch |
| EAC-CON-017 | Agent contract & step loop | DOMAIN_CONTRACT | AGENT_CONTRACTS_AND_ASSEMBLY | `agent_contract.py`, `contracts/agent_*` | EE, Nexus | Agent impls | Public | Yes | PARTIAL | CANONICAL | **INTERNAL-LEAK** via Nexus types EAC2-F-002 |
| EAC-CON-018 | UAEP protocol | DOMAIN_CONTRACT | AGENT_CONTRACTS_AND_ASSEMBLY | `uaep_protocol.py`, `contracts/uaep*` | EE, harness | Agents | Public | Yes | PARTIAL | CANONICAL | Imports `RuntimeContext` in executor |
| EAC-CON-019 | Runtime execution context (contract) | DOMAIN_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `runtime_execution_context.py` | Agents, UAEP | EE | Public | N/A | PARTIAL | CANONICAL | Preferred over Nexus `RuntimeContext` |
| EAC-CON-020 | Memory provider stores | PROVIDER_SPI | MEMORY | `intergrax/memory/contracts/*` | T3 wiring, RAG | Store plugins | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-021 | Memory reference read | STATE_REFERENCE_CONTRACT | MEMORY | `memory/contracts/memory_reference_read.py` | CE, ContextView adapters | Memory readers | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-022 | Memory runtime read (inspection) | PORT | MEMORY | `memory_runtime_read.py` | Observability inspection | Memory | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-023 | Knowledge reference read | STATE_REFERENCE_CONTRACT | RAG | `knowledge/contracts/knowledge_reference_read.py` | CE adapters | RAG stack | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-024 | ContextView composition (MP-5E) | DOMAIN_CONTRACT | COLLABORATIVE_WORK (MP-5) | `context_view_composition.py`, `collaborative_work/context_view_composition.py` | Agent, runtime consumer | MP-5 composer | Public | Strategy SPI | PARTIAL | CANONICAL | Pre–EBH-1-R1 owner CONTEXT_ENGINEERING — **HISTORICAL**; reconciled @ EBH-1-R1 |
| EAC-CON-025 | ContextView source ports (MP-5D) | PORT | COLLABORATIVE_WORK (MP-5) | `context_view_source_ports.py` | MP-5 composer | Domain readers via adapters | Public | Yes | PARTIAL | CANONICAL | Consumer-side ports; aligns MP-5F-B1/B2/B3 |
| EAC-CON-026 | ContextView visibility policy (MP-5C) | DOMAIN_CONTRACT | COLLABORATIVE_WORK (MP-5) | `context_view_visibility_policy.py` | MP-5 pipeline | Policy plugins | Public | Yes | PARTIAL | CANONICAL | Pre–EBH-1-R1 owner CONTEXT_ENGINEERING — **HISTORICAL** |
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
| EAC-CON-040 | Marketplace listing / acquisition (catalog semantics) | DOMAIN_CONTRACT | CAPABILITY_CATALOG_AND_DISCOVERY | `contracts/marketplace/*` | Distribution, Marketplace (consumer) | Catalog federator | Public | PARTIAL | PARTIAL | CANONICAL | ADR-MKT-01 aligned; Distribution not co-owner |
| EAC-CON-041 | Tool marketplace lifecycle handoff | PORT | TOOLS | `tools/marketplace_lifecycle_handoff.py` | Distribution, Marketplace | Tool hosts | Public | Yes | PARTIAL | CANONICAL | — |
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
| EAC-CON-058 | Self-healing / preventive (platform extension) | DOMAIN_CONTRACT | RELIABILITY_FAILURE_AND_HITL | `contracts/self_healing/*`, `preventive/*` | EE, GE (consequential remediation via EAC-CON-011) | Reliability plugins | Public | Yes | PARTIAL | CANONICAL | Healing policy owner; GE admission separate |
| EAC-CON-059 | Tool runtime read port (inspection) | PORT | TOOLS | `tool_runtime_read.py` | DIAG, OBS (aggregate) | Tool runtime | Public | Yes | PARTIAL | CANONICAL | Obs may aggregate; does not own tool state |
| EAC-CON-060 | Causal / platform evidence | EVENT_CONTRACT | OBSERVABILITY | `platform_causal_evidence.py` | DIAG, ERL | EE | Public | PARTIAL | PARTIAL | CANONICAL | — |
| EAC-CON-061 | Execution failure evidence | EVENT_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `execution_failure_evidence.py` | Observability | EE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-062 | Sandbox isolation policy / admission authority | DOMAIN_CONTRACT | GOVERNED_EXECUTION | `runtime_sandbox_isolation_authority.py` (policy surface) | UAEP, Tools, EE | GE evaluators | Public | PARTIAL | PARTIAL | CANONICAL | EE enforces via EAC-CON-097 |
| EAC-CON-063 | Session persistence (Nexus) | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `runtime/nexus/session/*` | Nexus, **T3 memory_wiring** | SessionStorage impls | **Internal** | Yes | N/A | **BYPASS** | **CL-EAC1-002** |
| EAC-CON-064 | Nexus runtime request/answer | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `responses/response_schema.py` | Agents (**leak**) | Nexus | **Internal** | No | N/A | **BYPASS** | **CL-EAC1-001** |
| EAC-CON-065 | Nexus runtime context | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `engine/runtime_context.py` | Agents (**leak**) | Nexus | **Internal** | No | N/A | **BYPASS** | **CL-EAC1-001** |
| EAC-CON-066 | Diagnostic payload (trace) | INTERNAL_CONTRACT | NEXUS_EXECUTION_FLOW | `tracing/trace_models.py` | Agents diagnostics | Nexus | **Internal** | No | N/A | LEGACY_SHIM | Prefer contract diagnostics |
| EAC-CON-067 | Tier-3 memory wiring composition | COMPOSITION_CONTRACT | TIER3_APPLICATION_ENVIRONMENT | `applications/_shared/memory_wiring.py` | T3 hosts | Memory plugins + **SessionManager** | Public wiring | PARTIAL | N/A | **PARTIAL** | **CONCRETE-COUPLING** EAC2-F-003 |
| EAC-CON-068 | Distributed KV (supporting) | PROVIDER_SPI | PLATFORM_FOUNDATION | `distributed/contracts/kv_store.py` | Multiple | Redis/SQLite | Public | Yes | PARTIAL | CANONICAL | Supporting SPI; not a composite owner label |
| EAC-CON-069 | External operations (integration provider commands) | COMMAND_CONTRACT | INTEGRATIONS | `external_operations/*` | ERL, Tools | Integration providers | Public | Yes | PARTIAL | CANONICAL | Tool invocation semantics: EAC-CON-098 |
| EAC-CON-070 | Delegated execution family | PORT | UNIFIED_EXECUTION_RUNTIME | `delegated_execution_*.py` | AW, providers | EE | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-071 | Step LLM router port | PORT | AGENT_CONTRACTS_AND_ASSEMBLY | `step_llm_router_port.py` | ACP authoring | LLM adapters | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-072 | Context / token budget (CE accounting) | DOMAIN_CONTRACT | CONTEXT_ENGINEERING | CE budget modules (`intergrax/context/` policy) | Agents, T3 | CE composers | Public | Hook SPI | PARTIAL | CANONICAL | CE-02 qualification; EE slice → EAC-CON-099 |
| EAC-CON-073 | Runtime invariant rules | STRATEGY_SPI | PLATFORM_FOUNDATION | `runtime_invariants.py` | EE | Rule packs | Public | Yes | PARTIAL | CANONICAL | — |
| EAC-CON-074 | Forecast / predictive analyzers | STRATEGY_SPI | DIAGNOSTICS | `predictive*.py`, `statistical_forecast_analyzer.py` | DIAG | Plugins | Public | Yes | PARTIAL | CANONICAL | Strategy SPI under DIAGNOSTICS authority |
| EAC-CON-075 | Semantic verification judges | STRATEGY_SPI | DECISION_SYSTEM | `semantic_verification.py` | Decision qual | Judges | Public | Yes | PARTIAL | CANONICAL | Strategy SPI under DECISION_SYSTEM authority |
| EAC-CON-076 | Execution environment isolation view (T3 profile) | STATE_REFERENCE_CONTRACT | TIER3_APPLICATION_ENVIRONMENT | `execution_environment_isolation.py` | EE, GE (EAC-CON-062 read) | T3 profiles | Public | PARTIAL | PARTIAL | CANONICAL | T3-owned view; EE enforces admission |
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
| EAC-CON-091 | Plugin memory store discovery / resolution port | PORT | PLATFORM_PLUGINS | `core/plugins`, `memory/resolver` | T3, MEMORY (store SPI) | Plugin registry, memory backends | Public | Yes | PARTIAL | CANONICAL | MEMORY implements stores; does not own discovery port |
| EAC-CON-092 | Hosting service registry | COMPOSITION_CONTRACT | APPLICATION_HOSTING | `hosting/services.py` | T3 host apps | Hosting | Public | PARTIAL | PARTIAL | CANONICAL | **CL-EAC1-005** |
| EAC-CON-093 | Multi-agent coordination governance (delegation auth eval) | DOMAIN_CONTRACT | GOVERNED_EXECUTION | `multi_agent_coordination_governance.py` | EE, CW | GE evaluators | Public | Yes | PARTIAL | CANONICAL | **CONTRACT-DUPLICATE** vs EAC-CON-011; **ADR-GOV-01** open |
| EAC-CON-094 | Provider effect uncertainty / reconciliation ports | PORT | ENTERPRISE_RELIABILITY_LAYER | ERL `provider_*` / runtime reconciliation ports | EE, Tools | ERL orchestrator | Public | Yes | PARTIAL | CANONICAL | Split from EAC-CON-016; dispatch remains TOOLS |
| EAC-CON-095 | Skill marketplace lifecycle handoff | PORT | SKILLS | `skills/.../marketplace_lifecycle_handoff.py` | Distribution, Marketplace | Skill hosts | Public | Yes | PARTIAL | CANONICAL | Split from EAC-CON-041 |
| EAC-CON-096 | Model runtime read port (inspection) | PORT | LLM_ADAPTERS | `model_runtime_read.py` | DIAG, OBS (aggregate) | LLM runtime | Public | Yes | PARTIAL | CANONICAL | Split from EAC-CON-059 |
| EAC-CON-097 | Sandbox execution environment enforcement | DOMAIN_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `runtime_sandbox_isolation_authority.py` (enforcement surface) | UAEP, Tools | EE runtime | Public | PARTIAL | PARTIAL | CANONICAL | Split from EAC-CON-062; policy owner GE |
| EAC-CON-098 | External work dispatch (tool semantic) | COMMAND_CONTRACT | TOOLS | `external_work.py` | ERL, Integrations | Tool drivers | Public | Yes | PARTIAL | CANONICAL | Split from EAC-CON-069 |
| EAC-CON-099 | Agent execution / resource budget slice | DOMAIN_CONTRACT | UNIFIED_EXECUTION_RUNTIME | `agent_budget.py` | Agents, T3 | EE ledger | Public | Hook SPI | PARTIAL | CANONICAL | Split from EAC-CON-072; CE token budget separate |

**Inventory metrics (@ grouped families, post–EAC-2R1 splits):**

| Metric | Count |
|--------|------:|
| Total contract families (EAC-CON rows) | **99** |
| Public cross-layer contracts | **75** |
| Internal-only / subordinate contracts | **14** |
| Strategy SPI families | **12** |
| Provider SPI families | **18** |
| Event contract families | **6** |
| Legacy contract families | **3** |
| Documented bypass / leak paths | **8** |

**Ownership metrics (EAC-2R1):**

| Metric | Count |
|--------|------:|
| Public canonical contract families | **75** |
| Families with exactly one semantic owner (incl. internal/legacy rows) | **99** |
| Families `OWNER UNRESOLVED` / explicit ADR-required owner cell | **0** |
| Families with multi-owner semantic owner cells | **0** |

ADR-GOV-01 and related conflicts remain **open** in Findings columns — not resolved via composite owner wording.

---

## 5. Per-domain owned / consumed contracts (Phase 4)

Compact register — all **34** domains audited. Empty **Owned** is acceptable.

| Domain | Public contracts owned (IDs) | Key consumed contracts | Ports provided | Ports consumed | Known bypasses |
|--------|------------------------------|------------------------|----------------|----------------|----------------|
| PLATFORM_FOUNDATION | EAC-CON-073 | All (meta) | — | — | — |
| UNIFIED_EXECUTION_RUNTIME | EAC-CON-001…005, 019, 061, 070, 097, 099 | EAC-CON-011, 006, 029, 076 | Admission, continuation, sandbox enforcement | Governance, Obs, T3 isolation view | — |
| ORCHESTRATION | EAC-CON-006 | EAC-CON-003, 007 | Topology ports | EE admission | — |
| NEXUS_EXECUTION_FLOW | EAC-CON-007, 008, 063…066, 085 | EE context, tool/agent contracts | **None public** | Tools, Agents (internal) | **008, 063–066** |
| DECISION_SYSTEM | EAC-CON-009, 010, 075 | EAC-CON-029, 054 | Decision SPI | EE host | EAC-CON-082 legacy |
| GOVERNED_EXECUTION | EAC-CON-011, 012, 062, 077, 078, 088, 093 | EAC-CON-001, 015, 014 (CW facts) | Policy evaluators | EE, Tools, CW | **CL-EAC1-004** |
| REASONING_AND_COGNITION | EAC-CON-054 | EAC-CON-024, 047 | Reasoning SPI | LLM | — |
| AGENT_CONTRACTS_AND_ASSEMBLY | EAC-CON-017, 018, 071, 083 | EAC-CON-011, 024, 015, **064–065** | Agent SPI | **Nexus leak** | **001, 083** |
| AGENT_DISTRIBUTION | EAC-CON-039, 080 | EAC-CON-038, 002 | Install ports | Catalog | — |
| LLM_ADAPTERS | EAC-CON-047, 096 | — | LLM backends, runtime read | Vendor APIs, DIAG | — |
| TOOLS | EAC-CON-015, 016, 041, 059, 098 | EAC-CON-011, 094 | Tool drivers, dispatch | GE, ERL, Integrations | — |
| SKILLS | EAC-CON-049, 095 | EAC-CON-015 | Skill hosts | Tools | — |
| INTEGRATIONS | EAC-CON-048, 069 | EAC-CON-015, 098 | Provider backends | Credentials, Tools | — |
| RAG | EAC-CON-023 | EAC-CON-021, 048 | Retrieval | Memory refs | — |
| MEMORY | EAC-CON-020, 021, 022 | EAC-CON-091 (consumer) | Store plugins | Obs (audit), Plugins | **067 wiring** |
| CONTEXT_ENGINEERING | EAC-CON-024, 025, 026, 072 | EAC-CON-021, 023, 028, 013, 099 (read) | Composers | Memory/RAG/UCL/CW/EE | **085 impl locus** |
| UNIFIED_CONTEXT_LIFECYCLE | EAC-CON-027, 028 | EAC-CON-024 | Optimizer | CE bundles | Scope gap B3 |
| MODALITY | EAC-CON-053 | EAC-CON-024 | Adapters | CE | — |
| OBSERVABILITY | EAC-CON-029, 030, 060, 081, 090 | EAC-CON-061, 059, 096 (aggregate reads) | Export sinks | EE events, domain read ports | — |
| DIAGNOSTICS | EAC-CON-031, 032, 033, 074, 086 | EAC-CON-029, 090 | Detectors | Obs evidence | — |
| RELIABILITY_FAILURE_AND_HITL | EAC-CON-034, 035, 058, 089 | EAC-CON-004, 011 | Recovery/HITL | EE lifecycle | — |
| ADAPTIVE_HARNESS_INTELLIGENCE | EAC-CON-057 | Research telemetry | — | — | — |
| ELASTIC_CAPACITY_AND_SCALING | EAC-CON-052 | EAC-CON-002 | Capacity | EE | — |
| EXPERIMENTATION_AND_DX | — (guides) | Qual contracts | — | — | — |
| TIER3_APPLICATION_ENVIRONMENT | EAC-CON-043, 044, 067, 076 | EAC-CON-002, 045, 091 | App manifests | Hosting, EE | **067, 005** |
| APPLICATION_HOSTING | EAC-CON-045, 046, 092 | EAC-CON-043, 002 | Host lifecycle | T3 manifest | **005** |
| CODE_CRAFT | EAC-CON-055 | CW, Tools | — | — | — |
| AUTONOMOUS_WORK | EAC-CON-050 | EAC-CON-002, 070 | Worker dispatch | EE | — |
| COLLABORATIVE_WORK | EAC-CON-013, 014 | EAC-CON-011, 009, 093 | CW stores | GE (**overlap**) | **004** |
| BACKGROUND_TASKS | EAC-CON-051 | EAC-CON-002 | Queues | EE | — |
| CAPABILITY_CATALOG_AND_DISCOVERY | EAC-CON-038, 040, 079 | Domain descriptors | Catalog read, marketplace listing | Distribution, Marketplace | — |
| PROOF_RECEIPTS | EAC-CON-056 | — | — | — | — |
| PLATFORM_PLUGINS | EAC-CON-042, 091 | Domain validation, MEMORY wiring | Plugin registry | Domains, MEMORY | Registry ≠ domain semantic owner |
| ENTERPRISE_RELIABILITY_LAYER | EAC-CON-036, 037, 087, 094 | EAC-CON-011, 004, 016, 029 | Reconciliation SPI | GE, EE, Tools | — |

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
| Memory store plugin | EAC-CON-020 | MEMORY | Yes | PASS |
| Plugin memory discovery port | EAC-CON-091 | PLATFORM_PLUGINS | Yes | PASS (MEMORY implements store) |
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
| EAC2-F-004 | CONTRACT-DUPLICATE | **HIGH** | EAC-CON-011 vs 013 vs 014/093 | GE; CW (separate owner per ID) | Parallel WHETHER for tools/workspace/delegation | **ADR** CL-EAC1-004 |
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
| Authorization WHETHER | EAC-CON-011 (GE), EAC-CON-013 (CW policy), EAC-CON-093 (delegation eval); EAC-CON-014 (CW delegation facts) | **Duplicate semantic** — ADR-GOV-01 |
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

Nexus `runtime/nexus/context/*` (EAC-CON-085) is **implementation locus** for CE assembly algorithms — **not** a second CONTEXT_ASSEMBLY authority. **Leak risk:** if Nexus types appear in CE public imports (monitor EAC-3). **Principal-scoped ContextView** semantic owner is **COLLABORATIVE_WORK / MP-5** ([ADR-MP-006](../../technical/adr/entries/2026-09-17/ADR-MP-006.md); EBH-1-R1) — distinct from CONTEXT_ASSEMBLY_AUTHORITY.

### CL-EAC1-004 — GE ↔ CW authority (**ADR REQUIRED / HIGH**)

**Mapped contracts:** EAC-CON-011 (`AgentRuntimePolicyProvider`, `CapabilityGrantResolverPort`), EAC-CON-013 (workspace/tool policy declarations), EAC-CON-014 (CW delegation facts), EAC-CON-093 (multi-agent delegation evaluators). **Duplicate WHETHER** for tool side-effects — **do not resolve in EAC-2 / EAC-2R1** (ownership normalized; ADR remains open).

### CL-EAC1-005 — Tier-3 ↔ Hosting (**OPEN / MEDIUM**)

**Handoff contracts:** EAC-CON-043 (manifest, `ApplicationProfile`) → EAC-CON-045/046 (hosted lifecycle, context ports). **Overlap:** both interpret profile/digest and operational ownership; no single **activation** contract bridging install (Distribution) vs deploy (Hosting). **Catalog does not mutate Distribution** — EAC-CON-038 read-only **PASS**.

---

## 15. EAC-3 dependency inputs (Phase 21 · hardened @ EAC-2R1)

**Edge rule:** `consumer domain → semantic contract owner domain` (one owner per **EAC-CON** row).

| Consumer → owner (semantic) | Canonical contract(s) | EAC-3 action |
|-----------------------------|-------------------------|--------------|
| Agents → UNIFIED_EXECUTION_RUNTIME | EAC-CON-002, 019 (not Nexus 064/065) | Prove no upward Nexus leak |
| Agents → GOVERNED_EXECUTION | EAC-CON-011 | Policy direction |
| Agents → TOOLS | EAC-CON-015 | Invocation |
| Agents → CONTEXT_ENGINEERING | EAC-CON-072 | Context budget / generic assembly |
| Agents → COLLABORATIVE_WORK (MP-5) | EAC-CON-024, 025, 026 | Principal-scoped ContextView |
| Agents → UNIFIED_EXECUTION_RUNTIME | EAC-CON-099 | Execution budget |
| CE → MEMORY | EAC-CON-021 | Reference read |
| CE → RAG | EAC-CON-023 | Reference read |
| CE → UNIFIED_CONTEXT_LIFECYCLE | EAC-CON-028 | Reference read |
| Reliability → UNIFIED_EXECUTION_RUNTIME | EAC-CON-004 | Continuation (consumer) |
| Reliability → RELIABILITY_FAILURE_AND_HITL | EAC-CON-035 | Recovery admission |
| ERL → UNIFIED_EXECUTION_RUNTIME | EAC-CON-036 | Reconciliation intent |
| ERL → UNIFIED_EXECUTION_RUNTIME | EAC-CON-004 | Checkpoint read (consumer) |
| ERL → GOVERNED_EXECUTION | EAC-CON-087 | Consequential continue material |
| ERL → GOVERNED_EXECUTION | EAC-CON-011 | Governed continue (consumer) |
| ERL → TOOLS | EAC-CON-016 | Dispatch consumption |
| Tools → ENTERPRISE_RELIABILITY_LAYER | EAC-CON-094 | Effect uncertainty / reconciliation |
| Marketplace → CAPABILITY_CATALOG_AND_DISCOVERY | EAC-CON-040, 079 | Listings |
| Marketplace → AGENT_DISTRIBUTION | EAC-CON-080 | Agent handoff |
| Marketplace → TOOLS | EAC-CON-041 | Tool lifecycle handoff |
| Marketplace → SKILLS | EAC-CON-095 | Skill lifecycle handoff |
| Tier3 → APPLICATION_HOSTING | EAC-CON-043 → 045 | Deploy/host |
| Tier3 → PLATFORM_PLUGINS | EAC-CON-091 | Plugin discovery |
| Diagnostics → OBSERVABILITY | EAC-CON-029, 090 | Evidence read |
| Diagnostics → TOOLS | EAC-CON-059 | Tool runtime inspection read |
| Diagnostics → LLM_ADAPTERS | EAC-CON-096 | Model runtime inspection read |
| GE → COLLABORATIVE_WORK | EAC-CON-014 (read facts) | CW not co-owner of WHETHER |

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
| ADR-GOV-01 | EAC-CON-011 vs 013/014/093 | **Open** |
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
| Agents | Execution | Lifecycle/admission | EAC-CON-002, 019 | UNIFIED_EXECUTION_RUNTIME | Mixed — often skips to Nexus | **Yes** | Partial | EAC2-F-001 |
| Agents | Nexus | Step/session | EAC-CON-064, 065 (**illegal public**) | NEXUS | `runtime.nexus.*` | **Yes** | **No** | CL-EAC1-001 |
| Decision | Execution | Host run | EAC-CON-009 | DECISION_SYSTEM | `decision_*` contracts | No | Yes | — |
| Execution | Governance | WHETHER | EAC-CON-011 | GOVERNED_EXECUTION | `agent_runtime_governance` | No | Partial | CL-EAC1-004 |
| Governance | Tools | Tool invocation policy | EAC-CON-011 | GOVERNED_EXECUTION | `agent_runtime_governance` | No | Partial | CL-EAC1-004 |
| Tools | Governance | Tool admission (consumer) | EAC-CON-015 | TOOLS | tool contracts | No | Yes | — |
| CW | Governance | Workspace/tool policy surface | EAC-CON-013 | COLLABORATIVE_WORK | `collaborative_work` | Doc conflict | Partial | CL-EAC1-004 |
| CW | Governance | Platform WHETHER | EAC-CON-011 | GOVERNED_EXECUTION | GE evaluators | Overlap | Partial | CL-EAC1-004 |
| CW | Governance | Delegation facts | EAC-CON-014 | COLLABORATIVE_WORK | `physical_delegation_governance` | No | Partial | ADR-GOV-01 |
| GE | CW | Delegation facts (read) | EAC-CON-014 | COLLABORATIVE_WORK | CW stores | No | Partial | ADR-GOV-01 |
| CE | Memory | Memory refs | EAC-CON-021 | MEMORY | `memory_reference_read` | No | Yes | — |
| CE | RAG | Knowledge refs | EAC-CON-023 | RAG | `knowledge_reference_read` | No | Yes | — |
| CE | UCL | Lifecycle refs | EAC-CON-028 | UCL | `ucl_reference_read` | No | Yes | EAC2-F-005 |
| Diagnostics | Observability | Evidence | EAC-CON-029, 090 | OBSERVABILITY | diagnostic ports | Adapter only | Yes | — |
| Reliability | Execution | Continue/pause signals | EAC-CON-004 | UNIFIED_EXECUTION_RUNTIME | continuation ports | No | Yes | — |
| Reliability | Execution | Recovery admission | EAC-CON-035 | RELIABILITY_FAILURE_AND_HITL | `recovery_admission` | No | Yes | — |
| ERL | Execution | Lifecycle intent (not mutate) | EAC-CON-036 | ENTERPRISE_RELIABILITY_LAYER | ERL contracts | No* | Yes | *canon |
| ERL | Execution | Checkpoint read (consumer) | EAC-CON-004 | UNIFIED_EXECUTION_RUNTIME | continuation ports | No* | Yes | *canon |
| ERL | Governance | Consequential continue material | EAC-CON-087 | ENTERPRISE_RELIABILITY_LAYER | `governance_decision` | No | Yes | — |
| ERL | Governance | Governed continue (consumer) | EAC-CON-011 | GOVERNED_EXECUTION | GE admission | No | Yes | — |
| ERL | Tools | Provider dispatch (consumer) | EAC-CON-016 | TOOLS | `provider_invocation_dispatch` | No | Yes | — |
| ERL | Observability | Reliability facts | EAC-CON-029 | OBSERVABILITY | events | No | Yes | — |
| Catalog | Distribution | Discovery handoff | EAC-CON-038 | CAPABILITY_CATALOG_AND_DISCOVERY | catalog read | No | Yes | — |
| Distribution | Catalog | Discovery (consumer) | EAC-CON-080 | AGENT_DISTRIBUTION | handoff port | No | Yes | — |
| Marketplace | Catalog | Listings | EAC-CON-040, 079 | CAPABILITY_CATALOG_AND_DISCOVERY | marketplace contracts | No | Yes | — |
| Marketplace | Distribution | Acquire | EAC-CON-080 | AGENT_DISTRIBUTION | handoff | No | Yes | — |
| Marketplace | Tools | Tool listing handoff | EAC-CON-041 | TOOLS | marketplace handoff | No | Yes | — |
| Marketplace | Skills | Skill listing handoff | EAC-CON-095 | SKILLS | marketplace handoff | No | Yes | — |
| Tier3 | Hosting | Deploy/host manifest | EAC-CON-043 | TIER3_APPLICATION_ENVIRONMENT | manifest | Partial | Partial | CL-EAC1-005 |
| Tier3 | Hosting | Hosted lifecycle | EAC-CON-045 | APPLICATION_HOSTING | lifecycle | Partial | Partial | CL-EAC1-005 |
| Plugins | Domain owners | Semantic validation | EAC-CON-042 | PLATFORM_PLUGINS | plugin SPI | No | Yes | — |
| Plugins | Memory | Store discovery handoff | EAC-CON-091 | PLATFORM_PLUGINS | resolver | No | Yes | — |
| EE | T3 | Isolation view (consumer) | EAC-CON-076 | TIER3_APPLICATION_ENVIRONMENT | isolation profile | No | Partial | — |
| EE | GE | Sandbox policy (consumer) | EAC-CON-062 | GOVERNED_EXECUTION | isolation authority | No | Partial | — |
| EE | EE | Sandbox enforcement | EAC-CON-097 | UNIFIED_EXECUTION_RUNTIME | runtime enforcement | No | Partial | — |

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
| ContextView composition (MP-5) | COLLABORATIVE_WORK (MP-5) | `intergrax/contracts/context_view_*` + `collaborative_work/` | Agent, runtime | MP-5 composer + adapters | **CORRECT** @ EBH-1-R1 (pre-R1: CONTEXT_ENGINEERING — **HISTORICAL**) |
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
| `physical_delegation_governance` | COLLABORATIVE_WORK | `intergrax/contracts/` | EE, GE | CW | **CONFLICT** ADR-GOV-01 |
| `multi_agent_coordination_governance` | GOVERNED_EXECUTION | `intergrax/contracts/` | EE, CW | GE evaluators | **CONFLICT** ADR-GOV-01 |
| `provider_invocation_dispatch` | TOOLS | `intergrax/contracts/` | ERL, Integrations | Providers | **CORRECT** |
| ERL provider reconciliation ports | ENTERPRISE_RELIABILITY_LAYER | `intergrax/contracts/enterprise_reliability/` | EE, Tools | ERL | **CORRECT** |
| Marketplace listing (catalog) | CAPABILITY_CATALOG_AND_DISCOVERY | `intergrax/contracts/marketplace/` | Marketplace | Catalog | **CORRECT** |
| Tool marketplace handoff | TOOLS | `tools/marketplace_lifecycle_handoff.py` | Distribution | Tool hosts | **CORRECT** |
| Skill marketplace handoff | SKILLS | `skills/.../marketplace_lifecycle_handoff.py` | Distribution | Skill hosts | **CORRECT** |
| `tool_runtime_read` | TOOLS | `intergrax/contracts/` | DIAG, OBS | Tool runtime | **CORRECT** |
| `model_runtime_read` | LLM_ADAPTERS | `intergrax/contracts/` | DIAG, OBS | LLM runtime | **CORRECT** |
| Sandbox isolation policy | GOVERNED_EXECUTION | `runtime_sandbox_isolation_authority.py` | UAEP, Tools, EE | GE evaluators | **CORRECT** |
| Sandbox execution enforcement | UNIFIED_EXECUTION_RUNTIME | `runtime_sandbox_isolation_authority.py` | UAEP, Tools | EE runtime | **CORRECT** |
| CE context/token budget | CONTEXT_ENGINEERING | `intergrax/context/` | Agents, T3 | CE composers | **CORRECT** |
| Agent execution budget | UNIFIED_EXECUTION_RUNTIME | `agent_budget.py` | Agents, T3 | EE ledger | **CORRECT** |
| T3 isolation view | TIER3_APPLICATION_ENVIRONMENT | `execution_environment_isolation.py` | EE, GE | T3 profiles | **CORRECT** |
| Plugin memory discovery | PLATFORM_PLUGINS | `core/plugins/` | T3, MEMORY | Plugins | **CORRECT** |

---

## 20. Quality metrics summary

**Finding counts are NON-EXCLUSIVE.** One contract family may appear in multiple finding classes (e.g. **INTERNAL_LEAK** and **BYPASS**). These counts do **not** sum to total family count (**99**) and are not an arithmetic partition of the inventory.

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

## 22. Validation checklist (Phase 29 · EAC-2R1 gates)

| Gate | Result |
|------|--------|
| V1 — All contract families inventoried | **PASS** (§4 — **99** rows) |
| V2 — Each canonical public family: one semantic owner or explicit ADR/UNRESOLVED cell | **PASS** |
| V3 — Multi-owner semantic owner cells | **PASS** (**0**) |
| V4 — Consumers/providers not encoded as owner | **PASS** |
| V5 — Composite families split where required | **PASS** (EAC-CON-093…099) |
| V6 — ADR-GOV-01 remains unresolved and visible | **PASS** (§14, §16) |
| V7 — CL-EAC1-001…005 correctly mapped | **PASS** (§14) |
| V8 — Finding metrics marked non-exclusive | **PASS** (§20) |
| V9 — Inventory counts recomputed | **PASS** (§4 metrics) |
| V10 — EAC-3 dependency inputs unambiguous | **PASS** (§15) |
| V11 — No production code changed | **PASS** (documentation only) |
| V12 — No runtime contract / canonical authority changed | **PASS** |
| V13 — All 34 domains audited (EAC-2 baseline) | **PASS** (§5) |
| V14 — No ADR resolved | **PASS** |

---

*EAC-2 artifact @ **EAC2_EVIDENCE_HEAD** `e1cdaa77e0b1bf7608573001b23a42ebf870507f` · **EAC-2R1** ownership normalization @ **EAC2R1_EVIDENCE_HEAD** (see commit). Upstream: **EAC1_CLOSE_COMMIT** `588180053c908c5960d5bca099b14e7b217849bb` · baseline **EAC2_INDEPENDENT_AUDIT_HEAD** `cffd71062d9145a7b52c00dff1f02dcdfec2e262`.*

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
