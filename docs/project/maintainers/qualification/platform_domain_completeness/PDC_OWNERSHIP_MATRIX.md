# PDC-X — ownership matrix (PDC-P3-01)

**Ticket:** T-IGX-036-D · **Status:** draft for architecture lock (OD-PDC-ADR-01 pending).

Rule: every PDC concern has **exactly one semantic owner** and **one composition root** (FRZ-OWN-01..05). Rows marked *decided here* resolve dual-owner findings from P2. Every resolution was checked against DUP-X; DUP-X records **no** disposition for any of them, so none overrides a certified decision.

## Domain owners

| Domain | Concern | Semantic owner | Composition root | Not an owner (consumers or adapters) |
| --- | --- | --- | --- | --- |
| D1 | Agent identity and version | `contracts/agent_contract_meta.AgentContract` | `agent_distribution/admin_service.AgentPlatformAdminService` | `runtime/registry/AgentRegistry` (runtime projection only) |
| D1 | Install / bind / activate / upgrade / revoke / retire | `agent_distribution` services (installation, binding, activation, revision) | `AgentPlatformAdminService` | `runtime/agent_governance` (runtime tool authorization plane, separate concern) |
| D1 | Skill lifecycle | `skills/host_lifecycle.SkillHostLifecycleService` | skills registry runtime | agent distribution (consumes skills; does not own them) |
| D1 | Plugin discovery | `core/plugins/discovery.py` (*decided here*, PDC-D1-R020) | catalog bootstrap | `runtime/vendor_knowledge/contribution_catalog.py`, `rag/bootstrap/entry_point_load.py`: loaders that must delegate to or register through the core discovery owner (P5-D1) |
| D1 | Plugin qualification suites | **unowned today** → assigned to PDC-X D1 for certification (*decided here*, PDC-D1-R021) | — | `plug_03`, `plug_04`, `plug_final` |
| D2 | Context planning / budget / compaction / degradation / hard window | CE-02 table (`CONTEXT_ENGINEERING.md` §147–165) | `ContextCompiler` | adapters (no semantic truncation) |
| D2 | Memory | `memory/contracts/memory_control.MemoryControlPlane` | `memory/default_memory_control_plane.py` | memory providers |
| D2 | `ContextFragment` name | `context/contracts.ContextFragment` (canonical) | — | `runtime/token_optimization/context_pack.ContextFragment` is a different concept (*decided here*: N/A as duplicate, PDC-D2-R017; a rename is optional and not required for freeze) |
| D3 | Adapter ABI / provider registry / catalog | `LLMAdapter`, `LLMAdapterRegistry`, `ModelCatalog` | `llm_adapters/routing/composition.py` | `intergrax/llm` (message models), `intergrax/model_inference` (separate non-LLM vision/ML plane, own registry by modality) |
| D3 | Context-window data | `ModelCatalog` (*decided here*, PDC-D3-R023) | — | provider-local tables become fallbacks resolved through the catalog (P5-D3) |
| D4 | Hosting lifecycle | `hosting/engine/HostedApplicationEngine` | `hosting/runner.run_hosted_application` | `intergrax/supervisor` (experimental LangGraph planning prototype; not hosting) |
| D4 | Tier-3 environment | `ApplicationEnvironmentProfile` | `applications/_shared/environment_wiring.wire_application_environment` | `intergrax/fastapi_core` (HTTP adapter; DI-only) |
| D5 | Evaluation contracts | new `intergrax/contracts/evaluation_*` (ADR-PDC-05) | `applications/_shared/evaluation_wiring.py` | promotion: **Governance**, never evaluation |
| D6 | Authorization / policy | `runtime/policy/PolicyEngine` (GOV-X2) | `applications/_shared/security_assembly_resolver.py` | `runtime/agent_governance` consumes policy for tool authorization |
| D6 | Credentials | `integrations/credentials/ScopedCredentialBroker` | integrations composition | — |
| D6 | Security audit trail | evidence/persistence plane (ADR-PDC-06) | `applications/_shared/security_audit_trail_wiring.py` | `SecurityAuditTrail` becomes a writer, not a store |
| D7 | Task handler registration | `queueing/worker/registry.TaskExecutionRegistry` (*decided here*, PDC-D7-R022) | `queueing/registry.py` | `background_tasks/registry.TaskRegistry` + `WorkerRuntime`: layered facade that wraps and binds the execution registry; it is **not** a second authority |
| D7 | Delayed / recurring execution | `runtime/long_running/SchedulerLedger` + `LongRunningScheduler` | `runtime/long_running/wiring.py` | queue providers (no queue-level delay authority; PDC-D7-R005 N/A) |
| D8 | Business artifacts | `collaborative_work` | Collaborative Work services | — |
| D8 | Execution / audit evidence artifacts | evidence plane (`runtime/evidence`, `ArtifactStore` behind governed port) | `runtime/execution/run_artifact_bundle_composition.py` | — |
| D8 | Context artifacts | `runtime/context_lifecycle` (UCL) | UCL repository | — |
| D8 | Temporary tool output | tool runtime (spill-to-reference, ADR-PDC-08) | tool invocation composition | — |
| D8 | Task artifact contracts | `contracts/task_artifacts.py` (*decided here*, PDC-D8-R017) | — | `applications/contracts/application_artifacts.py` is removed under COMPAT-X R5 (0 production importers) |
| D8 | `ArtifactRef` name | `contracts/artifact_ref.ArtifactRef` (*decided here*, PDC-D8-R018) | — | `TraceArtifactRef` keeps its own name; the `ArtifactRef` aliases in `runtime/nexus/artifacts/models.py` and `runtime/nexus/tracing/trace_models.py` are removed |

## Composition roots (no new roots introduced)

`AgentPlatformAdminService` · `ContextCompiler` / `default_memory_control_plane` · `llm_adapters/routing/composition.py` · `run_hosted_application` / `wire_application_environment` · `evaluation_wiring` · `security_assembly_resolver` · `long_running/wiring` / `queueing/registry` · `run_artifact_bundle_composition`.
