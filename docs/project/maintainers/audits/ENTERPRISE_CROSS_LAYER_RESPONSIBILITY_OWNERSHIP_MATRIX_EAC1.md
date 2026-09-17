# EAC-1 — Enterprise Cross-Layer Responsibility & Ownership Matrix

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-1 — Responsibility & Ownership Matrix (**EAC-1R2** strict authority uniqueness & role separation applied)  
**Type:** Read-only architecture audit (no remediation)  
**Authority:** EAC-0 R1 inventory + canonical domain pairs on `development`

### Provenance (normalized — no ambiguous “baseline” labels)

| Field | SHA | Meaning |
|-------|-----|---------|
| **EAC1_ORIGINAL_BASELINE_HEAD** | `d5252dd5ee79ba57fff2769587463fb7165a4451` | EAC-0 R1 anchor used to seed EAC-1 domain inventory |
| **EAC1_PUBLISH_COMMIT** | `41a145332921de5d49f8f78099aca85d5591ef51` | Original EAC-1 matrix publish on `development` |
| **EAC1R1_SESSION_START_HEAD** | `9a6319e0e710029fb7ccfed1f1c2e7588cda4ff0` | Working tree @ start of EAC-1R1 hardening session |
| **EAC1R1_EVIDENCE_HEAD** | `5563c5921500bc3a57ca8e6d816bfd18f5d76d8b` | Repository state reconciled for EAC-1R1 taxonomy pass |
| **EAC1R1_HARDENING_COMMIT** | `ab6d578296e7b234a364c8dbb8570d318fe28f62` | Committed EAC-1R1 documentation hardening |
| **EAC1R2_SESSION_START_HEAD** | `af4d84e370062776d2f051d99cd5a7f74fc6e243` | Working tree @ start of EAC-1R2 (HEAD == `origin/development`) |
| **EAC1R2_EVIDENCE_HEAD** | `af4d84e370062776d2f051d99cd5a7f74fc6e243` | Repository state reconciled for EAC-1R2 taxonomy (claims through this SHA unless LEGACY/QUAL) |

| Gate | Value |
|------|-------|
| **Branch** | `development` |
| **HEAD == origin/development @ EAC-1R2 session** | **YES** (`af4d84e370062776d2f051d99cd5a7f74fc6e243`) |
| **Upstream inventory** | [`ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md`](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md) |
| **Registry hub** | [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md) |

**EAC-1R1 → EAC-1R2 drift (commits after `EAC1R1_HARDENING_COMMIT`, inspected for ownership semantics):**

| Commit | Area | EAC-1R2 effect |
|--------|------|----------------|
| `0425caa4e512764fd541fbf25d5b84db8b89d1a6` | Governance GR-10 strategy qualification | **QUALIFICATION ONLY** — GR-10 OPEN → **PARTIAL**; **GOVERNANCE_AUTHORITY** remains **GOVERNED_EXECUTION** |
| `af4d84e370062776d2f051d99cd5a7f74fc6e243` | Context budgeting / compaction | CE/UCL qualification surface; **no** peer authority owner change |

**Drift watch:** Uncommitted working-tree deltas outside `EAC1R2_EVIDENCE_HEAD` are **out of scope** for this artifact.

**Subordinate to:** per-domain architecture/plan pairs. This matrix **MUST NOT** redefine domain semantics.

**EAC-1R2 enterprise rule (authority vs role):**

```text
ONE AUTHORITY TYPE → ONE CANONICAL PEER OWNER
(subordinate/internal authority → one internal owner + explicit parent; not peer-level)
DOMAIN ROLE → descriptive; MAY be shared; does NOT transfer authority
```

---

## 1. Scope

EAC-1 establishes for each of **34** canonical DOMAIN rows:

- canonical **OWNS** / **MUST NOT OWN** / **MAY DELEGATE**
- **CONSUMES** / **PRODUCES**
- **AUTHORITATIVE** vs **DERIVED** state
- **PUBLIC** vs **INTERNAL** boundaries
- **canonical authority types** (peer or subordinate) vs **non-authoritative domain roles**
- delegation semantics, duplicate-authority classification
- pairwise hotspot verdicts and findings for EAC-2…EAC-16

**In scope:** cross-layer ownership, boundaries, contracts (baseline families), information-object semantic owners, plugin authority containment.

**Out of scope:** remediation, contract moves, persistence mechanism audit (EAC-9), full dependency proofs (EAC-3), replaceability proofs (EAC-4), internal algorithms (**ID** handoffs).

---

## 2. Baseline SHA & method

**Method (per domain):**

1. Reconstruct DOMAIN list from EAC-0 §16 (34 rows).
2. Read canonical `docs/project/architecture/<DOMAIN>.md` (+ maintainer plan where cited in EAC-0).
3. Cross-check freeze/qualification records referenced in EAC-0 §10.
4. Sample production boundary evidence (`intergrax/` layout, `intergrax/contracts/*`, tier import grep) — **semantics over folder names**.
5. Compare doc canon vs observable caller patterns; record **CL** / **DOC** / **CODE** / **ADR** / **QUAL** / **ID** / **LEGACY**.

**Primary sources:** (unchanged — see EAC-1R1) [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md), registry hub, `SYSTEM_INVARIANTS.md`, `UNIFIED_EXECUTION_RUNTIME.md`, per-domain architecture files (EAC-0 §16).

---

## 3. Responsibility vocabulary

| Term | Meaning |
|------|---------|
| **OWNS** | Final semantic authority; canonical writer for listed state |
| **MUST NOT OWN** | Explicitly delegated to another DOMAIN |
| **MAY DELEGATE** | May request work downstream; authority transfer only where contract says so |
| **CONSUMES** | Reads/contracts/events owned elsewhere |
| **PRODUCES** | Emits for downstream (may be derived if labeled) |
| **AUTHORITATIVE STATE** | Canonical mutable truth for that semantic |
| **DERIVED STATE** | Projection/cache/read-model; rebuildable; not second truth |
| **PUBLIC BOUNDARY** | Legal cross-layer contract surface |
| **INTERNAL BOUNDARY** | Implementation-only; external consumers MUST NOT depend |
| **AUTHORITY TYPE** | Final canonical semantic ownership over a decision, lifecycle, state, mutation, or externally meaningful capability — **exactly one peer owner per type** |
| **DOMAIN ROLE** | Descriptive architectural participation — **not** canonical authority; **may** be shared across domains |

**Enterprise rules (audit lens):** one capability → one canonical owner; one **peer** authority type → one canonical owner; no duplicate mutable truth; plugins do not become platform authority; **implementation location ≠ authority**.

---

## 4. Controlled taxonomies (EAC-1R2)

### 4.A AUTHORITY TYPE (peer — one canonical owner each)

Authority means: *final canonical semantic ownership over a decision, lifecycle, state, mutation, or externally meaningful capability.*

| Authority Type | Canonical Owner | Scope (summary) |
|----------------|-----------------|-----------------|
| **IDENTITY_AUTHORITY** | UNIFIED_EXECUTION_RUNTIME | Platform execution identity namespaces (Run/Attempt/Execution tree); other domains only in declared local namespaces |
| **EXECUTION_AUTHORITY** | UNIFIED_EXECUTION_RUNTIME | Run/Attempt/Execution tree coordination & strategy routing |
| **EXECUTION_LIFECYCLE_AUTHORITY** | UNIFIED_EXECUTION_RUNTIME | Canonical Run/Attempt/pause/resume/terminal transitions |
| **DECISION_AUTHORITY** | DECISION_SYSTEM | Authoritative WHAT outcome & decision record lifecycle |
| **GOVERNANCE_AUTHORITY** | GOVERNED_EXECUTION | WHETHER proceed / side-effect admission (platform) |
| **ORCHESTRATION_STRATEGY_AUTHORITY** | ORCHESTRATION | Orchestration **strategy class** semantics |
| **TOOL_INVOCATION_AUTHORITY** | TOOLS | Tool invocation contract & effect channel |
| **MEMORY_RECORD_AUTHORITY** | MEMORY | Durable memory records, projections, consolidation **policy surface** |
| **MEMORY_REFERENCE_READ_AUTHORITY** | MEMORY | Scoped canonical **reference-first** read (`MemoryReferenceReadPort`) |
| **RETRIEVAL_ORCHESTRATION_AUTHORITY** | RAG | Retrieval/query/index orchestration over memory read models (not memory truth) |
| **CONTEXT_ASSEMBLY_AUTHORITY** | CONTEXT_ENGINEERING | Prompt/context bundle assembly & compile plan |
| **CONTEXT_LIFECYCLE_AUTHORITY** | UNIFIED_CONTEXT_LIFECYCLE | Conversation context optimization lifecycle (trim/compaction policy) |
| **EVIDENCE_AUTHORITY** | OBSERVABILITY | RuntimeEvent journal / reconstruction |
| **DIAGNOSTIC_AUTHORITY** | DIAGNOSTICS | Problem / diagnostic interpretation lifecycle |
| **CAPABILITY_DISCOVERY_AUTHORITY** | CAPABILITY_CATALOG_AND_DISCOVERY | Federated read/rank of capabilities |
| **AGENT_DISTRIBUTION_AUTHORITY** | AGENT_DISTRIBUTION | Agent package install/materialize/activate (RuntimeRevision path) |
| **PLUGIN_LIFECYCLE_AUTHORITY** | PLATFORM_PLUGINS | Extension discovery, enablement, trust vocabulary (not domain semantics) |
| **AGENT_ASSEMBLY_AUTHORITY** | AGENT_CONTRACTS_AND_ASSEMBLY | Agent contract, assembly, harness kernel (not tool invocation) |
| **SKILL_COMPOSITION_AUTHORITY** | SKILLS | Skill composition over tools |
| **APPLICATION_COMPOSITION_AUTHORITY** | TIER3_APPLICATION_ENVIRONMENT | Application manifests, profiles, host wiring composition |
| **WORKSPACE_COMPOSITION_AUTHORITY** | COLLABORATIVE_WORK | Workspace, membership, delegation, ContextView authority |
| **HOST_DEPLOYMENT_LIFECYCLE_AUTHORITY** | APPLICATION_HOSTING | Deployment/hosting lifecycle (host scope; not EE Run tree) |
| **WORKER_INSTANCE_STATE_AUTHORITY** | AUTONOMOUS_WORK | WorkerDefinition/Instance durable work semantics |
| **CAPACITY_COORDINATION_AUTHORITY** | ELASTIC_CAPACITY_AND_SCALING | Capacity admission, scale coordination leases |
| **BACKGROUND_DELIVERY_AUTHORITY** | BACKGROUND_TASKS | Queue/bus abstractions, worker intake & queue job state |
| **FAILURE_CLASSIFICATION_AUTHORITY** | RELIABILITY_FAILURE_AND_HITL | Failure taxonomy / classification |
| **RECOVERY_POLICY_AUTHORITY** | RELIABILITY_FAILURE_AND_HITL | Bounded retry/degrade/compensate **policy selection** |
| **HITL_INTERACTION_AUTHORITY** | RELIABILITY_FAILURE_AND_HITL | Human escalation, interrupt interaction, HITL decision records |
| **LLM_PROVIDER_CONTRACT_AUTHORITY** | LLM_ADAPTERS | LLM provider abstraction & model call contracts |

**Deprecated authority labels (EAC-1R2 — do not use as peer types):** generic `LIFECYCLE_AUTHORITY`, `ORCHESTRATION_AUTHORITY`, `CONTEXT_AUTHORITY`, `DISTRIBUTION_AUTHORITY`, `RETRIEVAL_AUTHORITY`, `COMPOSITION_AUTHORITY`, `SCALE_COORDINATION`, `PERSISTENCE_AUTHORITY`, `PROVIDER_ABSTRACTION`, `DEVELOPMENT_ONLY`, `DISCOVERY_AUTHORITY` (use scoped types above).

Subordinate hubs (Execution Continuation, NPSC-5E recovery plane, Marketplace Engine doc, ERL sub-hubs) **do not** receive separate **peer** DOMAIN authority — parent DOMAIN rows below.

### 4.B DOMAIN ROLE (non-authoritative — may be shared)

| Domain Role | Typical domains | Meaning | Authority transfer? |
|-------------|-----------------|---------|-------------------|
| **PROVIDER_ABSTRACTION_ROLE** | LLM_ADAPTERS, INTEGRATIONS, MODALITY, REASONING_AND_COGNITION | Vendor/model/driver indirection & adapter wiring | **NO** (contract authority named separately where it exists, e.g. LLM_PROVIDER_CONTRACT_AUTHORITY) |
| **DEVELOPMENT_SUPPORT_ROLE** | PLATFORM_FOUNDATION, EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE, PROOF_RECEIPTS, ADAPTIVE_HARNESS_INTELLIGENCE | DX, meta, qualification, research hooks — non-runtime truth | **NO** |
| **COMPOSITION_ROLE** | CODE_CRAFT, APPLICATION_HOSTING (wiring adjunct) | Participates in composition stacks without owning a scoped composition authority | **NO** |
| **PERSISTENCE_PARTICIPANT_ROLE** | Any domain using stores via ports | Uses persistence mechanisms; semantic truth owner is always a named **authority type** | **NO** |
| **EXTENSION_PROVIDER_ROLE** | PLATFORM_PLUGINS (loader), domain plugin hosts | Implements platform-defined contracts; validation remains domain owner | **NO** |
| **INTERNAL_IMPLEMENTATION_ROLE** | NEXUS_EXECUTION_FLOW (graph engine locus) | Implementation under parent strategy/admission | **NO** (scheduling authority is **subordinate**, §4.C) |

### 4.C SUBORDINATE / INTERNAL AUTHORITY

Subordinate authority is **valid only inside** the parent-owned boundary and **MUST NOT** be cited as peer-level platform authority.

| Subordinate Authority Type | Internal Owner | Parent peer authority | Validity boundary |
|----------------------------|----------------|----------------------|-------------------|
| **INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY** | NEXUS_EXECUTION_FLOW | ORCHESTRATION_STRATEGY_AUTHORITY + EE admission/lifecycle | Private graph/step scheduling under strategy; **no** public root API |
| **EFFECT_RECONCILIATION_AUTHORITY** | ENTERPRISE_RELIABILITY_LAYER | RECOVERY_POLICY_AUTHORITY (platform resilience plane) | External effect reliability / reconciliation composition slice only |

---

## 4.1 Strict peer authority register (EAC-1R2 gate)

| Authority Type | Canonical Owner | Scope | Subordinate Authority | Competing Peer Owner | Verdict |
|----------------|-----------------|-------|------------------------|----------------------|---------|
| EXECUTION_LIFECYCLE_AUTHORITY | UNIFIED_EXECUTION_RUNTIME | Platform run tree | — | NONE | PASS |
| EXECUTION_AUTHORITY | UNIFIED_EXECUTION_RUNTIME | Coordination & routing | — | NONE | PASS |
| DECISION_AUTHORITY | DECISION_SYSTEM | Decision records | — | NONE | PASS |
| GOVERNANCE_AUTHORITY | GOVERNED_EXECUTION | Platform WHETHER | — | **ADR REQUIRED** (CW declarative overlap — not second peer type) | PASS* |
| TOOL_INVOCATION_AUTHORITY | TOOLS | Invoke spine | — | NONE | PASS |
| MEMORY_RECORD_AUTHORITY | MEMORY | Memory truth | — | NONE | PASS |
| MEMORY_REFERENCE_READ_AUTHORITY | MEMORY | Reference-first read port | — | NONE | PASS |
| RETRIEVAL_ORCHESTRATION_AUTHORITY | RAG | Query/index orchestration | — | NONE | PASS |
| ORCHESTRATION_STRATEGY_AUTHORITY | ORCHESTRATION | Strategy class | INTERNAL_ORCHESTRATION_SCHEDULING → NEXUS | NONE | PASS |
| AGENT_DISTRIBUTION_AUTHORITY | AGENT_DISTRIBUTION | Agent packages | — | NONE | PASS |
| PLUGIN_LIFECYCLE_AUTHORITY | PLATFORM_PLUGINS | Extension enablement | — | NONE | PASS |
| AGENT_ASSEMBLY_AUTHORITY | AGENT_CONTRACTS_AND_ASSEMBLY | Agent/harness assembly | — | NONE | PASS |
| SKILL_COMPOSITION_AUTHORITY | SKILLS | Skills over tools | — | NONE | PASS |
| APPLICATION_COMPOSITION_AUTHORITY | TIER3_APPLICATION_ENVIRONMENT | T3 manifests/profiles | — | NONE | PASS |
| WORKSPACE_COMPOSITION_AUTHORITY | COLLABORATIVE_WORK | Workspace/ContextView | — | **ADR REQUIRED** (GE admission vs CW policy — CL-EAC1-004) | PASS* |
| CAPACITY_COORDINATION_AUTHORITY | ELASTIC_CAPACITY_AND_SCALING | Leases/admission | — | NONE | PASS |
| BACKGROUND_DELIVERY_AUTHORITY | BACKGROUND_TASKS | Queues/worker intake | — | NONE | PASS |
| CAPABILITY_DISCOVERY_AUTHORITY | CAPABILITY_CATALOG_AND_DISCOVERY | Catalog read/rank | — | NONE | PASS |
| CONTEXT_ASSEMBLY_AUTHORITY | CONTEXT_ENGINEERING | Bundle assembly | — | NONE | PASS |
| CONTEXT_LIFECYCLE_AUTHORITY | UNIFIED_CONTEXT_LIFECYCLE | UCL optimization | — | NONE | PASS |
| EVIDENCE_AUTHORITY | OBSERVABILITY | RuntimeEvent journal | — | NONE | PASS |
| DIAGNOSTIC_AUTHORITY | DIAGNOSTICS | Problem store | — | NONE | PASS |
| FAILURE_CLASSIFICATION_AUTHORITY | RELIABILITY_FAILURE_AND_HITL | Taxonomy | — | NONE | PASS |
| RECOVERY_POLICY_AUTHORITY | RELIABILITY_FAILURE_AND_HITL | Policy selection | EFFECT_RECONCILIATION → ERL | NONE | PASS |
| HITL_INTERACTION_AUTHORITY | RELIABILITY_FAILURE_AND_HITL | HITL records | — | NONE | PASS |
| HOST_DEPLOYMENT_LIFECYCLE_AUTHORITY | APPLICATION_HOSTING | Host deploy lifecycle | — | NONE | PASS |
| WORKER_INSTANCE_STATE_AUTHORITY | AUTONOMOUS_WORK | Worker instances | — | NONE | PASS |
| LLM_PROVIDER_CONTRACT_AUTHORITY | LLM_ADAPTERS | LLM contracts | — | NONE | PASS |
| IDENTITY_AUTHORITY | UNIFIED_EXECUTION_RUNTIME | Execution IDs | — | NONE | PASS |

\* **PASS** at taxonomy layer: no duplicate **peer** owner for the type; real GE↔CW **documentation conflict** tracked as **CL-EAC1-004** / **ADR-GOV-01** (not resolved in EAC-1R2).

### 4.2 Authority uniqueness map (mechanical)

```text
∀ peer AUTHORITY_TYPE: canonical_owner_count == 1
```

**EAC-1R2:** **29** peer authority types in §4.A; **2** subordinate authority types in §4.C; **0** peer types with dual canonical owners.

---

## 5. Thirty-four domain responsibility matrix

**Row count verification:** **34** (= EAC-0 §16).

| ID | Domain | OWNS | MUST NOT OWN | MAY DELEGATE | CONSUMES | PRODUCES | Authoritative State | Derived State | Public Boundary | Internal Boundary | Canonical Authority | Domain Roles | Confidence | Risk |
|----|--------|------|--------------|--------------|----------|----------|---------------------|---------------|-----------------|-------------------|---------------------|--------------|------------|------|
| EAC-DOM-001 | PLATFORM_FOUNDATION | Tier topology, import boundaries, spine CI gates, platform invariants registry | Domain feature semantics, runtime lifecycle | None (meta) | All domains (read-only index) | Tier rules, SYS-INV-* | None runtime | Doctor reports | `SYSTEM_INVARIANTS`, tier contract docs | CI scripts internals | **NONE** | DEVELOPMENT_SUPPORT_ROLE | STRONG | LOW |
| EAC-DOM-002 | UNIFIED_EXECUTION_RUNTIME | Run/Attempt lifecycle, ExecutionId coordination, strategy routing, terminal semantics, lifecycle fact emission, checkpoint **ports** coordination | Decision WHAT, governance WHETHER, Observability journal, Problem truth, Nexus orchestration internals | Orchestration strategies, Governance checks, Observability record, Recovery providers | Governance decisions, budget, checkpoint store ports | Lifecycle events, execution artifacts | Run, Attempt, Execution tree state, continuation tokens (UEA) | Scheduling projections inside strategies | `intergrax.contracts.execution*`, UEA | Engine adapter graphs, private routers | EXECUTION_AUTHORITY, EXECUTION_LIFECYCLE_AUTHORITY, IDENTITY_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE (checkpoint ports) | PROVEN | LOW |
| EAC-DOM-003 | ORCHESTRATION | Orchestration **strategy class** semantics, orchestration contracts as execution strategy | Execution identity/lifecycle, public Nexus API, admission | Nexus graph execution (internal) | Execution admission context | Strategy outcomes to EE | None (stateless strategy role) | Graph specs in flight | Orchestration contracts | Strategy impl details | ORCHESTRATION_STRATEGY_AUTHORITY | — | STRONG | MEDIUM |
| EAC-DOM-004 | NEXUS_EXECUTION_FLOW | Graph execution, fan-out/merge, internal scheduling **under** orchestration strategy | Public root API, execution admission, Run lifecycle, ExecutionId mint, governance | Tool/agent steps via contracts | EE checkpoints, execution context | Orchestration progress (internal) | Nexus topology/scheduling **projections** (non-canonical tree) | Readiness caches | **None** at platform root — consumer MUST use EE contracts | `NexusLoop`, `GraphExecutor`, `intergrax/runtime/nexus/context/*` | **INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY** (subordinate) | INTERNAL_IMPLEMENTATION_ROLE | STRONG | MEDIUM |
| EAC-DOM-005 | DECISION_SYSTEM | Decision lifecycle, authoritative outcome, version lineage, DecisionStrategy plugins | Execution lifecycle, authorization WHETHER, retry orchestration | Reasoning for material; Execution for hosting | Reasoning artifacts, evidence | Decision artifacts/records | Decision, DecisionVersion, outcome state | Deliberation scratch | `intergrax.contracts.decision*` | Strategy impl modules | DECISION_AUTHORITY | — | PROVEN | LOW |
| EAC-DOM-006 | GOVERNED_EXECUTION | Policy evaluation, side-effect admission, governance evidence (WHETHER) | Decision WHAT, execution scheduling, diagnostic Problem semantics | Policy plugins, external policy stores | Execution context, tool manifests | Allow/deny/interrupt decisions | Policy grants, admission records | Policy evaluation caches | `intergrax.contracts.governance*` | Policy engine internals | GOVERNANCE_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | PARTIAL | HIGH |
| EAC-DOM-007 | REASONING_AND_COGNITION | Cognitive strategies, reasoning material, candidate generation | Authoritative decision outcome, governance | LLM adapters | Context bundles, memory projections | Reasoning artifacts (non-final) | Session/scratch reasoning state | Cached chains | Reasoning contracts | Strategy internals | **NONE** | PROVIDER_ABSTRACTION_ROLE | PARTIAL | MEDIUM |
| EAC-DOM-008 | AGENT_CONTRACTS_AND_ASSEMBLY | Agent contract, assembly, author behavior, harness kernel, step loop contract | Nexus planning, execution admission, governance rules, **tool gateways / vendor execution / policy engines** | Tools, skills, LLM, context (via contracts) | Context bundles, **tool contracts** (`intergrax.contracts.tools*`), governance hooks | Agent step requests, **tool intent/request** (not invocation) | Agent merge hooks state (bounded) | Assembly caches | Agent/`HarnessKernel` contracts | Harness private loops | AGENT_ASSEMBLY_AUTHORITY | COMPOSITION_ROLE (stack participant) | STRONG | MEDIUM |
| EAC-DOM-009 | AGENT_DISTRIBUTION | Agent package distribution, installability, RuntimeRevision path | Marketplace catalog engine, execution lifecycle, **plugin lifecycle** | Catalog read for discovery | Catalog entries (read) | Installed agent metadata | Package install state | Discovery projections | `intergrax.contracts.agent_distribution*` | Installer internals | AGENT_DISTRIBUTION_AUTHORITY | — | STRONG | MEDIUM |
| EAC-DOM-010 | LLM_ADAPTERS | LLM provider abstraction, model call contracts | Agent planning ownership, governance | Vendor APIs | Provider config | Model outputs (non-authoritative) | None platform | Client caches | LLM provider contracts | Driver specifics | LLM_PROVIDER_CONTRACT_AUTHORITY | PROVIDER_ABSTRACTION_ROLE | STRONG | LOW |
| EAC-DOM-011 | TOOLS | Tool contracts, invocation spine, idempotency contract surface | Integration drivers, governance rule definitions | Integrations for IO | Governance admission, integrations | Tool results, effect records | Idempotency keys (tool scope) | Invocation telemetry | `intergrax.contracts.tools*` | Driver wiring | TOOL_INVOCATION_AUTHORITY | — | STRONG | MEDIUM |
| EAC-DOM-012 | SKILLS | Skill composition over tools (reusable capability) | Tool driver semantics, marketplace lifecycle | Tools | Tool registry | Composed skill invocations | None | Resolver caches | Skill contracts | Skill graph internals | SKILL_COMPOSITION_AUTHORITY | — | STRONG | LOW |
| EAC-DOM-013 | INTEGRATIONS | Integration profiles, provider wiring | Tool semantic contracts, execution lifecycle | Vendor systems | Credentials profiles | Integration IO | Vendor connection state | Profile caches | Integration contracts | Connector internals | **NONE** | PROVIDER_ABSTRACTION_ROLE | STRONG | LOW |
| EAC-DOM-014 | RAG | Retrieval orchestration, chunk/query semantics | Memory authoritative store, execution lifecycle | Memory projections, index providers | Memory read models, indexes | Retrieval results (derived reads) | Index segments (retrieval scope) | Query caches | RAG contracts | Pipeline impl | RETRIEVAL_ORCHESTRATION_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE (indexes) | PARTIAL | MEDIUM |
| EAC-DOM-015 | MEMORY | Memory records, projections, consolidation **policy surface**; scoped **reference-first read** (`MemoryReferenceReadPort`) | ContextView composition, CE assembly, foreign ContextView types, execution lifecycle, diagnostic Problems | Memory providers | Observability (audit only) | Memory records/projections; **canonical memory references** | Memory record truth (provider-backed) | Projection read models | Memory provider contracts; `intergrax/memory/contracts/memory_reference_read.py` | Store implementations | MEMORY_RECORD_AUTHORITY, MEMORY_REFERENCE_READ_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | PARTIAL | MEDIUM |
| EAC-DOM-016 | CONTEXT_ENGINEERING | Context assembly, collectors, compile plan | UCL optimization ownership, raw memory truth, execution | Memory/RAG/modality readers (**MemoryReferenceReadPort** / CE adapters) | Memory refs & projections, RAG, modality, CE policy | Context bundles for consumers | Assembly plan state (ephemeral) | Compiled context caches | Context contracts (`intergrax/context/`) | Collector registry internals | CONTEXT_ASSEMBLY_AUTHORITY | — | PARTIAL | MEDIUM |
| EAC-DOM-017 | UNIFIED_CONTEXT_LIFECYCLE | Conversation context **lifecycle optimization** (trim/compaction policy) | Raw memory authority, CE collector rules, execution | CE execution of plans | CE bundles, token budgets | Optimized context views | UCL optimization state | Compaction journals | UCL contracts (ADR-UCL-001) | Optimizer internals | CONTEXT_LIFECYCLE_AUTHORITY | — | PARTIAL | MEDIUM |
| EAC-DOM-018 | MODALITY | Multimodal ingestion contracts | Execution, memory truth | Adapters | Media sources | Normalized modality payloads | Media artifact refs | Transcode caches | Modality contracts | Adapter internals | **NONE** | PROVIDER_ABSTRACTION_ROLE | UNKNOWN | MEDIUM |
| EAC-DOM-019 | OBSERVABILITY | RuntimeEvent recording, reconstruction, export | Problem semantics, execution control admission | Export sinks | Lifecycle facts from EE | Evidence journal, reconstructions | **RuntimeEvent** journal (canonical evidence) | Export/read models | `intergrax.contracts.observability*` | Indexer internals | EVIDENCE_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | PROVEN | LOW |
| EAC-DOM-020 | DIAGNOSTICS | Deterministic interpretation → **Problem** state, operator read models | Evidence minting, retry/lifecycle control, execution identity | Detector plugins | RuntimeEvent/reconstruction | Problem records, assessments | **Problem** lifecycle store | Grouping hypotheses | `intergrax.contracts.diagnostics` | Detector pipelines | DIAGNOSTIC_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | STRONG | MEDIUM |
| EAC-DOM-021 | RELIABILITY_FAILURE_AND_HITL | Failure classification, recovery policy selection, bounded retry/degrade/compensate **recommendation**, HITL escalation & interaction records | Governance ALLOW/DENY, **canonical** pause/resume / execution lifecycle (EE), Problem truth | Human decision stores, EE for lifecycle consequences | EE lifecycle facts, checkpoints | Interrupt/resume **signals** (EE applies lifecycle); HITL decision records | HITL decision records (bounded); resilience policy artifacts | Attempt ledger projections (derived) | HITL contracts | Policy classifiers | FAILURE_CLASSIFICATION_AUTHORITY, RECOVERY_POLICY_AUTHORITY, HITL_INTERACTION_AUTHORITY | — | PARTIAL | MEDIUM |
| EAC-DOM-022 | ADAPTIVE_HARNESS_INTELLIGENCE | Harness intelligence, design-search hooks (non-prod authority) | Production governance authority | Research runtimes | Telemetry | Research artifacts | Research state | Experiments | AHI contracts (immature) | Search internals | **NONE** | DEVELOPMENT_SUPPORT_ROLE | PARTIAL | LOW |
| EAC-DOM-023 | ELASTIC_CAPACITY_AND_SCALING | Capacity admission, scale coordination leases | EE identity/recovery semantics | Workers/runtime | Capacity signals | Admission decisions | Lease metadata | Metrics rollups | Capacity contracts | Scheduler internals | CAPACITY_COORDINATION_AUTHORITY | — | PARTIAL | MEDIUM |
| EAC-DOM-024 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | Dev workflows, harness DX, qualification tooling boundaries | Runtime semantic ownership | Local proofs | Docs/tooling | DX artifacts | None production | Local caches | N/A (guides) | Cursor rules, scripts | **NONE** | DEVELOPMENT_SUPPORT_ROLE | STRONG | NONE |
| EAC-DOM-025 | TIER3_APPLICATION_ENVIRONMENT | Application manifests, profiles, host wiring composition | Execution engine internals | Hosting, plugins | Platform contracts | Application config bindings | Host profile state | Scaffold templates | Application env contracts | App-specific code | APPLICATION_COMPOSITION_AUTHORITY | — | STRONG | MEDIUM |
| EAC-DOM-026 | APPLICATION_HOSTING | Deployment/hosting lifecycle, always-on runtime hosting (**host scope**, not platform execution tree) | Tier-3 manifest semantics (owned by T3), canonical EE Run lifecycle | Execution admission | T3 manifests | Deployment records | Deployment/host runtime state | Health projections | Hosting contracts | Orchestrator internals | HOST_DEPLOYMENT_LIFECYCLE_AUTHORITY | COMPOSITION_ROLE (host wiring adjunct) | PARTIAL | MEDIUM |
| EAC-DOM-027 | CODE_CRAFT | Code manipulation domain services | Agent assembly ownership | Workspace IO | VCS/workspace | Code artifacts | Workspace drafts | Analysis caches | CodeCraft contracts | Tooling internals | **NONE** | COMPOSITION_ROLE | UNKNOWN | LOW |
| EAC-DOM-028 | AUTONOMOUS_WORK | WorkerDefinition/Instance durable work semantics | Execution lifecycle owner (EE) | Background intake | Execution, catalog | Worker instances | Worker state | Queue views | AW contracts | Worker runtime | WORKER_INSTANCE_STATE_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | PARTIAL | MEDIUM |
| EAC-DOM-029 | COLLABORATIVE_WORK | Workspace, membership, delegation, ContextView authority | Execution graph, decision outcome, governance WHETHER (platform) | GE for tool auth (declared overlap) | GE, Decision read | CW persistence events | Workspace/membership/ContextView | Presence projections | CW contracts | MP internals | WORKSPACE_COMPOSITION_AUTHORITY | **delegation to GE** — not peer GOVERNANCE_AUTHORITY | PARTIAL | HIGH |
| EAC-DOM-030 | BACKGROUND_TASKS | Queue/bus abstractions, worker intake | Execution identity/lifecycle | EE for execution | Task envelopes | Queue metadata | Queue job state | Consumer lag metrics | `TaskQueue`, BG contracts | Broker adapters | BACKGROUND_DELIVERY_AUTHORITY | PERSISTENCE_PARTICIPANT_ROLE | PARTIAL | MEDIUM |
| EAC-DOM-031 | CAPABILITY_CATALOG_AND_DISCOVERY | Federated catalog read, rank, govern discovery rows | Vertical lifecycle, execution, Nexus | Domain registries | Domain sources | Catalog snapshots | Catalog index (read model) | Rank caches | Catalog contracts | Federator internals | CAPABILITY_DISCOVERY_AUTHORITY | — | STRONG | MEDIUM |
| EAC-DOM-032 | PROOF_RECEIPTS | Qualification receipt semantics | Runtime execution truth | CI/proof runners | Proof outputs | Receipt records | Receipt store | Verification views | Proof contracts | Gate scripts | **NONE** | DEVELOPMENT_SUPPORT_ROLE | STRONG | NONE |
| EAC-DOM-033 | PLATFORM_PLUGINS | Extension discovery, enablement, trust vocabulary | Domain semantic ownership, **agent package distribution** | Domain validation | Packaged extensions | Plugin enablement state | Enablement registry | Discovery cache | EP contracts | Loader internals | PLUGIN_LIFECYCLE_AUTHORITY | EXTENSION_PROVIDER_ROLE | STRONG | LOW |
| EAC-DOM-034 | ENTERPRISE_RELIABILITY_LAYER | External effect reliability, UNKNOWN handling, reconciliation **composition** | Governance ALLOW/DENY (GE), canonical execution lifecycle | Provider invocations | Tool effects, observability | ERL assessment artifacts | ProviderInvocation state (emerging) | Reconciliation projections | ERL contracts (emerging) | Submodule hubs | **EFFECT_RECONCILIATION_AUTHORITY** (subordinate) | — | PARTIAL | MEDIUM |

### 5.1 Ownership confidence & risk (aggregate)

| Confidence | Count |
|------------|------:|
| PROVEN | 3 |
| STRONG | 18 |
| PARTIAL | 11 |
| CONFLICT | 0 |
| UNKNOWN | 2 |

| Risk | Count |
|------|------:|
| NONE | 2 |
| LOW | 6 |
| MEDIUM | 20 |
| HIGH | 2 |
| CRITICAL | 0 |

### 5.2 Peer authority coverage (EAC-1R2)

| Metric | Value |
|--------|------:|
| Peer authority types (§4.A) | 29 |
| Domain role types (§4.B) | 6 |
| Subordinate authority types (§4.C) | 2 |
| Peer types with exactly one canonical owner | 29 |
| Peer types with competing canonical owners | 0 |
| Architecture conflicts (ADR / CL register) | GE↔CW (**CL-EAC1-004**); not taxonomy duplicates |

---

## 6. Public vs internal boundaries (summary)

| Domain | PUBLIC (legal consumers) | INTERNAL (must not escape) | Boundary findings |
|--------|--------------------------|----------------------------|-------------------|
| UNIFIED_EXECUTION_RUNTIME | `intergrax.contracts.execution*`, admission APIs | Private engine graph | — |
| NEXUS_EXECUTION_FLOW | **No public root** — only via EE/ORCHESTRATION | `NexusLoop`, `GraphExecutor`, `intergrax/runtime/nexus/context/*` | **CL-BOUNDARY-VIOLATION-EAC1-001** — Tier-2 `agents/*` import `RuntimeContext`, `SessionManager`, notebooks import `NexusLoop` (grep @ `EAC1R2_EVIDENCE_HEAD`) |
| DECISION_SYSTEM | `intergrax.contracts.decision*` | Strategy impl | — |
| GOVERNED_EXECUTION | `intergrax.contracts.governance*` | Policy engine private | QUAL: enterprise cert NOT CERTIFIED; GR-10 **PARTIAL** @ `0425caa4` (qualification only) |
| OBSERVABILITY | Observability contracts, reconstruction APIs | Journal writers private | — |
| DIAGNOSTICS | Diagnostics contracts, Problem read APIs | Detectors | DOC: ADR-REG-003 plan topology |
| MEMORY | Memory provider ports; **`MemoryReferenceReadPort`** | Store drivers; ContextView types in Memory | Tier-3 wiring via `intergrax/applications/_shared/memory_wiring.py` → Nexus `SessionManager` (**CL-EAC1-002** narrowed) |
| CONTEXT_ENGINEERING | `intergrax/context` contracts | Collectors/registry; Nexus `ContextCompiler` is **consumer/impl locus**, not semantic owner | CE-02 qualification; semantic owner = CE per `CONTEXT_ENGINEERING.md` (**CL-EAC1-003**) |
| CAPABILITY_CATALOG_AND_DISCOVERY | Catalog contracts | Federator | Canon: MUST NOT import Nexus |
| TIER3 / HOSTING | App env + hosting contracts | App code | Overlap risk manifest vs deployment (§9) |

---

## 7. Authoritative state ownership

| State | Canonical writer | Canonical readers | Derived copies | Persistence boundary | Duplicate risk |
|-------|------------------|-------------------|----------------|----------------------|----------------|
| Run / Attempt / Execution tree | UNIFIED_EXECUTION_RUNTIME | Observability, Diagnostics, Governance | Task projections, BG queue views | EE checkpoint ports | **NO CONFLICT** (Nexus projections non-canonical per NEXUS doc) |
| Decision / DecisionVersion | DECISION_SYSTEM | Execution, CW (read) | Reasoning scratch | Decision artifact store | **NO CONFLICT** if strategies do not finalize (monitor Reasoning) |
| Governance grant / admission | GOVERNED_EXECUTION | Tools, Execution | Policy caches | Policy store | **DOCUMENTATION CONFLICT** vs CW declarative tool auth (ADR-GOV-01) |
| RuntimeEvent journal | OBSERVABILITY | DIAGNOSTICS, export | Read models | Observability store | **NO CONFLICT** |
| Problem | DIAGNOSTICS | Operators, automation (read) | Grouping hypotheses | Problem store | **NO CONFLICT** (DIAG must not mint evidence) |
| Memory record | MEMORY (**MEMORY_RECORD_AUTHORITY**) | RAG, CE, UCL | Projections | Provider stores | **NO CONFLICT** on read boundary @ `616af2f4`; **wiring/persistence** risk remains (**CL-EAC1-002** NARROWED) |
| Memory canonical reference | MEMORY (**MEMORY_REFERENCE_READ_AUTHORITY**) | CE / ContextView adapters | — | Ephemeral adapter caches | **NO CONFLICT** |
| Context bundle (assembled) | CONTEXT_ENGINEERING | Agent, Nexus consumer | UCL optimized views | Ephemeral + caches | **LOW (architecture debt)** — Nexus context modules = **implementation placement** (**CL-EAC1-003**) |
| Retrieval result | RAG (**RETRIEVAL_ORCHESTRATION_AUTHORITY**) | CE, agents | Indexes | RAG indexes | **NO CONFLICT** — RAG does not own memory truth |
| Tool invocation / effect | TOOLS (+ GE admission) | ERL, Observability | Idempotency cache | Tool stores | **REAL DUPLICATE CANDIDATE** GE vs CW side-effect path (ADR-GOV-01) |
| Catalog snapshot row | CAPABILITY_CATALOG_AND_DISCOVERY | Marketplace UI | Rank caches | Catalog index | **NO CONFLICT** |
| Agent install / RuntimeRevision | AGENT_DISTRIBUTION | Execution, Catalog | Discovery | Package store | **NO CONFLICT** with Catalog read; not **PLUGIN_LIFECYCLE** |
| Plugin enablement registry | PLATFORM_PLUGINS | Domains | Discovery cache | EP store | **NO CONFLICT** with **AGENT_DISTRIBUTION_AUTHORITY** |
| Worker instance | AUTONOMOUS_WORK | BG, Execution | Queue metadata | AW store | **PARTIAL** — EE still owns execution lifecycle |
| Queue job state | BACKGROUND_TASKS | Workers | Lag metrics | Queue store | **NO CONFLICT** with **CAPACITY_COORDINATION** |
| Capacity leases | ELASTIC_CAPACITY_AND_SCALING | Workers | Metrics rollups | Lease store | **NO CONFLICT** with BG delivery |
| Continuation / checkpoint payload | UNIFIED_EXECUTION_RUNTIME (coordination) | Recovery providers | Reliability projections | Checkpoint ports (NPSC-5E) | **DERIVED ONLY** for Reliability Attempt Ledger |
| HITL interrupt decision | RELIABILITY_FAILURE_AND_HITL | EE (consequences) | — | Decision store | **NO CONFLICT** if EE owns pause/resume lifecycle |

---

## 8. Delegation semantics (cross-layer)

| Flow | Request owner | Proceed decision | Executor | Result state owner | Failure semantics owner |
|------|---------------|------------------|----------|-------------------|-------------------------|
| Governed run | EE | GE (WHETHER) + EE (lifecycle) | EE → strategy | EE lifecycle; GE admission record | EE terminal; GE deny |
| Decision inside run | Decision System | Decision authority | EE hosts strategy | Decision outcome | Decision + EE lifecycle |
| Tool side effect | Tools | GE (and CW policy per workspace) | Tool driver | Effect record / ERL | Tools + GE |
| Context for step | CE (assembly) | CE/UCL policy | CE collectors | Ephemeral bundle; Memory unchanged | CE/UCL |
| Evidence | EE emits facts | N/A | Observability persists | Observability journal | Observability export |
| Problem detection | Diagnostics job | Diagnostics rules | Detector plugins | Problem store | Diagnostics |
| Catalog handoff | Catalog | N/A (read) | N/A | Domain lifecycle owner on acquire | Distribution domain |
| Agent activate | Distribution | GE/EE admission | EE | RuntimeRevision | Distribution + EE |

**Rule:** delegation MUST NOT silently transfer authority — CW/GE tool auth requires ADR resolution (§15).

---

## 9. Pairwise conflict matrix (hotspots)

| Concern | Canonical owner | Adjacent layer | Allowed relationship | Current evidence | Conflict? | Finding |
|---------|-----------------|----------------|----------------------|------------------|-----------|---------|
| execution lifecycle | UNIFIED_EXECUTION_RUNTIME | NEXUS | Nexus schedules inside strategy; EE owns tree | `UNIFIED_EXECUTION_RUNTIME.md`, `NEXUS_EXECUTION_FLOW.md` | No | — |
| execution identity | UNIFIED_EXECUTION_RUNTIME | NEXUS | No OrchestrationRunId / competing tree | NEXUS-INV-011, UEA | No | — |
| orchestration | ORCHESTRATION | NEXUS | Strategy delegates graph to Nexus internal | Registry §7 | No | — |
| decision outcome | DECISION_SYSTEM | REASONING | Reasoning produces material only | Decision FROZEN hub | No* | QUAL: monitor app-local judges (**LEGACY**) |
| governance authorization | GOVERNED_EXECUTION | COLLABORATIVE_WORK | CW workspace policy; GE platform admission | RB-5 **F**, ADR-GOV-01 | **Yes (DOC)** | CL-EAC1-004 |
| tool side effects | GOVERNED_EXECUTION | TOOLS | Tools invoke; GE admits | U5 zero-bypass qual | Partial | CL-EAC1-004 |
| memory truth | MEMORY | RAG / CE | Read projections; no second writer | MEMORY.md, `MemoryReferenceReadPort` @ `616af2f4` | Partial (wiring only) | CL-EAC1-002 **NARROWED** |
| memory reference read | MEMORY | CE / ContextView consumers | Scoped reference-first port | `memory_reference_read.py` | No | — |
| retrieval orchestration | RAG | MEMORY | RAG orchestrates reads; Memory owns refs/truth | RAG + MEMORY canon | No | EAC1R2-TAX-001 |
| context assembly | CONTEXT_ENGINEERING | UCL / Nexus | CE assembles; UCL optimizes lifecycle | ADR-CTX-01 | Low (impl locus) | CL-EAC1-003 |
| evidence | OBSERVABILITY | DIAGNOSTICS | DIAG consumes; no mint | DIAGNOSTICS.md §SSOT | No | — |
| diagnostics | DIAGNOSTICS | OBSERVABILITY | Problem vs RuntimeEvent | R1 freeze | No | — |
| continuation | UNIFIED_EXECUTION_RUNTIME | RELIABILITY | EE owns pause/resume lifecycle | UER §HITL consequences | No | — |
| recovery | UNIFIED_EXECUTION_RUNTIME (5E plane) | RELIABILITY | Recovery ports; attempt ledger derived | NPSC-5E freeze | No | — |
| HITL | RELIABILITY (+ GE) | EE | Human decision vs lifecycle consequence split | UER §411 | No | — |
| capability discovery | CAPABILITY_CATALOG | Marketplace hub | ME-RB1 supporting model | CAPABILITY_MARKETPLACE_ENGINE | No | ADR-MKT-01 |
| marketplace acquisition | CAPABILITY_CATALOG (+ product) | AGENT_DISTRIBUTION | Handoff to distribution | CC V1 audit | No* | DOC vertical boundaries |
| agent distribution vs plugins | AGENT_DISTRIBUTION | PLATFORM_PLUGINS | Separate authority types | EAC-1R2 §4.A | No | EAC1R2-TAX-001 |
| application composition | TIER3 | HOSTING | T3 manifest vs deploy lifecycle | EAC-0 §16 | Partial | CL-EAC1-005 |
| skill vs tool | SKILLS | TOOLS | Composition over invocation | SKILLS.md | No | — |
| scale vs background | ELASTIC_CAPACITY_AND_SCALING | BACKGROUND_TASKS | Separate coordination vs delivery | EAC-1R2 §4.A | No | EAC1R2-TAX-001 |

### 9.1 Critical pair verdicts (explicit)

| Pair | Verdict | Notes |
|------|---------|-------|
| **Execution ↔ Nexus** | **ALIGNED** | EE owns **EXECUTION_LIFECYCLE**; Nexus holds **INTERNAL_ORCHESTRATION_SCHEDULING** only (**CL-EAC1-001** = caller bypass risk). |
| **Decision ↔ Reasoning** | **ALIGNED (canon)** | Decision owns outcome; Reasoning = **PROVIDER_ABSTRACTION_ROLE** material only. |
| **GE ↔ CW ↔ Tools** | **DOCUMENTATION CONFLICT** | Declarative CW tool auth vs GE **GOVERNANCE_AUTHORITY** — **ADR-GOV-01**; severity **HIGH**. |
| **CE ↔ UCL ↔ Memory** | **ALIGNED (canon) + impl debt** | **MEMORY_RECORD** + **MEMORY_REFERENCE_READ**; CE = **CONTEXT_ASSEMBLY**; UCL = **CONTEXT_LIFECYCLE**. |
| **Catalog ↔ Marketplace ↔ Distribution** | **ALIGNED (canon)** | **CAPABILITY_DISCOVERY** vs **AGENT_DISTRIBUTION** handoff. |
| **Observability ↔ Diagnostics** | **ALIGNED** | **EVIDENCE_AUTHORITY** vs **DIAGNOSTIC_AUTHORITY**. |
| **Reliability/HITL ↔ Continuation ↔ Recovery** | **ALIGNED** | EE **EXECUTION_LIFECYCLE**; Reliability policy/HITL authorities. |
| **Hosting ↔ Tier-3** | **PARTIAL** | **APPLICATION_COMPOSITION** vs **HOST_DEPLOYMENT_LIFECYCLE** — CL-EAC1-005. |
| **Skills ↔ Tools** | **ALIGNED** | **SKILL_COMPOSITION** vs **TOOL_INVOCATION**. |
| **RAG ↔ Memory** | **ALIGNED (canon)** | **RETRIEVAL_ORCHESTRATION** ≠ **MEMORY_RECORD** / **MEMORY_REFERENCE_READ**. |
| **Agent Distribution ↔ Platform Plugins** | **ALIGNED (canon)** | **AGENT_DISTRIBUTION_AUTHORITY** vs **PLUGIN_LIFECYCLE_AUTHORITY**. |

---

## 10. Duplicate authority findings

| Authority | Canonical owner | Competing candidate | Evidence | Verdict |
|-----------|-----------------|---------------------|----------|---------|
| Execution lifecycle | UNIFIED_EXECUTION_RUNTIME | NEXUS (historical public root) | NEXUS doc forbids; P0 bypass inventory @ RB-2A | **HISTORICAL ONLY** / **SUBORDINATE_CONFIRMED** |
| Execution lifecycle | UNIFIED_EXECUTION_RUNTIME | BACKGROUND_TASKS direct run | BG canon MUST NOT own identity | **NO CONFLICT** (if enforced) |
| Decision outcome | DECISION_SYSTEM | REASONING_AND_COGNITION | Architecture split | **NO CONFLICT** |
| Decision outcome | DECISION_SYSTEM | Legacy CVL / critic | `intergrax/runtime/critic/*` | **LEGACY** → RB-3 |
| Side-effect admission | GOVERNED_EXECUTION | COLLABORATIVE_WORK | ADR-GOV-01, RB-5 | **DOCUMENTATION CONFLICT** / **ADR REQUIRED** |
| RuntimeEvent truth | OBSERVABILITY | DIAGNOSTICS | DIAGNOSTICS.md | **NO CONFLICT** |
| Problem truth | DIAGNOSTICS | OBSERVABILITY | DIAGNOSTICS.md | **NO CONFLICT** |
| Memory record truth | MEMORY | Tier-3 session wiring | `memory_wiring.py` | **WIRING BYPASS RISK** — CL-EAC1-002 **NARROWED** |
| Memory reference read | MEMORY | CE adapters | `MemoryReferenceReadPort` | **NO CONFLICT** @ `616af2f4` |
| Context assembly semantics | CONTEXT_ENGINEERING | Nexus context modules (impl) | CE canon vs `intergrax/runtime/nexus/context/*` | **IMPLEMENTATION DEBT** — CL-EAC1-003 |
| Catalog lifecycle | AGENT_DISTRIBUTION / domain | CAPABILITY_CATALOG | CC pure consumer canon | **NO CONFLICT** |
| Plugin semantic authority | DOMAIN owners | PLATFORM_PLUGINS | EP **PLUGIN_LIFECYCLE** only | **NO CONFLICT** |
| Governance enterprise truth | GOVERNED_EXECUTION | AHI (research) | AHI non-prod | **NO CONFLICT** |
| Shared DISTRIBUTION_AUTHORITY (EAC-1R1) | Split types @ R2 | AD + EP | §4.A | **TAXONOMY_FIXED** |
| Shared RETRIEVAL_AUTHORITY (EAC-1R1) | Split types @ R2 | MEMORY + RAG | §4.A | **TAXONOMY_FIXED** |
| Shared COMPOSITION_AUTHORITY (EAC-1R1) | Scoped types @ R2 | Multiple domains | §4.A | **TAXONOMY_FIXED** |
| PROVIDER_ABSTRACTION as authority (EAC-1R1) | Demoted @ R2 | — | §4.B | **ROLE_ONLY** |

---

## 11. Contract ownership baseline (EAC-2 input)

| Contract family | Semantic owner | Primary consumers |
|-----------------|----------------|-------------------|
| `intergrax.contracts.execution*` | UNIFIED_EXECUTION_RUNTIME | All strategies, Observability, Diagnostics |
| `intergrax.contracts.decision*` | DECISION_SYSTEM | EE, CW (read) |
| `intergrax.contracts.governance*` | GOVERNED_EXECUTION | EE, Tools |
| `intergrax.contracts.tools*` | TOOLS | Agents, Integrations, GE |
| `intergrax.contracts.observability*` | OBSERVABILITY | Diagnostics, export |
| `intergrax.contracts.diagnostics` | DIAGNOSTICS | Operators, automation |
| Memory provider contracts | MEMORY | RAG, CE, applications |
| `MemoryReferenceReadPort` / `MemoryReferenceReadScope` / `MemoryRecordCanonicalRef` | MEMORY | ContextView / CE adapters (consume refs only) |
| Context contracts (`intergrax/context`) | CONTEXT_ENGINEERING | Agent, Nexus (should consume ports / memory refs) |
| Agent / harness contracts | AGENT_CONTRACTS_AND_ASSEMBLY | Nexus, Tier-2 (**must not** re-export **TOOL_INVOCATION_AUTHORITY**) |
| Catalog contracts | CAPABILITY_CATALOG_AND_DISCOVERY | Marketplace, T3 |
| Agent distribution contracts | AGENT_DISTRIBUTION | Catalog, Execution |
| Skill / RAG / Integration / LLM | respective DOMAIN | Assembly stack |

**Misplacement flags (audit only):** Nexus-internal types used as cross-tier public API (**CL-EAC1-001**); Diagnostics implementation slices under Observability **plan** (**DOC** ADR-REG-003) — not runtime authority transfer.

---

## 12. Plugin authority containment baseline

| Extension mechanism | Contract owner | Authority owner | Illegal bypass risk |
|--------------------|----------------|-----------------|---------------------|
| DecisionStrategy | DECISION_SYSTEM | DECISION_SYSTEM | Strategy finalizing without Decision lifecycle — **monitor** |
| Memory provider | MEMORY | MEMORY (**MEMORY_RECORD_AUTHORITY**) | Direct store bypassing ports — **EAC-4** |
| Tool driver | TOOLS | TOOLS + GE admission | Ungoverned invoke — U5 qual |
| Policy plugin | GOVERNED_EXECUTION | GOVERNED_EXECUTION | Uncertified GE — **QUAL** |
| Platform plugin entry | PLATFORM_PLUGINS | Target DOMAIN semantics; EP = **PLUGIN_LIFECYCLE** only | EP does not own semantics |
| Problem detector | DIAGNOSTICS | DIAGNOSTICS | Minting Problems without evidence — frozen R1 forbids |
| Orchestration / Nexus graph | ORCHESTRATION / NEXUS | EE admission | Public Nexus entry — **forbidden** |

**Rule verified:** plugin **never** becomes canonical owner without DOMAIN contract + validation.

---

## 13. Information ownership baseline

| Object | Created by | Owned by | May mutate | May only read | May project |
|--------|------------|----------|------------|---------------|-------------|
| TaskId | EE / task intake | EE (scope) | EE | All | BG, AW |
| RunId | EE | EE | EE | Observability, DIAG | — |
| AttemptId | EE | EE | EE | Reliability | — |
| ExecutionId | EE | EE | EE | Nexus (internal) | Graph views |
| DecisionId | DECISION_SYSTEM | DECISION_SYSTEM | Decision engine | EE, CW | — |
| DecisionVersion | DECISION_SYSTEM | DECISION_SYSTEM | Decision engine | Consumers | — |
| authorization/grant | GOVERNED_EXECUTION | GOVERNED_EXECUTION | GE | Tools, EE | — |
| continuation_id | EE | EE | EE | Recovery | Reliability |
| checkpoint state | EE (coordination) | Checkpoint port owner | Provider | EE restore | — |
| memory record | MEMORY writers | MEMORY | MEMORY providers | RAG, CE | Projections |
| memory canonical reference | MEMORY (`MemoryReferenceReadPort`) | MEMORY | MEMORY read port | CE / ContextView adapters | Adapter-side materialization |
| context bundle | CE | CE (ephemeral) | CE/UCL | Agent | UCL |
| retrieval result | RAG | RAG (derived) | RAG index jobs | CE | — |
| tool invocation | TOOLS | TOOLS (+ GE) | Tool runtime | Observability | — |
| external effect record | TOOLS / ERL | ERL (emerging) | ERL | DIAG | — |
| RuntimeEvent | EE (emit) / Obs (persist) | OBSERVABILITY | Observability | DIAG | Export |
| Problem | DIAGNOSTICS | DIAGNOSTICS | DIAG lifecycle | Operators | — |
| capability descriptor | Domain sources | Domain | Domain lifecycle | Catalog | Catalog |
| plugin descriptor | PLATFORM_PLUGINS | PLATFORM_PLUGINS | EP | Domains | — |
| proof receipt | PROOF_RECEIPTS | PROOF_RECEIPTS | Proof gates | CI | — |

---

## 14. Internal-domain handoffs (ID)

| ID | Owner | Finding | Next stage |
|----|-------|---------|------------|
| ID-MEM-01 | MEMORY | Consolidation algorithms | Domain session |
| ID-RAG-01 | RAG | Retrieval heuristics | Domain session |
| ID-RC-01 | REASONING_AND_COGNITION | Strategy internals | Domain session |
| ID-NX-01 | NEXUS_EXECUTION_FLOW | Graph scheduler heuristics | Domain session |
| ID-DX-01 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | Cursor workflow placement | DX session |
| ID-CE-01 | CONTEXT_ENGINEERING | Budget/compaction module internals (CE-02) | CE qualification |
| ID-GOV-01 | GOVERNED_EXECUTION | Policy catalog runtime gaps | GOV session |

---

## 15. Architecture conflict classification (findings register)

| Finding ID | Status @ EAC-1R2 | Class | Severity | Notes |
|------------|------------------|-------|----------|-------|
| CL-EAC1-001 | **OPEN** | CL | HIGH | Nexus imports from Tier-2 — unchanged |
| CL-EAC1-002 | **NARROWED** | CL | MEDIUM | Read port closed; wiring/persistence risk remains |
| CL-EAC1-003 | **QUALIFICATION ONLY** | CL | LOW | CE vs Nexus impl locus — not peer authority duplicate |
| CL-EAC1-004 | **ADR REQUIRED** | ADR | HIGH | GE **GOVERNANCE_AUTHORITY** vs CW policy — **visible**; GR-10 **PARTIAL** @ `0425caa4` is qualification only |
| CL-EAC1-005 | **OPEN** | CL | MEDIUM | T3 vs Hosting handoff |
| DOC-EAC1-001 | DOC | LOW | DIAGNOSTICS, OBSERVABILITY | ADR-REG-003 plan topology |
| DOC-EAC1-002 | DOC | LOW | REGISTRY | ADR-REG-001..004 |
| QUAL-EAC1-001 | QUAL | MEDIUM | GOVERNED_EXECUTION | GOV-FINAL NOT CERTIFIED; GR-10 **PARTIAL** @ `0425caa4` |
| QUAL-EAC1-002 | QUAL | MEDIUM | OBSERVABILITY | DG-005 |
| QUAL-EAC1-003 | QUAL | MEDIUM | ENTERPRISE_RELIABILITY_LAYER | Plan NEXT |
| LEGACY-EAC1-001 | LEGACY | MEDIUM | DECISION_SYSTEM | `intergrax/runtime/critic/*` — RB-3 |

| EAC-1R2 taxonomy notes | Class | Severity |
|------------------------|-------|----------|
| EAC1R2-TAX-001 | TAXONOMY_FIXED | — | Split distribution / retrieval / composition / scale vs background |
| EAC1R2-TAX-002 | ROLE_ONLY | — | PROVIDER_ABSTRACTION, DEVELOPMENT_ONLY demoted to domain roles |
| EAC1R2-TAX-003 | SUBORDINATE_CONFIRMED | — | Nexus scheduling; ERL effect reconciliation |

---

## 16. ADR-required findings (unchanged resolution)

| ADR | Topic | Status |
|-----|-------|--------|
| ADR-GOV-01 | GE vs CW side-effect authority | **Open** — blocks HIGH confidence for GE/CW |
| ADR-CTX-01 | CE vs UCL vs Memory | Open |
| ADR-MKT-01 | Marketplace vs Catalog vs Distribution | Canon aligned |

No ADR resolved in EAC-1R2.

---

## 17. EAC-2 / EAC-3 inputs

- Per-domain contract file mapping under `intergrax/contracts/` (EAC-2).
- Forbidden import graph: Tier-2/3 → Nexus internals (EAC-3).
- Replaceability: memory provider, DecisionStrategy, detector plugins (EAC-4).
- Authority flow diagrams for GE↔CW and CE↔UCL using §4.A peer map (EAC-7).
- Checkpoint/port persistence writers (EAC-9).

---

## 18. Validation checklist

### EAC-1R1 (retained)

| Check | Result |
|-------|--------|
| V1 — 34 DOMAIN rows in §5 | **PASS** |
| V5 — Nexus subordinate/internal | **PASS** |
| V6 — `MemoryReferenceReadPort` under Memory | **PASS** |

### EAC-1R2 (strict uniqueness)

| Check | Result |
|-------|--------|
| V1 — 34 DOMAIN rows | **PASS** |
| V2 — Authority taxonomy ≠ Domain role taxonomy | **PASS** (§4.A vs §4.B) |
| V3 — Each peer authority → one canonical owner | **PASS** (§4.1–4.2) |
| V4 — No shared DISTRIBUTION_AUTHORITY (AD + Plugins) | **PASS** |
| V5 — Memory vs RAG scoped authorities | **PASS** |
| V6 — No generic COMPOSITION_AUTHORITY across domains | **PASS** |
| V7 — Provider abstraction = role | **PASS** |
| V8 — Development-only = role | **PASS** |
| V9 — Scale vs Background split | **PASS** |
| V10 — Nexus subordinate | **PASS** |
| V11 — GE/CW conflict visible (CL-EAC1-004) | **PASS** |
| V12 — No production code changed in EAC-1R2 session | **PASS** (documentation only) |

---

*EAC-1 artifact reconciled @ **EAC1R2_EVIDENCE_HEAD** `af4d84e370062776d2f051d99cd5a7f74fc6e243`. Prior hardening: **EAC1R1_HARDENING_COMMIT** `ab6d578296e7b234a364c8dbb8570d318fe28f62`.*

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
