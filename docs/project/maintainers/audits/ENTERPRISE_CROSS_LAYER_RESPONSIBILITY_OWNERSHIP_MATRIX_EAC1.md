# EAC-1 — Enterprise Cross-Layer Responsibility & Ownership Matrix

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-1 — Responsibility & Ownership Matrix (**EAC-1R1** authority-taxonomy & parallel-drift hardening applied)  
**Type:** Read-only architecture audit (no remediation)  
**Authority:** EAC-0 R1 inventory + canonical domain pairs on `development`

| Gate | Value |
|------|-------|
| **EAC1_BASELINE_HEAD** (original EAC-1 publish) | `41a145332921de5d49f8f78099aca85d5591ef51` |
| **EAC1R1_BASELINE_HEAD** (reconciliation @ hardening) | `5563c5921500bc3a57ca8e6d816bfd18f5d76d8b` |
| **EAC0_R1_ANCHOR** | `d5252dd5ee79ba57fff2769587463fb7165a4451` |
| **Branch** | `development` |
| **HEAD == origin/development @ EAC-1R1** | **YES** |
| **Upstream inventory** | [`ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md`](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md) |
| **Registry hub** | [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md) |

**EAC-1R1 parallel-drift reconciliation (committed `development`):**

| Commit | Area | EAC-1R1 effect |
|--------|------|----------------|
| `616af2f4a3a5e06c1acd7249a1ab2528111fd301` | `MemoryReferenceReadPort`, `MEMORY.md`, `MEMORY_ARCHITECTURE.md` | Memory public read boundary; **CL-EAC1-002** read slice **CLOSED** |
| `9a6319e0e710029fb7ccfed1f1c2e7588cda4ff0` | Memory canonical plane composition | Memory row / contract baseline refreshed; persistence wiring risks unchanged |
| `5563c5921500bc3a57ca8e6d816bfd18f5d76d8b` | MP-5F-B1 workspace isolation hardening | Reference-read invariants; **no** DOMAIN ownership change |
| `3aecd9bac` (between EAC-1 publish and R1) | Governance GR-8 evidence contract doc | GE evidence vocabulary only; **GE = WHETHER / CW = workspace** model unchanged |

**Drift watch:** Claims through **EAC1R1_BASELINE_HEAD** unless marked LEGACY/QUAL. Uncommitted working-tree deltas (context budget modules, etc.) are **out of scope** for this artifact.

**Subordinate to:** per-domain architecture/plan pairs. This matrix **MUST NOT** redefine domain semantics.

---

## 1. Scope

EAC-1 establishes for each of **34** canonical DOMAIN rows:

- canonical **OWNS** / **MUST NOT OWN** / **MAY DELEGATE**
- **CONSUMES** / **PRODUCES**
- **AUTHORITATIVE** vs **DERIVED** state
- **PUBLIC** vs **INTERNAL** boundaries
- authority taxonomy, delegation semantics, duplicate-authority classification
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

**Primary sources:**

- [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)
- [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
- [`SYSTEM_INVARIANTS.md`](../../technical/guides/SYSTEM_INVARIANTS.md)
- [`UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md)
- Per-domain architecture files (EAC-0 §16)

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

**Enterprise rules (audit lens):** one capability → one canonical owner; one role → one authority; no duplicate mutable truth without explicit owner; plugins do not become platform authority.

---

## 4. Authority taxonomy (controlled — EAC-1R1)

Each type has **one** peer-level canonical owner unless listed as **subordinate/internal** (hierarchical exception).

| Type | Description | Peer canonical owner |
|------|-------------|----------------------|
| **IDENTITY_AUTHORITY** | Mint/bind canonical IDs in a namespace | UNIFIED_EXECUTION_RUNTIME (execution tree); other domains only in declared local namespaces |
| **EXECUTION_LIFECYCLE_AUTHORITY** | Canonical Run/Attempt/pause/resume/terminal transitions for platform execution | UNIFIED_EXECUTION_RUNTIME |
| **EXECUTION_AUTHORITY** | Run/Attempt/Execution tree coordination & strategy routing | UNIFIED_EXECUTION_RUNTIME |
| **DECISION_AUTHORITY** | Authoritative WHAT outcome & decision record lifecycle | DECISION_SYSTEM |
| **GOVERNANCE_AUTHORITY** | WHETHER proceed / side-effect admission | GOVERNED_EXECUTION |
| **ORCHESTRATION_STRATEGY_AUTHORITY** | Orchestration **strategy class** semantics (execution strategy role) | ORCHESTRATION |
| **INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY** | Private graph/step scheduling **under** strategy (not peer orchestration) | NEXUS_EXECUTION_FLOW (**subordinate** to ORCHESTRATION + EE admission) |
| **FAILURE_CLASSIFICATION_AUTHORITY** | Failure taxonomy / classification for resilience | RELIABILITY_FAILURE_AND_HITL |
| **RECOVERY_POLICY_AUTHORITY** | Bounded retry/degrade/compensate **policy selection** (not lifecycle execution) | RELIABILITY_FAILURE_AND_HITL (+ EE recovery **ports** for canonical continuation) |
| **HITL_INTERACTION_AUTHORITY** | Human escalation, interrupt interaction, HITL decision **records** | RELIABILITY_FAILURE_AND_HITL |
| **PERSISTENCE_AUTHORITY** | Durable store semantic owner | Domain-scoped (Memory, Observability, …) — no shared mutable truth without row owner |
| **EVIDENCE_AUTHORITY** | RuntimeEvent journal / reconstruction | OBSERVABILITY |
| **DIAGNOSTIC_AUTHORITY** | Problem / diagnostic interpretation | DIAGNOSTICS |
| **DISCOVERY_AUTHORITY** | Federated read/rank of capabilities | CAPABILITY_CATALOG_AND_DISCOVERY |
| **DISTRIBUTION_AUTHORITY** | Package install/materialize/activate | AGENT_DISTRIBUTION / PLATFORM_PLUGINS (extension slice) |
| **CONTEXT_ASSEMBLY_AUTHORITY** | Prompt/context bundle assembly & compile plan | CONTEXT_ENGINEERING |
| **CONTEXT_LIFECYCLE_AUTHORITY** | Conversation context optimization lifecycle (trim/compaction policy) | UNIFIED_CONTEXT_LIFECYCLE |
| **MEMORY_AUTHORITY** | Durable memory records/projections & consolidation policy surface | MEMORY |
| **RETRIEVAL_AUTHORITY** | Scoped memory **reference-read** semantics & RAG/query orchestration | MEMORY (`MemoryReferenceReadPort`); RAG owns retrieval **orchestration** over memory read models |
| **TOOL_AUTHORITY** | Tool invocation contract & effect channel | TOOLS |
| **PROVIDER_ABSTRACTION** | Vendor/model/driver indirection | LLM / Integrations / Modality / cognitive adapters |
| **COMPOSITION_AUTHORITY** | Host wiring, skill composition, agent assembly (not tool execution) | AGENT_CONTRACTS_AND_ASSEMBLY, SKILLS, TIER3, CODE_CRAFT, CW (workspace composition) |
| **SCALE_COORDINATION** | Capacity/lease admission | ELASTIC_CAPACITY_AND_SCALING, BACKGROUND_TASKS |
| **DEVELOPMENT_ONLY** | DX/tooling; non-runtime truth | PLATFORM_FOUNDATION (meta), EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE, PROOF_RECEIPTS, AHI |

**Deprecated in EAC-1R1 (do not assign to DOMAIN rows):** generic **LIFECYCLE_AUTHORITY**, generic **ORCHESTRATION_AUTHORITY**, generic **CONTEXT_AUTHORITY** — replaced by scoped types above.

Subordinate hubs (Execution Continuation, NPSC-5E recovery plane, Marketplace Engine doc, ERL sub-hubs) **do not** receive separate DOMAIN authority — parent DOMAIN rows below.

### 4.1 Authority uniqueness check (34 rows @ EAC-1R1)

Peer-level canonical owners: **no duplicate** for EXECUTION_LIFECYCLE, DECISION, GOVERNANCE, TOOL, MEMORY (store), EVIDENCE, DIAGNOSTIC, DISCOVERY (catalog). **Hierarchical (allowed):** ORCHESTRATION_STRATEGY → NEXUS internal scheduling. **RETRIEVAL_AUTHORITY:** MEMORY owns reference-read port; RAG holds retrieval orchestration over indexes (complementary, not competing store truth). **Remaining peer tension (documented, not taxonomy duplicate):** GE vs CW governance (**CL-EAC1-004**, ADR-GOV-01).

---

## 5. Thirty-four domain responsibility matrix

**Row count verification:** **34** (= EAC-0 §16).

| ID | Domain | OWNS | MUST NOT OWN | MAY DELEGATE | CONSUMES | PRODUCES | Authoritative State | Derived State | Public Boundary | Internal Boundary | Authority Types | Confidence | Risk |
|----|--------|------|--------------|--------------|----------|----------|---------------------|---------------|-----------------|-------------------|-----------------|------------|------|
| EAC-DOM-001 | PLATFORM_FOUNDATION | Tier topology, import boundaries, spine CI gates, platform invariants registry | Domain feature semantics, runtime lifecycle | None (meta) | All domains (read-only index) | Tier rules, SYS-INV-* | None runtime | Doctor reports | `SYSTEM_INVARIANTS`, tier contract docs | CI scripts internals | DEVELOPMENT_ONLY | STRONG | LOW |
| EAC-DOM-002 | UNIFIED_EXECUTION_RUNTIME | Run/Attempt lifecycle, ExecutionId coordination, strategy routing, terminal semantics, lifecycle fact emission, checkpoint **ports** coordination | Decision WHAT, governance WHETHER, Observability journal, Problem truth, Nexus orchestration internals | Orchestration strategies, Governance checks, Observability record, Recovery providers | Governance decisions, budget, checkpoint store ports | Lifecycle events, execution artifacts | Run, Attempt, Execution tree state, continuation tokens (UEA) | Scheduling projections inside strategies | `intergrax.contracts.execution*`, UEA | Engine adapter graphs, private routers | EXECUTION_AUTHORITY, EXECUTION_LIFECYCLE_AUTHORITY, IDENTITY_AUTHORITY | PROVEN | LOW |
| EAC-DOM-003 | ORCHESTRATION | Orchestration **strategy class** semantics, orchestration contracts as execution strategy | Execution identity/lifecycle, public Nexus API, admission | Nexus graph execution (internal) | Execution admission context | Strategy outcomes to EE | None (stateless strategy role) | Graph specs in flight | Orchestration contracts | Strategy impl details | ORCHESTRATION_STRATEGY_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-004 | NEXUS_EXECUTION_FLOW | Graph execution, fan-out/merge, internal scheduling **under** orchestration strategy | Public root API, execution admission, Run lifecycle, ExecutionId mint, governance | Tool/agent steps via contracts | EE checkpoints, execution context | Orchestration progress (internal) | Nexus topology/scheduling **projections** (non-canonical tree) | Readiness caches | **None** at platform root — consumer MUST use EE contracts | `NexusLoop`, `GraphExecutor`, `intergrax/runtime/nexus/context/*` (implementation locus) | INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY (**subordinate**) | STRONG | MEDIUM |
| EAC-DOM-005 | DECISION_SYSTEM | Decision lifecycle, authoritative outcome, version lineage, DecisionStrategy plugins | Execution lifecycle, authorization WHETHER, retry orchestration | Reasoning for material; Execution for hosting | Reasoning artifacts, evidence | Decision artifacts/records | Decision, DecisionVersion, outcome state | Deliberation scratch | `intergrax.contracts.decision*` | Strategy impl modules | DECISION_AUTHORITY | PROVEN | LOW |
| EAC-DOM-006 | GOVERNED_EXECUTION | Policy evaluation, side-effect admission, governance evidence (WHETHER) | Decision WHAT, execution scheduling, diagnostic Problem semantics | Policy plugins, external policy stores | Execution context, tool manifests | Allow/deny/interrupt decisions | Policy grants, admission records | Policy evaluation caches | `intergrax.contracts.governance*` | Policy engine internals | GOVERNANCE_AUTHORITY | PARTIAL | HIGH |
| EAC-DOM-007 | REASONING_AND_COGNITION | Cognitive strategies, reasoning material, candidate generation | Authoritative decision outcome, governance | LLM adapters | Context bundles, memory projections | Reasoning artifacts (non-final) | Session/scratch reasoning state | Cached chains | Reasoning contracts | Strategy internals | PROVIDER_ABSTRACTION (cognitive) | PARTIAL | MEDIUM |
| EAC-DOM-008 | AGENT_CONTRACTS_AND_ASSEMBLY | Agent contract, assembly, author behavior, harness kernel, step loop contract | Nexus planning, execution admission, governance rules, **tool gateways / vendor execution / policy engines** | Tools, skills, LLM, context (via contracts) | Context bundles, **tool contracts** (`intergrax.contracts.tools*`), governance hooks | Agent step requests, **tool intent/request** (not invocation) | Agent merge hooks state (bounded) | Assembly caches | Agent/`HarnessKernel` contracts | Harness private loops | COMPOSITION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-009 | AGENT_DISTRIBUTION | Agent package distribution, installability, RuntimeRevision path | Marketplace catalog engine, execution lifecycle | Catalog read for discovery | Catalog entries (read) | Installed agent metadata | Package install state | Discovery projections | `intergrax.contracts.agent_distribution*` | Installer internals | DISTRIBUTION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-010 | LLM_ADAPTERS | LLM provider abstraction, model call contracts | Agent planning ownership, governance | Vendor APIs | Provider config | Model outputs (non-authoritative) | None platform | Client caches | LLM provider contracts | Driver specifics | PROVIDER_ABSTRACTION | STRONG | LOW |
| EAC-DOM-011 | TOOLS | Tool contracts, invocation spine, idempotency contract surface | Integration drivers, governance rule definitions | Integrations for IO | Governance admission, integrations | Tool results, effect records | Idempotency keys (tool scope) | Invocation telemetry | `intergrax.contracts.tools*` | Driver wiring | TOOL_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-012 | SKILLS | Skill composition over tools (reusable capability) | Tool driver semantics, marketplace lifecycle | Tools | Tool registry | Composed skill invocations | None | Resolver caches | Skill contracts | Skill graph internals | COMPOSITION_AUTHORITY | STRONG | LOW |
| EAC-DOM-013 | INTEGRATIONS | Integration profiles, provider wiring | Tool semantic contracts, execution lifecycle | Vendor systems | Credentials profiles | Integration IO | Vendor connection state | Profile caches | Integration contracts | Connector internals | PROVIDER_ABSTRACTION | STRONG | LOW |
| EAC-DOM-014 | RAG | Retrieval orchestration, chunk/query semantics | Memory authoritative store, execution lifecycle | Memory projections, index providers | Memory read models, indexes | Retrieval results (derived reads) | Index segments (retrieval scope) | Query caches | RAG contracts | Pipeline impl | RETRIEVAL_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-015 | MEMORY | Memory records, projections, consolidation **policy surface**; scoped **reference-first read** (`MemoryReferenceReadPort`) | ContextView composition, CE assembly, foreign ContextView types, execution lifecycle, diagnostic Problems | Memory providers | Observability (audit only) | Memory records/projections; **canonical memory references** (no ContextView types) | Memory record truth (provider-backed) | Projection read models | Memory provider contracts; `intergrax/memory/contracts/memory_reference_read.py` (`MemoryReferenceReadPort`, `DefaultMemoryReferenceReader`) | Store implementations | MEMORY_AUTHORITY, PERSISTENCE_AUTHORITY, RETRIEVAL_AUTHORITY (reference-read) | PARTIAL | MEDIUM |
| EAC-DOM-016 | CONTEXT_ENGINEERING | Context assembly, collectors, compile plan | UCL optimization ownership, raw memory truth, execution | Memory/RAG/modality readers (**MemoryReferenceReadPort** / CE adapters) | Memory refs & projections, RAG, modality, CE policy | Context bundles for consumers | Assembly plan state (ephemeral) | Compiled context caches | Context contracts (`intergrax/context/`) | Collector registry internals | CONTEXT_ASSEMBLY_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-017 | UNIFIED_CONTEXT_LIFECYCLE | Conversation context **lifecycle optimization** (trim/compaction policy) | Raw memory authority, CE collector rules, execution | CE execution of plans | CE bundles, token budgets | Optimized context views | UCL optimization state | Compaction journals | UCL contracts (ADR-UCL-001) | Optimizer internals | CONTEXT_LIFECYCLE_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-018 | MODALITY | Multimodal ingestion contracts | Execution, memory truth | Adapters | Media sources | Normalized modality payloads | Media artifact refs | Transcode caches | Modality contracts | Adapter internals | PROVIDER_ABSTRACTION | UNKNOWN | MEDIUM |
| EAC-DOM-019 | OBSERVABILITY | RuntimeEvent recording, reconstruction, export | Problem semantics, execution control admission | Export sinks | Lifecycle facts from EE | Evidence journal, reconstructions | **RuntimeEvent** journal (canonical evidence) | Export/read models | `intergrax.contracts.observability*` | Indexer internals | EVIDENCE_AUTHORITY, PERSISTENCE_AUTHORITY | PROVEN | LOW |
| EAC-DOM-020 | DIAGNOSTICS | Deterministic interpretation → **Problem** state, operator read models | Evidence minting, retry/lifecycle control, execution identity | Detector plugins | RuntimeEvent/reconstruction | Problem records, assessments | **Problem** lifecycle store | Grouping hypotheses | `intergrax.contracts.diagnostics` | Detector pipelines | DIAGNOSTIC_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-021 | RELIABILITY_FAILURE_AND_HITL | Failure classification, recovery policy selection, bounded retry/degrade/compensate **recommendation**, HITL escalation & interaction records | Governance ALLOW/DENY, **canonical** pause/resume / execution lifecycle (EE), Problem truth | Human decision stores, EE for lifecycle consequences | EE lifecycle facts, checkpoints | Interrupt/resume **signals** (EE applies lifecycle); HITL decision records | HITL decision records (bounded); resilience policy artifacts | Attempt ledger projections (derived) | HITL contracts | Policy classifiers | FAILURE_CLASSIFICATION_AUTHORITY, RECOVERY_POLICY_AUTHORITY, HITL_INTERACTION_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-022 | ADAPTIVE_HARNESS_INTELLIGENCE | Harness intelligence, design-search hooks (non-prod authority) | Production governance authority | Research runtimes | Telemetry | Research artifacts | Research state | Experiments | AHI contracts (immature) | Search internals | DEVELOPMENT_ONLY | PARTIAL | LOW |
| EAC-DOM-023 | ELASTIC_CAPACITY_AND_SCALING | Capacity admission, scale coordination leases | EE identity/recovery semantics | Workers/runtime | Capacity signals | Admission decisions | Lease metadata | Metrics rollups | Capacity contracts | Scheduler internals | SCALE_COORDINATION | PARTIAL | MEDIUM |
| EAC-DOM-024 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | Dev workflows, harness DX, qualification tooling boundaries | Runtime semantic ownership | Local proofs | Docs/tooling | DX artifacts | None production | Local caches | N/A (guides) | Cursor rules, scripts | DEVELOPMENT_ONLY | STRONG | NONE |
| EAC-DOM-025 | TIER3_APPLICATION_ENVIRONMENT | Application manifests, profiles, host wiring composition | Execution engine internals | Hosting, plugins | Platform contracts | Application config bindings | Host profile state | Scaffold templates | Application env contracts | App-specific code | COMPOSITION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-026 | APPLICATION_HOSTING | Deployment/hosting lifecycle, always-on runtime hosting (**host scope**, not platform execution tree) | Tier-3 manifest semantics (owned by T3), canonical EE Run lifecycle | Execution admission | T3 manifests | Deployment records | Deployment/host runtime state | Health projections | Hosting contracts | Orchestrator internals | COMPOSITION_AUTHORITY (host deployment) | PARTIAL | MEDIUM |
| EAC-DOM-027 | CODE_CRAFT | Code manipulation domain services | Agent assembly ownership | Workspace IO | VCS/workspace | Code artifacts | Workspace drafts | Analysis caches | CodeCraft contracts | Tooling internals | COMPOSITION_AUTHORITY | UNKNOWN | LOW |
| EAC-DOM-028 | AUTONOMOUS_WORK | WorkerDefinition/Instance durable work semantics | Execution lifecycle owner (EE) | Background intake | Execution, catalog | Worker instances | Worker state | Queue views | AW contracts | Worker runtime | PERSISTENCE_AUTHORITY (worker instance scope) | PARTIAL | MEDIUM |
| EAC-DOM-029 | COLLABORATIVE_WORK | Workspace, membership, delegation, ContextView authority | Execution graph, decision outcome, governance WHETHER | GE for tool auth (declared overlap) | GE, Decision read | CW persistence events | Workspace/membership/ContextView | Presence projections | CW contracts | MP internals | COMPOSITION_AUTHORITY, GOVERNANCE_AUTHORITY (delegation) | PARTIAL | HIGH |
| EAC-DOM-030 | BACKGROUND_TASKS | Queue/bus abstractions, worker intake | Execution identity/lifecycle | EE for execution | Task envelopes | Queue metadata | Queue job state | Consumer lag metrics | `TaskQueue`, BG contracts | Broker adapters | SCALE_COORDINATION | PARTIAL | MEDIUM |
| EAC-DOM-031 | CAPABILITY_CATALOG_AND_DISCOVERY | Federated catalog read, rank, govern discovery rows | Vertical lifecycle, execution, Nexus | Domain registries | Domain sources | Catalog snapshots | Catalog index (read model) | Rank caches | Catalog contracts | Federator internals | DISCOVERY_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-032 | PROOF_RECEIPTS | Qualification receipt semantics | Runtime execution truth | CI/proof runners | Proof outputs | Receipt records | Receipt store | Verification views | Proof contracts | Gate scripts | DEVELOPMENT_ONLY | STRONG | NONE |
| EAC-DOM-033 | PLATFORM_PLUGINS | Extension discovery, enablement, trust vocabulary | Domain semantic ownership | Domain validation | Packaged extensions | Plugin enablement state | Enablement registry | Discovery cache | EP contracts | Loader internals | DISTRIBUTION_AUTHORITY (extension) | STRONG | LOW |
| EAC-DOM-034 | ENTERPRISE_RELIABILITY_LAYER | External effect reliability, UNKNOWN handling, reconciliation **composition** | Governance ALLOW/DENY (GE), canonical execution lifecycle | Provider invocations | Tool effects, observability | ERL assessment artifacts | ProviderInvocation state (emerging) | Reconciliation projections | ERL contracts (emerging) | Submodule hubs | RECOVERY_POLICY_AUTHORITY (effect reconciliation slice) | PARTIAL | MEDIUM |

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

**Note:** CONFLICT at DOMAIN-row level is reserved for irreconcilable dual ownership; cross-layer conflicts are tracked in §10–§11 with finding IDs (GE↔CW, CE↔UCL↔Memory).

### 5.2 Authority type usage (domains may hold multiple)

| Authority type | Domain count (approx.) | Peer owner |
|----------------|----------------------:|------------|
| EXECUTION_LIFECYCLE_AUTHORITY | 1 | EE only |
| EXECUTION_AUTHORITY | 1 | EE |
| ORCHESTRATION_STRATEGY_AUTHORITY | 1 | ORCHESTRATION |
| INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY | 1 | NEXUS (subordinate) |
| DECISION_AUTHORITY | 1 | DECISION_SYSTEM |
| GOVERNANCE_AUTHORITY | 2 | GE + partial CW (**ADR-GOV-01**) |
| FAILURE_CLASSIFICATION / RECOVERY_POLICY / HITL_INTERACTION | 1 (+ ERL slice) | RELIABILITY (+ ERL recovery slice) |
| EVIDENCE_AUTHORITY | 1 | OBSERVABILITY |
| DIAGNOSTIC_AUTHORITY | 1 | DIAGNOSTICS |
| MEMORY_AUTHORITY | 1 | MEMORY |
| RETRIEVAL_AUTHORITY | 2 | MEMORY (reference-read port); RAG (orchestration) |
| CONTEXT_ASSEMBLY / CONTEXT_LIFECYCLE | 1 each | CE / UCL |
| TOOL_AUTHORITY | 1 | TOOLS |
| DISCOVERY_AUTHORITY | 1 | CATALOG |
| DISTRIBUTION_AUTHORITY | 3 | AD / EP / plugins |
| PROVIDER_ABSTRACTION | 5 | LLM, Integrations, Modality, Reasoning, … |
| COMPOSITION_AUTHORITY | 6 | Agent, Skills, T3, CodeCraft, CW, Hosting |
| SCALE_COORDINATION | 2 | Elastic, BG |
| DEVELOPMENT_ONLY | 4 | Foundation meta, DX, Proof, AHI |

---

## 6. Public vs internal boundaries (summary)

| Domain | PUBLIC (legal consumers) | INTERNAL (must not escape) | Boundary findings |
|--------|--------------------------|----------------------------|-------------------|
| UNIFIED_EXECUTION_RUNTIME | `intergrax.contracts.execution*`, admission APIs | Private engine graph | — |
| NEXUS_EXECUTION_FLOW | **No public root** — only via EE/ORCHESTRATION | `NexusLoop`, `GraphExecutor`, `intergrax/runtime/nexus/context/*` | **CL-BOUNDARY-VIOLATION-EAC1-001** — Tier-2 `agents/*` import `RuntimeContext`, `SessionManager`, notebooks import `NexusLoop` (grep @ baseline) |
| DECISION_SYSTEM | `intergrax.contracts.decision*` | Strategy impl | — |
| GOVERNED_EXECUTION | `intergrax.contracts.governance*` | Policy engine private | QUAL: enterprise cert NOT CERTIFIED |
| OBSERVABILITY | Observability contracts, reconstruction APIs | Journal writers private | — |
| DIAGNOSTICS | Diagnostics contracts, Problem read APIs | Detectors | DOC: ADR-REG-003 plan topology |
| MEMORY | Memory provider ports; **`MemoryReferenceReadPort`** (`intergrax/memory/contracts/memory_reference_read.py`) | Store drivers; ContextView types in Memory | Tier-3 wiring via `intergrax/applications/_shared/memory_wiring.py` → Nexus `SessionManager` (**CL-EAC1-002** narrowed) |
| CONTEXT_ENGINEERING | `intergrax/context` contracts | Collectors/registry; Nexus `ContextCompiler` is **consumer/impl locus**, not semantic owner | CE-02 qualification (committed + in-flight); semantic owner = CE per `CONTEXT_ENGINEERING.md` |
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
| Memory record | MEMORY | RAG, CE, UCL | Projections | Provider stores | **NO CONFLICT** on read boundary @ `616af2f4` (`MemoryReferenceReadPort`); **wiring/persistence** risk remains (**CL-EAC1-002** NARROWED) |
| Memory canonical reference | MEMORY (`MemoryReferenceReadPort`) | CE / ContextView adapters | — | Ephemeral adapter caches | **NO CONFLICT** — Memory returns refs, not ContextView types |
| Context bundle (assembled) | CONTEXT_ENGINEERING | Agent, Nexus consumer | UCL optimized views | Ephemeral + caches | **LOW (architecture debt)** — Nexus context modules = **implementation placement**; semantic assembly owner = CE (**CL-EAC1-003** refreshed) |
| Retrieval result | RAG | CE, agents | Indexes | RAG indexes | **NO CONFLICT** if RAG does not own memory truth |
| Tool invocation / effect | TOOLS (+ GE admission) | ERL, Observability | Idempotency cache | Tool stores | **REAL DUPLICATE CANDIDATE** GE vs CW side-effect path (ADR-GOV-01) |
| Catalog snapshot row | CAPABILITY_CATALOG_AND_DISCOVERY | Marketplace UI | Rank caches | Catalog index | **NO CONFLICT** |
| Agent install / RuntimeRevision | AGENT_DISTRIBUTION | Execution, Catalog | Discovery | Package store | **NO CONFLICT** with Catalog read |
| Worker instance | AUTONOMOUS_WORK | BG, Execution | Queue metadata | AW store | **PARTIAL** — EE still owns execution lifecycle |
| Continuation / checkpoint payload | UNIFIED_EXECUTION_RUNTIME (coordination) | Recovery providers | Reliability projections | Checkpoint ports (NPSC-5E) | **DERIVED ONLY** for Reliability Attempt Ledger |
| HITL interrupt decision | RELIABILITY_FAILURE_AND_HITL (human decision semantics) | EE (consequences) | — | Decision store | **NO CONFLICT** if EE owns pause/resume lifecycle |

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
| memory reference read | MEMORY | CE / ContextView consumers | Scoped reference-first port | `memory_reference_read.py` | No | — (B1 CLOSED per MEMORY.md) |
| context assembly | CONTEXT_ENGINEERING | UCL / Nexus | CE assembles; UCL optimizes lifecycle; Nexus = internal scheduler consumer | ADR-CTX-01, `CONTEXT_ENGINEERING.md` | Low (impl locus) | CL-EAC1-003 **QUAL / debt** |
| retrieval | RAG | MEMORY | RAG orchestrates reads | Catalog canon | No | — |
| evidence | OBSERVABILITY | DIAGNOSTICS | DIAG consumes; no mint | DIAGNOSTICS.md §SSOT | No | — |
| diagnostics | DIAGNOSTICS | OBSERVABILITY | Problem vs RuntimeEvent | R1 freeze | No | — |
| continuation | UNIFIED_EXECUTION_RUNTIME | RELIABILITY | EE owns pause/resume lifecycle | UER §HITL consequences | No | — |
| recovery | UNIFIED_EXECUTION_RUNTIME (5E plane) | RELIABILITY | Recovery ports; attempt ledger derived | NPSC-5E freeze | No | — |
| HITL | RELIABILITY (+ GE) | EE | Human decision vs lifecycle consequence split | UER §411 | No | — |
| capability discovery | CAPABILITY_CATALOG | Marketplace hub | ME-RB1 supporting model | CAPABILITY_MARKETPLACE_ENGINE | No | ADR-MKT-01 |
| marketplace acquisition | CAPABILITY_CATALOG (+ product) | AGENT_DISTRIBUTION | Handoff to distribution | CC V1 audit | No* | DOC vertical boundaries |
| distribution/activation | AGENT_DISTRIBUTION | CATALOG | Catalog read only | CC.md §Agent lifecycle | No | — |
| application composition | TIER3 | HOSTING | T3 manifest vs deploy lifecycle | EAC-0 §16 | Partial | CL-EAC1-005 |
| skill vs tool | SKILLS | TOOLS | Composition over invocation | SKILLS.md | No | — |
| RAG vs memory | MEMORY | RAG | Retrieval ≠ store owner | RAG campaign E | Partial | QUAL |

### 9.1 Critical pair verdicts (explicit)

| Pair | Verdict | Notes |
|------|---------|-------|
| **Execution ↔ Nexus** | **ALIGNED** | EE owns **EXECUTION_LIFECYCLE**; Nexus holds **INTERNAL_ORCHESTRATION_SCHEDULING** only. Violation risk = **callers** importing Nexus from agents (**CL-EAC1-001**), not dual lifecycle writers. |
| **Decision ↔ Reasoning** | **ALIGNED (canon)** | Decision owns outcome; Reasoning owns material. **LEGACY** app-local loops — RB-3; not second DOMAIN authority. |
| **GE ↔ CW ↔ Tools** | **DOCUMENTATION CONFLICT** | Declarative CW tool auth vs GE WHETHER — **ADR-GOV-01**; Tools remain mechanism. Severity **HIGH** until ADR closed. |
| **CE ↔ UCL ↔ Memory** | **ALIGNED (canon) + impl debt** | Memory = durable truth + **MemoryReferenceReadPort**; CE = **CONTEXT_ASSEMBLY**; UCL = **CONTEXT_LIFECYCLE**. Nexus context path = implementation location, not peer semantic owner (**CL-EAC1-003**). |
| **Catalog ↔ Marketplace ↔ Distribution** | **ALIGNED (canon)** | Catalog read; Distribution lifecycle; Marketplace = supporting hub (not DOMAIN). |
| **Observability ↔ Diagnostics** | **ALIGNED** | Evidence vs Problem — explicit SSOT hierarchy. |
| **Reliability/HITL ↔ Continuation ↔ Recovery** | **ALIGNED** | EE owns **EXECUTION_LIFECYCLE** & continuation ports; Reliability owns **FAILURE_CLASSIFICATION / RECOVERY_POLICY / HITL_INTERACTION** (not platform pause/resume lifecycle). |
| **Hosting ↔ Tier-3** | **PARTIAL** | T3 owns manifest semantics; Hosting owns deployment lifecycle — document overlap (CL-005). |
| **Skills ↔ Tools** | **ALIGNED** | Composition vs executable boundary. |
| **RAG ↔ Memory** | **ALIGNED (canon)** | Implementation coupling risk (**QUAL**), not dual memory authority in canon. |

---

## 10. Duplicate authority findings

| Authority | Canonical owner | Competing candidate | Evidence | Verdict |
|-----------|-----------------|---------------------|----------|---------|
| Execution lifecycle | UNIFIED_EXECUTION_RUNTIME | NEXUS (historical public root) | NEXUS doc forbids; P0 bypass inventory @ RB-2A | **HISTORICAL ONLY** |
| Execution lifecycle | UNIFIED_EXECUTION_RUNTIME | BACKGROUND_TASKS direct run | BG canon MUST NOT own identity | **NO CONFLICT** (if enforced) |
| Decision outcome | DECISION_SYSTEM | REASONING_AND_COGNITION | Architecture split | **NO CONFLICT** |
| Decision outcome | DECISION_SYSTEM | Legacy CVL / critic | `intergrax/runtime/critic/*` | **LEGACY** → RB-3 |
| Side-effect admission | GOVERNED_EXECUTION | COLLABORATIVE_WORK | ADR-GOV-01, RB-5 | **DOCUMENTATION CONFLICT** |
| RuntimeEvent truth | OBSERVABILITY | DIAGNOSTICS | DIAGNOSTICS.md | **NO CONFLICT** |
| Problem truth | DIAGNOSTICS | OBSERVABILITY | DIAGNOSTICS.md | **NO CONFLICT** |
| Memory record truth | MEMORY | Tier-3 session wiring | `memory_wiring.py` constructs Nexus `SessionManager` | **WIRING BYPASS RISK** — CL-002 **NARROWED** (read port closed) |
| Memory reference read | MEMORY | CE adapters | `MemoryReferenceReadPort` | **NO CONFLICT** @ `616af2f4` |
| Context assembly semantics | CONTEXT_ENGINEERING | Nexus context modules (impl) | CE canon vs `intergrax/runtime/nexus/context/*` | **IMPLEMENTATION DEBT** — not peer duplicate authority — CL-003 |
| Catalog lifecycle | AGENT_DISTRIBUTION / domain | CAPABILITY_CATALOG | CC pure consumer canon | **NO CONFLICT** |
| Plugin authority | DOMAIN owners | PLATFORM_PLUGINS | EP enables; domains validate | **NO CONFLICT** |
| Governance enterprise truth | GOVERNED_EXECUTION | AHI (research) | AHI non-prod | **NO CONFLICT** |

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
| Agent / harness contracts | AGENT_CONTRACTS_AND_ASSEMBLY | Nexus, Tier-2 (**must not** re-export TOOL_AUTHORITY) |
| Catalog contracts | CAPABILITY_CATALOG_AND_DISCOVERY | Marketplace, T3 |
| Agent distribution contracts | AGENT_DISTRIBUTION | Catalog, Execution |
| Skill / RAG / Integration / LLM | respective DOMAIN | Assembly stack |

**Misplacement flags (audit only):** Nexus-internal types used as cross-tier public API (**CL-001**); Diagnostics implementation slices under Observability **plan** (**DOC** ADR-REG-003) — not runtime authority transfer.

---

## 12. Plugin authority containment baseline

| Extension mechanism | Contract owner | Authority owner | Illegal bypass risk |
|--------------------|----------------|-----------------|---------------------|
| DecisionStrategy | DECISION_SYSTEM | DECISION_SYSTEM | Strategy finalizing without Decision lifecycle — **monitor** |
| Memory provider | MEMORY | MEMORY | Direct store bypassing ports — **EAC-4** |
| Tool driver | TOOLS | TOOLS + GE admission | Un governed invoke — U5 qual |
| Policy plugin | GOVERNED_EXECUTION | GOVERNED_EXECUTION | Uncertified GE — **QUAL** |
| Platform plugin entry | PLATFORM_PLUGINS | Target DOMAIN | EP does not own semantics |
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

| Finding ID | Status @ EAC-1R1 | Class | Severity | Affected domains | Evidence | Recommended stage |
|------------|----------------|-------|----------|------------------|----------|-------------------|
| CL-EAC1-001 | **OPEN** | CL | HIGH | NEXUS, AGENT_CONTRACTS, Tier-2 | `agents/*` imports `intergrax.runtime.nexus.*` (e.g. `RuntimeContext`, `NexusLoop` notebooks) @ `9a6319e0` | EAC-13 composition audit |
| CL-EAC1-002 | **NARROWED** | CL | MEDIUM | TIER3 wiring, MEMORY persistence | Read boundary **CLOSED** `616af2f4` (`MemoryReferenceReadPort`). **Remaining:** `intergrax/applications/_shared/memory_wiring.py` → Nexus `SessionManager`; duplicate persistence / host bypass risk | EAC-9 persistence + EAC-5 flow |
| CL-EAC1-003 | **QUALIFICATION ONLY** (impl debt) | CL | LOW | CE, NEXUS (impl) | Semantic owner = CE/UCL/Memory per canon; `intergrax/runtime/nexus/context/*` = **implementation location** not peer **CONTEXT_ASSEMBLY** authority | ADR-CTX-01 / EAC-7 (relocate vs document) |
| CL-EAC1-004 | **ADR REQUIRED** | ADR | HIGH | GOVERNED_EXECUTION, COLLABORATIVE_WORK, TOOLS | ADR-GOV-01, RB-5; GR-8 evidence doc @ `3aecd9bac` does not resolve CW vs GE admission | ADR close → EAC-7 |
| CL-EAC1-005 | **OPEN** | CL | MEDIUM | TIER3, APPLICATION_HOSTING | EAC-0 LIFECYCLE_OVERLAP_RISK; hosting row uses host-scope composition not EE lifecycle | EAC-8 handoff |
| DOC-EAC1-001 | DOC | LOW | DIAGNOSTICS, OBSERVABILITY | ADR-REG-003 plan topology | Registry ADR |
| DOC-EAC1-002 | DOC | LOW | REGISTRY | ADR-REG-001..004 | Registry update |
| QUAL-EAC1-001 | QUAL | MEDIUM | GOVERNED_EXECUTION | GOV-FINAL NOT CERTIFIED | Qualification |
| QUAL-EAC1-002 | QUAL | MEDIUM | OBSERVABILITY | DG-005 | Qualification |
| QUAL-EAC1-003 | QUAL | MEDIUM | ENTERPRISE_RELIABILITY_LAYER | Plan NEXT | ERL program |
| LEGACY-EAC1-001 | LEGACY | MEDIUM | DECISION_SYSTEM | `intergrax/runtime/critic/*` | RB-3 |

---

## 16. ADR-required findings (unchanged resolution)

| ADR | Topic | EAC-1 status |
|-----|-------|--------------|
| ADR-REG-001..004 | Registry / plan topology | Open — DOC only |
| ADR-GOV-01 | GE vs CW side-effect authority | **Blocks HIGH confidence for GE/CW** |
| ADR-CTX-01 | CE vs UCL vs Memory | Open — CL-003 **narrowed** (Memory read port closed B1; Nexus impl debt remains) |
| ADR-MKT-01 | Marketplace vs Catalog vs Distribution | Canon aligned; vertical product edges |

No ADR resolved implicitly in EAC-1.

---

## 17. EAC-2 / EAC-3 inputs

- Per-domain contract file mapping under `intergrax/contracts/` (EAC-2).
- Forbidden import graph: Tier-2/3 → Nexus internals (EAC-3).
- Replaceability: memory provider, DecisionStrategy, detector plugins (EAC-4).
- Authority flow diagrams for GE↔CW and CE↔UCL (EAC-7).
- Checkpoint/port persistence writers (EAC-9).

---

## 18. Validation checklist (EAC-1R1 PASS)

| Check | Result |
|-------|--------|
| V1 — 34 DOMAIN rows in §5 | **PASS** |
| V2 — No peer duplicate authority without explicit finding | **PASS** (GE↔CW documented **CL-EAC1-004**) |
| V3 — Agent Contracts does not own TOOL_AUTHORITY | **PASS** |
| V4 — Reliability does not own canonical Execution lifecycle | **PASS** |
| V5 — Nexus authority subordinate/internal | **PASS** (`INTERNAL_ORCHESTRATION_SCHEDULING_AUTHORITY`) |
| V6 — `MemoryReferenceReadPort` in Memory ownership | **PASS** |
| V7 — CL-EAC1-002 narrowed to remaining wiring scope | **PASS** |
| V8 — CL-EAC1-003 semantic owner vs impl locus | **PASS** |
| V9 — No ADR silently resolved | **PASS** |
| V10 — No production code changed in EAC-1R1 session | **PASS** (documentation only) |
| Every row has OWNS + MUST NOT OWN | **PASS** |
| Authority taxonomy §4 rebuilt | **PASS** |
| Parallel drift @ EAC1R1_BASELINE_HEAD | **PASS** |

---

*EAC-1 / EAC-1R1 audit artifact @ `5563c5921500bc3a57ca8e6d816bfd18f5d76d8b` (+ EAC-1R1 doc commit). Independent verification against GitHub `development` remains mandatory for production claims.*

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
