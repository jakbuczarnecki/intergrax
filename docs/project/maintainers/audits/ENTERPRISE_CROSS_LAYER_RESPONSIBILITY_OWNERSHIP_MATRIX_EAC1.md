# EAC-1 — Enterprise Cross-Layer Responsibility & Ownership Matrix

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-1 — Responsibility & Ownership Matrix  
**Type:** Read-only architecture audit (no remediation)  
**Authority:** EAC-0 R1 inventory + current canonical domain pairs on `development`

| Gate | Value |
|------|-------|
| **EAC1_BASELINE_HEAD** | `d5252dd5ee79ba57fff2769587463fb7165a4451` |
| **EAC0_R1_ANCHOR** | `d5252dd5ee79ba57fff2769587463fb7165a4451` |
| **Branch** | `development` |
| **HEAD == origin/development @ baseline** | **YES** |
| **Upstream inventory** | [`ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md`](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md) |
| **Registry hub** | [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md) |

**Drift watch:** No commits on `origin/development` between EAC-0R1 anchor and EAC-1 baseline on registry/architecture/governance/context/memory paths. Local working-tree changes exist (context budget, memory reference read, application wiring) — **not** incorporated into this audit; claims reference **committed** `development` @ baseline unless noted as DOC/QUAL gap.

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

## 4. Authority taxonomy (controlled)

| Type | Description |
|------|-------------|
| **IDENTITY_AUTHORITY** | Mint/bind canonical IDs in a namespace |
| **LIFECYCLE_AUTHORITY** | State transitions for a work unit |
| **DECISION_AUTHORITY** | Authoritative WHAT outcome |
| **GOVERNANCE_AUTHORITY** | WHETHER proceed / side-effect admission |
| **EXECUTION_AUTHORITY** | Run/Attempt/Execution tree coordination |
| **ORCHESTRATION_AUTHORITY** | In-run graph/step scheduling inside execution strategy |
| **PERSISTENCE_AUTHORITY** | Durable store semantic owner |
| **EVIDENCE_AUTHORITY** | RuntimeEvent journal / reconstruction |
| **DIAGNOSTIC_AUTHORITY** | Problem / diagnostic interpretation |
| **DISCOVERY_AUTHORITY** | Federated read/rank of capabilities |
| **DISTRIBUTION_AUTHORITY** | Package install/materialize/activate |
| **CONTEXT_AUTHORITY** | Prompt/context bundle assembly |
| **MEMORY_AUTHORITY** | Durable memory records/projections |
| **RETRIEVAL_AUTHORITY** | RAG/query/chunk orchestration |
| **TOOL_AUTHORITY** | Tool invocation contract & effect channel |
| **PROVIDER_ABSTRACTION** | Vendor/model/driver indirection |
| **COMPOSITION_AUTHORITY** | Host wiring / skill composition |
| **SCALE_COORDINATION** | Capacity/lease admission |
| **DEVELOPMENT_ONLY** | DX/tooling; non-runtime truth |

Subordinate hubs (Execution Continuation, NPSC-5E recovery plane, Marketplace Engine doc, ERL sub-hubs) **do not** receive separate DOMAIN authority — parent DOMAIN rows below.

---

## 5. Thirty-four domain responsibility matrix

**Row count verification:** **34** (= EAC-0 §16).

| ID | Domain | OWNS | MUST NOT OWN | MAY DELEGATE | CONSUMES | PRODUCES | Authoritative State | Derived State | Public Boundary | Internal Boundary | Authority Types | Confidence | Risk |
|----|--------|------|--------------|--------------|----------|----------|---------------------|---------------|-----------------|-------------------|-----------------|------------|------|
| EAC-DOM-001 | PLATFORM_FOUNDATION | Tier topology, import boundaries, spine CI gates, platform invariants registry | Domain feature semantics, runtime lifecycle | None (meta) | All domains (read-only index) | Tier rules, SYS-INV-* | None runtime | Doctor reports | `SYSTEM_INVARIANTS`, tier contract docs | CI scripts internals | DEVELOPMENT_ONLY | STRONG | LOW |
| EAC-DOM-002 | UNIFIED_EXECUTION_RUNTIME | Run/Attempt lifecycle, ExecutionId coordination, strategy routing, terminal semantics, lifecycle fact emission, checkpoint **ports** coordination | Decision WHAT, governance WHETHER, Observability journal, Problem truth, Nexus orchestration internals | Orchestration strategies, Governance checks, Observability record, Recovery providers | Governance decisions, budget, checkpoint store ports | Lifecycle events, execution artifacts | Run, Attempt, Execution tree state, continuation tokens (UEA) | Scheduling projections inside strategies | `intergrax.contracts.execution*`, UEA | Engine adapter graphs, private routers | EXECUTION_AUTHORITY, LIFECYCLE_AUTHORITY, IDENTITY_AUTHORITY | PROVEN | LOW |
| EAC-DOM-003 | ORCHESTRATION | Orchestration **strategy class** semantics, orchestration contracts as execution strategy | Execution identity/lifecycle, public Nexus API, admission | Nexus graph execution (internal) | Execution admission context | Strategy outcomes to EE | None (stateless strategy role) | Graph specs in flight | Orchestration contracts | Strategy impl details | ORCHESTRATION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-004 | NEXUS_EXECUTION_FLOW | Graph execution, fan-out/merge, internal scheduling **under** orchestration strategy | Public root API, execution admission, Run lifecycle, ExecutionId mint, governance | Tool/agent steps via contracts | EE checkpoints, execution context | Orchestration progress (internal) | Nexus topology/scheduling **projections** (non-canonical tree) | Readiness caches | **None** at platform root — consumer MUST use EE contracts | `NexusLoop`, `GraphExecutor`, internal context engine | ORCHESTRATION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-005 | DECISION_SYSTEM | Decision lifecycle, authoritative outcome, version lineage, DecisionStrategy plugins | Execution lifecycle, authorization WHETHER, retry orchestration | Reasoning for material; Execution for hosting | Reasoning artifacts, evidence | Decision artifacts/records | Decision, DecisionVersion, outcome state | Deliberation scratch | `intergrax.contracts.decision*` | Strategy impl modules | DECISION_AUTHORITY, LIFECYCLE_AUTHORITY (decision scope) | PROVEN | LOW |
| EAC-DOM-006 | GOVERNED_EXECUTION | Policy evaluation, side-effect admission, governance evidence (WHETHER) | Decision WHAT, execution scheduling, diagnostic Problem semantics | Policy plugins, external policy stores | Execution context, tool manifests | Allow/deny/interrupt decisions | Policy grants, admission records | Policy evaluation caches | `intergrax.contracts.governance*` | Policy engine internals | GOVERNANCE_AUTHORITY | PARTIAL | HIGH |
| EAC-DOM-007 | REASONING_AND_COGNITION | Cognitive strategies, reasoning material, candidate generation | Authoritative decision outcome, governance | LLM adapters | Context bundles, memory projections | Reasoning artifacts (non-final) | Session/scratch reasoning state | Cached chains | Reasoning contracts | Strategy internals | PROVIDER_ABSTRACTION (cognitive) | PARTIAL | MEDIUM |
| EAC-DOM-008 | AGENT_CONTRACTS_AND_ASSEMBLY | Agent assembly, harness kernel, step loop contract | Nexus planning, execution admission, governance rules | Tools, skills, LLM, context | Context, tools, governance hooks | Agent step requests | Agent merge hooks state (bounded) | Assembly caches | Agent/`HarnessKernel` contracts | Harness private loops | COMPOSITION_AUTHORITY, TOOL_AUTHORITY (invoke path) | STRONG | MEDIUM |
| EAC-DOM-009 | AGENT_DISTRIBUTION | Agent package distribution, installability, RuntimeRevision path | Marketplace catalog engine, execution lifecycle | Catalog read for discovery | Catalog entries (read) | Installed agent metadata | Package install state | Discovery projections | `intergrax.contracts.agent_distribution*` | Installer internals | DISTRIBUTION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-010 | LLM_ADAPTERS | LLM provider abstraction, model call contracts | Agent planning ownership, governance | Vendor APIs | Provider config | Model outputs (non-authoritative) | None platform | Client caches | LLM provider contracts | Driver specifics | PROVIDER_ABSTRACTION | STRONG | LOW |
| EAC-DOM-011 | TOOLS | Tool contracts, invocation spine, idempotency contract surface | Integration drivers, governance rule definitions | Integrations for IO | Governance admission, integrations | Tool results, effect records | Idempotency keys (tool scope) | Invocation telemetry | `intergrax.contracts.tools*` | Driver wiring | TOOL_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-012 | SKILLS | Skill composition over tools (reusable capability) | Tool driver semantics, marketplace lifecycle | Tools | Tool registry | Composed skill invocations | None | Resolver caches | Skill contracts | Skill graph internals | COMPOSITION_AUTHORITY | STRONG | LOW |
| EAC-DOM-013 | INTEGRATIONS | Integration profiles, provider wiring | Tool semantic contracts, execution lifecycle | Vendor systems | Credentials profiles | Integration IO | Vendor connection state | Profile caches | Integration contracts | Connector internals | PROVIDER_ABSTRACTION | STRONG | LOW |
| EAC-DOM-014 | RAG | Retrieval orchestration, chunk/query semantics | Memory authoritative store, execution lifecycle | Memory projections, index providers | Memory read models, indexes | Retrieval results (derived reads) | Index segments (retrieval scope) | Query caches | RAG contracts | Pipeline impl | RETRIEVAL_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-015 | MEMORY | Memory records, projections, consolidation **policy surface** | Context lifecycle (UCL), execution lifecycle, diagnostic Problems | Memory providers | Observability (audit only) | Memory records/projections | Memory record truth (provider-backed) | Projection read models | Memory provider contracts | Store implementations | MEMORY_AUTHORITY, PERSISTENCE_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-016 | CONTEXT_ENGINEERING | Context assembly, collectors, compile plan | UCL optimization ownership, raw memory truth, execution | Memory/RAG/modality readers | Memory, RAG, modality, CE policy | Context bundles for consumers | Assembly plan state (ephemeral) | Compiled context caches | Context contracts (`intergrax/context/`) | Collector registry internals | CONTEXT_AUTHORITY | PARTIAL | MEDIUM |
| EAC-DOM-017 | UNIFIED_CONTEXT_LIFECYCLE | Conversation context **lifecycle optimization** (trim/compaction policy) | Raw memory authority, CE collector rules, execution | CE execution of plans | CE bundles, token budgets | Optimized context views | UCL optimization state | Compaction journals | UCL contracts (ADR-UCL-001) | Optimizer internals | CONTEXT_AUTHORITY (lifecycle) | PARTIAL | MEDIUM |
| EAC-DOM-018 | MODALITY | Multimodal ingestion contracts | Execution, memory truth | Adapters | Media sources | Normalized modality payloads | Media artifact refs | Transcode caches | Modality contracts | Adapter internals | PROVIDER_ABSTRACTION | UNKNOWN | MEDIUM |
| EAC-DOM-019 | OBSERVABILITY | RuntimeEvent recording, reconstruction, export | Problem semantics, execution control admission | Export sinks | Lifecycle facts from EE | Evidence journal, reconstructions | **RuntimeEvent** journal (canonical evidence) | Export/read models | `intergrax.contracts.observability*` | Indexer internals | EVIDENCE_AUTHORITY, PERSISTENCE_AUTHORITY | PROVEN | LOW |
| EAC-DOM-020 | DIAGNOSTICS | Deterministic interpretation → **Problem** state, operator read models | Evidence minting, retry/lifecycle control, execution identity | Detector plugins | RuntimeEvent/reconstruction | Problem records, assessments | **Problem** lifecycle store | Grouping hypotheses | `intergrax.contracts.diagnostics` | Detector pipelines | DIAGNOSTIC_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-021 | RELIABILITY_FAILURE_AND_HITL | HITL escalation mechanisms, resilience policy **surface**, interrupt semantics | Governance ALLOW/DENY, canonical pause/resume lifecycle (EE), Problem truth | Human decision stores | EE lifecycle, checkpoints | Interrupt/resume signals | HITL decision records (bounded) | Attempt ledger projections | HITL contracts | Policy classifiers | LIFECYCLE_AUTHORITY (HITL slice) | PARTIAL | MEDIUM |
| EAC-DOM-022 | ADAPTIVE_HARNESS_INTELLIGENCE | Harness intelligence, design-search hooks (non-prod authority) | Production governance authority | Research runtimes | Telemetry | Research artifacts | Research state | Experiments | AHI contracts (immature) | Search internals | DEVELOPMENT_ONLY | PARTIAL | LOW |
| EAC-DOM-023 | ELASTIC_CAPACITY_AND_SCALING | Capacity admission, scale coordination leases | EE identity/recovery semantics | Workers/runtime | Capacity signals | Admission decisions | Lease metadata | Metrics rollups | Capacity contracts | Scheduler internals | SCALE_COORDINATION | PARTIAL | MEDIUM |
| EAC-DOM-024 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | Dev workflows, harness DX, qualification tooling boundaries | Runtime semantic ownership | Local proofs | Docs/tooling | DX artifacts | None production | Local caches | N/A (guides) | Cursor rules, scripts | DEVELOPMENT_ONLY | STRONG | NONE |
| EAC-DOM-025 | TIER3_APPLICATION_ENVIRONMENT | Application manifests, profiles, host wiring composition | Execution engine internals | Hosting, plugins | Platform contracts | Application config bindings | Host profile state | Scaffold templates | Application env contracts | App-specific code | COMPOSITION_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-026 | APPLICATION_HOSTING | Deployment/hosting lifecycle, always-on runtime hosting | Tier-3 manifest semantics (owned by T3) | Execution admission | T3 manifests | Deployment records | Deployment/host runtime state | Health projections | Hosting contracts | Orchestrator internals | LIFECYCLE_AUTHORITY (host scope) | PARTIAL | MEDIUM |
| EAC-DOM-027 | CODE_CRAFT | Code manipulation domain services | Agent assembly ownership | Workspace IO | VCS/workspace | Code artifacts | Workspace drafts | Analysis caches | CodeCraft contracts | Tooling internals | COMPOSITION_AUTHORITY | UNKNOWN | LOW |
| EAC-DOM-028 | AUTONOMOUS_WORK | WorkerDefinition/Instance durable work semantics | Execution lifecycle owner (EE) | Background intake | Execution, catalog | Worker instances | Worker state | Queue views | AW contracts | Worker runtime | LIFECYCLE_AUTHORITY (worker scope) | PARTIAL | MEDIUM |
| EAC-DOM-029 | COLLABORATIVE_WORK | Workspace, membership, delegation, ContextView authority | Execution graph, decision outcome, governance WHETHER | GE for tool auth (declared overlap) | GE, Decision read | CW persistence events | Workspace/membership/ContextView | Presence projections | CW contracts | MP internals | COMPOSITION_AUTHORITY, GOVERNANCE_AUTHORITY (delegation) | PARTIAL | HIGH |
| EAC-DOM-030 | BACKGROUND_TASKS | Queue/bus abstractions, worker intake | Execution identity/lifecycle | EE for execution | Task envelopes | Queue metadata | Queue job state | Consumer lag metrics | `TaskQueue`, BG contracts | Broker adapters | SCALE_COORDINATION | PARTIAL | MEDIUM |
| EAC-DOM-031 | CAPABILITY_CATALOG_AND_DISCOVERY | Federated catalog read, rank, govern discovery rows | Vertical lifecycle, execution, Nexus | Domain registries | Domain sources | Catalog snapshots | Catalog index (read model) | Rank caches | Catalog contracts | Federator internals | DISCOVERY_AUTHORITY | STRONG | MEDIUM |
| EAC-DOM-032 | PROOF_RECEIPTS | Qualification receipt semantics | Runtime execution truth | CI/proof runners | Proof outputs | Receipt records | Receipt store | Verification views | Proof contracts | Gate scripts | DEVELOPMENT_ONLY | STRONG | NONE |
| EAC-DOM-033 | PLATFORM_PLUGINS | Extension discovery, enablement, trust vocabulary | Domain semantic ownership | Domain validation | Packaged extensions | Plugin enablement state | Enablement registry | Discovery cache | EP contracts | Loader internals | DISTRIBUTION_AUTHORITY (extension) | STRONG | LOW |
| EAC-DOM-034 | ENTERPRISE_RELIABILITY_LAYER | External effect reliability, UNKNOWN handling, reconciliation **composition** | Governance ALLOW/DENY (GE), execution lifecycle | Provider invocations | Tool effects, observability | ERL assessment artifacts | ProviderInvocation state (emerging) | Reconciliation projections | ERL contracts (emerging) | Submodule hubs | LIFECYCLE_AUTHORITY (effect slice) | PARTIAL | MEDIUM |

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

| Authority type | Domain count (approx.) |
|----------------|----------------------:|
| LIFECYCLE_AUTHORITY | 8 |
| EXECUTION_AUTHORITY | 1 (EE) |
| ORCHESTRATION_AUTHORITY | 2 |
| DECISION_AUTHORITY | 1 |
| GOVERNANCE_AUTHORITY | 2 (GE + partial CW) |
| EVIDENCE_AUTHORITY | 1 |
| DIAGNOSTIC_AUTHORITY | 1 |
| MEMORY_AUTHORITY | 1 |
| CONTEXT_AUTHORITY | 2 (CE + UCL) |
| RETRIEVAL_AUTHORITY | 1 |
| TOOL_AUTHORITY | 2 |
| DISCOVERY_AUTHORITY | 1 |
| DISTRIBUTION_AUTHORITY | 3 |
| PROVIDER_ABSTRACTION | 5 |
| COMPOSITION_AUTHORITY | 5 |
| SCALE_COORDINATION | 2 |
| DEVELOPMENT_ONLY | 4 |

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
| MEMORY | Memory provider ports | Store drivers | Monitor host wiring in `applications/_shared` (composition — EAC-13) |
| CONTEXT_ENGINEERING | `intergrax/context` contracts | Collectors/registry | CE-02 qualification in progress (working tree — not baseline) |
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
| Memory record | MEMORY | RAG, CE, UCL | Projections | Provider stores | **DUPLICATE_TRUTH_RISK** if CE/Nexus session stores mirror memory semantics (CL-EAC1-002) |
| Context bundle (assembled) | CONTEXT_ENGINEERING | Agent, Nexus consumer | UCL optimized views | Ephemeral + caches | **MEDIUM** — CE vs UCL vs Nexus `ContextCompiler` (CL-EAC1-003) |
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
| memory truth | MEMORY | RAG / CE | Read projections; no second writer | MEMORY.md, RAG.md | Partial | CL-EAC1-002 |
| context assembly | CONTEXT_ENGINEERING | UCL / Nexus | CE assembles; UCL optimizes lifecycle | ADR-CTX-01 | Partial | CL-EAC1-003 |
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
| **Execution ↔ Nexus** | **ALIGNED** | EE owns lifecycle/identity; Nexus internal orchestration only. Violation risk = **callers** importing Nexus from agents (CL-001), not dual lifecycle writers. |
| **Decision ↔ Reasoning** | **ALIGNED (canon)** | Decision owns outcome; Reasoning owns material. **LEGACY** app-local loops — RB-3; not second DOMAIN authority. |
| **GE ↔ CW ↔ Tools** | **DOCUMENTATION CONFLICT** | Declarative CW tool auth vs GE WHETHER — **ADR-GOV-01**; Tools remain mechanism. Severity **HIGH** until ADR closed. |
| **CE ↔ UCL ↔ Memory** | **PARTIAL OVERLAP** | Memory = durable truth; CE = assembly; UCL = optimization lifecycle. Nexus `ContextCompiler` blurs **internal** CE boundary (CL-003). |
| **Catalog ↔ Marketplace ↔ Distribution** | **ALIGNED (canon)** | Catalog read; Distribution lifecycle; Marketplace = supporting hub (not DOMAIN). |
| **Observability ↔ Diagnostics** | **ALIGNED** | Evidence vs Problem — explicit SSOT hierarchy. |
| **Reliability/HITL ↔ Continuation ↔ Recovery** | **ALIGNED** | EE continuation/recovery canonical; Reliability HITL + derived attempt ledger. |
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
| Memory record truth | MEMORY | CONTEXT_ENGINEERING / Nexus session | Session/memory wiring patterns | **DERIVED ONLY** / **CODE CONFLICT** risk — CL-002 |
| Context truth | CONTEXT_ENGINEERING | UCL / Nexus context engine | ADR-CTX-01; nexus context modules | **REAL DUPLICATE CANDIDATE** (assembly locus) — CL-003 |
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
| Context contracts (`intergrax/context`) | CONTEXT_ENGINEERING | Agent, Nexus (should consume ports) |
| Agent / harness contracts | AGENT_CONTRACTS_AND_ASSEMBLY | Nexus, Tier-2 |
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

| Finding ID | Class | Severity | Affected domains | Evidence | Recommended stage |
|------------|-------|----------|------------------|----------|-------------------|
| CL-EAC1-001 | CL | HIGH | NEXUS, AGENT_CONTRACTS, Tier-2 | `git grep` agents → `intergrax.runtime.nexus.*` | EAC-13 composition audit |
| CL-EAC1-002 | CL | MEDIUM | MEMORY, TIER3, NEXUS session | `applications/_shared/memory_wiring.py` SessionManager | EAC-9 persistence + EAC-5 flow |
| CL-EAC1-003 | CL | MEDIUM | CE, UCL, NEXUS | `intergrax/runtime/nexus/context/*` vs `intergrax/context/` | ADR-CTX-01 / EAC-7 |
| CL-EAC1-004 | ADR | HIGH | GOVERNED_EXECUTION, COLLABORATIVE_WORK, TOOLS | ADR-GOV-01, RB-5 | ADR close → EAC-7 |
| CL-EAC1-005 | CL | MEDIUM | TIER3, APPLICATION_HOSTING | EAC-0 LIFECYCLE_OVERLAP_RISK | EAC-8 handoff |
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
| ADR-CTX-01 | CE vs UCL vs Memory | Open — CL-003 |
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

## 18. Validation checklist (EAC-1 PASS)

| Check | Result |
|-------|--------|
| 34 DOMAIN rows in §5 | **PASS** |
| Every row has OWNS + MUST NOT OWN | **PASS** |
| Authority classification per row | **PASS** |
| Authoritative state table + duplicate flags | **PASS** |
| All listed hotspots audited (§9) | **PASS** |
| Public/internal documented (§6) | **PASS** |
| Duplicate authority classified (§10) | **PASS** |
| Plugin authority rule (§12) | **PASS** |
| Delegation explicit (§8) | **PASS** |
| ID defects handed off (§14) | **PASS** |
| No production code / contract changes in EAC-1 session | **PASS** |

---

*EAC-1 audit artifact. Independent verification against GitHub `development` remains mandatory for production claims.*

*Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z GitHuba.*
