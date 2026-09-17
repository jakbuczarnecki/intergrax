# EAC-0 — Enterprise Cross-Layer Canonical Layer Inventory

**Program:** Enterprise Architecture Cross-Layer Audit (EAC)  
**Task:** EAC-0 — Canonical Layer Inventory  
**Type:** Read-only inventory / classification (no runtime semantics changed)  
**Authority:** Current `development` architecture registry and 1:1 domain pairs — **not** the 2026-08-18 campaign topology alone.

| Gate | Value |
|------|-------|
| **EAC0_BASELINE_HEAD** | `7163fca0ea086f09b75382efe340a8aa06a90bc2` |
| **FINAL HEAD** | `6350e7e02dcfb45a63c98dde339adbfc16cac69c` |
| **Branch** | `development` |
| **HEAD == origin/development @ baseline** | **YES** |
| **Registry hub** | [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md) |
| **Prior rebaseline context** | [`CROSS_LAYER_ARCHITECTURE_REBASE_RB0.md`](CROSS_LAYER_ARCHITECTURE_REBASE_RB0.md) (evidence only) |

**Subordinate to:** domain architecture/plan pairs. This inventory **MUST NOT** redefine domain semantics.

---

## 1. Scope

EAC-0 answers: what canonical layers exist on current `development`, how they are classified, who owns them, coarse maturity/freeze posture, first-order dependencies, and preliminary risk flags for EAC-1…EAC-16.

**In scope:** boundaries, ownership, contracts (references), dependency direction, pluginability/persistence/information-flow **baselines**, legacy authority separation, duplicate-authority candidates.

**Out of scope:** internal algorithms, remediation, contract implementation, full dependency audit (EAC-3), full persistence audit (EAC-9).

---

## 2. Baseline SHA and drift watch

Commits since RB-2A (`4bcc0255…`) touching registry-relevant areas (sample @ `7163fca0…`):

- Governance evidence spine, continuation lifecycle authority, GOVERNANCE-FINAL certification record (**NOT CERTIFIED**).
- Multiplayer / ContextView (MP-5D–5E), Collaborative Work scope narrowing.
- Memory enterprise audit documentation; Diagnostics DG-003 evidence positions.
- NPSC-5F re-freeze after observability delivery QoS qualification.
- Scheduling / background qualification tests.

**Action:** Rows for **GOVERNED_EXECUTION**, **COLLABORATIVE_WORK** / **MULTIPLAYER_AI**, **MEMORY**, **OBSERVABILITY**, **DIAGNOSTICS** refreshed against current HEAD narratives.

---

## 3. Classification rules

From the registry [`Architecture artifact classification register`](../../architecture/intergrax_runtime_architecture.md#architecture-artifact-classification-register):

| Class | Rule |
|-------|------|
| **META_ARCHITECTURE** | Platform-wide governance, cross-domain semantic models, indexes — no 1:1 implementation plan as domain owner |
| **DOMAIN** | One reusable capability — canonical `architecture/<DOMAIN>.md` ↔ `maintainers/plans/<DOMAIN>.md` (exceptions documented as **ADR**) |
| **FEATURE** | Cross-layer coordination — `capabilities/architecture` ↔ `capabilities/plan`; domains retain runtime semantics |
| **SUPPORTING_MODEL / SATELLITE** | Subordinate typed model, maintainer hub, or satellite — **not** a separate platform authority |
| **LEGACY / HISTORICAL** | Former authority; replacement documented |

**Stop rule:** If two docs conflict on ownership → **ADR — ARCHITECTURE DECISION REQUIRED** (recorded in §17). No silent reconciliation.

---

## 4. Canonical domains (summary)

**Registry primary 24:** includes one **LEGACY** row (`CRITIC_VERIFICATION`) — not counted as current DOMAIN authority.

**Registry additional canonical pairs (8):** `DECISION_SYSTEM`, `GOVERNED_EXECUTION`, `AGENT_DISTRIBUTION`, `PLATFORM_PLUGINS`, `PROOF_RECEIPTS`, `AUTONOMOUS_WORK`, `CAPABILITY_CATALOG_AND_DISCOVERY`, `ENTERPRISE_RELIABILITY_LAYER` (plan **NEXT** — architecture canon only).

**1:1 pairs present on `development` but absent from registry index (registry gap):**

| Domain | Evidence | EAC classification |
|--------|----------|-------------------|
| `COLLABORATIVE_WORK` | [`COLLABORATIVE_WORK.md`](../../architecture/COLLABORATIVE_WORK.md) ↔ plan | **DOMAIN** — **ADR-REG-001** registry omission |
| `BACKGROUND_TASKS` | [`BACKGROUND_TASKS.md`](../../architecture/BACKGROUND_TASKS.md) ↔ plan | **DOMAIN** — **ADR-REG-002** registry omission |

**Separate architecture hub, not a DOMAIN pair:**

| Artifact | Classification | Owner |
|----------|----------------|-------|
| [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md) | **DOMAIN** (interpretation plane); plan slices in [`OBSERVABILITY` plan](../../maintainers/plans/OBSERVABILITY.md) | Central Diagnostics — **ADR-REG-003** split plan ownership |
| [`CAPABILITY_MARKETPLACE_ENGINE.md`](../../architecture/CAPABILITY_MARKETPLACE_ENGINE.md) | **SUPPORTING_MODEL** (ME-RB1 frozen hub) | `CAPABILITY_CATALOG_AND_DISCOVERY` (+ vertical lifecycle owners) |

**Counted canonical DOMAIN rows (main table):** **33** (23 non-legacy primary + 8 additional + `COLLABORATIVE_WORK` + `BACKGROUND_TASKS` + `DIAGNOSTICS`).

---

## 5. Meta architecture

| Artifact | Class | Role |
|----------|-------|------|
| [`INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md) | META_ARCHITECTURE | Capability creation, ownership, adoption, proof order |
| [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md) | META_ARCHITECTURE | Technical index / classification register |
| [`ARCHITECTURE_OVERVIEW.md`](../../architecture/ARCHITECTURE_OVERVIEW.md) | META_ARCHITECTURE | Public mental model (non-SSOT for contracts) |
| [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) | META_ARCHITECTURE | Cross-domain execution identity tree, UEA-INV-* |
| [`SYSTEM_INVARIANTS.md`](../../technical/guides/SYSTEM_INVARIANTS.md) | META_ARCHITECTURE | Compact `SYS-INV-*` cross-layer MUST/MUST NOT |
| [`MATURITY_TAXONOMY.md`](../../technical/guides/MATURITY_TAXONOMY.md) | META_ARCHITECTURE | A/I/P/E vocabulary |
| [`LAYER_COMPLETION_MODE.md`](../../technical/guides/LAYER_COMPLETION_MODE.md) | META_ARCHITECTURE | Domain closeout workflow |
| [`UNIFIED_EXECUTION_ARCHITECTURE_DIAGRAMS.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE_DIAGRAMS.md) | SUPPORTING_MODEL | UEA diagram pack |
| [`../architecture/EXECUTION_ENGINE.md`](../architecture/EXECUTION_ENGINE.md) | SUPPORTING_MODEL | Maintainer qualification/freeze hub for UER |

---

## 6. Features (cross-layer)

| Feature | Architecture | Plan | Notes |
|---------|--------------|------|-------|
| `TOKEN_OPTIMIZATION` | [`capabilities/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) | [`capabilities/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md) | Implemented foundation |
| `LANGCHAIN_INDEPENDENCE` | [`capabilities/architecture/LANGCHAIN_INDEPENDENCE.md`](../../capabilities/architecture/LANGCHAIN_INDEPENDENCE.md) | [`capabilities/plan/LANGCHAIN_INDEPENDENCE.md`](../../capabilities/plan/LANGCHAIN_INDEPENDENCE.md) | Roadmap; implementation not started |
| `MULTIPLAYER_AI` | [`capabilities/architecture/MULTIPLAYER_AI.md`](../../capabilities/architecture/MULTIPLAYER_AI.md) | [`capabilities/plan/MULTIPLAYER_AI.md`](../../capabilities/plan/MULTIPLAYER_AI.md) | Coordinates CW, GE, Decision, Context, etc. |
| `PROOF_DATA_PACKAGE_DISTRIBUTION` | architecture doc only | *no plan pair* | **ADR-REG-004** — promote or demote |

**Anchor domain for multiplayer:** `COLLABORATIVE_WORK` (not the feature).

---

## 7. Supporting / subordinate capabilities

| Subordinate | Parent DOMAIN | Why not separate authority |
|-------------|---------------|----------------------------|
| Execution Continuation / checkpoint consumer paths | `UNIFIED_EXECUTION_RUNTIME` | Lifecycle subset of Execution Engine; NPSC-5E qualified |
| NPSC-5E recovery plane | `UNIFIED_EXECUTION_RUNTIME` | Recovery contracts under EE freeze |
| NPSC-5F evidence / journal / export | `OBSERVABILITY` (+ EE consumption) | Evidence plane frozen; not execution control |
| Child execution / strategy routing | `UNIFIED_EXECUTION_RUNTIME` | EE internal routing |
| `DECISION_VERIFICATION`, `DECISION_DELIBERATION` | `DECISION_SYSTEM` | Registry-declared subordinate hubs |
| `UNCERTAINTY_MANAGEMENT`, `RECONCILIATION`, `EXTERNAL_EFFECT_CONTRACTS`, `RECOVERY_AND_COMPENSATION` | `ENTERPRISE_RELIABILITY_LAYER` | ERL subordinate hubs |
| `APPLICATION_RUNTIME_GRAPH_MODEL`, `APPLICATION_DEPENDENCY_MODEL` | `TIER3_APPLICATION_ENVIRONMENT` / `APPLICATION_HOSTING` | Composition models, not runtime owners |
| `CAPABILITY_MARKETPLACE_ENGINE` | `CAPABILITY_CATALOG_AND_DISCOVERY` | Single discovery/acquisition plane doc (ME-RB1) |
| `DECISION_APPROVAL_GOVERNANCE` | `COLLABORATIVE_WORK` / `GOVERNED_EXECUTION` | MP-4 cross-reference hub |
| CVL / `CRITIC_VERIFICATION` stack | *LEGACY* → `DECISION_SYSTEM` | Historical verification; Council = `DecisionStrategy` |
| Public Nexus root APIs | *forbidden* — internal to `NEXUS_EXECUTION_FLOW` | ORCHESTRATION-only implementation surface |

---

## 8. Legacy / historical authorities

| Historical | Former responsibility | Current replacement | Reachable in production? | Migration |
|------------|----------------------|---------------------|--------------------------|-----------|
| `CRITIC_VERIFICATION` / CVL | Critic-led verification canon | `DECISION_SYSTEM` + verification/deliberation strategies | Legacy code paths may exist (`intergrax/runtime/critic/*`) — **RB-3 hardening** | Wrap as DecisionStrategy; doc **HISTORICAL** |
| Second decision runtime in apps | Local judge loops | Execution-hosted Decision capability | Per-app inventory **E** | RB-3 |
| Nexus as public integration root | Task entry / lifecycle | Governed execution → **Execution Engine** | **NO** per P0 bypass inventory (0 production bypasses @ RB-2A) | Frozen role |
| Campaign 2026-08-18 layer topology | 36-layer mental model | Current registry + RB-0 map | N/A (documentation) | Evidence only |

---

## 9. Enterprise maturity matrix (E1–E5)

| Code | Meaning |
|------|---------|
| **E1** | Enterprise certified / frozen — qualification + architecture alignment |
| **E2** | Enterprise core — requalification or adoption gaps |
| **E3** | Hardening required — known boundary/contract/plugin gaps |
| **E4** | Developing — architecture ahead of runtime/E2E proof |
| **E5** | Legacy / not current authority |

**Distribution @ EAC-0 (33 DOMAIN rows):** E1 **3** · E2 **16** · E3 **10** · E4 **4** · E5 **0** (`CRITIC_VERIFICATION` — §8 only, not a DOMAIN row).

---

## 10. Frozen / certified owners

| Mechanism | Frozen owner | Evidence | Downstream MAY modify | Downstream MUST NOT modify |
|-----------|--------------|----------|------------------------|----------------------------|
| Execution Engine semantics | `UNIFIED_EXECUTION_RUNTIME` | [`EXECUTION_ENGINE.md`](../architecture/EXECUTION_ENGINE.md), EE-FINAL cert, post-freeze gap **PASS** | Adapters behind contracts, non-semantic wiring | Identity hierarchy, sole lifecycle owner, zero-bypass semantics |
| Decision System architecture | `DECISION_SYSTEM` | Hub **FROZEN/CANONICAL**; qualification family in plan | New `DecisionStrategy` plugins | Second decision runtime; decision = authorization |
| NPSC-5E recovery | `UNIFIED_EXECUTION_RUNTIME` | `NPSC_5E_FINAL_*` freeze docs | Provider stores behind ports | Parallel retry authority outside recovery plane |
| NPSC-5F evidence | `OBSERVABILITY` | R1–R4 + Final freeze; EE-FINAL-02 re-freeze **PASS** | Export sinks, read models | Evidence deciding execution admission |
| Multi-agent governance (5D) | `UNIFIED_EXECUTION_RUNTIME` / Nexus consumer | NPSC-5D Final freeze | Graph specs under contract | Public Nexus root |
| Diagnostic single authority (R1) | `DIAGNOSTICS` | `DIAGNOSTIC_ENGINE_SINGLE_AUTHORITY_ARCHITECTURE_R1.md` | Detector plugins, read APIs | Lifecycle mutation, second truth minting |
| Marketplace engine architecture | `CAPABILITY_MARKETPLACE_ENGINE` hub | ME-RB1 **ARCHITECTURE FROZEN** | Vertical product surfaces | Lifecycle authority, execution |
| Governed Execution | `GOVERNED_EXECUTION` | GOV-FINAL records | Policy plugins, wiring | **NOT enterprise-frozen** — cert **NOT CERTIFIED** @ latest audit |

---

## 11. Preliminary dependency map (first-order)

| From | Relationship | To |
|------|--------------|-----|
| `TIER3_APPLICATION_ENVIRONMENT` / `APPLICATION_HOSTING` | HOSTS | Governed execution entry, application composition |
| `APPLICATION_HOSTING` | DELEGATES | Execution admission |
| `UNIFIED_EXECUTION_RUNTIME` | USES | `GOVERNED_EXECUTION` (WHETHER) |
| `UNIFIED_EXECUTION_RUNTIME` | HOSTS | `DECISION_SYSTEM` (WHAT) |
| `UNIFIED_EXECUTION_RUNTIME` | DELEGATES | `ExecutionStrategy` → ORCHESTRATION / AGENTIC / INFERENCE |
| `ORCHESTRATION` | DELEGATES | `NEXUS_EXECUTION_FLOW` (internal) |
| `NEXUS_EXECUTION_FLOW` | USES | `AGENT_CONTRACTS_AND_ASSEMBLY` |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | USES | `TOOLS`, `SKILLS`, `LLM_ADAPTERS`, context/memory projections |
| `TOOLS` | USES | `INTEGRATIONS`, `GOVERNED_EXECUTION` (side effects) |
| `RAG` | USES | retrieval providers; **OBSERVES** memory projections |
| `MEMORY` | PERSISTS THROUGH | provider contracts; **EXTENDS THROUGH** projections |
| `CONTEXT_ENGINEERING` / `UNIFIED_CONTEXT_LIFECYCLE` | USES | memory/RAG/modality inputs |
| `OBSERVABILITY` | PERSISTS THROUGH | `RuntimeEvent` journal |
| `DIAGNOSTICS` | OBSERVES | `OBSERVABILITY` evidence → `Problem` read models |
| `ENTERPRISE_RELIABILITY_LAYER` | EXTENDS THROUGH | tool/integration external effects |
| `RELIABILITY_FAILURE_AND_HITL` | USES | execution pause/resume; **not** diagnostics retry |
| `BACKGROUND_TASKS` | DELEGATES | Execution identity via admission |
| `CAPABILITY_CATALOG_AND_DISCOVERY` | USES | marketplace engine; **DELEGATES** lifecycle to vertical owners |
| `AUTONOMOUS_WORK` | DELEGATES | worker admission → Execution |
| `COLLABORATIVE_WORK` | USES | governance, decision approval, context views |
| `PROOF_RECEIPTS` | OBSERVES | qualification / proof artifacts |
| `PLATFORM_PLUGINS` | EXTENDS THROUGH | entry-point discovery |

---

## 12. Preliminary pluginability matrix (summary)

| Verdict | Domains (inventory-level) |
|---------|---------------------------|
| **EXPLICIT** | `LLM_ADAPTERS`, `INTEGRATIONS`, `TOOLS`, `SKILLS`, `PLATFORM_PLUGINS`, `MEMORY` (provider guides), `DECISION_SYSTEM` (`DecisionStrategy`), `UNIFIED_EXECUTION_RUNTIME` (strategy/work ports) |
| **PARTIAL** | `RAG`, `OBSERVABILITY` (export sinks), `CAPABILITY_CATALOG_AND_DISCOVERY`, `GOVERNED_EXECUTION` (policy plugins; catalog runtime gap), `DIAGNOSTICS` |
| **GAP** | `ENTERPRISE_RELIABILITY_LAYER` (plan NEXT), `AUTONOMOUS_WORK`, `ADAPTIVE_HARNESS_INTELLIGENCE` |
| **NOT APPLICABLE** | `PLATFORM_FOUNDATION`, `PROOF_RECEIPTS` (artifact semantics) |
| **UNKNOWN / EAC-4** | `CODE_CRAFT`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`, `MODALITY` (deep replaceability proof) |

---

## 13. Preliminary persistence ownership

| Domain | Persistent state? | Canonical writer | Storage contract? | Pluggable? | Duplicate truth risk |
|--------|-----------------|------------------|-------------------|------------|----------------------|
| `UNIFIED_EXECUTION_RUNTIME` | Yes (checkpoints, lineage) | Execution runtime services | Ports under `intergrax.contracts.execution*` | Partial | Low vs OBS journal if boundaries hold |
| `OBSERVABILITY` | Yes | Event persistence layer | `RuntimeEventPersistence` | Partial | **QUAL** — DG-005 cross-topology not proven |
| `DIAGNOSTICS` | Yes | Diagnostic engine → Problem store | `intergrax.contracts.diagnostics` | Partial | Must not duplicate RuntimeEvent truth |
| `MEMORY` | Yes | Memory domain services | Provider extension guides | **EXPLICIT** | **CL** risk vs context projections |
| `DECISION_SYSTEM` | Yes | Decision artifact stores | Decision contracts | Partial | Low |
| `GOVERNED_EXECUTION` | Yes | Policy/grant persistence | Policy contracts | Partial | **ADR** side-effect authority overlap w/ CW |
| `COLLABORATIVE_WORK` | Yes | CW persistence | CW contracts | Partial | Delegation vs governance grants |
| `CAPABILITY_CATALOG_AND_DISCOVERY` | Yes | Catalog services | Catalog contracts | Partial | vs marketplace projection |
| Others | Varies | Domain-specific | See EAC-9 | — | Flagged in main table |

---

## 14. Information object ownership (coarse)

| Information object | Canonical owner |
|--------------------|-----------------|
| `ExecutionRequest` / logical task intake | `APPLICATION_HOSTING` / Tier-3 intake → Execution admission |
| Execution identity (Task/Run/Attempt/Execution/Event) | `UNIFIED_EXECUTION_RUNTIME` (mint) · `OBSERVABILITY` (record) |
| `Decision` artifact / version lineage | `DECISION_SYSTEM` |
| Authorization / governance outcome | `GOVERNED_EXECUTION` |
| Context package / assembly | `CONTEXT_ENGINEERING` |
| Context lifecycle optimization | `UNIFIED_CONTEXT_LIFECYCLE` |
| Principal-scoped `ContextView` | `COLLABORATIVE_WORK` (MP-5) composer ports |
| Memory projection | `MEMORY` |
| Retrieval result | `RAG` |
| Tool invocation / effect descriptor | `TOOLS` (+ ERL external effect) |
| Continuation / checkpoint state | `UNIFIED_EXECUTION_RUNTIME` (NPSC-5E) |
| `RuntimeEvent` / evidence envelope | `OBSERVABILITY` |
| `Problem` / diagnostic read model | `DIAGNOSTICS` |
| Capability descriptor | `CAPABILITY_CATALOG_AND_DISCOVERY` |
| Plugin / provider metadata | `PLATFORM_PLUGINS` + vertical domain |
| Proof / qualification receipt | `PROOF_RECEIPTS` |
| Worker / Responsibility durable model | `AUTONOMOUS_WORK` |

---

## 15. Cross-layer risk flags (by layer)

See **Cross-Layer Risk** column in §16. Program-level hotspots:

- **DUPLICATE_AUTHORITY_RISK:** Decision vs Reasoning; Governance vs CW tool auth; Context Engineering vs UCL vs Memory; Marketplace vs Catalog vs Distribution.
- **LEGACY_BYPASS_RISK:** CVL paths; intake normalization residuals (**E**).
- **QUALIFICATION_GAP:** ERL, Governance enterprise cert, Observability DG-005, Background universal production qual.
- **CONTRACT_GAP_RISK:** GOV policy catalog runtime; ERL plan-only.
- **INFORMATION_FLOW_RISK:** Diagnostics vs Observability SSOT hierarchy (managed — explicit).

---

## 16. Required main table

| Layer | Class | Canonical Owner | OWNS | MUST NOT OWN | Architecture Doc | Implementation Area | Public Contract(s) | Pluginability | Persistence Role | Frozen? | Qualification Evidence | Enterprise Status | Cross-Layer Risk |
|-------|-------|-------------------|------|--------------|------------------|---------------------|----------------------|---------------|------------------|---------|------------------------|-------------------|------------------|
| PLATFORM_FOUNDATION | DOMAIN | Platform Foundation | Tier topology, dependency rules, spine gates | Domain feature semantics | `PLATFORM_FOUNDATION.md` | `intergrax/`, CI guards, doctor | `SYSTEM_INVARIANTS`, tier contracts | N/A | None authoritative | No | PF plan §6.1ax open | **E2** | CONTRACT_GAP_RISK |
| UNIFIED_EXECUTION_RUNTIME | DOMAIN | Execution Engine | Execution lifecycle, strategy routing, identity mint, terminal semantics | Decision semantics, governance ALLOW/DENY, observability journal ownership | `UNIFIED_EXECUTION_RUNTIME.md` + maintainer `EXECUTION_ENGINE.md` | `intergrax/runtime/execution/` | `intergrax.contracts.execution*`, UEA refs | EXPLICIT | Owns checkpoint/lineage ports | **Yes** | EE-FINAL, NPSC-5D/E, post-freeze PASS | **E1** | LEGACY_BYPASS_RISK (monitor) |
| ORCHESTRATION | DOMAIN | Orchestration capability | Strategy class semantics, orchestration contracts | Nexus internals, execution identity | `ORCHESTRATION.md` | `intergrax/runtime/orchestration/` | Orchestration contracts | PARTIAL | None | Role frozen | EE / UEA | **E2** | LIFECYCLE_OVERLAP_RISK |
| NEXUS_EXECUTION_FLOW | DOMAIN | Nexus execution flow | Graph execution, fan-out/merge **inside** ORCHESTRATION | Public root API, execution admission | `NEXUS_EXECUTION_FLOW.md` | `intergrax/runtime/nexus/` | Nexus internal contracts | PARTIAL | Consumer of checkpoints | **Yes** (role) | NPSC multi-agent 5D | **E2** | LEGACY_BYPASS_RISK |
| DECISION_SYSTEM | DOMAIN | Decision System | Decision lifecycle, authoritative outcome, version lineage | Execution lifecycle, authorization, retry, orchestration | `DECISION_SYSTEM.md` | `intergrax/runtime/execution/decision*` | `intergrax.contracts.decision*` | EXPLICIT | Decision artifacts | **Yes** (semantics) | Decision qual family | **E1** | DUPLICATE_AUTHORITY_RISK (Reasoning) |
| GOVERNED_EXECUTION | DOMAIN | Governed Execution | WHETHER — policy, side-effect admission, governance evidence | WHAT decisions, execution scheduling, diagnostics | `GOVERNED_EXECUTION.md` | `intergrax/runtime/governance/`, policy | `intergrax.contracts.governance*` | PARTIAL | Policy/grants | No (not E1) | GOV-FINAL **NOT CERTIFIED** | **E3** | DUPLICATE_AUTHORITY_RISK, QUALIFICATION_GAP |
| REASONING_AND_COGNITION | DOMAIN | Reasoning & Cognition | Cognitive strategies, reasoning material | Authoritative decision outcome | `REASONING_AND_COGNITION.md` | `intergrax/runtime/reasoning/` | Reasoning contracts | PARTIAL | Session/scratch per design | No | Domain plan | **E3** | DUPLICATE_AUTHORITY_RISK |
| AGENT_CONTRACTS_AND_ASSEMBLY | DOMAIN | Agent contracts | Agent assembly, harness kernel contract, step loop | Nexus graph planning, execution admission | `AGENT_CONTRACTS_AND_ASSEMBLY.md` | `intergrax/runtime/agent*`, harness | Agent/`HarnessKernel` contracts | EXPLICIT | Agent state merge hooks | No | ACP qual | **E2** | INFORMATION_FLOW_RISK |
| AGENT_DISTRIBUTION | DOMAIN | Agent Distribution | Agent package distribution, installability | Marketplace catalog engine, execution | `AGENT_DISTRIBUTION.md` | `agents/`, distribution runtime | `intergrax.contracts.agent_distribution*` | PARTIAL | Package metadata stores | No | AD audits | **E2** | DUPLICATE_AUTHORITY_RISK (Marketplace) |
| LLM_ADAPTERS | DOMAIN | LLM Adapters | LLM provider abstraction | Agent planning, governance | `LLM_ADAPTERS.md` | `intergrax/llm/` | LLM provider contracts | EXPLICIT | None | No | Provider qual partial | **E2** | VENDOR_COUPLING_RISK |
| TOOLS | DOMAIN | Tools | Tool contracts, invocation spine | Integration drivers, governance rules | `TOOLS.md` | `intergrax/tools/` | `intergrax.contracts.tools*` | EXPLICIT | Idempotency stores | No | U5 zero bypass | **E2** | DUPLICATE_AUTHORITY_RISK (GE) |
| SKILLS | DOMAIN | Skills | Skill composition over tools | Tool drivers, marketplace lifecycle | `SKILLS.md` | `intergrax/skills/` | Skill contracts | EXPLICIT | None | No | Domain plan | **E2** | DUPLICATE_AUTHORITY_RISK (Tools) |
| INTEGRATIONS | DOMAIN | Integrations | Integration profiles, provider wiring | Tool semantics, execution | `INTEGRATIONS.md` | `intergrax/integrations/` | Integration contracts | EXPLICIT | Vendor stores | No | PBA qual partial | **E2** | VENDOR_COUPLING_RISK |
| RAG | DOMAIN | RAG | Retrieval orchestration, chunk/query semantics | Memory authoritative store, execution | `RAG.md` | `intergrax/rag/` | RAG contracts | PARTIAL | Index stores | No | Campaign **E** findings | **E3** | IMPLEMENTATION_COUPLING_RISK |
| MEMORY | DOMAIN | Memory | Memory stores, projections, consolidation policy surface | Context lifecycle owner, execution | `MEMORY.md` | `intergrax/memory/` | Memory provider contracts | EXPLICIT | Owns memory truth | No | Recent enterprise audit docs | **E3** | PERSISTENCE_OWNERSHIP_RISK |
| CONTEXT_ENGINEERING | DOMAIN | Context Engineering | Context assembly, collectors | UCL optimization lifecycle, memory truth | `CONTEXT_ENGINEERING.md` | `intergrax/context/` | Context contracts | PARTIAL | Projection caches | No | Domain plan | **E3** | DUPLICATE_AUTHORITY_RISK (UCL/Memory) |
| UNIFIED_CONTEXT_LIFECYCLE | DOMAIN | UCL | Conversation context lifecycle optimization | Raw memory authority, CE assembly rules | `UNIFIED_CONTEXT_LIFECYCLE.md` | UCL runtime modules | UCL contracts (ADR-UCL-001) | PARTIAL | Optimization state | No | ADR-UCL-001 | **E3** | LIFECYCLE_OVERLAP_RISK |
| MODALITY | DOMAIN | Modality | Multimodal ingestion contracts | Execution, memory truth | `MODALITY.md` | modality adapters | Modality contracts | UNKNOWN | Media stores | No | Plan | **E4** | PLUGINABILITY_RISK |
| OBSERVABILITY | DOMAIN | Observability | RuntimeEvent recording, reconstruction, export | Diagnostic Problem semantics, execution control | `OBSERVABILITY.md` | `intergrax/runtime/observability/` | `intergrax.contracts.observability*` | PARTIAL | **Canonical** event journal | **Yes** (5F) | NPSC-5F finals | **E1** | QUALIFICATION_GAP (DG-005) |
| DIAGNOSTICS | DOMAIN | Central Diagnostics | Problem detection, operator read models | Retry, lifecycle, evidence minting | `DIAGNOSTICS.md` | `intergrax/runtime/diagnostics/` | `intergrax.contracts.diagnostics` | PARTIAL | Problem store | R1 arch frozen | DIAG hardening quals | **E3** | INFORMATION_FLOW_RISK |
| RELIABILITY_FAILURE_AND_HITL | DOMAIN | Reliability / HITL | Pause/resume, HITL escalation paths | Governance ALLOW/DENY, diagnostics | `RELIABILITY_FAILURE_AND_HITL.md` | reliability runtime | HITL contracts | PARTIAL | Checkpoint refs | Partial | Qualified paths | **E2** | LIFECYCLE_OVERLAP_RISK (Continuation) |
| ADAPTIVE_HARNESS_INTELLIGENCE | DOMAIN | AHI | Harness intelligence, design search hooks | Production governance authority | `ADAPTIVE_HARNESS_INTELLIGENCE.md` | AHI modules | AHI contracts | GAP | Research artifacts | No | Satellites | **E4** | DUPLICATE_AUTHORITY_RISK (GE) |
| ELASTIC_CAPACITY_AND_SCALING | DOMAIN | Elastic capacity | Capacity admission, scale coordination | EE identity/recovery semantics | `ELASTIC_CAPACITY_AND_SCALING.md` | capacity runtime | Capacity contracts | PARTIAL | Lease metadata | No | Plan | **E3** | LIFECYCLE_OVERLAP_RISK |
| EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | DOMAIN | Experimentation/DX | Dev workflows, harness DX boundaries | Runtime semantics | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | docs/tooling | N/A | N/A | None | No | P2-ARCH-13 | **E4** | NOT APPLICABLE |
| TIER3_APPLICATION_ENVIRONMENT | DOMAIN | Tier-3 environment | Application wiring, manifests, profiles | Execution engine internals | `TIER3_APPLICATION_ENVIRONMENT.md` | `applications/*`, scaffold | Application env contracts | PARTIAL | Host config | No | Composition **E** | **E2** | PLUGINABILITY_RISK |
| APPLICATION_HOSTING | DOMAIN | Application Hosting | Always-on hosting, deployment lifecycle | Tier-3 manifest semantics | `APPLICATION_HOSTING.md` | hosting runtime | Hosting contracts | PARTIAL | Deployment state | No | Domain plan | **E2** | LIFECYCLE_OVERLAP_RISK |
| CODE_CRAFT | DOMAIN | Code Craft | Code manipulation domain services | Agent assembly | `CODE_CRAFT.md` | `intergrax/codecraft/` | CodeCraft contracts | UNKNOWN | Workspace artifacts | No | Plan | **E4** | CONTRACT_GAP_RISK |
| AUTONOMOUS_WORK | DOMAIN | Autonomous Work | WorkerDefinition/Instance, durable work semantics | Execution lifecycle owner | `AUTONOMOUS_WORK.md` | autonomous work runtime | AW contracts | GAP | Worker state | No | AW audits **E** | **E3** | QUALIFICATION_GAP |
| COLLABORATIVE_WORK | DOMAIN | Collaborative Work | Workspace, membership, delegation, ContextView authority | Execution graph, decision outcome | `COLLABORATIVE_WORK.md` | `intergrax/runtime/collaborative*` | CW contracts | PARTIAL | CW persistence | MP-1 closed | MP-1..MP-6 ADRs | **E2** | DUPLICATE_AUTHORITY_RISK (GE) |
| BACKGROUND_TASKS | DOMAIN | Background Tasks | Queue/bus abstractions, worker intake | Execution identity/lifecycle | `BACKGROUND_TASKS.md` | queue runtime, worker | `TaskQueue`, BG contracts | PARTIAL | Queue metadata | No | BG-01, SCHED-01 partial | **E3** | LEGACY_BYPASS_RISK |
| CAPABILITY_CATALOG_AND_DISCOVERY | DOMAIN | Capability Catalog | Federated catalog read, rank, govern | Vertical lifecycle, execution | `CAPABILITY_CATALOG_AND_DISCOVERY.md` | catalog services | Catalog contracts | PARTIAL | Catalog indices | No | CC V1 audit | **E2** | DUPLICATE_AUTHORITY_RISK |
| PROOF_RECEIPTS | DOMAIN | Proof Receipts | Qualification receipt semantics | Runtime execution | `PROOF_RECEIPTS.md` | proof receipt runtime | Proof contracts | N/A | Receipt store | No | Proof gates | **E2** | NOT APPLICABLE |
| PLATFORM_PLUGINS | DOMAIN | Platform Plugins | Extension discovery, enablement | Domain semantics | `PLATFORM_PLUGINS.md` | plugin registry | EP contracts | EXPLICIT | None | Extension cert PASS | **E2** | PLUGINABILITY_RISK |
| ENTERPRISE_RELIABILITY_LAYER | DOMAIN | ERL | External effect reliability, UNKNOWN, reconciliation composition | Governance ALLOW/DENY | `ENTERPRISE_RELIABILITY_LAYER.md` | ERL runtime modules | ERL contracts (emerging) | GAP | ProviderInvocation | No | Plan **NEXT** | **E4** | CONTRACT_GAP_RISK, QUALIFICATION_GAP |

---

## 17. Open architecture decisions (ADR)

| ID | Topic | Conflict / gap |
|----|-------|----------------|
| **ADR-REG-001** | Registry completeness | `COLLABORATIVE_WORK` 1:1 pair exists; absent from registry additional table |
| **ADR-REG-002** | Registry completeness | `BACKGROUND_TASKS` 1:1 pair exists; absent from registry |
| **ADR-REG-003** | Diagnostics plan ownership | `DIAGNOSTICS.md` SSOT vs plan slices under `OBSERVABILITY` plan |
| **ADR-REG-004** | `PROOF_DATA_PACKAGE_DISTRIBUTION` | Architecture file without matching `capabilities/plan` pair |
| **ADR-GOV-01** | Side-effect authority | CW vs Governed Execution declarative tool auth (RB-5 **F**) |
| **ADR-CTX-01** | Context authority | Context Engineering vs UCL vs Memory projection boundaries |
| **ADR-MKT-01** | Marketplace vs Catalog vs Distribution | ME-RB1 frozen hub vs vertical lifecycle owners |

---

## 18. Internal-domain handoff findings (ID)

| ID | Owner | Finding |
|----|-------|---------|
| ID-MEM-01 | MEMORY | Internal consolidation/projection algorithms — out of EAC scope |
| ID-RAG-01 | RAG | Retrieval heuristic quality — domain session |
| ID-RC-01 | REASONING_AND_COGNITION | Strategy internals |
| ID-NX-01 | NEXUS_EXECUTION_FLOW | Graph scheduler heuristics |
| ID-DX-01 | EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE | Cursor workflow rules placement |

---

## 19. Cross-layer findings (CL / QUAL / LEGACY) — inventory only

| Class | Finding | Owner |
|-------|---------|-------|
| **LEGACY** | `intergrax/runtime/critic/*` CVL surface | DECISION_SYSTEM / RB-3 |
| **QUAL** | Governance enterprise **NOT CERTIFIED** @ GOV-FINAL | GOVERNED_EXECUTION |
| **QUAL** | Observability DG-005 cross-topology persistence | OBSERVABILITY |
| **QUAL** | ERL architecture without implementation plan closure | ENTERPRISE_RELIABILITY_LAYER |
| **CL** | Memory vs context projection duplicate truth risk | MEMORY + CONTEXT_ENGINEERING |
| **CL** | RAG store coupling to memory implementations | RAG + MEMORY |
| **ADR** | Duplicate side-effect authority (GE vs CW) | GOVERNANCE + COLLABORATIVE_WORK |

---

## 20. Next-stage inputs (EAC-1…)

- Responsibility matrix expansion from §16 OWNS/MUST NOT OWN.
- Contract inventory (`intergrax/contracts/*` mapping per DOMAIN) — EAC-2.
- Dependency direction proofs — EAC-3.
- Replaceability scenarios — EAC-4.
- Full information-flow and authority-flow graphs — EAC-5, EAC-7.

---

## 21. EAC-0 validation checklist

| Check | Result |
|-------|--------|
| Every registry DOMAIN (incl. additional 8) in main table | **PASS** |
| `COLLABORATIVE_WORK`, `BACKGROUND_TASKS`, `DIAGNOSTICS` documented | **PASS** (registry gap flagged) |
| Legacy `CRITIC_VERIFICATION` separated | **PASS** (§8) |
| Subordinate hubs not promoted to DOMAIN | **PASS** (§7) |
| One owner per DOMAIN row | **PASS** |
| OWNS / MUST NOT OWN per DOMAIN | **PASS** (§16) |
| Maturity per DOMAIN | **PASS** (§9, §16) |
| E1 claims have qualification refs | **PASS** (EE, Decision, 5F, partial 5E) |
| Frozen claims cite freeze docs | **PASS** (§10) |
| Pluginability references contracts | **PASS** (high-level; EAC-4 deepens) |
| Persistent domains declare writer or UNKNOWN | **PASS** (§13, §16) |

---

*Document generated by EAC-0 session. Independent code audit on GitHub remains required for production claims.*
