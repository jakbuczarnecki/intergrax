# EBH-1 — Canonical Boundary Baseline Reconciliation & Drift Audit

**Program:** Enterprise Boundary Hardening (EBH)  
**Task:** EBH-1 — Canonical Boundary Baseline Reconciliation & Drift Audit  
**Type:** As-built audit / baseline reconciliation (no production remediation in scope)  
**Supersedes as current baseline:** Reconciles and extends EAC-0, EAC-1, EAC-2, RB-0 — those artifacts remain **historical audit input** where this document disagrees.

| Gate | Value |
|------|-------|
| **EBH1_SESSION_START_HEAD** | `58a842d8fd5e5a7493607c88b9c629cb6c75ef0b` |
| **EBH1_EVIDENCE_HEAD** | `2c82f7253c51c39912eef5d88ed806e5c1960460` |
| **Branch** | `development` |
| **HEAD == origin/development @ session open** | **YES** (`58a842d8…`) |
| **HEAD vs origin/development @ artifact commit** | **AHEAD** — local `2c82f7253` (MP-6B doc close) atop `58a842d8`; push publishes both |
| **Production code changed by EBH-1** | **NO** |
| **EBH1_R1_SESSION_START_HEAD** | `190af29cc46d358f84c8d66254445284eae0171a` |
| **EBH1_R1_EVIDENCE_HEAD** | `4aca3ce86a55ab2f351bdb9ef00c27513dcc9a6b` |
| **Production code changed by EBH-1-R1** | **NO** — [R1 record](EBH-1-R1_CONTEXTVIEW_OWNERSHIP_BASELINE_CORRECTION.md) |

**SSOT strategy:** This document is the **single current enterprise boundary baseline** for ownership, authorities, contract families, and drift vs EAC/RB artifacts. Domain semantics remain owned by `docs/project/architecture/<DOMAIN>.md` pairs. Do not fork a parallel taxonomy — update this file or reconcile EAC rows in place in a follow-up doc pass (EBH-2+). **EBH-1-R1** corrects Principal-scoped ContextView ownership; EAC rows marked **HISTORICAL (pre-R1)** where reconciled in EAC1/EAC2.

---

## 1. Scope

EBH-1 answers:

1. Does current production code still respect declared ownership, public contracts, and dependency directions?
2. Where did documentation or code drift from canonical architecture?
3. What is the reconciled **DOMAIN → PUBLIC CONTRACT → CONSUMER → ADAPTER/STRATEGY → PROVIDER** map @ evidence HEAD?

**In scope:** 34 canonical DOMAIN rows (EAC-0 §16), 31 peer authority types (EAC-1 §4.A), 99 contract families (EAC-2 §4), import/boundary sampling, composition-root exceptions, Nexus privacy, evidence/control separation.

**Out of scope:** Remediation implementation, MP-6 delivery, new ADR resolution, full EAC-3…EAC-16 dependency proofs (deferred to EBH-2…EBH-10).

---

## 2. Git provenance

| Field | SHA / note |
|-------|------------|
| SESSION_START_HEAD | `58a842d8fd5e5a7493607c88b9c629cb6c75ef0b` |
| EVIDENCE_HEAD | `2c82f7253c51c39912eef5d88ed806e5c1960460` |
| origin/development @ finalize fetch | `58a842d8fd5e5a7493607c88b9c629cb6c75ef0b` |
| Intervening commit (MP-6 docs) | `2c82f7253` — no EBH-1 ownership semantic change identified |
| Parallel working-tree drift | **YES** — uncommitted deltas in `intergrax/runtime/nexus/*`, application factories, qualification tests (not certified in this artifact) |

**Git safety @ session:** `git reset` / `rebase` / `stash` / `clean` / `amend` / force-push: **NO**.

---

## 3. Historical baseline inputs

| Artifact | Role | EBH-1 assessment | Notes |
|----------|------|------------------|-------|
| [EAC0](ENTERPRISE_CROSS_LAYER_CANONICAL_LAYER_INVENTORY_EAC0.md) | Layer inventory @ `08b182e9…` | **PARTIAL** | 34 DOMAIN taxonomy **CURRENT**; evidence SHA **STALE**; post-EAC commits (GR-10, UCL, memory qual) not reflected |
| [EAC1](ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md) | Ownership @ `4403f1b5…` | **PARTIAL** | Peer authority register **CURRENT**; GE↔CW **ADR-GOV-01** still open |
| [EAC2](ENTERPRISE_CROSS_LAYER_CONTRACT_INVENTORY_EAC2.md) | Contracts @ `8552029b…` | **PARTIAL** | Family taxonomy **CURRENT**; module count **605** vs 601 @ EAC-2R1; bypass rows revalidated @ EBH1_HEAD |
| [RB0](CROSS_LAYER_ARCHITECTURE_REBASE_RB0.md) | Migration ledger @ `4bcc0255…` | **SUPERSEDED** for drift watch | Execution zero-bypass narrative **still directionally valid**; use P0 inventory + EBH-1 import audit for current proof |

### HISTORICAL CLAIMS TO REVALIDATE (summary)

| Claim | Source | EBH-1 status |
|-------|--------|--------------|
| EE sole execution lifecycle owner | RB0, EAC1 | **CONFIRMED** (structural) |
| 0 production execution bypasses @ RB-2A | RB0 | **PARTIAL** — not re-run full P0 inventory; no new bypass surfaced in EBH-1 sample |
| Nexus not public root | EAC2 EAC-CON-007/008 | **DRIFTED** — public `intergrax/contracts/*` still imports Nexus types |
| GE enterprise certified | EAC0 §10 | **CONFIRMED NOT CERTIFIED** |
| Diagnostics no lifecycle mutation | RB0 §6 | **CONFIRMED** @ grep sample (retry strings = store OCC / taxonomy, not EE control) |
| ERL does not own Run tree | EAC1R3 | **CONFIRMED** — `ExecutionLifecyclePort` documents handoff only |
| CVL legacy → Decision | EAC0 §8 | **CONFIRMED** — `runtime/critic/*` remains LEGACY |
| CW ContextView authority | EAC0, MP-5 | **CONFIRMED** — `intergrax/collaborative_work/` + contracts |
| Platform plugins ≠ domain semantics | EAC2 EAC-CON-042 | **CONFIRMED** |

---

## 4. Audit method

| Layer | Technique | Evidence class |
|-------|-----------|----------------|
| Domain inventory | `intergrax/` top-level tree + EAC-0 §16 | STRUCTURAL |
| Authorities | EAC-1 §4.A + spot reads (`execution_lifecycle_port.py`) | DOC + STRUCTURAL |
| Contracts | `intergrax/contracts/` census (**605** modules), EAC-2 register | STRUCTURAL |
| Nexus privacy | `rg 'from intergrax.runtime.nexus' intergrax/contracts agents intergrax/agents` | STATIC |
| Provider leakage | `rg default_/sqlite_` cross-tier samples | STATIC |
| Composition | `intergrax/applications/_shared/*` spot classification | STATIC |
| Replaceability | Test grep for fake/in-memory provider patterns | STATIC (not full EAC-4) |
| Evidence vs control | Diagnostics module grep for execution admission | STATIC |

**Not certified without EBH-9:** runtime replaceability, full import graph, behavioral boundary E2E.

---

## 5. Current domain inventory (code-grounded)

Production locus (verified paths exist):

| Domain / layer | Code locus (primary) | Canonical owner (semantic) | Status |
|----------------|----------------------|----------------------------|--------|
| Platform foundation | `intergrax/core/`, CI guards | PLATFORM_FOUNDATION | CURRENT |
| Unified execution | `intergrax/runtime/execution/` | UNIFIED_EXECUTION_RUNTIME | CURRENT |
| Orchestration | `intergrax/runtime/orchestration/`, EE strategy | ORCHESTRATION | CURRENT |
| Nexus (internal) | `intergrax/runtime/nexus/` | NEXUS_EXECUTION_FLOW | CURRENT (internal) |
| Decision | `intergrax/runtime/execution/decision*`, `contracts/decision*` | DECISION_SYSTEM | CURRENT |
| Governance | `intergrax/runtime/governance/`, `contracts/*governance*` | GOVERNED_EXECUTION | CURRENT |
| Agents / harness | `intergrax/agents/`, `agents/` | AGENT_CONTRACTS_AND_ASSEMBLY | CURRENT (leak watch) |
| Tools / skills | `intergrax/tools/`, `intergrax/skills/` | TOOLS, SKILLS | CURRENT |
| Integrations | `intergrax/integrations/` | INTEGRATIONS | CURRENT |
| LLM | `intergrax/llm_adapters/`, `intergrax/llm/` | LLM_ADAPTERS | CURRENT |
| Memory | `intergrax/memory/` | MEMORY | CURRENT |
| RAG | `intergrax/rag/`, `intergrax/knowledge/` | RAG | CURRENT |
| Context | `intergrax/context/` | CONTEXT_ENGINEERING | CURRENT |
| UCL | `intergrax/ucl/` | UNIFIED_CONTEXT_LIFECYCLE | CURRENT |
| Observability | `intergrax/runtime/observability/` | OBSERVABILITY | CURRENT |
| Diagnostics | `intergrax/runtime/diagnostics/` | DIAGNOSTICS | CURRENT |
| Reliability / HITL | `intergrax/runtime/human/`, reliability contracts | RELIABILITY_FAILURE_AND_HITL | CURRENT |
| ERL | `intergrax/runtime/enterprise_reliability/`, `contracts/enterprise_reliability/` | ENTERPRISE_RELIABILITY_LAYER | CURRENT (emerging) |
| Collaborative work | `intergrax/collaborative_work/` | COLLABORATIVE_WORK | CURRENT |
| Collaborative activity | `intergrax/contracts/collaborative_activity.py` (+ MP-6 tests) | COLLABORATIVE_ACTIVITY (feature) | PARTIAL / in flight |
| Background / queue | `intergrax/queueing/`, `intergrax/runtime/task/` | BACKGROUND_TASKS | CURRENT |
| Hosting | `intergrax/hosting/`, `intergrax/applications/` | APPLICATION_HOSTING, TIER3 | CURRENT |
| Plugins | `intergrax/core/plugins/` | PLATFORM_PLUGINS | CURRENT |
| Autonomous work | `intergrax/autonomous_work/` | AUTONOMOUS_WORK | CURRENT |
| Applications (T3) | `applications/` | TIER3_APPLICATION_ENVIRONMENT | CURRENT |

---

## 6. Canonical layer matrix (condensed)

Full OWNS/MUST NOT columns: **EAC-0 §16** (revalidated **CURRENT**). EBH-1 adds code-locus column only where EAC-0 used generic paths:

| Domain | Owns (semantic) | MUST NOT own | Public contracts (family) | Consumers | Providers / impl | Status |
|--------|-----------------|--------------|---------------------------|-----------|------------------|--------|
| UNIFIED_EXECUTION_RUNTIME | Lifecycle, identity, strategy routing | Decision, governance, Obs journal | EAC-CON-001…005, 019, 097, 099 | All runtime | EE services, checkpoint adapters | CONFIRMED |
| DECISION_SYSTEM | WHAT outcome | Lifecycle, WHETHER | EAC-CON-009, 010 | EE, CW read | DecisionStrategy plugins | CONFIRMED |
| GOVERNED_EXECUTION | WHETHER admission | WHAT, Run tree | EAC-CON-011, 012, 078, 093 | EE, Tools, Agents | Policy plugins | CONFIRMED* |
| COLLABORATIVE_WORK | Workspace, membership, delegation; **Principal-scoped ContextView** (MP-5) | Execution graph owner; generic CE assembly | EAC-CON-013, 014; MP-5 `context_view_*` | Agents, CE (generic assembly), GE read | CW stores; MP-5 composer in `collaborative_work/` | CONFIRMED* |
| ORCHESTRATION / NEXUS | Strategy topology / internal graph | Public entry, identity | EAC-CON-006, 007 (internal) | EE only | NexusLoop, GraphExecutor | CONFIRMED internal |
| MEMORY | Memory records; reference-first read | ContextView truth, CE bundle truth | EAC-CON-020, 021 | CE, MP-5 adapters | Memory providers | CONFIRMED |
| RAG / KNOWLEDGE | Retrieval orchestration; knowledge refs | Memory truth, ContextView truth | EAC-CON-023 | CE, MP-5 adapters | RAG stack | CONFIRMED |
| CONTEXT_ENGINEERING | Generic model-facing context assembly, budgeting, compile plan | Principal-scoped ContextView; UCL lifecycle; memory truth | EAC-CON-072, collectors | Agents, runtime, Nexus consumer | CE + Nexus context engine impl (**CL-EAC1-003**) | CONFIRMED with impl-locus watch |
| UNIFIED_CONTEXT_LIFECYCLE | Context lifecycle optimization | Memory truth; MP-5 composition | EAC-CON-027, 028 | CE, MP-5 adapters | UCL optimizer | CONFIRMED |
| OBSERVABILITY / DIAGNOSTICS | Facts vs Problems | Each other's authority | EAC-CON-029…033 | Cross-read | Stores, detectors | CONFIRMED |
| ERL | External uncertainty, reconciliation | GE WHETHER, EE tree direct mutation | EAC-CON-036, 094 | EE, Tools | ERL orchestrator | CONFIRMED |
| PLATFORM_PLUGINS | EP lifecycle | Domain semantics | EAC-CON-042 | T3, domains | Loaders | CONFIRMED |

\* **ADR-GOV-01** — overlapping WHETHER *dimensions* (GE platform vs CW membership/delegation); not duplicate peer owner in EAC-1 register.

---

## 7. Authority matrix

| Authority | Canonical owner | Public contract anchor | Duplicate peer authority? |
|-----------|-----------------|------------------------|----------------------------|
| EXECUTION_LIFECYCLE_AUTHORITY | UNIFIED_EXECUTION_RUNTIME | `intergrax/contracts/execution*` | **NO** |
| DECISION_AUTHORITY | DECISION_SYSTEM | `intergrax/contracts/decision*` | **NO** |
| GOVERNANCE_AUTHORITY | GOVERNED_EXECUTION | `runtime_execution_policy_admission`, `agent_runtime_governance` | **NO** (ADR overlap **YES**) |
| COLLABORATIVE_AUTHORITY (WHO / workspace) | COLLABORATIVE_WORK | `collaborative_work`, `physical_delegation_governance` | **NO** |
| ORCHESTRATION_AUTHORITY | ORCHESTRATION (+ internal NEXUS) | `orchestration_topology` | **NO** |
| MEMORY_AUTHORITY | MEMORY | `intergrax/memory/contracts/*` | **NO** |
| KNOWLEDGE_RETRIEVAL_AUTHORITY | RAG | `knowledge_reference_read` | **NO** |
| CONTEXT_LIFECYCLE_AUTHORITY | UNIFIED_CONTEXT_LIFECYCLE | `intergrax/ucl/contracts/*` | **NO** |
| CONTEXT_ASSEMBLY_AUTHORITY | CONTEXT_ENGINEERING | `intergrax/context/`, context budget contracts | **NO** |
| PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY | COLLABORATIVE_WORK (MP-5) | `context_view_*`, `intergrax/collaborative_work/context_view_*` | **NO** |
| COLLABORATIVE_ACTIVITY_AUTHORITY | MP-6 contracts (feature) | `collaborative_activity.py` | **NO** (immature) |

**Deprecated / historical authority label (pre–EBH-1-R1 — do not use in active baseline):** `CONTEXT_VIEW_COMPOSITION_AUTHORITY` → CONTEXT_ENGINEERING — **SUPERSEDED**; conflated generic CE assembly with MP-5 Principal-scoped ContextView ([ADR-MP-006](../../technical/adr/entries/2026-09-17/ADR-MP-006.md)).

### 7.1 Context ownership distinction (canonical @ EBH-1-R1)

| Semantic responsibility | Canonical owner | Consumers |
| ----------------------- | --------------- | --------- |
| General context assembly | CONTEXT_ENGINEERING | Agents / runtime |
| Context lifecycle optimization | UNIFIED_CONTEXT_LIFECYCLE | CE / runtime |
| Memory records / recall | MEMORY | ContextView / CE / RAG |
| Knowledge retrieval | RAG / KNOWLEDGE | ContextView / CE |
| Principal visibility | COLLABORATIVE_WORK / MP-5 | ContextView pipeline |
| Principal-scoped composition semantics | COLLABORATIVE_WORK / MP-5 | Agents / runtime consumers |
| ContextView source reads | Source-domain contracts (Memory, RAG, UCL, CW) | MP-5 (ports/adapters) |
| ContextView default composition implementation | MP-5 / Collaborative Work (`DefaultContextViewComposer`) | ContextView consumers |

**File location ≠ ownership:** `intergrax/contracts/context_view_*.py` are **public contract locus** for MP-5; they do **not** transfer semantic ownership to CONTEXT_ENGINEERING. Default implementations under `intergrax/collaborative_work/` are supporting evidence; ownership follows architecture + ADR-MP-006.
| DIAGNOSTIC_INTERPRETATION_AUTHORITY | DIAGNOSTICS | `contracts/diagnostics/*` | **NO** |
| OBSERVABILITY_AUTHORITY | OBSERVABILITY | `runtime_event`, observability contracts | **NO** |
| EXTERNAL_EFFECT_UNCERTAINTY_AUTHORITY | ENTERPRISE_RELIABILITY_LAYER | `enterprise_reliability/*` | **NO** |
| PLUGIN_LIFECYCLE_AUTHORITY | PLATFORM_PLUGINS | `core/plugins/*` | **NO** |

**Blocking duplicate-authority findings:** **NONE** at peer-type level (EAC-1R3 gate holds @ EBH1_HEAD).

---

## 8. Public contract inventory (reconciled counts)

| Class | Count @ EAC-2R1 | EBH-1 @ HEAD | Notes |
|-------|----------------:|-------------:|-------|
| EAC-CON families | 99 | **99** (register unchanged) | +4 contract modules since EAC-2R1 |
| CANONICAL public | 75 | **75** | Ownership rows unchanged |
| INTERNAL / Nexus | 14 | **14** | Leak paths persist |
| LEGACY | 3 | **3** | EAC-CON-082 critic |
| BYPASS / leak documented | 8 | **8** | Revalidated |
| UNOWNED semantic owner | 0 | **0** | — |

**Duplicates / bypass (high signal):**

| ID | Issue | Classification |
|----|-------|----------------|
| EAC-CON-008, 063–066 | Nexus surface reachable from agents/T3 wiring | BYPASS |
| EAC-CON-011 vs 013/014/093 | GE WHETHER vs CW facts/delegation | DUPLICATE semantics **ADR** |
| EAC-CON-024 vs 085 | CE owns contract; Nexus hosts context engine impl | INTERNAL impl locus |
| `collaborative_activity` | Parallel to CW history semantics | EMERGING (MP-6) |

---

## 9. Dependency direction matrix (major flows)

Expected: `Consumer → public contract ← provider`.

| Interaction | Expected | Observed @ EBH1_HEAD | Verdict |
|-------------|----------|----------------------|---------|
| T3 host → EE admission | Contract | `runtime_execution_admission` via wiring | VALID |
| EE → GE | Contract | Policy admission ports | VALID |
| EE → Nexus | Internal port | `runtime/execution/orchestration.py` | VALID INTERNAL |
| Agent → Nexus | Contract only | **Direct** `runtime.nexus` imports in `intergrax/agents/*`, `agents/*` | **CROSS-LAYER LEAK** |
| `contracts/*` → Nexus | Forbidden | **3 modules** import Nexus types | **CROSS-LAYER LEAK** |
| RAG → Memory | Reference port | `MemoryReferenceReadPort` pattern | VALID |
| DIAG → OBS | Read evidence | Reconstruction / contributor ports | VALID |
| ERL → EE lifecycle | Port handoff | `ExecutionLifecyclePort` | VALID |
| T3 → Memory default | Composition | `memory_control_wiring` → `DefaultMemoryControlPlane` | VALID COMPOSITION |
| Nexus session → T3 | Internal | `memory_wiring` + SessionManager | **PARTIAL BYPASS** (EAC2-F-003) |

---

## 10. Provider / plugin matrix

| Capability | Public contract | Default impl | Replaceable w/o consumer change? | Evidence |
|------------|-----------------|--------------|-----------------------------------|----------|
| Model provider | LLM adapter SPI | Vendor adapters | **PARTIAL** | Provider qual tests |
| Memory provider | EAC-CON-020 | Default control plane / stores | **PROVEN** (extensive qual + fakes) | MEM qual family |
| RAG retriever | RAG contracts | Engine + loaders | **DECLARED ONLY** | Partial integration tests |
| Vector store | Integration SPI | Vendor-specific | **PROVEN** (e2e vendor tests) | MEM/RAG e2e |
| Observability export | EAC-CON-030 | OTLP bridges | **PARTIAL** | NPSC-5F |
| Plugin loader | EAC-CON-042 | core/plugins | **PROVEN** | Extension cert |
| Integration | EAC-CON-048 | Provider bundles | **PARTIAL** | PBA qual |
| Diagnostics sink | EAC-CON-031 | Problem store ports | **PARTIAL** | DIAG qual |
| ERL reconciliation | EAC-CON-036 | `runtime/enterprise_reliability/*` | **DECLARED ONLY** | GR-7 qual samples |

---

## 11. Persistence boundary

| Domain | Pattern | Leakage finding |
|--------|---------|-----------------|
| EE decision checkpoints | `sqlite_decision_*_persistence.py` in `runtime/execution/` | **LOW** — adapter in execution package, behind ports |
| CW | `sqlite_repository.py`, `postgresql_repository.py` | **LOW** — domain-owned adapters |
| Nexus session | `sqlite_session_storage.py` | **MEDIUM** — reachable from T3 `memory_wiring` (composition coupling) |
| Memory consumers | No widespread `sqlite` in `intergrax/rag/` semantic core @ sample | **PASS** sample |

**Policy in persistence:** No **HIGH** finding @ sample — governance evaluators remain outside raw SQL modules.

---

## 12. Cross-domain reference ownership

Canonical reference DTO families (EAC-2): `MemoryReferenceRead`, `KnowledgeReferenceRead`, `UclReferenceRead`, execution identity IDs, `RecoveryLifecycleIntent`, delegation refs.

**Principal-scoped ContextView consumes canonical references; it does not own source-domain truth.**

| Source read contract | Canonical owner | MP-5 consumer surface |
|----------------------|-----------------|-------------------------|
| `MemoryReferenceReadPort` | MEMORY | `MemoryContextSourcePort` + B5 adapters |
| `KnowledgeReferenceReadPort` | RAG / KNOWLEDGE | `KnowledgeContextSourcePort` + adapters |
| `UclReferenceReadPort` | UNIFIED_CONTEXT_LIFECYCLE | `UclContextSourcePort` + adapters |
| `CollaborativeWorkReferenceReadPort` | COLLABORATIVE_WORK | `CollaborativeWorkContextSourcePort` + adapters |

MP-5D source ports remain **consumer-side abstractions** (COLLABORATIVE_WORK / MP-5); adapters translate domain refs only — no hydration of source payloads in ContextView composition.

**Duplicate ref models:** No new duplicate **peer** ref authority found @ EBH-1 sample. Watch: Nexus `RuntimeAnswer` used in **contracts** (`runtime_cost.py`, `runtime_mapping.py`) — **duplicate transport shape** vs contract-neutral DTO (**FINDING**).

---

## 13. Execution / Decision / Governance / Authority

| Separation | Owner unique? | Evidence |
|------------|---------------|----------|
| Decision WHAT | **YES** | `DECISION_SYSTEM` + frozen hub |
| Governance WHETHER | **YES** (peer) | `GOVERNED_EXECUTION`; ADR dimension overlap with CW |
| Authority WHO | **YES** | `intergrax/collaborative_work/authority.py` |
| Execution lifecycle | **YES** | EE runtime; P0 narrative |
| Orchestration topology | **YES** | ORCHESTRATION + internal Nexus |

Modules mixing authorities @ spot check: **Nexus tool inner governance** (uncommitted WIP) — potential **orchestration-side policy**; requires follow-up against GE boundary (not certified here).

---

## 14. Observability / Diagnostics / Evidence

| Check | Result |
|-------|--------|
| Obs records facts | **PASS** |
| DIAG interprets Problems | **PASS** |
| DIAG mutates EE lifecycle | **NO** @ `runtime/diagnostics` sample |
| Evidence drives retry/resume without policy | **NO** structural hit in production diagnostics |
| Functional evidence store vs RuntimeEvent | **WATCH** — dual evidence shapes (EAC / RB **E** class) |

---

## 15. Memory / RAG / UCL / ContextView

| Boundary | Status |
|----------|--------|
| Memory owns durable truth | **CONFIRMED** |
| RAG orchestrates retrieval, reads memory refs | **CONFIRMED** |
| UCL owns optimization lifecycle | **CONFIRMED** (`intergrax/ucl/`) |
| CE owns **generic** context assembly (`CONTEXT_ASSEMBLY_AUTHORITY`) | **CONFIRMED**; Nexus hosts CE assembly algorithms (**CL-EAC1-003**) — not Principal-scoped ContextView |
| CW / MP-5 owns **Principal-scoped ContextView** visibility & composition semantics | **CONFIRMED** ([ADR-MP-006](../../technical/adr/entries/2026-09-17/ADR-MP-006.md); `collaborative_work/context_view_*`) |
| Memory / RAG / UCL / CW expose source-domain read semantics via canonical contracts | **CONFIRMED** — consumed by MP-5 ports/adapters; ContextView does not own domain truth |

---

## 16. Collaborative Work / Activity

| Surface | Owner | Status |
|---------|-------|--------|
| Workspace, delegation, enforcement | COLLABORATIVE_WORK | **CONFIRMED** |
| Activity semantic history | `contracts/collaborative_activity.py` | **MP-6 in progress** — tests in `tests/unit/collaborative_work/test_mp6*` |
| MP-6 roadmap | External to EBH remediation | Record only |

---

## 17. ERL / Recovery

- Runtime: `intergrax/runtime/enterprise_reliability/` (orchestration, handoff, reconciliation).
- **ExecutionLifecyclePort** explicitly: ERL produces `RecoveryLifecycleIntent`; UER applies lifecycle (**CONFIRMED**).
- ERL `governance_orchestration.py` consumes GE **read** material — does not transfer GOVERNANCE_AUTHORITY (EAC-CON-087).

---

## 18. Nexus privacy

| Question | Answer |
|----------|--------|
| Public `intergrax/contracts/*` importing Nexus? | **YES** — `host_profile_slices.py` (`ContextBudgetPolicy`), `runtime_cost.py`, `runtime_mapping.py` (`RuntimeAnswer`) |
| Tier-2 `agents/*` importing Nexus? | **YES** — ~30 production agent modules (qualifiers, product agents) |
| `intergrax/agents/*` importing Nexus? | **YES** — UAEP, reference harness, bridges |
| EE importing Nexus? | **YES** — expected internal consumer |

**Nexus public dependency count (contracts package):** **3 modules** @ EBH1_HEAD.

---

## 19. Escape-hatch inventory (boundary-relevant only)

| Location | Mechanism | Class |
|----------|-----------|-------|
| Agent UAEP bridges | Nexus `RuntimeContext` / response schema types | **CONTRACT BYPASS** |
| `catalog_declarative_invoker` | Concrete Nexus tool dispatch | **LEGACY / QUAL** |
| Public contracts → Nexus DTOs | Type import | **CONTRACT BYPASS** |
| Composition roots | Concrete `Default*` wiring | **LEGITIMATE** when confined to `applications/_shared` |

Repository-wide `type: ignore` / `Any` not inventoried (EBH-8 scope).

---

## 20. Replaceability evidence

| Claim | EBH-1 verdict |
|-------|---------------|
| Memory providers | **PROVEN** |
| LLM adapters | **PARTIAL** |
| Tool/integration providers | **PARTIAL** |
| RAG stack | **DECLARED ONLY** |
| Nexus session store | **NOT PROVEN** for external agents (harness coupling) |
| Governance policy plugins | **PARTIAL** (GR-10 qual progress) |

---

## 21. Drift matrix (core)

| Historical claim | Source | Current code evidence | Status | Severity | Follow-up |
|------------------|--------|----------------------|--------|----------|-----------|
| Nexus not on public contract surface | EAC2, RB0 | 3× `contracts` → `runtime.nexus` | **DRIFTED** | HIGH | EBH-3 |
| Agents depend on UAEP contracts only | Tier rules | Widespread Nexus imports | **DRIFTED** | HIGH | EBH-3 |
| 34 DOMAIN one-owner table | EAC0 | Doc + code locus align | **CONFIRMED** | — | Maintain registry ADRs |
| EAC peer authority uniqueness | EAC1R3 | No second peer owner found | **CONFIRMED** | — | — |
| GE enterprise certified | EAC0 | GOV-FINAL not certified | **CONFIRMED** | MEDIUM | Governance qual |
| 0 EE production bypasses | RB-2A | Not re-proven @ 58a842d8 | **UNVERIFIED** | MEDIUM | Re-run P0 inventory |
| CVL legacy | EAC0 | `runtime/critic/*` present | **CONFIRMED** | LOW | RB-3 wrap |
| SessionManager in T3 memory wiring | EAC2 EAC-CON-067 | `memory_wiring.py` still couples | **PARTIAL** | MEDIUM | EBH-7 |
| ERL separate from RECOVERY_POLICY | EAC1R3 | Ports + package layout | **CONFIRMED** | — | — |
| Contract module count 601 | EAC2 | **605** modules | **PARTIAL** | INFO | EBH-2 ownership pass |
| Collaborative activity contracts | MP-6 plan | `collaborative_activity.py` + tests | **PARTIAL** | INFO | MP-6 (not EBH) |

---

## 22. Findings

### CRITICAL

*None* @ peer-authority duplication layer. (Nexus contract leakage classified **HIGH**, not duplicate lifecycle owner.)

### HIGH

| ID | Severity | Owner | Boundary | Evidence | Risk | Remediation task |
|----|----------|-------|----------|----------|------|------------------|
| EBH-F-H-001 | HIGH | PLATFORM_FOUNDATION + AGENT_CONTRACTS | Public contracts → Nexus | `intergrax/contracts/host_profile_slices.py`, `runtime_cost.py`, `runtime_mapping.py` | Stable Nexus coupling; blocks LangChain-independence / vendor-neutral contracts | **EBH-3** — extract DTOs to `contracts/` |
| EBH-F-H-002 | HIGH | AGENT_CONTRACTS_AND_ASSEMBLY | Tier-2/3 agents → Nexus | `rg from intergrax.runtime.nexus` in `agents/`, `intergrax/agents/` | Agents not replaceable without Nexus; tier boundary violation | **EBH-3** dependency audit + adapter shim |
| EBH-F-H-003 | HIGH | GOVERNED_EXECUTION + COLLABORATIVE_WORK | Side-effect WHETHER | EAC-CON-011 vs 013/014; ADR-GOV-01 | Inconsistent admission for delegated/membership tools | **ARCHITECTURE DECISION REQUIRED** then EBH-6 |
| EBH-F-H-004 | HIGH | EBH-1 baseline documentation | ContextView ownership conflation | EBH-1 §7/§15/§24 pre-R1; EAC2 EAC-CON-024…026 | Baseline assigned Principal-scoped ContextView to CE | **RESOLVED BY EBH-1-R1** — see [R1 record](EBH-1-R1_CONTEXTVIEW_OWNERSHIP_BASELINE_CORRECTION.md) |

### MEDIUM

| ID | Severity | Owner | Boundary | Evidence | Remediation |
|----|----------|-------|----------|----------|-------------|
| EBH-F-M-001 | MEDIUM | CONTEXT_ENGINEERING + NEXUS | CE contract vs Nexus impl | EAC-CON-085, `nexus/context/*` | Hidden CE authority in Nexus | EBH-6 communication audit |
| EBH-F-M-002 | MEDIUM | TIER3_APPLICATION_ENVIRONMENT | Session store wiring | `memory_wiring.py`, EAC-CON-063 | Internal session leaks to composition | EBH-7 |
| EBH-F-M-003 | MEDIUM | GOVERNED_EXECUTION | Enterprise certification | EAC0 §10 | Operator trust gap | Governance qual program |
| EBH-F-M-004 | MEDIUM | UNIFIED_EXECUTION_RUNTIME | P0 bypass inventory stale | RB-2A vs 58a842d8 | False confidence | Re-run P0 @ HEAD |

### LOW

| ID | Severity | Owner | Evidence | Remediation |
|----|----------|-------|----------|-------------|
| EBH-F-L-001 | LOW | DECISION_SYSTEM | `runtime/critic/*` legacy | RB-3 strategy wrap |
| EBH-F-L-002 | LOW | PLATFORM_FOUNDATION | ADR-REG-001…004 registry gaps | Doc registry update |

### INFO

| ID | Note |
|----|------|
| EBH-F-I-001 | Uncommitted Nexus governance WIP — parallel drift; not in EBH1_EVIDENCE_HEAD certification |
| EBH-F-I-002 | +4 contract modules since EAC-2R1 — EBH-2 should diff owners |

---

## 23. Required remediation tasks (ordered)

1. **EBH-2** — Public Contract Ownership Audit (605 modules, owner cells).
2. **EBH-3** — Dependency Direction & Private Boundary Audit (Nexus/agent/contract leaks).
3. **ADR-GOV-01** — Resolve GE vs CW WHETHER convergence rule (**ARCHITECTURE DECISION REQUIRED**).
4. **EBH-7** — Application composition & SessionManager decoupling.
5. **EBH-1-R1** — Principal-scoped ContextView ownership baseline correction — **CLOSED** (documentation only).
6. Re-run **PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY** @ current HEAD.

---

## 24. Canonical communication map

```text
Applications (applications/, intergrax/applications/_shared)
    ↓ composition wires concrete providers
Public capability contracts (intergrax/contracts/*, domain */contracts/*)
    ↓
Domain control plane (runtime/execution, governance, decision, collaborative_work, memory, rag, context, ucl, diagnostics, observability, enterprise_reliability)
    ↓ ports
Provider / adapter / plugin implementations (integrations, llm_adapters, memory stores, tools/providers, core/plugins)
```

**Cross-capability flows (code-backed):**

```text
ContextViewRequest → visibility (MP-5C) → policy decision → MP-5E composer
    → MP-5D source ports → B5 adapters → source-domain *ReferenceReadPort
    → canonical refs → isolation / scope compatibility → ContextView → Agent/runtime consumer
(Generic CE context assembly — separate path — CONTEXT_ASSEMBLY_AUTHORITY → CE bundle / compile plan)
Decision WHAT → GE WHETHER → EE admission / tool invoke
EE → ORCHESTRATION → Nexus (internal) → Tools/Agents
EE → Observability (RuntimeEvent) → Diagnostics (Problem)
ERL reconciliation → RecoveryLifecycleIntent → EE ExecutionLifecyclePort
```

---

## 25. Final EBH-1 status

| Result | **EBH-1 — BASELINE ESTABLISHED** (ContextView ownership clarified @ EBH-1-R1) |
|--------|-----------------------------------------------|

**Rationale:** Domain and peer-authority taxonomy from EAC **holds** on current `development`. **Material drift** persists on Nexus privacy (public contracts + agents) and **ADR-GOV-01**. No duplicate execution lifecycle owner found. **EBH-F-H-004** (ContextView ownership conflation) **resolved** by EBH-1-R1. Remaining HIGH findings (H-001…H-003) unchanged.

**Next recommended task:** **EBH-2** (public contract ownership @ 605 modules) then **EBH-3** (Nexus/agent leaks) per program table.

---

*Independent verification on GitHub @ `EBH1_EVIDENCE_HEAD` remains required for production certification claims.*
