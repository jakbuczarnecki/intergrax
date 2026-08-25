# Unified Execution Architecture — Diagram Pack Specification

**Status:** Canonical diagram-pack specification (asset contract only — no production graphics in-repo yet)  
**Classification:** `SUPPORTING_MODEL / SATELLITE` — subordinate to [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) (`META_ARCHITECTURE`); **not** a new DOMAIN and **not** paired with an implementation plan  
**Owner:** Intergrax Platform Architecture (visual semantics coordination)  
**Audience:** Principal architects, technical writers, graphic producers, Cursor documentation/integration sessions  
**Registered in:** [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md#architecture-artifact-classification-register)

---

## 1. Purpose

This document specifies the **canonical diagram pack** required to visualize the frozen Unified Execution Architecture (UEA). It exists so that:

1. **Architects** know exactly which visual views exist, what each must communicate, and which ambiguities each view must eliminate.
2. **Graphic producers** (outside Cursor) can generate assets without re-deciding architecture semantics.
3. **Implementers and Cursor sessions** can embed assets consistently in architecture docs, README, and future domain slices without inventing new execution semantics.

**Authority chain:** All diagram semantics **inherit** from [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md). This file defines **what to draw and where to place it**; it does **not** redefine UEA-INV-* invariants, identity hierarchy, or ownership boundaries.

**Out of scope for this document:** runtime code, Python/API/schema implementation, and **generation of final graphic binaries** (`.png`, `.svg`, `.drawio`, exported `.mmd`, screenshots).

---

## 2. Diagram pack principles

| Principle | Requirement |
|-----------|-------------|
| **Explain, do not invent** | Diagrams visualize frozen UEA semantics; they must not introduce new runtime concepts, owners, or invariants. |
| **Canonical and cross-consistent** | Shared notation (identity layers, Execution Tree, strategy branches, cross-cutting planes) must match across all twelve views. Color/shape legend is stable across the pack. |
| **README-grade vs deep technical** | One executive README-grade graphic (see [§6](#6-readme-promotion-candidate)); all other views are architecture-doc depth. README simplification is allowed only when semantics remain true. |
| **Topology ≠ runtime tree** | Orchestration definition/topology (`NodeId`) and runtime Execution Tree (`ExecutionId`) must never be conflated in a single ambiguous tree (**UEA-INV-004**, **UEA-INV-006**). |
| **Nexus ≠ universal executor** | No diagram may imply Nexus always runs, directly executes Agent internals, or owns Run lifecycle / execution identity / budget / DIAG / observability store (**UEA-INV-007**, **UEA-INV-008**). |
| **Agent ≠ Execution** | Agent/AgentDefinition is a reusable executor; Execution is the schedulable runtime unit (**UEA-INV-003**). |
| **Identity visible where relevant** | ExecutionId and Execution Tree parent/child links must appear wherever scheduling, governance, observability, cancellation, or recovery semantics are shown. Do not hide identity behind opaque boxes. |
| **Record vs invent** | Observability **records** truth; DIAG **interprets** evidence. Neither owns lifecycle or mints ExecutionId (**UEA-INV-015**, **UEA-INV-016**). |
| **Execution boundary coordinates** | Execution layer coordinates contracts; it does not absorb Governance, Budget, Observability, DIAG, Queue, Checkpoint, or Agent internals (**UEA-INV-018**). |
| **Light and dark variants** | Every production asset ships as paired light/dark variants for theme-aware embedding (aligned with existing platform README `<picture>` pattern). |

---

## 3. Asset strategy

### 3.1 Directory and naming convention

Future production assets live under:

```text
docs/assets/architecture/unified_execution/
docs/assets/architecture/unified_execution/fullsize/   # optional caption-rich companion pages (markdown only)
```

**Basename pattern:** `unified_execution_<view_slug>` where `<view_slug>` matches the diagram ID slug in [§4](#4-required-architecture-views).

**Per-view file set (contract — files are created by graphic production, not this slice):**

| Variant | Filename pattern | Required |
|---------|------------------|----------|
| SVG (light) | `<basename>.light.svg` | Yes |
| SVG (dark) | `<basename>.dark.svg` | Yes |
| PNG (light) | `<basename>.light.png` | Yes (README / GitHub rendering) |
| PNG (dark) | `<basename>.dark.png` | Yes |

**Example (diagram A):**

```text
docs/assets/architecture/unified_execution/unified_execution_full_architecture.light.svg
docs/assets/architecture/unified_execution/unified_execution_full_architecture.dark.svg
docs/assets/architecture/unified_execution/unified_execution_full_architecture.light.png
docs/assets/architecture/unified_execution/unified_execution_full_architecture.dark.png
```

**Alignment with existing repo conventions:** Application-level proofs use `*-light.png` / `*-dark.png` with `<picture>` + `prefers-color-scheme` (see root [`README.md`](../../../README.md)). Platform UEA assets use the `.light.` / `.dark.` infix before extension for clearer pairing with SVG sources. Optional `fullsize/<basename>.md` companions may hold extended captions and alt text (markdown only).

### 3.2 Asset register (summary)

| Diagram ID | Basename | Light/dark | SVG+PNG | Primary embedding target(s) |
|------------|----------|------------|---------|----------------------------|
| **UEA-DIAG-A** | `unified_execution_full_architecture` | Yes | Yes | UEA §intro or dedicated visuals subsection; runtime hub cross-link |
| **UEA-DIAG-B** | `unified_execution_simple_execute_flow` | Yes | Yes | UEA §3; UER entry-path docs (future link-only) |
| **UEA-DIAG-C** | `unified_execution_orchestration_nexus_flow` | Yes | Yes | UEA §6–§7; NEXUS_EXECUTION_FLOW (future link-only) |
| **UEA-DIAG-D** | `unified_execution_topology_vs_execution_tree` | Yes | Yes | UEA §5; ORCHESTRATION (future link-only) |
| **UEA-DIAG-E** | `unified_execution_identity_lifecycle` | Yes | Yes | UEA §3 |
| **UEA-DIAG-F** | `unified_execution_retry_pause_resume_cancel` | Yes | Yes | UEA §10, §14 |
| **UEA-DIAG-G** | `unified_execution_nested_orchestration` | Yes | Yes | UEA §6; scenario E |
| **UEA-DIAG-H** | `unified_execution_distributed_queue_worker` | Yes | Yes | UEA §11, §19 |
| **UEA-DIAG-I** | `unified_execution_governance_budget_inheritance` | Yes | Yes | UEA §12–§13 |
| **UEA-DIAG-J** | `unified_execution_observability_diag_causal_flow` | Yes | Yes | UEA §17–§18 |
| **UEA-DIAG-K** | `unified_execution_checkpoint_recovery` | Yes | Yes | UEA §16 |
| **UEA-DIAG-L** | `unified_execution_component_ownership` | Yes | Yes | UEA §20, §22 |

### 3.3 Embedding contract (for future integration sessions)

When assets exist, embed using theme-aware `<picture>` (PNG) or inline SVG reference (architecture docs). Each embedding **must** include:

1. Link to this specification section for the diagram ID.
2. Short caption (from **Caption guidance** below).
3. Alt text describing mandatory concepts, not decorative marketing copy.
4. Pointer to [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) for normative semantics.

Do **not** embed diagrams in README until the README promotion candidate (§6) is produced and approved in a dedicated slice.

---

## 4. Required architecture views

### UEA-DIAG-A — Full Unified Execution Architecture

| Field | Specification |
|-------|---------------|
| **Audience** | Principal architects, new platform engineers, Cursor sessions onboarding to execution model |
| **Purpose** | Single end-to-end mental model: identity hierarchy, strategy branches, Execution Tree, cross-cutting planes, and ownership boundaries |
| **Required concepts** | Application/caller; Task; Run; Attempt; root Execution; strategy split (**inference**, **agentic**, **orchestration**); AgentEngine/UAEP below agentic; Nexus below orchestration; child Executions; cross-cutting planes: governance, budget, observability, DIAG, checkpoint/recovery, queue/distributed execution |
| **Mandatory relationships/arrows** | Task → Run → Attempt → root Execution; strategy selection on Execution; orchestration path: Execution → Nexus → child Executions; agentic path: Execution → AgentEngine → UAEP; governance/budget flow Run → Execution → child Execution (narrowing); observability records events along identity spine; DIAG reads evidence (dashed/secondary); queue/worker attaches via transport → Execution admission |
| **Ambiguity to eliminate** | Nexus is always involved; Agent == Execution; Observability or DIAG own execution lifecycle |
| **Level of detail** | Hub-level: all major boxes, no UAEP step internals, no broker implementation detail |
| **Target embedding** | [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) (near §1–§2); link from [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) Unified Execution subsection |
| **Asset basename** | `unified_execution_full_architecture` |
| **Caption guidance** | *"Unified Execution Architecture: Task through Event identity, three execution strategies, Execution Tree composition, and cross-cutting governance, budget, observability, DIAG, checkpoint, and distributed execution planes. Nexus orchestrates Executions only when strategy is orchestration."* |
| **README-eligible** | **No** (source for README derivative — see §6) |

---

### UEA-DIAG-B — Simple `execute()` Flow

| Field | Specification |
|-------|---------------|
| **Audience** | Application developers, UER implementers |
| **Purpose** | Show the direct execution path that does **not** require Nexus |
| **Required concepts** | Caller/Application; `execution.execute(...)` entry; Task; Run; Attempt; single root Execution; one lightweight strategy path (inference or simple agentic — pick one canonical example, label the other as alternate dashed branch) |
| **Mandatory relationships/arrows** | Linear: caller → execute API → Task → Run → Attempt → root Execution → strategy executor → result; explicit **absence** of Nexus on this path (optional muted "not used" indicator) |
| **Ambiguity to eliminate** | Every request becomes orchestration; Nexus required for observability/governance |
| **Level of detail** | Minimal boxes; no child Executions |
| **Target embedding** | UEA §3; UEA §27 view #2 cross-reference |
| **Asset basename** | `unified_execution_simple_execute_flow` |
| **Caption guidance** | *"Direct execute() path: even the simplest call materializes Task → Run → Attempt → root Execution without Nexus (UEA-INV-008)."* |
| **README-eligible** | No |

---

### UEA-DIAG-C — Orchestration / Nexus Flow

| Field | Specification |
|-------|---------------|
| **Audience** | Orchestration and Nexus owners, implementers |
| **Purpose** | Show how an orchestration-strategy Execution is realized via Nexus and child Executions |
| **Required concepts** | Parent Execution (`strategy = orchestration`); `OrchestrationDefinition` input; Nexus (readiness, scheduling, fan-out/merge); child Execution instantiation; result flow/merge back to parent Execution |
| **Mandatory relationships/arrows** | OrchestrationDefinition → Nexus (configuration); parent Execution → Nexus; Nexus → child Executions (one or many); child results → merge → parent completion; boundary label: Nexus stops at Execution boundary — does not enter AgentEngine |
| **Ambiguity to eliminate** | Nexus directly runs agent internals; Node graph equals Execution Tree |
| **Level of detail** | Show 2–3 child Executions; one child may be agentic (icon only below boundary) |
| **Target embedding** | UEA §6–§7; future link from NEXUS_EXECUTION_FLOW |
| **Asset basename** | `unified_execution_orchestration_nexus_flow` |
| **Caption guidance** | *"Orchestration strategy: Nexus schedules child Executions from a validated OrchestrationDefinition; it does not execute Agent internals directly (UEA-INV-007)."* |
| **README-eligible** | No |

---

### UEA-DIAG-D — Orchestration Topology vs Runtime Execution Tree

| Field | Specification |
|-------|---------------|
| **Audience** | Orchestration architects, anyone modeling fan-out |
| **Purpose** | Side-by-side comparison of definition topology and runtime instances |
| **Required concepts** | **Left panel:** OrchestrationDefinition, Nodes (`NodeId`), edges; **Right panel:** Execution Tree (`ExecutionId`, `parent_execution_id`); fan-out example (one Node → many Executions) |
| **Mandatory relationships/arrows** | Left: static graph with NodeIds; Right: runtime tree with ExecutionIds; mapping annotation (dashed) from Node to **set** of Executions, not 1:1 identity equality |
| **Ambiguity to eliminate** | Node == Execution; single tree serves both definition and runtime |
| **Level of detail** | One fan-out node (e.g. `analyze_customer` → E101, E102, E103 per UEA §5) |
| **Target embedding** | UEA §5 |
| **Asset basename** | `unified_execution_topology_vs_execution_tree` |
| **Caption guidance** | *"Definition topology (NodeId) vs runtime Execution Tree (ExecutionId): same Node may materialize many Executions (UEA-INV-004, UEA-INV-006)."* |
| **README-eligible** | No |

---

### UEA-DIAG-E — Identity & Lifecycle

| Field | Specification |
|-------|---------------|
| **Audience** | All implementers touching identity, Observability, UER |
| **Purpose** | ID minting hierarchy and Attempt boundaries |
| **Required concepts** | TaskId → RunId → AttemptId → ExecutionId → EventId; root Execution; `parent_execution_id` on child Executions; Attempt boundary marker |
| **Mandatory relationships/arrows** | Vertical spine for one root Execution; branch for child Executions under same Attempt; labels for what mints what (Run retry → new AttemptId + new Execution instances) |
| **Ambiguity to eliminate** | Local retry mints new AttemptId; EventId interchangeable with ExecutionId |
| **Level of detail** | Identity types only — no retry table (see F) |
| **Target embedding** | UEA §3 |
| **Asset basename** | `unified_execution_identity_lifecycle` |
| **Caption guidance** | *"Canonical identity hierarchy: five ID layers; every Attempt has at least one root Execution; children link via parent_execution_id (UEA-INV-001, UEA-INV-002)."* |
| **README-eligible** | No (simplified IDs may appear in §6 derivative) |

---

### UEA-DIAG-F — Retry / Pause / Resume / Cancellation

| Field | Specification |
|-------|---------------|
| **Audience** | Reliability, UER, HITL implementers |
| **Purpose** | Operational semantics across retry, pause, resume, and cancel dimensions |
| **Required concepts** | Provider/tool retry; execution-level retry; whole-Run retry (new Attempt); pause/resume (HITL); cancel Run; cancel Execution subtree; identity columns or swimlanes for TaskId/RunId/AttemptId/ExecutionId |
| **Mandatory relationships/arrows** | Table or swimlane layout from UEA §10; cancel Run → entire tree; cancel Execution E → E + descendants; pause/resume loop on **same** ids |
| **Ambiguity to eliminate** | Pause/resume is retry; local retry mints new Attempt |
| **Level of detail** | Scenario rows, not policy engine internals |
| **Target embedding** | UEA §10, §14 |
| **Asset basename** | `unified_execution_retry_pause_resume_cancel` |
| **Caption guidance** | *"Recovery semantics: local retries preserve AttemptId; whole-Run retry mints new Attempt and new Execution instances; pause/resume preserves all runtime ids (UEA-INV-012–014)."* |
| **README-eligible** | No |

---

### UEA-DIAG-G — Nested Orchestration

| Field | Specification |
|-------|---------------|
| **Audience** | Orchestration/Nexus architects |
| **Purpose** | Legal hierarchical orchestration within one Run/Attempt |
| **Required concepts** | Parent Execution (orchestration) → Nexus → child Execution that **also** uses orchestration strategy → nested Nexus → deeper child Executions; same RunId/AttemptId throughout |
| **Mandatory relationships/arrows** | Multi-level Execution Tree; nested Nexus blocks; explicit label: **no new RunId** for nested orchestration (**UEA-INV-019**) |
| **Ambiguity to eliminate** | Architecture is permanently flat; nested orchestration requires new RunId / OrchestrationRunId |
| **Level of detail** | Two levels of orchestration sufficient |
| **Target embedding** | UEA §6; scenario E |
| **Asset basename** | `unified_execution_nested_orchestration` |
| **Caption guidance** | *"Nested orchestration: child Executions may themselves use orchestration strategy under the same Run and Attempt (UEA-INV-019)."* |
| **README-eligible** | No |

---

### UEA-DIAG-H — Distributed Queue / Worker Execution

| Field | Specification |
|-------|---------------|
| **Audience** | Background tasks, agent distribution, platform ops |
| **Purpose** | Transport vs runtime identity across queue/worker boundary |
| **Required concepts** | Queue/broker; transport ids (`message_id`, `delivery_id`, `lease_id`, `worker_id`); dispatch; worker; preserved TaskId/RunId/AttemptId/ExecutionId; causal admission before meaningful work (**UEA-INV-017**) |
| **Mandatory relationships/arrows** | Transport envelope → worker receives runtime ids (not mints Run); durable transport→Execution causal evidence → admission → execution; redelivery reuses same runtime ids (**UEA-INV-011**) |
| **Ambiguity to eliminate** | Worker/transport mints a new Run; broker task id equals RunId |
| **Level of detail** | One worker crash/redelivery annotation |
| **Target embedding** | UEA §11, §19 |
| **Asset basename** | `unified_execution_distributed_queue_worker` |
| **Caption guidance** | *"Distributed execution: transport identity is infrastructure-only; workers receive and continue the same runtime identity (UEA-INV-011, UEA-INV-017)."* |
| **README-eligible** | No |

---

### UEA-DIAG-I — Governance + Authority + Budget Inheritance

| Field | Specification |
|-------|---------------|
| **Audience** | Governance, budget, execution coordination implementers |
| **Purpose** | Narrow-only authority and hierarchical budget allowances |
| **Required concepts** | Run authority/budget; inheritance to Execution; narrowing to child Execution; Agent; Tool; budget subsystem as ledger owner |
| **Mandatory relationships/arrows** | Run → Execution → child Execution → Agent → Tool; authority arrows labeled **≤ parent**; budget reservation/consumption owned by budget subsystem; Nexus and Execution marked as **non-ledger** |
| **Ambiguity to eliminate** | Child may exceed parent authority/budget; Nexus owns budget ledger |
| **Level of detail** | Tree with narrow-only annotations |
| **Target embedding** | UEA §12–§13 |
| **Asset basename** | `unified_execution_governance_budget_inheritance` |
| **Caption guidance** | *"Authority and budget flow: children may narrow but never expand parent envelope; budget ledger owned by budget subsystem, not Nexus or Execution (UEA-INV-009, UEA-INV-010)."* |
| **README-eligible** | No |

---

### UEA-DIAG-J — Observability + DIAG Causal Flow

| Field | Specification |
|-------|---------------|
| **Audience** | Observability, DIAG, audit engineers |
| **Purpose** | Division of labor: facts, evidence, interpretation |
| **Required concepts** | Execution produces lifecycle facts; Observability records canonical evidence; DIAG interprets evidence; causal chain Event → Execution → parent Execution(s) → Attempt → Run → Task |
| **Mandatory relationships/arrows** | Solid: Execution → RuntimeEvents → Observability store; dashed: Observability → DIAG (read-only interpret); **no** arrow from DIAG to mint ExecutionId |
| **Ambiguity to eliminate** | DIAG reconstructs truth heuristically from logs only; Observability invents execution identity |
| **Level of detail** | One branch of Execution Tree in causal walk-back |
| **Target embedding** | UEA §17–§18 |
| **Asset basename** | `unified_execution_observability_diag_causal_flow` |
| **Caption guidance** | *"Observability records execution truth; DIAG interprets canonical evidence along Event → Execution → … → Task without minting identity (UEA-INV-015, UEA-INV-016)."* |
| **README-eligible** | No |

---

### UEA-DIAG-K — Checkpoint / Recovery

| Field | Specification |
|-------|---------------|
| **Audience** | Long-running execution, reliability engineers |
| **Purpose** | Run-scoped durable state and resume path |
| **Required concepts** | Run-scoped checkpoint; Attempt; Execution Tree state; per-Execution state; orchestration state; agent/step cursors; pending HITL; budget reservations (where required); resume path |
| **Mandatory relationships/arrows** | Checkpoint blob linked to RunId/AttemptId; contains tree snapshot; resume restores into **same** Attempt/Execution ids; checkpoint labeled **not** identity mint source |
| **Ambiguity to eliminate** | Checkpoint is separate source of identity truth; resume creates new Attempt by default |
| **Level of detail** | State categories as labeled compartments, not storage schema |
| **Target embedding** | UEA §16 |
| **Asset basename** | `unified_execution_checkpoint_recovery` |
| **Caption guidance** | *"Run-scoped checkpoint captures Execution Tree and pending work for resume; checkpoint does not mint ExecutionId (UEA-INV-002, migration gap aware)."* |
| **README-eligible** | No |

---

### UEA-DIAG-L — Component Ownership / Dependency View

| Field | Specification |
|-------|---------------|
| **Audience** | Architects partitioning work across domains |
| **Purpose** | Who owns what; execution boundary as coordinator |
| **Required concepts** | Ownership matrix/boundaries: UEA meta-architecture, UER, Orchestration, Nexus, Agent subsystem, Governance, Budget, Observability, DIAG, Checkpoint/distributed; execution coordination boundary |
| **Mandatory relationships/arrows** | Matrix or layered diagram from UEA §22; Execution boundary box **coordinates** arrows to subsystems; no absorption edges into Execution for Governance/OBS/DIAG/Budget |
| **Ambiguity to eliminate** | Execution boundary absorbs subsystem ownership; duplicate owners for same semantic |
| **Level of detail** | Owner names + link to domain docs; no API listings |
| **Target embedding** | UEA §20, §22 |
| **Asset basename** | `unified_execution_component_ownership` |
| **Caption guidance** | *"Component ownership: execution coordinates contracts across domain owners; it does not absorb Governance, Budget, Observability, DIAG, Queue, Checkpoint, or Agent internals (UEA-INV-018)."* |
| **README-eligible** | No |

---

## 5. Scenario mapping

Compact index: which diagram(s) best explain each reference scenario from UEA §28.

| Scenario | Primary diagram(s) | Secondary diagram(s) |
|----------|---------------------|----------------------|
| **A. Simple inference** | B | A, E |
| **B. Autonomous agent with tools** | B, A (agentic branch) | E, I |
| **C. A → B → C orchestration** | C | A, D, E |
| **D. Parallel fan-out** | D, C | A, E |
| **E. Nested orchestration** | G | C, A, E |
| **F. Local execution retry** | F | E |
| **G. Whole Run retry** | F, E | A |
| **H. HITL pause/resume** | F | E, I, K |
| **I. Remote worker crash/redelivery** | H | F, E |
| **J. Post-failure DIAG investigation** | J | E, A |

---

## 6. README promotion candidate

**Exactly one** future README-grade execution-core graphic is defined for a later promotion slice (not embedded in README in UE-DOC-0.3A).

| Field | Specification |
|-------|---------------|
| **Derivative of** | UEA-DIAG-A (`unified_execution_full_architecture`), simplified |
| **Working title** | *Unified Execution — platform core* |
| **Future asset basename** | `unified_execution_platform_core` |
| **Must show (simplified)** | Task → Run → Attempt → Execution; three strategy branches (inference, agentic, orchestration) with Nexus only on orchestration branch; AgentEngine/UAEP under agentic; child Executions under orchestration; muted cross-cutting bands for governance, budget, observability |
| **Must omit / simplify** | DIAG internals, checkpoint detail, queue wire protocol, per-domain API names, migration gaps |
| **Semantic truth** | All simplifications must remain consistent with UEA-INV-*; no implication that Nexus always runs or that Agent equals Execution |
| **Visual alignment** | Match existing platform/LKW README `<picture>` pattern: light/dark PNG pair, optional SVG, link to fullsize markdown companion |
| **Accompanying blurb (draft)** | *"Intergrax executes work through a governed identity spine (Task → Run → Attempt → Execution). Simple calls, agentic sessions, and multi-step orchestration share one model; Nexus appears only when an Execution's strategy requires coordinating child Executions."* |
| **Normative link** | [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) |
| **Embedding target (future)** | Root [`README.md`](../../../README.md) — execution/platform section only, after asset production slice |

---

## 7. Production workflow (out of band)

1. **Graphic producer** reads this specification + [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md); does not change semantics.
2. **Assets** are generated per §3 naming; placed under `docs/assets/architecture/unified_execution/`.
3. **Integration session** embeds assets per §3.3 into UEA and (optionally) domain docs via link-only updates.
4. **README promotion** uses §6 candidate in a dedicated slice after architect review.

**This slice (UE-DOC-0.3A) stops at step 0 — specification and documentation wiring only.**

---

## 8. Related documents

| Document | Relationship |
|----------|--------------|
| [`UNIFIED_EXECUTION_ARCHITECTURE.md`](UNIFIED_EXECUTION_ARCHITECTURE.md) | Normative meta-architecture (parent) |
| [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) | Hub index; registers this file as supporting artifact |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | UER domain owner — detailed contracts |
| UEA §27 / §28 | View checklist and reference scenarios (semantic source for §4–§5) |
