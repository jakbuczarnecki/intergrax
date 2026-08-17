# Observability — Implementation Plan

**Architecture (1:1):** [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Cross-plan — Agent layer (ACP):** Dual observability planes (architecture §31) — `AgentRunTrace` on `AgentRunResult` (Plane B) and `ApplicationRunSummary` on Task completion (Plane A). Delivered in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 3** (`ACP-OBS-1`, `ACP-OBS-2`) and **Wave 7** redaction (`ACP-PROD-8`). Trace spine changes MUST keep step records compatible with `AgentStepRecord` tool/RAG/LLM fields.

**Cross-plan — Event catalog (OBS-EVOL-9 · P1-ARCH-02):** Layered spine + `event_kind` (architecture §4.4 · ADR-OBS-003). Developers extend via `emit_domain_signal`, not new `RuntimeEventType`. Pre-release spine consolidation before publication.

**Cross-plan — Application Hosting (APP-HOST-3B):** Application Hosting publishes typed `HostedApplicationEvent` signals through the **platform observability signal** path on the existing HOS spine/export infrastructure (architecture § Execution-scoped vs non-execution observability signals). Hosting lifecycle signals **MUST NOT** synthesize `TaskId`/`RunId`/`AttemptId` or mint per-event `AttemptId`. Corrective implementation: **TRACE-1B-HOS-FIX**. No second event bus.

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). OBSERVABILITY owns token savings attribution, optimization receipts visibility, typed diagnostic payloads, metrics, and regression-gate reporting through the Harness Observability Spine. Token Optimization telemetry must be observable through the same observability spine — do not create a private telemetry bus for token optimization. **TOKEN-6A-lite** is an early telemetry-shape slice for savings attribution through the existing observability spine; it must not create a private telemetry bus. **OBS-HEALTH-lite** is a minimal operator-visible status slice for exporter/token telemetry health, not full observability production hardening. Full **OBS-VENDOR** production hardening remains **Planned**. **LKW-PF6** ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF) is the platform proof workload for token savings telemetry and regression gates.

**Last updated:** 2026-08-17 — **TRACE-ASOF-2** logical execution projection **Planned / In Review** · **TRACE-BITEMP-1** typed contracts **Done / Closed** (`d68c72177403fb634fd4ede2d0252e9814d7adee`) · **TRACE-ASOF-1** execution position + `AsOfBoundary` **Done / Closed** (`02462d96897daa4ea19d96dce776768a03cbbf53`) · **TRACE-1C** Done / Closed (`42f196715bbf19aa69667f4a73c6002de61b8dff`) · **TRACE-BITEMP-ARCH-SYNC-R6** unresolved position resolution ownership + lease/fencing + auditable terminalization · **TRACE-BITEMP-ARCH-SYNC-R5** watermark finality + gap semantics + idempotent acceptance requirements · **TRACE-BITEMP-ARCH-SYNC-R4** domain-owned revision ordering authority + pluggable provider contract · **TRACE-BITEMP-ARCH-SYNC-R3** revision watermark semantics + serialization decision boundary · **TRACE-1B** / **TRACE-1B-HOS-FIX** Done / Closed (`dd4cd73598652119010502bfd0f4ff06d4db0ed8`) · **TRACE-1B-ARCH-HOS** Done (`1a3dfa9244a052bd13417562dccc104048d0a492`) · **TRACE-BITEMP-ARCH-SYNC-R2** deterministic correction ordering · pre-production clean-cut policy.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (OBSERVABILITY plan).

- **Implement / audit default:** Hub Phase TRACE (§5–§10 arch) · [`plan/satellites`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) (active OECP register), [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) (closed phases). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + rows `TOKEN-OBS-1` / `TOKEN-OBS-2`; use HOS/domain-signal model, do not create private telemetry channel.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/OBSERVABILITY.md`](../../technical/guides/audit_slices/OBSERVABILITY.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Architecture documentation (P2)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-07** | Clarify observability event spine and event ownership | **Done** (2026-06-20) |

Architecture: [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md#observability-event-spine).

---

## Phase TRACE — Execution identity, unified journal strictness, as-of projections, and bitemporal state (Planned)

**Architecture:** [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §5–§10 · [`UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) §42.1.8  
**Design:** **TRACE-1** identity + journal + as-of + bitemporal target canon — **Done** (TRACE-ARCH-SYNC-1, TRACE-BITEMP-ARCH-SYNC, TRACE-BITEMP-ARCH-SYNC-R2, TRACE-BITEMP-ARCH-SYNC-R3, TRACE-BITEMP-ARCH-SYNC-R4, TRACE-BITEMP-ARCH-SYNC-R5, TRACE-BITEMP-ARCH-SYNC-R6, 2026-08-17)  
**Implementation:** **TRACE-1A / TRACE-1B / TRACE-1C** identity + journal strictness canon is **Done / Closed**. **TRACE-ASOF-1** execution position + `AsOfBoundary` is **Done / Closed** (`02462d96897daa4ea19d96dce776768a03cbbf53`). **TRACE-BITEMP-1** typed contracts are **Planned / In Review** (independent audit closes it). Production bitemporal persistence is **not implemented**.

**Pre-production clean-cut (canon):** Intergrax has no active production users. TRACE implementation **removes** unused legacy paths rather than preserving them. No permanent compatibility aliases, dual canonical schemas, or migrations for unused persisted formats (including old checkpoint shapes). Migrate live code to the canonical contract, then delete the old path. See architecture [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §5.7.

**Delivery rule:** one TRACE row per PR; no runtime work in TRACE-ARCH-SYNC-1 doc-only slice.

### TRACE-1 — Identity contracts and journal strictness

| ID | Priority | Status | Goal | Dependency | Acceptance (summary) |
|----|----------|--------|------|------------|----------------------|
| **TRACE-1A** | P1 | Done / Closed (`3eee3a860cd852e833fa3f2dd3d190ce9d96de08`) | Typed `TaskId`/`RunId`/`AttemptId` via `typing.NewType(..., str)`; strict validation; correct mint ownership; `RuntimeRequest.task_id`/`run_id`; remove canonical identity from metadata | TRACE-1 design complete | No `metadata["run_id"]`; no `task_id or run_id` fallbacks; wire strings only on boundary |
| **TRACE-1B** | P1 | Done / Closed (`dd4cd73598652119010502bfd0f4ff06d4db0ed8`) | `AttemptId` through `RuntimeExecutionContext`, `EmitContext`, `RuntimeEvent`; attempt-aware emit paths; `TASK_CREATED` as first journal event in A1 | TRACE-1A | Every canonical `RuntimeEvent` carries `TaskId`, `RunId`, `AttemptId`, `EventId` |
| **TRACE-1B-ARCH-HOS** | P1 | **Done** (`1a3dfa9244a052bd13417562dccc104048d0a492`) | Freeze execution-scoped `RuntimeEvent` vs non-execution platform observability signals; resolve hosting target; one HOS spine, no second bus | TRACE-1A | Architecture canon in [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) § Execution-scoped vs non-execution observability signals |
| **TRACE-1B-HOS-FIX** | P1 | Done / Closed (`dd4cd73598652119010502bfd0f4ff06d4db0ed8`) | Remove synthetic execution identity from hosting platform signals; route `HostedApplicationEvent` through canonical platform observability path on existing HOS spine/export — no `mint_attempt_id()` per hosting event, no synthetic `TaskId`/`RunId` | TRACE-1B-ARCH-HOS; TRACE-1B | `ObservabilityHostedApplicationEventPublisher` replaces runtime spine adapter; `ExportRecordKind.PLATFORM_SIGNAL`; no second bus; no compatibility alias or historical migration |
| **TRACE-1C** | P1 | Done / Closed (`42f196715bbf19aa69667f4a73c6002de61b8dff`) | **Strict journal + legacy removal:** zero identity fallback in journal build; migrate live code to canonical identity; delete unused legacy aliases/fallbacks/adapters; strict journal reconstruction; no `Any`/dynamic adapters on identity boundary; **no** permanent compatibility layer | TRACE-1B; TRACE-1B-HOS-FIX | `build_unified_run_journal` enforces strict contract; legacy aliased identity **removed** from canonical runtime (not merely recognized or flagged) |

**TRACE-1C evidence chain** (independently audited; final acceptance `42f196715bbf19aa69667f4a73c6002de61b8dff`): `94ad240e4a162ede5f6e1a03c55d1b412fffaee8` → `7d5afa7da0aca0c9b195e63613041e91fc5449e6` → `42f196715bbf19aa69667f4a73c6002de61b8dff`.

**TRACE-1B evidence chain** (independently audited; final acceptance `dd4cd73598652119010502bfd0f4ff06d4db0ed8` because TRACE-1B could not close before HOS semantics were corrected): `302669b7c1006be49e1a23d9087de148b9ccc6d1` → `7542adf384a69713fea7faf644a711ed42563ca7` → `1a3dfa9244a052bd13417562dccc104048d0a492` → `dd4cd73598652119010502bfd0f4ff06d4db0ed8`.

### TRACE-ASOF — First-class as-of projections

| ID | Priority | Status | Goal | Dependency | Acceptance (summary) |
|----|----------|--------|------|------------|----------------------|
| **TRACE-ASOF-1** | P1 | Done / Closed (`02462d96897daa4ea19d96dce776768a03cbbf53`) | Resolve deterministic historical boundary: run-scoped `ExecutionEventPosition` at persistence acceptance; `PositionedRuntimeEvent` wrapper; typed inclusive `AsOfBoundary`; positioned read prefix; no timestamp-only ambiguity | TRACE-1C | `append` returns canonical position; idempotent `EventId` reuse; journal/list order follows execution position |
| **TRACE-ASOF-2** | P1 | Planned | Typed logical projections from Unified Run Journal; deterministic state-as-of reconstruction; attempt-aware; source `RuntimeEvent` references; compatible with bitemporal temporal basis (does not freeze valid/system-time semantics) | TRACE-ASOF-1; TRACE-BITEMP-1 | Logical projection rebuildable from journal; no new source of truth |
| **TRACE-ASOF-3** | P2 | Planned (**conditional**) | Only if projections are materialized: immutable projection revisions; explicit `revision_id`; explicit `supersedes`; rebuildability | TRACE-ASOF-2 | Skippable if materialization not needed after TRACE-ASOF-2 |
| **TRACE-ASOF-4** | P1 | Planned | Typed public/internal query contract for as-of; provenance to source events; no dynamic projection registry; operator/debug/audit consumption | TRACE-ASOF-2; TRACE-ASOF-3 only if materialization shipped | Query surface does not treat projection as proof/evidence |

**TRACE-ASOF-1 evidence chain** (independently audited; final acceptance `02462d96897daa4ea19d96dce776768a03cbbf53`): `ae618fc81817497dbbcf018d92c95856f2d44115` → `d88253dbcfaa470597f93d91eec6a80a30e77007` → `98a2d186d9b512048c01024b67f1e707d72240ee` → `a7a931c6a5c4356e9bd49d7d9f8b5787e9a826b6` → `02462d96897daa4ea19d96dce776768a03cbbf53`.

**Non-goals (canon):** new event store; mandatory materialization. Full bitemporal valid-time / system-time architecture is an **accepted target capability** — see TRACE-BITEMP-* (not a non-goal).

### TRACE-BITEMP — First-class bitemporal historical state

**Architecture:** [`OBSERVABILITY.md`](../../architecture/OBSERVABILITY.md) §8  
**Design:** **Done** (TRACE-BITEMP-ARCH-SYNC, TRACE-BITEMP-ARCH-SYNC-R2, TRACE-BITEMP-ARCH-SYNC-R3, TRACE-BITEMP-ARCH-SYNC-R4, TRACE-BITEMP-ARCH-SYNC-R5, TRACE-BITEMP-ARCH-SYNC-R6, 2026-08-17) · **TRACE-BITEMP-1** typed contracts **Planned / In Review** · production persistence **Planned** (TRACE-BITEMP-2).

**Implementation sequence (semantic):**

```text
TRACE-BITEMP-1
    semantic contract
    scope decision
    canonical strategy decision
    provider contract design

        ↓

TRACE-BITEMP-2
    typed provider interface
    canonical implementation
    persistence
    DI/host selection

        ↓

TRACE-BITEMP-3
    provider-independent reconstruction

        ↓

TRACE-BITEMP-4
    provider-independent query/audit surface

        ↓

TRACE-BITEMP-5
    canonical proof
    reusable provider qualification/conformance proof
```

**Dependency order (semantic):**

```text
TRACE-1A → TRACE-1B → TRACE-1B-HOS-FIX → TRACE-1C → TRACE-ASOF-1 → TRACE-BITEMP-1 → TRACE-ASOF-2
  → TRACE-BITEMP-2 → TRACE-BITEMP-3 → TRACE-ASOF-3 (conditional) → TRACE-ASOF-4
  → TRACE-BITEMP-4 → TRACE-BITEMP-5
```

(Architecture-only prerequisite **TRACE-1B-ARCH-HOS** is Done; it does not block runtime sequencing beyond informing **TRACE-1B-HOS-FIX**.)

| ID | Priority | Status | Goal | Dependency | Acceptance (summary) |
|----|----------|--------|------|------------|----------------------|
| **TRACE-BITEMP-1** | P1 | Planned / In Review | Canonical temporal semantics and contracts **only** (no production storage): frozen types in `intergrax.contracts.bitemporal_knowledge` — `ValidTimeBasis`, `SystemTimeBasis`, `BitemporalKnowledgeBasis`, `KnowledgeRevisionId`, `RevisionAcceptanceKey`, `KnowledgeRevisionPosition`, `KnowledgeRevisionPositionLifecycle`, `KnowledgeRevisionWatermark`, `KnowledgeOrderingScope` (TENANT), `RevisionOrderingAuthority`. Canonical strategy: tenant-scoped transactional allocation + acceptance. Failure matrix A–J frozen. Architecture §8.11. **MUST NOT** implement production persistence | TRACE-1C; TRACE-ASOF-1 | Identity and execution boundary stable; correction ordering is not timestamp-only; `supersedes` does not define total ordering; position lifecycle and finalized watermark contract frozen; allocated ≠ accepted; `KnowledgeRevisionId` bound to idempotent acceptance; same key + same revision → same K; same key + different revision → conflict; failure/crash scenarios A–J specified; TENANT scope + transactional canonical strategy selected with documented rationale; extension contract remains capable of qualified alternatives; no semantic weakening by provider selection; no storage implementation |
| **TRACE-BITEMP-2** | P1 | Planned | Implement: (1) typed `RevisionOrderingAuthority` contract, (2) canonical first-party provider selected by TRACE-BITEMP-1, (3) host/deployment DI/composition path, (4) persistence integration, (5) revision-position / watermark query support, (6) **typed position lifecycle**, (7) **stable acceptance/dedup identity** — retry same acceptance returns same semantic position/result, (8) idempotent acceptance, (9) atomic failure behavior, (10) **canonical watermark advancement** per finalized-contiguous semantics, (11) **crash recovery for unresolved positions**, (12) **authoritative resolution transition API/path** (`UNRESOLVED → ACCEPTED` or `TERMINAL_NON_COMMITTED`), (13) **lease/ownership state** where required, (14) **fencing/generation semantics**, (15) **stale-writer rejection**, (16) **bounded unresolved scanner/recovery mechanism**, (17) **safe reconciliation after crash/restart**, (18) **immutable resolution record persistence** (reason/source/provenance), (19) **race-safe finalization**, (20) **watermark advancement after terminalization**, (21) **idempotent recovery**, (22) **manual/governance fallback through same authority**, (23) **no new K for lifecycle voiding**. Architecture §8 R6. Bitemporal revision/fact persistence **implementing the TRACE-BITEMP-1 contract**: persist accepted vs terminal-non-committed vs unresolved status as required; prevent unresolved gap from being hidden below watermark; revision acceptance and authoritative position persisted consistently; no accepted revision without authoritative ordering identity; no ambiguous partial acceptance; no half-accepted revision; idempotent retries; deterministic concurrent correction acceptance; repository can query by revision position / watermark; correction ordering reconstructable without timestamp heuristics; preserves prior system-time belief; additive correction/backdating/future effectiveness and causal lineage; no destructive overwrite; explicit provenance linkage. Canonical provider **MUST** work out of the box. Contract **MUST NOT** hardcode the canonical implementation. **MUST NOT** invent semantics — implements TRACE-BITEMP-1 + R6 canon only. Physical representation of terminal gaps not required to be uniform across providers. Third-party qualified providers are **not** required here. **No** database vendor selection | TRACE-BITEMP-1 | Corrections additive; prior belief preserved; ordering auditable from persistence; position lifecycle persisted per contract; watermark advances per finalized-contiguous rules; no half-accepted revision; retries idempotent with stable dedup identity; concurrent acceptance deterministic; canonical default provider operational via host DI; contract swappable without semantic change; **R6:** old writer cannot commit after newer resolution authority/fence; no unresolved position silently disappears; resolution decision audit-queryable; terminalized K cannot later become accepted; watermark advances only after legitimate finalization; no provider-specific semantics leak through public contract |
| **TRACE-BITEMP-3** | P1 | Planned | Historical execution reconstruction **depending only on the canonical `RevisionOrderingAuthority` contract**, not concrete provider implementation. Reconstruction semantics **MUST** be provider-independent — E + K + valid/system-time basis produces equivalent semantics for every qualified provider. **MUST** consume only finalized watermark semantics; **MUST NOT** treat highest-allocated as stable knowledge boundary; deterministically exclude unresolved positions beyond watermark; correctly interpret terminal non-committed positions. **R6:** treat `TERMINAL_NON_COMMITTED` **K** as finalized but containing **no** accepted knowledge revision; **never** synthesize a knowledge fact from a resolution record; exclude stale late writes rejected after terminalization; remain deterministic regardless of which provider performed recovery; preserve historical belief and system-time semantics independently from lifecycle resolution metadata. Knowledge state at revision watermark K; wall-clock system-time basis resolved to finalized K; execution state at E; combined E + K + valid/system-time basis where needed. Keep all semantics distinct. Do **not** call E + K + temporal basis “bitemporal state”. Distinguish execution-boundary state, valid-time state, system-time historical belief, knowledge/revision-position/watermark state, and combined historically reproducible execution state; execution ordering and knowledge/revision ordering remain independent | TRACE-ASOF-2; TRACE-BITEMP-2 | Reconstruction modes remain distinct; bitemporal state remains valid-time + system-time only; watermark is not a temporal axis; finalized watermark semantics enforced; provider-independent reconstruction proven by contract abstraction; **R6:** terminal non-committed positions produce no accepted knowledge fact; resolution records never reconstructed as knowledge revisions; late stale writes excluded |
| **TRACE-BITEMP-4** | P1 | Planned | Temporal query/audit surface consuming **domain abstractions only** — including: What is the highest **finalized** knowledge watermark? Is position K accepted, terminal non-committed, or unresolved? Why can the watermark not advance beyond K? Which positions below/at the watermark are accepted revisions? Which positions represent terminal non-committed outcomes? What knowledge was safely visible at watermark K? What was the authoritative knowledge watermark at system time S? What revisions were accepted ≤ K? What was known at K? What did execution E operate against using knowledge watermark K? What do we now know was valid for the time associated with E? What was believed to be valid at the time E executed? **R6 resolution/lease/fencing audit:** Which K positions are currently `UNRESOLVED`? Why is K unresolved? Which authority currently owns resolution? Is the current lease/ownership stale? What blocks the watermark? Who/what finalized K? Why was K terminalized? What resolution reason was recorded? What fencing/generation authority authorized finalization? When did resolution occur? Which stale/late commits were rejected after finalization? Which terminalized gap allowed watermark to advance? Plus valid-time as-of; system-time as-of; combined bitemporal basis; correction-order queries (before/after revision position K; authoritative K1 → K2 → K3 order); correction history; provenance to source revisions/events. Wall-clock audit questions must **not** rely on timestamp sorting alone. Canonical API **MUST NOT** expose provider-specific gap semantics as the audit model. Provider diagnostics **MAY** separately expose implementation details (transaction ID, DB row/version, 2PC coordinator details, internal lease ID, vendor tokens) as diagnostic provenance only — not canonical semantics. Audit surfaces **MUST NOT** expose implementation-specific concepts (database sequence IDs, Kafka offsets, Redis counters, vendor-specific transaction identifiers) as canonical audit meaning — only **KnowledgeRevisionPosition** / **KnowledgeRevisionWatermark** (or TRACE-BITEMP-1 frozen names) | TRACE-BITEMP-3 | Typed query API; deterministic correction-order and finalized watermark-resolution answers; position finality queries supported; no vendor lock-in; canonical audit semantics provider-independent; **R6:** resolution/lease/fencing audit queries supported at domain level |
| **TRACE-BITEMP-5** | P1 | Planned | Production proof/invariants + **provider qualification proof matrix**: (A) canonical provider passes complete proof suite; (B) provider conformance suite reusable for alternatives; (C) alternate provider cannot be marked production-qualified without passing equivalent semantic proof; (D) host selection cannot activate an unqualified provider in production-grade profile without explicit fail-closed behavior defined by architecture. Proof areas: **concurrent acceptance**; **clock skew**; **idempotent retry**; **allocation/persistence failure**; **watermark resolution**; **scoped ordering**; **cross-scope behavior**; **repeated historical query**; **deterministic repeated reconstruction**; **causal lineage independence**; **historical immutability**; **audit reconstruction**; **R5 proof scenarios**: (1) transaction rollback gap — K2 allocated then rolled back, K3/K4 succeed, watermark can advance across finalized K2; (2) sequencer orphan — K issued but acceptance never commits; (3) crash between allocation and acceptance; (4) crash after durable acceptance but before caller response; (5) retry after successful-but-unacknowledged acceptance returns same semantic K; (6) duplicate retry does not create second accepted revision; (7) permanent terminal gap does not freeze watermark forever; (8) unresolved gap blocks watermark advancement; (9) highest-allocated ≠ watermark proof; (10) provider swap equivalence — two qualified providers produce equivalent canonical watermark/reconstruction semantics; (11) repeated reconstruction at same finalized watermark is deterministic; (12) wall-clock T → K resolution never returns boundary containing unresolved gaps; **R6 proof scenarios**: (1) writer crashes while K is UNRESOLVED; (2) unresolved K pins watermark; (3) recovery worker detects stale K; (4) lease expires but K is NOT blindly voided; (5) recovery obtains newer fencing authority; (6) original writer returns using stale fence and is rejected; (7) recovery proves no durable acceptance and terminalizes K; (8) immutable resolution record exists; (9) watermark advances across terminalized K; (10) writer commits before recovery obtains newer authority → ACCEPTED wins; (11) recovery wins race first → late writer rejected; (12) ambiguous state remains UNRESOLVED and watermark stays pinned; (13) restart recovers outstanding unresolved positions; (14) hung/in-doubt 2PC operation remains unresolved until safely classified; (15) governance/manual resolution uses same authority/fencing/audit rules; (16) TERMINAL_NON_COMMITTED cannot transition back to ACCEPTED; (17) later legitimate correction uses new acceptance key + new K; (18) provider conformance — two qualified providers produce equivalent lifecycle/resolution/watermark semantics under same crash/race history; (19) resolution record is not reconstructed as a knowledge revision; (20) repeated recovery is idempotent; plus existing: concurrent corrections on same fact; delayed correction; multiple correction chain P1 → P2 → P3; equal/ambiguous timestamps; execution independence; backdated correction; future-effective change; historical-belief reconstruction; current-corrected-history reconstruction; no history destruction. Do **not** implement qualification engine here unless separately scoped | TRACE-BITEMP-4 | All required platform proof scenarios pass including R5 + R6 scenarios; canonical and alternative providers measured against same semantic suite; unqualified provider activation fail-closed in production-grade profile |

**TRACE-BITEMP-1 decision ownership:** contract/design only. Frozen in architecture §8.11 pending independent audit (do **not** mark Done here). TRACE-BITEMP-2 **MUST NOT** invent architecture or re-select the canonical strategy / ordering scope. TRACE-BITEMP-2 implements persistence only.

### Downstream trace roadmap (reference)

Detailed delivery for later trace phases may live in other canonical plans. OBSERVABILITY retains cross-layer observability ownership; add dependency links only — do not duplicate full phase specs here.

| ID | Status | Depends on (observability) | Notes |
|----|--------|----------------------------|-------|
| **TRACE-2** | Planned | TRACE-1C | Downstream — export/sink alignment with strict identity |
| **TRACE-3** | Planned | TRACE-1C | Downstream — operator/debug surfaces |
| **TRACE-4** | Planned | TRACE-ASOF-2, TRACE-BITEMP-3 | Downstream — eval/OECP consumption of as-of and bitemporal reconstruction |
| **TRACE-5** | Planned | TRACE-1C, TRACE-ASOF-2, TRACE-BITEMP-3 | Downstream — cross-tier proof/evidence linkage |
| **TRACE-DOC** | Planned | TRACE-1, TRACE-ASOF-*, TRACE-BITEMP-* | Downstream — operator docs and audit slices |

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) | **OECP** — eval control plane implementation register (active) |
| [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) | audit history (closed phases) |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-OBS — Token optimization telemetry and regression gates (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P1 after TOKEN-UER-1; TOKEN-OBS-1 may ship before CE/MEM integrations, TOKEN-OBS-2 after first optimized source exists.  
**Delivery rule:** one `TOKEN-OBS-*` row per PR; emit through HOS or approved domain-signal path only.

**LKW proof:** **LKW-PF6** is the platform proof workload for token savings telemetry and regression gates ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-PF). **LKW-PF6-0** proof design (**Done / Closed**) defines required observability attribution fields and redaction rules — see [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md) §LKW-PF6-0 and [`applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md`](../../../../applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md) §10.8. No private telemetry bus.

**Early slices (before full TOKEN-OBS-1/2):** **TOKEN-6A-lite** defines the token-savings telemetry payload shape only; HOS emission/exporter wiring remains future work — no private telemetry bus. **TOKEN-6A** adds helper-only telemetry summary/counter payloads for TOKEN-2..4; it does not emit through HOS, does not register runtime subscribers, and does not wire exporters. **TOKEN-6A-R** hardens telemetry summary metadata before future HOS/exporter wiring. Summary metadata passthrough is allow-listed and unsafe/raw-content-like metadata is rejected or dropped. This still does not emit through HOS, does not register subscribers, and does not wire exporters. **TOKEN-6B** adds a deterministic helper-only token regression benchmark runner with default fixtures for `tool_schema`, `context_pack`, and `memory_summary`, plus `scripts/check_token_regression_benchmarks.py` as a local gate; it does not emit through HOS, does not wire exporters, does not execute LKW proof runs, and does not add LLM-as-a-Judge. **TOKEN-6B-R** refines TOKEN-6B validation expectation handling to fail closed: `expect_validation_pass=True` accepts only `passed` / `not_applicable` statuses; `unknown`, `missing`, `failed`, `runner_error`, or unexpected statuses fail the fixture; no HOS/exporter/runtime wiring or LLM-as-a-Judge was added. **TOKEN-OBS-1A** (**Done / Closed**) adds a helper-only token optimization domain signal model, safe metadata sanitizer, builders for optimization outcomes and regression results, and in-memory/no-op sinks; it does not emit through HOS, does not register runtime subscribers, does not wire exporters, does not integrate runtime hot paths, does not execute LKW proof runs, and does not add LLM-as-a-Judge. **TOKEN-OBS-1A-R** refines TOKEN-OBS-1A receipt reference safety: `receipt_ref` metadata is sanitized before being attached to signals; raw content/prompt/context/evidence cannot bypass the sanitizer through `receipt_ref.metadata`; receipt identity fields are preserved; no HOS/exporter/runtime wiring or LLM-as-a-Judge was added. **TOKEN-OBS-1B** (**Done / Closed**) adds a typed token optimization `RuntimeEventPayload`, domain event kind registration (`intergrax.token_optimization.signal`), safe `TokenOptimizationSignal` → payload conversion, and explicit `emit_token_optimization_domain_signal(...)` through `emit_domain_signal(...)`; payload metadata and receipt_ref metadata are sanitized; no optimizer auto-emission, no regression runner auto-emission, no runtime subscribers, no observability exporter wiring, no Elasticsearch/Kibana wiring, no LKW proof execution, and no LLM-as-a-Judge was added. **TOKEN-OBS-1C** (**Done / Closed**) adds explicit opt-in emission helpers that combine the safe signal builders with the HOS domain-signal adapter (`emit_token_optimization_outcome`, `emit_token_regression_result`, `emit_token_regression_summary`); optional dry-run/no-emit mode is supported; metadata and receipt_ref metadata remain sanitized; no optimizer auto-emission, no regression runner auto-emission, no runtime subscribers, no observability exporter wiring, no Elasticsearch/Kibana wiring, no LKW proof execution, and no LLM-as-a-Judge was added. **TOKEN-OBS-1D** (**Done / Closed**) adds policy-gated `maybe_emit_*` runtime emission hooks (`maybe_emit_token_optimization_outcome`, `maybe_emit_token_regression_result`, `maybe_emit_token_regression_summary`); emission policy defaults to disabled; enabled policy emits through the existing safe domain-signal adapter; kind-level gates cover outcomes, regression results, and regression summaries; dry-run policy mode builds signal/payload without emitting; metadata and receipt_ref metadata remain sanitized; no optimizer auto-emission, no regression runner auto-emission, no runtime subscribers, no observability exporter wiring, no Elasticsearch/Kibana wiring, no LKW proof execution, and no LLM-as-a-Judge was added. **TOKEN-OBS-1E** (**Done / Closed**) adds `run_token_regression_benchmarks_with_emission(...)` wrapper around the existing deterministic regression runner; per-result and summary emission use the TOKEN-OBS-1D `maybe_emit_*` helpers only from the wrapper; default policy disabled; `emit_results` / `emit_summary` flags gate attempts; core `regression.py` and `scripts/check_token_regression_benchmarks.py` unchanged; no optimizer auto-emission, no exporter wiring, no LKW proof, and no LLM-as-a-Judge was added. **TOKEN-OBS-2A** (**Done / Closed**) adds a redaction-safe regression benchmark report artifact (`TokenRegressionReport`, `build_token_regression_report`, dict/formatter helpers) for `TokenRegressionSummary` and optional `TokenRegressionEmissionRunResult`; report items include only safe scalar benchmark fields; metadata uses the existing token optimization metadata sanitizer; emission section aggregates status counts without event payloads; core `regression.py`, `regression_emission.py`, and `scripts/check_token_regression_benchmarks.py` unchanged; no LKW integration, no exporter wiring, and no LLM-as-a-Judge was added. **TOKEN-OBS-2B** (**Done / Closed**) expands the deterministic regression benchmark default fixtures with a minimal eval matrix covering compactable savings, protected-region preservation, and validation-failure fallback across `tool_schema`, `context_pack`, and `memory_summary`; expectations are explicit for pass/fail, validation, fallback, savings bounds, and receipt presence; report items expose safe eval metadata (`eval_case`, `expected_behavior`, `expectation_status`) without raw fixture bodies; `scripts/check_token_regression_benchmarks.py` unchanged and still exits 0 for the default suite; no LKW integration, no exporter wiring, and no LLM-as-a-Judge was added. **TOKEN-OBS-2C** (**Done / Closed**) adds a formal regression gate artifact (`TokenRegressionGateThresholds`, `evaluate_token_regression_gate`, dict/formatter helpers) over `TokenRegressionSummary` with optional `TokenRegressionReport` cross-checks; default thresholds pass the current 7-fixture suite; stable failure reason codes cover fixture pass/fail, missing receipts, expectation status, unexpected fallback, and optional aggregate savings floors; metadata uses the existing token optimization metadata sanitizer; core `regression.py`, `regression_report.py`, `regression_emission.py`, and `scripts/check_token_regression_benchmarks.py` unchanged and do not import the gate module; no LKW integration, no exporter wiring, and no LLM-as-a-Judge was added. **TOKEN-OBS-2D** (**Done / Closed**) exposes existing regression report and gate artifacts through `scripts/check_token_regression_benchmarks.py` via `--report`, `--report-json`, `--gate`, `--gate-json`, and gate-only `--min-total-saved-ratio` / `--min-total-saved-tokens`; default and `--json` behavior unchanged; output modes are mutually exclusive; exit code remains non-zero on benchmark failures and gate modes also fail on gate `fail`; report/gate output stays redaction-safe through existing helpers; no files written; no LKW integration, no exporter wiring, and no LLM-as-a-Judge was added. Full **TOKEN-OBS-1** hot-path wiring and **TOKEN-OBS-2** regression-gate reporting remain future work. **OBS-HEALTH-lite** adds a minimal operator-visible health/status shape for exporter and token telemetry; it is not full observability production hardening. Full **OBS-VENDOR** production hardening (auth/TLS, retention, batching, dashboards-as-code, CI/live proof automation, and related operational closeout) remains **Planned**.

**Maturity bar (LKW-PF0):** Platform proof vs operational proof vs production-grade readiness is defined in [`applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md`](../../../../applications/local_workspace_application/docs/PLATFORM_PROOF_LOOP.md) §9. **Elasticsearch/Kibana** observability export is **closed for platform proof** (see LKW §9.6 example); **OBS-VENDOR** production hardening rows in this plan remain the production hardening backlog — closing the platform proof does not imply production-grade readiness.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-OBS-1** | Code | P1 | Planned | `intergrax/runtime/token_optimization/telemetry.py` with typed optimization summary payload, receipt visibility, and counters/spans for saved tokens, failures, fallbacks, source type, strategy, output profile, model/provider | No private telemetry bus; attribution includes run/step/source/model/provider/strategy/profile; redaction rules followed; compatible with unified run journal; `uv run pytest tests/unit/runtime/observability/ -q`; `uv run pytest tests/unit/runtime/token_optimization/ -q` |
| **TOKEN-OBS-2** | Test/Gate | P1 | Planned | Token-vs-quality regression benchmark runner and scripts `check_compression_receipts.py`, `check_token_regression_benchmarks.py` | CI can fail on uncontrolled token growth, missing receipts, protected-region failures, or quality regression; benchmark fixtures cover output policy, tool catalog, and context pack cases; `uv run python scripts/check_compression_receipts.py`; `uv run python scripts/check_token_regression_benchmarks.py` |

**Explicit exclusions:** no new `RuntimeEventType` unless ADR/OBS review requires it; prefer typed payload/domain-signal style consistent with OBS event ownership; no raw prompt/completion persistence in production traces.

---

## Phase OBS-EXPORT — External observability export boundary (In progress)

**Purpose:** Define the platform-level export boundary for external observability sinks such as JSONL/file, OTLP, Elasticsearch, Langfuse, Arize/Phoenix.

**OBS-EXPORT-1 status (2026-06-28):**

- **OBS-EXPORT-1-EXPORT-BOUNDARY** — **Done**
- Normalized **`ObservabilityExportEnvelope`** added (`observability_export_envelope.v1`)
- **`ObservabilityExporter`** async protocol + **`NoOpObservabilityExporter`** + **`InMemoryObservabilityExporter`** / **`TestObservabilityExporter`**
- Typed source models (`RuntimeEventExportSource`, `GatewayCallExportSource`) and safe mapping helpers from `RuntimeEvent`, `ToolCallRecord`, `RagCallRecord`, and `JournalRef`
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix, Elasticsearch, OTLP deferred to OBS-EXPORT-4/5)
- Lifecycle wiring and failure isolation **deferred to OBS-EXPORT-2** (superseded below)

**OBS-EXPORT-2 status (2026-06-28):**

- **OBS-EXPORT-2-EXPORT-POLICY-AND-WIRING** — **Done**
- Typed **`ObservabilityExportPolicy`** with explicit allow/drop/hash posture (`apply_observability_export_policy`, `try_export_observability_envelope`)
- Default local-first posture remains **disabled by default**; **`export_content=false`** by default; strict redaction by default
- Raw prompts, documents, RAG chunks, synthesized content, tool args, secrets, and full local file paths are **not exported by default**
- Exporter failure isolation added — exporter exceptions are logged and never fail product/runtime runs
- Minimal runtime lifecycle wiring implemented via **`make_observability_export_runtime_plugin`** (optional bus subscriber; defaults to **`NoOpObservabilityExporter`**; export runs after canonical bus recording)
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- JSONL/file exporter remains **OBS-EXPORT-3** (superseded below)
- OTLP/Elasticsearch remains **OBS-EXPORT-4**

**OBS-EXPORT-3 status (2026-06-28):**

- **OBS-EXPORT-3-SAFE-JSONL-FILE-EXPORTER** — **Done**
- **`JsonlObservabilityExporter`** added — local-first, explicit opt-in JSONL/file sink implementing **`ObservabilityExporter`**
- Exporter writes normalized **`ObservabilityExportEnvelope`** records (one JSON object per line, UTF-8, append by default)
- Exporter does **not** register globally; platform bootstrap registration remains deferred unless explicitly planned later
- Redaction/export policy remains upstream — exporter writes the envelope it receives; sanitized metadata-only records when used through **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Raw content export remains **unsupported/disabled by default** (`export_content=false`)
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- OTLP/Elasticsearch remains **OBS-EXPORT-4**

**OBS-EXPORT-4A status (2026-06-28):**

- **OBS-EXPORT-4A-APPLICATION-ATTRIBUTES-CONTRACT** — **Done**
- Typed **`ApplicationObservabilityAttributes`** contract added — application developers extend/customize through inheritance (e.g. `LocalWorkspaceObservabilityAttributes`, `BillingObservabilityAttributes`)
- No arbitrary public `dict[str, Any]` metadata boundary — safe scalar/list values only (`str`, `int`, `float`, `bool`, `None`, `list[str]`)
- Namespaced attribute keys (`{namespace}.{field}`) with stable schema/version fields
- **`sanitize_application_observability_attributes`** applies policy/redaction before export — unsafe/raw/sensitive values rejected, dropped, or path-like values hashed
- **`ObservabilityExportEnvelope`** integrates optional `application_attributes` (pre-policy input) and `sanitized_application_attributes` (post-policy export)
- **`JsonlObservabilityExporter`** consumes already-normalized sanitized attributes via envelope JSON serialization
- Vendor adapters **not implemented** (Langfuse, Arize, Phoenix remain OBS-EXPORT-5)
- LKW remains **unchanged**

**OBS-EXPORT-4 status (2026-06-28):**

- **OBS-EXPORT-4-OTLP-ADAPTER** — **Done**
- **`OtlpObservabilityExporter`** added as first remote backend adapter — platform observability package only (not applications, not LKW-specific)
- Adapter consumes normalized **`ObservabilityExportEnvelope`** only; expects policy-approved envelopes from **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Adapter consumes **`sanitized_application_attributes`** only — does **not** read or export raw **`application_attributes`**
- Adapter maps sanitized application attributes into OTLP-safe attribute keys (namespaced keys preserved)
- Adapter does **not** export raw content; **`export_content=false`** posture unchanged
- Injectable **`OtlpTransport`** protocol — no vendor SDK coupling; no network in unit tests
- Explicit opt-in — adapter is **not** globally registered in platform bootstrap
- Elasticsearch remains **deferred** unless separately planned
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**
- Global bootstrap registration remains **deferred** unless explicitly planned later

**OBS-EXPORT-4B status (2026-06-28):**

- **OBS-EXPORT-4B-OTLP-HTTP-TRANSPORT** — **Done**
- **`OtlpHttpTransport`** added — concrete HTTP POST transport implementing **`OtlpTransport`**
- Transport belongs to platform observability only (not applications, not LKW-specific)
- Transport is explicit opt-in — **not** globally registered in platform bootstrap
- Transport sends only OTLP-safe JSON payloads produced by **`OtlpObservabilityExporter`** from policy-sanitized envelopes
- Redaction and failure isolation remain upstream — **`apply_observability_export_policy`** / **`try_export_observability_envelope`**
- Transport does **not** read or export raw **`application_attributes`**; no raw content export
- No vendor SDK coupling — lightweight **`httpx`** client only; injectable client for tests (no real network in unit tests)
- Operator/bootstrap wiring remains **deferred** unless explicitly planned later
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**

**OBS-EXPORT-4C status (2026-06-28):**

- **OBS-EXPORT-4C-EXPLICIT-OTLP-OPERATOR-WIRING** — **Done**
- Explicit OTLP operator wiring helper added — **`build_otlp_observability_exporter`**, **`build_otlp_observability_export_runtime_plugin`**
- Typed operator config added — **`ObservabilityExportOperatorConfig`**, **`OtlpExportOperatorConfig`**
- Disabled by default (`enabled=false`); **`export_content=false`** by default and enforced in runtime plugin wiring
- No global bootstrap registration — operator/platform code must explicitly construct and register the plugin
- No LKW wiring — platform observability package only
- No raw content export; no raw **`application_attributes`** export — policy sanitization remains upstream
- OTLP HTTP export can now be explicitly assembled from operator config (**`ObservabilityExportPolicy`** + **`OtlpObservabilityIntegration`** + **`OtlpObservabilityExporter`** + **`OtlpHttpTransport`** + **`make_observability_export_runtime_plugin`**)
- Injectable transport for tests — no network in unit tests
- Langfuse / Arize / Phoenix remains **OBS-EXPORT-5**
- Elasticsearch remains **deferred** unless separately planned

**INTEGRATIONS-1C status (OTLP observability integration alignment):**

- **INTEGRATIONS-1C** — **Done**
- **`OtlpObservabilityIntegration`** added — first concrete observability vendor integration
- Derives from **`ObservabilityVendorIntegrationContract`**; `provider_id=otlp`
- Wraps existing **`OtlpObservabilityExporter`** / **`OtlpTransport`** as lower-level implementation details
- Operator wiring (**`build_otlp_observability_integration`**) constructs integration-backed OTLP export path explicitly
- Consumes only policy-sanitized envelopes; **`sanitized_application_attributes`** only — never raw **`application_attributes`**
- **`JsonlObservabilityExporter`** unchanged — classified as local file export sink, not remote observability vendor
- No global bootstrap registration; no LKW change; no Langfuse/Arize/Phoenix/Elasticsearch adapters

**INTEGRATIONS-1D status (LKW/local workspace observability platform wiring):**

- **INTEGRATIONS-1D** — **Done**
- **`build_local_workspace_observability_plugins`** added in LKW host — composes platform **`ObservabilityExportOperatorConfig`** only; no LKW-specific exporter
- Disabled by default; **`export_content=false`** enforced via platform **`build_otlp_observability_export_runtime_plugin`**
- LKW factory accepts optional **`observability_export`** and registers returned **`RuntimePlugin`** only at LKW bootstrap — no global registration
- OTLP integration-backed path only (**`build_otlp_observability_export_runtime_plugin`** → **`OtlpObservabilityIntegration`**); no direct **`OtlpHttpTransport`** from LKW
- LKW.2 pipeline unchanged; no vendor SDK in LKW; raw content/local paths not exported by default
- **OBS-EXPORT-5** remains **deferred** until Langfuse/Arize/Phoenix vendor adapters are implemented

**OBS-EXPORT-5 status (2026-06-28 — post INTEGRATIONS-2C):**

| Sub-deliverable | Status | Notes |
|-----------------|--------|-------|
| Contract adapters | **Complete** | All **`observability_backend`** slugs have **`ObservabilityVendorIntegrationContract`** subclasses (**INTEGRATIONS-2C**) |
| Production vendor transports | **Pending** | Injectable transport protocols + fake transports in tests only — no Langfuse/Arize/New Relic/etc. network exporters |
| Operator / bootstrap wiring | **Pending** | No global registration; LKW unchanged; registry v2 deferred |
| Production export end-to-end | **Not done** | Contract layer only — do **not** treat OBS-EXPORT-5 as “production export done” |

**OBS-EXPORT-5 status (dependency on INTEGRATIONS-1A / INTEGRATIONS-1B / INTEGRATIONS-1C):**

- **OBS-EXPORT-5** remains **paused** until remaining observability integration alignment is complete (OTLP done in INTEGRATIONS-1C; Langfuse/Arize/Phoenix deferred)
- Generic **`PlatformIntegrationContract`** added in **INTEGRATIONS-1A** — **Done**
- Observability vendor specialization **`ObservabilityVendorIntegrationContract`** added in **INTEGRATIONS-1B** — **Done**
- OTLP aligned with **`ObservabilityVendorIntegrationContract`** in **INTEGRATIONS-1C** — **Done**
- Concrete Langfuse / Arize / Phoenix adapters remain **deferred** — must subclass **`ObservabilityVendorIntegrationContract`**, not ad-hoc exporter classes
- JSONL remains a local file export sink — not migrated to observability vendor integration contract in INTEGRATIONS-1C
- **No LKW change** in INTEGRATIONS-1C

**INTEGRATIONS-2A status (provider category contracts):**

- **`observability_backend`** provider folder aligns with existing **`ObservabilityVendorIntegrationContract`** — no duplicate observability backend contract
- Runtime **`integration_kind`** for observability vendor integrations remains **`observability_vendor`**; folder name **`observability_backend`** is documented via **`PlatformIntegrationKind.OBSERVABILITY_BACKEND`**
- Existing **`observability_backend`** catalog providers (Langfuse, Arize, Phoenix, Elasticsearch, Datadog, …) are **still awaiting migration/adaptation** to category contracts
- **OBS-EXPORT-5** remains **deferred** until those providers are adapted as concrete **`ObservabilityVendorIntegrationContract`** subclasses

**INTEGRATIONS-2B-LANGFUSE pilot status (2026-06-28):**

- Existing Langfuse **`observability_backend`** provider adapted to **`ObservabilityVendorIntegrationContract`** as reference pilot
- **Pattern hardened (INTEGRATIONS-2B-FOLLOWUP):** canonical provider package layout, scaffold idempotency, `enabled=True` requires transport at construction
- **`LangfuseObservabilityIntegration`** consumes policy-sanitized envelopes only; legacy **`ObservabilityBackend`** query facade unchanged
- **Registry v2 / contract registry wiring deferred** — `register_langfuse_integration()` still registers legacy query facade only
- **OBS-EXPORT-5** remains **not complete** until remaining observability_backend providers (Arize, Phoenix, …) are adapted

**INTEGRATIONS-2C status (observability_backend provider migration — 2026-06-28):**

- All existing **`observability_backend`** provider packages adapted to **`ObservabilityVendorIntegrationContract`** (Langfuse reference + 25 batch slugs)
- Each provider: contract-based `integration.py`, `create_<slug>_observability_integration` factory, lazy public API; legacy **`ObservabilityBackend`** query facade unchanged
- Sanitized **`ObservabilityExportEnvelope`** only; raw **`application_attributes`** rejected; no raw content export
- Injectable transport protocol only — **no real vendor network transports** in this task; no vendor SDK imports in `integration.py`
- **`enabled=True`** without transport fails at construction (**`IntegrationConfigurationError`**)
- Parametrized conformance tests in **`test_observability_provider_contract_migration.py`**
- **Registry v2 / contract registry wiring deferred** — `register_<slug>_integration()` still legacy-only
- **OBS-EXPORT-5 progress:** contract adapters **complete**; production vendor transports and operator/bootstrap wiring **pending** — not production export done
- No LKW change

**Required decisions:**

- External sinks are optional subscribers/export targets, not semantic owners of Intergrax observability.
- Intergrax RuntimeEvent / trace / journal / diagnostic payloads remain the canonical source.
- Vendor SDKs must not be called directly from runtime hot paths, agents, or LKW product code.
- Exporter failure must never fail product runs.
- Raw prompts, raw documents, raw RAG chunks, raw synthesized content, secrets, and full local file paths are not exported by default.
- Redaction/export policy must run before external export.
- Default posture for local-first apps such as LKW: disabled by default, strict redaction, `export_content=false`.
- OBS-EXPORT depends on LKW.2.4 pipeline proof as a representative multi-agent workload.

**Delivery rule:** one `OBS-EXPORT-*` row per PR; export through normalized envelope only; no vendor SDK in runtime hot paths.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **OBS-EXPORT-1** | Code | P2 | **Done** | Normalized export envelope and exporter interface | Defines stable export envelope, exporter interface, no-op exporter, and test exporter. Uses existing spine/journal/runtime metadata as source. No vendor SDK. |
| **OBS-EXPORT-2** | Code | P2 | **Done** | Redaction/export policy and failure isolation | Explicit allow/drop/hash policy for exported fields. Export timeout/failure does not fail the run. Tests prove raw content is not exported. Minimal runtime plugin wiring via `make_observability_export_runtime_plugin`. |
| **OBS-EXPORT-3** | Code | P2 | **Done** | Safe JSONL/file exporter | **`JsonlObservabilityExporter`** writes policy-sanitized **`ObservabilityExportEnvelope`** JSONL records; local-first explicit opt-in; no global bootstrap registration; no vendor SDK; raw content export disabled by default. |
| **OBS-EXPORT-4A** | Code | P2 | **Done** | Typed application observability attributes contract | **`ApplicationObservabilityAttributes`** base + subclass extension; namespaced safe metadata; policy sanitization before export; envelope integration; no arbitrary public dict boundary; no vendor SDK. |
| **OBS-EXPORT-4** | Code | P2 | **Done** | First remote backend adapter: OTLP | **`OtlpObservabilityExporter`** + **`OtlpObservabilityExporterConfig`** + injectable **`OtlpTransport`**; maps policy-sanitized envelopes (including **`sanitized_application_attributes`**) to OTLP-safe log payloads; no vendor SDK; explicit opt-in; no global bootstrap registration. Elasticsearch deferred. |
| **OBS-EXPORT-4B** | Code | P2 | **Done** | OTLP HTTP transport | **`OtlpHttpTransport`** POSTs OTLP JSON to configured endpoint; explicit opt-in; platform observability only; policy-sanitized payloads only; failure isolation via **`try_export_observability_envelope`**; no vendor SDK; no global bootstrap registration. Operator wiring deferred. |
| **OBS-EXPORT-4C** | Code | P2 | **Done** | Explicit OTLP operator wiring | **`ObservabilityExportOperatorConfig`** + **`OtlpExportOperatorConfig`** + **`build_otlp_observability_exporter`** + **`build_otlp_observability_export_runtime_plugin`**; disabled by default; **`export_content=false`** enforced; composes policy + OTLP exporter + HTTP transport + runtime plugin; no global registration; no LKW wiring; no raw content or raw **`application_attributes`** export. |
| **OBS-EXPORT-5** | Code | P3 | **In progress** | Observability vendor export | **Contract adapters complete** (INTEGRATIONS-2C). **Production transports pending.** **Operator wiring pending.** Not production export done. Adapters derive from **`ObservabilityVendorIntegrationContract`**; sanitized envelope only; JSONL remains local file sink. |

---

## Phase OBS-VENDOR — Production observability vendor integration rollout (Planned)

**Observability projections:** **Elasticsearch/Kibana**, **Prometheus/Grafana**, **Tempo** (or equivalent), and **Sentry** are different observability projections — structured event/log timeline, metrics/SLO dashboards, distributed traces/spans, and error issue triage respectively. They complement each other; none replaces the others. Platform contracts remain vendor-neutral; backends are replaceable. See LKW strategic roadmap **LKW-PF5** in [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md).

**Purpose:** Move from the LKW OTLP proof path to a production-grade, vendor-agnostic observability export model where runtime/LKW call only the contract and vendors own backend I/O.

**Cross-plan — LKW proof workload:** [`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) §LKW-OBS.

**Platform contract:** [`intergrax/runtime/integrations/observability.py`](../../../../intergrax/runtime/integrations/observability.py) · **Operator wiring:** [`intergrax/runtime/observability/operator_wiring.py`](../../../../intergrax/runtime/observability/operator_wiring.py) · **LKW wiring:** [`applications/local_workspace_application/host/observability_wiring.py`](../../../../applications/local_workspace_application/host/observability_wiring.py).

### Current state — LKW OTLP proof path (Done)

The following LKW/platform tasks are **Done** and establish the baseline export spine:

| ID | Status | Summary |
|----|--------|---------|
| **LKW-OBS-OTLP-1A** | **Done** | LKW env-driven OTLP observability export configuration |
| **LKW-OBS-OTLP-1B** | **Done** | LKW Docker Compose self-hosted OpenTelemetry Collector + persisted JSONL sink |
| **LKW-OBS-OTLP-1C** | **Done** | Manual Swagger/Compose proof — runtime events exported as OTLP logs to JSONL |
| **LKW-OBS-OTLP-DUP-1** | **Done** | Duplicate export runtime events fixed; duplicate check for current run = 0 |
| **LKW-OBS-VIEW-1A** | **Done** | Lightweight OTLP log inspector (`inspect_otlp_logs.py`, `inspect-otlp-logs.bat`); focused tests 5 passed; manual duplicate check = 0 |

**Current end-to-end path (proof only):**

```text
LKW runtime
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ OtlpObservabilityIntegration.export()   # contract
→ OtlpObservabilityExporter / OtlpHttpTransport
→ OpenTelemetry Collector
→ persisted JSONL (.observability/otel/lkw-otlp-logs.jsonl)
→ local inspector (inspect-otlp-logs.bat)
```

**Target end-to-end path (production):**

```text
Intergrax runtime / LKW
→ ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract.export()
→ vendor-specific integration
→ vendor-specific deliver_payload()
→ Langfuse / Arize / Phoenix / Elasticsearch / OTLP / custom backend
```

### Observability Vendor Integration Invariant

All observability vendor export must flow through:

```text
ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract.export()
→ vendor deliver_payload()
```

**LKW, agents, runtime loops, and application code must not call vendor SDKs/APIs directly.**

Vendor SDK/API calls are allowed only inside concrete provider implementations under:

```text
intergrax/integrations/providers/observability_backend/<vendor>/
```

Additional developer metadata must use **`ApplicationObservabilityAttributes`**.

Artifacts must be exported only as references: **`artifact_ref`**, **`sha256`**, **`safe_relative_path`**, **`schema_id`**.

Raw content, prompts, chunks, tool args, secrets, and absolute paths must not be exported by default.

**Core rule:** Platform knows the contract. Vendor knows its backend. LKW does not know the vendor.

### Layer responsibilities

| Layer | Location | Owns | Must not contain |
|-------|----------|------|------------------|
| **Base / contract** | `intergrax/runtime/integrations/observability.py` | `ObservabilityVendorIntegrationContract`, `ObservabilityVendorPayload`, `ObservabilityVendorSignal`, `map_envelope()`, `export()`, shared envelope→vendor-neutral mapping, policy-safe boundary | Vendor SDK/API imports or vendor-specific network I/O |
| **Vendor providers** | `intergrax/integrations/providers/observability_backend/<vendor>` | `integration.py`, optional `transport.py`, config/factory, manifest/register/bundle per existing convention, vendor-specific `deliver_payload()`, vendor-specific payload mapping only when necessary, vendor-specific tests | Direct calls from LKW, agents, or runtime loops |
| **Wiring / config** | `intergrax/runtime/observability/operator_wiring.py`, application host settings (e.g. LKW `host/settings.py`, `host/observability_wiring.py`) | Typed backend selection, integration factory, `ObservabilityExporter`-compatible plugin assembly | Vendor SDK calls; vendor-specific branching in LKW product code |

Runtime and LKW always invoke:

```python
await observability_integration.export(envelope)
```

Wiring selects the concrete integration; product code never branches on vendor SDKs.

### OBS-VENDOR task register

**Delivery rule:** one `OBS-VENDOR-*` row per PR; no vendor SDK in runtime hot paths, LKW, or agents; exporter failure must never fail product runs.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **OBS-VENDOR-0** | Docs | P2 | **Done** | Close LKW OTLP inspector status in LKW implementation plan | Implementation plan references `applications/local_workspace_application/scripts/inspect_otlp_logs.py` and `inspect-otlp-logs.bat`; states duplicate check = 0; states focused inspector tests = 5 passed; no code changes |
| **OBS-VENDOR-1** | Docs/Code | P1 | **Done** | Define canonical vendor integration execution model | Plan states platform/runtime/LKW call only contract-level `export()`; vendor SDK/API calls belong only in provider implementations; LKW remains vendor-agnostic; direct Langfuse/Elastic/Phoenix/Arize calls from LKW, agents, runtime loops, or application code are forbidden |
| **OBS-VENDOR-2** | Code | P1 | **Done** | Open plugin-based observability backend selection in operator/runtime config | Plan defines where backend selection belongs; invalid backend_id format fails fast; valid but unregistered backend_id fails at builder registry lookup; selection produces `ObservabilityExporter`-compatible integration; existing OTLP behavior preserved. **Scope:** config/wiring shape only — do not implement all vendors at once. **Built-in today:** `otlp`. **Extension:** register additional backend builders by `backend_id`. **Expected env:** `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED=true`, `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND=otlp` |
| **OBS-VENDOR-3** | Docs/Code | P1 | **Done** | Formalize safe extension metadata API | `ApplicationObservabilityAttributes` documented as official extension path; artifact refs (`artifact_ref`, `sha256`, `safe_relative_path`, `schema_id`) as reference-only path; raw artifact content not exported; forbidden fields remain blocked by export policy; arbitrary `RuntimeEvent.payload` fields not auto-exported; optional helper API (e.g. `emit_observability_event(..., application_attributes=..., artifact_ref=...)`) only if needed |
| **OBS-VENDOR-4A** | Code | P1 | **Done** | **First concrete vendor adapter: Elasticsearch/OpenSearch** | Contract subclass under `intergrax/integrations/providers/observability_backend/elasticsearch`; injectable transport for indexing policy-safe `ObservabilityVendorPayload`; no raw content export; no policy bypass; unit tests with fake transport prove policy-safe delivery, disabled config does not send, unsafe content not exported; no LKW direct dependency on Elasticsearch/OpenSearch SDK |
| **OBS-VENDOR-5** | Code | P1 | **Done** | Wire selected vendor backend into runtime operator config | `LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND=elasticsearch` builds `ElasticsearchObservabilityIntegration.from_transport(...)`; `otlp` continues `OtlpObservabilityIntegration`; runtime plugin receives only contract/`ObservabilityExporter` object; LKW does not branch on vendor SDK; misconfigured credentials/endpoint fail fast at build time; exporter failures do not fail product runs |
| **OBS-VENDOR-6** | Code | P2 | Planned | Vendor-specific operational hardening | Bounded transport timeout; retry/backoff; explicit batching or non-batching decision documented; failure isolation; rate-limit handling; dead-letter or failed-export diagnostics; structured error reason; health check capability; exporter failure never breaks LKW run; tests cover transport failure isolation |
| **OBS-VENDOR-7** | Test/Docs | P1 | Planned | End-to-end vendor proof (Elasticsearch/OpenSearch first) | LKW run → envelope → policy → vendor integration → backend → query/readback by `run_id`/`event_id`; proof records exact commands, `run_id`, backend query result; `tool_requested`/`tool_completed` appear once; duplicate check = 0; no raw query/content/prompt/chunks/secrets indexed; documented in LKW runbook or implementation plan |
| **OBS-VENDOR-8** | Docs/Code | P3 | Planned / Later | Langfuse/Phoenix/Arize semantic mapping phase | After first event/log-oriented adapter: map Intergrax records (`runtime_event`, `tool_call`, `rag_call`, `llm_call`, `journal_ref`, `diagnostic`) to vendor concepts (`trace`, `span`, `generation`, `event`, `score`, `metadata`); preserve `run_id`/`task_id`/`event_id` correlation; no direct SDK calls outside provider package; no raw prompts/completions unless future explicit content-export mode is designed and approved; policy remains metadata-only by default |

**OBS-VENDOR-1 status:**

Done — canonical execution model closed. Runtime and applications call only **`ObservabilityVendorIntegrationContract.export()`**; vendor SDK/API calls are restricted to concrete provider implementations under **`observability_backend/<vendor>`**; LKW remains vendor-agnostic. Export policy runs before external export; exporter failure must never fail product runs.

**OBS-VENDOR-2 status:**

Done — observability backend selection is open and plugin-based. Operator config uses normalized `backend_id` strings rather than a closed vendor enum. Built-in registered backends today: `otlp`, `elasticsearch`. Additional vendors are selected by `backend_id` and become available when a backend builder is registered; valid but unregistered `backend_id`s fail fast with a missing-builder error.

**OBS-VENDOR-2A correction:**

Closed backend enums are forbidden as runtime/operator selectors. Well-known vendor names may exist as documentation/scaffold constants only, not as the extensibility boundary.

**OBS-VENDOR-2B correction:**

Generic backend selection no longer carries OTLP-specific configuration requirements. OTLP endpoint is required only when `backend_id="otlp"`. Non-OTLP `backend_id`s are valid at config time and become usable when a backend builder is registered. Generic builder APIs are vendor-neutral; OTLP transport injection is kept only on OTLP-specific helper functions and tests may use custom registries for injection.

**OBS-VENDOR-3 status:**

Done — safe application extension metadata is formalized. Applications attach typed metadata through ApplicationObservabilityAttributes and artifact references through ObservabilityArtifactReference. The official helper attaches these extensions to ObservabilityExportEnvelope without accepting arbitrary payload dictionaries. Export policy remains the sanitization boundary: raw application_attributes are cleared, sanitized_application_attributes are emitted, artifact references remain reference-only, and raw prompts/content/query/chunks/tool args/secrets/full paths remain forbidden.

**OBS-VENDOR-4A status:**

Done — concrete Elasticsearch/OpenSearch observability export transport added under `intergrax/integrations/providers/observability_backend/elasticsearch/transport.py`. `ElasticsearchHttpObservabilityTransport` maps policy-safe `ObservabilityVendorPayload` to index documents and delivers via provider-owned `ElasticsearchRestClient.index_document()`. `create_elasticsearch_observability_transport()` and `create_elasticsearch_observability_integration()` expose injectable transport wiring; `enabled=False` sends nothing; `enabled=True` without transport fails at factory time with `IntegrationConfigurationError`. Unit tests use fake transport/indexer only. Follow-up: transport dispatches sync provider-owned indexing via asyncio.to_thread() and indexes append-only by default; event_id/correlation_id remain query fields, not default document IDs.

**OBS-VENDOR-5 status:**

Done — `backend_id="elasticsearch"` now resolves through open `ObservabilityExportBackendRegistry`/operator wiring to `ElasticsearchObservabilityIntegration` with provider-owned transport; OTLP remains working; no LKW vendor SDK/client dependency; Docker/E2E proof deferred.

**OBS-VENDOR-6A status:**

Done — Elasticsearch export failures now produce safe provider-owned delivery diagnostics with retriable/non-retriable classification. This closes failure classification for the first vendor backend; batching, retry/backoff execution, dead-letter storage, auth/TLS, and health checks remain separate OBS-VENDOR-6 follow-ups.

**OBS-VENDOR-6B status:**

Done — Elasticsearch observability delivery now supports bounded provider-owned retry/backoff for retriable delivery failures, with LKW env overrides for deployment-specific calibration. Full OBS-VENDOR-6 remains Planned until batching, dead-letter storage, auth/TLS, health checks, and broader operational hardening are complete.

**OBS-VENDOR-6C-A status:**

Done — Elasticsearch observability export now exposes a provider-owned, optional failed-delivery sink contract with safe diagnostic records only (no raw document or content fields). Invocation occurs on ultimate delivery failure (immediate for non-retriable errors; after retry exhaustion for retriable errors). Full OBS-VENDOR-6C operational hardening (durable dead-letter storage) and full OBS-VENDOR-6 remain Planned.

**OBS-VENDOR-6C-B1 status:**

Done — Elasticsearch provider now ships a file-backed failed-delivery sink (`FileElasticsearchFailedDeliverySink`) that appends one UTF-8 JSON object per line using only safe `ElasticsearchFailedDeliveryRecord` fields. No LKW wiring, env configuration, or operational runbook yet. Full OBS-VENDOR-6C remains Planned until the file sink is wired/configurable and operationally documented. Full OBS-VENDOR-6 remains Planned.

**OBS-VENDOR-6C-B2 status:**

Done — typed operator config (`ElasticsearchExportOperatorConfig.failed_delivery_file_path`) now wires the provider-owned file failed-delivery sink through runtime Elasticsearch export wiring into `create_elasticsearch_observability_transport()`. When unset, transport keeps the no-op failed-delivery sink default. No LKW env, Docker, or operational runbook yet. Full OBS-VENDOR-6C remains Planned until LKW env wiring and operational docs exist. Full OBS-VENDOR-6 remains Planned.

**OBS-VENDOR-6C-B3 status:**

Done — LKW deployment settings now read `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH` and pass a stripped deployment-owned path into `ElasticsearchExportOperatorConfig.failed_delivery_file_path` when `backend_id=elasticsearch`; empty or whitespace disables the file sink. LKW does not implement file/JSON writes or instantiate the provider sink. Full OBS-VENDOR-6C remains Planned until operational docs/proof (Docker, scripts, live proof) are complete. Full OBS-VENDOR-6 remains Planned.

**OBS-VENDOR-6C-B4 status:**

Done — LKW operational docs and read-only failed-delivery JSONL inspector added (`applications/local_workspace_application/docs/BUILD_AND_DEPLOY.md`, `applications/local_workspace_application/scripts/inspect_elasticsearch_failed_deliveries.py`, `inspect-elasticsearch-failed-deliveries.bat`). Documents env path, safe fields, controlled local proof steps, and inspector validation. Full OBS-VENDOR-6C remains Planned until live operational proof is complete. Full OBS-VENDOR-6 remains Planned.

**OBS-VENDOR-6C-B5 status:**

Done — live local controlled-failure proof recorded. File-backed Elasticsearch failed-delivery JSONL path is operationally complete (OBS-VENDOR-6C file-backed dead-letter proof closed). Full OBS-VENDOR-6C remains **Planned** until rotation/retention, auth/TLS, health checks, batching, and index-based dead-letter storage are done. Full OBS-VENDOR-6 remains Planned.

**OBS-VENDOR-6C-B5 proof evidence (2026-07-01):**

| Field | Value |
|-------|-------|
| Date | 2026-07-01 |
| `failed_delivery_file_path` | `applications/local_workspace_application/.observability/elasticsearch/failed-deliveries.jsonl` |
| Failure mode | `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL=http:/127.0.0.1:59200` (unreachable endpoint); `LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_ATTEMPTS=1` |
| LKW run | `POST http:/127.0.0.1:8099/v1/local_workspace/run` (`capability=local.workspace.search`); `run_id=run_f4870c18fced4b83b61c38c8359e6be9` |
| Inspector command | `applications/local_workspace_application/scripts/inspect-elasticsearch-failed-deliveries.bat --check-safety` |
| Inspector result | `Records: 36`; `Reason counts: connection_error: 36`; `Validation: all records contain exactly the safe failed-delivery fields`; exit code 0 |
| Safety result | Passed — each JSONL line contains only `provider_id`, `operation`, `index`, `status_code`, `reason`, `retriable`, `attempts`, `exhausted`; no raw document, prompt, chunks, tool args, secrets, tokens, or absolute payload paths |

Sample failed-delivery record:

```json
{"provider_id": "elasticsearch", "operation": "send_observability_payload", "index": "intergrax-lkw-observability", "status_code": null, "reason": "connection_error", "retriable": true, "attempts": 1, "exhausted": true}
```

**OBS-VENDOR-7B status:**

OBS-VENDOR-7B tooling done: Elasticsearch/OpenSearch readback inspector added for list-runs, run timeline, duplicate check, and safety-key check (`applications/local_workspace_application/scripts/inspect_elasticsearch_observability.py`, `inspect-elasticsearch-observability.bat`). Follow-up: Elasticsearch inspector safety-key check now derives forbidden keys from the canonical runtime export boundary (`FORBIDDEN_EXPORT_CONTENT_FIELDS`) instead of maintaining an independent ad-hoc list. Full OBS-VENDOR-7 remains **Planned** until a live Docker Compose proof records a real `run_id` and backend query result.

### OBS-VENDOR-1 — execution model (reference)

1. Runtime emits **`ObservabilityExportEnvelope`** (from spine/journal/runtime metadata).
2. **`ObservabilityExportPolicy`** sanitizes the envelope (`apply_observability_export_policy` / `try_export_observability_envelope`).
3. Runtime plugin calls **`ObservabilityVendorIntegrationContract.export(envelope)`**.
4. Base contract **`map_envelope()`** maps to **`ObservabilityVendorPayload`** (vendor-neutral).
5. Concrete vendor overrides **`deliver_payload()`** — vendor-specific network/SDK I/O allowed **only** inside provider `transport.py` / `deliver_payload()`.
6. LKW and agents **must not** call vendor SDKs directly.

**Forbidden in LKW, agents, runtime loops, and application code:**

```python
langfuse_client.trace(...)
elasticsearch.index(...)
phoenix.log(...)
arize.log(...)
```

### OBS-VENDOR-3 — safe metadata and artifact extension (reference)

| Path | Allowed | Forbidden |
|------|---------|-----------|
| Additional safe metadata | **`ApplicationObservabilityAttributes`** (namespaced, policy-sanitized) | Arbitrary `RuntimeEvent.payload` fields auto-exported |
| Artifact references | `artifact_ref`, `sha256`, `safe_relative_path`, `schema_id` | Raw prompt, completion, message content, document content, raw chunks, query text, tool args, secrets, tokens, absolute paths, full file paths, synthesized content |

Export policy blocks forbidden content by default (`export_content=false`). Vendor adapters consume **`sanitized_application_attributes`** only.

### OBS-VENDOR-4A — first adapter decision

**Recommended first path: Elasticsearch/OpenSearch (`OBS-VENDOR-4A`).**

Rationale: current export records are event/log-oriented (`run_id`, `event_type`, `agent_id`, `tool_id`, `latency_ms`, `status`, `tenant_id`, `workspace_id`). Langfuse, Phoenix, and Arize require additional semantic mapping (trace, span, generation, observation, score) — deferred to **OBS-VENDOR-8**.

**Note:** INTEGRATIONS-2C delivered contract stubs for all `observability_backend` slugs including Elasticsearch; **OBS-VENDOR-4A** implements the **production transport** and end-to-end delivery — not the contract scaffold alone.

### OBS-VENDOR-6 — batching decision

**Deferred for initial rollout:** batching is not required for OBS-VENDOR-4A/5/7. Document explicit non-batching in transport; revisit batching in a follow-up row if throughput requires it.

### Relationship to OBS-EXPORT-5

| OBS-EXPORT-5 sub-deliverable | OBS-VENDOR coverage |
|------------------------------|---------------------|
| Contract adapters (INTEGRATIONS-2C) | **Done** — stubs exist; production I/O pending |
| Production vendor transports | **OBS-VENDOR-4A**, **OBS-VENDOR-6** |
| Operator / bootstrap wiring | **OBS-VENDOR-2** **Done**; **OBS-VENDOR-5** **Done** |
| Production export end-to-end | **OBS-VENDOR-7** |
| LLM-trace semantic vendors | **OBS-VENDOR-8** (later) |

---

## Phase OBS-PROBLEM — Plugin-extensible problem/error signal contract (Planned)

**Purpose:** Define a vendor-neutral platform contract for problem/error/issue signals before implementing Sentry or other issue-monitoring providers.

**Why:** Event/log timeline answers “what happened”. Problem/error signal answers “what broke and needs attention”. The LKW proof path already validates normal event/log timeline export through the observability spine; the missing platform proof is problem classification, safe error context, plugin-defined application/agent-specific problem metadata, reference-only problem artifacts, and future vendor mapping (especially Sentry-style issue triage).

**Relationship to existing observability:** Problem signals must flow through the same observability spine:

```text
ObservabilityExportEnvelope
→ ObservabilityExportPolicy
→ ObservabilityVendorIntegrationContract.export()
→ vendor deliver_payload()
```

**Vendor boundary:** Sentry is a later vendor projection for issue triage. Sentry must not define the platform problem model. LKW, agents, runtime loops, and application code must not call Sentry SDK/API directly.

**Plugin extensibility:** The problem signal must support typed application/agent/plugin-specific extensions. Platform owns the core problem/error signal contract; application/agent/plugin code may define typed custom extensions; policy owns sanitization and redaction; vendor providers own delivery/projection.

**Reuse / alignment:** Custom problem metadata should reuse or explicitly align with:

- **`ApplicationObservabilityAttributes`** — typed, namespaced, declared custom fields
- **`SanitizedApplicationObservabilityAttributes`** — policy-sanitized export surface
- **`ObservabilityArtifactReference`** — reference-only artifact metadata

Custom fields must be typed, namespaced, declared, and sanitized. No arbitrary raw `dict` payloads.

**Core platform fields (non-final candidate list):**

| Field | Role |
|-------|------|
| `problem_id` | Stable problem/issue identity |
| `problem_kind` | Classification (e.g. exception, policy_violation, tool_failure) |
| `severity` | Operator triage priority |
| `source_layer` | Tier/layer origin (runtime, agent, application) |
| `source_component` | Component within layer |
| `status` | Problem lifecycle status |
| `safe_message` | Redacted human-readable summary |
| `error_code` | Stable machine-readable code |
| `exception_type` | Exception class name when applicable |
| `run_id` | Correlation to run |
| `task_id` | Correlation to task |
| `event_id` | Correlation to spine event |
| `agent_id` | Originating agent |
| `tool_id` | Originating tool when applicable |
| `capability` | Capability context |
| `correlation_id` | Cross-signal correlation |
| `occurred_at` | Problem timestamp |

**Extension fields:** Applications/agents may define typed custom attributes through inheritance, for example:

- `LkwProblemAttributes(ApplicationObservabilityAttributes)`
- `AgentProblemAttributes(ApplicationObservabilityAttributes)`

**Artifact refs:** Problem artifacts must be reference-only using **`ObservabilityArtifactReference`** or a directly compatible model (`artifact_ref`, `sha256`, `safe_relative_path`, `schema_id`). No raw artifact bodies in problem signals.

**Redaction / safety — explicitly forbidden in problem signals and export:**

- raw prompts
- raw queries
- raw documents
- raw chunks
- raw completions
- raw tool args
- secrets
- absolute paths
- full local paths
- arbitrary raw payload dictionaries

Policy owns sanitization before export; forbidden fields are dropped or hashed.

**LKW role:** LKW is the proof workload for controlled failure scenarios. LKW is not the owner of the problem/error integration. LKW must not create an LKW-only issue model.

### OBS-PROBLEM task register

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **OBS-PROBLEM-0** | Docs/Architecture | P1 | **Done** | Define plugin-extensible platform problem/error signal contract | Plan states core platform fields, plugin extension model, policy/redaction boundary, reference-only artifacts, vendor boundary, and LKW proof workload role |
| **OBS-PROBLEM-1** | Code | P1 | **Done** | Minimal vendor-neutral `ProblemSignal` model | `intergrax/runtime/observability/problem_signal.py` — `PlatformProblemSignal`; plugin-extensible taxonomy is string-based (well-known constants, not closed enums); typed app/agent attributes reuse `ApplicationObservabilityAttributes`; artifacts reuse `ObservabilityArtifactReference`; export mapping in OBS-PROBLEM-2 |
| **OBS-PROBLEM-2** | Code | P1 | **Done** | Map problem signals through observability export policy/envelope | `ExportRecordKind.PROBLEM_SIGNAL` added; `intergrax/runtime/observability/problem_export.py` — minimal `PlatformProblemSignal` → `ObservabilityExportEnvelope` mapping (`problem_kind`, `problem_severity`, `problem_error_code` only); plugin taxonomy remains string-based and open; `application_attributes` sanitized through existing export policy; primary artifact ref only mapped (multi-artifact full projection deferred); `agent_attributes` mapping deferred; `safe_message` export deferred; no Sentry/vendor/LKW proof |
| **LKW-PF-ERR-1** | Test/Docs | P1 | **Done** | Controlled LKW failure proof for platform problem signals | Added platform-reusable `problem_reporter.py` helper (`ProblemReportContext`, `ProblemReporter`, `build_problem_signal`, `build_problem_export_envelope`, `report_problem`); LKW proof uses helper — not manual low-level signal/envelope/policy calls; proof validates controlled LKW-shaped `lkw.retrieve_failed`; proof validates ProblemSignal → Envelope → Policy/export helper; no raw content leakage; no Sentry/vendor/routing; runtime automatic failure emission deferred |
| **OBS-PROBLEM-3** | Docs | P1 | **Done** | Runtime/application problem emission boundary | Architecture canon defines allowed/discouraged emission boundaries, duplicate prevention, `RuntimeEvent` relationship, safety rules, and developer examples; boundary/contract only — no Sentry, Elastic, OTLP, routing/fanout, automatic exception catching, `ObservabilityEmitter.emit_problem`, or LKW endpoint wiring |
| **OBS-ROUTING-0** | Code/Docs | P1 | **Done** | Routing/fanout design for `problem_signal` | Vendor-neutral `FanoutObservabilityExporter` operates on policy-safe `ObservabilityExportEnvelope`; route filters for `record_kind` / `problem_kind` / `problem_severity` / `problem_error_code`; per-route failure isolation; `export_with_result(...)` for explicit fanout diagnostics; `export(...)` remains protocol-compatible; no Sentry/Elastic/OTLP provider; no runtime auto-emission; no `ObservabilityEmitter.emit_problem`; no `RuntimeEventBus` subscriber; no LKW endpoint wiring; no operator bootstrap |
| **OBS-VENDOR-PROBLEM-0** | Code | P1 | **Done** | Shared problem signal fields in observability vendor contract | `ObservabilityVendorSignal.PROBLEMS` added; `ObservabilityVendorPayload` preserves `problem_kind` / `problem_severity` / `problem_error_code`; `problem_signal` maps to `PROBLEMS`; Sentry and Elasticsearch declare `PROBLEMS` in `supported_signals` through the same base contract; no Sentry SDK; no Elasticsearch transport changes; no runtime auto-emission; no LKW wiring; no operator bootstrap |
| **OBS-SENTRY-1** | Code | P2 | **Done** | Sentry provider maps sanitized problem signals to Sentry issues | Sentry SDK-backed provider transport added under observability backend provider package; `ObservabilityVendorPayload(PROBLEMS)` maps to Sentry issue-shaped event; `sentry_sdk` imported lazily only in provider transport/client; runtime/LKW/agents do not call Sentry SDK; no LKW proof/docker compose/operator bootstrap yet |

---

## Phase OBS-SENTRY — Sentry error-monitoring integration proof (Done — LKW wiring)

**OBS-SENTRY-1 status:** **Done** — platform-level Sentry SDK-backed provider transport; `ObservabilityVendorPayload(PROBLEMS)` → Sentry issue-shaped event via `ObservabilityVendorIntegrationContract`; lazy/provider-owned `sentry_sdk` import; registry construction remains SDK-free.

**LKW-OBS-SENTRY-0 status:** **Done** — LKW controlled problem proof through platform operator wiring (`SentryExportOperatorConfig`, `sentry_export_wiring`), proof helper scripts, and `SENTRY_OBSERVABILITY.md`.

**LKW-OBS-SENTRY-1 status:** **Done** — repo-owned local Sentry Docker Compose proof stack (trimmed self-hosted **24.8.0** services, UI `http://127.0.0.1:9000`, bootstrap local DSN into `docker/sentry-proof/generated.env`), LKW app-level proof endpoint (`POST /v1/local_workspace/proof/sentry-error`), proof helper calls LKW over HTTP.

**LKW-OBS-SENTRY-1F status:** **Done** — all-in-one startup fixed: `sentry.services.yml` internal fragment (no `docker-compose.*.yml` double-discovery), `run-local-docker-all.sh`, `sentry-upgrade` lifecycle, local-proof `SENTRY_SECRET_KEY`, atomic `generated.env`, canonical one-script docs.

**OBS-SENTRY live proof status:**

**Done (controlled proof path)** — LKW can emit a controlled platform problem signal through the shared observability vendor contract into **local Docker Sentry** using operator configuration and the Sentry overlay. Canonical proof does not require external SaaS DSN. Deterministic unit tests and the proof helper validate the shared export path.

**Remaining out of scope (production / hardening):**

- production alert policy
- production Sentry org/project provisioning
- production secret management
- production auth/compliance posture
- global exception catching
- PII/default user capture
- stacktrace/raw exception capture unless separately approved

**LKW relationship:** LKW is the proof workload ([`applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`](../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) **LKW-OBS-SENTRY-1**); platform owns the provider integration and operator wiring.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.9, §11 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL observability rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-5.3 | §5 Policy | Governance health dashboard (GOV-PROD.1) | P4 | **Done** |
| AUDIT-IDEAL-21.1 | §21 Observability | Causal diagnostics beyond trace bridge (ops tooling) | P2 | **Done** |
| AUDIT-IDEAL-21.2 | §21 Observability | Quality / governance / cost health dashboard contracts | P2 | **Done** |
| AUDIT-IDEAL-21.3 | §21 Observability | Unified product observability dashboard | P4 | **Done** |
| AUDIT-IDEAL-30.2 | §30 Ops | Real deploy SLO window evidence (prod `W_OPS_RELEASE_CYCLES`) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---
