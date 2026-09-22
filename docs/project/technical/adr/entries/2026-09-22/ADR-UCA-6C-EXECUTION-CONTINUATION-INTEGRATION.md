# ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION: Execution-owned continuation-aware bound tool invocation

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-09-22 |
| **Reconciled** | 2026-09-22 (UCA-6C-ADR1-R1 — EBCI boundary reconciliation with ADR-HARNESS-003) |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | ADR-GR-5-001 · ADR-HARNESS-001 · ADR-HARNESS-003 · ADR-PLATFORM-PLUGIN-001 · UCA-6C-ARCH-R2 |

## 1. Context

UCA-6C qualified `code.exec` runs inside an active `ExecutionRuntime` via `CodeCraftQualifiedCapabilityExecutionHandler` → `CodeCraftBoundCapabilityExecutionPort` → `ExecutionBoundCatalogToolInvoker` (domain L2) → L3 `NexusExecutionBoundCatalogToolInvoker` → `RuntimeToolInvoker`. When declarative policy enforces `REQUIRE_HITL`, the direct catalog path raises `DeclarativePolicyHitlRequiredError` without the canonical bridge used by `catalog_dispatch` / `tool_loop`. `CanonicalExecutionRuntimeAdapter` wraps that into `CanonicalExecutionInvocationFailed`, conflating **REQUIRE_HUMAN** with **FAILED**.

The platform already owns declarative-tool HITL (bridge → pending → `DeclarativeHitlApprovalGrant` → resume → policy re-eval → MSE → backend) and `ExecutionContinuationPort` for same-Execution pause/resume. The gap is **Execution-owned continuation integration behind the existing catalog L2 seam**, not a new universal consumer-facing invoker.

**ADR-HARNESS-003** freezes L1/L2/L3 ownership, domain-owned L2 contracts, Nexus-internal L3, per-call immutable invocation scope on L2, and `ExecutionIdentityBinding` as EE-internal only. This ADR is reconciled to that freeze (UCA-6C-ADR1-R1).

## 2. Problem

Specialized consumers (UCA-6C CodeCraft; future siblings) need mandatory ToolRuntime, canonical HITL for the exact protected invocation, and stable domain L2 typing — without AW/UCA pre-approval transport, without leaking Nexus, RuntimeState, continuation ports, or canonical HITL grants to consumers.

Current L3 `NexusExecutionBoundCatalogToolInvoker` bypasses `raise_hitl_pause_from_tool_invocation` and enables parallel AW pre-approval via `ToolInvocationGovernanceApprovalEvidence` with non-bridge `uca6c-scope:*` ids.

## 3. Existing architecture

See ADR-HARNESS-003 and UCA-6C-ARCH-R2 proof. Frozen consumer/catalog L2: `ExecutionBoundCatalogToolInvoker`. Bridged declarative dispatch: `ExecutionBoundDeclarativeToolInvoker` / `catalog_dispatch` (wrong shape for typed `code.exec` catalog input). Canonical continuation: `ExecutionContinuationPort`. HITL: `declarative_hitl` types and bridge-owned `invocation_scope_id` (`dhr_*`).

## 4. Constraints

Nexus internal only; frozen `ExecutionContinuationPort` and identity minting; no second HITL owner or grant store; ToolRuntime mandatory; no new universal L2 megacontract; no `DeclarativeHitlApprovalGrant` on domain L2 requests; no `uca6c-scope:*`; no `Any` on new seams.

## 5. Frozen owners

| Concern | Owner |
|---------|-------|
| Execution lifecycle / continuation | Execution Engine |
| WHETHER (policy) | Governance |
| Human resolution / canonical grant | Canonical declarative HITL + EE continuation state |
| Physical tool invoke | ToolRuntime (`RuntimeToolInvoker`, L3) |
| UCA coordination | UCA (not pause/resume, not grant transport) |
| AW worker consumer | AW (not `ExecutionContinuation.resume`) |
| Craft/session artifact eligibility | CodeCraft local gate (not platform HITL) |

## 6. Reconcile with ADR-HARNESS-003

**Decision (R1):** Keep **`ExecutionBoundCatalogToolInvoker`** as the canonical **domain/platform L2** consumer surface for catalog `code.exec`. Move continuation awareness to an **EE-owned L3 continuation host** that delegates to the **same canonical bridge path** as `catalog_dispatch` (not a second orchestration stack).

**Reject** introducing `ExecutionBoundContinuationAwareToolInvocationPort` (or any equivalent) as a **new universal domain L2** consumer API — that would violate HARNESS-003 (“common invariant ≠ common megacontract”).

Optional **public Execution composition** contracts (B2) are **not required** for UCA-6C: the composition owner is already `ExecutionRuntime` / qualified-capability dispatch; the seam is **L3-internal** unless a future ADR defines a host-only composition port that is explicitly **not** a tool consumer API.

## 7. Options considered (post-reconciliation)

| Option | Description | HARNESS-003 | Verdict |
|--------|-------------|-------------|---------|
| **B1** | Existing L2 + EE-internal continuation-aware L3 adapter/host → bridge → ToolRuntime | Compatible | **Accept** |
| **B2** | B1 + public EE composition seam (host-only, not CodeCraft tool API) | Compatible if host-only | **Defer** — not needed for UCA-6C R6 |
| **B3** | New generic EE consumer-facing continuation invoker (L2) | Conflicts with frozen L2 taxonomy | **Reject** |
| A | L2 as-is without bridge integration | No canonical HITL | Reject |
| C | UCA/CodeCraft local bridge + AW pre-approval | Second HITL transport | Reject |
| D | Public Nexus bridge | Nexus public | Reject |
| E | Hidden binding mutation | Mutable bind debt | Reject |
| F | TIGAE pre-approval only | Duplicates canonical HITL | Reject |

## 8. Decision

**Option B1** — Execution Engine owns continuation-aware integration **behind** `ExecutionBoundCatalogToolInvoker`:

```text
CodeCraft / UCA composition
    → ExecutionBoundCatalogToolInvoker                    [L2 — frozen consumer surface]
    → EE continuation-aware catalog host (L3 adapter)   [new R6 — EE-internal only]
    → canonical declarative HITL bridge (catalog_dispatch semantics)
    → ExecutionContinuationPort (on REQUIRE_HITL — EE only)
    → RuntimeToolInvoker                                  [L3 — ToolRuntime]
```

- **No** new universal L2 port for CodeCraft/UCA.
- **No** consumer-visible `CONTINUATION_REQUIRED` outcome on L2; continuation is EE-internal (see §9).
- R6 implements the L3 host and rewires default catalog L3 implementation; does **not** expand L2 into a megacontract.

## 9. L2 vs L3 responsibilities

### 9.1 L2 (`ExecutionBoundCatalogToolInvoker`)

- Typed domain invocation (`tool_id`, `BaseModel` input, **per-call immutable catalog execution scope**: `tenant_id`, `run_id`, `task_id`, `agent_id`, `step_id`, correlation/idempotency as today).
- **Does not** mint `RunId` / `AttemptId` / `ExecutionId`; **does not** expose `ExecutionIdentityBinding` or full four-ID EE binding unless a **separate frozen work-unit contract** already requires it (catalog plane does not).
- **Does not** manage continuation, pause/resume, or HITL stores.
- **Does not** import Nexus; **does not** accept `DeclarativeHitlApprovalGrant`.
- **Target (R6):** remove `ToolInvocationGovernanceApprovalEvidence` from `ExecutionBoundCatalogToolInvokeRequest` (see §13).

**Sync semantics:** L2 `invoke()` remains `ToolExecutionResult` — no typed suspend outcome. On `REQUIRE_HITL`, the **EE L3 host** (invoked from within active `ExecutionRuntime`) integrates with the canonical bridge and `ExecutionContinuationPort`: active Execution pauses, stack unwinds; **reconstruction owner is EE**, which re-enters the L3 host after resume with grant read from continuation state — not from the consumer request. CodeCraft/UCA **must not** own pause, resume, or continuation advancement.

Whether the outer qualified-capability handler appears to “block until tool completes” is an **EE orchestration** concern; consumers **must not** observe `CONTINUATION_REQUIRED` as an L2 result type.

### 9.2 L3 (EE continuation-aware catalog host)

**May:** project admitted execution to `RuntimeState`; call Nexus **internally**; map `REQUIRE_HITL` to continuation lifecycle; persist/replay operation payload via existing checkpoint/continuation/task state; materialize `DeclarativeHitlApprovalGrant` internally on resume; reuse `catalog_dispatch` bridge path.

**Must not:** mint identity; be public API; create a new HITL owner or approval store; expose Nexus types to consumers.

**Classification:** `L3 / EE internal orchestration seam` — **not** `public Execution composition contract` for UCA-6C (unless B2 is opened in a future ADR for non-tool hosts).

## 10. Identity and grants (frozen targets)

| Item | Consumer / L2 | EE / L3 |
|------|---------------|---------|
| `ExecutionIdentityBinding` | **No** | Internal |
| `AttemptId` / `ExecutionId` on catalog request | **No** (unless future frozen work unit) | Internal admission |
| Catalog scope fields on request | **Yes** (per HARNESS-003 Model A) | Validate vs admission |
| `DeclarativeHitlApprovalGrant` | **No** | **Yes** (resume path internal) |
| `invocation_scope_id` minting | **No** | Bridge `dhr_*` only |
| `uca6c-scope:*` | **Remove** from architecture | **Remove** |

`bind_execution_identity`: **do not** expand in R6; mark transitional debt per HARNESS-003 M6 target (deprecate → remove).

## 11. Sequence flow (canonical)

```text
AW → qualified capability dispatch [EE]
  → CodeCraft handler → ExecutionBoundCatalogToolInvoker [L2]
  → EE continuation-aware catalog host [L3]
  → bridge → ToolRuntime → Governance → REQUIRE_HITL
  → EE: ExecutionContinuationPort pause (same Execution)
  → human approves → EE restores grant internally
  → EE reconstructs invocation → bridge → ToolRuntime → MSE → backend
  → L2 returns ToolExecutionResult (success/failure/rejection — not continuation token)
```

Consumer does **not** transport approval. Post-approve: policy re-eval, fresh MSE, then backend (no reuse of stale MSE allow).

## 12. Failure semantics

`REQUIRE_HITL` ≠ `FAILED` / `REJECTED` / `CapabilityGap` at the execution adapter boundary. After `REQUIRE_HUMAN`: no rediscovery, reacquisition, requalification, or re-binding. Idempotency: reuse existing pre-effect coordination; no new idempotency mechanism.

## 13. TIGAE (`ToolInvocationGovernanceApprovalEvidence`)

**Final verdict for UCA-6C production path:** **REMOVE** — no AW/UCA ingress, no L2 generic approval carrier, no `governance_approval_evidence` on catalog invoke request after R6.

**Contract retention:** Today `ToolInvocationGovernanceApprovalEvidence` remains referenced from `WorkerQualifiedCapabilityResume` and related AW paths — **not** canonical `ExecutionContinuation.resume`. That is **out of UCA-6C canonical HITL** and must not be conflated with bridge grants. R6 removes UCA/catalog pre-approval and the grant↔DTO↔grant round-trip in `NexusExecutionBoundCatalogToolInvoker`; **full contract deletion** is a follow-on only if no independent legitimate production consumer remains after AW worker path migration (design-only in R1; do not delete in R1).

**R6 cleanup candidates (design):** `derive_qualified_capability_governance_step_id`, `derive_qualified_capability_governance_invocation_scope_id`, `validate_governance_approval_evidence_for_execution_request` when no non-preapproval use remains.

## 14. CodeCraft local authorization

**KEEP** as **craft/session artifact eligibility** (`resolve_codecraft_exec_authorization`) — not platform policy approval for `code.exec`. ToolRuntime governance owns policy/MSE.

**Terminology debt:** CodeCraft `pending_hitl` where it is **not** canonical HITL — document only; no refactor in R1.

## 15. Catalog binding

`CatalogDeclarativeRunBinding.declarative_hitl_grant` — L3 internal compatibility only; not a consumer transport surface.

## 16. ADR-HARNESS-003 compatibility matrix

| HARNESS-003 invariant | Proposed UCA continuation model | Compatible? |
|-----------------------|---------------------------------|----------:|
| L1/L2/L3 separation | L2 catalog unchanged; continuation host is L3 | Yes |
| No common megacontract | No new universal L2 invoker | Yes |
| Domain-owned L2 | `ExecutionBoundCatalogToolInvoker` remains surface | Yes |
| Nexus internal | Only L3 host → Nexus | Yes |
| No identity mint on L2 | L2 validates scope only | Yes |
| Per-call immutable scope | Request fields authoritative (catalog plane) | Yes |
| RuntimeState internal projection | L3 host only | Yes |
| No lifecycle ownership on L2 | EE owns continuation | Yes |
| No retry/recovery ownership on L2 | EE strategies | Yes |
| Replaceable L2 implementation | Fake invoker without Nexus still valid | Yes |
| No mutable bind as canonical target | R6 must not worsen `bind_*`; debt noted | Yes (with debt) |

**Overall:** **PASS** for R6 architectural gate (implementation must not violate rows).

## 17. ADR-UCA-6C correction matrix (ADR1 → R1)

| ADR1 statement | Keep | Change | Remove | Reason |
|----------------|-----:|-------:|-------:|--------|
| Option B — EE owns continuation integration | ✓ | | | Aligns with HARNESS-003 when L3-scoped |
| `ExecutionBoundContinuationAwareToolInvocationPort` as public contract | | ✓ | ✓ | Becomes L3 host; not universal L2 |
| Request = tool + four-ID + optional grant | | | ✓ | Catalog L2 scope ≠ EE four-ID binding; no grant on L2 |
| Outcome `CONTINUATION_REQUIRED` on public port | | | ✓ | Continuation internal to EE |
| TIGAE RESTRICT → REMOVE | | ✓ | | Final: remove from UCA path; contract TBD for AW-only |
| CodeCraft local gate | ✓ | | | Artifact eligibility only |
| Internal `catalog_dispatch` reuse | ✓ | | | Canonical bridge |
| EE adapter drives `ExecutionContinuationPort` | ✓ | | | Unchanged owner |

## 18. Implementation phases (R6 — conceptual only)

1. **L3** EE continuation-aware catalog host (bridge-aligned).
2. Rewire default catalog implementation (replace or wrap direct `NexusExecutionBoundCatalogToolInvoker` HITL bypass).
3. UCA/CodeCraft wiring unchanged at L2 type.
4. Remove TIGAE from catalog request path and UCA pre-approval helpers (as eligible).
5. Adapter/execution-runtime mapping: `REQUIRE_HUMAN` ≠ `FAILED`.
6. Architecture regression gates (see §19) + targeted E2E.

## 19. Future architecture test targets (permanent gates)

- Domain L2 contracts must not import `declarative_hitl` grant types unless domain-HITL-specific.
- Domain L2 must not import `runtime.nexus`.
- UCA/AW must not import execution continuation for catalog HITL resume.
- UCA/AW must not create approval grants or mint `invocation_scope_id`.
- Consumer L2 must not expose full four-ID binding unless frozen work-unit contract requires it.
- L3 continuation implementation stays under `intergrax/runtime/execution/**` and `intergrax/runtime/nexus/**` composition.

## 20. R6 readiness

R6 may start when this reconciliation is accepted on GitHub audit:

| Criterion | R1 verdict |
|-----------|------------|
| HARNESS-003 compatibility | PASS |
| No universal L2 megacontract | PASS |
| Existing domain L2 preserved | PASS |
| No grant in consumer API | PASS (target) |
| No four-ID surface leak | PASS (target) |
| Nexus internal only | PASS |
| Identity / continuation frozen | PASS |
| Canonical `dhr_*` scope | PASS |
| TIGAE removal from UCA path defined | PASS |
| Exact invocation reconstruction from existing EE state (UCA-6C-ADR1-R2) | **FAIL** — see §22 |
| **R6 ready (implementation)** | **NO** — blocked until §22 gap closed or ADR decision |

## 21. R6 forbidden scope

Identity authority; `ExecutionContinuationPort` semantics; Governance policy semantics; ToolRuntime core semantics; public Nexus; new HITL subsystem; new grant store; universal L2 invoker; full HARNESS M2–M6 migration in one R6 slice.


## 22. UCA-6C-ADR1-R2 — exact invocation reconstruction (state reuse proof)

**Status:** Proposed addendum (2026-09-22). **Verdict:** existing persisted EE state is **insufficient** for UCA QCE code.exec exact resume without new durable operation materialization or EE work-reentry API (ADR decision required before R6).

### 22.1 Canonical resumable operation (UCA-6C)

**Owner level:** ToolExecutionRequest (materialized inside EE L3 from ExecutionBoundCatalogToolInvokeRequest). L2 carries the typed catalog scope + CodeExecInput; EE L3 owns bridge, continuation, and exact replay.

PendingExecutionContinuation / continuation_id identify **lifecycle + four-ID only** — not operation payload (see intergrax/contracts/execution_continuation.py).

### 22.2 QCE vs Task-graph canonical HITL

| Mechanism | Task-graph / catalog_dispatch | UCA QCE path (today) |
|-----------|--------------------------------|----------------------|
| declarative_hitl_pending durable | Task checkpoint + governance | **Absent** — no Task host |
| Tool input for replay | Plan/checkpoint or re-planned PlannedToolCall | **Ephemeral** — CodeCraftSessionManager (in-memory) + per-invoke RuntimeState |
| ExecutionContinuationPort.resume() work reentry | Task intake / graph executor resumes node | **None** — delegate is single execute() pass |
| Ingress ledger | N/A | _IngressLedgerEntry — **ingress dedup only** (in-memory) |

### 22.3 FIRST MISSING LINK

EE continuation-aware catalog host (L3, B1) **plus** durable exact-invocation datum (not present in PendingExecutionContinuation, DeclarativeHitlPendingApproval, or QCE dispatch ledger).

### 22.4 R6 gate

R6 **must not** start as “surgical wiring only” until architecture records **where** exact ToolExecutionRequest / CodeExecInput is durably owned for QCE, or accepts a new ADR for durable operation descriptor / work-reentry without violating frozen continuation contracts.


## Compliance

ADR-GR-5-001, ADR-HARNESS-001/003, ADR-PLATFORM-PLUGIN-001; UCA_6C_CANONICAL_HITL_REENTRY_PROOF.md.
