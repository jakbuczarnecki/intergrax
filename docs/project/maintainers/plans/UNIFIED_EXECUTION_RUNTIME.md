# Unified Execution Runtime - Implementation Plan

**Architecture (1:1):** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

**Cross-feature - Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md). UER owns runtime policy resolution, shared contract placement, output profile resolution, compression-level selection, and safety bypass enforcement.

**Meta-architecture (frozen):** [`UNIFIED_EXECUTION_ARCHITECTURE.md`](../../architecture/UNIFIED_EXECUTION_ARCHITECTURE.md) - semantic authority over UER target model. UER plan rows must not contradict Execution-centric identity, neutral Execution Boundary, or strategy resolution semantics.

### Architecture sync - UE-DOC-0.4 (2026-08-25)

**Target model (from rewritten UER hub):**

- Fundamental unit: **Execution** (`TaskId` → `RunId` → `AttemptId` → `ExecutionId` → `EventId`)
- Public entry: `execution.execute(request=..., output_type=...)` - no public engine/mode selection
- Strategies: inference · agentic (AgentEngine → UAEP) · orchestration (Nexus → child Executions)
- UAEP is **agent-specific**; Nexus is **not** required for direct inference or ordinary agentic execution
- UER coordinates Governance/Budget/Observability/Checkpoint; does not own their authorities

**Known implementation gaps (CURRENT):** canonical UER foundations exist (`ExecutionId`, `ExecutionBoundary`, `StrategyExecutionRouter`); full entry-path adoption, five-ID background propagation, subtree cancellation, and budget/authority convergence **PARTIAL**; `UnifiedTaskRunner` routes through Nexus on many paths; agent-centric `GraphExecutor`; incomplete hierarchical budget dimensions on some paths.

**High-level migration order:** see UER hub [Implementation readiness §5](../../architecture/UNIFIED_EXECUTION_RUNTIME.md#5-migration-order-high-level). Detailed code mapping: [`UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md`](../../architecture/UNIFIED_EXECUTION_IMPLEMENTATION_MAP.md) (**UE-DOC-0.9**).

### Architecture sync - UE-DOC-0.9 (2026-08-26)

**Status:** canonical implementation map delivered - **no runtime implementation in this slice**. Use map for UE-1+ slice derivation; do not reopen UEA semantics.

**Plan debt:** substantial row restructuring against Execution-centric slices remains in plan rows - align incrementally per map waves §26.

### Implementation gate - UE-DOC-0.10 (2026-08-26)

**Status:** final pre-runtime consistency audit complete - [`UNIFIED_EXECUTION_IMPLEMENTATION_READINESS.md`](../../architecture/UNIFIED_EXECUTION_IMPLEMENTATION_READINESS.md). **IMPLEMENTATION GATE: PASS** (baseline `fc7c76c999e3d49d0532c4bdd07941c688e2553c`). UE-1+ runtime work may proceed; first slice UE-1A.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (UNIFIED_EXECUTION_RUNTIME plan).

- **Implement / audit default:** §6.1 UAEP maintenance · R-Policy / SEC / COST open rows · phase satellites on demand
- **Token Optimization:** read feature pair + rows `TOKEN-UER-1` / `TOKEN-UER-2`; do not read unrelated closed UAEP queues.
- **Use** `Read` with offset/limit - open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) read-scope block only.
- **Platform audit:** [`docs/audit_results/AUDIT_PROTOCOL.md`](../../audit_results/AUDIT_PROTOCOL.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/UNIFIED_EXECUTION_RUNTIME_appendices.md`](plan/UNIFIED_EXECUTION_RUNTIME_appendices.md) | appendices |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

### Protocol v2 remediation - STRATEGIC_HARNESS_MODEL (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-INIT.

| Block | Status | Findings | Scope |
|-------|--------|----------|-------|
| **SHM-FIX-A** | ACCEPTED / PLANNED | [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-02`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-03`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-04`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Equivalent governed step boundary for normal and resume; fail closed without kernel context |
| **SHM-FIX-B** | ACCEPTED / PLANNED | [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-08`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-09`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | AttemptId continuity across bridges; typed RuntimeExecutionContext at ACP boundary |
| **SHM-FIX-C** | ACCEPTED / PLANNED | [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md), [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-07`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Structural production host/profile; remove product-shaped result promotion from core runtime |
| **SHM-FIX-D** | ACCEPTED / PLANNED | [`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10`](../../audit_results/2026-08-18/STRATEGIC_HARNESS_MODEL.md) | Recertify mandatory-path and maturity claims after A–C verification |

**Remediation rules:**

- Revalidate each finding against then-current `development` HEAD before implementation.
- If parallel work already fixed a finding, do not duplicate - independently verify before lifecycle advancement.
- Implementer may advance finding status only through **IMPLEMENTED**; independent verification required for **VERIFIED**; **CLOSED** only after campaign/remediation rollup confirms closure ([`AUDIT_REMEDIATION_PROTOCOL.md`](../../audit_results/AUDIT_REMEDIATION_PROTOCOL.md)).

---

### Protocol v2.2 remediation - IDENTITY_TRUST (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/IDENTITY_TRUST.md`](../../audit_results/2026-08-18/IDENTITY_TRUST.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-IDENTITY-TRUST-PERSIST.

#### IDT-FIX-D - Execution identity closure

**Status:** `ACCEPTED / PLANNED`
**Source:** [`AUDIT-20260818-IDENTITY_TRUST-05`](../../audit_results/2026-08-18/IDENTITY_TRUST.md)

**Acceptance criteria:**

- No production HITL/lifecycle/runtime-event path substitutes `TaskId` for `RunId`.
- Canonical `RunId` + `AttemptId` obtained from `ActiveExecutionIdentity` or explicit canonical runtime context.
- APPROVE / REJECT / ESCALATE parity across HITL verdict branches.
- Hooks/events carry correct execution identity.
- Contract/gate tests cover all three HITL verdict branches and lifecycle provenance.

**Remediation rules:** same as SHM-FIX block above.

---

### Protocol v2.2 remediation - EXECUTION_RUNTIME (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/EXECUTION_RUNTIME.md`](../../audit_results/2026-08-18/EXECUTION_RUNTIME.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-BATCH-PERSIST-2.

| Block | Status | Findings | Acceptance intent |
|-------|--------|----------|-------------------|
| **UER-FIX-A** | ACCEPTED / PLANNED | UER-01 | Canonical runtime policy propagation into direct ACP/UAEP/kernel |
| **UER-FIX-B** | ACCEPTED / PLANNED | UER-02 | Atomic step commit; no state contradicting `outcome_applied=false` |
| **UER-FIX-C** | ACCEPTED / PLANNED | UER-03 | Resume without retry preserves AttemptId; checkpoint carries identity |
| **UER-FIX-D** | ACCEPTED / PLANNED | UER-04 | Normal runtime exceptions → typed FAILED terminal results |
| **UER-FIX-E** | ACCEPTED / PLANNED | UER-05, UER-06 | Cooperative ACP cancellation + checkpoint invalidation |

---

### Protocol v2 remediation - SECURITY_BOUNDARIES (2026-08-18)

**Audit:** [`docs/audit_results/2026-08-18/SECURITY_BOUNDARIES.md`](../../audit_results/2026-08-18/SECURITY_BOUNDARIES.md) · campaign [`README`](../../audit_results/2026-08-18/README.md)
**Status:** ACCEPTED findings - **PLANNED** remediation only. **Not implemented** by audit persistence task AUDIT-20260818-SECURITY-BOUNDARIES-PERSIST.

#### SEC-DATA-PROTECTION-INTEGRITY - encryption fail-closed and cryptographic honesty

**Priority:** P0
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-SECURITY_BOUNDARIES-03`](../../audit_results/2026-08-18/SECURITY_BOUNDARIES.md)

**Acceptance intent:**

- Configured secure backend resolution failure blocks or fails startup when cryptographic protection is required.
- No silent downgrade to `HarnessEnvelopeEncryptor` / Base64 envelope on product paths.
- Lab/demo transforms explicitly labeled non-cryptographic; preserve `SecretsStorePayloadEncryptor` provider-neutral shape.

#### SEC-AUDIT-AUTHORITY-INTEGRITY - durable immutable audit authority

**Priority:** P1
**Status:** `ACCEPTED / PLANNED`
**Findings:** [`AUDIT-20260818-SECURITY_BOUNDARIES-06`](../../audit_results/2026-08-18/SECURITY_BOUNDARIES.md)

**Acceptance intent:**

- Separate in-memory audit simulation from production audit authority.
- Define `ImmutableSecurityAuditTrail` persistence port; multi-region qualification requires independently durable replicas and explicit replication/tamper evidence.
- No specific cloud vendor required.

Cross-link **SEC-DEFENSE-QUALIFICATION-INTEGRITY** in [`TIER3_APPLICATION_ENVIRONMENT` plan](TIER3_APPLICATION_ENVIRONMENT.md) for signing and defense toggle wiring that requires runtime enforcement position verification.

**Remediation rules:** same as SHM-FIX block above.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_06_closed_queues.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_06_closed_queues.md) | 06 closed queues |
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_implementation_history.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_implementation_history.md) | implementation history |
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_embedded_detail.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** file per session unless RESUME cites more.

---

## Phase TOKEN-UER - Token Optimization runtime policy foundation (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../../capabilities/plan/TOKEN_OPTIMIZATION.md)
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../../capabilities/architecture/TOKEN_OPTIMIZATION.md)
**Priority:** P1 after docs sync; first implementation slice for Token Optimization  
**Delivery rule:** one `TOKEN-UER-*` row per PR; do not wire CE/TOOLS/MEMORY behavior before shared contracts land.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-UER-1** | Code | P1 | Planned | Shared package `intergrax/runtime/token_optimization` with contracts, protected-region validator, compression receipts, and contract check script | Contracts import cleanly; no CE/TOOLS/MEMORY hot-path imports; protected regions preserve code/paths/URLs/API names/env vars/enums/hashes/dates/errors; receipts hash original/optimized content and record token savings; `uv run pytest tests/unit/runtime/token_optimization/ -q`; `uv run python scripts/check_token_optimization_contracts.py` |
| **TOKEN-UER-2** | Code | P1 | Planned | `OutputPolicyResolver` and runtime output profiles (`minimal`, `terse`, `standard`, `full`, `audit`, `machine_receipt`, `debug_verbose`) | Output profile resolved by runtime policy, not prompt-only wording; structured outputs and high-risk contexts can force clarity/full mode; no model-specific prompt hacks; `uv run python scripts/check_output_policy_wiring.py` |

**TOKEN-2 note:** `OutputPolicyResolver` in `intergrax/runtime/token_optimization/output_policy.py` is an internal policy-resolution helper only. It is not wired into model calls, prompt assembly, or runtime execution yet.

**Explicit exclusions:** no `ToolSchemaOptimizer`, no `ContextPackOptimizer`, no `MemorySummaryCompressor`, no adaptive policy auto-apply, no `docs/project/maintainers/plans/TOKEN_OPTIMIZATION.md`.

---

## Phase AUDIT-IDEAL - Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.2–3.3, §23–§24 · baseline **32/32 L3**
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** - incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-4.1 | §4 Identity | Cryptographic signing / audit-protect for critical actions | P2 | **Done** |
| AUDIT-IDEAL-4.2 | §4 Identity | Hard tenant storage isolation (Postgres multi-tenant RFC → ship) | P1 | **Done** |
| AUDIT-IDEAL-5.1 | §5 Policy | Pre-output policy hooks on all LLM response paths | P1 | **Done** |
| AUDIT-IDEAL-5.2 | §5 Policy | Compliance profile templates per regulated domain class | P2 | **Done** |
| AUDIT-IDEAL-23.1 | §23 Security | Immutable multi-region security audit trail | P2 | **Done** |
| AUDIT-IDEAL-23.2 | §23 Security | Retrieval poisoning + tool injection live on product hosts | P1 | **Done** |
| AUDIT-IDEAL-24.1 | §24 Cost | Cost forecasting from historical run patterns | P2 | **Done** |
| AUDIT-IDEAL-24.2 | §24 Cost | Automated cost optimization recommendations (AHI) | P2 | **Done** |
| AUDIT-IDEAL-24.3 | §24 Cost | CPU/memory/concurrency quotas with tenant fairness | P2 | **Done** |
| UAEP-AUDIT-01 | §8 Runtime | Populate `tenant_id` on all `RuntimeEvent` emitters (UAEP + trace middleware) | P2 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

### 6.1av Harness implementation queue - UAEP audit maintenance

**Source:** Interactive layer audit (2026-06-19) - `UNIFIED_EXECUTION_RUNTIME` layers 4, 5, 8, 23–24 · [`../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md`](../../../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md) · prior: [`../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md`](../../../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md)
**Priority ladder:** **Band 1** (§6.1) - incremental after gate maintenance; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **UAEP-AUDIT-01** | Code | P2 | **Done** | `tenant_id` on `RuntimeEvent` in `UAEPExecutor._emit`, `TraceEmittingMiddleware`, and any orphan emitters | §42.44.2; regression gate on event tenant propagation |
| 2 | **UAEP-MAINT-02** | Code | P3 | **Done** | Dedup `STEP_COMPLETED` - canonical emitter in `HarnessKernel`; adjust `TraceEmittingMiddleware` to avoid duplicate journal entries | Single `STEP_COMPLETED` per step boundary in unified run journal |
| 3 | **UAEP-MAINT-03** | Docs | P3 | **Done** | Security middleware layout diagram in `AGENT_CREATION_GUIDE.md` Appendix H (`runtime/architecture` + Tier-3 `*_wiring.py` map) | No new mechanisms; author onboarding clarity |
| 4 | **UAEP-MAINT-04** | Test | P3 | **Done** | Regression gate: at most one `STEP_COMPLETED` per step boundary (`HarnessKernel` canonical; middleware must not duplicate) | `test_kernel_emits_single_step_completed_per_step` + `test_trace_middleware_does_not_emit_step_completed_on_after_step`; gate green |

**Suggested PR order:** none - §6.1av queue closed (2026-06-19).

**Explicitly excluded:** `EscalationRouter` SUPERVISOR_AGENT target (§42.38 lab-minimal - deferred); FLOW-8 product host; GOV-PROD.1 - [§6.3](PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

---

## Cross-domain ORCH/flow registers removed

See [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md).

---
