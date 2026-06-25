# Unified Execution Runtime — Implementation Plan

**Architecture (1:1):** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). UER owns runtime policy resolution, shared contract placement, output profile resolution, compression-level selection, and safety bypass enforcement.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (UNIFIED_EXECUTION_RUNTIME plan).

- **Implement / audit default:** §6.1 UAEP maintenance · R-Policy / SEC / COST open rows · phase satellites on demand
- **Token Optimization:** read feature pair + rows `TOKEN-UER-1` / `TOKEN-UER-2`; do not read unrelated closed UAEP queues.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md`](../guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/UNIFIED_EXECUTION_RUNTIME_appendices.md`](plan/UNIFIED_EXECUTION_RUNTIME_appendices.md) | appendices |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_06_closed_queues.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_06_closed_queues.md) | 06 closed queues |
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_audit_history.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_audit_history.md) | audit history |
| [`plan/satellites/UNIFIED_EXECUTION_RUNTIME_embedded_detail.md`](plan/satellites/UNIFIED_EXECUTION_RUNTIME_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** file per session unless RESUME cites more.

---

## Phase TOKEN-UER — Token Optimization runtime policy foundation (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P1 after docs sync; first implementation slice for Token Optimization  
**Delivery rule:** one `TOKEN-UER-*` row per PR; do not wire CE/TOOLS/MEMORY behavior before shared contracts land.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-UER-1** | Code | P1 | Planned | Shared package `intergrax/runtime/token_optimization/` with contracts, protected-region validator, compression receipts, and contract check script | Contracts import cleanly; no CE/TOOLS/MEMORY hot-path imports; protected regions preserve code/paths/URLs/API names/env vars/enums/hashes/dates/errors; receipts hash original/optimized content and record token savings; `uv run pytest tests/unit/runtime/token_optimization/ -q`; `uv run python scripts/check_token_optimization_contracts.py` |
| **TOKEN-UER-2** | Code | P1 | Planned | `OutputPolicyResolver` and runtime output profiles (`minimal`, `terse`, `standard`, `full`, `audit`, `machine_receipt`, `debug_verbose`) | Output profile resolved by runtime policy, not prompt-only wording; structured outputs and high-risk contexts can force clarity/full mode; no model-specific prompt hacks; `uv run python scripts/check_output_policy_wiring.py` |

**Explicit exclusions:** no `ToolSchemaOptimizer`, no `ContextPackOptimizer`, no `MemorySummaryCompressor`, no adaptive policy auto-apply, no `docs/plan/TOKEN_OPTIMIZATION.md`.

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.2–3.3, §23–§24 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

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

### 6.1av Harness implementation queue — UAEP audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `UNIFIED_EXECUTION_RUNTIME` layers 4, 5, 8, 23–24 · [`../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md`](../audit_results/2026-06-19/UNIFIED_EXECUTION_RUNTIME.md) · prior: [`../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md`](../audit_results/2026-06-18/UNIFIED_EXECUTION_RUNTIME.md)  
**Priority ladder:** **Band 1** (§6.1) — incremental after gate maintenance; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **UAEP-AUDIT-01** | Code | P2 | **Done** | `tenant_id` on `RuntimeEvent` in `UAEPExecutor._emit`, `TraceEmittingMiddleware`, and any orphan emitters | §42.44.2; regression gate on event tenant propagation |
| 2 | **UAEP-MAINT-02** | Code | P3 | **Done** | Dedup `STEP_COMPLETED` — canonical emitter in `HarnessKernel`; adjust `TraceEmittingMiddleware` to avoid duplicate journal entries | Single `STEP_COMPLETED` per step boundary in unified run journal |
| 3 | **UAEP-MAINT-03** | Docs | P3 | **Done** | Security middleware layout diagram in `AGENT_CREATION_GUIDE.md` Appendix H (`runtime/architecture/` + Tier-3 `*_wiring.py` map) | No new mechanisms; author onboarding clarity |
| 4 | **UAEP-MAINT-04** | Test | P3 | **Done** | Regression gate: at most one `STEP_COMPLETED` per step boundary (`HarnessKernel` canonical; middleware must not duplicate) | `test_kernel_emits_single_step_completed_per_step` + `test_trace_middleware_does_not_emit_step_completed_on_after_step`; gate green |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** `EscalationRouter` SUPERVISOR_AGENT target (§42.38 lab-minimal — deferred); FLOW-8 product host; GOV-PROD.1 — [§6.3](../plan/PLATFORM_FOUNDATION.md#63-end-of-plan--deferred-product-work-only).

---

## Cross-domain ORCH/flow registers removed

See [`ORCHESTRATION.md`](ORCHESTRATION.md) · [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md).

---
