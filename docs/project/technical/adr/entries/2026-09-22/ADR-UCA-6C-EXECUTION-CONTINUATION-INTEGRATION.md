# ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION: Execution-owned continuation-aware bound tool invocation

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-09-22 |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | ADR-GR-5-001 · ADR-HARNESS-001 · ADR-HARNESS-003 · ADR-PLATFORM-PLUGIN-001 · UCA-6C-ARCH-R2 |

## 1. Context

UCA-6C qualified `code.exec` runs inside an active `ExecutionRuntime` via `CodeCraftQualifiedCapabilityExecutionHandler` → `CodeCraftBoundCapabilityExecutionPort` → public `ExecutionBoundCatalogToolInvoker` → internal `RuntimeToolInvoker`. When declarative policy enforces `REQUIRE_HITL`, the path raises `DeclarativePolicyHitlRequiredError` without the canonical bridge used by `catalog_dispatch` / `tool_loop`. `CanonicalExecutionRuntimeAdapter` wraps that into `CanonicalExecutionInvocationFailed`, conflating **REQUIRE_HUMAN** with **FAILED**.

The platform already owns a complete declarative-tool HITL loop (bridge → pending → `DeclarativeHitlApprovalGrant` → resume → policy re-eval → MSE → backend) and `ExecutionContinuationPort` for same-Execution pause/resume. The gap is a **public, Execution-owned seam** for *continuation-aware execution-bound tool invocation* inside an active Execution.

## 2. Problem

Specialized consumers (UCA-6C CodeCraft today; future siblings) need mandatory ToolRuntime, canonical HITL for the exact protected invocation, same four-ID Execution scope, and typed **CONTINUATION_REQUIRED** distinct from failure.

Current `ExecutionBoundCatalogToolInvoker` returns only `ToolExecutionResult`, bypasses `raise_hitl_pause_from_tool_invocation`, and enables parallel AW pre-approval via `ToolInvocationGovernanceApprovalEvidence` with non-bridge `uca6c-scope:*` ids.

## 3. Existing architecture

See ADR-HARNESS-003 and UCA-6C-ARCH-R2 proof. Key public contracts: `ExecutionBoundCatalogToolInvoker`, `ExecutionBoundDeclarativeToolInvoker` (bridged dispatch, wrong shape for `code.exec`), `ExecutionContinuationPort`, `declarative_hitl` types, `QualifiedCapabilityExecutionDispatchPort`.

## 4. Constraints

Nexus internal only; frozen `ExecutionContinuationPort` and identity; no second HITL; ToolRuntime mandatory; bridge-owned `invocation_scope_id` (`dhr_*`); no `Any` on new contracts.

## 5. Frozen owners

Execution Engine — lifecycle/continuation; Governance — WHETHER; canonical HITL — human resolution/grant; ToolRuntime — enforcement; UCA — coordination; AW — worker consumer; CodeCraft — artifact consumer.

## 6. Reuse existing public contracts?

**NO** end-to-end; **YES** for `ExecutionContinuationPort`, `DeclarativeHitlApprovalGrant`, bridge internals. `ExecutionBoundCatalogToolInvoker` is an insufficient outcome/HITL model.

## 7. Options considered

| Option | Description | Reuses canonical HITL | Public Nexus? | New authority? | Frozen core change? | Pluginable? | Verdict |
| ------ | ----------- | --------------------: | ------------: | -------------: | ------------------: | ----------: | ------- |
| A | Reuse `ExecutionBoundCatalogToolInvoker` as-is | No | No | No | No | Yes | Reject |
| B | Minimal EE-owned continuation-aware bound tool invocation contract | Yes | No | No | Additive | Yes | **Accept** |
| C | UCA/CodeCraft local bridge + AW pre-approval | Duplicates | No | Yes | No | No | Reject |
| D | Public Nexus bridge | Yes | Yes | No | Violates freeze | No | Reject |
| E | Hidden binding mutation | Partial | No | Hidden | No | No | Reject |
| F | TIGAE pre-approval only | No | No | Duplicates | No | No | Reject |

## 8. Decision

**Option B** — `ExecutionBoundContinuationAwareToolInvocationPort` (conceptual) under `intergrax/contracts/execution/`, EE-composed, delegating to canonical bridge + `ExecutionContinuationPort`. R6 implements via internal `catalog_dispatch` path, not a second orchestration stack.

## 9. Why chosen

Ownership correctness; proven ARCH-R2 gap; reuse canonical HITL/continuation/ToolRuntime; Nexus encapsulated; provider-neutral; exact invocation continuity; fail-closed; minimal surface.

## 10. Contract ownership

Execution Engine. Request: tool + four-ID identity + optional resume `DeclarativeHitlApprovalGrant`. Outcomes: `COMPLETED`, `CONTINUATION_REQUIRED`, `REJECTED`, `FAILED`, `UNAVAILABLE` — no Nexus exceptions on public API. Port does not own pause persistence; EE adapter drives `ExecutionContinuationPort`.

## 11. Sequence flow

```text
AW → Execution dispatch → EE handler → ExecutionBoundContinuationAwareToolInvocationPort
→ [EE: bridge → ToolRuntime → Governance → HITL → ExecutionContinuationPort]
→ CONTINUATION_REQUIRED → human → grant → resume same Execution → ToolRuntime → MSE → backend → COMPLETED
```

## 12. Failure semantics

CONTINUATION_REQUIRED ≠ FAILED/REJECTED/CapabilityGap. After REQUIRE_HUMAN: no rediscovery/reacquisition/requalification/re-binding. Post-approve: policy re-eval, then MSE, then backend.

## 13. TIGAE

**RESTRICT** then **REMOVE** — forbid AW/UCA ingress; R6 uses canonical grant only; remove from bound invoke request and pre-approval fixtures.

## 14. CodeCraft local authorization

**KEEP AS DISTINCT ARTIFACT ELIGIBILITY GATE** (`resolve_codecraft_exec_authorization`) vs ToolRuntime governance (policy/MSE). Future cleanup: do not map platform pause to CodeCraft `REJECTED`.

| Gate | Authorizes | Owner |
|------|------------|-------|
| CodeCraft local | Craft/session supervised eligibility | CodeCraft |
| ToolRuntime governance | `code.exec` policy + MSE | Governance |

## 15. Catalog binding

`CatalogDeclarativeRunBinding.declarative_hitl_grant` — internal compatibility only.

## 16. Implementation phases (R6)

1. Public contract 2. EE adapter 3. UCA wiring 4. TIGAE cleanup 5. E2E tests 6. architecture gates.

## 17. R6 readiness

YES.

## Compliance

ADR-GR-5-001, ADR-HARNESS-001/003, ADR-PLATFORM-PLUGIN-001; UCA_6C_CANONICAL_HITL_REENTRY_PROOF.md.
