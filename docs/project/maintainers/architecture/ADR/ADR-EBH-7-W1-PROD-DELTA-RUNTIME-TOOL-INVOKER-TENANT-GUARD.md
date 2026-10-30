# ADR-EBH-7-W1-PROD-DELTA: RuntimeToolInvoker attempt-scoped tenant guard

| Field | Value |
| --- | --- |
| **Status** | Accepted (OD-EBH7-01 — retain pending reconciliation) |
| **Date** | 2026-10-30 |
| **Deciders** | Operator (OD-EBH7-01); Runtime/Nexus maintainers |
| **Related** | [`EBH_7_OD_EBH7_01_ENTRY_READINESS_RECORD.md`](../../qualification/EBH_7_OD_EBH7_01_ENTRY_READINESS_RECORD.md) · W1-PROD-DELTA |

## Context

Gap-closure wave W1 introduced `_require_current_attempt_authorization` in `RuntimeToolInvoker`, comparing `ToolExecutionRequest.invocation_context.tenant_id` to `RuntimeState.tenant_id` and raising `ToolScopeViolationError` on mismatch.

Historical introduction predates OD-EBH7-01; operator disposition **RETAIN — PENDING ARCHITECTURAL RECONCILIATION** forbids silent revert and forbids a second parallel tenant resolver or governance admission path.

Canonical authority for root execution remains `RootExecutionAuthorityAdmissionService` and runtime policy admission. Tool access policy and catalog admission remain upstream of physical invocation.

## Decision

**Retain** the invoker guard as an **attempt-scoped fail-closed seam** (defense in depth at physical tool execution), not as a replacement for governance admission or ToolAccessPolicy.

**Rejected:**

- Revert without operator ADR — violates OD-EBH7-01 RETAIN disposition.
- Extract a new shared “tenant guard service” duplicated across Nexus and Governance — violates “no duplicate tenant authority.”
- Promote the guard alone to satisfy TEN-E2E-04 EXACT — insufficient; integrated ToolRuntime → admission chain still required for qualification.

## Consequences

### Positive

- Cross-tenant `ToolInvocationContext` projections fail closed at invoke with trace evidence.
- Aligns with ST-01 adversarial probe in EBH-7 W1/W2 reconciliation tests.

### Negative

- Product delta remains under architectural review until TEN-04 integrated proof is EXACT or operator directs consolidation into an existing canonical boundary.

## Compliance

- Tier boundaries preserved (`intergrax/` does not import `agents/` / `applications/`).
- No new governance port; composition unchanged.
- Qualification must not auto-promote TEN-04 to EXACT on guard alone.

## Implementation notes

- Code: `intergrax/runtime/nexus/tools/invoker.py` — `_require_current_attempt_authorization`.
- Probes: `tests/qualification/ebh_7/tenant_platform_e2e/_ebh7_w1w2_reconciliation_probes.py`.
- Verification: `uv run pytest tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_w1w2_qualification_reconciliation.py -q`.
