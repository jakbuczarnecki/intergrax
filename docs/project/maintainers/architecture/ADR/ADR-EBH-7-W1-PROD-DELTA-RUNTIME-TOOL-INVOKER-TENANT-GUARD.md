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

## Reconciliation addendum (EBH-7 master completion Wave A3)

**Date:** 2026-10-31 · **Audit HEAD:** `8b7661af15641a6a22d4eaa9127a5372d9968bfd`

| Question | Finding |
| --- | --- |
| Is the guard necessary defense in depth? | **Yes** — physical invoke remains a distinct seam from governance admission; mismatch on `ToolInvocationContext.tenant_id` vs runtime state must fail closed. |
| Does it duplicate admission authority? | **No semantic ownership transfer** — guard does not mint permissions; it rejects inconsistent attempt context only. |
| Parallel owner risk? | **Mitigated** — no second resolver service; single method on existing invoker. |
| Further product change required? | **Optional** — TEN-04 EXACT requires ToolRuntime chain evidence (Batch D); guard retained regardless. |
| Additional operator gate? | **OD-EBH7-INVOKER-RECON** optional sign-off before B5; not blocking authorized implementation. |

**Adversarial entrypoint matrix (qualification):** `test_ebh7_ten_adversarial_semantic_a2.py` + `assert_exec_tool_cross_tenant_via_tool_runtime` cover invoker and ToolRuntime paths; other entrypoints must remain covered by TEN/XDM integrated proofs (Wave D).
