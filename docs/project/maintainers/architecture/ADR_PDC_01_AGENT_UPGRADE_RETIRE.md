# ADR-PDC-01 — Explicit agent UPGRADE and RETIRE operations on the distribution authority

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-01 |
| **Status** | **Accepted** (OD-PDC-ADR-01, 2026-11-12) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D1 Agent Lifecycle |
| **Requirements** | PDC-D1-R011, PDC-D1-R014 (and evidence for PDC-D1-R017) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-CMP-05, FRZ-EVOL-07, FRZ-PLG-08, FRZ-OBS-05 |

## Context

- `AgentLifecycleState.RETIRED` exists only as a routing filter (`runtime/registry/agent_routing_policy.py`); there is no retire operation.
- Upgrade happens implicitly as a new install into the same `installation_slot_id` followed by promote; no compatibility check is a first-class step.
- Hub invariants ADR-AGENT-001..003: Nexus is not the agent, HarnessKernel does not plan, `AgentRuntime.advance_step` is glue only (PDC-P1-F06).

## Decision (proposed)

- Add `upgrade_agent(...)` and `retire_agent(...)` to `AgentPlatformAdminService` (`intergrax/agent_distribution/admin_service.py`) and the agent manager command facade. No new registry, no new service authority.
- `upgrade_agent` = install candidate into the existing slot → `mark_verified` (trust + attestation) → **compatibility check** of the new `AgentContract` version against the bound configuration (COMPAT-X version policy) → `promote_verified_to_active`; failure leaves the previous revision `INSTALLED_ACTIVE` and records a typed rejection.
- `retire_agent` = transition to `RETIRED`, remove routing eligibility, refuse new activations, keep installation and evidence records (no tombstone); existing executions finish under their pinned revision.
- Every transition emits one canonical lifecycle event through the existing evidence path.

- **OD-PDC-SCOPE-01 = A (implement):** add `disable_skill`, `remove_skill` and `upgrade_skill` to `SkillHostLifecycleService`, extending the Skills manifest model (GAP_AUDIT §16 PRESERVE/EXTEND). Agent upgrade/retire for skill-bearing agents depends on it.

## Alternatives rejected

- A separate upgrade manager or lifecycle registry (shadow authority; FRZ-OWN-04).
- Retire implemented in the runtime step loop (violates ADR-AGENT-003).

## Verification

`tests/qualification/pdc_x/d1/`: upgrade accept/reject on compatibility, rollback to previous revision, retire blocks new routing while in-flight work completes, transition evidence completeness.

## Status note

Accepted by OD-PDC-ADR-01 (Operator, 2026-11-12, recorded under T-IGX-036-E). This authorizes PDC-P4 design. Product merges still follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8), which is pending.
