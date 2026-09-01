# ADR-EXTWORK-001: Provider-neutral external work contracts (money + status)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / GEC-1 |
| **Related** | GEC host [`IMPLEMENTATION_PLAN.md`](../../../applications/governed_contractor_application/docs/IMPLEMENTATION_PLAN.md) · [`intergrax/contracts/external_work.py`](../../../../../../intergrax/contracts/external_work.py) · [`intergrax/contracts/money.py`](../../../../../../intergrax/contracts/money.py) · Platform consolidation [`governed_external_execution.md`](../../../platform/governed_external_execution.md) |

## Context

Governed external contractor flows need shared types for contractor identity, task correlation, quote, acceptance evidence, deliverable refs, and normalized external-work status. Existing Intergrax models cover many adjacent concerns (task/run ids, HITL decision ids, artifact refs, LLM float cost rollups, Nexus `TaskState`) but none represent commercial quotes or external-work commercial stages without duplication or category errors.

## Decision

1. Place reusable primitives under `intergrax/contracts` as flat modules (`money.py`, `external_work.py`) - not under Tier-3/Tier-2 GEC packages, and not a new `contracts/contractor` subtree.
2. Introduce `MoneyAmount` (`Decimal` + ISO 4217 alphabetic code). Reject reuse of `AgentRunCost` / token cost floats for commercial money.
3. Introduce `ExternalWorkStatus` for provider-neutral external-work progress. Do **not** extend Nexus `TaskState` with quote/commercial stages.
4. Compose (do not duplicate): Intergrax `task_id`/`run_id`/`correlation_id` strings, integration `provider_id`, `ActorIdentity`, optional HITL `decision_id` / `interrupt_id` string refs, `sha256:` digest convention, `ValidationResult` for pure quote/acceptance matching.
5. Keep contracts free of transport, authz, payment, policy evaluation, and receipt persistence.

Rejected:

- Owning contracts in `applications/governed_contractor_application` or `agents/external_contractor_adapter`
- Extending Nexus `TaskState` with `quote_pending` / `waiting_for_acceptance`
- Reusing `AgentRunCost.total_usd: float` as commercial money
- Embedding `HumanDecisionRecord` / `PolicyDecision` objects inside acceptance evidence (refs only)

## Consequences

### Positive

- GEC and future external-work applications share one platform vocabulary
- Nexus lifecycle stays orchestration-focused
- Commercial amounts remain exact (`Decimal`)

### Negative

- Callers must map `ExternalWorkStatus` - Nexus `TaskState` at adapter/host boundaries
- Currency validation is alphabetic ISO shape only (no full ISO table dependency)

## Compliance

- Tier boundaries preserved (`intergrax` has no `agents` / `applications` imports)
- Authorization remains outside contracts
- Transport deferred to GEC-2

## Implementation notes

- Modules: `intergrax/contracts/money.py`, `intergrax/contracts/external_work.py`
- Tests: `tests/unit/contracts/test_money.py`, `tests/unit/contracts/test_external_work.py`
- Verify: `uv run pytest tests/unit/contracts/test_money.py tests/unit/contracts/test_external_work.py -q`
