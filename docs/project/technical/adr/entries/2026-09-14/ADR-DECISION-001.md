# ADR-DECISION-001: Public authoritative decision exposure at execution boundary

| Field | Value |
| --- | --- |
| **Status** | Proposed |
| **Date** | 2026-09-14 |
| **Deciders** | Platform architecture / Scenario #1 track |
| **Related** | [`SCENARIO_1_P0_B_D1_AUTHORITATIVE_DECISION_RESULT_EXPOSURE.md`](../../../../architecture/SCENARIO_1_P0_B_D1_AUTHORITATIVE_DECISION_RESULT_EXPOSURE.md) · [`DECISION_SYSTEM.md`](../../../../architecture/DECISION_SYSTEM.md) |

## Context

`DecisionFlowResult` carries authoritative acceptance and resolution records, but Nexus graph execution propagates only `ValidationResult` via `decision_flow_result_to_validation_result`. Applications cannot distinguish execution completion from decision acceptance without private coupling or trace inference. Scenario #1 (P0-B) requires a typed, scenario-neutral public contract.

## Decision

1. Add a **contracts-layer discriminated union** `AuthoritativeDecisionExposure[T]` reusing `AuthoritativeAcceptedDecision[T]` and `AuthoritativeResolutionRecord`, plus `ExposureUnevaluated`.
2. Map **once** from `DecisionFlowResult` in Decision runtime adapters; graph runner keeps **dual channels**: validation + authoritative exposure.
3. Compose optional exposure on **`TaskResult`** (non-generic); `ScenarioRuntimeExecutionResult` delegates without duplicating data.
4. **Do not** expose full `DecisionFlowResult` or extend `ValidationResult` with acceptance fields.

## Consequences

### Positive

- Preserves Decision / Execution / Application boundaries.
- Enforces constructive invariants; supports hosts without Decision gates.
- Unblocks P0-B-R1 and Scenario authoritative consumption.

### Negative

- New contracts module and propagation through Nexus graph + task result.
- Host boundary uses `AgentExecutionResult` as carrier until P0-C artifact projection matures.

## Compliance

- Tier boundaries preserved (contracts ← runtime ← applications).
- No scenario-specific platform types.
- Implementation tracked under SCENARIO-1-P0-B-D1-I1 after design audit.

## Implementation notes

- See design doc §14–§30 for shapes, migration, and test matrix.
- Verification: contract tests, graph decision integration, host task result propagation.
