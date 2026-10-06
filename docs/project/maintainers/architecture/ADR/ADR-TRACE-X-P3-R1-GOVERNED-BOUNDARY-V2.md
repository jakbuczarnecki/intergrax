# ADR: TRACE-X-P3-R1 — Governed execution boundary v2

## Context

`governed_execution_boundary_event.v1` records task/run and provider/policy/proof bindings but does not carry canonical `AttemptId` / `ExecutionId`. That blocks fail-closed FRZ-TRC-04/06 attribution without heuristic joins.

## Decision

Introduce `governed_execution_boundary_event.v2` under the same execution-evidence composition owner. Propagate canonical `TaskId`, `RunId`, `AttemptId`, and `ExecutionId` from Execution only via `ExecutionIdentitySection` and `GovernedExecutionResultV2`. New governed side effects write v2 boundary events and `execution_evidence.proof_receipt.v2`.

## Compatibility

v1 boundary and v1 receipt remain readable and verifiable; v1 shapes are not mutated. v2 is the sole canonical writer for new production governed effects.

## Receipt consequence

`ProofReceiptV2` pairs only with `ExecutionBoundaryEventV2`; cross-version receipt/event pairs fail closed in the offline verifier.

## Rejected

- In-place v1 mutation
- Reliability projection as attribution truth
- `GovernedProofProfile.execution_ref` as `ExecutionId`
- Correlation/timestamp joins
- New attribution service
- Host-minted canonical `ExecutionId` (`exec-{uuid}` fallback)
