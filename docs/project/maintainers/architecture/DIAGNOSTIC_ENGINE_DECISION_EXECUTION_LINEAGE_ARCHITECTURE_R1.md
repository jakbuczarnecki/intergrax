# Central Diagnostic Engine — Decision↔Execution Lineage (R4 / R1 delivery)

**Task:** `DIAGNOSTIC-ENGINE-DECISION-EXECUTION-DIAGNOSTIC-LINEAGE-R1`

## Invariant

```text
Decision System     → owns decision FACTS
Execution Runtime   → owns execution FACTS
Central Diagnostics → owns INTERPRETATION + Problem lifecycle
```

Decision evidence is **context**. Causality requires `PlatformCausalEvidence` (out of scope for this slice).

## Target flow

```text
Decision System → Decision Evidence (correlation + facts)
Execution Runtime → Execution Evidence (lineage, failure events)
Central Diagnostic Engine → Failure boundary + Decision context + Execution context → Problem
```

## Contracts

| Type | Role |
| ---- | ---- |
| `DecisionExecutionCorrelationRecord` | Immutable mapping decision scope ↔ execution scope |
| `DecisionExecutionCorrelationKind` | Why the link exists (bound execution, retry attempt) |
| `DecisionContextView` | Operator read — related decisions + bounded facts + limitations |
| `DiagnosticEvidenceContributor` | SPI name freeze; `DecisionEvidenceContributor` marker |

Correlation **does not** imply `decision → cause`.

## Read model

`DiagnosticReadService` accepts optional `DecisionContextProvider`. Occurrence views expose `decision_context` when configured.

Forbidden projections:

- decision as root cause
- decision confidence as diagnostic certainty
- decision score as failure certainty

## Persistence

`DecisionExecutionCorrelationPersistence` — append-only correlation evidence (adapter-neutral). Production adapters are follow-on; qualification uses `InMemoryDecisionExecutionCorrelationPersistence`.

## SPI (Etap 6)

Applications contribute decision facts via `DecisionEvidenceContributor` implementing the frozen contributor protocol — no per-app diagnostic engine.

## Quality gates

| Gate | Expectation |
| ---- | ----------- |
| `ONE_DIAGNOSTIC_ENGINE` | Single `intergrax.runtime.diagnostics` authority |
| `DECISION_SYSTEM_AUTHORITY` | Unchanged — no Problem store in Decision System |
| `NO_CAUSAL_INFERENCE` | No decision-cause finding kinds |
| `TENANT_ISOLATION` | Correlation queries scoped by `tenant_id` |
| `NO_LOCAL_DIAGNOSTICS` | No `DecisionDiagnosticEngine` / `DecisionProblemStore` |
