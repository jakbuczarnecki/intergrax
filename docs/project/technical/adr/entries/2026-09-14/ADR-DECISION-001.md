# ADR-DECISION-001: Public authoritative decision exposure at execution boundary

| Field | Value |
| --- | --- |
| **Status** | Proposed |
| **Date** | 2026-09-14 (R1 update: multi-scope + trust + terminality) |
| **Deciders** | Platform architecture / Scenario #1 track |
| **Related** | [`SCENARIO_1_P0_B_D1_AUTHORITATIVE_DECISION_RESULT_EXPOSURE.md`](../../../../architecture/SCENARIO_1_P0_B_D1_AUTHORITATIVE_DECISION_RESULT_EXPOSURE.md) · [`DECISION_SYSTEM.md`](../../../../architecture/DECISION_SYSTEM.md) |

## Context

`DecisionFlowResult` carries authoritative acceptance and resolution records, but Nexus graph execution propagates only `ValidationResult` via `decision_flow_result_to_validation_result`. Applications cannot distinguish execution completion from decision acceptance without private coupling or trace inference. Scenario #1 (P0-B) requires a typed, scenario-neutral public contract.

A single task may run **multiple** Decision evaluations (`DecisionFlowScope.GRAPH_FINAL`, `DecisionFlowScope.UAEP_STEP`, future scopes) with distinct `DecisionScope.subject` values. D1-R1 hardening requires a formal model for selecting **one** public `TaskResult.authoritative_decision_exposure` without enum precedence as a substitute for semantics.

## Decision

1. Add a **contracts-layer discriminated union** `AuthoritativeDecisionExposure[T]` reusing `AuthoritativeAcceptedDecision[T]` and `AuthoritativeResolutionRecord`, plus `ExposureUnevaluated` with explicit reasons (including execution failed/cancelled before Decision evaluation).
2. Map **once per evaluation** from `DecisionFlowResult` in Decision runtime adapters (`decision_flow_result_to_authoritative_exposure`); graph runner keeps **dual channels**: validation + per-evaluation exposure fragment.
3. **Execution host** owns public terminal selection via **`DecisionExposurePublicationPolicy`** + **`DecisionExposureSelectionStrategy`** over run-scoped **`DecisionExposureCandidate`** records (MS-B). Decision System owns correctness of each evaluation, not which scope is the host’s public final authority.
4. Compose **one** optional exposure on terminal **`TaskResult`** (non-generic); `ScenarioRuntimeExecutionResult` delegates without duplicating data.
5. **Do not** expose full `DecisionFlowResult` or extend `ValidationResult` with acceptance fields.
6. **Fail closed** when two eligible terminal candidates conflict within the same effective attempt (no silent last-write-wins). Attempt-aware selection prefers the **effective terminal attempt**.

### Selection owner (final)

| Concern | Owner |
| --- | --- |
| Per-evaluation authority | Decision System (`DecisionFlowGate.evaluate` → mapper) |
| Host terminal scope eligibility | Execution host (`DecisionExposurePublicationPolicy`) |
| Public outcome pick | Execution host (`DecisionExposureSelectionStrategy`; default or plugin) |
| Business RESOLVED / UNRESOLVED | Application only |

Default graph host policy: public terminal scope **`GRAPH_FINAL`** when evaluated; `UAEP_STEP` outcomes on graph hosts are **intermediate** for public terminal purposes. UAEP-only hosts may declare **`UAEP_STEP`** as eligible terminal scope.

### Trust semantics

- **Type validity ≠ authority authenticity.** Python allows manual construction of type-valid `AuthoritativeDecisionExposure` and nested records.
- **Trusted producer:** platform Decision runtime + trusted execution host path that maps, collects, selects, and attaches exposure to `TaskResult`.
- **Consumer:** application / external caller — may **read** exposure from execution results; must **not** supply exposure as proof of pre-approved Decision on platform intake APIs.
- **Provenance:** `DecisionIdentity.execution` (and nested accepted/resolution records) carries `run_id`, `task_id`, `attempt_id`, `decision_id`, version — no duplicate lineage on the envelope.
- **Security wording:** use **trusted runtime-issued**, not “unforgeable”. No cryptographic attestation in v1.

**Required contract documentation:**

> The type represents a platform-issued authoritative outcome when received from the trusted execution boundary. Construction of an equivalent value by application code does not constitute platform-issued authority.

> `AuthoritativeDecisionExposure` is authoritative only as the outcome of a trusted platform execution path; manual construction by application code does not confer platform authority.

### Terminal exposure invariants (post-migration)

| `TaskResult` lifecycle | `authoritative_decision_exposure` |
| --- | --- |
| Non-terminal (e.g. HITL `WAITING_FOR_HUMAN`, `NEEDS_INPUT`) | `None` |
| Terminal + Decision accepted (selected) | `ExposureAccepted` |
| Terminal + Decision resolution (selected) | `ExposureResolution` |
| Terminal + no gate / scope not evaluated | `ExposureUnevaluated` |
| Terminal + failed/cancelled before Decision on terminal path | `ExposureUnevaluated` with `EXECUTION_FAILED_BEFORE_DECISION` / `EXECUTION_CANCELLED_BEFORE_DECISION` — **not** `ExposureResolution` |

**Invariant:** terminal `TaskResult` with `authoritative_decision_exposure is None` is a **wiring/configuration violation** after migration (not a legacy default).

### Multi-scope semantics (summary)

- Collect **candidates** during execution (run-scoped collector; not trace, not metadata dict).
- Classify each evaluation: intermediate vs host-terminal-candidate vs non-publishable (e.g. `PENDING_HUMAN`).
- **Deterministic** selection; structured `reason_code` for observability.
- **Single** public field on `TaskResult`; optional observability lists all scopes evaluated.

See design doc **§D1-R1** for matrices, collision rules, attempt semantics, host matrix, and test strategy.

## Consequences

### Positive

- Preserves Decision / Execution / Application boundaries.
- Enforces constructive invariants; supports hosts without Decision gates and UAEP-only hosts.
- Pluginable, host-neutral selection without Scenario types in platform code.
- Unblocks P0-B-R1 and Scenario authoritative consumption after audit + I1.

### Negative

- New contracts (exposure + selection) and propagation through Nexus graph + task finalize.
- Host boundary uses `AgentExecutionResult` as carrier until P0-C artifact projection matures.
- Checkpoint/resume candidate reconstruction deferred beyond I1 v1 unless explicitly scoped.

## Compliance

- Tier boundaries preserved (contracts ← runtime ← applications).
- No scenario-specific platform types; no business semantics in selector.
- Implementation tracked under SCENARIO-1-P0-B-D1-I1 (split I1-A/B/C per design doc) **after D1-R1 design audit**.

## Implementation notes

- See design doc §14–§30 and **§D1-R1** for shapes, migration, terminality matrix, and tests.
- Verification: contract tests, selection strategy tests, graph decision integration, host task result propagation, negative guardrails (no trace/metadata selection).
