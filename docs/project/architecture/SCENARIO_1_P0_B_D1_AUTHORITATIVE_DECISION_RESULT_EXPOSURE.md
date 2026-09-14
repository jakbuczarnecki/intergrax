<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
-->

# SCENARIO-1-P0-B-D1 — Authoritative Decision Result Exposure

**Task:** SCENARIO-1-P0-B-D1  
**Status:** Architecture design (implementation not in scope)  
**Qualification driver:** Scenario #1 — AI incident investigation (`ai_incident_investigation`)  
**Branch target:** `development`  
**Blocks:** P0-B-R1 (Authoritative Result Boundary Cleanup)

**Related artifacts:**

| Artifact | Role |
| --- | --- |
| [`DECISION_SYSTEM.md`](DECISION_SYSTEM.md) | Decision System ownership |
| [`ENTERPRISE_RELIABILITY_LAYER.md`](ENTERPRISE_RELIABILITY_LAYER.md) | Parallel “authoritative facts → public boundary” precedent |
| [`ERL_DIAG_001_EXTERNAL_EFFECT_RELIABILITY_OPERATOR_DIAGNOSTICS.md`](ERL_DIAG_001_EXTERNAL_EFFECT_RELIABILITY_OPERATOR_DIAGNOSTICS.md) | Design doc pattern |
| [`ADR-DECISION-001`](../technical/adr/entries/2026-09-14/ADR-DECISION-001.md) | ADR companion |
| `intergrax.contracts.decision_record` | `AuthoritativeAcceptedDecision`, `DecisionArtifact` |
| `intergrax.contracts.decision_resolution` | `AuthoritativeResolutionRecord` |
| `intergrax.runtime.decision_flow` | `DecisionFlowResult` (runtime composition) |
| `intergrax.runtime.decision_flow_host` | `decision_flow_result_to_validation_result` |
| `intergrax.runtime.nexus.orchestration.graph_runner` | Loss point today |
| `intergrax.runtime.task.task` | `TaskResult` |
| `intergrax.applications._shared.scenario_runtime_baseline` | `ScenarioRuntimeExecutionResult` |

---

## 1. Executive summary

**Gap:** After `DecisionFlowGate.evaluate`, the platform retains full `DecisionFlowResult[T]` only inside the graph validation phase. The only propagated signal is a boolean-ish `ValidationResult` derived via `decision_flow_result_to_validation_result`. Applications receive `TaskResult` / `ScenarioRuntimeExecutionResult` without any typed authoritative Decision outcome.

**Recommendation:** Introduce a **minimal public contracts-layer envelope** — `AuthoritativeDecisionExposure[T]` — as a **discriminated union** over existing authoritative record types, plus an explicit **unevaluated** variant. Map once from `DecisionFlowResult` in the Decision runtime adapter layer. Propagate through **dual channels** on graph phase outcome and **optional composition** on `TaskResult`. `ScenarioRuntimeExecutionResult` remains a thin adapter over `TaskResult` (no second source of truth).

**Rejected:** Exposing full `DecisionFlowResult` at the application boundary (Option A). Extending `ValidationResult` with acceptance fields.

---

## 2. Scenario business driver

Scenario #1 requires the platform to support an investigation that may conclude with **business UNRESOLVED** inside an **authoritatively accepted** decision — without the application inferring acceptance from `TaskState.COMPLETED`, trace replay, or agent payload heuristics alone.

Today, `platform_proofs/scenarios/ai_incident_investigation` still projects domain conclusions from `task_result.execution_result` (see `scenario.py`), which violates P0-B single-authority intent and blocks P0-B-R1 cleanup.

---

## 3. Confirmed platform gap (code-backed)

### 3.1 Rich internal outcome

`DecisionFlowResult[T]` (`intergrax/runtime/decision_flow.py`) carries:

- `accepted_decision: AuthoritativeAcceptedDecision[T] | None`
- `resolution_record: AuthoritativeResolutionRecord | None`
- `candidate`, `verification_result`, `lifecycle_state`, `authorization`, `revision_decision`, HITL pending, etc.

### 3.2 Reduction at graph boundary

In `NexusGraphRunner.run` (`graph_runner.py` ~407–424), after `evaluate_agent_execution_flow`:

```text
flow_result → decision_flow_result_to_validation_result(flow_result) → final_validation
```

`decision_flow_result_to_validation_result` (`decision_flow_host.py`) maps only:

- `CONTINUE` → `ValidationResult(valid=True)`
- otherwise → `ValidationResult(valid=False, errors=[authority_reason])`

All acceptance semantics are **lost** for downstream consumers.

### 3.3 Application envelope

`ScenarioRuntimeExecutionResult` (`scenario_runtime_baseline.py`) contains `task_result`, ids, and deferred trace finalize — **no** decision authority fields.

`TaskResult` (`task.py`) exposes `state`, `execution_result`, `summary`, `metadata` — **no** authoritative decision slot.

---

## 4. Current data flow (before)

```text
DecisionFlowGate.evaluate
        ↓
DecisionFlowResult[T]
        ↓
NexusGraphRunner (GRAPH_FINAL scope)
        ↓
decision_flow_result_to_validation_result  ← authoritative collapse
        ↓
ValidationResult
        ↓
GraphPhaseOutcome.final_validation
        ↓
NexusLoop._finish_task(validation=…)
        ↓
TaskResult (state + execution_result + summary)
        ↓
HostTaskExecution.execute → TaskResult
        ↓
ScenarioRuntimeExecutionResult.task_result
        ↓
Application domain projection (today: execution_result payload inference)
```

---

## 5. Exact loss point

| Step | What survives | What is lost |
| --- | --- | --- |
| After `evaluate` | Full `DecisionFlowResult` | — |
| `decision_flow_result_to_validation_result` | `valid` + error string | `AuthoritativeAcceptedDecision`, `AuthoritativeResolutionRecord`, `decision_id`, version/lineage, artifact, scope, governance disposition |
| `GraphPhaseOutcome` | `final_validation` only | Entire decision flow result |
| `TaskResult` | Task execution state | Any decision authority |
| `ScenarioRuntimeExecutionResult` | Same as `TaskResult` | Same |

**Primary loss point:** mapping in `graph_runner.py` immediately after evaluation, before `GraphPhaseOutcome` is returned.

**Secondary gap:** no slot on `TaskResult` / host result to carry a propagated exposure even if graph retained it.

---

## 6. Layer ownership

| Concern | Owner | Must not |
| --- | --- | --- |
| Decision identity, version, lineage | Decision System (`intergrax.contracts.decision_*`) | Execution redefining acceptance |
| Candidate, verification, revision lifecycle | Decision System (runtime gate + contracts) | Application driving revision |
| `AuthoritativeAcceptedDecision` / `AuthoritativeResolutionRecord` | Decision System contracts | Application constructing spoofed authority |
| Public exposure envelope shape | **Contracts** (cross-layer, Decision-led) | Living only in `applications/` or scenario baseline |
| `DecisionFlowResult` → exposure mapping | Decision runtime adapter (`decision_flow_host` or sibling) | Nexus re-deriving acceptance from validation |
| Task/run/attempt lifecycle, retries, cancellation | Execution Engine | Becoming second Decision System |
| `ValidationResult` / Nexus validation semantics | Nexus validation plane | Carrying authoritative acceptance |
| Domain RESOLVED/UNRESOLVED, UI, reports | Application / Scenario | Reinterpreting platform acceptance |
| Trace / observability | Observability spine | Serving as application authority contract |

---

## 7. Existing contracts reused

| Contract | Reuse in design |
| --- | --- |
| `AuthoritativeAcceptedDecision[T]` | `Accepted` variant payload |
| `AuthoritativeResolutionRecord` | `Resolution` variant (REJECTED / UNRESOLVED **technical**) |
| `DecisionIdentity`, `DecisionVersionLineage`, `DecisionArtifact[T]` | Embedded in accepted variant (no field duplication) |
| `DecisionResolution` | Only inside `AuthoritativeResolutionRecord` |
| `ValidationResult` | Unchanged Nexus channel |
| `DecisionFlowScope` (runtime today) | Mapped to new contracts enum `DecisionEvaluationScope` in implementation (avoid app importing runtime) |

**Not reused at public boundary:** `DecisionFlowResult`, `CandidateDecision`, `VerificationResult`, `DecisionRevisionDecision`, raw `DecisionHumanReviewPending` (HITL handled via execution; see §23).

---

## 8. Design requirements (checklist)

- Typed public authoritative outcome (AC-D1-6)
- Accepted + terminal resolution (AC-D1-7)
- Hosts without Decision gate (AC-D1-8)
- Business UNRESOLVED via accepted payload (AC-D1-9)
- HITL / governance after full lifecycle for terminal exposure (AC-D1-10)
- Retry/attempt lineage via `DecisionIdentity.execution` (AC-D1-11)
- Scenario-neutral platform API (AC-D1-12)
- No metadata/trace fallback (AC-D1-4)
- No `TaskState` as acceptance proxy (AC-D1-5)
- Dual channel: validation ≠ authority (§30)
- Constructive illegal-state prevention (§11)

---

## 9. Option A — Expose `DecisionFlowResult[T]`

**Shape:** `TaskResult.decision_flow_result: DecisionFlowResult[T]` or on `ScenarioRuntimeExecutionResult`.

| Pros | Cons |
| --- | --- |
| No new mapping types | Leaks verification internals, revision, candidate |
| Full fidelity for debugging | Tied to one host flow shape |
| | Forces Decision runtime import at application boundary |
| | Weak invariants (`accepted` + `resolution` both optional) |
| | Poor host neutrality (UAEP vs Graph vs future hosts) |
| | Generics infect `TaskResult` or erases unsafely |

**Verdict:** **Reject** for public application boundary. May remain internal to graph runner during transition only.

---

## 10. Option B — Public authoritative outcome envelope

**Shape:** New contracts type `AuthoritativeDecisionExposure[T]` as discriminated union:

```text
AuthoritativeDecisionExposure[T]
  = ExposureAccepted[T]
  | ExposureResolution
  | ExposureUnevaluated
```

**Pros:** Minimal surface, strong invariants, reuse DS-CORE-02/04 records, contracts-first.

**Cons:** Requires new mapping function and scope enum in contracts; generic `T` at host boundary needs policy (§22).

**Verdict:** **Core of recommended design.**

---

## 11. Option C — Execution result composition

**Shape:** `TaskResult` gains optional `authoritative_decision: AuthoritativeDecisionExposure[…] | None` without making all of `TaskResult` generic.

**Pros:** Execution aggregates subsystem outputs; hosts without Decision keep `None` or `ExposureUnevaluated`; aligns with `HostTaskExecution` as canonical host return.

**Cons:** `TaskResult` gains optional Decision dependency (contracts only — acceptable).

**Verdict:** **Recommended propagation carrier** combined with Option B envelope.

---

## 12. Comparative matrix

| Kryterium | Option A | Option B | Option C |
| --- | ---: | ---: | ---: |
| Layering | Poor | Excellent | Excellent |
| Coupling | High | Low | Low–medium |
| Type safety | Misleading generics | Strong union | Strong + host-erased `T` |
| Reusability | Low | High | High |
| Host neutrality | Low | High | High |
| Migration risk | Medium | Low–medium | Low–medium |
| API minimalism | Poor (bloated) | Excellent | Excellent |
| Future extensibility | Accidental | Controlled | Controlled |

**Zwycięzca:** **Option B + C** (envelope in contracts, composed on `TaskResult`).

---

## 13. Recommended target architecture

```text
DecisionFlowGate.evaluate
        ↓
DecisionFlowResult[T]
        ├─→ decision_flow_result_to_validation_result → ValidationResult (Nexus)
        └─→ decision_flow_result_to_authoritative_exposure → AuthoritativeDecisionExposure[T]
                  ↓
           GraphPhaseOutcome
             final_validation
             authoritative_decision_exposure
                  ↓
           NexusLoop._finish_task
                  ↓
           TaskResult
             state, execution_result, …
             authoritative_decision_exposure: AuthoritativeDecisionExposure[AgentExecutionResult] | None
                  ↓
           HostTaskExecution.execute → TaskResult
                  ↓
           ScenarioRuntimeExecutionResult (adapter — reads TaskResult field)
                  ↓
           Application typed projection (domain plugin / artifact kind)
```

**Single mapping authority:** `decision_flow_result_to_authoritative_exposure` owns translation from flow result to public envelope. Graph runner must not reconstruct acceptance from validation.

**Terminal scope policy (v1):** For task-level public exposure, publish the **effective terminal** decision outcome for **`DecisionEvaluationScope.GRAPH_FINAL`** when evaluated; otherwise the highest configured terminal scope evaluated during the run (precedence: `GRAPH_FINAL` > `UAEP_STEP`). Partial/node scopes (future) do not override final exposure. Multi-scope simultaneous finals are a Decision System invariant violation and must fail mapping.

---

## 14. Public contract proposal (typed shape)

**Package (implementation):** `intergrax/contracts/decision_authoritative_exposure.py`  
**Ownership:** Decision System contracts (exported via `intergrax.contracts.decision` or dedicated `__init__`).

```python
class DecisionEvaluationScope(StrEnum):
    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"
    # future: NODE_PARTIAL, …

class ExposureUnevaluatedReason(StrEnum):
    NO_DECISION_GATE = "no_decision_gate"
    SCOPE_NOT_EVALUATED = "scope_not_evaluated"

@dataclass(frozen=True, slots=True)
class ExposureAccepted(Generic[T]):
    scope: DecisionEvaluationScope
    accepted: AuthoritativeAcceptedDecision[T]

@dataclass(frozen=True, slots=True)
class ExposureResolution:
    scope: DecisionEvaluationScope
    resolution: AuthoritativeResolutionRecord

@dataclass(frozen=True, slots=True)
class ExposureUnevaluated:
    scope: DecisionEvaluationScope | None  # None = no gate at all
    reason: ExposureUnevaluatedReason

AuthoritativeDecisionExposure = ExposureAccepted[T] | ExposureResolution | ExposureUnevaluated
```

**Mapping rules (`decision_flow_result_to_authoritative_exposure`):**

| `DecisionFlowResult` condition | Exposure |
| --- | --- |
| `accepted_decision is not None` | `ExposureAccepted` |
| `resolution_record is not None` (and no acceptance) | `ExposureResolution` |
| `host_action is PENDING_HUMAN` | **No terminal exposure** — return `None` at graph phase; task non-terminal |
| Gate absent / scope skipped | `ExposureUnevaluated` |
| Both accepted and resolution set | **Mapping error** — refuse; indicates DS bug |

**Factory trust:** Only Decision runtime module may call constructors after gate evaluation; contracts validate invariants in `__post_init__` (mirror `AuthoritativeResolutionRecord` rules).

---

## 15. Invariants

1. `ExposureAccepted` and `ExposureResolution` are mutually exclusive at exposure level.
2. `AuthoritativeResolutionRecord.resolution` never `ACCEPTED` (already enforced in DS-CORE-04).
3. Terminal public exposure is present only when decision lifecycle reached terminal authority for the published scope.
4. `TaskState.COMPLETED` does not imply `ExposureAccepted`.
5. `ExposureAccepted` does not imply business RESOLVED.
6. `DecisionResolution.UNRESOLVED` (technical) ≠ business UNRESOLVED in artifact payload.
7. No duplicate copies of identity/lineage outside accepted/resolution records.

---

## 16. Decision vs execution semantics

| Signal | Meaning |
| --- | --- |
| `TaskResult.state` | Execution lifecycle terminalization |
| `ValidationResult.valid` | Nexus/graph validation gate (includes decision block) |
| `AuthoritativeDecisionExposure` | Substantive Decision authority outcome |

Example valid combination:

```text
TaskState.COMPLETED
+ ExposureAccepted(payload.status = business UNRESOLVED)
```

---

## 17. Graph propagation

Extend `GraphPhaseOutcome`:

```python
final_validation: ValidationResult | None
authoritative_decision_exposure: AuthoritativeDecisionExposure[AgentExecutionResult] | None
```

- On `PENDING_HUMAN` early return: `authoritative_decision_exposure=None`, execution pauses.
- On successful terminal graph phase: set exposure from **final** `flow_result` for `GRAPH_FINAL`.
- UAEP hosts: same pattern on UAEP integration point (future I1 scope).

---

## 18. Task / host propagation

`TaskResult` (non-generic) field:

```python
authoritative_decision_exposure: AuthoritativeDecisionExposure[AgentExecutionResult] | None = None
```

Semantics:

| Value | Meaning |
| --- | --- |
| `None` | Not yet terminal **or** non-decision task path (legacy default during migration) |
| `ExposureUnevaluated` | Terminal task, decision subsystem not engaged for published scope |
| `ExposureAccepted` / `ExposureResolution` | Terminal authoritative outcome |

`HostTaskExecution.execute` returns `TaskResult` unchanged — no reinterpretation.

**Do not** add generics to `TaskResult` model; use fixed `AgentExecutionResult` carrier at execution host boundary. Applications recover domain `T` via artifact kind + typed projection (P0-C).

---

## 19. Application boundary

Applications import **only** `intergrax.contracts.*` exposure types and existing decision records.

Forbidden:

- `metadata["decision"]`, trace reconstruction, private `NexusGraphRunner` access
- Inferring acceptance from `task_result.state == COMPLETED`
- Importing `DecisionFlowResult` in `applications/` or `platform_proofs/` scenario code

`ScenarioRuntimeExecutionResult` may expose a **read-only property** delegating to `task_result.authoritative_decision_exposure` (adapter only).

---

## 20. Scenario #1 consumption (future)

```text
ScenarioRuntimeExecutionResult
        ↓
task_result.authoritative_decision_exposure
        ↓
match ExposureAccepted:
    AuthoritativeAcceptedDecision[AgentExecutionResult]
        ↓
application artifact projection (P0-C)
        ↓
InvestigationConclusion (RESOLVED / UNRESOLVED business)
```

Rejection path:

```text
ExposureResolution → NOT_ACCEPTED (application enum)
ExposureUnevaluated → explicit NOT_EVALUATED (≠ rejection)
```

---

## 21. Business UNRESOLVED semantics

Platform **acceptance** wraps domain payload. Scenario business `UNRESOLVED` lives in **application artifact content** inside `DecisionArtifact[T]`, not as `AuthoritativeResolutionRecord`.

Technical `DecisionResolution.UNRESOLVED` remains for “no accepted version” terminal resolutions — distinct from business wording.

---

## 22. Governance

Governance runs inside `DecisionFlowGate.evaluate` before exposure is minted. Application receives exposure **after** authorization and finalization for terminal paths. `DecisionExecutionAuthorization` stays inside flow result / trace — not duplicated on exposure (audit via decision identity + observability).

---

## 23. HITL

`DecisionFlowHostAction.PENDING_HUMAN` → graph early exit (`NEEDS_INPUT` / HITL runner). **No terminal `AuthoritativeDecisionExposure`**. Resume continues decision lifecycle; terminal exposure appears only on final host return after acceptance or resolution.

Existing task HITL fields remain execution-owned. Public decision exposure does not replace HITL pause records.

---

## 24. Retry / attempts / lineage

- Each attempt may evaluate decision with distinct `DecisionExecutionLineage.attempt_id`.
- Public final exposure reflects **final attempt’s** terminal authority for the published scope.
- Historical attempts: observability + decision event stream — not embedded in `TaskResult`.
- Retries must not duplicate exposure on partial returns; only terminal `TaskResult` carries final exposure.

---

## 25. Observability

Operators distinguish:

| Question | Source |
| --- | --- |
| Execution completed? | `TaskResult.state` |
| Nexus validation passed? | `TaskResult.summary` / validation metadata |
| Decision accepted? | `ExposureAccepted` |
| Decision technically rejected/unresolved? | `ExposureResolution` |
| Business outcome? | Application projection only |

Trace enriches audit; **not** authoritative for application logic.

---

## 26. Security / trust

Applications cannot mint `AuthoritativeAcceptedDecision` without going through gate (existing contract constructors + platform-only mapping). Exposure mapping function lives in Tier-1 runtime, invoked only from trusted orchestration paths.

---

## 27. Serialization / persistence

- **v1:** In-process host boundary — Pydantic-friendly optional serialization on `TaskResult` if already serialized for checkpoints; use JSON round-trip for nested dataclasses only if task persistence requires it (verify during I1).
- **Not required:** New DB tables for exposure; decision checkpoint store remains authoritative for replay.
- **API:** If HTTP hosts surface `TaskResult`, exposure serializes as discriminated object with stable field names.

---

## 28. Compatibility impact

| Surface | Change |
| --- | --- |
| `GraphPhaseOutcome` | Add field |
| `NexusGraphRunner` | Retain `flow_result`, dual map |
| `NexusLoop._finish_task` | Pass exposure into `_build_result` |
| `TaskResult` | Add optional field (clean cut default `None`) |
| `ScenarioRuntimeExecutionResult` | Optional delegate property |
| `decision_flow_result_to_validation_result` | Unchanged |
| UAEP decision integration | Same dual channel (I1) |
| Tests | New contract + graph + host suites (§30) |

---

## 29. Migration plan (clean cut)

1. **I1:** Add contracts + mapper + graph phase field + `TaskResult` field.
2. **I2:** Wire `NexusLoop` / `_build_result`; scenario reads new field; remove acceptance inference from scenario authority path (P0-B-R1).
3. **I3:** UAEP parity if not in I1.
4. Deprecate any documentation implying `ValidationResult.valid` means accepted decision.
5. No `legacy_decision_result` parallel fields.

---

## 30. Test strategy (future implementation)

### Contract tests

- Accepted / resolution / unevaluated variants
- Illegal dual acceptance+resolution in mapper raises
- Immutability
- Serialization round-trip (if enabled)

### Graph tests

- `DecisionFlowResult` → exposure preserves accepted decision identity
- Resolution propagates
- Validation projection unchanged
- No gate → `ExposureUnevaluated`

### Task / host tests

- Accepted propagates to `TaskResult`
- `COMPLETED` without `ExposureAccepted`
- Governance deny → resolution or block (not accepted)
- HITL: no terminal exposure until resume completes

### Application tests

- Scenario consumes exposure
- Business UNRESOLVED from accepted artifact
- Rejection → `NOT_ACCEPTED`
- No task-state inference

### Negative tests

- No metadata fallback
- No trace reconstruction requirement
- No private graph coupling in scenario code

---

## 31. Regression plan

Decision System unit suites, decision flow host, graph decision integration/parity, host task terminal publisher, scenario runtime baseline integration, UAEP decision paths, architecture boundary gates (`intergrax` must not import applications), P0-B authority tests.

---

## 32. Risks

| Risk | Mitigation |
| --- | --- |
| `AgentExecutionResult` as `T` at host boundary | P0-C artifact projection + artifact kind registry |
| Multiple scopes in one task | Precedence policy + single terminal exposure |
| Checkpoint serialization size | Exposure is small (refs + artifact) |
| Consumers confuse validation vs acceptance | Dual channel + docs + tests |

---

## 33. Rejected alternatives

- Full `DecisionFlowResult` on `TaskResult` (Option A)
- `ValidationResult.accepted_decision`
- Metadata / trace as authority
- Scenario-specific `IncidentDecisionResult` in platform layer
- `DecisionOutcomeProviderPlugin` for envelope itself

---

## 34. Platform layers changed by future implementation

| Layer | Change |
| --- | --- |
| `intergrax/contracts` | New exposure types |
| `intergrax/runtime/decision_flow_host` | Mapper |
| `intergrax/runtime/nexus/orchestration` | Propagation |
| `intergrax/runtime/task` | `TaskResult` field |
| `intergrax/applications/_shared` | Optional adapter property only |
| Applications / proofs | Consume exposure (separate tasks) |

---

## 35. Explicit layer-boundary validation

```text
intergrax/contracts  ← AuthoritativeDecisionExposure (public)
        ↑
intergrax/runtime/decision_flow*, nexus, task  (map + propagate only)
        ↑
intergrax/applications, platform_proofs  (import contracts only)
```

Decision System does not import applications. Execution does not import scenario code.

---

## 36. Design verdict

**PASS — READY FOR IMPLEMENTATION** (pending independent design audit per task §80).

---

## 37. Recommended implementation task name

**`SCENARIO-1-P0-B-D1-I1 — AUTHORITATIVE DECISION EXPOSURE PROPAGATION`**

(Contracts + mapper + graph + `TaskResult` + tests; scenario consumption deferred to P0-B-R1 / P0-C where appropriate.)

---

## 38. Updated roadmap snippet

| Etap | Status after D1 |
| --- | --- |
| P0-B-D1 Design | ✅ Complete (this document) |
| P0-B-D1 Design Audit | ⏳ Required before I1 |
| P0-B-D1-I1 Implementation | ⏳ Unblocked after audit |
| P0-B-R1 | ⏳ Blocked until I1 lands |
