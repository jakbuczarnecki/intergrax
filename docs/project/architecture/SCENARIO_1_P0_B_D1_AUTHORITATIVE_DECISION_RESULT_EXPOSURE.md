<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
-->

# SCENARIO-1-P0-B-D1 — Authoritative Decision Result Exposure

**Task:** SCENARIO-1-P0-B-D1  
**Status:** Architecture design — D1-R2 effective-attempt semantics (implementation not in scope)  
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

**Terminal public outcome (v1, D1-R1):** Per-evaluation mapping still flows through `decision_flow_result_to_authoritative_exposure`. **Which** mapped outcome becomes the single `TaskResult.authoritative_decision_exposure` is **not** an enum precedence hack; it is selected by the **execution host** via a platform `DecisionExposureSelectionStrategy` fed by run-scoped `DecisionExposureCandidate` records (see §D1-R1).

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
    EXECUTION_FAILED_BEFORE_DECISION = "execution_failed_before_decision"
    EXECUTION_CANCELLED_BEFORE_DECISION = "execution_cancelled_before_decision"

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

**Type validity:** Contracts validate structural invariants in `__post_init__` (mirror `AuthoritativeResolutionRecord` rules). **Authority authenticity** is separate — see §D1-R1 (trust boundary); Python does not prevent manual construction of syntactically valid exposure values.

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

Semantics (post-migration invariant — see §D1-R1 terminality matrix):

| Value | Meaning |
| --- | --- |
| `None` | **Non-terminal** task lifecycle (HITL pause, in-flight execution, pending authority) |
| `ExposureUnevaluated` | **Terminal** task without a publishable Decision authority outcome for the host’s declared public terminal scope |
| `ExposureAccepted` / `ExposureResolution` | **Terminal** task with selected public authoritative Decision outcome |

**Post-migration:** `terminal TaskResult` + `authoritative_decision_exposure is None` is an **invariant violation** (configuration bug or incomplete host wiring), not a legacy default.

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

See **§D1-R1 — Trust model** for the authoritative statement. Summary: **type validity ≠ authority authenticity**; exposure is **trusted runtime-issued** only when produced by the platform execution path and selected for publication by the host. Application-constructed values are not accepted as platform authority **input** to any API.

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
| Multiple scopes in one task | Host `DecisionExposureSelectionStrategy` + fail-closed collision rules (§D1-R1) |
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

## D1-R1 HARDENING — Multi-scope selection, trust, terminality

**Task:** SCENARIO-1-P0-B-D1-R1  
**Supersedes (partial):** informal `GRAPH_FINAL > UAEP_STEP` precedence in §13; weak trust wording in §26; migration-ambiguous `None` semantics in §18.

### D1-R1.1 Problem statement

A single task/run may evaluate **multiple** `DecisionFlowGate.evaluate` calls at different `DecisionFlowScope` values and different `DecisionScope.subject` values (`DecisionFlowRequest.identity_seed.scope`). Each evaluation yields a correct `DecisionFlowResult[T]` for **that** scope/subject. The platform must still expose **exactly one** public `AuthoritativeDecisionExposure` on terminal `TaskResult` when the host declares a single public terminal authority — without assuming one task equals one decision.

**Decision scope ≠ publication priority.** Enum ordering is not semantics. Required model:

```text
Decision evaluation (per scope/subject/attempt)
  + publication eligibility (host terminal scope policy)
  + terminality (task lifecycle + host finalization)
  + selection strategy (deterministic, fail-closed)
→ effective public authoritative_decision_exposure
```

### D1-R1.2 Preserved D1 architecture (B + C)

Unchanged target shape:

```text
DecisionFlowResult
  ├── decision_flow_result_to_validation_result → ValidationResult
  └── decision_flow_result_to_authoritative_exposure → per-evaluation exposure fragment

Run-scoped candidates → host selection → TaskResult.authoritative_decision_exposure
```

Single field on `TaskResult` remains **`authoritative_decision_exposure`** (not a collection at the public application boundary). Optional observability may list all evaluated scopes; applications **match** the single selected exposure only (§D1-R1.20).

### D1-R1.3 Ownership: selection policy (S1 / S2 / S3)

| Option | Owner | Verdict |
| --- | --- | --- |
| **S1** Decision System picks global publishable terminal | Decision runtime | **Reject** — conflates per-evaluation correctness with host-final semantics |
| **S2** Execution host declares which scope is terminal for this host run | Execution host (Nexus, UAEP, future workflow) | **Accept (primary)** |
| **S3** Explicit host configuration object | Host composition | **Accept (configuration surface for S2)** |

**Split of concerns (target):**

- **Decision System:** each `DecisionFlowResult` is authoritative **for its evaluation** (`flow_scope`, `identity_seed.scope`, attempt lineage).
- **Execution host:** declares **which scope(s) may become the host’s public terminal authority** and runs **selection** over collected candidates.

This matches existing code: `DecisionFlowGateCapabilities.scopes` is composed by the hosting application (`application_decision_composition.py`); `DecisionFlowScope` is a **host invocation scope**, not a global platform ranking.

### D1-R1.4 Multi-scope options (MS-A … MS-D)

| ID | Approach | Summary |
| --- | --- | --- |
| **MS-A** | Hardcoded static precedence in mapper | Mapper picks “winner” via fixed enum order |
| **MS-B** | Execution-host selection contract | Host accumulates candidates; strategy selects one public outcome |
| **MS-C** | Decision-system publication contract | DS marks “publishable terminal” per evaluation |
| **MS-D** | Expose collection on `TaskResult` | Application chooses among all outcomes |

#### Comparative matrix

| Kryterium | MS-A | MS-B | MS-C | MS-D |
| --- | ---: | ---: | ---: | ---: |
| Layer ownership | Poor (mapper) | **Strong (host)** | Split / muddy | Weak (app) |
| Pluginability | Poor | **Strong (strategy plugin)** | Medium | N/A |
| Determinism | Fragile | **Strong (pure strategy)** | Medium | App-dependent |
| Host neutrality | Poor | **Strong** | Medium | Medium |
| Application simplicity | Medium | **Strong (single field)** | Strong | Poor |
| Auditability | Poor | **Strong (selection reason)** | Medium | Medium |
| Future extensibility | Poor | **Strong** | Medium | Medium |

**Recommendation: MS-B** — execution-host selection contract with pluggable `DecisionExposureSelectionStrategy`, default implementation `HostTerminalDecisionExposureSelector` (name illustrative only; not Scenario-specific).

MS-A rejected: smuggles host semantics into Decision mapper; non-pluginable `if graph elif uaep`.  
MS-C rejected: Decision System should not own “this host’s final public outcome”.  
MS-D rejected: violates single public authority invariant for Scenario #1; pushes platform gap to applications.

### D1-R1.5 Selection contract (design-only pseudocode)

**Package (future):** `intergrax/contracts/decision_authoritative_exposure.py` (types) + `intergrax/contracts/decision_exposure_selection.py` (strategy) — or co-located; **no implementation in D1-R1**.

```python
@dataclass(frozen=True, slots=True)
class DecisionExposureCandidate(Generic[T]):
    """Minimal input for selection; not a dump of DecisionFlowResult."""

    evaluation_scope: DecisionEvaluationScope  # maps from DecisionFlowScope
    decision_scope: DecisionScope  # namespace + subject from identity_seed
    execution_lineage: DecisionExecutionLineage  # run_id, task_id, attempt_id, …
    host_publication_class: HostPublicationClass  # see below
    exposure: AuthoritativeDecisionExposure[T]  # mapped fragment for this evaluation
    evaluation_ordinal: int  # monotonic per effective attempt (collector partition); stable ordering within attempt


class HostPublicationClass(StrEnum):
    INTERMEDIATE = "intermediate"  # e.g. UAEP_STEP on graph host
    HOST_TERMINAL_CANDIDATE = "host_terminal_candidate"  # eligible if policy says so
    NON_PUBLISHABLE = "non_publishable"  # PENDING_HUMAN path fragments


@dataclass(frozen=True, slots=True)
class DecisionExposurePublicationPolicy:
    """Host-declared terminal scope semantics (S3 config for S2)."""

    eligible_terminal_scopes: frozenset[DecisionEvaluationScope]
    # Graph host default: {GRAPH_FINAL}; UAEP-only: {UAEP_STEP}; no gate: ∅


@dataclass(frozen=True, slots=True)
class DecisionExposureSelectionDecision(Generic[T]):
    selected: AuthoritativeDecisionExposure[T]
    reason_code: str  # structured, no CoT; e.g. "host_terminal_scope_graph_final"
    considered_candidates: int


class DecisionExposureSelectionStrategy(Protocol[T]):
    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: Sequence[DecisionExposureCandidate[T]],
    ) -> DecisionExposureSelectionDecision[T] | DecisionExposureSelectionFailure:
        ...
```

**Collision:** two `HOST_TERMINAL_CANDIDATE` exposures with the same `(evaluation_scope, decision_scope.subject)` **within the same effective attempt** and both terminal-eligible → **`DecisionExposureSelectionFailure` (fail closed)** — not last-write-wins. (`attempt_id` appears only for identity correlation in lineage, not ordering — see **§D1-R2**.)

**Attempt-aware (corrected in D1-R2):** the selection strategy **does not** choose execution attempts. **Execution Engine** resolves **`effective_attempt_id`** before selection; superseded attempts’ Decision outcomes are not public terminal candidates.

**Determinism (within effective attempt only):** sort key `(evaluation_ordinal, evaluation_scope, decision_scope.namespace, decision_scope.subject)` before tie-break rules; no dict iteration order; **`AttemptId` must not appear in chronological sort keys**.

### D1-R1.6 Host semantics

| Host | Possible evaluation scopes | Public terminal selection |
| --- | --- | --- |
| **Graph host** (Nexus) | `UAEP_STEP` (intermediate) + `GRAPH_FINAL` (terminal candidate) | Policy `eligible={GRAPH_FINAL}`; intermediate UAEP never overrides finalized `GRAPH_FINAL`; if `GRAPH_FINAL` never evaluated → `ExposureUnevaluated(SCOPE_NOT_EVALUATED)` when gate configured |
| **UAEP-only host** | `UAEP_STEP` only | Policy `eligible={UAEP_STEP}`; subject = step identity; **last terminal UAEP evaluation is not automatic** — strategy picks among UAEP candidates per policy (e.g. configured final step subject or max ordinal) |
| **Direct agent / no Decision gate** | none | Terminal task → `ExposureUnevaluated(NO_DECISION_GATE)` |
| **Future workflow host** | configured | `DecisionExposurePublicationPolicy` + strategy from host composition |

**Scenario #1 rule:** when graph host completes with evaluated `GRAPH_FINAL`, **no** intermediate `UAEP_STEP` outcome may be selected as public terminal — host policy + `HostPublicationClass.INTERMEDIATE`, not global enum rank.

### D1-R1.7 Subject semantics

Two evaluations at `UAEP_STEP` with subjects `planning-step-1` vs `planning-step-2` are **distinct candidates**. Selection uses `(evaluation_scope, decision_scope.namespace, decision_scope.subject, attempt_id)` — not “last UAEP_STEP wins”.

Whether `UAEP_STEP` can ever be task-final public outcome: **yes, only on hosts whose `DecisionExposurePublicationPolicy.eligible_terminal_scopes` includes `UAEP_STEP`** (UAEP-only mode). On graph hosts, UAEP outcomes are **intermediate** for public terminal purposes.

### D1-R1.8 Collector / execution-local state

- **Owner:** execution host runtime (e.g. NexusLoop run context, UAEP host context) — **run/task scoped**, not global singleton or thread-local magic.
- **Shape:** run-scoped collector keyed/partitioned by `AttemptId`; append within partition; **finalize reads only the Execution-resolved effective attempt partition** (§D1-R2).
- **Population:** after each `evaluate`, mapper produces exposure fragment; host classifies publication class; append candidate.
- **Not allowed:** trace as storage; `TaskResult.metadata` dict; hidden module state.
- **v1 persistence:** in-process only; **resume/HITL after checkpoint** may require reconstructing candidates from decision checkpoint store — **out of scope for I1 v1** unless checkpoint replay already re-executes gates; document as **remaining gap** (§D1-R1.19).

### D1-R1.9 Pluginability / external strategy

- Host composition selects `DecisionExposureSelectionStrategy` (built-in default or **external plugin** implementing the protocol).
- **Integration point:** same host composition layer that wires `DecisionFlowGate` today (`application_decision_composition` / execution profile) — **reuse platform plugin framework**, no Scenario-specific selector types in platform code.
- Selector **must not** inspect business payload, hypothesis, diagnosis, or application RESOLVED/UNRESOLVED.

### D1-R1.10 Trust model

**Required statement:**

> `AuthoritativeDecisionExposure` is authoritative **only** as the outcome of a trusted platform execution path. Manual construction of a type-valid instance by application code does **not** confer platform authority.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Trusted producer? | Decision runtime path (`evaluate` → mapper) + host selection on trusted execution boundary |
| 2 | Consumer? | Application / external caller (untrusted for minting) |
| 3 | Manual construction possible? | **Yes** (Python) |
| 4 | Why manual ≠ authority? | Authority from **provenance** (lineage inside accepted/resolution records), **trusted path**, not datatype |
| 5 | Platform API intake of exposure? | **No** — not accepted proof of pre-approved Decision |
| 6 | Provenance verification? | `DecisionIdentity.execution` (`run_id`, `task_id`, `attempt_id`, `decision_id`, version) embedded in nested records; match to current `TaskResult` execution context |
| 7 | Cryptography now? | **No** — in-process trusted boundary sufficient for v1 |

**Asymmetry:** public **output** may be `AuthoritativeDecisionExposure`; public **input** must not treat caller-built exposure as authority. Do **not** claim “unforgeable”; use **trusted runtime-issued**.

**Public contract documentation (must appear in ADR + contracts docstring):**

> The type represents a platform-issued authoritative outcome when received from the trusted execution boundary. Construction of an equivalent value by application code does not constitute platform-issued authority.

No fake security (private constructors, magic tokens) as authority mechanism.

### D1-R1.11 Terminality matrix

| Task state | Decision evaluated for host terminal scope? | `authoritative_decision_exposure` |
| --- | ---: | --- |
| `COMPLETED` | accepted | `ExposureAccepted` |
| `COMPLETED` | resolution (no acceptance) | `ExposureResolution` |
| `COMPLETED` | gate configured, terminal scope not reached/evaluated | `ExposureUnevaluated` (`SCOPE_NOT_EVALUATED`) |
| `COMPLETED` | no gate | `ExposureUnevaluated` (`NO_DECISION_GATE`) |
| `FAILED` (before any Decision evaluation on terminal path) | no | `ExposureUnevaluated` (`EXECUTION_FAILED_BEFORE_DECISION`) — **≠** `ExposureResolution` |
| `CANCELLED` (before Decision evaluation) | no | `ExposureUnevaluated` (`EXECUTION_CANCELLED_BEFORE_DECISION`) |
| `WAITING_FOR_HUMAN` / `NEEDS_INPUT` | pending | `None` |
| `PARTIALLY_COMPLETED` | non-terminal for authority | `None` while lifecycle open; if host treats as terminal without Decision → `ExposureUnevaluated` per policy |

### D1-R1.12 Attempt matrix (examples)

Superseded ordering column semantics: use **§D1-R2.23** (execution `generation`, not `AttemptId`).

| Exec. ordering (`generation`) | Attempt ID (identity) | Scope | Outcome | Effective? | Public candidate? |
| --- | --- | --- | --- | ---: | ---: |
| 1 | random A | `GRAPH_FINAL` | Resolution | no | no |
| 2 | random Z | `GRAPH_FINAL` | Accepted | **yes** | **yes** |
| 1 | random A | `UAEP_STEP` / step-1 | Accepted | no | no (intermediate on graph host) |
| 2 | random Z | `GRAPH_FINAL` | Accepted | **yes** | **yes** |
| 1 | random A | `UAEP_STEP` / step-2 | Accepted | no | **yes** on UAEP-only host only when attempt 1 is effective |
| — | — | none | — | — | `ExposureUnevaluated(NO_DECISION_GATE)` |

### D1-R1.13 Multiple decision identities

`TaskResult` carries **one** public exposure field. Host policy + selection strategy must guarantee **at most one** eligible terminal authority per terminal task. If configuration yields two semantically co-equal terminal candidates **same attempt** → **fail closed** (selection failure → host surfaces task `FAILED` or explicit invariant error at finalize — exact host error mapping is I1 detail).

Two independent final Decision identities on one task without a declared selection rule → **configuration error**, not silent multi-authority.

### D1-R1.14 Observability

Structured events (not embedded in exposure): all evaluated scopes/subjects, candidate count, `DecisionExposureSelectionDecision.reason_code`, selected identity. Trace remains observability-only.

### D1-R1.15 Test strategy additions (future I1)

**Multi-scope:** (A) UAEP + GRAPH_FINAL → GRAPH_FINAL selected; (B) UAEP-only host → configured terminal; (C) duplicate eligible finals same attempt → fail closed; (D) attempt 2 over attempt 1; (E) intermediate never overrides terminal; (F) no gate → Unevaluated.

**Trust:** (A) app-built exposure not accepted as platform input; (B) execution output includes exposure; (C) no pre-approved intake API; (D) lineage preserved.

**Terminality:** matrix rows §D1-R1.11; failure/cancel ≠ resolution; HITL → `None`.

**Negative:** no last-write-wins; no metadata/trace selection; no task-state-as-acceptance; no business-payload selector; no Scenario-specific policy type in platform.

### D1-R1.16 Implementation split (recommendation)

| Task | Scope |
| --- | --- |
| **I1-A** | Contracts (`AuthoritativeDecisionExposure`, candidates, policy, strategy protocol), mapper, default selector, run-scoped collector |
| **I1-B** | Graph phase dual channel + NexusLoop finalize selection + `TaskResult` field |
| **I1-C** | UAEP host parity (same collector + strategy wiring) |

I1-B may start after I1-A contracts freeze; I1-C can follow B.

### D1-R1.17 Remaining gaps (v1)

- Checkpoint/resume reconstruction of candidate list without re-evaluation.
- Cross-process / API trust (serialization attestation) — deferred.

### D1-R1.18 Layer boundaries (validation)

```text
intergrax/contracts  ← exposure + selection protocol
        ↑
Decision runtime     ← per-evaluation map only
        ↑
Execution host       ← collector + policy + strategy (default | plugin)
        ↑
Application          ← consume single exposure; no minting as input
```

---

## D1-R2 — Effective Attempt Selection Semantics

**Task:** SCENARIO-1-P0-B-D1-R2  
**Supersedes (partial):** D1-R1 §D1-R1.5 attempt ordering via `AttemptId`; any “highest attempt_id” or lexicographic attempt sort; implicit selector-owned retry lifecycle.

### D1-R2.1 Hard rule — `AttemptId` identity-only

`AttemptId` values are opaque identities (e.g. `attempt_<uuid>` from `mint_attempt_id()` / `mint_retry_attempt_id()` in `intergrax/contracts/execution_identity.py` and `intergrax/runtime/execution/identity_authority.py`). **Permitted:** equality, lineage correlation, lookup, deduplication. **Forbidden:** `max`/`min`, lexicographic sort, “newer than”, retry ordering, effective-attempt resolution.

### D1-R2.2 Effective attempt — definition

**Effective attempt** (for public Decision exposure at terminal `TaskResult` finalization): the execution attempt the **Execution Engine** treats as authoritative for the terminal task/run outcome after retry supersession, cancellation, and recovery — **not** the lexicographically largest UUID and **not** the last Decision callback arrival order.

### D1-R2.3 Canonical Execution Engine source of truth (HEAD audit)

| Mechanism | Location | Role |
| --- | --- | --- |
| **`AttemptLifecycleState`** | `intergrax/contracts/attempt_lifecycle.py` | Durable record: `active_attempt_id`, `previous_attempt_id`, **`generation`** (≥1, monotonic per run retry sequence), `transition_reason` |
| **`AttemptLifecycleService`** | `intergrax/runtime/execution/attempt_lifecycle/service.py` | **Canonical authority** for attempt transitions: `record_initial_attempt`, `transition_to_next_attempt`, `get_active_attempt_id`, `get_current_generation` |
| **`ExecutionAttemptRetryService.transition_for_retry`** | `intergrax/runtime/execution/retry/service.py` | Orchestrates policy-eligible retry → lifecycle transition → **`rebind_active_attempt_for_retry`**; seals superseded attempt in lineage (`RETRY_SUPERSEDED`) |
| **In-process active identity** | `require_active_execution_identity()` in `intergrax/contracts/execution_identity.py` | Current `(run_id, attempt_id)` for executing work; rebinding follows durable transition (`transition_retry` on facade is **explicitly non-authoritative**) |

**Retry ordering signal:** `AttemptLifecycleState.generation` (and transition graph), **not** `AttemptId`.

**Not authoritative for execution-attempt ordering:** Nexus `RetryRecord` / node-level `RetryEngine` (`intergrax/runtime/nexus/retry/`) — agent/step retries, not run-scoped effective attempt.

### D1-R2.4 Target flow (required)

```text
Execution attempt lifecycle (AttemptLifecycleService + retry orchestration)
        ↓
effective_attempt_id resolved (Execution Engine)
        ↓
Decision exposure candidates for effective attempt only
        ↓
DecisionExposurePublicationPolicy (host)
        ↓
DecisionExposureSelectionStrategy (scope/subject/terminal eligibility)
        ↓
single authoritative_decision_exposure on TaskResult
```

### D1-R2.5 Negative flows (forbidden)

```text
all attempts → sort AttemptId → pick max
```

```text
pick attempt whose Decision outcome is Accepted / “better” business result
```

```text
DecisionExposureSelectionStrategy chooses which retry won
```

### D1-R2.6 Effective attempt resolution at finalize (design contract)

At host **`TaskResult` finalization** (I1 detail, design invariant now):

1. Resolve **`effective_attempt_id`** from Execution lifecycle — **prefer durable** `AttemptLifecycleService.get_active_attempt_id(tenant_id, run_id)` when run lifecycle is durable; **in-process** finalize on the same worker must match `require_active_execution_identity()[1]` after canonical retry rebind.
2. If lifecycle cannot resolve an effective attempt for a terminal task → **fail closed** (`DecisionExposureSelectionFailure` / invariant error); do not pick arbitrary candidates.
3. **`candidates = collector.for_attempt(effective_attempt_id)`** (run-scoped collector partitioned by `AttemptId`).
4. **`strategy.select(policy, candidates)`** — inputs **only** effective-attempt candidates.

Application-supplied `effective_attempt_id` is **never** trusted input.

### D1-R2.7 Ownership table (effective attempt)

| Concern | Owner |
| --- | --- |
| Attempt identity | Execution Engine |
| Retry ordering / supersession | Execution Engine (`generation`, lifecycle transitions) |
| **Effective terminal attempt** | **Execution Engine** |
| Decision authority within attempt | Decision System |
| Public scope/subject inside effective attempt | Execution host (`DecisionExposurePublicationPolicy` + `DecisionExposureSelectionStrategy`) |
| Business outcome | Application |

### D1-R2.8 Supersession semantics

When attempt A fails/eligible-retry → `transition_to_next_attempt` mints attempt B, updates `active_attempt_id`, increments `generation`, records `previous_attempt_id=A`. Attempt A is **superseded** for terminal public authority; its Decision outcomes remain historically correct **for A** but are **not** eligible public terminal candidates after B is effective.

**Effective ≠ Decision success:** if effective attempt 2 yields `ExposureResolution` and superseded attempt 1 had `ExposureAccepted`, public exposure is **Resolution from attempt 2**.

### D1-R2.9 Concurrency and late completion

Overlapping or late-finishing superseded attempts must not override exposure: only **effective attempt** candidates participate at finalize. Arrival order of Decision callbacks or trace events is **not** ordering authority.

| Attempt | Lifecycle position (`generation`) | Completion time | Effective |
| --- | ---: | ---: | ---: |
| A | older (1) | later | no |
| B | newer (2) | earlier | **yes** |

Design test: `attempt_old_id = "attempt_ffff…"`, `attempt_new_id = "attempt_0000…"` — effective is the **newer generation**, not lexicographic ID order.

### D1-R2.10 Recovery / resume

After checkpoint/resume, durable `AttemptLifecycleState` restores `active_attempt_id` and `generation` (qualified in NPSC-5E recovery tests). Effective attempt for exposure finalize must be recovered from **the same lifecycle store**, not recomputed from `AttemptId`. **I1 v1** may still defer **rebuilding** the decision candidate collector from checkpoints without re-evaluation (D1-R1.17 gap remains); lifecycle effective attempt itself is **not** a platform gap at HEAD.

### D1-R2.11 Collector options — EA-A … EA-D

| ID | Model | Verdict |
| --- | --- | --- |
| **EA-A** | Selector compares / sorts `AttemptId` | **Reject** — violates identity-only rule and Execution ownership |
| **EA-B** | Selector receives all attempts + ordinal | **Reject** — selector still owns attempt winner |
| **EA-C** | Execution resolves effective attempt; selector receives **only** that attempt’s candidates | **Accept (preferred default)** |
| **EA-D** | Attempt-scoped collector created/handoff per active attempt by Execution host | **Accept** when host lifecycle already isolates per-attempt context |

#### Comparative matrix

| Kryterium | EA-A | EA-B | EA-C | EA-D |
| --- | ---: | ---: | ---: | ---: |
| Layer ownership | Poor | Weak | **Strong** | **Strong** |
| Determinism | Wrong | Medium | **Strong** | **Strong** |
| Retry correctness | Wrong | Medium | **Strong** | **Strong** |
| Concurrency safety | Poor | Medium | **Strong** | **Strong** |
| Simplicity | — | Low | **Strong** | Medium |
| Reuse of lifecycle | No | Partial | **Strong** | **Strong** |
| Pluginability (scope selection) | N/A | N/A | **Strong** | **Strong** |

**Recommendation:** **EA-C** with run-scoped partitioned collector (matches D1-R1.8 shape); consider **EA-D** if I1 wiring already binds collectors to attempt-local execution scopes.

Effective attempt resolution is **not** a plugin strategy — it is a **lifecycle invariant** implemented via existing `AttemptLifecycleService` / retry orchestration.

### D1-R2.12 Selector contract (revised)

```python
# After Execution resolves effective_attempt_id:
candidates_for_effective = collector.candidates_for_attempt(effective_attempt_id)
decision = strategy.select(policy, candidates_for_effective)
```

`DecisionExposureSelectionStrategy` chooses **scope**, **subject**, and **terminal eligibility** among candidates **inside one effective attempt** only.

### D1-R2.13 Observability (selection event)

Include: `effective_attempt_id`, lifecycle resolution method (`durable_active_attempt` / `in_process_identity`), `generation` when available, candidate count, selected `DecisionIdentity`, structured `reason_code`. No chain-of-thought.

### D1-R2.14 Test strategy additions (future I1)

- **ID order independence:** smaller lexicographic UUID wins when `generation` is higher.
- **Retry supersession:** attempt 1 Resolution, attempt 2 Accepted → public Accepted from attempt 2 (via generation, not ID).
- **Late old attempt:** attempt 2 terminal before attempt 1 completes → no override.
- **Same attempt collision:** two eligible finals same effective attempt → fail closed (unchanged).
- **Multi-scope + retry:** attempt 1 UAEP + GRAPH_FINAL Resolution; attempt 2 UAEP + GRAPH_FINAL Accepted → effective attempt 2, selected GRAPH_FINAL Accepted.
- **UAEP-only retry:** Execution picks attempt; selector picks terminal UAEP candidate within attempt.
- **No effective attempt at finalize:** fail closed.
- **Cancelled / superseded:** cancelled attempt not public accepted authority when not effective terminal.

### D1-R2.15 Platform gap verdict

**No BLOCKED gap** for execution-attempt ordering at HEAD: canonical durable + in-process mechanisms exist. Remaining gaps: candidate collector reconstruction on resume (D1-R1.17), not missing effective-attempt contract.

### D1-R2.16 Acceptance mapping (R2)

| AC | Status |
| --- | --- |
| AC-R2-1 … AC-R2-15 | Addressed in this § |

---

## 36. Design verdict

**PASS — READY FOR FINAL DESIGN AUDIT** after **D1-R2**; **blocked** on independent **D1-R2 design audit** before P0-B-D1-I1 (supersedes D1-R1 audit gate).

---

## 37. Recommended implementation task name

**`SCENARIO-1-P0-B-D1-I1 — AUTHORITATIVE DECISION EXPOSURE PROPAGATION`**

(Contracts + mapper + graph + `TaskResult` + tests; scenario consumption deferred to P0-B-R1 / P0-C where appropriate.)

---

## 38. Updated roadmap snippet

| Etap | Status after D1-R2 |
| --- | --- |
| P0-B-D1 Design | ✅ Baseline (e3139f2f…) |
| P0-B-D1-R1 Hardening | ✅ §D1-R1 (attempt ordering corrected in R2) |
| P0-B-D1-R2 Effective attempt | ✅ This §D1-R2 |
| D1 FINAL DESIGN AUDIT | ⏳ Required before I1 |
| P0-B-D1-I1 Implementation | ⏳ Blocked until D1 final audit |
| P0-B-R1 | ⏳ Blocked until I1 lands |
