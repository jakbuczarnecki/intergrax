# TOOL-LIFE-X P0 — Canonical Tool Lifecycle Closed-World Pre-Audit

| Field | Value |
| --- | --- |
| **Task** | `TOOL-LIFE-X-P0-CANONICAL-LIFECYCLE-PREAUDIT` |
| **Parent** | `TOOL-LIFE-X` (§3.0.5 roadmap) — **not CLOSED** |
| **Predecessor** | `SKILL-X` = **CLOSED — INDEPENDENTLY ACCEPTED** @ `552e379845a9512c275f3ebe520ac51767c97cf8` |
| **Successor (blocked)** | `DIST-SCALE-X` until `TOOL-LIFE-X` closes |
| **Branch** | `development` |
| **START_HEAD / audit SHA** | `76a98cb3f3e4576ab694f03a5b3e270eb7694f9e` |
| **Origin** | `https://github.com/jakbuczarnecki/intergrax.git` |
| **Production delta (P0)** | **0** |
| **pytest @ P0** | **0** (read-only inventory; no platform-wide replay) |
| **Disposition** | **`TOOL-LIFE-X P0 — READY FOR INDEPENDENT AUDIT`** |

**Canonical authorities read @ START_HEAD:**

- [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.5
- [`PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- [`ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md`](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) (ECAP-P1/P2/P4/P6)

**Non-closure:** This record does **not** close `TOOL-LIFE-X`, does **not** promote any global `FRZ-*` PASS, and does **not** substitute `HARNESS-W4`, `GOV-X2`, `STATE-X`, or `QUAL-X` parent closure.

---

## 1. Project identity gate

| Check | Result |
| --- | --- |
| Repository | `jakbuczarnecki/intergrax` @ `origin` |
| Branch | `development` (not detached) |
| `HEAD` | `76a98cb3f3e4576ab694f03a5b3e270eb7694f9e` (= **EXPECTED_START_HEAD**) |
| Working tree @ audit | **clean** |
| `EXPECTED_START_HEAD` ancestor of `HEAD` | **yes** |
| Conflicting active stage | **none** — roadmap §5 row `TOOL-LIFE-X` = **NEXT / PLANNED / MANDATORY** |
| `SKILL-X` prerequisite | **satisfied** (reconciliation [`SKILL_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md`](SKILL_X_PARENT_FORMAL_CLOSURE_RECONCILIATION.md)) |
| Unauthorized production delta in P0 | **none** (documentation only) |

---

## 2. Canonical tool invocation lifecycle owner (single owner)

**Answer:** Per-invocation **semantic enforcement** is owned by **`RuntimeToolInvoker`** (`intergrax/runtime/nexus/tools/invoker.py`), composed on strict production paths via **`build_production_runtime_tool_invoker`** (`runtime_tool_invoker_composition.py`). **Run / Attempt / Execution** lifecycle, deadline authority, and protected-work admission are owned by **`ExecutionRuntime`** / UER (`intergrax/runtime/execution/runtime.py`, architecture [`UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md)).

**Sanctioned ingress (no shadow registry):**

```text
Agent / UAEP / Skill-resolved plan
  → ToolRuntime.invoke / RuntimeToolGateway.invoke (§42.12)
  → catalog_dispatch.invoke_catalog_tool_* 
  → ToolInvokerProtocol → RuntimeToolInvoker.invoke
  → ToolExecutor + provider handlers (intergrax/tools/**)
```

**Governance ≠ execution:** `ToolScopePolicy` + `ToolAccessPolicy` + agent governance ports gate **admission**; they do not execute tools. **Composition root:** host runtime config / `build_production_runtime_tool_invoker` (Tier-1), not agents or applications constructing invokers locally (see `test_platform_execution_unification_u5_final_zero_bypass.py`).

**Read seam (observability, not authority):** `intergrax/contracts/tool_runtime_read.py` — `ToolRuntimeInvocationOutcome` is a **read model** (`requested|completed|denied|failed`), not a competing lifecycle state machine.

**Continuation / long-running catalog work:** `ContinuationAwareCatalogToolHost` + suspended-operation store — **child of Execution** identity, re-enters via sanctioned ToolRuntime after resume; does not introduce a parallel Tool-only owner.

---

## 3. Closed-world inventory and classification

| Surface | Role | Class |
| --- | --- | --- |
| `ExecutionRuntime` (`runtime/execution/runtime.py`) | Run/Attempt lifecycle, deadline scope bind, capacity admission | **CANONICAL** (Execution) |
| `ExecutionDeadlineAuthority` + `deadline_scope` + `protected_work_admission` | Global/step deadline, protected-work denial | **CANONICAL** |
| `RuntimeToolInvoker` | Per-tool timeout, retry loop, cooperative cancel, idempotency pre-effect, external-op attempt | **CANONICAL** (Tool invocation) |
| `DependencyAttemptExecutionBoundary` | Queue wait / worker permit; detach on caller timeout | **CANONICAL** (transport admission) |
| `CancellationCoordinator` | Cooperative cancel metadata | **CANONICAL** |
| `ToolExternalOperationAttempt` + `ExternalOperationStateStore` | Durable external-op identity, cancel intent, stale terminal guard | **CANONICAL** |
| `ToolContract` (`timeout_ms`, `retry_policy`, `side_effect_retry_safety`) | Typed per-tool attempt budget | **CANONICAL** contract |
| `ToolEffectCertainty` / `InvocationStatus.UNCERTAIN` | Unknown-outcome representation | **CANONICAL** typed semantics |
| `IdempotencyPreEffectCoordinator` + stores | Side-effect claim / uncertain blocking | **CANONICAL** |
| `ToolRuntime` / `RuntimeToolGateway` / `catalog_dispatch` | Callers orchestrating plans & §42.12 requests | **CALLER** |
| `ToolScopePolicy` / `ToolAccessPolicy` | Allow-list / scope intersection | **GOVERNANCE** |
| `ToolRegistryRead` + `ToolExecutor` + `intergrax/tools/providers/**` | Provider adapters | **ADAPTER** |
| `budget_ticks` | Tool-call budget ticks vs execution budget | **GOVERNANCE** (budget plane) |
| `tool_runtime_read` port | Historical invocation facts | **OBSERVABILITY** |
| Direct `ThreadPoolExecutor` in non-production test harnesses | Local doubles | **LEGACY/UNQUALIFIED** for TOOL-LIFE-X closure |

**Extended surfaces (directly relevant):**

- Graph/step: `graph_runner.py`, `graph_executor.py` — pass `cancelled=` into retry eligibility; cooperative batch cancel (W4-A).
- MCP / HTTP / SQL: provider modules under `intergrax/tools/providers/` and integrations — **must** remain behind `RuntimeToolInvoker`; no `runtime/nexus/tools` MCP bypass found @ HEAD.
- Qualification: `test_runtime_tool_invoker_dependency_admission.py`, `test_tools_side_effect_safety.py`, `test_enterprise_scale_resilience_w4_a_cancellation_qualification.py`, `test_p0_safety_8_retry_redelivery_authorization.py`, harness W4 parent replay @ `f38e455ccc391dad9a6a64d7867a33cfc5c9b21d`.

---

## 4. Lifecycle and authority pre-audit (§3.0.5 mapping)

Roadmap states **`CREATED → ADMITTED → STARTED → RUNNING`** with terminals **`SUCCEEDED | FAILED | TIMED_OUT | CANCELLED | ABANDONED`** (+ optional **`RETRY_SCHEDULED`**).

**Mapping without a second state machine (recommended certification model):**

| Roadmap state | Canonical owner | Existing equivalent @ HEAD |
| --- | --- | --- |
| CREATED | Caller | `ToolExecutionRequest` / `ToolRequest` constructed |
| ADMITTED | Governance + scope + budget + protected-work admission | `ToolAccessPolicy`, `ToolScopePolicy`, `_protected_work_admission_tool_denial`, governance ports |
| STARTED | RuntimeToolInvoker | `tool_invocation_start` trace + optional idempotency claim |
| RUNNING | RuntimeToolInvoker + DependencyAttempt + external-op record | Thread pool / async execution; `ToolExternalOperationAttempt` |
| RETRY_SCHEDULED | RuntimeToolInvoker | Inter-attempt `cooperative_delay_seconds` + attempt counter |
| SUCCEEDED | ToolExecutionResult.ok | `effect_certainty=COMPLETED` |
| FAILED | ToolExecutionResult.fail | Definite failure classification |
| TIMED_OUT | RuntimeToolInvoker | `FuturesTimeoutError` → fail + `effect_certainty` NOT_STARTED or UNCERTAIN |
| CANCELLED | CancellationCoordinator + invoker | `task_cancelled` / cooperative abort |
| ABANDONED | **Gap (typed)** | Partial: retry exhaustion, uncertain idempotency, external-op ambiguity — **no single `ABANDONED` enum on tool path**; `InvocationStatus.UNCERTAIN` + compensation jobs approximate unknown-outcome stall |

**UNKNOWN_OUTCOME:** Represented as **`ToolEffectCertainty.UNCERTAIN`** on `ToolExecutionResult` and **`InvocationStatus.UNCERTAIN`** on idempotency ledger — **not** equated to `FAILED`. Certification must prove operators/agents cannot blind-retry (see `test_side_effect_timeout_does_not_blind_retry`, `test_side_effect_timeout_marks_uncertain_and_blocks_replay`).

**Transition authority:** Only **`RuntimeToolInvoker`** (and Execution runtime for parent cancel/deadline) may emit terminal tool outcomes on production paths; adapters return raw results; invoker normalizes. **Forbidden:** terminal resurrection — `ExternalOperationStateStore` raises on stale state; late results require explicit reconciliation (W4-C scope — **VALIDATION REQUIRED** for full TOOL-LIFE-X).

---

## 5. Timeout / deadline / resource hierarchy

**Hierarchy @ HEAD (partial certification):**

```text
Execution deadline (ExecutionRuntime + deadline_scope + graph_runner global_deadline_monotonic)
  → Protected-work admission (tool denial when deadline exceeded)
  → ToolContract.timeout_ms (per attempt; default 30_000 ms; min 1 — no None default)
  → DependencyAttemptExecutionBoundary (queue wait with WAIT_WITH_TIMEOUT)
  → Provider connect/read timeouts (HTTP `timeout_s`, sandbox caps, integration clients)
```

| Layer | Mechanism | Gap |
| --- | --- | --- |
| Execution → Step | `peek_active_execution_global_deadline_monotonic` in graph runner | **VALIDATION REQUIRED** — prove step tools inherit remaining budget on all wired paths |
| Queue wait vs attempt timeout | Admission timeout before worker submit | **VALIDATION REQUIRED** — scenario 21 (queue wait eats deadline) |
| Result size / rows | Per-provider limits (RAG, records `sample_limit`, excel `max_rows_per_sheet`) | **IMPLEMENTATION REQUIRED** — unified SQL/`database.query` row/byte ceiling not one TOOL-LIFE contract |
| Long-running policy | Suspended operations + continuation host | **VALIDATION REQUIRED** — enumerate tools needing async job pattern (§3.0.5 long-running jobs) |

**Real unbounded wait cases (production, @ HEAD scan):** Default tool contract forbids `timeout_ms < 1`; **no** platform-wide `timeout = None` on `ToolContract`. Residual risk: **provider-internal** blocking without contract-mapped timeout (integration-specific) — **provider matrix §9**.

---

## 6. Concurrency, retry, idempotency, unknown-outcome

| Tool class (roadmap) | Contract knob | Behavior @ HEAD |
| --- | --- | --- |
| Read-only | `side_effect_retry_safety` default | Retries allowed only if policy + safety permit |
| Idempotent write | `RETRY_SAFE` + idempotency key | Key preserved across attempts (`test_idempotency_key_preserved_across_retry_attempts`) |
| Non-idempotent write | `NOT_RETRY_SAFE` (default) | `_effective_max_attempts` → **1** |
| Unknown outcome | `ToolEffectCertainty.UNCERTAIN` | Ledger UNCERTAIN; blocks blind replay |
| Externally deduplicated | Idempotency store + fence/lease | `sqlite_idempotency_store` lease semantics |
| Compensatable | PCM / compensation jobs | Agent persistence qual — **scoped**, not full TOOL-LIFE |

**Retry budget:** `ToolRetryPolicy.max_attempts` capped by invoker; global execution retry separate (`execution/retry/policy.py`). **Amplification:** W4 proves bounded dependency admission — **historical @ `f38e455…`**, not TOOL-LIFE-X closure.

**Timeout ≠ failure:** Invoker sets `effect_certainty` on timeout based on whether executor may have started (`_ExternalEffectBoundary`).

---

## 7. TL1–TL15 exit matrix

Evidence SHA for **inventory @ P0** = `76a98cb3f3e4576ab694f03a5b3e270eb7694f9e`. Historical harness evidence cited **only** with original acceptance scope.

| ID | Classification | Canonical owner | Typed contract | Existing mechanism @ HEAD | Evidence (scope) | Missing | Mandatory qual | FRZ (related) | Depends |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TL1** | **VALIDATION REQUIRED** | RuntimeToolInvoker + ToolContract | `timeout_ms >= 1` default 30s | No None default on contract | `tools/core/contracts.py` @ audit SHA; side-effect safety tests | Prove **all** production tool registrations honor contract; long-running allow-list | Per-tool registry audit + gate | FRZ-REL-01 | TL2 |
| **TL2** | **VALIDATION REQUIRED** | ExecutionRuntime + invoker | `execution_deadline` hierarchy §3.0.5 | deadline_scope + protected-work denial + contract timeout | W4 partial; `invoker._tool_result_deadline_exceeded` | End-to-end hierarchy certification across Step→Tool→Attempt→transport | Integrated deadline tests | FRZ-REL-01 | TL1, TL14 |
| **TL3** | **VALIDATION REQUIRED** | CancellationCoordinator + invoker + boundary | Cooperative cancel + external-op cancel port | W4-A hardening in invoker/LLM/policy | W4-A @ `f38e455…` (**harness scope only**) | DB/MCP/HTTP physical cancel paths | Real-infra cancel suite | FRZ-REL-04,05 | TL9 |
| **TL4** | **ALREADY PROVEN** (scoped) | RuntimeToolInvoker | `ToolRetryPolicy`, `_effective_max_attempts` | Bounded attempts; NOT_RETRY_SAFE → 1 | `test_p0_safety_8_*`, W4 retry replay | Per-provider global retry budget certification | TOOL-LIFE adversarial + DIST-SCALE | FRZ-REL-07,11 | TL5, TL6 |
| **TL5** | **VALIDATION REQUIRED** | Invoker + idempotency | `ToolEffectCertainty`, `InvocationStatus.UNCERTAIN` | Typed uncertain; not auto-success | `test_tools_side_effect_safety.py` @ audit SHA | Formal UNKNOWN_OUTCOME operator runbooks; cross-provider | Scenario 22 + operator matrix | FRZ-REL-03,05 | TL6 |
| **TL6** | **ALREADY PROVEN** (scoped) | RuntimeToolInvoker + governance | `SideEffectRetrySafety` | Blind retry blocked on timeout | `test_side_effect_timeout_does_not_blind_retry` | Governance on **retry attempt** re-auth full matrix | Batch B negatives | FRZ-REL-03, FRZ-GOV-* | TL5 |
| **TL7** | **IMPLEMENTATION REQUIRED** | Execution + watchdog (future) | STALLED → terminal | W4 stuck detection partial (RUNNING + deadline) | W4-A inventory | Heartbeat/STALLED certification per roadmap (design-only in roadmap) | Watchdog/reconciliation qual | FRZ-REL-01 | TL8, TL15 |
| **TL8** | **VALIDATION REQUIRED** | External op store + worker recovery | Lease + durable op state | W4-C external operation semantics | `f38e455…` W4-C (**not** full tool re-drive) | Worker loss mid-tool takeover rules | Scenarios 15–16, 23–24 | FRZ-REL-05, FRZ-REC-* | TL7, TL9 |
| **TL9** | **VALIDATION REQUIRED** | ExternalOperationStateStore | Stale state errors | Stale terminal guard | W4-C qual doc | Late result after TIMED_OUT on all providers | Scenario 14 | FRZ-REL-05 | TL3, TL8 |
| **TL10** | **IMPLEMENTATION REQUIRED** | SQL/database tool providers | Row/byte/statement timeouts | Provider-local limits | RAG/records limits only | Unified `database.query` boundedness contract + tests | Scenario 20 + Docker DB | FRZ-REL-10 | TL2 |
| **TL11** | **VALIDATION REQUIRED** | MCP adapters via invoker | Same as ToolContract path | MCP tools as catalog providers | **No** TOOL-LIFE MCP qual @ HEAD | Hung MCP, partial response, restart | Real MCP process suite | FRZ-REL-06 | TL14 |
| **TL12** | **VALIDATION REQUIRED** | HTTP provider (`tools/providers/http`) | `timeout_s` on params | HTTP client timeout | Unit-level | Lost ack, 429 storm under tool retry | Real HTTP test service | FRZ-REL-06,11 | TL4, TL5 |
| **TL13** | **VALIDATION REQUIRED** | Qualification stage | §3.0.5 real infrastructure | W4 uses mix; mocks dominate tool providers | HARNESS-W4 (**insufficient alone**) | Docker DB, fault injection, MCP live | Batch E qual | FRZ-REG-06,08 | TL10–12 |
| **TL14** | **ALREADY PROVEN** (scoped) | Architecture gates | No agent-local invoker | U5 zero-bypass tests | `test_platform_execution_unification_u5_final_zero_bypass.py` | Continuous gate on new bypass paths | Architecture gate in TOOL-LIFE closure | FRZ-BND-03, FRZ-EXE-02 | TL1–TL12 |
| **TL15** | **BLOCKED** | Parent TOOL-LIFE-X | EC1–EC15 aggregate | P0 pre-audit only | This document | TL1–TL14 closure | Parent independent audit | — | all |

---

## 8. Mandatory test matrix (roadmap §3.0.5 — 24 scenarios)

| # | Scenario | Existing tests (representative) | Gap | Qual type |
| --- | --- | --- | --- | --- |
| 1 | Normal success | `test_tool_invocation_registry.py`, provider unit tests | — | unit |
| 2 | Fast failure | invoker error mapping tests | — | unit |
| 3 | Connection timeout | HTTP provider unit | Real network | targeted + real-infra |
| 4 | Execution timeout | invoker timeout paths | All tool classes | unit + integration |
| 5 | Lock timeout | PostgreSQL search timeout test | DB lock held | real-infra |
| 6 | Cancellation | W4-A cancellation qual | Tool mid-executor | integration |
| 7 | User abort | cooperative cancel | — | W4-A |
| 8 | Parent abort | graph concurrent work cancel | Child tool cascade rules | integration |
| 9 | Retryable error | retry policy tests | — | unit |
| 10 | Non-retryable | NOT_RETRY_SAFE | — | unit |
| 11 | Retry exhaustion | invoker attempt loop | ABANDONED typing | integration |
| 12 | Lost acknowledgement | partial external-op | **missing dedicated** | real-infra |
| 13 | Duplicate response | idempotency tests | HTTP duplicate body | integration |
| 14 | Late response | W4-C stale external op | Tool result resurrection | integration |
| 15 | Process crash | crash injection port tests | Production path | qual |
| 16 | Worker lost | W4 distributed cancel | Tool permit detach | integration |
| 17 | Provider unavailable | circuit/retry qual W4-D | — | harness-scoped |
| 18 | Stalled operation | **gap** | STALLED semantics | implementation + test |
| 19 | Huge response | modality/preview bounds | Per-tool payload ceiling | implementation |
| 20 | Huge DB result | **gap** unified | SQL row cap | real-infra |
| 21 | Queue wait > deadline | dependency admission timeout | Prove deadline includes queue | integration |
| 22 | Unknown outcome | side_effect_safety UNCERTAIN | Operator reconciliation | unit + qual |
| 23 | Abandonment | retry exhaust + uncertain | Typed ABANDONED terminal | implementation |
| 24 | Successful recovery | W3 recovery admission, suspended op | Tool-specific recovery | integration |

**Reusable historical evidence (do not re-attribute):**

| Artifact | SHA | Original scope |
| --- | --- | --- |
| HARNESS-W4 parent consolidation | `f38e455ccc391dad9a6a64d7867a33cfc5c9b21d` | Scale/resilience/cancellation — **not** TOOL-LIFE-X |
| HARNESS-W4-R1 strict production admission | `cde779f149cc8696fb5c1fd86c29083ffd90f1bd` | Dependency admission chain |
| GOV-X2 | `e2d7a40b291c658677cc6d981183e20957c7cf0e` / code `b233c61b123464726124dc02daaee21f55d3b856` | Governance ≠ execution ingress |
| SKILL-X implementation tests | `e00a0aab20d98a7050dc26a0e46324f635afe798` | Skill lifecycle only |
| STATE-X / TRACE-X | checklist historical rows | Durability/trace — not tool lifecycle closure |

---

## 9. Provider / adapter qualification matrix (bounded rows)

| Provider class | Adapter locus | Timeout/cancel @ HEAD | Real-infra qual | Classification |
| --- | --- | --- | --- | --- |
| SQL/database | integrations + data tools | Integration-specific; query canceled test in product search | **Required** Batch D | VALIDATION REQUIRED |
| HTTP/API | `tools/providers/http` | `timeout_s` param | Fault-inject service | VALIDATION REQUIRED |
| MCP | catalog MCP providers | Invoker envelope only | Hung server process | VALIDATION REQUIRED |
| Browser/sandbox | `sandbox/extended_service` | `timeout_s` 30–300 | Long automation abandon | VALIDATION REQUIRED |
| Worker shutdown | host runtime + W4 cancel | Cooperative shutdown paths | Running tool + shutdown | VALIDATION REQUIRED |
| Child executions | UER child admission | GOV-X2 child rules | Parent cancel → child | VALIDATION REQUIRED |
| Late results | `external_operation_state_store` | Stale guard | Timed out then late | VALIDATION REQUIRED |
| Response size | previews / RAG limits | Partial | Huge payload rejection | IMPLEMENTATION REQUIRED |

Test doubles (`RegistryToolExecutor`, harness reference host) = **not** real-provider certification.

---

## 10. ECAP / enterprise safety (P0)

| Invariant | Assessment |
| --- | --- |
| Exactly-one tool lifecycle owner | **PASS @ design** — `RuntimeToolInvoker` + UER; no second Tool-only SM proposed |
| Exactly-one composition root | **PASS** — production invoker composition |
| Governance ≠ Execution | **PASS** — historical GOV-X2; Skills use ToolRuntime ceiling |
| Proposal ≠ Permission ≠ Execution | **PASS** — governance ports on invoker |
| Execution identity continuity | **PASS** — `require_active_execution_identity` on continuation host |
| Typed transitions | **PARTIAL** — ABANDONED/STALLED not fully typed on tool seam |
| Tenant isolation | **PASS** scoped TENANT-X / SKILL-X — not re-run @ P0 |
| Provider replaceability | **PASS** architecture — qual per TOOL-LIFE |
| No shadow registry | **PASS** @ scanned ingress |
| No adapter bypass | **PASS** U5 gates — maintain regression |
| Fail-closed | **PASS** missing policy/admission |
| FRZ-BND-03 | **OPEN** globally — no promotion |
| Global FRZ PASS | **0** — ECAP-P4 |

**Architecture decision:** **not required** for P0 — existing owners sufficient if certification extends semantics in-place (no new coordinator/manager per W4-A decision).

---

## 11. Implementation batch plan (post-P0)

### Batch A — Canonical lifecycle contracts & deadline hierarchy

- **Scope:** Map roadmap states to existing UER + invoker events; document `remaining_budget` propagation; strengthen protected-work + step deadline tests.
- **Owners:** UER / execution runtime + Nexus tools.
- **Modules:** `runtime/execution/deadline_*`, `graph_runner.py`, `invoker.py`, contracts `execution_deadline/*`.
- **Prerequisites:** P0 independent audit.
- **Delivers:** Typed lifecycle fact emission spec; TL2 closure tests.
- **Risk:** Medium — touch core execution path.
- **Acceptance:** TL2 green on integrated pytest + architecture gate.
- **Audit boundary:** Independent SHA after Batch A only.

### Batch B — Cancellation, retry/idempotency, unknown-outcome

- **Scope:** TL3–TL6, TL5 operator semantics; retry re-authorization matrix; UNCERTAIN → ABANDONED policy table.
- **Owners:** `RuntimeToolInvoker`, idempotency coordinator, governance evidence.
- **Modules:** `invoker.py`, `idempotency_*`, `test_p0_safety_8_*`, governance bridges.
- **Prerequisites:** Batch A deadline proofs.
- **Delivers:** Full negative package for blind retry / unknown outcome.
- **Risk:** High — side-effect safety.
- **Acceptance:** TL4–TL6 **PASS**; scenarios 9–11, 22.
- **Audit boundary:** Batch B qual doc.

### Batch C — Stalled/abandoned, worker loss, recovery

- **Scope:** TL7–TL9, TL8; watchdog/reconciliation design; worker takeover rules.
- **Owners:** Execution runtime + external operations + long_running.
- **Modules:** `external_operations/*`, `continuation_aware_catalog_tool_host.py`, recovery qual patterns from W3/W4.
- **Prerequisites:** Batch B uncertain semantics frozen.
- **Delivers:** STALLED/ABANDONED typed terminals without second SM.
- **Risk:** High — distributed correctness.
- **Acceptance:** Scenarios 15–16, 18, 23–24.
- **Audit boundary:** Batch C.

### Batch D — Provider SQL/HTTP/MCP/browser boundaries

- **Scope:** TL10–TL12; unified DB limits; MCP/HTTP real-infra qual.
- **Owners:** Tools providers + integrations (adapters only).
- **Modules:** `tools/providers/http`, database tools, MCP wiring, sandbox.
- **Prerequisites:** Batch A deadline hierarchy.
- **Delivers:** Provider qual rows §9 closed.
- **Risk:** Medium — vendor-specific.
- **Acceptance:** TL10–TL12; scenarios 3–5, 12–14, 19–20.
- **Audit boundary:** Batch D (+ Docker/real-infra logs under `.tmp/proof/`).

### Batch E — Integrated adversarial + parent TOOL-LIFE-X acceptance

- **Scope:** TL13–TL15; roadmap §3.0.5 exit rollup; DIST-SCALE-X handoff package.
- **Owners:** Qualification stage + operator independent audit.
- **Prerequisites:** Batches A–D.
- **Delivers:** Parent closure reconciliation doc (pattern SKILL-X).
- **Risk:** Audit gating.
- **Acceptance:** All TL1–TL15 **PASS**; **not** ARCH-FREEZE.

---

## 12. Blockers and authorizations

| Item | Type |
| --- | --- |
| Independent audit of **this P0 commit** | Mandatory before Batch A |
| Real-infrastructure qualification (Docker DB, MCP, HTTP fault) | Operator authorization per env-vendor rule |
| Global `FRZ-REL-*` promotion | **Forbidden** from TOOL-LIFE-X alone (CTRL-X / PROD-Q remain) |
| Heartbeat/STALLED implementation | Operator decision if roadmap certification requires code vs observability-only |

---

## 13. P0 deliverable summary

| Deliverable | Status |
| --- | --- |
| Closed-world inventory | §3 |
| TL1–TL15 matrix | §7 |
| 24-scenario test matrix | §8 |
| Provider matrix | §9 |
| Batch plan A–E | §11 |
| Production code changes | **none** |
| Changed files (P0) | This qualification record only |

**Next mandatory implementation batch:** **TOOL-LIFE-X Batch A** (canonical lifecycle + deadline hierarchy).

---

**TOOL-LIFE-X P0 — READY FOR INDEPENDENT AUDIT**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
