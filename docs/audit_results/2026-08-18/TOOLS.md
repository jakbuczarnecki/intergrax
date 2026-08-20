# TOOLS — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** TOOLS
- **Constituent domains:** TOOLS (ToolRuntime spine · catalog · invocation · idempotency)
- **Tier(s):** Tier-0 `ToolContract` · Tier-1 `ToolRuntime` / `RuntimeToolInvoker` / planner-loop · Nexus tool gateway surfaces
- **audited_sha:** `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/TOOLS.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/TOOLS.md`
- **Scope in:**
  - `ToolRuntime` as canonical execution facade and gateway convergence
  - `ToolContract` / `ToolRegistry` typed catalog model
  - `resolve_allowed_tools_from_config()` and explicit caller allow-list vs `RuntimePolicyBundle.tool_access`
  - `RuntimeToolInvoker` registry/input/output enforcement, declarative policy when wired, timeout/retry
  - `ToolPlanningService` filtering against `allowed_tool_ids`
  - planner-loop invocation ordering (`_invoke_planned_call`, `run_bounded_tool_loop`, `run_tools_context`)
  - hard tool-call budget (`enforce_tool_call_budget`, `BudgetEnforcer`)
  - `IdempotentToolInvoker` / `IdempotencyStore` identity and outcome semantics
  - `ToolContract` side-effect + retry metadata vs automatic retry behavior
  - parallel planner read-only vs mutating tool separation
  - Tool / Skill / Integration responsibility separation
- **Scope out:**
  - remediation implementation
  - second Tool Runtime design
  - TOOL-PRODUCT-ROI catalog expansion
  - TOKEN-TOOLS-1B compact catalog wiring
  - full Governed Execution / meaningful-side-effect re-audit beyond tool-path touchpoints
  - MCP export / sandbox / catalog scale gates (positive controls only)
- **Prior audit reference(s):** Phase **TOOL-ENG** **closed** (36/36); AUDIT-IDEAL §11 **Done**; Protocol v2 [`AGENT_SYSTEM`](AGENT_SYSTEM.md) (AGSYS-03 tool permission bypass — separate layer); [`POLICY_GOVERNANCE`](POLICY_GOVERNANCE.md) (PG-FIX tool-scope spine)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Executive summary

**Verdict: FAIL.** Six accepted findings (5 HIGH, 1 MEDIUM) show explicit caller tool allow-lists can override rather than intersect `RuntimePolicyBundle.tool_access`; `ToolContract.timeout_ms` does not bound wall-clock latency on thread-pool timeout exit; hard tool-call budget is checked after invocation with stale `tool_traces` accounting and may be swallowed as tools-context error; idempotency keys omit canonical tool/operation identity and permit cross-tool cache collision; side-effectful tools may be automatically retried without proven retry-safe authorization; and idempotency ledger records failures as `COMPLETED` without outcome-state distinction. Positive controls: `ToolRuntime` remains the canonical execution facade; `ToolContract` / registry model holds; invoker enforces schema/policy when wired; `ToolPlanningService` respects `allowed_tool_ids`; parallel planner separates read-only from mutating tools; Tool / Skill / Integration split intact; finding set does not require a second Tool Runtime.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-TOOLS-01

**Explicit caller allow-list overrides runtime policy bundle instead of intersecting**

- **Severity:** HIGH
- **Category:** ARCHITECTURE / AUTHORIZATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-GOVERNED-BOUNDARY-INTEGRITY
- **Claim falsified:** Effective tool permissions are a monotonic intersection of applicable authorities (host availability ∩ agent/skill declaration ∩ runtime policy ∩ modality ∩ per-call/invoker scope). No caller-provided list may expand a stricter policy authority.
- **Observation:** `resolve_allowed_tools_from_config()` returns `explicit` immediately when an explicit caller allow-list is supplied. The `RuntimePolicyBundle.tool_access` allow-list is therefore ignored instead of intersected. The gate test `test_explicit_allowed_tools_win_over_bundle` explicitly codifies this precedence.
- **Location:**
  - `intergrax/runtime/policy/tool_policy_resolution.py` — `resolve_allowed_tools_from_config()` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/nexus/tools/tool_access_policy.py` — policy application surfaces @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `tests/unit/runtime/policy/test_tool_policy_resolution.py` — `test_explicit_allowed_tools_win_over_bundle` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. `git show 65aaf33a6a6dba9b336162ec547cd677f4edad91:intergrax/runtime/policy/tool_policy_resolution.py` — early return on explicit allow-list without bundle intersection.
  2. `git show 65aaf33a6a6dba9b336162ec547cd677f4edad91:tests/unit/runtime/policy/test_tool_policy_resolution.py` — test asserts explicit wins over bundle.
- **Impact:** Runtime policy bundle can be bypassed when a caller supplies a broader explicit allow-list; violates monotonic narrowing invariant.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOOLS-02

**Thread-pool timeout does not bound wall-clock latency**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / RESOURCE BOUNDARY
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-GOVERNED-BOUNDARY-INTEGRITY
- **Claim falsified:** `ToolContract.timeout_ms` represents a real execution latency boundary; timeout handling does not synchronously wait for the timed-out worker to finish.
- **Observation:** `RuntimeToolInvoker._execute_once()` uses `ThreadPoolExecutor(max_workers=1)` with `future.result(timeout=timeout_s)`. On timeout, exiting the executor context performs `shutdown(wait=True)`, so the caller may still wait for the worker to finish. Status can report `TIMEOUT` while wall-clock latency remains bounded by actual handler completion, not `ToolContract.timeout_ms`. The timeout unit test checks the `TIMEOUT` result code but not elapsed wall-clock duration.
- **Location:**
  - `intergrax/runtime/nexus/tools/invoker.py` — `RuntimeToolInvoker._execute_once()` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/tools/core/contracts.py` — `ToolContract.timeout_ms` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `tests/unit/runtime/nexus/tools/test_runtime_tool_invoker_policy.py` — timeout result-code test @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. `git show 65aaf33a6a6dba9b336162ec547cd677f4edad91:intergrax/runtime/nexus/tools/invoker.py` — `with ThreadPoolExecutor` + `future.result(timeout=...)`.
  2. Observe context-manager shutdown waits for worker completion after timeout exception.
  3. Timeout test asserts result code only — no wall-clock bound assertion.
- **Impact:** Callers and budget governance may observe latency beyond declared contract timeout; local thread timeout cannot undo an already-running external side effect — cancellation/abandon semantics are not explicit.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOOLS-03

**Hard tool-call budget checked after invocation with stale trace accounting**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / BUDGET GOVERNANCE
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-GOVERNED-BOUNDARY-INTEGRITY
- **Claim falsified:** Hard tool-call budget is reserved/checked before the next invocation can cross the side-effect boundary; usage accounting uses authoritative invocation state, not a trace list populated later. Budget violations defined as hard abort/HITL preserve canonical semantics and are not swallowed as ordinary tool-context errors.
- **Observation:** Planner-loop `_invoke_planned_call()` executes `invoker.invoke()` first and calls `enforce_tool_call_budget(state)` afterward. Budget check uses `tool_calls=len(state.tool_traces)`, but planner-loop traces are assigned to `state.tool_traces` only after `run_bounded_tool_loop()` returns, so mid-loop budget checks can observe stale counts (including zero) after real invocations occurred. Additionally `run_tools_context()` catches general exceptions other than the special HITL pause and records them as error telemetry rather than propagating them, so a budget exception may become a local tools-context error rather than a hard execution stop.
- **Location:**
  - `intergrax/runtime/nexus/tools/tool_loop.py` — `_invoke_planned_call()`, `run_bounded_tool_loop()` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/nexus/tools/plan_context_invocation.py` — `run_tools_context()` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/nexus/budget/budget_ticks.py` — `enforce_tool_call_budget()` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. `git show 65aaf33a6a6dba9b336162ec547cd677f4edad91:intergrax/runtime/nexus/tools/tool_loop.py` — invoke before budget enforcement in loop path.
  2. Trace assignment deferred until loop return — mid-loop `len(state.tool_traces)` stale.
  3. `run_tools_context()` broad exception catch → error telemetry path for budget failures.
- **Impact:** Hard tool-call limits may be exceeded; budget violations may not abort execution with canonical HITL/hard-stop semantics.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOOLS-04

**Idempotency key not bound to canonical tool/operation identity**

- **Severity:** HIGH
- **Category:** IDENTITY / IDEMPOTENCY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-SIDE-EFFECT-SAFETY
- **Claim falsified:** Idempotency identity canonically binds the key to the logical operation. Stored/claimed identity validates tool identity; repeated key with different operation identity fails closed rather than returning unrelated cached result.
- **Observation:** `IdempotentToolInvoker` and `IdempotencyStore` key invocation state only by `(tenant_id, idempotency_key)`. They do not bind ledger identity to `tool_id` or another canonical logical operation fingerprint. A completed invocation for one tool can collide with a later invocation for a different tool using the same tenant/key; the latter can return the cached `ToolExecutionResult` from the first tool without executing or checking tool identity. Unit test validates same-tool same-key dedup only — no cross-tool collision test.
- **Location:**
  - `intergrax/runtime/tools/idempotent_invoker.py` — `IdempotentToolInvoker` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/tools/in_memory_idempotency_store.py` — store keying @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/contracts/idempotency_store.py` — `IdempotencyStore` contract @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `tests/unit/runtime/tools/test_idempotent_invoker.py` — same-tool dedup only @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. Complete tool A with `(tenant, key)` → `COMPLETED`.
  2. Invoke tool B with same `(tenant, key)` → cached result from tool A without identity check.
  3. No cross-tool collision test in unit suite.
- **Impact:** Wrong tool output returned under shared idempotency key; fail-open cross-operation replay.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOOLS-05

**Automatic retry of side-effectful tools without proven retry safety**

- **Severity:** HIGH
- **Category:** SIDE-EFFECT / RETRY ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-SIDE-EFFECT-SAFETY
- **Claim falsified:** Automatic retry of side-effectful tools is positively proven safe via canonical mechanism (idempotent operation semantics + correctly scoped idempotency identity, or explicit retry-safe classification). Unknown-outcome mutating failures are not blindly retried.
- **Observation:** `ToolContract` independently allows `side_effects=True` and `retry_policy.max_attempts > 1`. `RuntimeToolInvoker` retries general exceptions according to `max_attempts` without requiring idempotency key, `IdempotentToolInvoker` presence, explicit retry-safe contract metadata, or retryable error classification. This permits automatic retry of an externally mutating operation whose prior outcome may be unknown.
- **Location:**
  - `intergrax/tools/core/contracts.py` — `ToolContract.side_effects`, `ToolRetryPolicy` @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/tools/execution_models.py` — execution result models @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/nexus/tools/invoker.py` — `RuntimeToolInvoker` retry loop @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. Register mutating tool with `side_effects=True`, `max_attempts > 1`.
  2. Provoke retried exception path — invoker retries without idempotency or retry-safe gate.
  3. No requirement for `IdempotentToolInvoker` wrapper on retry path.
- **Impact:** Duplicate external mutations possible when outcome after failure is unknown; not universal exactly-once against external providers.
- **Confidence:** CONFIRMED

### AUDIT-20260818-TOOLS-06

**Idempotency ledger cannot distinguish outcome states for safe retry**

- **Severity:** MEDIUM
- **Category:** FAILURE / IDEMPOTENCY STATE MODEL GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** TOOLS-SIDE-EFFECT-SAFETY
- **Claim falsified:** Idempotency ledger state/outcome semantics explicitly distinguish outcomes needed for deterministic safe retry decisions. Failures with unknown external outcome are not automatically treated as retry-safe completed operations.
- **Observation:** `RuntimeToolInvoker` commonly returns `ToolExecutionResult.fail(...)` rather than raising. `IdempotentToolInvoker` unconditionally calls `record_completed(...)` on the returned result without inspecting `result.success`. `IdempotencyStore` state model exposes only `STARTED` / `COMPLETED`. It cannot distinguish successful completed operation, known failed-before-effect operation, and failed operation with unknown external outcome. Subsequent same-key calls return the cached failed result as `COMPLETED`.
- **Location:**
  - `intergrax/runtime/tools/idempotent_invoker.py` — `record_completed` without `success` check @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/contracts/idempotency_store.py` — `STARTED` / `COMPLETED` only @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
  - `intergrax/runtime/nexus/tools/invoker.py` — `ToolExecutionResult.fail` return path @ `65aaf33a6a6dba9b336162ec547cd677f4edad91`
- **Reproduction:**
  1. Side-effect invocation fails → `ToolExecutionResult.fail` returned.
  2. `IdempotentToolInvoker` records `COMPLETED` for failed result.
  3. Retry with same key returns cached failure — no distinction for unknown-outcome failures.
- **Impact:** Safe retry decisions cannot be made deterministically; fail-closed behavior for unknown external outcome is not preserved.
- **Confidence:** CONFIRMED

## Falsification log (negative results)

1. **ToolRuntime not canonical execution facade** — not falsified; all audited paths converge on `ToolRuntime` → gateway → `RuntimeToolInvoker`.
2. **ToolContract / ToolRegistry typed catalog invalid** — not falsified; registry lookup and schema enforcement remain on invoker path.
3. **RuntimeToolInvoker lacks registry/input/output enforcement** — not falsified when wired; schema validation and policy hooks present on primary path.
4. **ToolPlanningService ignores allowed_tool_ids** — not falsified; planner filters against allowed set.
5. **Parallel planner does not separate read-only from mutating** — not falsified at audited SHA.
6. **Tool / Skill / Integration responsibility collapse** — not falsified; boundaries documented and observed on primary paths.
7. **Second Tool Runtime required** — not falsified; gaps are boundary integrity and side-effect safety on existing spine.
8. **Prior TOOL-ENG closeout never occurred** — not falsified; historical **Done** / **closed** rows remain valid delivery facts; this audit records Protocol v2 residual gaps.

## Prior-audit comparison

First canonical Protocol v2 `TOOLS` layer snapshot at `65aaf33a6a6dba9b336162ec547cd677f4edad91`. Supplements — does not rewrite — Phase **TOOL-ENG** harness closeout and AUDIT-IDEAL §11 delivery. Discoveries are governed-boundary integrity (permissions, timeout, budget ordering) and side-effect safety (idempotency identity, retry authorization, outcome-state model) beyond prior **Done** registers.

## Provider / backend abstraction

`NOT APPLICABLE — TOOLS scope is platform tool execution boundary, catalog contracts, and invocation governance; external provider behavior is referenced only where timeout/retry/idempotency cannot guarantee vendor exactly-once.`

## Positive controls

1. **ToolRuntime canonical facade** — agents and Nexus converge on `ToolRuntime` for side effects @ audited SHA.
2. **ToolContract / ToolRegistry** — typed catalog model with schema, risk, retry/timeout metadata @ audited SHA.
3. **RuntimeToolInvoker enforcement** — registry lookup, input/output schema validation, declarative policy when wired @ audited SHA.
4. **ToolPlanningService allowed_tool_ids filter** — planner narrows against allowed set @ audited SHA.
5. **Parallel planner read/mutate separation** — read-only tools separated from mutating tools in parallel batch path @ audited SHA.
6. **Tool / Skill / Integration split** — skills compose `tool_ids`; integrations behind handlers @ audited SHA.
7. **No second Tool Runtime required** — findings target boundary integrity and side-effect semantics on existing spine.

**FAIL qualification:** verdict means permission intersection, resource timeout, budget ordering, idempotency identity, retry safety, and outcome-state gaps remain — **not** that the ToolRuntime model or TOOL-ENG delivery is invalid.

## Root-cause remediation grouping

Planning only — **audit persistence does NOT implement remediation.**

### TOOLS-GOVERNED-BOUNDARY-INTEGRITY — permissions, timeout, budget ordering

**Findings:** 01, 02, 03

**Primary plan owner:** TOOLS plan hub

One fail-closed execution boundary where permissions, resource limits, and budgets cannot be weakened by caller precedence, post-execution checks, or stale trace accounting. Reuse canonical policy/tool-scope owners and `RunBudget` / `BudgetEnforcer` — no second budget subsystem.

### TOOLS-SIDE-EFFECT-SAFETY — idempotency identity, retry safety, outcome states

**Findings:** 04, 05, 06

**Primary plan owner:** TOOLS plan hub

Safe deterministic mutating-tool semantics: canonical operation identity on idempotency keys, positive authorization for side-effect retries, explicit ledger outcome states. Do not claim universal exactly-once against external providers.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `65aaf33a6a6dba9b336162ec547cd677f4edad91`; current `development` HEAD was not re-audited.
- TOOLS-02 does not prescribe unsafe thread killing — abandonment/cancellation semantics must be explicit.
- TOOLS-03 does not invent a second budget subsystem — reuse `RunBudget` / `BudgetEnforcer`.
- TOOLS-05 does not claim universal exactly-once against external providers.
- Tests are supporting evidence, not standalone proof.
- Remediation not performed in this task.

## Open questions / blocked items

- TOOLS-02: explicit abandon-vs-cancel semantics for in-flight external effects — operator decision deferred to remediation.
- TOOLS-06: typed idempotency outcome enum vs extended ledger fields — prefer typed contract over string concatenation identity.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-TOOLS-01` … `AUDIT-20260818-TOOLS-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
