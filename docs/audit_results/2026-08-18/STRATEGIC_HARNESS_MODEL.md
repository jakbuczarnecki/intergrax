# STRATEGIC_HARNESS_MODEL — audit snapshot

**Layer:** STRATEGIC_HARNESS_MODEL  
**Protocol:** [Protocol v2](../AUDIT_PROTOCOL.md)  
**Campaign:** [2026-08-18](README.md)

## Verdict

| Field | Value |
|-------|-------|
| **Verdict** | **FAIL** |
| **audited_sha** | `9658224495c775fcefd55ab52bbcc7a94c84fb50` |
| **post_sync_sha** | `363a8a1f10ea4198d479c3a708af6122ac72144b` |
| **Operator decision** | accepted 2026-08-18 |
| **Findings** | 10 total — 10 ACCEPTED |

**FAIL means:** the strategic harness invariant is not yet proven or enforced universally.

**FAIL does NOT mean:** the entire runtime is non-functional or all execution is ungoverned.

## Positive evidence (material strengths)

- **HarnessKernel is real**, not paper architecture — policy, trace, gateways, state merge, and budgets exist on wired paths.
- **Nexus** wires substantial policy, middleware, events, workspace, sandbox, memory, and context infrastructure.
- **RuntimeExecutionContext** tool invocation is routed through a gateway on governed paths.
- **RuntimeToolGateway** provides allow-list, tool-hook, and runtime enforcement.
- Meaningful **external side-effect policy is fail-closed** where explicitly used.
- Typed **TaskId / RunId / AttemptId** contracts exist in the platform model.
- **UER documentation** already distinguishes platform implementation maturity from full production qualification in several places.

## Findings summary

| ID | Severity | Category | Title | Status |
|----|----------|----------|-------|--------|
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01 | HIGH | ARCHITECTURE DEFECT | HarnessKernel is not the universal pre-execution safety boundary | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-02 | HIGH | BOUNDARY VIOLATION | UAEP resume path bypasses HarnessKernel | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-03 | MEDIUM | BOUNDARY VIOLATION | UAEP execute_step fails open when kernel session is absent | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-04 | HIGH | IMPLEMENTATION / ARCHITECTURE DRIFT | Nexus and direct ACP do not universally share one canonical author hook | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05 | MEDIUM | ARCHITECTURE DEFECT | Production host/profile requirements are not structural | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06 | HIGH | IMPLEMENTATION / ARCHITECTURE DRIFT | Critical author surface is not typed-only | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-07 | MEDIUM | BOUNDARY VIOLATION | Core execution runtime promotes product/LKW-shaped result keys | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-08 | HIGH | RELIABILITY | Nexus → ACP bridge breaks AttemptId continuity | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-09 | HIGH | IMPLEMENTATION DEFECT / TEST GAP | Direct ACP constructs RuntimeExecutionContext without required attempt_id | ACCEPTED |
| AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10 | MEDIUM | PROCESS / CLAIM | Current architecture-complete/platform-ready claims exceed current verified state | ACCEPTED |

---

## FINDING 01

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01  
**Severity:** HIGH  
**Category:** ARCHITECTURE DEFECT  
**Title:** HarnessKernel is not the universal pre-execution safety boundary

**Evidence paths at audited_sha:**

- `intergrax/agents/authoring/step_loop.py`
- `intergrax/agents/authoring/uaep_step_bridge.py`
- `intergrax/contracts/runtime_execution_context.py`
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`

**Core evidence:**

- direct ACP calls `agent.on_next_step(...)` before `HarnessKernel.execute_step(...)`
- UAEP calls `agent.run_step(...)` before `HarnessKernel.execute_step(...)`
- author/runtime paths can invoke immediate tool execution before the kernel processes the resulting StepOutcome
- tool gateways have separate enforcement — not every tool call is ungoverned
- the defect is that HarnessKernel itself is not the universal pre-execution boundary promised by the strategic model

**Consequence:** Platform safety enforcement is distributed rather than structurally guaranteed by one universal boundary; direct side effects performed by agent code can occur before kernel policy/state/budget handling.

**Target state:** Either (A) agent author code produces pure/controlled intent and all effects happen only after kernel authorization, or (B) architecture explicitly defines a compositional enforcement model with mandatory equivalent boundaries and proves that no meaningful side effect can bypass them.

**Status:** ACCEPTED

---

## FINDING 02

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-02  
**Severity:** HIGH  
**Category:** BOUNDARY VIOLATION  
**Title:** UAEP resume path bypasses HarnessKernel

**Evidence:**

- `intergrax/agents/uaep.py`
- `_execute_step_with_resume(...)`
- direct `agent.resume_step(...)`
- fallback direct `uaep_agent.run_step(...)`
- no `execute_uaep_step_via_kernel(...)`
- no `HarnessKernel.execute_step(...)`

**Qualification:** Outer UAEP middleware/governance still exists. The proven defect is specifically loss of kernel-equivalent step semantics.

**Consequence:** Normal and resumed execution can have different policy/state/budget/trace/declarative-action semantics.

**Target:** Resume must re-enter the same canonical governed step boundary as normal execution, while preserving resume semantics.

**Status:** ACCEPTED

---

## FINDING 03

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-03  
**Severity:** MEDIUM  
**Category:** BOUNDARY VIOLATION  
**Title:** UAEP execute_step fails open when kernel session is absent

**Evidence:**

- `intergrax/agents/uaep.py`
- if kernel context exists → kernel bridge
- otherwise → direct `agent.run_step(...)`

**Qualification:** No separate normal production path without a kernel was proven during this audit; severity is MEDIUM rather than HIGH.

**Target:** Production/certified execution must fail closed when required kernel context is missing. Any dev/test bypass must be explicit and separately named.

**Status:** ACCEPTED

---

## FINDING 04

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-04  
**Severity:** HIGH  
**Category:** IMPLEMENTATION / ARCHITECTURE DRIFT  
**Title:** Nexus and direct ACP do not universally share one canonical author hook

**Evidence:**

- `intergrax/agents/authoring/base.py`
- `intergrax/agents/authoring/patterns/base.py`
- `intergrax/agents/authoring/acp_uaep_shim.py`
- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`

**Observed semantics:**

- direct ACP → `on_next_step()`
- generic IntergraxAgent UAEP → `run_step()` → authored `@step`
- CognitiveAgent UAEP → `run_step()` → shim → `on_next_step()`

The architecture claim that both paths universally converge on `on_next_step()` is not true for all IntergraxAgent subclasses.

**Target:** Choose and enforce one canonical author execution contract, OR define explicit, typed, intentionally different contracts whose guarantees are equivalent. Do not document accidental class-dependent semantics as one universal path.

**Status:** ACCEPTED

---

## FINDING 05

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05  
**Severity:** MEDIUM  
**Category:** ARCHITECTURE DEFECT  
**Title:** Production host/profile requirements are not structural

**Evidence:**

- `intergrax/agents/agent_engine.py`
- `_DEFAULT_UAEP = UAEPExecutor()`
- static `AgentEngine.run_agent(...)`
- direct ACP allows absence of host context
- `intergrax/runtime/policy/runtime_policy_engine.py`

**Qualification:** Meaningful external side-effect policy itself is fail-closed where explicitly used. The defect is ambiguity between fully wired production execution and weaker hostless/dev execution.

**Target:** Production execution requires an explicit production host/profile contract. Hostless execution is explicitly marked and constrained to dev/test/lab use.

**Status:** ACCEPTED

---

## FINDING 06

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06  
**Severity:** HIGH  
**Category:** IMPLEMENTATION / ARCHITECTURE DRIFT  
**Title:** Critical author surface is not typed-only

**Evidence:**

- `intergrax/contracts/agent_step_context.py` — `state_snapshot: dict[str, Any]`, `metadata: dict[str, Any]`, `llm_router: object | None`
- `intergrax/agents/authoring/step_outcome.py` — untyped state/output/artifacts/diagnostics/requested_actions maps
- `intergrax/agents/authoring/acp_uaep_shim.py` — hidden `"uaep_exec_ctx"` escape hatch
- architecture claims untyped dict author surface is not supported

**Target:** A typed capability/context model at the critical author boundary. Internal maps may exist where appropriate, but authors must not depend on unbounded `dict[str, Any]` or hidden RuntimeExecutionContext escape hatches for core execution semantics.

**Status:** ACCEPTED

---

## FINDING 07

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-07  
**Severity:** MEDIUM  
**Category:** BOUNDARY VIOLATION  
**Title:** Core execution runtime promotes product/LKW-shaped result keys

**Evidence:**

- `intergrax/agents/uaep.py`
- `intergrax/agents/authoring/acp_run.py`
- hard-coded promotion of: `search_summary`, `ingest_summary`, `domain_summary`
- direct comment references LKW search/index handoff

**Consequence:** Core platform execution has knowledge of product/application output vocabulary.

**Target:** Generic typed structured-result / diagnostic / artifact transport. Application/product layers interpret domain-specific payloads.

**Status:** ACCEPTED

---

## FINDING 08

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-08  
**Severity:** HIGH  
**Category:** RELIABILITY  
**Title:** Nexus → ACP bridge breaks AttemptId continuity

**Evidence:**

- `intergrax/runtime/nexus/nexus_loop.py` binds/mints active AttemptId
- `intergrax/runtime/nexus/responses/response_schema.py` — RuntimeRequest carries task_id/run_id but no attempt_id
- `intergrax/agents/runtime_request_bridge.py` preserves task_id/run_id but not attempt_id
- `intergrax/agents/authoring/acp_run.py` — `_resolve_acp_session_identity()` always mints a new AttemptId

**Consequence:** One logical Nexus execution can appear as Task T, Run R, Attempt A1 in Nexus, Attempt A2 inside ACP without a retry boundary. This conflicts with canonical retry/resume AttemptId semantics.

**Target:** Preserve the active AttemptId across Nexus → ACP, OR introduce an explicit typed parent/child execution identity model if nesting is truly intended.

**Status:** ACCEPTED

---

## FINDING 09

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-09  
**Severity:** HIGH  
**Category:** IMPLEMENTATION DEFECT / TEST GAP  
**Title:** Direct ACP constructs RuntimeExecutionContext without required attempt_id

**Evidence:**

- `intergrax/contracts/runtime_execution_context.py` requires `attempt_id: AttemptId`
- `intergrax/agents/authoring/acp_uaep_shim.py` — `attach_acp_catalog_exec_ctx()` constructs RuntimeExecutionContext without attempt_id
- `intergrax/agents/authoring/acp_run.py` invokes `attach_acp_catalog_exec_ctx()` in the normal session loop
- `tests/unit/agents/authoring/test_acp_session_identity.py` mocks `attach_acp_catalog_exec_ctx()` in key identity tests

**Evidence limitation:** Based on exact static construction/reachability evidence. The independent auditor did **not** execute a local failing runtime test during this audit.

**Target:** Propagate the canonical active ACP AttemptId into RuntimeExecutionContext and add a regression test exercising the real helper without mocking away the boundary.

**Status:** ACCEPTED

---

## FINDING 10

**ID:** AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10  
**Severity:** MEDIUM  
**Category:** PROCESS / CLAIM  
**Title:** Current architecture-complete/platform-ready claims exceed current verified state

**Evidence:**

- `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md`
- relevant UER maturity/mandatory-path claims

**Qualification:** Do not rewrite history to claim previous gaps were never closed. Previous gaps may have been correctly closed against the scope known then. Protocol v2 discovered a new class of issues; **current** maturity/completeness claims must be reopened/qualified.

**Target:** Preserve historical completion records, but current maturity must explicitly acknowledge unresolved accepted Protocol v2 findings.

**Status:** ACCEPTED

---

## Remediation grouping (not implemented by this task)

These blocks are accepted remediation groupings only — **not** implementation performed by audit persistence.

### SHM-FIX-A — Execution boundary

**Findings:** 01, 02, 03, 04  
**Goal:** one canonical governed execution semantics across direct, Nexus, normal, and resume paths.

### SHM-FIX-B — Identity and typed context

**Findings:** 06, 08, 09  
**Goal:** typed critical author/runtime boundary and correct Task/Run/Attempt continuity.

### SHM-FIX-C — Host and platform neutrality

**Findings:** 05, 07  
**Goal:** explicit production hosting requirements and product-neutral result transport.

### SHM-FIX-D — Maturity and recertification

**Finding:** 10 plus verification requirements produced by A–C  
**Goal:** re-run certification only after implementation blocks are independently verified; current historical Done records remain historical.
