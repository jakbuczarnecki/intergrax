# STRATEGIC_HARNESS_MODEL — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** STRATEGIC_HARNESS_MODEL
- **Tier(s):** conceptual cross-domain slice spanning Tier-1 execution/runtime, Tier-2 agent authoring, and Tier-3 production-host contract where relevant
- **audited_sha:** `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Status:** COMPLETE
- **Auditor:** OpenAI ChatGPT / GPT-5.6 Sol — independent auditor
- **Architecture doc(s):**
  - `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`
  - `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md`
  - `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`
- **Scope in:**
  - direct ACP author execution
  - Nexus/UAEP graph-node execution
  - HarnessKernel boundary semantics
  - resume and missing-kernel behavior
  - Task/Run/Attempt identity bridge
  - critical author context typing
  - production host/profile boundary
  - core structured-result neutrality
  - current maturity/completeness claims
- **Scope out:**
  - implementation/remediation
  - unrelated platform domains
  - broad product production qualification
  - tool/integration internals except where directly required as boundary evidence
- **Prior audit reference(s):** pre-Protocol-v2 working STRATEGIC_HARNESS_MODEL audit was revalidated; this Protocol v2 result at audited_sha is the canonical result. No prior Protocol v2 canonical layer result.
- **post_sync_sha:** `def29be1adf2e099c300b7a8471c32b946e9c957`
- **Exact audit-start time:** not captured before first Protocol v2 persistence; date-level UTC precision is preserved rather than fabricating a clock time.

### Correction provenance (Protocol v2 conformance — not a re-audit)

- **initial_sync_sha:** `363a8a1f10ea4198d479c3a708af6122ac72144b`
- **initial_traceability_sha:** `1c6341021c830eeba365f23e000a8028aee0c676`
- Independent verification found documentation/protocol-conformance defects in the persisted layer artifact.
- This correction completes required Protocol v2 evidence/schema fields.
- **No finding meaning, severity, verdict, accepted status, or audited_sha changed.**
- Category spelling and primary-category normalization (`IMPLEMENTATION/ARCHITECTURE DRIFT`, `IMPLEMENTATION DEFECT` with related TEST GAP note for finding 09) were Protocol conformance only — not an operator decision change.
- Git history remains the provenance of the originally persisted form.

## Executive summary

**Verdict: FAIL.** The strategic harness invariant is not yet proven or enforced universally. Ten accepted findings (6 HIGH, 4 MEDIUM) show distributed rather than structurally guaranteed pre-effect governance, resume/kernel divergence, identity bridge gaps, untyped author surfaces, product-shaped core transport, and maturity claims that exceed independently verified state. HarnessKernel, Nexus wiring, and tool gateways are materially real on wired paths — FAIL does not mean the entire runtime is non-functional or all execution is ungoverned.

## Verdict

**FAIL**

## Findings

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01

**HarnessKernel is not the universal pre-execution safety boundary**

- **Severity:** HIGH
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** HarnessKernel (or equivalent compositional boundary) is the universal pre-execution safety boundary before meaningful side effects; architecture presents kernel-governed execution as the structural guarantee (`docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` L87-L89, L133 @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`).
- **Observation:** Direct ACP calls `agent.on_next_step(...)` before `HarnessKernel.execute_step(...)`. UAEP bridge calls `agent.run_step(...)` before `HarnessKernel.execute_step(...)`. Author/runtime paths can invoke immediate tool execution before the kernel processes the resulting `StepOutcome`. Tool gateways provide separate enforcement on governed paths — not every tool call is ungoverned; the defect is that HarnessKernel itself is not the universal pre-execution boundary.
- **Location:**
  - `intergrax/agents/authoring/step_loop.py:L48-L57` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/uaep_step_bridge.py:L197-L204` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md:L87-L89,L133` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/step_loop.py` — inspect `advance_step`: `on_next_step` at L48 precedes `HarnessKernel.execute_step` at L57.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/uaep_step_bridge.py` — inspect `execute_uaep_step_via_kernel`: `run_step` at L197 precedes `HarnessKernel.execute_step` at L204.
- **Impact:** Platform safety enforcement is distributed rather than structurally guaranteed by one universal boundary; direct side effects performed by agent code can occur before kernel policy/state/budget handling under production stresses.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-02

**UAEP resume path bypasses HarnessKernel**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Normal, retry, and resume execution use the same canonical governed step boundary with HarnessKernel-equivalent semantics (UER architecture and Protocol v2 target invariants).
- **Observation:** `_execute_step_with_resume(...)` calls `agent.resume_step(...)` or fallback `uaep_agent.run_step(...)` with no `execute_uaep_step_via_kernel(...)` and no `HarnessKernel.execute_step(...)`. Outer UAEP middleware/governance still exists; the proven defect is specifically loss of kernel-equivalent step semantics on resume.
- **Location:** `intergrax/agents/uaep.py:L730-L744` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/uaep.py`
  2. Inspect `_execute_step_with_resume` (L730-L744): no `HarnessKernel` or `execute_uaep_step_via_kernel` on resume path.
  3. Compare `execute_step` (L784-L798): kernel bridge present when `KERNEL_SESSION` metadata exists.
- **Impact:** Normal and resumed execution can have different policy/state/budget/trace/declarative-action semantics.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-03

**UAEP execute_step fails open when kernel session is absent**

- **Severity:** MEDIUM
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Certified production paths must fail closed when required kernel/governance context is missing.
- **Observation:** `execute_step` routes through kernel bridge only when `ctx.metadata` contains `KERNEL_SESSION`; otherwise falls through to direct `agent.run_step(step, ctx)`. No separate normal production path without kernel was proven during this audit.
- **Location:** `intergrax/agents/uaep.py:L784-L798` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/uaep.py`
  2. Inspect `execute_step` L794-L798: `kernel_ctx is not None` → kernel bridge; else → `run_step` without fail-closed guard.
- **Impact:** Missing kernel context allows weaker execution semantics if such a path is reachable in production configuration.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-04

**Nexus and direct ACP do not universally share one canonical author hook**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Claim falsified:** Both entry paths converge on the same author hook — `on_next_step()` — before platform-governed execution (`docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` L94, L127-L132 @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`).
- **Observation:** Direct ACP → `on_next_step()`. Generic `IntergraxAgent` UAEP → `run_step()` → authored `@step`. `CognitiveAgent` UAEP → `run_step()` → shim → `on_next_step()`. Architecture claim that both paths universally converge on `on_next_step()` is not true for all `IntergraxAgent` subclasses.
- **Location:**
  - `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md:L94,L127-L132` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/base.py:L172-L205` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_uaep_shim.py:L219` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — read L94 and L127-L132 convergence claims.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/base.py` — `on_next_step` calls `run_step` (L172); generic `run_step` invokes `@step` methods (L205-L209).
  3. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_uaep_shim.py` — shim path calls `on_next_step` (L219).
- **Impact:** Class-dependent author semantics undermine one canonical certified contract; governance and typing guarantees may differ by agent subclass and entry path.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-05

**Production host/profile requirements are not structural**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** Production execution requires explicit production host/profile wiring; hostless execution is dev/test/lab only.
- **Observation:** `AgentEngine` uses static `_DEFAULT_UAEP = UAEPExecutor()` when no custom executor/event bus is supplied. `run_agent` accepts `RuntimeRequest` without structural production-host/profile requirement. Meaningful external side-effect policy itself is fail-closed where explicitly used; defect is ambiguity between fully wired production execution and weaker hostless/dev execution.
- **Location:**
  - `intergrax/agents/agent_engine.py:L34,L157,L160-L175` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/runtime/policy/runtime_policy_engine.py:L36-L50` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/agent_engine.py` — `_DEFAULT_UAEP` at L34; `_resolve_static_executor` returns it at L157; `run_agent` has no host/profile gate at L160-L175.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/runtime/policy/runtime_policy_engine.py` — `evaluate_meaningful_side_effect` fail-closed default at L36-L50.
- **Impact:** Production and dev execution boundaries are not structurally separated; weaker default executor path may be used without explicit lab marking.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-06

**Critical author surface is not typed-only**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Claim falsified:** `AgentStepContext` / `StepOutcome` and related author surfaces at the critical boundary are typed capability models, not unbounded `dict[str, Any]` author dependencies (architecture claims untyped dict author surface is not supported).
- **Observation:** `AgentStepContext` exposes `state_snapshot: dict[str, Any]`, `metadata: dict[str, Any]`, `llm_router: object | None`. `StepOutcome` uses untyped state/output/artifacts/diagnostics/requested_actions maps. `acp_uaep_shim` provides hidden `"uaep_exec_ctx"` escape hatch in metadata.
- **Location:**
  - `intergrax/contracts/agent_step_context.py:L30-L32` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/step_outcome.py:L17-L34` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_uaep_shim.py:L77,L132` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/contracts/agent_step_context.py` — L30-L32 untyped fields.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/step_outcome.py` — L17-L34 untyped maps.
  3. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_uaep_shim.py` — `uaep_exec_ctx` metadata channel L77, L132.
- **Impact:** Authors can depend on unbounded dict channels and hidden runtime escape hatches at the critical boundary, weakening platform-owned effect control and static verification.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-07

**Core execution runtime promotes product/LKW-shaped result keys**

- **Severity:** MEDIUM
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Claim falsified:** Product/application vocabulary must not live in core execution result transport; core platform execution must use generic typed structured-result transport.
- **Observation:** `uaep.py` and `acp_run.py` hard-code promotion of `search_summary`, `ingest_summary`, `domain_summary` into structured result/route extras. Direct comment in `acp_run.py` references LKW search/index handoff.
- **Location:**
  - `intergrax/agents/uaep.py:L560-L567` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_run.py:L565-L570` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/uaep.py` — L562-L567 domain summary promotion loop.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_run.py` — L565-L570 LKW handoff comment and promotion loop.
- **Impact:** Core platform execution encodes product/application output vocabulary, coupling Tier-1 runtime to LKW/product semantics.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-08

**Nexus → ACP bridge breaks AttemptId continuity**

- **Severity:** HIGH
- **Category:** RELIABILITY
- **Status at publication:** ACCEPTED
- **Claim falsified:** `TaskId`/`RunId`/`AttemptId` propagate across internal bridges (e.g. Nexus → ACP) unless an explicit typed parent/child attempt relationship exists.
- **Observation:** `NexusLoop` binds/mints active `AttemptId`. `RuntimeRequest` carries `task_id`/`run_id` but no `attempt_id`. `runtime_request_bridge` preserves `task_id`/`run_id` but not `attempt_id`. `_resolve_acp_session_identity()` always mints a new `AttemptId`.
- **Location:**
  - `intergrax/runtime/nexus/nexus_loop.py:L386-L397` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/runtime/nexus/responses/response_schema.py:L151-L152` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/runtime_request_bridge.py:L54-L55` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_run.py:L73-L89` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/runtime/nexus/nexus_loop.py` — bind/mint AttemptId L386-L397.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/runtime/nexus/responses/response_schema.py` — RuntimeRequest fields L151-L152 (no attempt_id).
  3. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/runtime_request_bridge.py` — metadata task_id/run_id only L54-L55.
  4. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_run.py` — `mint_attempt_id()` at L89.
- **Impact:** One logical Nexus execution can appear as Task T, Run R, Attempt A1 in Nexus and Attempt A2 inside ACP without a retry boundary, conflicting with canonical retry/resume AttemptId semantics and trace correlation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-09

**Direct ACP constructs RuntimeExecutionContext without required attempt_id**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Status at publication:** ACCEPTED
- **Claim falsified:** `RuntimeExecutionContext` requires `attempt_id: AttemptId`; canonical active ACP AttemptId must propagate into runtime execution context at the ACP boundary.
- **Observation:** `attach_acp_catalog_exec_ctx()` constructs `RuntimeExecutionContext` at L102-L108 without `attempt_id`, while the contract requires it at L84. `acp_run.py` invokes `attach_acp_catalog_exec_ctx()` in the normal session loop. Key identity tests mock away `attach_acp_catalog_exec_ctx()`.
- **Location:**
  - `intergrax/contracts/runtime_execution_context.py:L82-L107` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_uaep_shim.py:L102-L108` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `intergrax/agents/authoring/acp_run.py:L471-L479` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `tests/unit/agents/authoring/test_acp_session_identity.py:L117,L135` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/contracts/runtime_execution_context.py` — required `attempt_id` L84.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_uaep_shim.py` — construction without `attempt_id` L102-L108.
  3. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:intergrax/agents/authoring/acp_run.py` — call site in session loop L471-L479.
  4. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:tests/unit/agents/authoring/test_acp_session_identity.py` — mocks at L117, L135.
- **Impact:** Runtime execution context at the ACP catalog boundary may lack canonical attempt identity; regression risk is masked by tests that mock the defective helper.
- **Confidence:** CONFIRMED — No independent failing runtime test was executed during the audit. Confidence is based on directly read constructor contract + directly read reachable call path.

### AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10

**Current architecture-complete/platform-ready claims exceed current verified state**

- **Severity:** MEDIUM
- **Category:** PROCESS / CLAIM
- **Status at publication:** ACCEPTED
- **Claim falsified:** Current maturity/completeness claims (A4/I4, architecture-complete, platform-ready gates Done) accurately reflect independently verified invariant closure at audited_sha.
- **Observation:** Architecture and plan documents at audited_sha assert ACP + ACP-CLOSE + ACP-FINISH + AUDIT-IDEAL Done, A4/I4 maturity, and architecture-complete DoD closure while implementation exhibits accepted Protocol v2 boundary, identity, typing, and neutrality gaps. Do not rewrite history — prior gaps may have been correctly closed against then-known scope; **current** claims must be reopened/qualified.
- **Location:**
  - `docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md:L35,L368-L374` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md:L220-L226,L238` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
  - `docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md:L29-L41,L77` @ `9658224495c775fcefd55ab52bbcc7a94c84fb50`
- **Reproduction:**
  1. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:docs/project/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — platform-ready Done claim L35; A4/I4 maturity L368-L374.
  2. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — A4/I4 maturity L220-L226, L238.
  3. `git show 9658224495c775fcefd55ab52bbcc7a94c84fb50:docs/project/maintainers/plans/AGENT_CONTRACTS_AND_ASSEMBLY.md` — Done rows L29-L41, L77.
  4. Cross-check against findings 01–09 evidence at same audited_sha.
- **Impact:** Operators and implementers may treat harness path as fully invariant-closed when accepted Protocol v2 findings show material open gaps; remediation priority and production qualification posture may be misjudged.
- **Confidence:** CONFIRMED

## Falsification log (negative results)

1. **HarnessKernel is real** — provides substantive policy/state/budget/trace behavior on wired paths (`step_loop.py`, `uaep_step_bridge.py`, kernel session wiring in `uaep.py`).
2. **Nexus wires real infrastructure** — policy, middleware, events, workspace, sandbox, memory, and context infrastructure are substantively present on governed Nexus paths.
3. **Tool gateway on RuntimeExecutionContext** — tool invocation through `RuntimeExecutionContext` uses a tool gateway on governed paths; finding 01 does **not** claim every tool call is ungoverned.
4. **Meaningful external side-effect policy** — fail-closed where explicitly used (`runtime_policy_engine.py` `evaluate_meaningful_side_effect`).
5. **Finding 02 qualification** — does **not** claim resume bypasses all governance; outer UAEP middleware/governance remains; defect is kernel-equivalent step semantics only.
6. **Finding 03 qualification** — did **not** prove a normal production path intentionally running without kernel context.
7. **Finding 09 qualification** — did **not** observe a failing runtime execution; evidence was static contract + reachable call path only.

## Prior-audit comparison

Revalidation against pre-Protocol-v2 working STRATEGIC_HARNESS_MODEL audit (no prior Protocol v2 canonical SHA recovered):

- Prior resume bypass severity **CRITICAL → Protocol v2 HIGH** — no concrete exploit/outage evidence justified CRITICAL.
- Fail-open fallback **HIGH → MEDIUM** — normal production reachability of missing-kernel path was not proven.
- Canonical-hook finding refined — generic `IntergraxAgent` has `on_next_step`, but generic UAEP does not universally execute through it; `CognitiveAgent` shim does.
- Product-result leakage confirmed (`search_summary`, `ingest_summary`, `domain_summary` promotion).
- Typed-context finding strengthened to include `StepOutcome` untyped maps.
- `AttemptId` continuity break confirmed across Nexus → ACP bridge.
- Maturity drift confirmed — A4/I4 claims exceed verified invariant closure.
- **NEW under revalidation:** universal pre-effect kernel-boundary defect (finding 01).
- **NEW under revalidation:** `RuntimeExecutionContext` construction missing `attempt_id` / test mock gap (finding 09).

Protocol v2 canonical audit remains bound only to audited_sha `9658224495c775fcefd55ab52bbcc7a94c84fb50`.

## Open questions / blocked items

- **SHM-01:** Architectural decision needed — pure-intent kernel model vs explicitly compositional mandatory enforcement.
- **SHM-03:** Separate normal production reachability of the missing-kernel fallback was not demonstrated.
- **SHM-04:** Decide one universal author hook vs explicitly separate typed contracts.
- **SHM-09:** Runtime regression reproduction still required during remediation.

These are **not** reasons to change ACCEPTED status.

## Operator acceptance

- **Date:** 2026-08-18
- **Accepted findings:** all 10 IDs (`AUDIT-20260818-STRATEGIC_HARNESS_MODEL-01` … `AUDIT-20260818-STRATEGIC_HARNESS_MODEL-10`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none

## Remediation grouping (planning only — not implemented by this task)

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
