# EXECUTION_RUNTIME — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** EXECUTION_RUNTIME
- **Tier(s):** Tier-1 Unified Execution Runtime · ACP/UAEP · HarnessKernel · cancellation/checkpoint
- **layer_audited_at:** 2026-08-19
- **audited_sha:** `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Status:** COMPLETE
- **Auditor:** independent ChatGPT platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-19
- **Architecture doc(s):**
  - `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`
  - `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`
  - `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`
- **Scope in:**
  - canonical runtime policy propagation into direct ACP
  - atomic step commit semantics
  - resume attempt identity continuity
  - runtime exception containment
  - cooperative cancellation and checkpoint invalidation
- **Scope out:**
  - remediation implementation
  - claim that process-fatal exceptions must be swallowed
  - claim that cancellation can never stop later graph execution (finding concerns active ACP session)
- **Prior audit reference(s):** [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md); [`IDENTITY_TRUST`](IDENTITY_TRUST.md)
- **architecture_sync:** COMPLETE after Commit A
- **plan_sync:** COMPLETE after Commit A
- **post_sync_sha:** `pending Commit A`

## Executive summary

**Verdict: FAIL.** Six accepted findings (5 HIGH, 1 MEDIUM) show direct ACP constructing default policy engine, non-atomic step state commit on failure, resume minting new AttemptId, unexpected agent exceptions escaping without typed terminal failure, cancellation not reaching active ACP loops, and cancel clearing task pointers without invalidating persisted ACP checkpoints. Positive controls: declarative invoker abstraction, provider-neutral runtime policy contracts, checkpoint store port. No new vendor leak.

## Verdict

**FAIL** — 0 CRITICAL / 5 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-EXECUTION_RUNTIME-01

**Direct ACP constructs fresh default PolicyEngine rather than receiving canonical host policy**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Related classification:** SECURITY · IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-A
- **Claim falsified:** Direct ACP receives and uses the same canonical host/Nexus policy environment as other production execution paths.
- **Observation:** `run_acp_session` builds `StepKernelContext(policy_engine=PolicyEngine())`. `ACPSessionHostContext` has no canonical policy-engine/policy-view carrier. Direct ACP can use policy semantics distinct from host/Nexus. Finding is propagation/ownership, not absence of all policy.
- **Location:**
  - `intergrax/agents/authoring/acp_run.py:L292` — `policy_engine=PolicyEngine()` @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/agents/authoring/acp_run.py:L285-L300` — `StepKernelContext` construction @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/authoring/acp_run.py` — fresh `PolicyEngine()` in kernel context.
  2. `git grep -n "policy_engine" df7aaac19b20e84c06d6233492cdb4365a892f4f -- intergrax/agents/authoring/acp_run.py`
- **Impact:** Direct ACP sessions can run under a different policy universe than application/Nexus composition.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXECUTION_RUNTIME-02

**Failed step can retain already-merged state while record says `outcome_applied=False`**

- **Severity:** HIGH
- **Category:** RELIABILITY
- **Related classification:** IMPLEMENTATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-B
- **Claim falsified:** Step outcome semantics are atomic — `outcome_applied=False` cannot coexist with committed platform state describing success-like effects for the same step.
- **Observation:** `HarnessKernel` merges state and assigns `kernel_ctx.state_root` before declarative action execution and post-policy. Later tool failure or post-policy DENY can return `outcome_applied=False`. Merged state is not rolled back. Runtime state can describe success-like effects for a failed step.
- **Location:**
  - `intergrax/runtime/kernel/step_kernel.py:L243-L300` — merge before actions/post-policy @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/runtime/kernel/step_kernel.py` — `kernel_ctx.state_root = merge_result.state` before tool/post-policy failure paths returning `outcome_applied=False`.
- **Impact:** Resume, audit, and downstream steps can observe state inconsistent with step record.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXECUTION_RUNTIME-03

**ACP resume mints new AttemptId although resume-without-retry must preserve it**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Related classification:** OPERABILITY · RELIABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-C
- **Claim falsified:** Resume without retry preserves canonical `AttemptId`; retry mints new `AttemptId`.
- **Observation:** UER canon distinguishes retry vs resume-without-retry attempt identity. `_resolve_acp_session_identity()` always mints `AttemptId`. Resume checkpoint resolution occurs afterward. `AgentRunCheckpoint` does not persist `AttemptId`. Direct ACP resume cannot preserve canonical attempt continuity.
- **Location:**
  - `intergrax/agents/authoring/acp_run.py:L73-L89` — always `mint_attempt_id()` @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/contracts/side_effect.py:L63-L76` — checkpoint without AttemptId @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/authoring/acp_run.py` — identity resolver.
  2. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/contracts/side_effect.py` — checkpoint fields.
- **Impact:** Resume breaks attempt-identity continuity required by UER/Observability spine.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXECUTION_RUNTIME-04

**Unexpected agent exception can escape UER without typed terminal failure**

- **Severity:** HIGH
- **Category:** RELIABILITY
- **Related classification:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-D
- **Claim falsified:** Normal runtime exceptions from agent/domain/provider paths become typed terminal `AgentRunResult` FAILED with evidence and cleanup.
- **Observation:** `AgentRuntime.advance_step` catches budget exception but not normal unexpected agent/domain/provider exceptions. Outer direct ACP run has identity-reset `finally` but no universal typed terminal exception boundary around `_run_acp_session_bound`. `on_run_start`/`on_next_step` exceptions can escape. Process-fatal exceptions are out of scope.
- **Location:**
  - `intergrax/agents/authoring/step_loop.py:L47-L58` — only `AcpBudgetExceededError` caught @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/agents/authoring/acp_run.py:L217-L229` — `_run_acp_session_bound` without broad exception containment @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/authoring/step_loop.py` — `advance_step` exception scope.
  2. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/authoring/acp_run.py` — session bound call without universal catch.
- **Impact:** Callers may receive raw exceptions instead of governed terminal runtime results.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXECUTION_RUNTIME-05

**Accepted task cancellation does not propagate into already-running ACP loop**

- **Severity:** HIGH
- **Category:** RELIABILITY
- **Related classification:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-E
- **Claim falsified:** Cooperative cancellation reaches already-running ACP work at meaningful iteration/LLM/tool/side-effect boundaries.
- **Observation:** Task-control cancellation mutates Task metadata. `GraphExecutor` checks cancellation between graph batches. Active ACP session has its own context/snapshot and loop. ACP iteration/kernel has no canonical shared cancellation check/channel. Operator may receive accepted cancellation while active agent continues until node returns. Later graph cancellation is not claimed broken.
- **Location:**
  - `intergrax/runtime/cancellation/coordinator.py:L26-L35` — metadata flag only @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/agents/authoring/acp_run.py:L473-L507` — ACP loop without cancellation check @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/runtime/nexus/execution/graph_executor.py:L209-L212` — graph batch cancellation (contrast) @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/authoring/acp_run.py` — session loop.
  2. `git grep -n "CancellationCoordinator\|should_cancel" df7aaac19b20e84c06d6233492cdb4365a892f4f -- intergrax/agents/authoring/`
- **Impact:** Cancel acceptance does not stop in-flight ACP iterations promptly.
- **Confidence:** CONFIRMED

### AUDIT-20260818-EXECUTION_RUNTIME-06

**Cancel clears task checkpoint pointers but cannot invalidate persisted ACP checkpoint**

- **Severity:** MEDIUM
- **Category:** RELIABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** UER-FIX-E
- **Claim falsified:** Cancellation invalidates or tombstones resumable checkpoint authority; cancelled checkpoint cannot be treated as ordinary resumable state without new authorized transition.
- **Observation:** `CancellationCoordinator.clear_checkpoint_state` explicitly only clears task-side resume pointers. `AgentCheckpointStore` exposes save/get_latest only. No invalidate/delete/cancelled state. Direct ACP resume can locate historical checkpoint by run_id+tenant if resume flag supplied. No claim current public API automatically exploits this.
- **Location:**
  - `intergrax/runtime/cancellation/coordinator.py:L62-L69` — pointer clear only @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
  - `intergrax/agents/persistence/checkpoint_store.py:L16-L23` — save/get_latest only @ `df7aaac19b20e84c06d6233492cdb4365a892f4f`
- **Reproduction:**
  1. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/runtime/cancellation/coordinator.py` — `clear_checkpoint_state` docstring and behavior.
  2. `git show df7aaac19b20e84c06d6233492cdb4365a892f4f:intergrax/agents/persistence/checkpoint_store.py` — port surface.
- **Impact:** Cancelled runs may retain resumable ACP checkpoint material in store.
- **Confidence:** CONFIRMED

## Provider / backend abstraction

| concern | classification | notes |
|---------|----------------|-------|
| `AgentCheckpointStore` | `ABSTRACTION_PRESERVED` | port over SQLite/local backends |
| SQLite checkpoint store | `PROVIDER_LOCAL` | lab/reference implementation |
| declarative invoker | `ABSTRACTION_PRESERVED` | kernel tool path |
| LLM through `LLMAdapter` | `ABSTRACTION_PRESERVED` | runtime policy provider-neutral |
| runtime policy | `ABSTRACTION_PRESERVED` | UER-01 is propagation, not vendor leak |

No new vendor leak in this layer.

## Falsification log

1. **No policy anywhere on ACP path** — policy engine exists; defect is default fresh engine (not promoted as absence finding).
2. **Graph cancellation never works** — graph batch cancellation exists (positive control); ACP gap remains.
3. **Process-fatal exceptions must be swallowed** — explicitly out of scope for UER-04.

## Prior-audit comparison

Extends strategic harness and identity themes into UER-specific propagation, atomicity, attempt continuity, exception containment, and cancellation. First canonical Protocol v2.2 `EXECUTION_RUNTIME` snapshot.

## Open questions / blocked items

- Shared cancellation authority between Nexus graph and ACP session — planning only (**UER-FIX-E**).
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-19
- **Accepted findings:** all 6 (`AUDIT-20260818-EXECUTION_RUNTIME-01` … `AUDIT-20260818-EXECUTION_RUNTIME-06`)
- **Remediation blocks:** UER-FIX-A … UER-FIX-E — all **ACCEPTED / PLANNED** only
