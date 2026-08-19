# LLM_ADAPTERS — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** LLM_ADAPTERS
- **Tier(s):** cross-domain Tier-0 LLM contracts · Tier-1 Nexus planning/classification · Tier-1 ACP `StepLLMRouter` · provider adapters
- **layer_audited_at:** 2026-08-19
- **audited_sha:** `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Status:** COMPLETE
- **Auditor:** independent ChatGPT platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-19
- **Architecture doc(s):**
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/maintainers/plans/LLM_ADAPTERS.md`
- **Scope in:**
  - PRE_MODEL governance boundary coverage for all provider-bound inference
  - decision-to-execution model/provider binding
  - failover governance
  - LLM execution identity on planning/classification paths
- **Scope out:**
  - remediation implementation
  - claim that initial Tier-3 profile resolver is broken at composition time
- **Prior audit reference(s):** [`IDENTITY_TRUST`](IDENTITY_TRUST.md) (LLM-FIX-D / IDT-FIX-D shared identity closure)
- **architecture_sync:** COMPLETE after Commit A
- **plan_sync:** COMPLETE after Commit A
- **post_sync_sha:** `pending Commit A`

## Executive summary

**Verdict: FAIL.** Six accepted findings (4 HIGH, 2 MEDIUM) show classifier and planner retry inference outside universal PRE_MODEL boundary, decision-plane model_id not execution-bound, failover changing provider/model without per-candidate authorization, trace/provider attribution drift, and TaskId substituted for run_id on LLM calls. Positive controls: `LLMAdapter` abstraction preserved, initial composition-time adapter wiring, and provider SDK isolation behind adapters. No new `VENDOR_LEAK` finding in this layer.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-LLM_ADAPTERS-01

**LLM classifier bypasses canonical PRE_MODEL governance**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Related classification:** SECURITY · IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-A
- **Claim falsified:** Every provider-bound inference, including classification, crosses the same canonical PRE_MODEL governance boundary used for planning and agent-step model access.
- **Observation:** Nexus invokes classifier before planning PRE_MODEL evaluation. `LlmTaskClassifier` can invoke LLM directly during classification. PRE_MODEL phase model does not include classification. Classification inference is therefore outside the same pre-model boundary used for planning/agent-step model access.
- **Location:**
  - `intergrax/runtime/nexus/orchestration/planning_runner.py:L127-L196` — classify before `evaluate_pre_llm` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/runtime/nexus/llm_task_classifier.py:L99-L117` — direct `generate_messages` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/runtime/nexus/orchestration/planning_runner.py` — `self.classifier.classify(task)` precedes PRE_MODEL planning evaluation.
  2. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/runtime/nexus/llm_task_classifier.py` — `_infer_capability` calls LLM adapter.
- **Impact:** Classification inference can evade PRE_MODEL policy/budget/trace guarantees.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LLM_ADAPTERS-02

**Selected model is not execution-bound to the model actually invoked**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Related classification:** SECURITY · OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-B
- **Claim falsified:** The model/provider authorized by routing/policy is the same candidate actually invoked at execution time for each step/call.
- **Observation:** `StepLLMRouter` resolves `model_id` and policy/budget/trace operate on it. `LLMAdapterCompletePort.complete()` discards `model_id`/`provider`. Concrete provider adapter uses model configured on adapter instance. Per-step dynamic model hint does not necessarily replace adapter instance. Decision plane can diverge from execution plane. Initial Tier-3 profile resolver correctly builds adapter at composition time — finding concerns per-call/per-step binding.
- **Location:**
  - `intergrax/agents/authoring/llm_router.py:L68-L75` — `del model_id, provider` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/agents/authoring/llm_router.py:L167-L231` — trace uses `self.provider` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/agents/authoring/llm_router.py` — completion port ignores resolved model/provider.
  2. Compare `resolve_model` / `LlmStepResult.provider=self.provider` vs adapter instance configuration.
- **Impact:** Policy and trace can describe a different model/provider than the one actually invoked.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LLM_ADAPTERS-03

**Failover may change provider/model after policy decision**

- **Severity:** HIGH
- **Category:** SECURITY
- **Related classification:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-C
- **Claim falsified:** Every actual provider/model candidate invoked in a call is individually authorized or belongs to an explicitly pre-authorized immutable candidate set.
- **Observation:** `FailoverLLMAdapter` can try primary then fallback adapters inside one `generate_messages()` call. Candidates may carry different provider/model. PRE_MODEL policy is evaluated before outer invocation and is not re-run for each fallback candidate. Structural guarantee missing. Not every configured fallback is forbidden — governance completeness is the defect.
- **Location:**
  - `intergrax/llm_adapters/registry/failover_adapter.py:L78-L118` — sequential failover in one call @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/runtime/policy/pre_model_policy_bridge.py:L63-L78` — single PRE_MODEL evaluation before inner router @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/llm_adapters/registry/failover_adapter.py` — `_execute_with_failover`.
  2. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/runtime/policy/pre_model_policy_bridge.py` — one `evaluate_pre_model_policy` before delegate.
- **Impact:** Fallback candidate can execute without fresh authorization for that provider/model.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LLM_ADAPTERS-04

**Planner PRE_MODEL runs once while parser retries may invoke provider multiple times**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Related classification:** RELIABILITY · SCALABILITY / COST
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-A
- **Claim falsified:** PRE_MODEL governance runs immediately before each actual provider/model invocation attempt.
- **Observation:** Planning policy evaluation occurs before planner call. Planner supports `1 + planner_parse_retries` provider calls. Retry calls do not each receive a fresh PRE_MODEL boundary. Architecture claim does not hold per actual inference attempt.
- **Location:**
  - `intergrax/runtime/nexus/planning/nexus_plan_bridge.py:L106-L120` — retry loop without PRE_MODEL @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/runtime/nexus/orchestration/planning_runner.py:L184-L196` — single planning PRE_MODEL @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/runtime/nexus/planning/nexus_plan_bridge.py` — `max_attempts = 1 + planner_parse_retries`.
  2. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/runtime/nexus/orchestration/planning_runner.py` — PRE_MODEL once before planner.
- **Impact:** Retried planner inference can bypass per-attempt PRE_MODEL guarantees.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LLM_ADAPTERS-05

**Trace/result can report provider different from actual runtime provider**

- **Severity:** MEDIUM
- **Category:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-B
- **Claim falsified:** LLM usage/trace attributes the actual runtime provider/model invoked for each call.
- **Observation:** Router computes local effective provider from runtime adapter. Execution uses effective provider. `LlmCallRecord`/`LlmStepResult` use `self.provider`; default can be `stub`. Usage/trace can attribute a real call to the wrong provider.
- **Location:**
  - `intergrax/agents/authoring/llm_router.py:L170-L173` — effective provider from adapter @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/agents/authoring/llm_router.py:L214-L227` — `provider=self.provider` on record/result @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git show b1e4de1d776acc64e8461f7dcdce09cd03d07b80:intergrax/agents/authoring/llm_router.py` — compare effective provider computation vs recorded provider field.
- **Impact:** Cost/audit attribution can misidentify provider for real calls.
- **Confidence:** CONFIRMED

### AUDIT-20260818-LLM_ADAPTERS-06

**Planner/classifier LLM usage passes TaskId as run_id**

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION DEFECT
- **Related classification:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** LLM-FIX-D / IDT-FIX-D
- **Claim falsified:** LLM usage/cost/trace correlation uses canonical execution `RunId`/`AttemptId`, not `TaskId`.
- **Observation:** Classifier/planner pass `run_id=task.task_id` into LLM adapter paths. Same planning phase already has real active `RunId`/`AttemptId` for `RuntimeEvent`. Cross-reference IDT-FIX-D as shared remediation dependency.
- **Location:**
  - `intergrax/runtime/nexus/llm_task_classifier.py:L114-L117` — `run_id=task.task_id` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
  - `intergrax/runtime/nexus/planning/nexus_plan_bridge.py:L113-L116` — `run_id=task.task_id` @ `b1e4de1d776acc64e8461f7dcdce09cd03d07b80`
- **Reproduction:**
  1. `git grep -n "run_id=task.task_id" b1e4de1d776acc64e8461f7dcdce09cd03d07b80 -- intergrax/runtime/nexus/`
- **Impact:** LLM telemetry mis-correlates with execution identity spine.
- **Confidence:** CONFIRMED

## Provider / backend abstraction

| concern | canonical abstraction | classification | notes |
|---------|-----------------------|----------------|-------|
| `LLMAdapter` | `ABSTRACTION_PRESERVED` | vendor SDKs behind adapter |
| `LLMProfile` | `ABSTRACTION_PRESERVED` | composition-time profile |
| `LLMAdapterRegistry` | `COMPOSITION_ONLY` | selection at wiring |
| initial `ModelRouter` / `resolve_llm_adapter` | `ABSTRACTION_PRESERVED` | composition-time binding OK |
| per-step dynamic routing | `PAPER_ABSTRACTION` | LLM-02 decision/execution drift |
| `FailoverLLMAdapter` | abstraction preserved, governance incomplete | LLM-03 |
| vendor SDKs | `PROVIDER_LOCAL` | no new `VENDOR_LEAK` |

## Falsification log

1. **Initial Tier-3 adapter resolution broken** — composition-time profile resolver works; defect is per-step binding (not promoted).
2. **All fallback forbidden** — not claimed; per-candidate authorization missing only.
3. **Vendor SDK leakage in Nexus** — adapters preserve abstraction (positive control).

## Prior-audit comparison

Builds on identity/trust execution-identity themes and Nexus flow planning narrative. First canonical Protocol v2.2 `LLM_ADAPTERS` immutable snapshot.

## Open questions / blocked items

- Whether failover candidates should be pre-authorized sets vs per-candidate PRE_MODEL — planning only (**LLM-FIX-C**).
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-19
- **Accepted findings:** all 6 (`AUDIT-20260818-LLM_ADAPTERS-01` … `AUDIT-20260818-LLM_ADAPTERS-06`)
- **Remediation blocks:** LLM-FIX-A, LLM-FIX-B, LLM-FIX-C, LLM-FIX-D — all **ACCEPTED / PLANNED** only; **LLM-FIX-D** cross-references **IDT-FIX-D**
