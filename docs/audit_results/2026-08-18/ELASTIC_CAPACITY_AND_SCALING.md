# ELASTIC_CAPACITY_AND_SCALING - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** ELASTIC_CAPACITY_AND_SCALING
- **Tier(s):** Tier-1 `intergrax/runtime/capacity/` · Tier-3 `intergrax/applications/_shared/scaling_wiring.py`
- **audited_sha:** `d2b65885ad1b472bf48254a1e7314dc6a53ca677`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 6 HIGH / 0 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md`
- **Scope in:**
  - `CapacitySignal`, `ScalingRule`, `ScalingPolicy`, `ScalingAction`, `ScalingActionPlan` contracts
  - `ScalingEvaluator` - signal indexing, rule match, hysteresis, cooldown, hourly rate limits
  - `ScalingProvisioner` - backend dispatch, action outcome semantics
  - `CapacityActionGate` - `BEFORE_CAPACITY_ACTION` authorization
  - `CapacityApprovalQueue` / `CapacityApprovalRecord` - HITL approval path
  - `CapacityScheduler` - plan drain and multi-action apply loop
  - `BoundedOrchestrationCeilingPatcher` - orchestration ceiling backend
  - `wire_application_scaling()` - canonical reference host wiring
  - Historical ECP **Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - Kubernetes HPA / Celery autoscale replacement claim
  - production cluster deployment evidence
  - second autoscaling subsystem invention
- **Prior audit reference(s):** legacy ECP audits under `docs/audit_results/legacy/` - historical only; Protocol v2 snapshot at pinned SHA supersedes for campaign register
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `9894be22c9d266584c88101ad8a89f9d2450f544`

## Executive summary

**Verdict: FAIL.** Six accepted HIGH findings show that `ScalingEvaluator` indexes capacity signals by `metric_name` only and can bind a rule to another target's signal; `ScalingRule` accepts semantically impossible action/threshold combinations and the provisioner can record false success on no-op backends; canonical `wire_application_scaling()` installs a permissive `CapacityActionGate()` without Governed Execution callback; HITL approval is satisfied by `plan_id` possession without authoritative approver evidence; cooldown and hourly limits live only in process memory and are recorded at plan generation not successful apply; and multi-action plans ignore per-action results without plan-level COMPLETE/PARTIAL/FAILED semantics. Positive controls: backpressure and scaling remain correctly separated; Orchestration owns scheduling inside capacity; ECP owns capacity action execution; `ScalingPolicy` is disabled by default; scheduler runs outside Nexus hot path; missing K8s/Celery backend fails visibly; typed `ScalingTarget` / `ScalingActionKind` contracts exist; architecture does not claim ECP replaces HPA/Celery autoscale; graceful external scale-down remains explicitly incomplete; K8s/Celery production deployment evidence is not falsely claimed; maturity remains A4/I3/P2/E3; findings require hardening existing ECP, not a second autoscaling subsystem. Remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 0 CRITICAL / 6 HIGH / 0 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-01

- **Severity:** HIGH
- **Category:** SIGNAL IDENTITY / WRONG-TARGET SCALING
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-SIGNAL-ACTION-INTEGRITY
- **Claim falsified:** Canonical capacity signal identity includes target + metric_name and required scope; rule evaluation consumes only signals belonging to its exact target/scope.
- **Observation:** `CapacitySignal` and `ScalingRule` both contain `target` and `metric_name`. `ScalingEvaluator` indexes signals only by `metric_name`: `by_metric = {s.metric_name: s for s in signals}` and resolves each rule with `by_metric.get(rule.metric_name)`. It does not validate `signal.target == rule.target`. If the same metric name exists for multiple targets, dictionary ordering can make a rule consume another target's signal and generate a scaling action for the wrong capacity domain.
- **Location:**
  - `intergrax/runtime/capacity/contracts.py` - `CapacitySignal`, `ScalingRule`
  - `intergrax/runtime/capacity/evaluator.py` - signal indexing and rule resolution
- **Reproduction:** Supply two signals with the same `metric_name` but different `target` values; configure a rule for one target - observe evaluator may consume the wrong signal depending on dict ordering.
- **Impact:** Scaling actions can target the wrong capacity domain - wrong replicas, workers, or ceiling mutations under multi-target policies.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-02

- **Severity:** HIGH
- **Category:** CONTRACT INTEGRITY / FALSE SUCCESS
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-SIGNAL-ACTION-INTEGRITY
- **Claim falsified:** Scaling contracts reject semantically impossible rule/action combinations; backend outcome distinguishes APPLIED / NO_CHANGE / FAILED and evidence reflects actual capacity effect.
- **Observation:** `ScalingRule` does not validate: `scale_up_threshold > scale_down_threshold`, positive base delta, `action_kind` / `target` compatibility, or whether an action kind supports scale-down. Evaluator negates `rule.delta` for every scale-down trigger. This allows e.g. `RAISE_ORCHESTRATION_CEILING` with a negative delta. `BoundedOrchestrationCeilingPatcher` treats `delta <= 0` as no-op. `ScalingProvisioner` nevertheless records the action as applied, emits `SCALE_APPLIED`, and returns `True`.
- **Location:**
  - `intergrax/runtime/capacity/contracts.py` - `ScalingRule`
  - `intergrax/runtime/capacity/evaluator.py` - scale-down delta negation
  - `intergrax/runtime/capacity/ceiling_patcher.py` - no-op on non-positive delta
  - `intergrax/runtime/capacity/provisioner.py` - success recording
- **Reproduction:** Configure a scale-down rule with `RAISE_ORCHESTRATION_CEILING` and positive delta - observe negative effective delta, patcher no-op, provisioner still reports applied success.
- **Impact:** Operators and audit evidence can show successful capacity change when no physical or logical capacity changed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-03

- **Severity:** HIGH
- **Category:** GOVERNANCE / CAPACITY MUTATION FAIL-OPEN
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-GOVERNED-ACTION-INTEGRITY
- **Claim falsified:** Capacity-mutating modes have explicit authority semantics; for production mutation, unavailable required Governance authority fails closed; reuse Governed Execution - do not create a second permission engine.
- **Observation:** `CapacityActionGate.authorize()` returns `True` when no `before_action` hook is configured. Canonical `wire_application_scaling()` creates `CapacityActionGate()` with no Governance callback. Therefore when `ScalingPolicy.enabled=True`, capacity actions supported by the configured backend may execute without canonical Governed Execution approval. Architecture honestly describes the current action gate as local/optional rather than full Governance coverage; Protocol v2 classifies that as a residual architecture defect for mutation-capable production posture.
- **Location:**
  - `intergrax/runtime/capacity/action_gate.py` - `CapacityActionGate.authorize()`
  - `intergrax/applications/_shared/scaling_wiring.py` - `wire_application_scaling()`
  - `intergrax/runtime/capacity/provisioner.py` - gate invocation before apply
- **Reproduction:** Enable scaling policy on reference host wiring; trigger a supported ceiling action - observe apply proceeds with default permissive gate and no Governed Execution hook.
- **Impact:** Production capacity mutations can bypass canonical policy/approval spine when host wiring omits Governance callback.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-04

- **Severity:** HIGH
- **Category:** HUMAN APPROVAL / AUTHORITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-GOVERNED-ACTION-INTEGRITY
- **Claim falsified:** HITL-gated capacity action consumes canonical approval evidence bound to exact plan/actions, scope/tenant/environment, approver, policy/version, and decision/time/expiry where applicable; reuse canonical Governance/HITL approval authority.
- **Observation:** `CapacityApprovalRecord` stores plan/status/requested_at but no approver identity or approval evidence. `CapacityApprovalQueue.approve()` accepts only `plan_id`. Calling `approve(plan_id)` changes `hitl_required` plan to `planned` and queues it for execution. `CapacityScheduler` drains approved plans and applies them. Possession of queue + `plan_id` is sufficient to manufacture local approval state; no authoritative human decision is proven.
- **Location:**
  - `intergrax/runtime/capacity/approval_queue.py` - `CapacityApprovalRecord`, `CapacityApprovalQueue.approve()`
  - `intergrax/runtime/capacity/scheduler.py` - approved plan drain and apply
  - `intergrax/runtime/capacity/governance.py` - approve/deny helpers
- **Reproduction:** Create HITL-required plan; call `approve(plan_id)` without approver identity or external approval evidence - observe scheduler applies plan on next tick.
- **Impact:** Scale-up and other HITL-gated capacity changes lack authoritative human decision provenance.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-05

- **Severity:** HIGH
- **Category:** MULTI-HOST / RESOURCE GOVERNANCE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-DISTRIBUTED-EXECUTION-INTEGRITY
- **Claim falsified:** Production anti-flapping/rate limits use a shared scope-aware state authority or equivalent version-fenced coordination contract; lifecycle distinguishes planned, approved, attempted, applied, and failed; restart/multi-host execution cannot silently reset a global policy bound.
- **Observation:** `ScalingEvaluator` stores cooldown and hourly action history only in process: `_last_action_at`, `_action_timestamps`. Restart clears both. Separate hosts each maintain independent limits. Additionally `_record_action()` executes when a plan is generated, before provisioning succeeds. Failed or HITL-separated execution therefore does not have truthful planned/attempted/applied accounting.
- **Location:**
  - `intergrax/runtime/capacity/evaluator.py` - `_last_action_at`, `_action_timestamps`, `_record_action()`
  - `intergrax/runtime/capacity/scheduler.py` - plan generation vs apply separation
- **Reproduction:** Generate plan that fails at provision; restart evaluator process; observe cooldown/rate state reset and action counted before successful apply.
- **Impact:** Multi-host or restart deployments can exceed intended rate limits and misrepresent action lifecycle in evidence.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-06

- **Severity:** HIGH
- **Category:** PARTIAL EXECUTION / PLAN CONSISTENCY
- **Status at publication:** ACCEPTED
- **Remediation block:** ECP-DISTRIBUTED-EXECUTION-INTEGRITY
- **Claim falsified:** Plan execution produces authoritative per-action + plan-level outcome COMPLETE / PARTIAL / FAILED as appropriate; partial plans create deterministic compensation/reconciliation obligations; logical consistency and recoverability are required - not a distributed physical transaction.
- **Observation:** `ScalingActionPlan` may contain multiple actions. `CapacityScheduler._apply_plan()` loops over actions and calls `ScalingProvisioner.apply(action)`. The boolean result is ignored. There is no plan-level execution result, partial state, compensation rule, or reconciliation obligation. A plan can apply one physical/logical capacity action, fail another, and complete the scheduler tick without representing a PARTIAL outcome.
- **Location:**
  - `intergrax/runtime/capacity/contracts.py` - `ScalingActionPlan`
  - `intergrax/runtime/capacity/scheduler.py` - `_apply_plan()`
  - `intergrax/runtime/capacity/provisioner.py` - per-action apply result
- **Reproduction:** Submit multi-action plan where first backend succeeds and second fails - observe scheduler completes without PARTIAL plan outcome or reconciliation semantics.
- **Impact:** Partial capacity mutations can leave infrastructure in inconsistent state without recoverable plan-level accountability.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Backpressure and scaling remain correctly separated | NOT falsified |
| Orchestration owns scheduling inside capacity | NOT falsified |
| ECP owns capacity action execution | NOT falsified |
| `ScalingPolicy` is disabled by default | NOT falsified |
| Scheduler runs outside Nexus hot path | NOT falsified |
| Missing K8s/Celery backend fails visibly rather than silently succeeding | NOT falsified |
| Typed `ScalingTarget` / `ScalingActionKind` contracts exist | NOT falsified |
| Architecture does not claim ECP replaces HPA/Celery autoscale | NOT falsified |
| Graceful external scale-down remains explicitly incomplete | NOT falsified |
| K8s/Celery production deployment evidence is not falsely claimed | NOT falsified |
| Maturity remains A4/I3/P2/E3 | NOT falsified |
| Findings require hardening existing ECP, not a second autoscaling subsystem | NOT falsified - remediation targets existing evaluator/gate/queue/scheduler/provisioner path |

## Historical ECP delivery vs Protocol-v2 residual defects

Historical **ECP-DEPTH Done** delivery facts remain valid - contracts, collector, evaluator, scheduler, provisioner, HITL queue, action gate hook surface, mocked K8s/Celery adapter paths, metrics, and reference wiring were delivered as claimed. The six accepted Protocol-v2 findings document **residual signal identity, contract integrity, governance authority, HITL evidence, distributed anti-flapping, and multi-action plan consistency gaps** at `audited_sha` - they harden the existing ECP path; they do **not** reopen closed ECP wave rows or require a second autoscaling subsystem.

## Root-cause remediation grouping

### ECP-SIGNAL-ACTION-INTEGRITY - capacity signal and action contract integrity

**Findings:** `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-01`, `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-02`

Capacity signals and actions carry exact target semantics; impossible or no-op actions cannot masquerade as successful capacity changes. Applied evidence must reflect actual backend effect (APPLIED / NO_CHANGE / FAILED).

### ECP-GOVERNED-ACTION-INTEGRITY - governed mutation and HITL authority

**Findings:** `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-03`, `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-04`

Capacity mutation and HITL use canonical Governed Execution / approval evidence; fail-closed where production policy requires authority. Cross-link [`GOVERNED_EXECUTION`](../../project/architecture/GOVERNED_EXECUTION.md) / [`IDENTITY_TRUST`](../../project/architecture/IDENTITY_TRUST.md) rather than duplicate approval infrastructure.

### ECP-DISTRIBUTED-EXECUTION-INTEGRITY - distributed limits and plan outcomes

**Findings:** `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-05`, `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-06`

Cooldown/rate limits remain correct across restart/multi-host execution; multi-action plans expose deterministic COMPLETE/PARTIAL/FAILED outcomes with reconciliation. Reuse existing Observability and distributed coordination mechanisms rather than duplicating them.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `d2b65885ad1b472bf48254a1e7314dc6a53ca677`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical ECP **Done** plan rows remain valid delivery facts - not rewritten.

## Open questions / blocked items

- Finding 05: shared scope-aware rate-limit store vs version-fenced coordination contract - deferred to remediation reusing platform coordination patterns.
- Finding 06: compensation semantics for mixed K8s/Celery/ceiling actions - deferred to remediation design without second provisioner stack.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-01` … `AUDIT-20260818-ELASTIC_CAPACITY_AND_SCALING-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
