# Execution Engine — Production Runbooks (EE-B4-C)

**Authority:** Certification operator contract. **Not** a second governance plane.  
**Prerequisite:** EE-B4-A operational assessment + durable runtime evidence where available.

**Diagnostic query minimum:** `tenant_id`, `run_id`, `attempt_id`, `execution_id`, time window, failure category.

---

## RB-01 ENGINE_NOT_READY

### Title

ENGINE_NOT_READY — readiness `not_ready` or health `unhealthy` without classified root cause

### Trigger

Readiness probe or `assess_execution_operational_state` reports `not_ready` / `unhealthy`; or host refuses new root admission without capacity saturation.

### Severity

SEV-1 when mandatory persistence or global execution halt; otherwise SEV-2 until classified.

### Scope

Establish global vs tenant vs profile vs dependency scope before action.

### Symptoms

New root work rejected; readiness failing; compound signals (shutdown + evidence) possible.

### Canonical signals

`ReadinessClassification.not_ready`; `HealthClassification.unhealthy`; `ExecutionRuntimeShutdownPhase`; mandatory auditability health; root capacity assessment (`assess_root_execution_capacity`).

### Immediate safe action

1. Record scope identifiers (`tenant_id`, `run_id`, `attempt_id`, `execution_id`).  
2. Run EE-B4-A operational assessment with current facts.  
3. Map to INC-01–INC-14; open specific runbook.  
4. Stabilize intake (do not force new roots) until classified.

### Do NOT

Restart process without classification; delete evidence; delete checkpoint; disable mandatory persistence; disable governance; force allow admission.

### Diagnosis

Compare readiness reason codes to taxonomy. Pull durable runtime events for the window. Distinguish INC-02 (mandatory evidence) from INC-01 (capacity) from INC-11 (shutdown).

### Recovery path

Follow child runbook. Resubmit work only via **canonical platform execution entry** (host admission + identity intake) after readiness returns.

### Verification

Readiness `ready`; health not `unhealthy` for global scope; capacity permits released; no worker/task leaks per EE-B4-B checks.

### Escalation

SEV-1 immediate platform owner; include assessment snapshot and incident window.

### Post-incident evidence

Post-incident package per incident model §13.

---

## RB-02 CAPACITY_SATURATED

### Title

CAPACITY_SATURATED — root capacity REJECT/DEFER (INC-01)

### Trigger

`admission_reject_defer_rate` elevated; `capacity_utilization` at limit; readiness `not_ready` with saturation reason.

### Severity

SEV-2 global sustained reject; SEV-3 if tenant-scoped admission policy only.

### Scope

Global root capacity (EE-B1.2) unless deployment proves profile-specific cap.

### Symptoms

New roots deferred/rejected; active execution count at limit; clients see backpressure.

### Canonical signals

`assess_root_execution_capacity`; active root execution counter; admission outcome events; drain progress during shutdown overlap.

### Immediate safe action

1. Confirm saturation via capacity/readiness diagnosis (not permit counters).  
2. Check active execution count trend and terminal completion rate.  
3. Distinguish backpressure from deadlock (stuck attempts without terminal events).  
4. If shutdown active, prefer EE-B4-B drain path over scale-up.

### Do NOT

Manually increment capacity state; clear internal permit counters; bypass admission port.

### Diagnosis

Time-series of `capacity_utilization` SLI; per-`run_id` terminal latency; verify permit release on terminal outcomes (EE-B1.2).

### Recovery path

Allow completions; tune deployment capacity limits via supported config; relieve upstream load. New work resumes when capacity/readiness diagnosis shows admit.

### Verification

`capacity_utilization` below limit; readiness `ready`; admission accepts representative probe execution via canonical entry.

### Escalation

SEV-2 if global reject persists beyond operator window with zero terminal progress (possible deadlock — escalate with `run_id` samples).

### Post-incident evidence

Capacity snapshots, admission outcomes, sample `run_id` terminal states.

---

## RB-03 MANDATORY_EVIDENCE_UNAVAILABLE

### Title

MANDATORY_EVIDENCE_UNAVAILABLE — fail closed (INC-02)

### Trigger

Mandatory persistence failures; auditability health fail; readiness `not_ready` with evidence reason.

### Severity

SEV-1

### Scope

Global or profile mandatory path (per deployment contract).

### Symptoms

Readiness fails; mandatory write errors; execution must not admit new safe work.

### Canonical signals

`mandatory_evidence_persistence_failure_rate`; runtime event store health; `HealthClassification.unhealthy`.

### Immediate safe action

1. Stop treating engine as ready (**FAIL CLOSED**).  
2. Preserve storage subsystem; engage storage owner.  
3. Collect failed write correlation IDs from runtime diagnostics.

### Do NOT

Disable mandatory persistence; switch to best-effort for mandatory path; delete failed events; delete evidence.

### Diagnosis

Identify store outage vs auth vs disk; confirm scope (global vs tenant). Use durable event reconstruction when readable.

### Recovery path

Restore mandatory store; verify write probe; allow readiness to return via EE-B4-A assessment only.

### Verification

Mandatory write success; readiness `ready`; health not `unhealthy`; sample execution persists terminal event.

### Escalation

SEV-1 immediate — no production traffic until mandatory path restored or explicit DR contract engaged.

### Post-incident evidence

Persistence error codes, incident window, store recovery timeline.

---

## RB-04 OBSERVABILITY_EXPORT_DEGRADED

### Title

OBSERVABILITY_EXPORT_DEGRADED — OTLP / best-effort export (INC-03)

### Trigger

Export failures; health `degraded` with export reason; OTLP backend unreachable.

### Severity

SEV-4

### Scope

Export plane only — execution and canonical store continue.

### Symptoms

Missing external telemetry; export error metrics; health degraded.

### Canonical signals

Export policy outcomes; `try_export_observability_envelope` failures; health `degraded` with readiness still `ready`.

### Immediate safe action

1. Acknowledge **DEGRADED** — do not stop Execution Engine solely for OTLP outage.  
2. Notify observability owner; rely on durable runtime evidence for facts.

### Do NOT

Stop accepting work only because OTLP failed; delete canonical evidence; disable mandatory persistence.

### Diagnosis

Separate INC-03 from INC-02 using readiness and mandatory SLI.

### Recovery path

Restore exporter/backend; verify envelopes flow; health returns to `healthy` when only export was impacted.

### Verification

Readiness remains or returns `ready`; export success probe; execution_success_rate unaffected.

### Escalation

SEV-4 to observability owner; SEV-1 only if coupled to INC-02.

### Post-incident evidence

Export error summary, readiness/health assessment during window.

---

## RB-05 WORKER_DRAIN_STUCK

### Title

WORKER_DRAIN_STUCK — drain timeout during shutdown or boundary close (INC-11 partial)

### Trigger

`shutdown_drain_duration` exceeded; drain phase timeout; active count not decreasing.

### Severity

SEV-2

### Scope

Hosting shutdown executor + active work port.

### Symptoms

Shutdown hung in DRAIN; workers remain active past policy timeout.

### Canonical signals

`ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS`; active execution counter; worker task registry; EE-B4-B phase timestamps.

### Immediate safe action

1. Classify stuck `run_id` / `execution_id` set.  
2. Apply hosting shutdown policy (bounded cancel) per EE-B4-B — not ad hoc kill.  
3. Verify capacity permit release after terminal/cancel.

### Do NOT

Restart entire platform without phase classification; clear internal permit counters.

### Diagnosis

Identify blocking attempt vs pool leak vs external dependency hang; check sibling worker health.

### Recovery path

Complete EE-B4-B phases; if timeout policy applied, document terminal/cancel outcomes; proceed flush/persist/terminate in order.

### Verification

Zero managed worker tasks; permits released; shutdown success report per EE-B4-B.

### Escalation

SEV-2 platform owner if mandatory flush blocks termination.

### Post-incident evidence

Drain timeline, stuck execution IDs, cancel outcomes.

---

## RB-06 WORKER_FAILURE

### Title

WORKER_FAILURE — isolated worker / pool degradation (INC-04, INC-05 scoped)

### Trigger

`worker_failure_rate` elevated; worker-isolated terminal failures; pool partial unavailable.

### Severity

SEV-3 default; SEV-2 if pool fully unavailable.

### Scope

Worker pool / orchestration fan-out — not global engine unless capacity impacted.

### Symptoms

Subset of dispatches fail; containment events; siblings may be healthy.

### Canonical signals

Worker containment classification (EE-B1.3); per-`attempt_id` terminal outcome; nexus fan-out errors distinct from root runtime fatal.

### Immediate safe action

1. Determine isolation scope (worker, pool, orchestration step).  
2. Check sibling health and global readiness (may remain `ready`).  
3. Confirm affected executions reached terminal outcome and released resources.

### Do NOT

Restart entire platform without scope proof; conflate Nexus fan-out with ExecutionRuntime fatal.

### Diagnosis

For INC-05: orchestration/fan-out failure signals vs root admission failures. Correlate `execution_id` across events.

### Recovery path

Replace/restart affected worker scope via hosting plane; replay only through **canonical platform execution entry** for new work.

### Verification

`worker_failure_rate` normalized; no permit leaks; global readiness unchanged unless INC-01 co-occurring.

### Escalation

SEV-2 if all workers unavailable or mandatory evidence flush fails during recovery.

### Post-incident evidence

Worker IDs, failure classification, terminal outcomes.

---

## RB-07 RECOVERY_FAILURE

### Title

RECOVERY_FAILURE — NPSC-5E recovery disposition failed (INC-08)

### Trigger

`recovery_success_rate` drop; recovery terminal failure; operator-initiated recovery rejected.

### Severity

SEV-2

### Scope

Per execution / subsystem under recovery plane.

### Symptoms

Recovery cannot complete; checkpoint present but recovery ineligible.

### Canonical signals

Recovery plane disposition; checkpoint read API; governance/admission outcome on recovery attempt.

### Immediate safe action

1. Inspect canonical checkpoint (read-only).  
2. Validate identity, tenant, authority/governance.  
3. Determine recovery eligibility per NPSC-5E — no manual mutation.

### Do NOT

Edit checkpoint JSON; change `run_id` or `tenant_id` manually; delete checkpoint.

### Diagnosis

Map failure to governance denial vs incompatible checkpoint vs provider unknown state (RB-11).

### Recovery path

Use supported recovery APIs and **canonical platform execution entry** for resubmit when eligibility allows.

### Verification

`recovery_success_rate` for cohort; terminal state consistent with evidence; no duplicate side effects.

### Escalation

SEV-2 reliability owner with checkpoint metadata (no secrets).

### Post-incident evidence

Recovery result, checkpoint version, governance outcome.

---

## RB-08 CHECKPOINT_INCOMPATIBLE

### Title

CHECKPOINT_INCOMPATIBLE — version/identity mismatch (INC-09)

### Trigger

Recovery admission rejects checkpoint; migration required; corruption detected.

### Severity

SEV-2

### Scope

Affected executions only.

### Symptoms

Recovery blocked; incompatibility errors in recovery plane.

### Canonical signals

Checkpoint schema/version fields; tenant binding; recovery admission reason codes.

### Immediate safe action

1. Read checkpoint metadata only.  
2. Compare to supported migration matrix (canonical migration path if exists).  
3. Halt blind resubmit for affected `run_id`.

### Do NOT

Delete checkpoint; rewrite version manually; change tenant_id or run_id manually.

### Diagnosis

Determine migration vs corruption vs wrong tenant scope.

### Recovery path

Apply canonical migration tooling only; otherwise escalate for data repair — not manual JSON edit.

### Verification

Recovery eligibility restored; recovery attempt succeeds under NPSC-5E.

### Escalation

SEV-2 with checkpoint version and migration status.

### Post-incident evidence

Checkpoint identity fields, admission errors.

---

## RB-09 PROVIDER_DEPENDENCY_OUTAGE

### Title

PROVIDER_DEPENDENCY_OUTAGE — tool/provider adapter unavailable (INC-06)

### Trigger

`dependency_failure_rate` spike; capability health `unavailable` for scoped provider.

### Severity

SEV-3 scoped; SEV-2 if mandatory-for-all-profiles dependency.

### Scope

Classify mandatory vs optional vs profile-specific provider.

### Symptoms

Tool invocations fail; dependency errors; executions terminal-fail or retry per policy.

### Canonical signals

Capability health projection; per-provider error taxonomy; `attempt_id` terminal classification.

### Immediate safe action

1. Identify provider scope and mandatory flag for active profiles.  
2. Do not invoke provider outside tool/integration plane.  
3. Communicate degradation to tenants if profile-bound.

### Do NOT

Invoke provider directly from operator shell; bypass tool surface.

### Diagnosis

Separate provider outage from governance denial (RB-12) using policy result fields.

### Recovery path

Restore adapter/deployment dependency; optional failover via **adapter appendix** (deployment-specific). Core engine remains provider-neutral.

### Verification

Dependency SLI normalized; sample tool invocation via platform path succeeds.

### Escalation

SEV-2 if mandatory global dependency; otherwise integration owner.

### Post-incident evidence

Provider scope, profile impact, sample `attempt_id` outcomes.

---

## RB-10 RETRY_EXHAUSTED

### Title

RETRY_EXHAUSTED — recovery plane exhausted retries (INC-07)

### Trigger

Terminal retry exhaustion; recovery policy max attempts reached.

### Severity

SEV-3

### Scope

Per `run_id` / `attempt_id`.

### Symptoms

Execution failed terminal; no further automatic retry scheduled.

### Canonical signals

NPSC-5E retry counters; terminal failure classification (retryable vs terminal vs unknown side effect).

### Immediate safe action

1. Classify outcome: retryable failure vs terminal vs unknown side effect.  
2. If unknown side effect — open RB-11.  
3. Do not manually restart retry chain.

### Do NOT

Rerun tool manually; copy request and invoke outside platform; force hidden retry.

### Diagnosis

Evidence timeline for external operation acceptance/timeout.

### Recovery path

Operator-initiated new run only via **canonical platform execution entry** when policy confirms safe; use recovery plane when eligible.

### Verification

New attempt has fresh identity; prior attempt remains terminal in evidence.

### Escalation

SEV-2 if systemic misconfiguration causes mass exhaustion.

### Post-incident evidence

Retry counts, terminal reason, external correlation IDs.

---

## RB-11 UNKNOWN_SIDE_EFFECT_STATE

### Title

UNKNOWN_SIDE_EFFECT_STATE — external operation outcome unknown (INC-07 / INC-14)

### Trigger

Timeout/ambiguous provider response; reconciliation incomplete.

### Severity

SEV-2

### Scope

Per external operation tied to `attempt_id`.

### Symptoms

Uncertain if mutation occurred; retry policy blocked.

### Canonical signals

External operation diagnostic integration; terminal classification `unknown_side_effect` if emitted.

### Immediate safe action

1. **NO BLIND RETRY** — halt automated and manual retries until reconciled.  
2. Query durable evidence and external system of record per adapter appendix.  
3. Document ambiguity window.

### Do NOT

Rerun tool manually; invoke provider directly; hidden retry.

### Diagnosis

Reconcile provider state with evidence; classify to retryable or terminal.

### Recovery path

After reconciliation, use recovery plane or new canonical execution entry — never duplicate mutation blindly.

### Verification

Reconciliation record; single terminal outcome; no duplicate side effect.

### Escalation

SEV-2 business + reliability owner for financial/shipping class operations.

### Post-incident evidence

Ambiguity window, reconciliation result, `attempt_id`.

---

## RB-12 SECURITY_GOVERNANCE_DENIAL_SPIKE

### Title

SECURITY_GOVERNANCE_DENIAL_SPIKE — DENY spike (INC-10)

### Trigger

Admission/governance DENY rate high; security monitors alert.

### Severity

SEV-3 default; SEV-1 if breach confirmed.

### Scope

May be attack, bad client, wrong policy, or config — not automatically engine unhealthy.

### Symptoms

Many DENY outcomes; legitimate traffic may be blocked.

### Canonical signals

Governance admission result per request; `tenant_id`; policy id; rate by client principal.

### Immediate safe action

1. Classify scope: attack vs misconfiguration vs policy change.  
2. Do not declare Execution Engine unhealthy from DENY alone.  
3. Preserve audit evidence.

### Do NOT

Disable governance; skip policy; force allow; use allowing admission adapters in production.

### Diagnosis

Compare DENY reason codes; correlate with deploy/config changes.

### Recovery path

Fix policy/config/client; rate-limit bad actors via edge controls — governance plane remains authoritative.

### Verification

DENY rate normal; legitimate canonical execution entry succeeds.

### Escalation

SEV-1 security incident if cross-tenant breach suspected.

### Post-incident evidence

DENY samples (redacted), policy version, time window.

---

## RB-13 SHUTDOWN_INCOMPLETE

### Title

SHUTDOWN_INCOMPLETE — EE-B4-B phase failure (INC-11)

### Trigger

Shutdown does not reach success; mandatory evidence flush failure; final state persistence failure; worker leak.

### Severity

SEV-2

### Scope

Global process lifecycle.

### Symptoms

Partial shutdown; readiness `not_ready`; workers remain.

### Canonical signals

`ExecutionRuntimeShutdownPhase`; flush outcome; persist outcome; EE-B4-B compound failure ordering.

### Immediate safe action

1. Identify failing phase (drain, flush, persist, terminate).  
2. Cross-reference RB-05 for drain; RB-03 for mandatory flush failure.  
3. Classify before any process restart.

### Do NOT

Delete evidence; skip flush; terminate workers before persist order violated.

### Diagnosis

Map to EE-B4-B model §6–§7.

### Recovery path

Complete canonical shutdown sequence or controlled restart only when safe restart conditions met (incident model §10).

### Verification

Health/readiness per EE-B4-A; zero worker leaks; capacity released.

### Escalation

SEV-2 platform owner.

### Post-incident evidence

Phase timeline, flush/persist results.

---

## RB-14 TENANT_SCOPED_DEGRADATION

### Title

TENANT_SCOPED_DEGRADATION — impact isolated to tenant/profile (INC-13)

### Trigger

Failure metrics concentrated on one `tenant_id`; other tenants healthy.

### Severity

SEV-3

### Scope

tenant-only / profile-only / dependency-only — prove before global escalation.

### Symptoms

Single-tenant errors; global readiness may remain `ready`.

### Canonical signals

Per-tenant SLI slices; `tenant_id` on terminal events; capability health for tenant profile.

### Immediate safe action

1. Prove isolation with tenant-scoped diagnostic query.  
2. Open RB-06 / RB-09 / RB-12 as appropriate — not RB-01 global halt.  
3. Notify tenant owner.

### Do NOT

Escalate to global outage without cross-tenant evidence; disable global governance.

### Diagnosis

Compare tenant cohort to global baseline for same window.

### Recovery path

Tenant-specific config/provider fix; resubmit via **canonical platform execution entry**.

### Verification

Tenant SLI normalized; global SLIs unchanged.

### Escalation

Tenant support owner; global only if mandatory persistence or capacity globally impacted.

### Post-incident evidence

`tenant_id`, incident window, scoped root cause.

---

## Adapter and deployment appendices

Vendor-specific remediation (cloud provider consoles, model vendor status pages, credential rotation runbooks) lives in **deployment/adapter documentation** — not in core Execution Engine runbooks above. Core runbooks refer only to **provider adapter unavailable** and canonical platform paths.
