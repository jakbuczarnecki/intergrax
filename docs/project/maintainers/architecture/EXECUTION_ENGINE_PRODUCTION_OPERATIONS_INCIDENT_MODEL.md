# Execution Engine — Production Operations & Incident Model (EE-B4-C)

**Status:** Certified reference on `development` (EE-B4-C).  
**Scope:** Operator incident taxonomy, severity, degraded modes, and diagnostic-first response — **no** second incident runtime.

## 1. Purpose

Prove that a production operator can, using only canonical platform signals and runbooks:

1. Recognize an incident  
2. Classify it (taxonomy + severity)  
3. Take an approved action  
4. Verify recovery  

without manual bypass of Execution Engine contracts.

```text
runtime fact
    ↓
diagnostics / health / evidence (EE-B4-A)
    ↓
incident classification (this model)
    ↓
approved operator action (runbooks)
    ↓
canonical recovery / operational mechanism (NPSC-5E, admission, hosting)
    ↓
verification
```

**Forbidden:**

```text
incident → manual bypass → direct DB mutation → hidden retry
→ checkpoint deletion → evidence deletion → forced execution
```

## 2. Relationship to EE-B4-A / EE-B4-B

| Plane | Owner doc | Operator use |
| ----- | --------- | ------------ |
| Health / readiness / liveness | EE-B4-A | Classify global vs scoped impact before action |
| Shutdown / drain | EE-B4-B | INC-11 and safe restart preconditions |
| SLI facts | `testing_support/execution_operational_readiness/sli_catalog.py` | Tie incidents to error budget consumers |

Operational assessment reference: `testing_support/execution_operational_readiness` (not runtime authority).

## 3. Incident taxonomy (canonical minimum)

| ID | Title | Primary runbook |
| -- | ----- | --------------- |
| INC-01 | Capacity saturation | RB-02 |
| INC-02 | Mandatory evidence persistence unavailable | RB-03 |
| INC-03 | Best-effort observability exporter unavailable | RB-04 |
| INC-04 | Worker failure / pool degradation | RB-06 |
| INC-05 | Nexus / fan-out degradation | RB-06 + §8 (not second engine) |
| INC-06 | Tool/provider dependency outage | RB-09 |
| INC-07 | Retry exhaustion | RB-10 |
| INC-08 | Recovery failure | RB-07 |
| INC-09 | Checkpoint incompatibility / corruption | RB-08 |
| INC-10 | Security / governance denial spike | RB-12 |
| INC-11 | Graceful shutdown incomplete | RB-13 |
| INC-12 | ExecutionRuntime fatal / unhealthy | RB-01 |
| INC-13 | Tenant-scoped degradation | RB-14 |
| INC-14 | Unknown / unclassified runtime failure | RB-01 → classify |

Typed catalog: `testing_support/operations/incident_taxonomy.py`.

## 4. Operational severity (single framework)

| Level | Guidance |
| ----- | -------- |
| **SEV-1** | Global execution impossible; mandatory evidence integrity unavailable; cross-tenant security breach; systemic data corruption |
| **SEV-2** | Major degradation; recovery subsystem unavailable globally; global capacity unavailable; widespread mandatory dependency failure |
| **SEV-3** | Scoped provider outage; tenant-local degradation; partial worker pool degradation; retry exhaustion for bounded cohort |
| **SEV-4** | Best-effort observability outage; minor diagnostics degradation; non-blocking operational anomaly |

Severity is assigned **after** scope identification — never from a single raw metric.

## 5. Diagnostics-first invariant

Every incident response starts with:

```text
identify scope (tenant / profile / global / dependency)
    ↓
collect canonical evidence (durable runtime events, checkpoint read, admission outcomes)
    ↓
classify failure (taxonomy + severity)
    ↓
select runbook
```

Not: `restart everything` without classification.

## 6. Operator identifiers

Runbooks require explicit correlation fields when available:

- `tenant_id`, `task_id`, `run_id`, `attempt_id`, `execution_id`  
- `ExecutionRuntimeShutdownPhase` (EE-B4-B)  
- Capacity admission outcome (EE-B1.2)  
- Recovery disposition (NPSC-5E)  

Avoid unqualified “check logs” without state/event/diagnostic/scope.

## 7. Evidence as source of truth

When durable runtime events / causal evidence reconstruction is available, operators use that plane — not raw OTLP export — to establish facts (NPSC-5F ownership unchanged).

## 8. Nexus vs ExecutionRuntime

Nexus orchestration / fan-out failures are **INC-05**. Nexus is not a second Execution Engine. Global readiness follows EE-B4-A; orchestration degradation is scoped unless root capacity or mandatory persistence is impacted.

## 9. Degraded mode map (summary)

| Incident | Typical disposition |
| -------- | ------------------- |
| INC-01 | `stop_accepting` (not ready) |
| INC-02 | `fail_closed` |
| INC-03 | `continue_degraded` (ready may remain true) |
| INC-04–07, 13 | `continue_degraded` (scope-dependent) |
| INC-08–09 | `fail_closed` for affected executions |
| INC-10 | `continue_normally` until scope proves engine fault |
| INC-11 | `stop_accepting` during shutdown |
| INC-12, 14 | `fail_closed` / `stop_accepting` until classified |

Full enum: `DegradedModeDisposition` in incident taxonomy module.

## 10. Safe vs unsafe process restart

**Safe restart when all hold:**

- Stop accepting confirmed or incident does not require in-flight unknown side effects  
- Mandatory evidence semantics known (persisted or explicitly fail-closed)  
- No unknown external side-effect state (RB-11)  
- Checkpoint identity consistent or recovery not attempted blindly  
- Operator knows canonical recovery path (NPSC-5E)  

**Unsafe restart:**

- Unknown provider side effect  
- Unpersisted mandatory evidence  
- Checkpoint identity conflict  
- Active execution with uncertain external mutation  

**Invariant:** `process restart ≠ execution recovery` (recovery uses NPSC-5E + canonical platform execution entry).

## 11. Diagnostic query model (minimum)

| Dimension | Use |
| --------- | --- |
| tenant | Isolate INC-13 |
| run / attempt / execution | Trace terminal outcome and recovery |
| time window | Correlate denial spikes |
| failure category | Map to taxonomy |
| policy/governance result | INC-10 triage |

## 12. SLI / error budget linkage

Incidents consume deployment error budget via EE-B4-A SLIs (targets configured by operator):

| SLI | Typical incident consumers |
| --- | -------------------------- |
| `execution_failure_rate` | INC-07, INC-12, INC-14 |
| `admission_reject_defer_rate` | INC-01 |
| `capacity_utilization` | INC-01 |
| `mandatory_evidence_persistence_failure_rate` | INC-02 |
| `recovery_success_rate` | INC-08 |
| `worker_failure_rate` | INC-04 |
| `dependency_failure_rate` | INC-06 |
| `shutdown_drain_duration` | INC-11 |

No SLO runtime is introduced in EE-B4-C.

## 13. Post-incident evidence package (minimum)

- `tenant_id`, `task_id`, `run_id`, `attempt_id`, `execution_id`  
- Incident window (UTC)  
- Failure category (INC-xx)  
- Policy/governance result (if applicable)  
- Recovery result  
- Final terminal state  

## 14. Architecture gates (certification)

| Forbidden second owner | Count in `intergrax/` for EE-B4-C |
| ---------------------- | ----------------------------------- |
| IncidentRuntime | 0 |
| RunbookEngine / OperatorRuntime | 0 |
| Second recovery / retry / evidence / governance / execution owner | 0 (unchanged planes) |

## 15. Runbook index

Canonical operator procedures: `docs/project/maintainers/runbooks/EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md`.

Provider-specific remediation belongs in deployment/adapter appendices only (§ pluginability).
