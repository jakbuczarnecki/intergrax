# Execution Engine — Operational Readiness, SLO & Health Model (EE-B4-A)

**Status:** Certified reference model on `development` (EE-B4-A).  
**Scope:** Operator-visible classification only — no second operational runtime.

## 1. Purpose

Answer whether an operator can determine, without guessing from a single metric:

- Is the Execution Engine **alive**?
- Is it **ready** to accept new work safely?
- Is it **healthy** as a whole?
- Is it **degraded**, **saturated**, or **unable** to meet mandatory persistence?

```text
raw runtime facts
      ↓
existing diagnostics / observability / admission contracts
      ↓
typed operational assessment (projection)
      ↓
HEALTH / READINESS / LIVENESS / SATURATION
      ↓
operator decision
```

**Forbidden pattern:** `metric → hidden runtime authority`. Operational diagnostics do not steer execution outside canonical policy/admission owners.

## 2. Ownership

| Concern | Owner | Location |
| --- | --- | --- |
| Runtime facts (events, persistence outcomes, capacity counters) | Execution + NPSC planes | `intergrax/runtime/execution/**`, events, recovery |
| Root capacity admission | EE-B1.2 | `ExecutionCapacityAdmissionPort`, `assess_root_execution_capacity` |
| Mandatory auditability readiness | Observability contract | `intergrax/runtime/observability/auditability_health.py` |
| Best-effort export | Observability export boundary | `export_policy`, `try_export_observability_envelope` |
| Shutdown phase semantics | EE-B1.1 | `ExecutionRuntimeShutdownPhase` |
| Capability / dependency health (scoped) | Tier-3 projection | `applications/contracts/capability_health/**` |
| Hosted component health | Hosting plane | `intergrax/hosting/**` |
| **Operational assessment composition** | Operator / deployment layer | Reference: `testing_support/execution_operational_readiness` (certification only) |
| **SLI computation** | Operational / observability layer | Derived from facts + exporters |
| **SLO targets** | Deployment / operator config | Not hardcoded in execution core |

## 3. Hard separation

```text
runtime fact ≠ diagnostic interpretation ≠ operational readiness ≠ execution authority
```

## 4. Health model

**Health** — does the system function correctly as a whole (for the assessed scope)?

| State | Meaning |
| --- | --- |
| `healthy` | No material degradation for the scope |
| `degraded` | Functioning with material operational limitation (capacity pressure, export outage, shutdown in progress, scoped dependency/worker degradation) |
| `unhealthy` | Cannot meet mandatory persistence / auditability requirements when they apply |

Health does **not** automatically imply readiness to accept new work.

Canonical cross-domain capability status (Tier-3): `CapabilityHealthStatus` (`ready` / `degraded` / `unavailable`) — scoped projection, not global execution authority.

## 5. Readiness model

**Readiness** — can the system safely accept **new** work **now** (for the assessed scope)?

| State | Meaning |
| --- | --- |
| `ready` | Startup complete; accepting work; mandatory evidence path available; capacity allows admission |
| `not_ready` | Stop accepting, drain, saturated root capacity (REJECT/DEFER), or mandatory evidence unavailable |
| `starting` | Process up but startup not complete |

Example: **live + capacity exhausted ⇒ not ready** (EE-B1.2 semantics unchanged).

**Global vs profile-specific:** Global readiness uses root capacity and global mandatory persistence. Profile-specific readiness may additionally require capability health for dependencies declared mandatory for that profile (`EffectiveCapabilityHealth`).

## 6. Liveness model

**Liveness** — is the process/runtime alive and able to progress without external restart?

| State | Meaning |
| --- | --- |
| `live` | Process alive and not in terminal worker termination phase |
| `not_live` | Process down or `TERMINATE_WORKERS` phase reached |

Liveness is **not** tied to all dependency health checks or OTLP availability.

## 7. Degradation semantics

| Condition | Typical health | Typical readiness |
| --- | --- | --- |
| OTLP / best-effort export unavailable | `degraded` | `ready` (execution + canonical store continue — EE-B2) |
| Mandatory evidence persistence unavailable | `unhealthy` | `not_ready` |
| Capacity saturated (REJECT) | `degraded` | `not_ready` |
| Shutdown `STOP_ACCEPTING_NEW_WORK` | `degraded` | `not_ready` |
| Drain active executions | `degraded` | `not_ready` for new work; liveness may remain `live` |
| Worker isolated failure | `degraded` (pool scope) | `ready` at global level unless capacity/policy says otherwise |
| Single dependency provider failure | `degraded` at dependency scope | global readiness unchanged unless dependency is mandatory for all profiles |

## 8. Saturation model

Derived from EE-B1.2 `ExecutionCapacityAssessmentContext` + `assess_root_execution_capacity`:

| Classification | Rule |
| --- | --- |
| `normal` | Decision `ALLOW` and not one-slot-remaining |
| `approaching_saturation` | `ALLOW` with `active == capacity_limit - 1` (when limit > 1) |
| `saturated` | Decision `REJECT` or `DEFER` due to full slots |

No arbitrary utilization thresholds in execution core; operators may compute `active/capacity` as an SLI.

## 9. Readiness matrix (global scope)

| Condition | Liveness | Readiness | Health |
| --- | ---: | ---: | --- |
| Normal | true | true | healthy |
| Capacity saturated | true | false | degraded |
| Mandatory evidence unavailable | true | false | unhealthy |
| OTLP unavailable | true | true | degraded |
| `STOP_ACCEPTING_NEW_WORK` | true | false | degraded |
| Worker isolated failure | true | true/degraded (scoped) | degraded |
| Fatal process down | false | false | unhealthy |

## 10. Security & governance

| Event | Operational interpretation |
| --- | --- |
| Policy DENY | Successful enforcement — **does not** lower global health |
| Security denial / abuse rejection | **does not** lower global health |
| User error (invalid request) | Request outcome — not system health |
| Capacity rejection | Backpressure — not security failure |
| Dependency failure | Scoped degradation |
| Runtime failure | Health / readiness per facts |

## 11. Mandatory vs optional dependencies

| Class | Readiness impact |
| --- | --- |
| Mandatory runtime event / evidence persistence | Fail-closed `not_ready` when unavailable |
| Required diagnostics wiring (`diagnostics_required`) | Fail-closed per `resolve_auditability_ready` |
| Best-effort observability export | Degraded health only |
| Optional dependency provider | Scoped `dependency` health only |

## 12. Multi-tenant scope

Tenant-scoped resource failure must not imply global unhealthy when architecture isolates tenants. Assessment carries `scope`: `global`, `tenant`, `execution_profile`, `dependency`, `worker_pool`.

## 13. Health endpoints (hosting)

Hosting exposes component-level models (`HostedApplicationComponentHealth` with `healthy`, `ready`, `state`, `detail_code`, `safe_message`). Semantics:

| Probe | Intent |
| --- | --- |
| `/live` | Process alive — no heavy dependency I/O |
| `/ready` | May include required operational dependencies |
| `/health` | Aggregate operator view |

Probes must be read-only (no retry, recovery, or queue mutation). No self-DoS (full DB scan, full reconstruction).

## 14. SLI catalog

Execution Engine emits **facts**; SLIs are defined with numerator, denominator, window, owner, and scope. Canonical catalog: `testing_support/execution_operational_readiness/sli_catalog.py` (`EXECUTION_ENGINE_SLI_CATALOG`).

Minimum SLIs: execution success/failure rate, admission reject/defer rate, execution latency, admission wait, capacity utilization, worker failure rate, dependency failure rate, mandatory evidence persistence failure rate, recovery success rate, shutdown/drain duration.

## 15. SLO contract shape

SLO targets are **deployment/operator configurable**. Core defines shape only:

```text
sli_id
target_description (operator-defined)
measurement_window
target_owner = deployment_operator
error_budget_model = allowed_bad_events_over_window
```

No arbitrary availability percentages in execution core. Burn rate is an operational derived metric.

## 16. Provider neutrality

Core does not import Prometheus, Datadog, Grafana, or vendor SDKs. Exporters/sinks remain pluginable via existing observability export abstractions.

## 17. Performance & safety

Operational evaluation is O(number of registered inputs), side-effect free, no unbounded scans. No secrets in status payloads.

## 18. Cross-session exclusions (NPSC-5F)

EE-B4-A does **not** modify or requalify:

- `intergrax/runtime/observability/causal_evidence.py`
- `intergrax/runtime/observability/causal_evidence_export.py`
- `intergrax/runtime/observability/export_boundary.py`
- `intergrax/runtime/background_execution/**`
- NPSC-5F protected fingerprints, baselines, resignoff

Findings on those surfaces → document handoff to parallel session.

## 19. Certification reference implementation

`testing_support/execution_operational_readiness/assessment.py` — `assess_execution_operational_state` composes existing contracts for gate tests. It is **not** wired as runtime authority.

## 20. Forbidden second runtimes

No `HealthRuntime`, `MetricsRuntime`, `SloEngine`, `OperationalEventBus`, `ReadinessScheduler`, `HealthStateMachineRuntime` in production (`intergrax/`).
