# END_TO_END_SYSTEM — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** END_TO_END_SYSTEM
- **Owning architecture/program:** NEXUS_EXECUTION_FLOW · TIER3_APPLICATION_ENVIRONMENT · RELIABILITY_FAILURE_AND_HITL (cross-layer execution composition)
- **Tier(s):** Tier-3 `intergrax/applications/_shared/` (host runtime, MCP, task routes, async dispatch); Tier-1 `intergrax/runtime/task/` (UnifiedTaskRunner, ActiveTaskRegistry)
- **audited_sha:** `563076c553fd7b9d2611b71fd4137b8164a58d81`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`
- **Scope in:**
  - cross-layer canonical execution spine under falsification
  - Tier-3 host composition vs surface execution semantics
  - tenant/model routing identity from Task/Run execution context
  - configured UnifiedTaskRunner / task enricher parity across HTTP/MCP/async surfaces
  - task-control plane (autonomy, cancel) vs Governance authorization
  - ActiveTaskRegistry ownership and concurrency
  - durable async terminal-result recovery after process restart
  - public async error contract vs internal diagnostics
  - positive controls: canonical spine, UER/Nexus split, non-duplication of owned layer findings
- **Scope out:**
  - remediation implementation
  - source/test/CI/script changes
  - duplicating ITI TaskId==RunId defects (INTERFACE_TASK_INTAKE)
  - duplicating IDENTITY_TRUST principal→Task tenant binding
  - duplicating PCM checkpoint CAS/multi-host defects
  - duplicating observability durability defects
  - duplicating T3 EnvironmentSnapshot provenance (T3-SNAPSHOT-PROVENANCE-INTEGRITY)
  - duplicating SECURITY API-key/admin defects
  - inventing a second end-to-end runtime subsystem
- **Prior audit reference(s):** [`INTERFACE_TASK_INTAKE`](INTERFACE_TASK_INTAKE.md) (ITI-FIX-C direct-Nexus bypass — distinct from runner enricher parity); [`IDENTITY_TRUST`](IDENTITY_TRUST.md); [`POLICY_GOVERNANCE`](POLICY_GOVERNANCE.md); [`SECURITY_BOUNDARIES`](SECURITY_BOUNDARIES.md); [`OBSERVABILITY_EVIDENCE`](OBSERVABILITY_EVIDENCE.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `d4e3ec2398261e791deb946f26c52b336ed58371`

## Scope / ownership mapping

| Concept | Canonical ownership |
|---------|---------------------|
| Audit unit (Protocol v2 layer code) | **END_TO_END_SYSTEM** |
| Canonical task execution service / runner convergence | **NEXUS_EXECUTION_FLOW** |
| Host composition / tenant-aware LLM wiring / one configured runner | **TIER3_APPLICATION_ENVIRONMENT** |
| Durable async terminal outcome / safe external errors / control identity | **RELIABILITY_FAILURE_AND_HITL** |
| Autonomy authorization / governance transitions | **POLICY_GOVERNANCE** — cross-link; do not duplicate |
| Safe error mapping / redaction policy | **SECURITY_BOUNDARIES** · **OBSERVABILITY_EVIDENCE** — cross-link |
| Per-layer report | `docs/audit_results/2026-08-18/END_TO_END_SYSTEM.md` |
| Target invariants (Nexus flow) | `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` — [Protocol v2 END_TO_END_SYSTEM target invariants (2026-08-18)](#protocol-v2-end-to-end-system-target-invariants-2026-08-18) |
| Target invariants (Tier-3 host) | `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — [Protocol v2 END_TO_END_SYSTEM Tier-3 composition target invariants (2026-08-18)](#protocol-v2-end-to-end-system-tier3-composition-target-invariants-2026-08-18) |
| Target invariants (reliability) | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` — [Protocol v2 END_TO_END_SYSTEM async/control target invariants (2026-08-18)](#protocol-v2-end-to-end-system-asynccontrol-target-invariants-2026-08-18) |

## Canonical flow under test

```text
surface (HTTP / MCP / queue / async)
  → canonical normalized intake (where applicable)
  → Task
  → UnifiedTaskRunner (with host-owned task enricher)
  → NexusLoop
  → TaskResult
```

Cross-layer falsification asked whether every supported surface receives the **same configured execution service**, whether runtime identity (tenant, routing context) derives from the **concrete Task/Run**, whether **task-control** mutations cross **Governance**, whether **ActiveTaskRegistry** binds the **exact active execution**, and whether **async** completion survives restart as a **user-retrievable outcome** with **safe external errors**.

## Executive summary

**Verdict: FAIL.** Four accepted HIGH and two accepted MEDIUM findings show that host runtime can pass tenant context to environment composition yet wire Nexus LLM resolution with literal `tenant_id="default"`; MCP can construct `UnifiedTaskRunner(nexus_loop)` without the canonical reliability task enricher while HTTP uses `build_reliability_task_enricher()`; harness autonomy route mutates live task governance without canonical Governance authorization; SQLite async index persists terminal status but not `TaskResult` so restart yields status-only completion; `ActiveTaskRegistry` silently overwrites colliding `TaskId` keys; and async failure strings expose raw `ExceptionClass: message` to callers. Positive controls: the canonical architectural spine remains sound; `UnifiedTaskRunner` correctly owns registry lifecycle and `llm_tenant_scope`; resume checkpoint identity conflict checks exist; Nexus vs UER responsibility split remains appropriate; remediation is composition and contract convergence — not a new runtime. Remediation is **ACCEPTED / PLANNED**, not implemented.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-END_TO_END_SYSTEM-01 (E2E-01)

- **Severity:** HIGH
- **Category:** CROSS-LAYER IDENTITY / MODEL ROUTING
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-EXECUTION-CONTEXT-INTEGRITY
- **Claim falsified:** LLM routing context for product execution derives from the current canonical Task/Run execution identity — not a hard-coded default tenant.
- **Observation:** `build_harness_host_runtime` accepts `tenant_id` and uses it in environment composition, but passes literal `tenant_id="default"` to `resolve_environment_llm_adapter()` when constructing Nexus. `resolve_environment_llm_adapter` / LLM resolver builds `RoutingContext` from `tenant_id`; routing context participates in environment/model routing. A tenant-specific Task may therefore execute through an adapter/routing context materialized as tenant `"default"`.
- **Location:**
  - `intergrax/applications/_shared/harness_host_runtime.py` — `build_harness_host_runtime`, Nexus LLM wiring
  - `intergrax/applications/_shared/llm_resolver.py` — `resolve_environment_llm_adapter`, `RoutingContext`
- **Impact:** Tenant-specific tasks may route models/environments under wrong tenant authority.
- **Confidence:** CONFIRMED

### AUDIT-20260818-END_TO_END_SYSTEM-02 (E2E-02)

- **Severity:** HIGH
- **Category:** SURFACE PARITY / RUNTIME SEMANTICS
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-EXECUTION-CONTEXT-INTEGRITY
- **Claim falsified:** Every supported surface consumes one Tier-3–materialized configured task execution service with all mandatory host-owned enrichment — not independent `UnifiedTaskRunner(nexus_loop)` reconstruction.
- **Observation:** `build_nexus_mcp_server` constructs `UnifiedTaskRunner(nexus_loop)` directly. Canonical task-control/Tier-3 wiring uses `build_reliability_task_enricher()`, which applies reliability defaults and optional checkpoint, compensation, and idempotency-store enrichment. `UnifiedTaskRunner` applies those semantics only when configured with a `task_enricher`. MCP can therefore use the nominal canonical runner/Nexus path while executing a different effective task configuration from a host surface supplied with the canonical enricher.
- **Location:**
  - `intergrax/applications/_shared/mcp_nexus_server.py` — `build_nexus_mcp_server`, `UnifiedTaskRunner(nexus_loop)`
  - `intergrax/applications/_shared/task_control_wiring.py` — `build_reliability_task_enricher()`
  - `intergrax/runtime/task/unified_task_runner.py` — `task_enricher` application
- **Impact:** Surface-dependent execution semantics (reliability enrichment, checkpoint/compensation/idempotency) without operator-visible divergence.
- **Confidence:** CONFIRMED

### AUDIT-20260818-END_TO_END_SYSTEM-03 (E2E-03)

- **Severity:** HIGH
- **Category:** CONTROL PLANE / GOVERNANCE BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-CONTROL-AUTHORITY-INTEGRITY
- **Claim falsified:** Autonomy change is a governed control-plane operation: authenticated principal + Task/Run + requested transition → canonical Governance authorization → authorized transition evidence → runtime state application.
- **Observation:** Harness task route `POST /{task_id}/autonomy` is protected only by the generic harness API-key dependency and calls `set_task_autonomy()`. `set_task_autonomy` directly mutates `task.options.governance.autonomy_level` and metadata on the live active Task. No canonical Governance authorization/evaluation, actor-bound decision, tenant/resource authorization, allowed-transition policy, or durable authority event is proven at this operation boundary.
- **Location:**
  - `intergrax/applications/_shared/harness_task_routes.py` — `POST /{task_id}/autonomy`
  - `intergrax/applications/_shared/task_control.py` — `set_task_autonomy()`
- **Impact:** Security-sensitive autonomy transitions without governed authorization evidence.
- **Confidence:** CONFIRMED

### AUDIT-20260818-END_TO_END_SYSTEM-04 (E2E-04)

- **Severity:** HIGH
- **Category:** ASYNC WORKFLOW / TERMINAL RESULT DURABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-ASYNC-OUTCOME-INTEGRITY
- **Claim falsified:** Durable async execution retains a canonical terminal-result relation: TaskId + RunId → durable `TaskResult` / result reference / execution-journal projection recoverable after process restart.
- **Observation:** PRODUCT/STRICT async task resolver defaults to `SqliteAsyncTaskIndex`. During the live process `AsyncTaskHandle` retains `TaskResult` and `get_async_status()` can expose terminal state + answer. `SqliteAsyncTaskIndex` persists only `task_id`, `status`, `error`, `state`. On `get()` after restart it reconstructs `AsyncTaskHandle` without `TaskResult`. Terminal task status survives restart but the user-visible terminal answer/result does not.
- **Location:**
  - `intergrax/applications/_shared/async_task_dispatch.py` — in-memory handle + status exposure
  - `intergrax/applications/_shared/async_task_index_resolver.py` — PRODUCT/STRICT default index
  - `intergrax/applications/_shared/sqlite_async_task_index.py` — persisted fields, `get()` reconstruction
- **Impact:** Completed async tasks appear status-complete without retrievable user outcome after restart.
- **Confidence:** CONFIRMED

### AUDIT-20260818-END_TO_END_SYSTEM-05 (E2E-05)

- **Severity:** MEDIUM
- **Category:** ACTIVE EXECUTION REGISTRY / CONCURRENCY
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-CONTROL-AUTHORITY-INTEGRITY
- **Claim falsified:** Registry registration is ownership-aware: duplicate `TaskId` is an explicit conflict or registration binds `TaskId` + `RunId`/attempt/registration token; unregister removes only the owned registration.
- **Observation:** `ActiveTaskRegistry` is a global dict keyed only by `task_id`. `register()` silently overwrites an existing task with the same `TaskId`. `unregister(task_id)` unconditionally removes whatever currently owns that key. Concurrent/delayed executions with colliding `TaskId` can replace each other's control-plane registration, cause an older run to unregister a newer active run, and make cancel/autonomy controls target or lose the wrong execution.
- **Location:**
  - `intergrax/runtime/task/active_task_registry.py` — `register()`, `unregister()`
  - `intergrax/runtime/task/unified_task_runner.py` — registry lifecycle integration
- **Impact:** Wrong-task control targeting under concurrent or delayed colliding identities.
- **Confidence:** CONFIRMED

### AUDIT-20260818-END_TO_END_SYSTEM-06 (E2E-06)

- **Severity:** MEDIUM
- **Category:** ERROR BOUNDARY / INFORMATION LEAKAGE
- **Status at publication:** ACCEPTED
- **Remediation block:** E2E-ASYNC-OUTCOME-INTEGRITY
- **Claim falsified:** External async failure contract exposes stable `reason_code`, safe message, and correlation/run identifier — not raw internal exception strings.
- **Observation:** Both in-memory and SQLite async dispatch store failures as `f"{exc.__class__.__name__}: {exc}"`. `get_async_status()` returns that raw error string to the caller. No user-safe error mapping is applied at the external boundary.
- **Location:**
  - `intergrax/applications/_shared/async_task_dispatch.py` — failure capture and status return
  - `intergrax/applications/_shared/sqlite_async_task_index.py` — persisted error field
- **Impact:** Internal diagnostic detail may leak to external callers.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| Canonical spine: surface → Task → UnifiedTaskRunner → NexusLoop → TaskResult | NOT falsified |
| UnifiedTaskRunner owns ActiveTaskRegistry lifecycle and `llm_tenant_scope` | NOT falsified |
| Resume checkpoint identity conflict checks exist | NOT falsified |
| Nexus vs UER responsibility split remains sound | NOT falsified |
| No new end-to-end runtime subsystem required | NOT falsified |
| Remediation is composition and contract convergence | NOT falsified |

## Duplicate ownership / cross-links

| Existing finding / domain | Relationship |
|-----------------------------|--------------|
| **INTERFACE_TASK_INTAKE / ITI-FIX-C** | Direct Nexus bypass and runner convergence — E2E-02 is **equal runner enricher semantics**, not the already-recorded bypass |
| **IDENTITY_TRUST** | Principal→Task tenant binding — cross-link; E2E-01 is runtime routing-context materialization at Nexus wiring |
| **LLM_ADAPTERS** | Inference/routing plane — cross-link; do not duplicate LLM-FIX blocks |
| **POLICY_GOVERNANCE** | Canonical authorization for autonomy transitions — E2E-03 requires reuse, not a second policy engine |
| **SECURITY_BOUNDARIES / SEC-AUTHORITY-BOUNDARY-INTEGRITY** | API-key/admin boundary — cross-link; E2E-03 is governance on live task control |
| **OBSERVABILITY_EVIDENCE / journal** | Durable canonical result projection where it already exists — E2E-04 may reference, not duplicate |
| **PCM / checkpoint CAS** | Multi-host checkpoint defects — explicitly not duplicated |
| **T3-SNAPSHOT-PROVENANCE-INTEGRITY** | EnvironmentSnapshot provenance — explicitly not duplicated |
| **OBS observability durability** | Explicitly not duplicated |

## Root-cause remediation grouping

### E2E-EXECUTION-CONTEXT-INTEGRITY — configured runner + routing identity

**Priority:** P0  
**Findings:** E2E-01, E2E-02  
**Owners:** TIER3_APPLICATION_ENVIRONMENT · NEXUS_EXECUTION_FLOW  

Every supported surface receives the same configured execution service; runtime identity/routing context derives from the concrete Task/Run. Cross-link **ITI-FIX-C**, **IDENTITY_TRUST**, **LLM_ADAPTERS**.

### E2E-CONTROL-AUTHORITY-INTEGRITY — governed control + registry ownership

**Priority:** P0  
**Findings:** E2E-03, E2E-05  
**Owners:** NEXUS_EXECUTION_FLOW · RELIABILITY_FAILURE_AND_HITL  

Live task control operates on exact execution identity; security-sensitive transitions require canonical Governance authorization. Cross-link **POLICY_GOVERNANCE**, **SECURITY_BOUNDARIES**.

### E2E-ASYNC-OUTCOME-INTEGRITY — durable terminal outcome + safe errors

**Priority:** P0/P1  
**Findings:** E2E-04, E2E-06  
**Owner:** RELIABILITY_FAILURE_AND_HITL  

Async tasks retain durable user-retrievable terminal outcome after restart; external failures use safe reason contracts. Cross-link **OBSERVABILITY_EVIDENCE** / Unified Run Journal where appropriate.

## Architecture / plan sync state

| Doc | Section | Status |
|-----|---------|--------|
| `docs/project/architecture/NEXUS_EXECUTION_FLOW.md` | Protocol v2 END_TO_END_SYSTEM target invariants | SYNCED |
| `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md` | Protocol v2 END_TO_END_SYSTEM Tier-3 composition target invariants | SYNCED |
| `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` | Protocol v2 END_TO_END_SYSTEM async/control target invariants | SYNCED |
| `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md` | E2E-EXECUTION-CONTEXT-INTEGRITY, E2E-CONTROL-AUTHORITY-INTEGRITY | SYNCED |
| `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` | E2E-EXECUTION-CONTEXT-INTEGRITY cross-ref | SYNCED |
| `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md` | E2E-ASYNC-OUTCOME-INTEGRITY | SYNCED |

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `563076c553fd7b9d2611b71fd4137b8164a58d81`; current `development` HEAD was not re-audited beyond persistence sync.
- Remediation not performed in this task.
- No source, test, CI, or script changes.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 6 (`AUDIT-20260818-END_TO_END_SYSTEM-01` … `06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
