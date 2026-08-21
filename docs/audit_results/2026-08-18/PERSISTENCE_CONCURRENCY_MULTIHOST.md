# PERSISTENCE_CONCURRENCY_MULTIHOST — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Audit unit:** PERSISTENCE_CONCURRENCY_MULTIHOST
- **Owning architecture/program:** RELIABILITY_FAILURE_AND_HITL · PLATFORM_FOUNDATION (cross-layer)
- **Tier(s):** Tier-1 `intergrax/runtime/` (idempotency, long-running scheduler, runtime events); Tier-2 `agents/persistence/` (checkpoints, compensation queue); Tier-3 `intergrax/applications/_shared/` (reliability wiring); Tier-0 `intergrax/integrations/` (relational store contract)
- **audited_sha:** `a786e3a2202b105f0d3a38afff8f79ea34255f05`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 6 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 7 ACCEPTED 2026-08-21
- **Architecture doc(s):**
  - `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/architecture/PLATFORM_FOUNDATION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/maintainers/plans/PLATFORM_FOUNDATION.md`
- **Reference architecture (evidence only — not modified):**
  - `docs/project/architecture/AGENT_DISTRIBUTION.md` — §§23–25, §34
- **Scope in:**
  - production topology vs process-local/durable/shared persistence capability qualification
  - idempotency crash consistency and false exactly-once claims
  - compensation queue multi-worker duplicate consumption
  - agent checkpoint lost-update / monotonic revision
  - long-running scheduler multi-host duplicate resume
  - domain persistence port concurrency semantics vs minimal `RelationalStore`
  - SQLite schema evolution fail-open startup
  - positive controls: Agent Distribution CAS target, honest single-process limitations, Redis idempotency capability existence
- **Scope out:**
  - remediation implementation
  - source/test/CI/script/schema/provider changes
  - modifying `AGENT_DISTRIBUTION` (reference CAS precedent only)
  - duplicating MEMORY-04, AHI activation CAS, or ECP multi-host anti-flapping findings
  - reopening historical REL/REL-ADV/RELIABILITY-LC Done rows
- **Prior audit reference(s):** [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md); [`TOOLS`](TOOLS.md); [`MEMORY`](MEMORY.md); [`ADAPTIVE_HARNESS_INTELLIGENCE`](ADAPTIVE_HARNESS_INTELLIGENCE.md); [`ELASTIC_CAPACITY_AND_SCALING`](ELASTIC_CAPACITY_AND_SCALING.md)
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** —

## Scope / ownership mapping

| Concept | Canonical ownership |
|---------|---------------------|
| Audit unit (Protocol v2 layer code) | **PERSISTENCE_CONCURRENCY_MULTIHOST** |
| Recovery / side-effect coordination / checkpoint / scheduler honesty | **RELIABILITY_FAILURE_AND_HITL** |
| Cross-layer persistence topology capability qualification | **PLATFORM_FOUNDATION** |
| CAS / serving_pointer_revision target precedent | **AGENT_DISTRIBUTION** (reference only) |
| Per-layer report | `docs/audit_results/2026-08-18/PERSISTENCE_CONCURRENCY_MULTIHOST.md` |
| Target invariants (recovery) | `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md` — [Protocol v2 persistence/concurrency multihost target invariants (2026-08-18)](#protocol-v2-persistence-concurrency-multihost-target-invariants-2026-08-18) |
| Target invariants (topology) | `docs/project/architecture/PLATFORM_FOUNDATION.md` — [Protocol v2 persistence topology target invariants (2026-08-18)](#protocol-v2-persistence-topology-target-invariants-2026-08-18) |

## Executive summary

**Verdict: FAIL.** Six accepted HIGH and one accepted MEDIUM finding show that production/STRICT composition can rely on process-local idempotency without mechanical topology qualification; idempotency state machine can permanently block after crash between external effect and completion; compensation and scheduler consumption lack atomic claim/lease semantics for multi-worker operation; checkpoint writes can regress newer execution state; minimal `RelationalStore` does not encode domain concurrency guarantees required for provider substitutability; and ad-hoc schema migration can fail open on unexpected errors. Positive controls: `SQLiteRuntimeEventStore` demonstrates sound local atomic sequencing; Agent Distribution documents honest single-process V1 and deferred multi-instance scale-out; LongRunningScheduler self-identifies as lab/single-process-first; Reliability architecture disclaims HA/full durable operator workflow; Redis distributed idempotency capability already exists and remediation stays provider-neutral. Remediation is **PLANNED**, not implemented. Findings define cross-layer persistence invariants — they do not duplicate domain-owned MEMORY, AHI, or ECP remediation.

## Verdict

**FAIL** — 0 CRITICAL / 6 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-01 (PCM-01)

- **Severity:** HIGH
- **Category:** PRODUCTION TOPOLOGY / FAIL-OPEN PERSISTENCE
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-PERSISTENCE-TOPOLOGY-INTEGRITY
- **Claim falsified:** STRICT/multi-host production composition mechanically requires restart-safe/shared persistence capability per stateful mechanism; process-local stores cannot satisfy shared-multi-host requirements merely by implementing a store interface.
- **Observation:** Tier-3 reliability wiring creates `SQLiteIdempotencyStore` only when `idempotency_db_path` is explicitly supplied; otherwise `InMemoryIdempotencyStore` when idempotency is enabled. Reliability assembly validation only requires a non-None store. `HarnessHostRuntime` accepts optional `idempotency_db_path` and passes the resolved store into Nexus. `RuntimeContext.build` defaults missing `idempotency_store` to `InMemoryIdempotencyStore`. `production_mode`/`STRICT` does not mechanically require a restart-safe/shared idempotency backend.
- **Location:**
  - `intergrax/applications/_shared/reliability_wiring.py`
  - `intergrax/applications/_shared/reliability_assembly_resolver.py`
  - `intergrax/applications/_shared/harness_host_runtime.py`
  - `intergrax/runtime/nexus/engine/runtime_context.py`
- **Impact:** Multi-host or restart-sensitive deployments can pass assembly with insufficient idempotency topology.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-02 (PCM-02)

- **Severity:** HIGH
- **Category:** IDEMPOTENCY / CRASH CONSISTENCY
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-SIDE-EFFECT-COORDINATION-INTEGRITY
- **Claim falsified:** Idempotency state machine models execution uncertainty; does not claim exactly-once unless the complete external-effect protocol proves it.
- **Observation:** `SQLiteIdempotencyStore` atomically records STARTED and later transitions STARTED → COMPLETED. `IdempotentToolInvoker` performs `record_started` → external side effect → `record_completed`. Crash after the external side effect and before `record_completed` leaves a permanent STARTED entry. Later invocation sees STARTED and raises indefinitely. No owner, lease, expiry, fencing token, UNCERTAIN state, or reconciliation protocol exists. The class claims exactly-once semantics, which is stronger than proven behavior.
- **Location:**
  - `intergrax/runtime/tools/sqlite_idempotency_store.py`
  - `intergrax/runtime/tools/idempotent_invoker.py`
- **Impact:** Irreversible side effects can become permanently blocked without operator reconciliation path.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-03 (PCM-03)

- **Severity:** HIGH
- **Category:** MULTI-WORKER / DUPLICATE SIDE EFFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-SIDE-EFFECT-COORDINATION-INTEGRITY
- **Claim falsified:** Durable compensation consumption has atomic claim semantics: PENDING → CLAIMED/RUNNING(owner, lease/fence) → COMPLETED / RETRYABLE / FAILED.
- **Observation:** `CompensationQueueStore` exposes `list_pending` + `mark_completed`/`mark_failed`. No atomic claim / RUNNING transition / lease / fencing owner exists. `drain_pending_compensation_jobs` lists pending jobs, invokes compensation, then updates status. Two workers can obtain the same PENDING job and invoke the same compensation concurrently. The passed `idempotency_key` is not a queue ownership guarantee and may itself use a process-local store.
- **Location:**
  - `intergrax/agents/persistence/compensation_queue_store.py`
  - `intergrax/agents/persistence/compensation_queue_worker.py`
- **Impact:** Duplicate compensation side effects under multi-worker or multi-process operation.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-04 (PCM-04)

- **Severity:** HIGH
- **Category:** CHECKPOINT CONSISTENCY / LOST UPDATE
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-CHECKPOINT-SCHEDULER-INTEGRITY
- **Claim falsified:** Checkpoint mutation is version-fenced / monotonic; stale concurrent writer receives explicit conflict.
- **Observation:** `SQLiteAgentCheckpointStore` stores one row per `(run_id, tenant_id)`. `save()` blindly UPSERTs payload/saved_at on conflict. `AgentRunCheckpoint` includes `step_index`, side-effect ledger and trace step count, but persisted update does not enforce monotonic step/revision or expected-current value. A stale concurrent run/resume writer can overwrite a newer checkpoint.
- **Location:**
  - `intergrax/agents/persistence/checkpoint_store.py`
- **Impact:** Resume can regress execution state after concurrent writers or stale resume paths.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-05 (PCM-05)

- **Severity:** HIGH
- **Category:** SCHEDULER / MULTI-HOST DUPLICATE RESUME
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-CHECKPOINT-SCHEDULER-INTEGRITY
- **Claim falsified:** Shared/multi-host topology requires atomic due-item claim/lease/fence or canonical distributed worker/message bus with equivalent semantics; single-process limitation remains documented until verified distributed implementation exists.
- **Observation:** `LongRunningScheduler` explicitly documents itself as lab/single-process-first. `ScheduledResume` store `list_due()` reads PENDING rows without claim. Scheduler does: `list_due` → `ledger.has_action` → `resume_task` → `ledger.record_action` → `mark_completed`. `has_action` and `record_action` are separate operations. Two scheduler instances can pass all prechecks and resume the same task/human timeout.
- **Location:**
  - `intergrax/runtime/long_running/store.py`
  - `intergrax/runtime/long_running/scheduler.py`
- **Impact:** Duplicate resume under multi-host scheduler deployment.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-06 (PCM-06)

- **Severity:** HIGH
- **Category:** ARCHITECTURE / PROVIDER SUBSTITUTABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-PERSISTENCE-TOPOLOGY-INTEGRITY
- **Claim falsified:** Domain persistence ports own concurrency semantics (CAS, lease/claim, transactional commit, required isolation); provider adapter implements a domain port only when it can satisfy that port's guarantees; replacing SQLite with PostgreSQL alone does not create multi-host correctness.
- **Observation:** Canonical `RelationalStore` contract only exposes `connect` / `execute` / `fetch_all` / `close`. Provider wrappers mirror that minimal contract. The platform increasingly requires stronger domain persistence guarantees: CAS, optimistic revisions, atomic claims, fencing, multi-record transactions and recoverable activation boundaries. Agent Distribution already documents the correct target pattern: `serving_pointer_revision` CAS, `binding_revision` optimistic locking, one atomic activation boundary, rollback on partial persistence failure.
- **Location:**
  - `intergrax/integrations/contracts/relational_store.py`
  - `intergrax/integrations/providers/relational_store/sqlite/paths.py`
- **Impact:** Provider swap without domain port semantics gives false confidence in multi-host production correctness.
- **Confidence:** CONFIRMED

### AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-07 (PCM-07)

- **Severity:** MEDIUM
- **Category:** SCHEMA EVOLUTION / FAIL-OPEN STARTUP
- **Status at publication:** ACCEPTED
- **Remediation block:** PCM-SCHEMA-EVOLUTION-INTEGRITY
- **Claim falsified:** Schema evolution distinguishes expected idempotent migration conditions from real persistence failures and fails closed on unknown schema errors.
- **Observation:** `SQLiteTaskCheckpointStore` schema setup executes `ALTER TABLE` for `runtime_checkpoint_json` and catches every `sqlite3.OperationalError` with `pass`. Intent is to tolerate an already-existing column, but unrelated migration errors are also hidden.
- **Location:**
  - `intergrax/runtime/long_running/store.py` (task checkpoint store path via long-running persistence)
- **Impact:** Storage corruption or unexpected schema failures may be silently ignored at startup.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| `SQLiteRuntimeEventStore` — BEGIN IMMEDIATE + atomic sequence allocation + rollback | NOT falsified |
| Agent Distribution defines CAS, optimistic revision, atomic traffic activation target semantics | NOT falsified |
| Agent Distribution reference production V1 honestly states single OS process, process-local lifecycle stores, restart loses state, multi-instance scale-out unsupported | NOT falsified |
| Durable/multi-instance Agent Distribution explicitly deferred, not falsely claimed shipped | NOT falsified |
| `LongRunningScheduler` explicitly identifies itself as single-process/lab-first | NOT falsified |
| Reliability architecture explicitly disclaims HA/full durable operator workflow | NOT falsified |
| `SQLiteHumanDecisionStore` not falsely represented as distributed operator queue | NOT falsified |
| Process-local / durable-single-host stores remain valid for their declared topology | NOT falsified |
| Not every store needs to be globally distributed | NOT falsified |
| Redis distributed idempotency capability already exists; remediation stays provider-neutral | NOT falsified |
| MEMORY-04 blind profile overwrite remains owned by MEMORY — cross-link only | NOT falsified |
| AHI activation CAS findings remain owned by AHI | NOT falsified |
| ECP multi-host anti-flapping findings remain owned by ECP | NOT falsified |

## Duplicate-finding avoidance / cross-links

| Existing finding / domain | Relationship |
|---------------------------|--------------|
| **MEMORY-04** | Profile overwrite — owned by MEMORY; cross-link only |
| **AHI activation CAS** | Owned by ADAPTIVE_HARNESS_INTELLIGENCE — reuse pattern, do not duplicate |
| **ECP distributed execution** | Owned by ELASTIC_CAPACITY_AND_SCALING — cross-link only |
| **PBA-FIX-A / IDT-FIX-C** | Remain PLANNED — checkpoint port consumption and human decision provenance orthogonal |
| **TOOLS idempotency remediation** | Coordinate PCM-02 side-effect semantics |
| **AGENT_DISTRIBUTION §§23–25, §34** | Reference CAS/revision/activation boundary precedent — not modified |
| **PROVIDER_BACKEND_ABSTRACTION** | Cross-link PCM-06 domain port semantics |

## Single-host vs multi-host distinction

| Topology class | Intended use | Audit posture |
|----------------|--------------|---------------|
| **PROCESS_LOCAL** | Lab, single-process reference hosts | Valid when explicitly declared; must not satisfy STRICT/shared-multi-host requirements |
| **DURABLE_SINGLE_HOST** | Restart-safe single host (e.g. local SQLite file) | Valid for declared mechanisms on one host; not equivalent to shared multi-worker claim semantics |
| **SHARED_MULTI_HOST** | Production scale-out, multiple workers/schedulers | Requires qualified shared persistence capability, atomic claim/lease/fence or canonical worker/bus primitives per mechanism |

Process-local and durable-single-host stores remain legitimate for their declared topology. This audit identifies gaps when those stores are treated as sufficient for multi-host or STRICT composition without mechanical qualification.

## Root-cause remediation grouping

### PCM-PERSISTENCE-TOPOLOGY-INTEGRITY — topology capability qualification and domain port semantics

**Priority:** P0  
**Findings:** PCM-01, PCM-06  
**Owner:** PLATFORM_FOUNDATION (plan) · cross-layer with RELIABILITY  

Every stateful runtime mechanism declares the persistence capability required by its deployment topology. STRICT/multi-host composition mechanically rejects process-local or otherwise insufficient stores. Provider-neutral domain persistence ports own concurrency semantics. Cross-links: PROVIDER_BACKEND_ABSTRACTION, Agent Distribution CAS model.

### PCM-SIDE-EFFECT-COORDINATION-INTEGRITY — uncertain execution and exclusive compensation consumption

**Priority:** P0  
**Findings:** PCM-02, PCM-03  
**Owner:** RELIABILITY_FAILURE_AND_HITL  

External side effects and compensations have truthful uncertain-execution and exclusive-consumption semantics across crash/restart/multi-worker operation. Cross-links: TOOLS idempotency remediation, Governance where operator reconciliation is required.

### PCM-CHECKPOINT-SCHEDULER-INTEGRITY — monotonic checkpoints and claimed scheduled resumes

**Priority:** P0/P1  
**Findings:** PCM-04, PCM-05  
**Owner:** RELIABILITY_FAILURE_AND_HITL  

Checkpoint writes cannot regress execution state; scheduled resumes are claimed exactly once at the scheduler/work-dispatch layer for multi-host topologies. Reuse canonical CAS/lease/worker primitives from Agent Distribution reference pattern.

### PCM-SCHEMA-EVOLUTION-INTEGRITY — fail-closed schema migration

**Priority:** P2  
**Findings:** PCM-07  
**Owner:** RELIABILITY_FAILURE_AND_HITL  

Store schema evolution distinguishes expected idempotent migration conditions from real persistence failures and fails closed on unknown schema errors.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `a786e3a2202b105f0d3a38afff8f79ea34255f05`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests and CI gates are supporting evidence, not standalone proof of multi-host production correctness.
- Remediation not performed in this task.
- `AGENT_DISTRIBUTION` used as reference evidence only — not modified.

## Operator acceptance

- **Date:** 2026-08-21
- **Accepted findings:** all 7 (`AUDIT-20260818-PERSISTENCE_CONCURRENCY_MULTIHOST-01` … `07`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
