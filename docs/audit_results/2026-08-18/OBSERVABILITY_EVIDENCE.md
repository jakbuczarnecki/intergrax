# OBSERVABILITY_EVIDENCE — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** OBSERVABILITY_EVIDENCE
- **Tier(s):** Tier-1 `intergrax/runtime/events/` · Tier-1 `intergrax/runtime/observability/`
- **audited_sha:** `f21a85e88a6204a7fc83f0a4c091bc69c549aaf9`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/OBSERVABILITY.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/OBSERVABILITY.md`
- **Scope in:**
  - Harness Observability Spine (HOS) canonical `RuntimeEvent` persistence and bus acceptance
  - `RuntimeEventPersistence` contract and SQLite / null durable providers
  - `RuntimeEventBus` store path and subscriber dispatch after persistence failure
  - EventId idempotency and canonical event identity equivalence
  - Unified Run Journal read model (`build_unified_run_journal`, `load_positioned_run_journal_through`)
  - Journal export (`journal_export`, `make_journal_export_runtime_plugin`) vs canonical export boundary
  - `ObservabilityExportEnvelope` / safe runtime event payload projection
  - Tenant identity on persistence scope vs event payload
  - Task-level ordering via `list_for_task()` vs run-local `ExecutionEventPosition`
  - Historical TRACE-1A/1B/1C, TRACE-ASOF-1/2, TRACE-BITEMP-1/3 **Done** delivery facts (positive control)
- **Scope out:**
  - remediation implementation
  - TRACE-BITEMP-2 provider acceptance closeout re-audit
  - OBS-VENDOR production hardening re-qualification
  - OECP code phases
  - full E + K + Valid Time + System Time public query API
  - second event bus / second persistence authority invention
  - silent runtime fixes in production source
- **Prior audit reference(s):** legacy observability audits under `docs/audit_results/legacy/` — historical only; Protocol v2 snapshot at pinned SHA supersedes for campaign register
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `d6c71def219a23f4a741fad73493416a66ed1adb`

## Executive summary

**Verdict: FAIL.** Four HIGH and two MEDIUM accepted findings show fail-open canonical evidence durability on persistence errors, EventId-only idempotency without canonical content equivalence, journal export bypass of the redacted-by-default `ObservabilityExportEnvelope` boundary via full `RuntimeEvent.model_dump()`, silently truncated full-run Unified Run Journal reads, divergent persistence routing tenant vs event tenant, and run-local position misused as task-global ordering. Positive controls: structural `TaskId`/`RunId`/`AttemptId`/`EventId` requirements remain; typed identity validators active; no legacy journal identity reconstruction from trace metadata; SQLite run position allocation is transactional; `(tenant, run, execution_position)` uniqueness holds; `load_positioned_run_journal_through()` is fail-closed for incomplete as-of prefixes; Unified Run Journal remains a read model not a second authority; export boundary is strong when used; external sinks remain destinations not semantic owners; docs remain conservative A4/I4/P2/E3; full E+K+bitemporal/OECP/OBS-VENDOR production hardening are not falsely claimed shipped. Residual defects are Protocol-v2 evidence-integrity gaps distinct from historical TRACE delivery — remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-01

- **Severity:** HIGH
- **Category:** EVIDENCE DURABILITY / FAIL-OPEN
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-EVIDENCE-DURABILITY-INTEGRITY
- **Claim falsified:** Canonical evidence durability policy is explicit; persistence acceptance is distinct from in-memory bus history and subscriber dispatch; incomplete canonical evidence cannot masquerade as complete.
- **Observation:** `RuntimeEventBus._store_event()` catches every `RuntimeEventPersistence.append()` exception, logs it, and continues. The event may already be added to in-memory bus history, and `publish()`/`record()` still dispatch subscribers after persistence failure. Therefore an execution transition can continue and trigger downstream observability handlers even though canonical execution history was not durably accepted.
- **Location:**
  - `intergrax/runtime/events/event_bus.py` — `_store_event()` exception swallowing; continued publish/record
  - `intergrax/runtime/events/persistence_contract.py` — `RuntimeEventPersistence.append()`
- **Reproduction:** Static inspection at `audited_sha`: trace emit/publish path through `_store_event()`; inject `append()` failure; observe bus history/subscriber dispatch proceeds without durable acceptance or explicit incomplete-evidence marking.
- **Impact:** Audit-grade runs may appear complete in memory and downstream observability while canonical execution history was never durably accepted — undermines evidence-required and audit-required modes.
- **Confidence:** CONFIRMED

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-02

- **Severity:** HIGH
- **Category:** IDEMPOTENCY / EVIDENCE IDENTITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-EVIDENCE-DURABILITY-INTEGRITY
- **Claim falsified:** EventId replay is idempotent only when canonical event identity and content match the originally accepted event; same EventId + different canonical event fails closed.
- **Observation:** `RuntimeEventPersistence` defines append idempotency by `event_id`. `SQLiteRuntimeEventStore` has global `UNIQUE(event_id)` and on duplicate `event_id` returns the previously stored event/position. It does not verify that the new append has the same tenant, task, run, attempt, event type, or canonical content/payload. `NullRuntimeEventPersistence` has the same conceptual event-id-only acceptance.
- **Location:**
  - `intergrax/runtime/events/persistence_contract.py` — idempotency contract
  - `intergrax/runtime/events/stores/sqlite_runtime_event_store.py` — duplicate `event_id` handling
  - `intergrax/runtime/events/stores/validating_runtime_event_store.py` — wrapper semantics
- **Reproduction:** Append event with `EventId` E1 and payload A; retry append with same E1 and different run/task/type/payload; observe silent idempotent return of first event rather than conflict error.
- **Impact:** Conflicting event reusing an existing `EventId` can silently alter or suppress execution evidence; breaks observational equivalence across durable providers.
- **Confidence:** CONFIRMED

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-03

- **Severity:** HIGH
- **Category:** SECURITY / DATA EXPORT / REDACTION BYPASS
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-EXPORT-CONTENT-INTEGRITY
- **Claim falsified:** Every external/log/vendor observability projection passes through one canonical content-safety/redaction policy; journal export does not bypass `ObservabilityExportEnvelope` by serializing arbitrary `RuntimeEvent.payload`.
- **Observation:** Canonical `ObservabilityExportEnvelope` is redacted-by-default. `runtime_event_export_source_from_event()` copies only a safe allowlist from `RuntimeEvent.payload` and the export boundary explicitly defines forbidden raw content fields. However `journal_export.serialize_runtime_event()` returns `event.model_dump(mode="json")` and `build_journal_export_snapshot` therefore contains the full `RuntimeEvent` payload. `make_journal_export_runtime_plugin()` logs the full snapshot via logger extra as `journal_export` and also logs derived OTLP JSON. Journal export is enabled by default unless `INTERGRAX_EXPORT_JOURNAL` disables it.
- **Location:**
  - `intergrax/runtime/observability/journal_export.py` — `serialize_runtime_event()`, snapshot build, runtime plugin
  - `intergrax/runtime/observability/export_boundary.py` — forbidden fields / safe boundary (bypassed by journal path)
  - `intergrax/runtime/observability/export_bridge.py` — safe projection contrast
- **Reproduction:** Enable default journal export; emit `RuntimeEvent` with sensitive payload keys (`prompt`, `content`, `query`, etc.); inspect logged `journal_export` extra and OTLP JSON for full payload escape.
- **Impact:** Default journal/log export can exfiltrate raw runtime payload content that the canonical export boundary explicitly forbids — undermines redaction-by-default posture and operator trust.
- **Confidence:** CONFIRMED

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-04

- **Severity:** HIGH
- **Category:** EVIDENCE COMPLETENESS / AUDIT INTEGRITY
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-JOURNAL-IDENTITY-INTEGRITY
- **Claim falsified:** A canonical full-run journal is either proven complete, explicitly paginated with continuation semantics, or explicitly marked/fails as truncated.
- **Observation:** `load_positioned_run_journal_through()` correctly proves prefix completeness by increasing the limit and failing closed when completeness cannot be established. But `build_unified_run_journal()` performs a single `list_positioned_for_run(..., limit=limit)` and returns the result without determining whether the returned list is complete or truncated. Default limit is commonly 2000, including journal export. A run with more events can therefore produce a silently truncated "Unified Run Journal" and an `event_count` that looks authoritative.
- **Location:**
  - `intergrax/runtime/events/unified_run_journal.py` — `build_unified_run_journal()`, contrast with `load_positioned_run_journal_through()`
  - `intergrax/runtime/observability/journal_export.py` — default limit usage
- **Reproduction:** Persist >2000 events for one run; call `build_unified_run_journal()` with default limit; observe truncated list and authoritative-looking count without truncation signal.
- **Impact:** Operators and downstream export surfaces may treat incomplete journals as complete execution history — conflicts with TRACE-1C strict-journal semantics and audit integrity.
- **Confidence:** CONFIRMED

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-05

- **Severity:** MEDIUM
- **Category:** TENANT IDENTITY / CONTRACT CONSISTENCY
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-JOURNAL-IDENTITY-INTEGRITY
- **Claim falsified:** Canonical persistence has one tenant truth; explicit persistence scope tenant and event tenant must match exactly or fail.
- **Observation:** `RuntimeEvent` carries optional `tenant_id`. `RuntimeEventPersistence.append` separately accepts `tenant_id`. `resolve_event_tenant_id()` uses explicit tenant first, then `event.tenant_id`, without validating equality when both exist. SQLite persists routing/index tenant from the explicit scope and serialized `RuntimeEvent` JSON containing its original `tenant_id`. Therefore a row can be indexed under tenant A but deserialize to an event whose `tenant_id` says B.
- **Location:**
  - `intergrax/runtime/events/runtime_event.py` — optional `tenant_id`
  - `intergrax/runtime/events/persistence_contract.py` — append scope tenant
  - `intergrax/runtime/events/stores/sqlite_runtime_event_store.py` — index vs serialized event tenant
- **Reproduction:** Call `append(tenant_id=A, event=RuntimeEvent(..., tenant_id=B))`; read back; observe index under A with deserialized event tenant B.
- **Impact:** Tenant-scoped queries and audit reconstructions can associate events with the wrong tenant identity — undermines isolation and provenance truthfulness.
- **Confidence:** CONFIRMED

### AUDIT-20260818-OBSERVABILITY_EVIDENCE-06

- **Severity:** MEDIUM
- **Category:** ORDERING CONTRACT DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** OBS-JOURNAL-IDENTITY-INTEGRITY
- **Claim falsified:** Run-local `ExecutionEventPosition` is not used as a task-global ordering coordinate; task query semantics are explicit and truthful.
- **Observation:** `RuntimeEventPersistence.list_for_task()` documents canonical execution order. `ExecutionEventPosition` is allocated per `(tenant_id, run_id)`. SQLite `list_for_task()` queries all events for a task but orders only by `execution_position`. Positions restart for every run, so task events spanning multiple runs have no canonical total order under that coordinate.
- **Location:**
  - `intergrax/runtime/events/persistence_contract.py` — `list_for_task()` contract
  - `intergrax/runtime/events/stores/sqlite_runtime_event_store.py` — task query ordering
- **Reproduction:** Persist events for same `TaskId` across two runs; call `list_for_task()`; observe order by run-local positions without run grouping or real task-level coordinate.
- **Impact:** Task-scoped history reads can misrepresent cross-run chronology — undermines audit reconstruction and any consumer assuming documented canonical execution order.
- **Confidence:** CONFIRMED

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| `RuntimeEvent` structurally requires `TaskId` / `RunId` / `AttemptId` / `EventId` | NOT falsified |
| Typed identity validators remain active | NOT falsified |
| No legacy journal identity reconstruction from trace metadata found | NOT falsified |
| SQLite run position allocation is transactional | NOT falsified |
| `(tenant, run, execution_position)` is unique | NOT falsified |
| `load_positioned_run_journal_through()` fail-closed for incomplete as-of prefixes and missing exact boundaries | NOT falsified |
| Unified Run Journal remains a read model, not a second persistence authority | NOT falsified |
| `ObservabilityExportEnvelope` is a strong redacted-by-default canonical boundary when used | NOT falsified |
| Safe runtime event payload projection is allowlist-based | NOT falsified |
| External sinks remain destinations rather than semantic owners | NOT falsified |
| Current docs remain conservative A4/I4/P2/E3 | NOT falsified |
| Full E+K+Valid Time+System Time, OECP, OBS-VENDOR production hardening not falsely claimed shipped | NOT falsified |
| Findings require hardening existing HOS, not creating another observability subsystem | NOT falsified — remediation targets existing spine |

## Provider / backend abstraction

| concern | canonical abstraction | provider boundary | composition owner | observed provider(s) | classification | evidence/finding |
|---------|-----------------------|-------------------|-------------------|----------------------|----------------|------------------|
| Runtime event durability | `RuntimeEventPersistence` | SQLite / null stores behind contract | `RuntimeEventBus` + host DI | SQLite, null | **IMPLEMENTATION DEFECT** (fail-open store path) | OBS-EVID-01 |
| Event append idempotency | `RuntimeEventPersistence.append` | store implementations | persistence contract | SQLite, null | **IMPLEMENTATION DEFECT** (identity equivalence) | OBS-EVID-02 |
| External/log export safety | `ObservabilityExportEnvelope` + export policy | vendor integrations as sinks | export plugins / journal export | journal export default-on | **SECURITY** (redaction bypass) | OBS-EVID-03 |
| Full-run journal completeness | Unified Run Journal read model | positioned list APIs | `unified_run_journal` | SQLite list with limit | **EVIDENCE COMPLETENESS** | OBS-EVID-04 |
| Tenant routing identity | persistence scope + event tenant | SQLite index columns | append resolver | SQLite | **CONTRACT CONSISTENCY** | OBS-EVID-05 |
| Task history ordering | `list_for_task()` | SQLite query | persistence contract | SQLite | **ORDERING CONTRACT** | OBS-EVID-06 |

## Historical TRACE delivery vs Protocol-v2 residual defects

Historical **TRACE-1A**, **TRACE-1B**, **TRACE-1B-HOS-FIX**, **TRACE-1C**, **TRACE-ASOF-1**, **TRACE-ASOF-2**, **TRACE-BITEMP-1**, and **TRACE-BITEMP-3** **Done / Closed** delivery facts remain valid — typed identity, strict journal identity removal, positioned as-of prefix authority, and K-only reconstruction were delivered as claimed. The six accepted Protocol-v2 findings document **residual evidence durability, export safety, journal completeness, tenant identity, and ordering semantics gaps** at `audited_sha` — they harden the existing HOS and read models; they do **not** reopen TRACE closeout rows or require a second event bus/store.

## Root-cause remediation grouping

### OBS-EVIDENCE-DURABILITY-INTEGRITY — canonical RuntimeEvent acceptance durability and EventId equivalence

**Findings:** `AUDIT-20260818-OBSERVABILITY_EVIDENCE-01`, `AUDIT-20260818-OBSERVABILITY_EVIDENCE-02`

Explicit evidence-required vs best-effort durability policy on the existing bus/store path; persistence acceptance distinct from in-memory history and subscriber dispatch; EventId replay validates full canonical event equivalence; conflicting EventId reuse fails closed across all durable providers.

### OBS-EXPORT-CONTENT-INTEGRITY — journal/log/vendor export subordinate to safe export boundary

**Findings:** `AUDIT-20260818-OBSERVABILITY_EVIDENCE-03`

All journal/log/vendor export passes through canonical redaction-safe `ObservabilityExportEnvelope` policy; no raw `RuntimeEvent.payload` escape via `journal_export`. Cross-link existing OBS-EXPORT boundary work — do not weaken the safe export boundary.

### OBS-JOURNAL-IDENTITY-INTEGRITY — journal completeness, tenant truth, ordering semantics

**Findings:** `AUDIT-20260818-OBSERVABILITY_EVIDENCE-04`, `AUDIT-20260818-OBSERVABILITY_EVIDENCE-05`, `AUDIT-20260818-OBSERVABILITY_EVIDENCE-06`

Reuse `load_positioned_run_journal_through()` completeness machinery for full-run journal; one tenant truth on append; explicit task ordering contract (grouped runs or real task coordinate). Cross-link TRACE-1C / TRACE-ASOF-1 positioned authority — do not build another journal authority.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `f21a85e88a6204a7fc83f0a4c091bc69c549aaf9`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification.
- Remediation not performed in this task.
- Historical TRACE **Done / Closed** plan rows remain valid delivery facts — not rewritten.

## Open questions / blocked items

- Finding 01: exact host-configurable durability mode naming (`evidence-required`, `audit-required`, explicit best-effort) — deferred to remediation design on existing HOS spine.
- Finding 06: preferred task ordering contract (grouped runs vs task-level position vs documented weaker semantics) — deferred to remediation design without second store.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-OBSERVABILITY_EVIDENCE-01` … `AUDIT-20260818-OBSERVABILITY_EVIDENCE-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.
