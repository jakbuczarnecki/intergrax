# PROD-Q-R1-R7-Q — GCA SQLite durable host bundle and partial-finalization recovery decision 01

| Field | Value |
|---|---|
| Status | **RATIFIED — SCOPED OPERATOR DECISION; IMPLEMENTATION NOT QUALIFIED** |
| Date | 2026-10-18 |
| Operator-approved source HEAD | `5470c376bc1404d6bbb3e7d833032ea9ccedcea4` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — **NOT CLOSED** |
| Canonical plan | `../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` |
| Checklist | `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` |
| Preserved decisions | Reflex ADR-01 A+B; ownership corrective ADR; F02 Option B; R2 OD-R2-01…04 |
| Production changes | **0 — architecture ratification only** |

## 1. Operator ratification and bounded target

Operator explicitly approved a **dedicated SQLite-backed GCA Host Store Bundle** as the first strict-production implementation of the existing four GCA persistence contracts and **correction of host partial-finalization semantics in the same parent**. No new generic persistence framework, provider invocation store, event store, orchestrator, registry, or parallel authority is authorized.

Exactly-one GCA semantic owner stays `GovernedExternalWorkOrchestrator`. `ProductionProcessComposition` remains the sanctioned process-lifetime composition owner through platform-defined typed ports. The existing `SQLiteProviderInvocationStore` (PIS) remains the **separate** intent/outcome authority; the new GCA backend must not reimplement PIS or infer snapshots from PIS/EBE.

Existing GCA protocols: `GovernedExecutionStore` (including `GovernedExternalWorkCreateFollowOnRecordV1`), `ProofReceiptStore`, `PolicyBundleArtifactStore`, `ContinuationStateStore`. The backend may be one SQLite DB with a coherent transaction scope or tightly coordinated SQLite-backed adapters but must prove exactly-one GCA persistence ownership, identity binding, atomicity/recovery and structural replaceability via these protocols. Do not expand semantic API surface or create an additional orchestration owner.

## 2. Mandatory correctness and failure-ordering

Current `applications/governed_contractor_application/host/orchestrator.py` around `put_result(ger)` → `_write_create_follow_on_record` catches failures and calls `put_state(EXECUTION_FAILED)` even when physical CREATE already occurred. **This is NOT acceptable** under F02 operator ratification (§11).

Implement within existing orchestrator/host-state authority a precise outcome model distinguishing:
- provider/physical-effect outcome (truth owned by integration + durable PIS);
- GCA host-finalization state, including **INCOMPLETE / RECOVERY REQUIRED**;
- availability of authoritative F02 record for follow-on.

An effected CREATE with failed GER/F02/finalization write must **never** be presented as ordinary retryable physical CREATE failure; never re-dispatch CREATE on normal re-entry. Reject accept/cancel while authoritative F02 is absent, corrupt, stale, tenant/principal mismatched or not committed. Specify crash windows (before provider dispatch; after provider success before GER; after GER before F02; after F02 before host state/event/receipt), authoritative state after each, allowable sanctioned recovery and explicit deny/error semantics.

If existing `GovernedExternalWorkHostState` or related contracts cannot truthfully express partial-finalization, introduce **only the minimal typed extension of that existing authority**, and require STOP — ARCHITECTURE DECISION REQUIRED for any new semantic owner, fallback pathway or broader architectural change. Do not put execution truth in observability or event payloads.

GER/F02/state and any necessary receipt/bundle/continuation updates must have explicitly analyzed write ordering and transactions or recoverable, fail-closed state transitions. SQLite transaction design must prevent optimistic success on partial commits. Treat corrupt/unreadable data as authoritative deny, not "missing → do CREATE".

## 3. Media durability and production admission

Implement explicit SQLite file ownership and lifecycle binding beside, but not inside, existing PIS in the approved production opener. Enforce startup rejection of absent/unqualified GCA bundle; no `InMemory*` or `FilesystemHostStore` fallback. Keep lab/demo paths separate.

Use production-conscious SQLite settings and proved semantics (journaling, synchronous writes, transactions, connection lifetime, concurrency/locking, busy/corruption handling, migrations/schema identity, close/reopen, file permissions where applicable). Never claim `is_durable=True` alone proves qualification; grant the marker only to an implementation whose actual media guarantees and recovery paths are tested. No fake provider or mock media may count as production restart evidence.

Preserve pluginability: all consumers use GCA protocols / typed bundle port; no SQLite dependence in Tier-2, contracts, orchestrator semantics or GCA business logic. Do not introduce cross-layer imports from shared to GCA implementations.

## 4. Mandatory qualification gates before parent acceptance

- SQLite GCA store round-trip and **close/reopen with a NEW backend instance** for every persisted semantic record.
- CREATE → committed GER/F02/state → restart → sanctioned accept/cancel; positive and negative tests.
- Fault injection at each listed crash window; PIS/provider physical outcome unchanged, no ordinary CREATE redispatch.
- F02 conflict, digest mismatch, tenant/principal mismatch, corrupt record and missing record → deny.
- Concurrent/reentrant invocations, idempotency and SQLite lock/error scenarios.
- Restart/migration compatibility, recovery from partially finalized transaction, failure of receipt/event/continuation persistence.
- Strict startup without GCA bundle or with InMemory/demo/unqualified store → deny.
- Bundle/backend replacement regression through the unchanged contracts.
- No new PIS, dispatch gate, shadow follow-on or registry authority.
- Relevant F02 G-F02-P01…P04 and G-F02-N01…N06 plus FRZ-REL-03/05, FRZ-PRD-08, FRZ-SEC-01/04, FRZ-OWN/BND/CTR/TYP/RPL/REG-02/03.
- Integrated strict HTTP → AC3 → orchestrator → PIS/provider and Package P(B) remain separately mandatory; do not claim them from store unit tests.

Plan tightly bounded foreground deterministic tests first; avoid broad tests, subprocess spawning, parallel Python/uv and expensive infra. Unexecuted gates = **NOT PROVEN**. Approval for this architecture **does not authorize resource-heavy test runs**.

## 5. Acceptance and stop rules

A child may reach at most **READY FOR AUDIT**, followed by independent exact-SHA GitHub audit; no autonomous CLOSE. Global `FRZ-*`, IMPL-01, DIM20, R7-Q and PROD-Q remain OPEN/BLOCKED until execution evidence. Cursor must not silently widen parent scope. Any missing authority contract, inability to reconcile physical outcome truth, or need for a new persistence framework triggers **STOP — ARCHITECTURE DECISION REQUIRED** with an exact blocker; do not fabricate success or introduce unsanctioned compatibility paths.

Operator ratification applies to the **bounded backend choice and host-finalization correction** only; a fully qualified backend, restart evidence or HTTP E2E have **not yet been demonstrated**.
