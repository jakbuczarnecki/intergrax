# PROD-Q-R1-R7-Q — Trusted history anchor mechanism — Decision 03 (Kubernetes / multi-instance)

| Field | Decision |
|---|---|
| Status | **RATIFIED ARCHITECTURAL TARGET / NOT IMPLEMENTED / DEPLOYMENT BLOCKED PENDING AUTHORITY & STORAGE DECISIONS** |
| Operator selection | 2026-10-18: Kubernetes, multiple replicas |
| Baseline | `e9b0e3b96969a22b095efc76bf2b13e333a0e970` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN |
| Prior | Legacy Compatibility Decision 01; History Provenance Authority Decision 02; Stable CREATE Identity Decision 01; GCA SQLite Recovery Decision 01; PIS ADR-03; R2 composition; F02 Option B |
| Production delta | **0** |

## 1. Deployment model and non-negotiable invariant

Target is Kubernetes with multiple potentially simultaneous application replicas, independent pod restarts, rescheduling, backups, persistent-volume restores and concurrent CREATE requests. **Pod-local SQLite files are not a shared durable admission authority.** A Kubernetes Lease, ConfigMap, annotation, pod UID, PVC name, filesystem path, schema version or an in-database marker is not by itself a trusted irreversible anti-rollback anchor or exactly-once physical-effect guarantee.

Never physically call a provider for CREATE unless (a) the current environment/history generation is proven by a trusted, independent continuity authority, (b) the dispatch has been admitted through **one logically global, linearizable provider-invocation intent authority** for all relevant replicas, (c) the authority is fenced against competing/stale writers and TOCTOU between validation and dispatch, and (d) unknown legacy history stays quarantined. This is a desired contract, not a claim that current SQLite PIS meets it.

## 2. Authority and recommended technology-neutral mechanism

**One existing semantic owner**: `ProductionControlPlaneGovernance` / production process lifecycle issues a narrowly typed `CreateHistoryCompatibilityVerdict` from an authorized, independent, durable **history continuity authority**; `GovernedProviderInvocationDispatchGate` is the sole physical CREATE enforcer. PIS remains the sole invocation intent/outcome owner; GCA remains the execution/GER/F02 owner. Do not add a second invocation ledger, gate, execution store or shadow registry.

The continuity anchor is a small, governance-owned record, *not* a list of provider invocations. Conceptual fields: `environment_id`, `history_scope_id`, `identity_epoch`, `generation`, `pis_history_id`, `pis_committed_watermark`, `gca_history_id`, `gca_committed_watermark`, `admission_mode` (FRESH_INITIALIZATION / AUTHORIZED_RECONCILIATION), `issuer_authority`, `proof_revision`, `status` (ACTIVE / QUARANTINED / REVOKED), `fencing_token`, `integrity_binding`. Types, serialization and source of authority require an implementable contract review; no implicit trust in operator-supplied booleans.

**Recommended backing semantics**: a strongly consistent, independently administered control-plane authority with linearizable compare-and-swap and durable monotonic revisions **whose disaster-recovery/backup domain cannot roll back with PIS/GCA**. A managed transactional database with independently protected append-only/monotonic head, or an already sanctioned equivalent, may qualify *only when its exact durability, rollback protection, privileges and recovery semantics are evidenced*. A Kubernetes API `resourceVersion` or Lease is not automatically sufficient: control-plane restore may roll back API data as well. A third SQLite file on the same volume/backup boundary is not sufficient.

The current control-plane implementation has **no proven such anchor**. This decision ratifies requirements and the Kubernetes topology, **not** a new database/service/provider, deployment, IAM permissions or production persistence mechanism. If a separate durable anchor/contract must be introduced, **STOP — ARCHITECTURE DECISION REQUIRED — ANCHOR BACKEND & TRUST DOMAIN** before code.

## 3. Fresh deployment and legacy upgrade

FRESH_INITIALIZATION can be issued only by privileged provisioning before the first eligible CREATE, based on authenticated creation of the bounded histories and their externally anchored identifiers; never from empty files discovered at pod startup. AUTHORIZED_RECONCILIATION requires human-authorized, separately evidenced complete reconciliation or quarantine of legacy effects, without new physical CREATE or heuristic `inv-*` to `gca-pinv:*` matching. Scope without complete proof remains QUARANTINED and physically denied. The default scope is process/environment when tenant completeness is not independently proven; no cross-tenant payload scan.

Restoring one history file, both histories, or the history-plus-local-proof from old snapshots must be rejected against the current external head. Restoring the external anchor itself from an old snapshot must not be considered safe; explicit cross-domain recovery/quarantine is required unless a separately protected non-rollback head survives. A signed but old token alone cannot prove freshness.

## 4. Continuous history binding, transactions and fencing

Static file hashes cannot be used as live generation assertions over mutable databases. Implementations must define trustworthy **monotonic committed watermarks / media incarnation IDs**, advancement semantics for *each* relevant durable intent/outcome and GER/F02 commit, partial commit/timeout uncertainty handling, and an externally observed contiguous continuity chain that never authorizes a missing history segment.

Because PIS and GCA are independent stores, no cross-database transaction is assumed. Define a fail-closed multi-step transition/recovery protocol; unknown intermediate states deny CREATE until reconciled. Stale replicas/old proof revisions must be fenced; revocation and generation change take effect before further physical dispatch. A one-time startup attestation or check-before-insert without a bound dispatch lease/transaction does not satisfy this guarantee.

For Kubernetes scale, establish a single authoritative concurrency domain for PIS first. Multiple isolated `SQLiteProviderInvocationStore` files (one per pod) can each return INSERTED for the same CREATE. SQLite on a shared network filesystem is not presumed to provide safe multi-writer global coordination. A single active owner with fenced dispatch or an approved shared transactional PIS implementation is an **additional, separately provable deployment prerequisite**, not silently delivered by this anchor.

Provider network partitions and timeouts mean that local admission + durable intent cannot prove that a non-idempotent external provider performed an effect exactly once. Distinguish **no automatic second dispatch under known intent** from true provider exactly-once; require provider-native idempotency/reconciliation guarantees before making any stronger claim.

## 5. Enforcement location and operation boundary

The existing `GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted` remains the only entry to external execution. Only `operation == "create_work"` consumes mandatory current history evidence; ensure enforcement both before provider `execute()` and across the validation-to-dispatch critical interval. Invalid, absent, expired, stale, unscoped, revoked or mismatch evidence → deny, zero physical provider CREATE, no silent migration, no fallback to default allow.

`submit_quote_acceptance` and `cancel_work` do not gain this CREATE history predicate; their own distributed idempotency/authority requirements must still be independently satisfied. All strict HTTP, AC3 and lifecycle entry paths must share the same guarded implementation. Preserve existing GER/F02 legacy ID reuse where authoritative.

## 6. Separate decisions and delivery gates

**Gate D03-A — trusted anchor backing and trust domain**: name the actual governance-owned control-plane persistent backing, monotonic CAS/immutable head semantics, independent recovery/backup boundary, issuer authorization and revocation, including threat model for etcd/cluster-admin or cloud-admin rollback. If absent, STOP — ANCHOR BACKEND & TRUST DOMAIN.

**Gate D03-B — globally coordinated PIS / dispatch**: prove that *all* replicas share one first-writer admission domain with fencing and durable intent (and that history continuity and dispatch cannot be separated by stale generation). If current per-process SQLite is used with multiple replicas, STOP — GLOBAL PIS/DISPATCH AUTHORITY.

**Gate D03-C — continuity write protocol**: prove monotonic media incarnation/watermarks, recovery for interrupted PIS↔GCA↔anchor transitions and nonbypassable CREATE-only enforcement. If missing, STOP — CONTINUITY PROTOCOL REQUIRED.

No gate authorizes adding a new production store, Kubernetes resource, control-plane service, provider, distributed coordinator or migrations. Each missing capability requires a separately approved minimal design and implementation plan. Avoid serial broad inventories: produce exact owner, capability and testable contract needed.

## 7. Qualification matrix (NOT RUN)

Behavioral proof required before strict production admission: (1) two replicas racing identical CREATE → one globally admitted physical attempt; (2) pod crash after intent / before network effect and after provider effect / before outcome → no unsafe redispatch; (3) restore PIS only, GCA only, both, and both + old proof → deny; (4) anchor rollback/failover or stale replica → deny; (5) loss/restore of cluster control-plane data → deny absent surviving anti-rollback head; (6) partially committed continuity advancement → deny and controlled recovery; (7) new fresh deployment with authorized provisioning → allow; (8) orphaned legacy PIS-only → deny; (9) authorized reconciliation/quarantine → scoped outcome; (10) accept/cancel unchanged and separately protected; (11) strict HTTP/AC3 cannot bypass gate; (12) tenant/environment isolation; (13) old leader fencing on network partition. Distinguish simulations from actual multi-process/Kubernetes tests.

`pytest = 0`, `uv run = 0` until explicit authorization. Q1/Q2 scoped PASS reports do not prove multi-instance behavior, real process restarts, or this anchor.

## 8. Status

Decision-03 **RATIFIED FOR TARGET REQUIREMENTS**. Deployment backend/anchor trust domain, global PIS/dispatch, and cross-store continuity protocol **NOT RATIFIED, NOT IMPLEMENTED, NOT QUALIFIED**. IMPL-01, R7-Q, PROD-Q and FRZ-* remain OPEN / BLOCKED. No claim of production readiness.
