# PROD-Q-R1-R7-Q — D03-A Trusted Anchor Backend & Trust Domain Proposal

> **PROPOSAL — NOT RATIFIED — NO IMPLEMENTATION AUTHORIZED**

| Field | Value |
|---|---|
| Task | `PROD-Q-R1-R7-Q-D03-A-TRUSTED-ANCHOR-BACKEND-TRUST-DOMAIN` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` |
| Branch | `development` |
| Required and inspected START_HEAD | `d7adda0d24230470e5e6a8d9ca4f51876f30b098` |
| Origin at inspection | `https://github.com/jakbuczarnecki/intergrax.git` |
| Baseline relation | local `HEAD == origin/development == START_HEAD`; clean tree before this document |
| Decision-03 authority | Kubernetes / multi-instance target requirements ratified; backend, global PIS and continuity protocol not ratified |
| Production delta | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |
| Maximum status | **READY FOR AUDIT — ARCHITECTURE DECISION PENDING OPERATOR RATIFICATION** |

This proposal performs the D03-A mechanism selection only. It does not authorize a
provider, database, Kubernetes resource, credential, migration, runtime
configuration, D03-B global PIS implementation or D03-C continuity protocol.

## 1. Decision summary

### 1.1 Preferred mechanism

Select an **External Linearizable Continuity Register with an Independent
Immutable Recovery Witness** (`ELCR-IIRW`) as the required infrastructure class.
It is one continuity authority composed of:

1. a managed, consensus-backed, strongly consistent transactional register for
   the current head, transition log, linearizable reads and compare-and-swap; and
2. a separately administered immutable recovery witness that seals every
   committed head revision outside the primary register's backup/restore and
   administrator domain.

The primary register and witness form one security mechanism; the witness is not
a second semantic history owner. The register owns the authoritative typed
continuity state. The witness only proves that an older register image is not
current and supplies a recovery high-water mark.

No repository or deployment evidence identifies an already deployed service
satisfying this class. A concrete managed service pair, accounts/projects,
regions, retention policy, service identities, endpoints and credentials remain
operator-controlled selections. The proposal therefore recommends the mechanism
class, not a named or approved technology.

### 1.2 Rejected mechanism classes

- The existing Agent Distribution SQLite backing has useful transactional CAS
  precedents but is single-host/multi-process storage, can be rolled back with
  its database image, has a different semantic owner and is not a shared
  Kubernetes continuity authority.
- Kubernetes Lease, ConfigMap, CRD status or another API object is linearizable
  only relative to the current etcd history. Cluster/etcd restore can roll the
  object and its `resourceVersion` back with the control plane.
- A third SQLite file, including one on the same persistent volume or backup
  policy as PIS/GCA, cannot detect rollback of the complete local history set.

## 2. Canonical baseline and current roadmap position

The following current controls were inspected at the exact baseline:

- [Platform Enterprise Completion Roadmap](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md)
- [Platform Enterprise Freeze Acceptance Checklist](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- [Enterprise Continuous Architecture Assurance Protocol](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md)
- [Trusted History Anchor Mechanism Decision-03](PROD_Q_R1_R7_Q_TRUSTED_HISTORY_ANCHOR_MECHANISM_DECISION_03.md)
- [History Provenance Authority Decision-02](PROD_Q_R1_R7_Q_LEGACY_HISTORY_PROVENANCE_AUTHORITY_DECISION_02.md)
- [Legacy CREATE Compatibility Decision-01](PROD_Q_R1_R7_Q_LEGACY_CREATE_ADMISSION_COMPATIBILITY_DECISION_01.md)
- [Stable CREATE Invocation Identity Decision-01](PROD_Q_R1_R7_Q_STABLE_CREATE_INVOCATION_IDENTITY_DECISION_01.md)
- [GCA SQLite Durable Recovery Decision-01](PROD_Q_R1_R7_Q_GCA_SQLITE_DURABLE_RECOVERY_DECISION_01.md)
- [F02 Option B](PROD_Q_R1_R7_Q_F02_DURABLE_FOLLOW_ON_SNAPSHOT_DECISION.md)
- [R2 Strict Composition Decision](PROD_Q_R1_R7_Q_REFLEX_EXTERNAL_WORK_STRICT_COMPOSITION_AUTHORITY_DECISION_R2.md)
- [PIS ADR-03](PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md)
- `docs/project/technical/guides/SYSTEM_INVARIANTS.md`
- [Vendor/Backend Neutrality Evidence Specification](VENDOR_BACKEND_NEUTRALITY_AND_REPLACEABILITY_EVIDENCE.md)

The roadmap records Decision-03 as **RATIFIED TARGET CONTRACT ONLY — NO BACKEND
OR GLOBAL PIS AUTHORIZED** and explicitly identifies Gate D03-A for the anchor
trust domain. It also keeps `IMPL-01`, `R7-Q`, `PROD-Q` and global `FRZ-*` open.
This proposal is therefore aligned with, and does not advance beyond, the
canonical roadmap.

## 3. Current authority and dependency graph

### 3.1 Existing production graph

```text
ProductionControlPlaneGovernance
  ├─ trusted RequestIdentity
  ├─ ControlPlaneMutationAuthorizationBoundary
  └─ ApplicationEnvironmentTenantResolver
       │
       v
open_production_process_lifecycle / open_strict_production_serving_session
  ├─ Agent Distribution SQLite bundle
  │    └─ revision/serving CAS for agent activation (not history continuity)
  ├─ SQLiteProviderInvocationStore (PIS; separate file/path)
  ├─ SQLite GCA host store bundle (separate file/path)
  └─ ProductionProcessComposition
       │
       v
strict HTTP / revision-bound AC3
  → GovernedSanctionedExternalWorkOperationPort
  → GovernedExternalWorkOrchestrator
      ├─ GCA execution/GER/F02/state owner
      └─ ExternalWorkAdapter._tracked_execute
           → GovernedProviderInvocationDispatchGate
              → ProviderInvocationStore.insert_invocation_if_absent
              → provider execute callback
              → ProviderInvocationStore outcome persistence
```

The graph has no `HistoryContinuityAuthority`,
`CreateHistoryCompatibilityVerdict`, history incarnation, identity epoch or
continuity-head implementation. `ProductionControlPlaneGovernance` is an
immutable wiring record; it has no durable monotonic backing.

### 3.2 Closed-world owner, contract and caller inventory

| Concern | Existing owner / contract / caller | Current persistence or behavior | D03-A finding |
|---|---|---|---|
| Provider intent/outcome | `ProviderInvocationStore`; `ProviderInvocation`; `ProviderInvocationOutcome` | Provider-neutral intent/outcome authority keyed by `invocation_id` | Remains exactly-one PIS owner; not an anchor |
| Atomic invocation admission | `ProviderInvocationIntentWriteResult`; `admit_provider_invocation_intent` | `INSERTED` versus `EQUIVALENT_NOOP` | Correct local semantic fact, but current backing is not a global Kubernetes first-writer domain |
| PIS backend | `SQLiteProviderInvocationStore` | Process-local SQLite, WAL, `synchronous=FULL`, schema version and uncertain-write marker | Durable restart evidence only; no history incarnation/watermark or anti-rollback |
| Stable CREATE identity | `derive_governed_external_work_create_provider_invocation_id` | Stable `gca-pinv:*` derived from tenant, principal, execution, provider and idempotency key | Prevents ordinary same-history duplicate admission; cannot detect restored history |
| Physical dispatch | `GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted` | Sole gate; executes only after atomic PIS `INSERTED` | Must remain sole execution gate |
| GCA execution truth | `GovernedExternalWorkOrchestrator`; GCA store protocols; F02 Option B | Separate SQLite GCA bundle with GER, state, event, follow-on, receipt, bundle and continuation tables | Exactly-one GCA owner; no anchor/history incarnation/watermark |
| Strict composition | `ProductionProcessComposition`; `open_production_process_lifecycle`; strict GCA opener | Opens and retains process-owned Agent Distribution, PIS, continuation and GCA resources | Sanctioned future composition locus, not proof of shared/independent storage |
| Governance | `ProductionControlPlaneGovernance` | Principal, mutation authorization and environment tenant resolver | Semantic owner precedent only; no continuity persistence |
| Existing durable control plane | Agent Distribution store protocols and `SqliteAgentDistributionStoreBundle` | Transactional CAS for serving pointers/revisions in one SQLite DB | CAS precedent, but wrong concern, wrong failure domain and rollbackable media |
| Attestation | Package/runtime digest and signature contracts | Integrity/authenticity evidence | A valid signature over stale state does not prove currentness |
| Strict serving storage declaration | `StrictProductionProcessStorage` | Four configured filesystem paths, including Agent Distribution, PIS and GCA SQLite | No evidenced shared Kubernetes or independent anti-rollback backing |

### 3.3 Proposed dependency graph

```text
Independent security / recovery trust domain
  └─ Immutable Recovery Witness (append-only revision seals; no runtime writes)
            ^
            │ verify/seal
Managed external linearizable continuity register
  └─ one typed head per EnvironmentHistoryScopeIdentity
            ^
            │ platform-defined contract only
Production process lifecycle under ProductionControlPlaneGovernance
  ├─ authorized provisioning/reconciliation service
  └─ runtime continuity client (read + narrowly scoped compare-and-advance)
            │
            v
GovernedProviderInvocationDispatchGate (unchanged sole execution gate)
  └─ CREATE-only D03-C guarded transition
       ├─ D03-B global PIS first-writer admission
       ├─ current PIS/GCA incarnation + watermark binding
       └─ provider execute callback
```

The anchor does not store provider invocations, provider outcomes, GERs, F02
records or execution payloads.

## 4. Root cause

Current safety is relative to the histories visible to a process:

1. stable CREATE identity derives the same `gca-pinv:*` for the same authorized
   logical effect;
2. the gate dispatches only when the PIS atomically reports `INSERTED`; and
3. the orchestrator rejects ordinary redispatch when current GCA state/GER/F02
   proves a prior CREATE.

If PIS is restored before its intent and GCA is restored before its GER/F02
correlation, both local checks can truthfully observe absence. The same stable
identity can then be newly inserted into the restored PIS and appear eligible.
Restoring only one history creates other unsafe or unresolved combinations.
Schema versions, integrity checks, file timestamps, signatures and local
uncertainty markers establish structure or integrity, not that the media is the
latest authorized history.

The missing fact is an authoritative, independently recoverable, monotonic head
binding the current environment scope, identity epoch, PIS incarnation/watermark
and GCA incarnation/watermark. It must remain current across replicas and outside
the rollback domain of both histories.

## 5. Explicit threat model

### 5.1 Protected assets and required safety

- No physical `create_work` after undetected PIS/GCA rollback, replacement or
  incompatible restore.
- No stale replica or restored cluster control plane can mint current evidence.
- No PIS/GCA backup operator can make old histories current by restoring media.
- No history proof crosses environment or tenant scope.
- Missing, stale, unverifiable or unavailable evidence denies CREATE.

### 5.2 Adversary and failure capabilities in scope

- Restore, copy, replace, truncate or independently roll back PIS or GCA.
- Restore both PIS and GCA, including an old pod-local proof.
- Restart/reschedule pods and run multiple replicas concurrently.
- Partition replicas from the anchor or Kubernetes control plane.
- Restore etcd/Kubernetes API state, including old Lease/ConfigMap/CRD objects.
- Reuse a stale leader, workload token, fence or signed proof.
- Crash or time out at any point in compare-and-advance.
- Compromise the PIS/GCA storage or backup operator.
- Compromise the primary anchor operator/account without compromising the
  independent witness domain.
- Perform disaster recovery from an old primary-register backup.

### 5.3 Required independent trust assumptions

- The primary register supplies linearizable reads, serializable transactions
  and conditional writes for all replicas in the supported deployment.
- The witness supplies a strongly consistent highest sealed revision and
  immutable append semantics; its production namespace cannot be restored
  in-place to a lower revision.
- Primary-register administration, witness administration and PIS/GCA
  backup/restore administration are non-overlapping roles with separately
  controlled break-glass credentials.
- Workload identity is issued by an external IAM trust root and grants runtime
  replicas only scoped read/advance rights, never initialize, reconcile, revoke,
  retention-change or restore rights.
- The authority namespace and issuer trust roots cannot be selected solely by
  rollbackable pod configuration.

If the same cloud super-administrator can silently restore or impersonate both
the primary register and witness, the proposal does not claim protection. The
operator must either establish independent accounts/providers and dual control
or reject this candidate. Suspected compromise of either domain forces
quarantine; it never permits continued CREATE.

### 5.4 Explicitly out of scope

- Provider-side exactly-once physical effects.
- Automatic replay or reconciliation of ambiguous external effects.
- D03-B selection/implementation of a global PIS.
- D03-C cross-store write ordering and dispatch protocol implementation.
- Protection after simultaneous compromise of all independent trust roots.

## 6. Preferred ELCR-IIRW mechanism

### 6.1 Mandatory primary-register properties

The selected managed backing must demonstrate:

- linearizable current-head reads and compare-and-swap;
- serializable atomic append of a transition and mutation of its scoped head;
- durable monotonic `ContinuityRevision` and `FenceEpoch`;
- one globally reachable authority namespace for all replicas of an environment;
- explicit tenant/environment partition keys and authorization;
- idempotent command identity for uncertain client outcomes;
- no eventual-consistency or client-side last-write-wins path;
- fail-closed unavailable/read-error behavior;
- audit records for initialization, transitions, quarantine, revocation and
  recovery;
- restore into a new authority incarnation, never silent in-place replacement of
  the production head with a lower revision.

### 6.2 Mandatory independent-witness properties

The witness must:

- be outside the PIS, GCA, Kubernetes/etcd and primary-register backup domains;
- use a separately administered production namespace and retention policy;
- conditionally append exactly the next scoped revision and transition digest;
- expose the strongly consistent highest sealed revision and receipt;
- prohibit overwrite, deletion and restore-to-lower-head for the retention
  period, including by normal platform and primary backup operators;
- record all retention, break-glass and recovery actions in a separately
  retained audit trail;
- deny or quarantine when its latest state cannot be proven after disaster
  recovery.

An ordinary log sink, signature service or eventually consistent object listing
does not satisfy these properties.

### 6.3 Authority roles

| Principal role | Permitted operations | Explicitly forbidden |
|---|---|---|
| `AnchorProvisioner` | First initialization with approved provisioning evidence | Runtime advance, PIS/GCA restore, witness retention changes |
| `AnchorTransitionWriter` | Scoped compare-and-advance and fence/session operations required by future D03-C | Initialize, reconcile, revoke, restore, cross-scope access |
| `AnchorReconciler` | Dual-controlled authorized reconciliation; may create a new identity epoch and authority incarnation | Physical provider dispatch; rewriting prior revisions |
| `AnchorSecurityRevoker` | Monotonic quarantine/revoke | Reactivation, provider dispatch, PIS/GCA mutation |
| `AnchorRuntimeReader` | Linearizable read and proof verification for its exact environment/history scope | Any mutation |
| `AnchorAuditReader` | Read immutable transition/audit evidence | Runtime admission or mutation |
| `WitnessWriter` | Seal primary transition digests through the authority service only | PIS/GCA or primary restore |
| PIS/GCA backup operator | Backup/restore those histories under approved runbook | Any anchor/witness issue, update, revoke or retention permission |

Human approval is represented by typed authorization evidence issued through the
existing governance boundary. It is never a mutable `allow=true` field.

## 7. Typed continuity contract proposal

These are conceptual platform-defined types for later ratification. They do not
authorize source files or implementation.

### 7.1 Identities and scalar types

| Type | Required semantics |
|---|---|
| `EnvironmentHistoryScopeIdentity` | Frozen identity containing `environment_id`, `history_scope_id`, optional proven `tenant_id`, and an authority namespace; all fields non-empty and domain typed |
| `HistoryIncarnationIdentity` | Frozen pair of `PISHistoryIncarnationId` and `GCAHistoryIncarnationId`; generated only during authorized media provisioning/reconciliation, never inferred from path/schema/emptiness |
| `IdentityEpoch` | Monotonic scoped epoch separating identity derivation regimes and authorized reconciliation generations |
| `ContinuityRevision` | Unsigned monotonic scoped revision; no wrap, decrement or reuse |
| `HistoryWatermark` | Typed contiguous committed position for its named store; PIS and GCA watermark types are not interchangeable |
| `FenceEpoch` | Monotonic authority-issued writer/dispatch fence |
| `TransitionId` | Globally unique idempotency identity for one requested head transition |
| `AnchorAuthorityIncarnationId` | Identifies the primary authority deployment/DR generation |
| `WitnessReceipt` | Typed witness namespace, revision, transition digest, previous digest, service incarnation and immutable receipt reference |

### 7.2 Enumerations

- `HistoryAdmissionMode = FRESH_INITIALIZATION | AUTHORIZED_RECONCILIATION`
- `HistoryContinuityStatus = ACTIVE | QUARANTINED | REVOKED`
- `HistoryTransitionState = PREPARED | WITNESS_SEALED | COMMITTED`
- `HistoryCompatibilityOutcome = ELIGIBLE | DENY`
- `HistoryCompatibilityDenyReason` includes:
  `MISSING_HEAD`, `UNPROVEN_INITIALIZATION`, `SCOPE_MISMATCH`,
  `TENANT_MISMATCH`, `ENVIRONMENT_MISMATCH`,
  `PIS_INCARNATION_MISMATCH`, `GCA_INCARNATION_MISMATCH`,
  `IDENTITY_EPOCH_STALE`, `CONTINUITY_REVISION_STALE`,
  `HEAD_QUARANTINED`, `HEAD_REVOKED`, `ISSUER_UNAUTHORIZED`,
  `WITNESS_MISSING`, `WITNESS_MISMATCH`, `FENCE_STALE`,
  `FRESHNESS_EXPIRED`, `TRANSITION_PENDING`, `LEGACY_UNRESOLVED`,
  `AUTHORITY_UNAVAILABLE`, `HISTORY_UNREADABLE` and `RECOVERY_REQUIRED`.

Unknown enum values fail closed. No string fallback or reflection-based semantic
dispatch is permitted.

### 7.3 Records

`HistoryContinuityHeadV1` contains:

- schema version and exact `EnvironmentHistoryScopeIdentity`;
- `AnchorAuthorityIncarnationId`;
- `HistoryIncarnationIdentity`;
- `IdentityEpoch`;
- current PIS and GCA committed watermarks;
- `HistoryAdmissionMode`;
- `HistoryContinuityStatus`;
- `ContinuityRevision` and `FenceEpoch`;
- issuer identity, governance authorization reference and authorization policy
  revision;
- `TransitionId`, prior-head digest and current-head digest;
- transition state and `WitnessReceipt`;
- authoritative commit timestamp and optional quarantine/revocation reason.

`CreateHistoryCompatibilityVerdictV1` contains:

- exact scope, history incarnations, identity epoch, revision and fence observed
  by a linearizable read;
- `ELIGIBLE` or `DENY` plus explicit typed deny reason;
- proof issuer/reader identity, issued-at and short `not_after`;
- witness receipt and head digest;
- the stable CREATE invocation identity to which a future one-use admission is
  bound.

It is a read-only verdict for the sole dispatch gate, not a reusable bearer
permission and not an invocation record.

`HistoryContinuityTransitionRequestV1` contains:

- exact scope and expected head revision/digest/fence;
- transition id and requested next PIS/GCA incarnations/watermarks;
- admission mode and target status;
- typed issuer and governance authorization evidence;
- reason code and exact prior witness receipt.

### 7.4 Contract operations and preconditions

| Operation | Preconditions | Result / failure |
|---|---|---|
| `read_current_authoritative_head(scope)` | Authorized exact-scope reader; linearizable primary and witness reads | Verified head, or typed DENY; cache-only success is forbidden |
| `initialize_fresh_history(request)` | Head absent; `AnchorProvisioner`; authenticated creation of both histories before any CREATE; witness revision absent | Revision 1 `ACTIVE/FRESH_INITIALIZATION`, or fail closed |
| `compare_and_advance(request)` | Exact expected revision, digest, fence, incarnations and witness receipt; authorized writer; current status allows transition | One serialized next revision, conflict/stale fence, or uncertain result requiring read-back |
| `quarantine(scope, expected_head, reason)` | Authorized reconciler/security principal; monotonic CAS | New `QUARANTINED` revision; never deletion or toggle |
| `revoke(scope, expected_head, reason)` | `AnchorSecurityRevoker`; monotonic CAS | Terminal `REVOKED` revision |
| `authorized_reconciliation(request)` | Dual-controlled evidence; complete affected-scope reconciliation; no provider dispatch; new identity epoch when identity/history changed | New revision in `AUTHORIZED_RECONCILIATION`, otherwise quarantine |
| `resolve_transition(transition_id)` | Exact scope and idempotency identity | Returns committed, pending or absent; absent/unknown never implies allow |
| `recover_authority(new_incarnation, witness_head, authorization)` | Primary unavailable/restore; witness latest proven; dual control | Reconstructed/quarantined authority with a new incarnation; no silent activation |

There is no `set_current`, `force_allow`, decrement, delete-head or restore-head
operation.

## 8. Monotonicity and anti-rollback

### 8.1 Authoritative current revision

A head is current only when all are true:

1. the primary register returns it through a linearizable read;
2. the transition is `COMMITTED`;
3. its revision, transition digest and predecessor match the highest immutable
   witness receipt;
4. its authority incarnation and scope match the configured external trust root;
5. status is `ACTIVE`, proof freshness is valid and the fence is current; and
6. current PIS/GCA incarnations and watermarks match the head under the future
   D03-C protocol.

No individual signature, database schema marker or locally cached head is
sufficient.

### 8.2 Register/witness transition protocol

The anchor's own anti-rollback transition is:

1. Primary serializable CAS appends revision `n` as `PREPARED` and blocks CREATE
   for the affected scope.
2. The witness conditionally seals `(scope, n, transition_digest,
   prior_witness_digest)` only when `n-1` is its current highest revision.
3. Primary serializable CAS attaches the receipt and marks revision `n`
   `COMMITTED`.
4. Readers require exact primary/witness equality. Any pending or mismatched
   state denies CREATE.

This is intentionally fail-closed without a distributed transaction:

- crash before step 1: no transition;
- crash after step 1: `PREPARED` denies; the same `TransitionId` may resume;
- witness committed but primary finalize timed out: witness is ahead, so readers
  deny until idempotent finalize/recovery proves the transition;
- primary claims a revision not sealed by the witness: deny and quarantine;
- outcome unknown: read by `TransitionId`; never retry with a new transition id
  or infer success.

### 8.3 Restore cases

- **PIS rollback only:** incarnation/watermark mismatch with the external head;
  deny.
- **GCA rollback only:** incarnation/watermark mismatch; deny.
- **Both histories rolled back:** both may agree with each other but disagree
  with the external head/witness; deny.
- **Histories plus local proof rolled back:** proof revision/fence is below the
  external highest witness revision; deny.
- **Primary anchor restored to an old snapshot:** the immutable witness is ahead;
  deny and enter recovery quarantine.
- **Witness disaster:** no CREATE until the latest witness revision is proved.
  Restoring a lower witness head is not an allowed recovery mode.
- **Whole Kubernetes control-plane rollback:** neither primary nor witness state
  is stored in etcd. Restored workload identity has no initialize/reconcile/
  revoke authority; stale fences and revisions fail.

An old snapshot cannot silently become authoritative because restore-in-place to
the production authority namespace is forbidden, a restored primary has a new
authority incarnation, and activation requires equality with the independently
retained highest witness revision. If the selected infrastructure cannot enforce
those facts, it does not qualify.

### 8.4 Concurrency and fencing

- Concurrent primary transitions use one linearizable expected-revision CAS;
  exactly one can prepare revision `n`.
- Every runtime writer and future dispatch reservation carries an authority
  issued `FenceEpoch`; an old leader cannot advance or authorize work after a
  newer fence exists.
- Positive CREATE evidence is never a long-lived cached boolean. D03-C must bind
  a short-lived, one-use verdict/reservation to the exact stable invocation,
  revision, fence and PIS/GCA state.
- Quarantine/revocation is effective for new CREATE when its committed revision
  is visible. A provider call already in flight is external-effect uncertainty,
  not proof that revocation canceled it.

## 9. Kubernetes deployment contract

### 9.1 Replica identity and access

- Each replica uses workload identity, not a shared static credential.
- Authorization binds service, cluster/deployment identity, environment and
  history scope.
- Runtime replicas may read and perform only future D03-C scoped transitions.
- Provisioning, reconciliation, revocation, retention and restore use separate
  identities unavailable to pods.
- Every replica reaches the same external authority/witness namespace for the
  environment; pod-local authority caches cannot authorize CREATE.

### 9.2 Restart, eviction and partitions

- Pod eviction, crash or rescheduling loses no authority state.
- A new replica performs linearizable authority/witness validation before any
  CREATE eligibility.
- Network partition, DNS ambiguity, TLS/issuer mismatch, expired workload
  identity, primary outage or witness outage denies CREATE.
- Kubernetes API/control-plane unavailability does not permit fallback to a
  Lease, cached ConfigMap, local file or last-known positive verdict.

### 9.3 Revocation and stale readers

- No positive verdict may outlive its `not_after`, revision or fence.
- The future D03-C critical interval must revalidate/consume the one-use
  reservation immediately before physical dispatch.
- Stale reads, follower reads without linearizability, disconnected reads and
  client-clock-only freshness are forbidden for CREATE admission.
- Required revocation latency and maximum verdict lifetime are operator-selected
  SLOs and must be tested; until selected they are **NOT PROVEN**.

### 9.4 Disaster recovery and audit

- Primary recovery creates a new authority incarnation in quarantine.
- The recovery service reconstructs from the highest witness-sealed transition
  chain and independently retained audit evidence.
- Cutover to the recovered incarnation requires dual-controlled authorization
  and a new witness-sealed revision.
- Provisioning, reconciliation, revocation, IAM, retention and restore actions
  must be attributable to principal, policy revision, request id and exact
  artifact/configuration SHA without secrets in evidence.

## 10. CREATE admission integration boundary

`GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted` remains
the sole physical provider gate. The later implementation must extend or compose
that existing typed gate; it must not add a peer gate in HTTP, AC3, the
orchestrator, the adapter or a storage adapter.

Only `ProviderInvocation.operation == "create_work"` receives the history
continuity predicate. `submit_quote_acceptance` and `cancel_work` retain their
existing separate admission and F02 semantics. They must not be blocked by a
CREATE-only requirement or weakened by its introduction.

The proposed producer is a governance-owned continuity service composed by
`open_production_process_lifecycle` under `ProductionControlPlaneGovernance`.
The consumer is the existing dispatch gate. The platform boundary must be a
Tier-0 typed contract; provider-specific and storage-specific types must remain
inside the eventual adapter/composition leaf.

### 10.1 Mandatory D03-C TOCTOU obligation

A linearizable read followed by ordinary PIS insert followed by `execute()` is
not sufficient. D03-C must define one gate-owned, fail-closed state machine that:

1. obtains a one-use anchor reservation bound to scope, current head/fence,
   stable CREATE invocation identity and expected PIS/GCA states;
2. performs D03-B global first-writer PIS admission under that binding;
3. advances/finalizes the anchor to include the admitted PIS state before
   physical dispatch eligibility;
4. revalidates or consumes the reservation at the last possible point before
   `execute()`; and
5. treats every partial commit, timeout or unknown outcome as deny/recovery
   required.

The provider network call cannot be made transactionally atomic with local
stores. D03-C must preserve external-effect uncertainty and must not claim
provider exactly-once behavior.

## 11. D03-B and D03-C boundaries

### 11.1 D03-B — independent mandatory blocker

Current `SQLiteProviderInvocationStore` explicitly describes itself as
process-local. `StrictProductionProcessStorage` supplies a filesystem path per
process, and no inspected composition proves one globally shared linearizable
first-writer domain across Kubernetes replicas.

Therefore:

- two pod-local PIS files are forbidden;
- SQLite on a shared network filesystem is not presumed safe;
- a distributed lock is not an approved substitute for globally durable PIS
  atomic admission;
- D03-A does not make local PIS global; and
- multi-replica CREATE remains blocked until D03-B is separately selected,
  ratified, implemented and qualified.

### 11.2 D03-C — independent mandatory blocker

Current PIS and GCA schemas expose schema versions and local uncertainty
handling, but no externally bound media incarnation identities or monotonic
committed watermarks. D03-C must define those types, write ordering, anchor
reservation/finalization, crash recovery and non-bypassable CREATE-only gate
integration.

No D03-C design or implementation is authorized by this proposal.

## 12. Mechanism comparison

| Evaluation | Existing Agent Distribution SQLite control-plane backing | Preferred ELCR-IIRW | Kubernetes-native API object / Lease / CRD |
|---|---|---|---|
| Atomic CAS and linearizability | Transactional CAS on one SQLite file; single-host/multi-process claim only | Required linearizable CAS and serializable transactions across replicas | API writes may be linearizable while current etcd is healthy |
| Durable monotonic head | Durable relative to current SQLite media | Monotonic primary head plus independently sealed highest witness revision | `resourceVersion`/object generation is relative to restorable etcd history |
| Independent from PIS/GCA backup domain | Not evidenced; path-based local production storage | Mandatory separate primary and witness recovery domains | Separate from PVCs, but inside cluster control-plane recovery domain |
| Anti-rollback after recovery | No; old DB image can restore old pointer/revision | Yes within stated split-domain trust model; mismatch forces quarantine | No; etcd snapshot restore can restore old object and lease |
| Replica fencing | Optimistic revision guards, not Kubernetes-wide history fencing | Monotonic `FenceEpoch`, one-use reservations and CAS | Lease holder/resource version can become stale after restore |
| Privilege separation | No D03-A roles; existing semantics unrelated | Explicit runtime/provision/reconcile/revoke/witness/backup role split | Cluster-admin commonly controls object, RBAC and restore |
| Crash recovery | SQLite transaction recovery; no external rollback witness | Idempotent transition id plus prepared/sealed/committed recovery | Object update recovers, but control-plane rollback remains undetected |
| Portability/replacement | Concrete SQLite implementation behind unrelated contracts | Platform-defined contract; provider adapter at composition leaf | Kubernetes semantics leak into production authority |
| Operational complexity | Low, but does not satisfy threat model | High: two security domains, IAM, retention and DR runbook | Medium, but safety requirement fails |
| Required new authority | Would misuse an existing semantic authority; rejected | Yes: narrowly scoped history continuity contract and approved infrastructure | New Kubernetes semantic authority; rejected |
| Demonstrable testability | Local CAS/restart only | Contract, fault, multi-process and Kubernetes DR tests possible | Healthy-cluster tests possible; safe etcd-restore anti-rollback cannot be shown without external witness |
| Decision | **REJECT** | **PREFERRED — PENDING OPERATOR RATIFICATION** | **REJECT** |

The Agent Distribution implementation may inform transaction/CAS style but may
not be reused as the D03-A semantic store. Kubernetes-native state may be used
for non-authoritative availability coordination only; it must never be the
continuity proof.

## 13. Future adversarial acceptance matrix — not execution evidence

| ID | Scenario | Required result | Future evidence class |
|---|---|---|---|
| D03-A01 | Valid fresh provision and correct PIS/GCA history | `ELIGIBLE` at D03-A verdict level; physical CREATE still requires D03-B/C | Contract + integration + eventual Kubernetes |
| D03-A02 | Unproven empty PIS/GCA | `DENY/UNPROVEN_INITIALIZATION`; zero provider call | Contract + fault injection |
| D03-A03 | PIS rollback only | `DENY/PIS_INCARNATION_MISMATCH` or stale watermark | Fault injection + restore |
| D03-A04 | GCA rollback only | `DENY/GCA_INCARNATION_MISMATCH` or stale watermark | Fault injection + restore |
| D03-A05 | Both PIS and GCA rolled back | DENY against higher external head/witness | Fault injection + restore |
| D03-A06 | Both histories plus stale local proof restored | `DENY/CONTINUITY_REVISION_STALE` or `FENCE_STALE` | Fault injection + process restart |
| D03-A07 | Primary anchor or witness rolled back | DENY and external recovery quarantine; never silent allow | Managed-service DR fault injection + eventual deployment test |
| D03-A08 | Stale replica/fencing token | `DENY/FENCE_STALE`; zero provider call | Multi-process + Kubernetes partition |
| D03-A09 | Concurrent compare-and-advance | Exactly one serialized winner; losers receive typed conflict/stale result | Contract + multi-process + eventual Kubernetes |
| D03-A10 | Crash/timeout during authority update | DENY until `TransitionId` resolves to matching committed primary/witness state | Fault injection |
| D03-A11 | Unresolved legacy PIS-only CREATE | `DENY/LEGACY_UNRESOLVED`; no heuristic match or provider call | Contract + integration |
| D03-A12 | Authorized history reconciliation | Eligible only for exact proven scope, new revision/epoch and sealed evidence | Contract + integration + operator workflow |
| D03-A13 | Cross-tenant/environment evidence reuse | `DENY/SCOPE_MISMATCH`, `TENANT_MISMATCH` or `ENVIRONMENT_MISMATCH` | Contract + adversarial multi-tenant |
| D03-A14 | Anchor unavailable or unreadable | `DENY/AUTHORITY_UNAVAILABLE`; no cached allow | Fault injection + Kubernetes network partition |
| D03-A15 | Accept/cancel operation | No unauthorized CREATE-specific history predicate; existing separate semantics retained | Contract + integration regression |
| D03-A16 | Alternate strict production/HTTP path | Cannot reach provider CREATE without the existing guarded dispatch gate | Static reachability + integrated HTTP + eventual Kubernetes |

All rows are future specifications. None was executed in this task.

## 14. ECAP Change Delta and Architecture Decision Audit

### 14.1 Change Delta Audit

| ECAP step | Result |
|---|---|
| A1 changed files/symbols | This proposal document only; no public code symbol |
| A2 dependencies | Documentation references only; no runtime import/config dependency |
| A3 production callers | No production-effective change |
| A4 semantic/composition owners | Existing PIS, GCA, dispatch and lifecycle owners preserved; future contract remains unratified |
| A5 production reachability | `no` for this delta |
| A6 controls | OWN, BND, CTR, TYP, PLG, RPL, SEC, REL, PRD, REG, GOV/EXE, OBS/TRC, CMP/STA/REC |
| A7 historical certification | No status changed; historical PASS remains scoped to its accepted SHA |
| A8 regression selection | Future D03-A01–A16 only; `pytest=0`, `uv run=0` |

Change class: architecture decision proposal with no production delta. ECAP does
not permit full-platform or freeze-ready claims from this delta.

### 14.2 Enterprise audit matrix

`PASS` below means the proposal preserves the invariant at design-document
scope only. It does not promote any `FRZ-*` criterion or prove an implementation.

| Dimension | Invariant | Existing reference | Proposed delta | Evidence available now | Evidence still required | Status |
|---|---|---|---|---|---|---|
| OWN | Exactly one provenance owner, PIS owner, GCA owner and dispatch gate; no shadow authority | `ProductionControlPlaneGovernance`; `ProviderInvocationStore`; orchestrator; gate | Add one narrowly scoped continuity authority, not an invocation/execution ledger | Closed-world owner graph and absence of an existing anchor | Ratified owner/role model; implementation reachability and duplicate-authority gates | **PASS** |
| BND | Tier-0 contract; Tier-3 composition; no reverse imports or provider leakage | `ProductionProcessComposition`; GCA bundle port; `SYS-INV-01/02` | External adapter hidden behind platform contract | Existing layer directions and composition precedent | Exact contract home/import proof and static boundary gates | **PASS** |
| CTR | Consumers depend on one narrow platform contract | `ProviderInvocationStore`; `ProviderInvocationDispatchPort` | New continuity head/read/CAS contract only if ratified | Conceptual operation set in §7 | Code-level contract review, versioning and consumer inventory | **NOT PROVEN** |
| TYP | No `Any`, `object`, dict/string pseudo-contract or reflection | Frozen Pydantic provider/GCA records; typed enums | Types in §7 with explicit enums and deny reasons | Conceptual schema | Implemented models, parser/version negatives and pyright/static gates | **NOT PROVEN** |
| PLG | Backend adapter cannot self-activate or expand authority | Existing explicit production composition | Adapter selected only by approved typed configuration/composition | Composition owner precedent | Approved selection contract, negative reachability and alternate adapter proof | **NOT PROVEN** |
| RPL | Backend replaceable without vendor types in consumers | ECAP RPL-A..H; VBN evidence spec | Neutral continuity port plus adapter leaf | Technology-neutral proposal | Concrete provider inventory; RPL-A..H, alternate backend and recovery proof | **BLOCKED** |
| SEC | Tenant/environment isolation, least privilege and independent recovery roles | Governance principal/resolver; `FRZ-SEC-01/04` | Scope-bound IAM and non-overlapping admin roles | Threat model and role matrix | Concrete IAM, secretless workload identity, cross-scope negatives and compromise exercises | **BLOCKED** |
| REL | Monotonic continuity, fencing, crash ambiguity and fail-closed availability | PIS atomic admission; local uncertain-write handling; `FRZ-REL-03/05` | External head/witness and transition recovery | Current local safety mechanisms and conceptual protocol | Real backend CAS/fault/DR evidence plus D03-B/C | **BLOCKED** |
| PRD | No production bypass; all replicas use same authority | Strict opener, factory and sole dispatch gate; `FRZ-PRD-08` | CREATE-only continuity at the existing gate | Current single-process route inventory | Multi-replica/Kubernetes integrated proof and zero alternate path | **BLOCKED** |
| REG | Corrected blocker has positive and negative regression gates | ECAP and `FRZ-REG-02/03` | D03-A01–A16 future gates | Acceptance specifications only | Executed gates tied to exact implementation SHA and CI ownership | **NOT PROVEN** |
| Governance / Execution | Governance issues scoped evidence; execution remains the gate/provider path | `ProductionControlPlaneGovernance`; `GovernedProviderInvocationDispatchGate`; `FRZ-GOV-01/05/07/10`, `FRZ-EXE-01/02/07` | Governance-owned verdict consumed by sole gate | Existing separation and proposed producer/consumer boundary | Fresh authorization and non-bypassable integration proof | **NOT PROVEN** |
| Observability / evidence | Audit observes transitions but does not mint currentness or execution truth | `RuntimeEvent` principles; package attestation diagnostic boundaries; `FRZ-OBS-01/02/04/06/07` | Authoritative head remains separate from audit projections | Canonical observability/evidence distinction | Redacted audit schema, loss behavior, correlation and exact-SHA export proof | **NOT PROVEN** |
| Compatibility / recovery | Versioned schema; old media/proofs fail closed; recovery cannot create current truth | Versioned PIS/GCA schemas; `FRZ-CMP-03/06/08`, `FRZ-STA-01/03/05`, `FRZ-REC-01/02/04/06/08/09/10` | Versioned anchor records and witness-governed recovery | Historical platform criteria and conceptual restore model | Migration policy, old/new reader matrix, real backup/restore and corruption tests | **BLOCKED** |

### 14.3 Architecture decision conclusion

The preferred option is technically coherent only if the operator can establish
the independent witness and privilege split. Repository code does not prove
that infrastructure. The decision is therefore suitable for independent audit,
but backend activation remains blocked pending explicit operator ratification.

## 15. Applicable FRZ evidence requirements

No checklist row is changed by this proposal.

| FRZ family / IDs | Current status preserved | Required future D03 evidence |
|---|---|---|
| `FRZ-OWN-01..05` | OPEN | Exactly-one continuity owner and composition owner; no shadow PIS, GCA, gate, registry, reconciliation or recovery authority |
| `FRZ-BND-01..05` | OPEN | Contract/import graph; Tier-0 neutral port; Tier-3 adapter/configuration; no shared-to-application reverse dependency |
| `FRZ-CTR-01..06` | OPEN | Narrow canonical contract, provider leakage zero, no pseudo-contract dictionaries, public/internal classification and versioning |
| `FRZ-TYP-01..06` | OPEN | Typed records/enums/results; `Any`, generic `object`, reflection, string dispatch, masking casts and type-ignore all absent at semantic seams |
| `FRZ-PLG-01..07` | OPEN | Explicit provider selection and activation owner; backend cannot self-register or self-expand authority; failure is fail-closed |
| `FRZ-RPL-01..04` and ECAP `RPL-A..H` | OPEN | Concrete dependency inventory, neutral contract, sanctioned alternate implementation, no consumer changes, configuration selection, behavioral/failure equivalence, regression gate and honest extensibility classification |
| `FRZ-SEC-01/04` | OPEN | Cross-tenant/environment denial, workload identity, least privilege, admin separation, no production bypass and compromise response |
| `FRZ-REL-03/05` | OPEN | Explicit idempotency/fencing ownership, termination/uncertainty semantics, no stale or duplicate CREATE dispatch |
| `FRZ-PRD-08` | OPEN | Closed-world strict HTTP/AC3/lifecycle route proves zero alternate CREATE path and all replicas use the same authority |
| `FRZ-REG-02/03` | OPEN | Mechanical positive/negative gates for every D03-A blocker, including rollback, stale fence, unavailable anchor and bypass |
| `FRZ-GOV-01/05/07/10` | Historical PASS preserved | Fresh governed authorization before control-plane mutation; evidence does not itself execute or self-authorize |
| `FRZ-EXE-01/02/07` | Historical PASS preserved | Sole dispatch gate and canonical execution path remain unchanged; no scheduler/worker/anchor execution authority |
| `FRZ-OBS-01/02/04/06/07` | OPEN | Canonical redacted events, correlation and reconstructable audit; observability never mints current head or permission |
| `FRZ-TRC-01/04/06/09` | Historical PASS preserved | Exact scope/revision/fence/transition/provider invocation attribution across restart without heuristic joins |
| `FRZ-STA-01/03/05` | Historical PASS preserved | One truth owner, explicit cross-store transaction limits and stale state rejection for the new state family |
| `FRZ-REC-01/02/04/06/08/09/10` | Historical PASS preserved | Deterministic crash handling, authority-preserving restart, no divergent replay, partial-state deny, explicit backup/restore, identity/tenant preservation and corruption detection |
| `FRZ-CMP-03/06/08` | Historical PASS preserved | Versioned persisted schema, explicit migration/reader policy and no compatibility shim that becomes a parallel authority |
| `FRZ-CFG-01..08` | Historical PASS preserved | Concrete backend and witness selected through approved typed environment configuration; no secrets, implicit defaults, fallback or alternate constructor |

Existing PASS rows are historical/current-platform evidence, not automatic proof
for this proposed new mechanism. All D03-specific contributions require
independent exact-SHA acceptance before any status consideration.

## 16. Unresolved findings and exact operator ratifications

### 16.1 Findings

| Finding | Classification | Required disposition |
|---|---|---|
| No existing approved history continuity backend or independent witness is present in the inspected composition | **IN-SCOPE BLOCKER** | Ratify or reject ELCR-IIRW and select concrete infrastructure |
| Concrete primary/witness administrative independence and break-glass model is unknown | **IN-SCOPE BLOCKER** | Approve a separation-of-duties and compromise model |
| Current per-process SQLite PIS is not a global Kubernetes first-writer domain | **IN-SCOPE BLOCKER** (D03-B; outside D03-A implementation scope) | Separate D03-B architecture decision and evidence |
| Current PIS/GCA lack externally bound history incarnations and watermarks; no cross-store critical protocol exists | **IN-SCOPE BLOCKER** (D03-C; outside D03-A implementation scope) | Separate D03-C architecture decision and evidence |
| Global `FRZ-*`, `R7-Q`, `PROD-Q` and `IMPL-01` remain open/not qualified | **TRACKED FREEZE DEBT** | Preserve roadmap gates; no promotion from this proposal |
| Backend portability, alternate implementation and RPL-A..H evidence do not exist | **TRACKED FREEZE DEBT** | EBH-5/6-compatible VBN artifacts and qualification |
| Managed-service CAS, witness immutability, IAM, revocation latency and DR behavior have not been exercised | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Vendor documentation plus controlled sandbox/qualification evidence |
| D03-A01–A16, multi-process and Kubernetes tests were not executed | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Future explicitly authorized qualification on exact implementation SHA |

### 16.2 Operator ratifications required

1. **OD-D03-A-01 — mechanism:** ratify or reject ELCR-IIRW as the D03-A
   mechanism class.
2. **OD-D03-A-02 — concrete infrastructure:** name the managed primary register
   and immutable witness services, accounts/projects, regions and production
   namespaces. No service is inferred by this proposal.
3. **OD-D03-A-03 — trust model:** approve the non-overlapping platform,
   security/witness, PIS/GCA backup and break-glass administrators, including the
   accepted residual risk of simultaneous trust-root compromise.
4. **OD-D03-A-04 — scope:** approve environment-level history scope as the
   conservative default and define when a tenant sub-scope is independently
   complete enough to authorize.
5. **OD-D03-A-05 — IAM:** approve runtime, provisioner, reconciler, revoker,
   witness and audit-reader workload identities and their exact permissions.
6. **OD-D03-A-06 — recovery:** approve no-restore-in-place, witness retention,
   new authority incarnation, quarantine and dual-controlled cutover rules.
7. **OD-D03-A-07 — freshness/SLO:** set maximum verdict lifetime, revocation
   latency, availability/degraded-mode policy and regional DR objectives.
8. **OD-D03-A-08 — contract/composition:** approve a platform-owned typed
   continuity contract composed by the production process lifecycle, with the
   existing dispatch gate as sole CREATE consumer/enforcer.
9. **OD-D03-A-09 — implementation authority:** if decisions 01–08 are ratified,
   separately authorize the smallest contract/adapter/configuration
   implementation. This proposal does not do so.
10. **OD-D03-A-10 — separate gates:** keep D03-B global PIS and D03-C continuity
    write protocol as independent mandatory decisions; neither is implied by
    D03-A ratification.

## 17. Explicit NO-GO conditions

The proposal must be rejected or the deployment must remain quarantined if any
of the following holds:

- no independently protected witness/high-water mark exists;
- primary and witness can both be silently rolled back by one normal
  administrator or backup workflow;
- the witness can restore its production namespace to a lower head;
- authority selection, issuer trust or positive verdict comes only from
  rollbackable Kubernetes/local configuration;
- the primary lacks linearizable reads, serializable CAS or idempotent
  transition resolution;
- runtime pods can initialize, reconcile, revoke, alter retention or restore;
- PIS/GCA backup operators can mutate the anchor/witness;
- missing/unreadable/stale state falls back to allow;
- Kubernetes Lease, ConfigMap, CRD, `resourceVersion`, schema metadata,
  signature, local hash or third SQLite file is treated as currentness proof;
- two pod-local PIS instances or a distributed lock are presented as D03-B;
- D03-C uses a non-atomic read-check-execute sequence without reservation,
  fencing and fail-closed partial-state recovery;
- any CREATE path bypasses `GovernedProviderInvocationDispatchGate`;
- accept/cancel is accidentally subjected to the CREATE-specific predicate;
- the anchor stores provider invocations or GCA execution records and becomes a
  shadow PIS/GCA;
- a vendor type, `Any`, generic `object`, dictionary pseudo-contract,
  reflection or dynamic admission appears at the semantic boundary;
- any claim of physical provider exactly-once is made without provider-native
  guarantees and separate qualification; or
- implementation begins before operator ratification and explicit
  implementation authorization.

## 18. Final recommendation and status

**Recommend ELCR-IIRW for operator ratification**, subject to all decisions in
§16.2. It is the only evaluated class that can provide linearizable
cross-replica continuity while detecting rollback of the primary authority
itself within an explicit split-domain trust model.

The existing Agent Distribution SQLite backing and Kubernetes-native
control-plane objects are rejected for D03-A because neither survives its own
authorized recovery/restore domain as a trustworthy monotonic currentness
authority.

This document contains no production implementation and no qualification
execution. D03-B and D03-C remain mandatory blockers. Recommended task status:

**READY FOR AUDIT — ARCHITECTURE DECISION PENDING OPERATOR RATIFICATION**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
