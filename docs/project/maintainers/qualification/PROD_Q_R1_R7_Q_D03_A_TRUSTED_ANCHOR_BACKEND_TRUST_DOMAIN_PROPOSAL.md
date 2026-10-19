# PROD-Q-R1-R7-Q — D03-A Trusted Anchor Backend & Trust Domain Proposal

> **PROPOSAL — NOT RATIFIED — NO IMPLEMENTATION AUTHORIZED**

| Field | Value |
|---|---|
| Task | `PROD-Q-R1-R7-Q-D03-A-TRUSTED-ANCHOR-BACKEND-TRUST-DOMAIN` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` |
| Branch | `development` |
| Required and inspected START_HEAD | `cee58e4751eb13169d55056c2293584e792fc49c` |
| Origin at inspection | `https://github.com/jakbuczarnecki/intergrax.git` |
| Baseline relation | local `HEAD == origin/development == START_HEAD`; clean tree before this document |
| Decision-03 / OD-D03-A-01 authority | Kubernetes / multi-instance target and ELCR-IIRW mechanism class ratified; concrete services, trust domain, global PIS and continuity protocol not ratified |
| Production delta | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |
| Maximum status | **READY FOR AUDIT — OPERATOR RATIFICATION REQUIRED** |

This R1 proposal performs only the required D03-A concrete infrastructure and
trust-domain selection for OD-D03-A-02–08. It does not authorize a provider
subscription, database, Kubernetes resource, credential, migration, runtime
configuration, D03-B global PIS implementation or D03-C continuity protocol.

## 1. Decision summary

### 1.1 Preferred concrete service pair

Select this single preferred ELCR-IIRW pair for operator ratification:

1. **Primary register — Google Cloud Spanner Enterprise Plus, base
   multi-region configuration `eur3`**, in a dedicated GCP project and database.
   Only strong reads and serializable read-write transactions are admissible.
2. **Independent witness — Azure Blob Storage, Standard general-purpose v2,
   RA-GZRS, primary region `North Europe`**, in a dedicated subscription in a
   separately governed Microsoft Entra tenant. The witness is a deterministic
   chain of fixed-size append-blob blocks under a locked container-level
   time-based WORM policy with protected append writes.

Spanner provides the externally consistent transaction and strongly current
read required for the authoritative head. Azure Blob provides strong
read-after-write consistency, an exact append-position precondition and WORM
protection for already appended blocks. A witness adapter derives the highest
sealed `ContinuityRevision` from the last valid fixed-size block; Azure does not
natively expose an ELCR revision type. The derivation, segment rollover and
unknown-outcome handling therefore remain properties to qualify by execution.

The services form one governance-continuity mechanism but remain separate
security and recovery domains. The witness never becomes a PIS, GCA or mutable
second current-head owner. It proves only the immutable maximum seal and detects
a primary prefix/rollback.

### 1.2 Why this pair is preferred

- Spanner documents external consistency, strong reads and serializable
  read-write transactions across its multi-region configuration.
- `eur3` keeps the database in the EU, uses read-write regions in Belgium and
  the Netherlands and a quorum witness in Finland, and documents 99.999%
  multi-region availability.
- Azure Append Block accepts `x-ms-blob-condition-appendpos`; an offset mismatch
  fails with HTTP 412. Azure Storage documents strong consistency after a
  successful write.
- A locked container WORM policy prevents overwrite/deletion of retained blocks,
  permits only protected appends when configured before locking, and prevents
  deletion of a non-empty protected account.
- Separate GCP and Azure identity roots make a non-overlapping human
  administration model possible. This independence is an operator-controlled
  topology, not a guarantee created merely by choosing two vendors.
- The pair retains platform-neutral contracts; all service API types and
  credentials remain in provider adapters and sanctioned composition.

The choice is conditional on OD-D03-A-02–08 and future sandbox/deployment
evidence. It is not a provisioning instruction or production approval.

### 1.3 Evidence classification

| Class | Meaning in this proposal |
|---|---|
| **DOC** | Explicit vendor documentation cited in §17; supports a service primitive, not deployment correctness |
| **OP** | Operator-controlled fact requiring ratification or inspection: accounts, administrators, region eligibility, budget, retention, network and break-glass |
| **TEST** | Property requiring execution evidence: adapter protocol, IAM negatives, crash resolution, regional recovery, latency, rollover and Kubernetes fencing |

## 2. Canonical baseline and current roadmap position

The bounded canonical set inspected at the exact baseline is:

- [Platform Enterprise Completion Roadmap](../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md)
- [Platform Enterprise Freeze Acceptance Checklist](PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md)
- [Enterprise Continuous Architecture Assurance Protocol](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md)
- [Trusted History Anchor Mechanism Decision-03](PROD_Q_R1_R7_Q_TRUSTED_HISTORY_ANCHOR_MECHANISM_DECISION_03.md)
- this proposal; and
- [OD-D03-A-01 operator ratification](PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md).

The roadmap records Decision-03 as **RATIFIED TARGET CONTRACT ONLY — NO BACKEND
OR GLOBAL PIS AUTHORIZED** and OD-D03-A-01 as **RATIFIED CLASS ONLY — NO
BACKEND/IMPLEMENTATION**. It explicitly makes OD-D03-A-02–08 the next task and
keeps D03-B, D03-C, `IMPL-01`, `R7-Q`, `PROD-Q` and global `FRZ-*` open. No
canonical order or scope change was found.

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
Azure witness security / recovery trust domain
  └─ Blob Storage RA-GZRS / North Europe
       └─ locked WORM append-blob chain per history scope
            ^
            │ strong read / conditional fixed-position append
GCP primary-register trust domain
  └─ Cloud Spanner Enterprise Plus / eur3
       ├─ one typed head per EnvironmentHistoryScopeIdentity
       └─ durable idempotent transition records
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

The anchor does not store provider payloads, provider outcomes, GERs or F02
records. It stores only typed continuity identities, watermarks, transition
digests, fences and authorization references. Spanner and Blob resource
identifiers remain adapter configuration and do not enter the platform contract.

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

### 5.3 Trust assumptions requiring operator confirmation

- **OP-TRUST-01:** the Spanner project belongs to the platform database domain;
  the Azure subscription belongs to a separate security-custodian Entra tenant;
  PIS/GCA backup operators belong to neither administration group.
- **OP-TRUST-02:** no person, group, CI identity, federation administrator,
  emergency account or vendor-management automation can administer/restore both
  domains. Two subscriptions or projects under one shared super-administrator
  do **not** satisfy this assumption.
- **OP-TRUST-03:** break-glass credentials are different, hardware-protected,
  monitored and require two-person approval independently in each domain.
- **OP-TRUST-04:** an independent hold remains until formally dual-controlled
  decommissioning. The ten-year interval previously named here is an unadopted
  historical candidate. §6.2.1 leaves the production duration unset. It is not
  a binding configuration. `ELCR-ACTIVE` is not adopted.
- **OP-TRUST-05:** Kubernetes workload federation is bound to the exact cluster
  issuer, namespace, service account and environment. Runtime credentials are
  short-lived and have no initialize, reconcile, revoke, IAM, retention,
  backup, restore or resource-delete permission.
- **OP-TRUST-06:** resource identifiers and issuer roots are delivered through
  approved typed configuration and external IAM, never accepted from
  rollbackable pod-local state as authority.
- **OP-TRUST-07:** an etcd/control-plane restore and stale honest replica are in
  scope and must be fenced. Active arbitrary-code compromise of the correctly
  federated `AnchorTransitionWriter` is not prevented by storage immutability:
  it can at least append an invalid forward seal and cause denial. If resistance
  to that actor is required, direct pod write authority is rejected and a
  separately trusted transition service requires another architecture decision.

The services document primitives that can support these assumptions; they do
not prove the actual organization chart or IAM graph. Suspected compromise,
missing separation or an uninspectable root administrator forces quarantine.
Simultaneous compromise of both independent identity roots remains residual
risk and is not claimed to be prevented.

### 5.4 Explicitly out of scope

- Provider-side exactly-once physical effects.
- Automatic replay or reconciliation of ambiguous external effects.
- D03-B selection/implementation of a global PIS.
- D03-C cross-store write ordering and dispatch protocol implementation.
- Protection after simultaneous compromise of all independent trust roots.

## 6. Preferred ELCR-IIRW mechanism

### 6.1 Primary register mapping — Cloud Spanner `eur3`

| Required property | Concrete mapping | Evidence class |
|---|---|---|
| Linearizable current read | Spanner default strong read; bounded/exact staleness is forbidden for admission | **DOC** |
| Transactional CAS | Serializable read-write transaction reads the scoped head, checks expected revision/digest/fence, inserts one transition and updates the head | **DOC** primitive; **TEST** schema/adapter |
| Idempotent resolution | Unique `(scope, transition_id)` row and immutable request digest; same ID/same digest resumes, mismatch denies | **TEST**; do not rely on a short client retry window |
| Monotonic revision/fence | Checked integer successor and non-decreasing fence in the same transaction; commit timestamp is audit metadata, not the CAS token | **TEST** |
| Multi-region availability | Enterprise Plus `eur3`: Belgium and Netherlands read-write regions, Finland quorum witness; documented 99.999% availability | **DOC** |
| Independent restore | Spanner backup/PITR restoration is performed only into a new database and new `AnchorAuthorityIncarnationId`; restored data remains quarantined until compared with Azure | **DOC** restore capability; **OP/TEST** policy |
| Workload identity | Kubernetes projected OIDC token exchanged through Google Workload Identity Federation; exact-scope database role | **DOC** primitive; **OP/TEST** binding |
| Security audit | Cloud Audit Logs Admin Activity, Data Access and System Event logs exported to a separately retained audit destination | **DOC** primitive; **OP/TEST** retention |

Only the primary endpoint and strong reads are authoritative. A Spanner backup,
PITR image, export, read-only stale read or locally cached result cannot establish
currentness. Database deletion protection is mandatory defense-in-depth but is
not anti-rollback evidence.

### 6.2 Witness mapping — Azure immutable append blob

For each `EnvironmentHistoryScopeIdentity`, the witness adapter uses a
deterministic append-blob chain. Every append block is a canonical, fixed-size
4 KiB `WitnessSealV1` containing scope hash, global revision, transition ID,
primary authority incarnation, transition digest, predecessor seal digest,
segment ordinal and record ordinal. Readers verify the chain and range-read the
last block after `Get Blob Properties`.

| Required property | Concrete mapping | Evidence class |
|---|---|---|
| Strong current read | Azure Storage strong consistency for successful insert/update; primary endpoint only | **DOC** |
| Conditional next append | `Append Block` with `x-ms-blob-condition-appendpos` equal to the expected fixed-size offset; mismatch returns 412 | **DOC** primitive; **TEST** encoding/races |
| Highest sealed revision | committed block count/content length plus strong tail range read; record revision and predecessor digest must match | **TEST** protocol over **DOC** primitives |
| Immutability | Locked container-level time-based retention. Duration unset. On that policy the protected-append properties are mutually exclusive: the append-blob mapping is `allowProtectedAppendWrites=true` and `allowProtectedAppendWritesAll` is not true on the same policy. The independent hold and the combined test are in §6.2.1. Not a deployed setting and not an operator-ratified production configuration | **DOC** for the 2026-04-01 immutability-policy contract; **OP/TEST** for duration, lock, and the combined hold |
| No lower-head restore | existing retained blocks cannot be overwritten/deleted; no service restore operation is accepted as currentness | **DOC/OP** |
| Account deletion resistance | protected non-empty locked container prevents account deletion; Azure Resource Manager `CannotDelete` lock adds defense-in-depth | **DOC** |
| Identity | external Kubernetes OIDC federation to Entra; custom data role limited to read and `blobs/add/action`; Shared Key disabled | **DOC** primitive; **OP/TEST** binding |
| Audit | Activity Log for control plane, Storage resource logs for data plane, policy audit log for locked retention changes; separately retained export | **DOC** primitive; **OP/TEST** retention |

Append blobs permit at most 50,000 committed blocks. The adapter closes a
segment at 49,000 with an immutable `SEGMENT_CLOSED` record naming the one
deterministic successor. The successor genesis block binds the prior segment
digest. A crash between close and successor creation is `RECOVERY_REQUIRED`.
Readers follow the immutable link rather than trust object listing. Extra data
after `SEGMENT_CLOSED`, a second successor, a gap or a malformed tail is
quarantine, never truncation or allow. Rollover is **not** a native vendor
transaction and requires adversarial execution proof.

RA-GZRS copies to the paired Azure region asynchronously. The secondary endpoint
is never a currentness source. An unplanned account failover can expose a prefix,
so it always forces comparison with Spanner and quarantine; it cannot
automatically resume CREATE. This is safe fail-closed behavior, not zero-RPO
regional availability.

### 6.2.1 Production-target configuration profile

Documentation only. No account, policy, or credential is created by this
section. The profile below is the single preferred production target. It does
not ratify OD-D03-A-06 or OD-D03-A-07 and does not open infrastructure
qualification.

#### Azure protected-append mapping

Pages read on 2026-10-19, without a cloud account:

- [Create Or Update Immutability Policy, api-version 2026-04-01](https://learn.microsoft.com/en-us/rest/api/storagerp/blob-containers/create-or-update-immutability-policy?view=rest-storagerp-2026-04-01).
  `allowProtectedAppendWrites` and `allowProtectedAppendWritesAll` are properties
  of the time-based immutability policy and are mutually exclusive. The published
  sample sets only `allowProtectedAppendWrites: true`. When that property is
  enabled, new blocks can be written to an append blob and existing blocks cannot
  be modified or deleted. The property cannot be changed through
  ExtendImmutabilityPolicy.
- [Container-level WORM policies](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-container-level-worm-policies).
  `allowProtectedAppendWrites` allows new blocks on an append blob and does not
  change block-blob or page-blob immutability. `allowProtectedAppendWritesAll`
  adds new-block writes for block blobs. After the time-based policy is locked,
  neither property can be changed. For a protected append blob, effective
  retention is the interval after the blob's last modification time.
- [Immutable storage overview](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-storage-overview).
  A time-based policy and a legal hold can be set together. The minimum interval
  is one day and the maximum is 146,000 days. A locked policy can be extended
  and cannot be shortened. A container-level locked policy allows at most five
  increases. Append Block is otherwise prohibited while a time-based policy
  and/or a legal hold is in effect.
- [Set Legal Hold, api-version 2026-04-01](https://learn.microsoft.com/en-us/rest/api/storagerp/blob-containers/set-legal-hold?view=rest-storagerp-2026-04-01).
  The hold is a separate resource. Its only protected-append property is
  `allowProtectedAppendWritesAll`. The operation has no
  `allowProtectedAppendWrites` property.

These pages do not state that `allowProtectedAppendWrites=true` on the
time-based policy still permits Append Block when the hold's
`allowProtectedAppendWritesAll` is omitted or false. They also do not state
that setting the hold property true conflicts with
`allowProtectedAppendWrites=true` on the immutability policy. Mutual exclusion
is stated only for the two properties of that policy. A secondary endpoint,
Last Sync Time, object listing, or unplanned-failover prefix remains unable to
authorize CREATE.

**Time-based policy, append-blob witness — `RECOMMENDED TECHNICAL CONFIGURATION — PENDING SANDBOX VERIFICATION`:**

- `allowProtectedAppendWrites=true`
- `allowProtectedAppendWritesAll` is not true on this same policy
- `immutabilityPeriodSinceCreationInDays` is unset
- a production policy stays unlocked until that combined test has passed on a sandbox container

This is not a deployed setting and not an operator-ratified production
configuration. Ten years is not adopted. `allowProtectedAppendWritesAll=false`
on the legal-hold resource is not part of this recommendation: the hold
contract does not establish that value.

**Combined locked retention and independent hold:** the single-flag choice is
unresolved.

Exact remaining compatibility test, for a later authorized sandbox only. This
section does not authorize that sandbox:

1. Use general-purpose v2, Hot, RA-GZRS, North Europe, one append blob, shared
   key disabled, and blob point-in-time restore and last-access tracking
   disabled.
2. Create an unlocked container time-based policy with
   `allowProtectedAppendWrites=true`, with `allowProtectedAppendWritesAll` not
   true, and with an interval of at least one day labeled sandbox.
3. Show that Append Block succeeds and that modifying or deleting an existing
   block fails.
4. Add a container legal hold with `allowProtectedAppendWritesAll` omitted or
   false. Repeat the append and the existing-block modification or delete.
5. If the append fails, set `allowProtectedAppendWritesAll=true` on the hold
   only, leave the time-based policy unchanged, and record whether the service
   accepts that pair.
6. Lock the time-based policy. Show that the protected-append property cannot
   be changed and that the interval cannot be shortened.
7. Show that a secondary-endpoint read and an unplanned-failover prefix do not
   authorize CREATE.

If steps 4 and 5 both fail to keep append-only writes under a locked time-based
policy and a distinct hold, the result is `STOP — ARCHITECTURE DECISION REQUIRED`.
Do not select another provider, edition, region, or storage mechanism from that
result.

#### Retention profile

The local contract `HistoryContinuityRetentionPolicy` already requires
`retention_seconds` to sit inside `ceiling_seconds` and `legal_hold` to be
true. Its text says this is not a production period. That contract remains the
non-weakening local control. It is not Azure immutability and it is not IAM.
No new semantic contract is added.

The duration rule is readiness §7.5.3:

```text
required_retention >
max(
    PIS_MAX_RESTORABLE_AGE,
    GCA_MAX_RESTORABLE_AGE,
    ANCHOR_PRIMARY_MAX_RESTORABLE_AGE
) + INVESTIGATION_WINDOW
```

When `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM` is declared and applicable,
`required_retention` must also be greater than or equal to that minimum. A
missing input is not zero. The rule is not evaluated. Azure measures protected
append-blob retention from the last modification time, so a future interval is
not a creation-time wall clock.

| Element | Value | Admission |
|---|---|---|
| Production-target retention | unset | Fail closed. A production witness policy is not duration-compliant while any required input is missing. |
| Sandbox retention | unset | A later authorized sandbox interval is labeled sandbox. It does not assert production compliance. |
| Operational preservation hold | required for the active environment | Local `legal_hold=true` is this flag. It is not WORM and not a legal finding. The Azure container hold is a separate unset cloud control. |
| Legal-purpose hold | unset | Used only after a legal or contractual duty is declared. The name `legal_hold` and the unadopted label `ELCR-ACTIVE` do not create it. |
| Dual-controlled decommissioning | required; procedure unset | Hold and retention are released only after formal retirement, by separated authorities. No approvers are named. |

#### Capacity profile

One architecture. Spanner Enterprise Plus, base multi-region `eur3`, strong
current reads only. Azure general-purpose v2 Standard, Hot, RA-GZRS, North
Europe. Conditional append and exact read-back. Independent administrative
domains. Short-lived workload federation. No static runtime administrator
credentials. Fail-closed recovery. No automatic CREATE on incomplete evidence.

Capacity stays variable. The figures below are the accepted preparation P1
limits. They were not re-fetched for this correction.

| Quantity | Disposition |
|---|---|
| Smallest documented Spanner candidate | 100 processing units, then multiples of 100 below 1000. Prerequisite: a later authorized create shows that the API accepts 100 processing units for `eur3`. This is a documented floor, not a production recommendation. |
| Managed-autoscaler minimum | 1000 processing units or one node. Outside this profile unless the operator accepts that floor. Enabling the autoscaler scales a smaller instance up to the minimum. |
| Recommended production capacity | unset. No workload evidence. |
| Capacity proven under production workload | unset. No measurement. |
| Azure stored capacity | variable `retained_gib`. It grows for the whole retention period. No account-size minimum is adopted. |

#### Cost and SLO profile

Safety invariants remain readiness §7.3 and §8.1. Authorization RPO 0 means an
unverified older history is never accepted. It does not mean that physical data
loss cannot occur.

The 5-second positive-verdict lifetime, the 5-second committed-revocation
visibility, the 500-millisecond transition p99, and the 4-hour CREATE recovery
target remain proposals in readiness §8.4. They are not vendor claims and they
are not measured achievements. Spanner's 99.999% multi-region availability
statement stays a vendor claim for the instance configuration. It is not a
measured cross-cloud SLO and not a purchased SLA for this platform.

Cost continues to use preparation P2 price inputs and readiness §8.3 formulas.
No price total, budget maximum, or SLA is stated. A total stays blocked while
these inputs are missing: Enterprise Plus `eur3` node-hour, storage, backup,
replication, and egress rates; North Europe Hot RA-GZRS write, read, and other
operation rates; processing units; transition volume; `retained_gib`; audit-sink
size and retention; private-path choice; discount and tax. Immutable storage
adds no extra capacity charge on the cited overview page. A lower price is not
obtained by weakening consistency, availability, or administrative independence.

### 6.3 IAM / authorization matrix

| Principal / domain | Spanner | Azure witness | PIS/GCA | Explicit prohibition |
|---|---|---|---|---|
| `AnchorRuntimeReader` (pod WIF) | strong `SELECT` through a read-only database role | properties/range read only | none | all writes, restore, IAM, retention |
| `AnchorTransitionWriter` (pod WIF) | narrowly scoped `SELECT/INSERT/UPDATE` on head/transition tables through database role | custom container role: read plus `Microsoft.Storage/storageAccounts/blobServices/containers/blobs/add/action` | no backup/restore; future D03-C may use a separate PIS port | initialize/reconcile/revoke, delete/overwrite, IAM, retention, cross-scope |
| `AnchorProvisioner` | create genesis rows only under governed workflow | create deterministic genesis append blob | authenticate newly provisioned history identities only | ordinary runtime transition or provider dispatch |
| `AnchorReconciler` | create new incarnation/epoch in quarantine | append recovery seal only | read approved reconciliation evidence | prior-row rewrite, delete, physical dispatch |
| `AnchorSecurityRevoker` | CAS to `QUARANTINED/REVOKED` | append matching revocation seal | none | reactivation or history mutation |
| GCP Spanner DBA / restore admin | instance/database/backup operations; separate `roles/spanner.restoreAdmin` | none | none | Azure/IAM witness or PIS/GCA restore |
| Azure witness custodian | retention, legal hold, account lock, Entra/RBAC under dual control | management only; no primary/PIS/GCA role | none | Spanner or PIS/GCA administration |
| PIS/GCA backup operator | none | none | backup/restore under its runbook | any anchor/witness operation |
| `AnchorAuditReader` | read transition/audit evidence | read witness and audit evidence | read approved evidence exports | mutation or admission |

Built-in broad data-writer roles are not approved for runtime. Exact custom GCP
database grants, Azure DataActions/ABAC name conditions, token audiences and
deny tests must be recorded before provisioning. Account keys, service-account
keys, SAS tokens and credentials in pod secrets are forbidden. Human approval
is typed governance evidence, never a mutable `allow=true` field.

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

### 7.5 Vendor-neutral port to provider mapping

| Platform operation | Spanner adapter | Azure witness adapter | Forbidden leakage |
|---|---|---|---|
| `read_current_authoritative_head` | strong read of scoped head and transition | properties plus exact tail range read; verify chain/highest seal | Spanner timestamps, Azure ETags or HTTP codes in domain result |
| `prepare_transition` | serializable transaction with expected revision/digest/fence and unique transition ID | no call | SQL/schema objects in consumer |
| `seal_prepared_transition` | read exact immutable prepared payload | conditional 4 KiB append at expected offset | blob URL/SAS/request headers in contract |
| `mark_witness_sealed` | serializable CAS attaching typed `WitnessReceipt` | strong read-back resolves uncertain append | treating response receipt alone as currentness |
| `commit_transition` | serializable CAS `WITNESS_SEALED → COMMITTED` | strong highest-seal revalidation | client-side last-write-wins |
| `resolve_transition` | strong lookup by scope + transition ID + request digest | expected-offset and tail lookup | retry with a new transition ID |
| `recover_authority` | new database/incarnation, reconstructed in quarantine | read immutable chain from genesis/current retained root | restore-in-place or automatic activation |

The platform contract owns types, deny reasons and state-machine semantics.
Provider adapters translate vendor conflicts/timeouts into typed outcomes such
as `EXPECTED_REVISION_CONFLICT`, `APPEND_POSITION_CONFLICT`,
`OUTCOME_UNKNOWN`, `AUTHORITY_UNAVAILABLE` and `RECOVERY_REQUIRED`. Selection is
through an approved typed profile at sanctioned production composition.
Consumers and the dispatch gate never branch on `spanner`, `azure`, region,
endpoint, SQL state, ETag or HTTP status.

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

The anchor has no cross-cloud distributed transaction. One scope permits at most
one non-terminal transition:

1. **PREPARED:** one Spanner serializable transaction checks committed head
   `n-1`, exact digest/fence/incarnations, and inserts immutable transition
   `TransitionId` + request digest for `n`; the scoped head becomes
   `PREPARED`. CREATE is denied.
2. **SEAL:** the adapter strong-reads Azure's current tail, verifies `n-1`, then
   appends exactly one fixed-size seal for `n` using
   `x-ms-blob-condition-appendpos=<expected_offset>`. HTTP 412 is a conflict,
   never a retry-as-new.
3. **WITNESS_SEALED:** after strong read-back of the exact range and digest, a
   Spanner CAS attaches `WitnessReceipt(blob identity, segment, byte range,
   seal digest, service ETag/request correlation)` and changes only
   `PREPARED → WITNESS_SEALED`.
4. **COMMITTED:** a second Spanner CAS rechecks the receipt, predecessor,
   authority incarnation and fence, then advances the authoritative head and
   changes only `WITNESS_SEALED → COMMITTED`.
5. A reader strong-reads both services and requires the same scope, revision,
   transition digest and authority incarnation. Any pending, missing, malformed,
   stale or unequal state denies CREATE.

Idempotent crash resolution is explicit:

- before PREPARED commit: transition is absent; caller may submit the same
  `TransitionId` and payload;
- after PREPARED: resolve the durable row and resume only that ID/payload;
- Azure append response lost: read the expected tail offset; exact seal means
  proceed, absent means retry the same append/offset, any other content means
  quarantine;
- after witness append but before WITNESS_SEALED: witness is ahead; reconstruct
  the receipt by exact read-back and CAS the existing transition;
- after WITNESS_SEALED but before COMMITTED: idempotently finalize only when the
  same prepared head and highest seal still match;
- primary claims a seal not present as Azure's highest valid seal, or Azure is
  more than the one expected pending revision ahead: quarantine;
- no resolver creates a new transition ID, decrements either head, truncates a
  blob or infers success from timeout.

`COMMITTED` here means only that the continuity transition is durably sealed.
It does not prove provider execution or authorize physical redispatch.

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
- **Witness rollback/prefix after Azure failover:** Spanner is ahead; deny and
  quarantine. RA-GZRS `Last Sync Time` is diagnostic only and cannot authorize.
- **Witness disaster:** no CREATE until the latest retained chain is proved.
  A lower prefix or newly empty account is not an allowed recovery mode.
- **Whole Kubernetes control-plane rollback:** neither primary nor witness state
  is stored in etcd. Restored workload identity has no initialize/reconcile/
  revoke authority; stale fences and revisions fail.

An old snapshot cannot silently become authoritative because a Spanner restore
must use a new database and authority incarnation, while the retained Azure
chain remains ahead. Activation requires equality with the independently read
highest witness revision and a newly sealed dual-controlled recovery revision.
The simultaneous loss/compromise of both independent roots is residual risk.

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
- Resolution of PREPARED/WITNESS_SEALED/COMMITTED never invokes the provider.
  If a crash also leaves PIS/GCA or external-effect state ambiguous, D03-C must
  preserve that ambiguity and deny; it must not automatically redispatch
  `create_work`.

## 9. Kubernetes deployment contract

### 9.0 Deployment, administration and failure-domain topology

```text
Kubernetes application replicas (region(s) to be operator-confirmed)
  ├─ short-lived OIDC federation → GCP anchor project
  │    └─ Spanner Enterprise Plus eur3 / dedicated ELCR database
  ├─ short-lived OIDC federation → separate Entra tenant
  │    └─ Azure Storage GPv2 RA-GZRS / North Europe / locked WORM container
  └─ no cloud account keys, restore rights or retention rights

GCP platform DBA domain ── administer/backup/restore Spanner only
Azure security-custodian domain ── administer WORM witness only
PIS/GCA backup domain ── restore PIS/GCA only
Independent audit domain ── read/export audit evidence only
```

| Failure/compromise | Surviving authority | Required result |
|---|---|---|
| pod/replica/cluster restart or etcd rollback | external Spanner + Azure current heads | stale revision/fence denies; no local fallback |
| active cluster admin impersonates the authorized transition-writer subject | immutable past survives, but writer can attempt forward mutation | fail closed on invalid chain; incident quarantine; outside rollback guarantee per OP-TRUST-07 |
| PIS/GCA backup operator rollback | both ELCR domains | incarnation/watermark mismatch denies |
| Spanner DBA restore/corruption | immutable Azure high-water mark | primary prefix mismatch; new-incarnation recovery in quarantine |
| Azure custodian error or regional prefix | Spanner committed head | mismatch denies; no stale secondary allow |
| GCP regional failure inside `eur3` | Spanner multi-region quorum plus Azure | service behavior per vendor; adapter still strong-read only |
| Azure North Europe regional failure | Spanner survives; Azure secondary may lag | CREATE unavailable/quarantined until exact chain is proven |
| simultaneous compromise/loss of GCP and Azure roots | no independent currentness source | outside guaranteed protection; incident quarantine and operator decision |

### 9.1 Replica identity and access

- Each replica uses a projected Kubernetes service-account token federated
  independently to Google Workload Identity Federation and Microsoft Entra
  Workload ID; no shared static credential.
- Authorization binds OIDC issuer, service-account subject, token audience,
  cluster/deployment identity, environment and history scope.
- Runtime replicas may read and perform only future D03-C scoped transitions
  through the neutral ports and narrow database/container roles.
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
- Revocation latency and verdict-lifetime values proposed in §9.5 remain
  unratified and **NOT PROVEN** until operator selection and execution tests.

### 9.4 Disaster recovery and audit

- Spanner recovery restores a backup to a new database, assigns a new authority
  incarnation and remains quarantined.
- The recovery service reconstructs/verifies from the highest Azure
  witness-sealed transition chain and independently retained audit evidence.
- Cutover to the recovered incarnation requires dual-controlled authorization
  and a new witness-sealed revision.
- Provisioning, reconciliation, revocation, IAM, retention and restore actions
  must be attributable to principal, policy revision, request id and exact
  artifact/configuration SHA without secrets in evidence.

Azure unplanned failover is not a transparent recovery path because
geo-replication is asynchronous. A planned failover may be tested for zero data
loss, but neither planned nor unplanned failover can bypass exact head
comparison, authority-incarnation checks and a new recovery seal.

### 9.5 Availability, latency and cost implications

- Spanner `eur3` documents 99.999% multi-region availability and incurs
  cross-region quorum latency; only strong reads/transactions are allowed.
- Spanner supports granular capacity from 100 PU, but managed autoscaling has a
  1000-PU minimum. Enterprise Plus multi-region remains materially more
  expensive than a regional service; capacity and budget require explicit
  operator sizing rather than an invented one-node minimum.
- Every transition adds an inter-cloud Azure append/read-back and possible
  egress cost. Blob capacity cost is small, but transactions, RA-GZRS, immutable
  retention, audit export and private connectivity are recurring costs.
- The safety availability objective is **RPO 0 for authorization**: an
  unverified prefix is never accepted. This intentionally allows CREATE
  unavailability during witness/primary uncertainty.
- Proposed operator targets are: positive verdict lifetime at most 5 seconds,
  committed revocation visible to new reservations within 5 seconds, anchor
  transition p99 at most 500 ms from approved EU cluster regions, and
  safety-preserving CREATE recovery target of 4 hours after a regional event.
  These are **OP/TEST** targets, not documented vendor guarantees. Failure to
  meet latency or RTO requires an operator decision, never weaker consistency.

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

## 12. Concrete service-pair comparison

At most three pairs were assessed:

| Pair | Primary findings | Witness findings | DR / cost / decision |
|---|---|---|---|
| **A — Cloud Spanner Enterprise Plus `eur3` + Azure Blob Storage GPv2 RA-GZRS immutable append blob** | External consistency, strong reads and serializable transactions support head + transition CAS; durable ID-based read-back handles unknown commit outcome | Strong reads plus append-position condition and protected WORM append support one ordered seal chain; split Entra tenant can be independently governed | Multi-region primary, fail-closed Azure regional recovery; highest fixed cost and cross-cloud latency. **PREFERRED — OD-D03-A-02–08 REQUIRED** |
| **B — DynamoDB MRSC global table + Azure Blob witness as above** | MRSC gives cross-region strong reads and conditional writes, but AWS documents that MRSC global tables do not support transactional operations. Regional transactions are ACID only in their invocation region | Witness is viable under the same conditions as Pair A | Does not meet the selected primary design's atomic head + separate durable transition requirement in multi-region mode without weakening or redesigning the contract. **REJECT** |
| **C — Cloud Spanner `eur3` + Amazon S3 Object Lock** | Primary qualifies as in Pair A | S3 is strongly consistent and Object Lock is WORM, but ordinary S3 has no atomic update across keys and no protected append. Conditional PUT protects one key only; append exists only in S3 Express One Zone directory buckets, where Object Lock is unavailable | WORM objects alone do not provide a conditional next-revision/highest-head primitive. **REJECT** |

Pair A is preferred only within the stated split-domain model. The Azure
append-blob chain is an application protocol requiring execution proof; the
proposal does not relabel WORM storage itself as monotonic currentness.
Kubernetes Lease/ConfigMap/CRD, a third SQLite file and two accounts with one
shared administrator remain rejected without re-evaluation.

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
| D03-A17 | Two writers append revision `n` at one Azure offset | One append succeeds; loser receives conflict; exact tail resolves winner | Azure sandbox concurrency |
| D03-A18 | Azure accepted append but response is lost | Exact range read recovers the same seal; no new transition ID or second seal | Proxy fault injection |
| D03-A19 | Crash before/after `SEGMENT_CLOSED` and successor genesis | CREATE denied until one deterministic linked successor is verified | Azure sandbox rollover fault matrix |
| D03-A20 | Azure unplanned RA-GZRS failover exposes a prefix | Spanner/witness mismatch and quarantine; no automatic CREATE | Controlled vendor sandbox DR |
| D03-A21 | Spanner backup restored to a new database | New authority incarnation remains quarantined until highest witness seal and dual control are proven | GCP sandbox restore |
| D03-A22 | Runtime identity attempts init/reconcile/revoke/delete/retention/restore or cross-scope access | IAM deny and auditable event in both clouds | Dual-cloud IAM adversarial test |
| D03-A23 | Primary or witness admin attempts to administer the other domain | No effective entitlement or impersonation chain | Identity-graph inspection + break-glass exercise |

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
| A8 regression selection | Future D03-A01–A23 only; `pytest=0`, `uv run=0` |

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
| RPL | Backend replaceable without vendor types in consumers | ECAP RPL-A..H; freeze RPL controls | Neutral ports; Spanner/Azure mappings remain adapter-local | Concrete inventory and two rejected alternatives | RPL-A..H implementation evidence, alternate conforming adapters and recovery equivalence | **BLOCKED** |
| SEC | Tenant/environment isolation, least privilege and independent recovery roles | Governance principal/resolver; `FRZ-SEC-01/04` | Separate GCP/Entra roots, WIF and narrow role matrix | Vendor IAM capabilities and explicit OP assumptions | Actual principal graph, custom roles, no-shared-admin attestation, cross-scope/impersonation/break-glass tests | **BLOCKED** |
| REL | Monotonic continuity, fencing, crash ambiguity and fail-closed availability | PIS atomic admission; local uncertain-write handling; `FRZ-REL-03/05` | Spanner PREPARED/WITNESS_SEALED/COMMITTED plus conditional Azure seal | Documented service primitives and crash resolver design | Real CAS/append unknown-outcome, rollover, restore, failover and D03-B/C evidence | **BLOCKED** |
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
| Preferred Spanner/Azure pair is not operator-ratified, provisioned or present in production composition | **IN-SCOPE BLOCKER** | Ratify or reject OD-D03-A-02–08; later authorize provisioning separately |
| Actual GCP project, Entra tenant/subscription, administrators and break-glass principals are unknown | **IN-SCOPE BLOCKER** | Inspect and approve a no-shared-admin identity graph before provisioning |
| Azure fixed-block chain, segment rollover and unknown append outcome are proposed adapter semantics, not a native ELCR feature | **IN-SCOPE BLOCKER** | Sandbox concurrency, crash, malformed-tail and rollover qualification |
| RA-GZRS unplanned failover can lose an asynchronously replicated suffix | **RESIDUAL AVAILABILITY RISK** | Preserve quarantine/no automatic failover; test exact recovery runbook |
| Spanner Enterprise Plus `eur3` carries premium multi-region cost; managed autoscaling starts at 1000 PU although manual granular capacity can start at 100 PU | **OPERATOR COST DECISION** | Size and approve budget or reject the pair; do not weaken consistency to reduce cost |
| Current per-process SQLite PIS is not a global Kubernetes first-writer domain | **IN-SCOPE BLOCKER** (D03-B; outside D03-A implementation scope) | Separate D03-B architecture decision and evidence |
| Current PIS/GCA lack externally bound history incarnations and watermarks; no cross-store critical protocol exists | **IN-SCOPE BLOCKER** (D03-C; outside D03-A implementation scope) | Separate D03-C architecture decision and evidence |
| Global `FRZ-*`, `R7-Q`, `PROD-Q` and `IMPL-01` remain open/not qualified | **TRACKED FREEZE DEBT** | Preserve roadmap gates; no promotion from this proposal |
| Backend portability, alternate implementation and RPL-A..H evidence do not exist | **TRACKED FREEZE DEBT** | EBH-5/6-compatible VBN artifacts and qualification |
| Managed-service CAS, witness immutability, IAM, revocation latency and DR behavior have not been exercised | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Vendor documentation plus controlled sandbox/qualification evidence |
| D03-A01–A23, multi-process and Kubernetes tests were not executed | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Future explicitly authorized qualification on exact implementation SHA |

### 16.2 Exact operator choices OD-D03-A-02–08

Each decision is independently accept/reject; silence is not approval.

1. **OD-D03-A-02 — concrete services and regions:** ratify Cloud Spanner
   Enterprise Plus `eur3` as primary and Azure Blob Storage GPv2 Standard
   RA-GZRS, primary `North Europe`, locked immutable append-blob container as
   witness. Record real project, database, tenant, subscription, account and
   container identities in a later provisioning record; none is invented here.
2. **OD-D03-A-03 — trust/failure domains:** ratify separate GCP and Entra
   identity roots, no shared human/machine/super-administrator, a third PIS/GCA
   backup domain, independent two-person break-glass and the residual risk of
   simultaneous compromise/loss of both roots. Also accept OP-TRUST-07 or require
   a separate trusted transition-service decision before implementation.
3. **OD-D03-A-04 — history scope:** ratify environment-level scope as the
   conservative default. Tenant sub-scope is forbidden until complete,
   independently restorable PIS/GCA tenant histories and cross-tenant
   non-interference are proven.
4. **OD-D03-A-05 — IAM:** ratify §6.3, short-lived dual-cloud workload
   federation, custom table/container data roles, Shared Key/static-key/SAS
   prohibition, separate provision/reconcile/revoke/restore/retention roles and
   immutable independent audit export.
5. **OD-D03-A-06 — retention/recovery:** ratify locked container time-based
   retention under the §6.2.1 append-blob mapping, an independent hold, and a
   duration computed only after the readiness §7.5 inputs exist. Ten years and
   `ELCR-ACTIVE` are not adopted. Also ratify the deterministic segment chain,
   no primary restore-in-place, a new authority incarnation in quarantine,
   no automatic Azure unplanned failover, and dual-controlled sealed cutover.
6. **OD-D03-A-07 — freshness/SLO/cost:** ratify no degraded allow, RPO 0 for
   authorization, verdict lifetime ≤5 s, committed-revocation target ≤5 s,
   transition p99 target ≤500 ms from approved EU clusters, safety-preserving
   CREATE RTO target 4 h, and Enterprise Plus/cross-cloud recurring cost subject
   to budget approval and execution measurement.
7. **OD-D03-A-08 — contract/composition:** ratify the vendor-neutral typed ports
   and records in §7, Spanner/Azure adapters only at sanctioned composition, the
   production lifecycle under `ProductionControlPlaneGovernance` as composer,
   and the existing `GovernedProviderInvocationDispatchGate` as the sole
   CREATE-only consumer/enforcer.

Even unanimous OD-D03-A-02–08 ratification does not authorize provisioning or
code. A later explicit implementation/provisioning authority is required.
D03-B global PIS and D03-C continuity/TOCTOU remain independent mandatory gates.

## 17. Authoritative vendor evidence

Documentation inspected 2026-10-18:

### 17.1 Preferred primary — Google Cloud

- [Spanner: TrueTime and external consistency](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency)
  — external consistency and default strong reads.
- [Transactions overview](https://docs.cloud.google.com/spanner/docs/transactions)
  — serializable read-write transactions and strong read semantics.
- [Regional, dual-region, and multi-region configurations](https://docs.cloud.google.com/spanner/docs/instance-configurations)
  — exact `eur3` topology, availability, latency and cost trade-offs.
- [Disaster recovery overview](https://cloud.google.com/spanner/docs/backup/disaster-recovery-overview)
  and [Backups overview](https://docs.cloud.google.com/spanner/docs/backup)
  — PITR, backup, copy and restore capabilities.
- [Spanner IAM](https://docs.cloud.google.com/spanner/docs/iam) and
  [security/audit logging](https://docs.cloud.google.com/spanner/docs/security-privacy-compliance)
  — data/admin roles and audit-log classes.
- [Workload Identity Federation with Kubernetes](https://cloud.google.com/iam/docs/workload-identity-federation-with-kubernetes)
  — projected Kubernetes OIDC identity without service-account keys.
- [Spanner editions](https://docs.cloud.google.com/spanner/docs/editions-overview)
  and [managed autoscaler](https://docs.cloud.google.com/spanner/docs/managed-autoscaler)
  — Enterprise Plus and minimum-capacity/cost implications.

### 17.2 Preferred witness — Microsoft Azure

- [Manage Blob Storage concurrency](https://learn.microsoft.com/en-us/azure/storage/blobs/concurrency-manage)
  — strong consistency.
- [Append Block REST API](https://learn.microsoft.com/en-us/rest/api/storageservices/append-block)
  — exact append-position condition, 412 conflict and required RBAC DataAction.
- [Append blob structure and limits](https://learn.microsoft.com/en-us/rest/api/storageservices/understanding-block-blobs--append-blobs--and-page-blobs)
  — append-only blocks and 50,000-block limit.
- [Container-level WORM policies](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-container-level-worm-policies)
  and [immutable storage overview](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-storage-overview)
  — locked retention, protected appends, legal holds and deletion behavior.
  Re-read 2026-10-19 for §6.2.1, together with
  [Create Or Update Immutability Policy, api-version 2026-04-01](https://learn.microsoft.com/en-us/rest/api/storagerp/blob-containers/create-or-update-immutability-policy?view=rest-storagerp-2026-04-01)
  and [Set Legal Hold, api-version 2026-04-01](https://learn.microsoft.com/en-us/rest/api/storagerp/blob-containers/set-legal-hold?view=rest-storagerp-2026-04-01).
  The rest of §17 was not re-fetched on that date.
- [Azure Storage redundancy](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy)
  and [disaster-recovery guidance](https://learn.microsoft.com/en-us/azure/storage/common/storage-disaster-recovery-guidance)
  — synchronous primary-zone versus asynchronous secondary-region replication
  and possible unplanned-failover data loss.
- [Microsoft Entra Workload Identity Federation](https://learn.microsoft.com/en-us/entra/workload-id/workload-identity-federation)
  — any Kubernetes cluster, including GKE/EKS/on-premises.
- [Blob authorization with Entra ID](https://learn.microsoft.com/en-us/azure/storage/blobs/authorize-access-azure-active-directory)
  and [custom Azure roles](https://learn.microsoft.com/en-us/azure/role-based-access-control/custom-roles)
  — identity-based, least-privilege data access.
- [Blob monitoring and audit scenarios](https://learn.microsoft.com/en-us/azure/storage/blobs/blob-storage-monitoring-scenarios)
  — Activity Log versus data-plane Storage resource logs.

### 17.3 Rejected-pair evidence

- [DynamoDB global tables](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/V2globaltables_HowItWorks.html)
  and [DynamoDB transactions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html)
  — MRSC strong conditional writes but no MRSC transactions; regional-only
  transactional atomicity.
- [S3 consistency model](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html),
  [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html),
  [Object Lock](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html)
  and [directory-bucket append](https://docs.aws.amazon.com/AmazonS3/latest/userguide/directory-buckets-objects-append.html)
  — strong per-key operations and WORM, but no atomic cross-key next-head;
  append is limited to S3 Express One Zone and does not supply Object Lock.

These pages document service behavior only. They do not prove purchased edition,
regional feature compatibility, configured IAM/retention, actual administrator
separation, adapter correctness, effective SLA, latency or recovery behavior.
Those claims remain **OP** or **TEST**.

## 18. Explicit NO-GO conditions

The proposal must be rejected or the deployment must remain quarantined if any
of the following holds:

- no independently protected witness/high-water mark exists;
- primary and witness can both be silently rolled back by one normal
  administrator or backup workflow;
- GCP and Azure ultimately share a person, group, CI principal, federation
  administrator or break-glass account with authority over both histories;
- the witness can restore its production namespace to a lower head;
- Azure Shared Key, SAS, or static secrets are enabled for runtime; the
  time-based policy is unlocked for production use; both protected-append
  properties are true on that policy; the §6.2.1 combined hold test has not
  passed; or the active hold and locked retention cannot be independently
  verified;
- an Azure secondary endpoint, `Last Sync Time`, object listing or unplanned
  failover prefix is treated as a current highest seal;
- append-blob segment rollover, malformed tail or unknown append outcome lacks a
  fail-closed resolver;
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

## 19. Final recommendation and status

**Recommend Cloud Spanner Enterprise Plus `eur3` plus independently governed
Azure Blob Storage RA-GZRS immutable append witness for OD-D03-A-02–08 operator
ratification.** The documented primitives can support ELCR-IIRW when combined
with the exact conditional-seal protocol and truly non-overlapping identity,
backup and break-glass domains described here.

This recommendation does not claim that WORM alone provides currentness, that
RA-GZRS has synchronous regional RPO, or that separate cloud accounts are
independent without the operator-inspected administration graph.

This document contains no production implementation and no qualification
execution. D03-B and D03-C remain mandatory blockers. Recommended task status:

**READY FOR AUDIT — OPERATOR RATIFICATION REQUIRED**

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
