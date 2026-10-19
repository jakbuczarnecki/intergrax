# D03-A — Infrastructure qualification preparation

| Field | Value |
|---|---|
| Status | **INDEPENDENTLY ACCEPTED — SCOPED PREPARATION ONLY** at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`. Not infrastructure ratification. This acceptance-recording delta is **READY FOR AUDIT**. |
| Accepted evidence | `4bbd786ffbd82000ef58b330854a6cce78cd2d9f` |
| Audit basis | Exact-SHA GitHub review of the documentation delta at that SHA. No independent execution of cloud tests. No independent qualification of vendor runtime behavior. No cloud resources created. No approval to incur cloud costs. |
| Protected-append flags | On the time-based immutability policy the two properties remain mutually exclusive. Proposal §6.2.1 sets `allowProtectedAppendWrites=true` on that policy and does not set `allowProtectedAppendWritesAll` true on the same policy. That mapping is `RECOMMENDED TECHNICAL CONFIGURATION — PENDING SANDBOX VERIFICATION`, not a deployed or ratified setting. The combined hold test stays unresolved. This acceptance record does not run that test. |
| Acceptance boundary | **INDEPENDENTLY ACCEPTED — SCOPED PREPARATION ONLY** remains the disposition of `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`. Sentences here that previously said proposal §6.2 still sets both flags are corrected so they match §6.2.1. That correction is a later documentation delta. It is not a new independent acceptance and it does not reopen P1–P6. |
| Date checked | 2026-10-19 |
| Branch | `development` |
| START_HEAD | `343cc5627f6c1e2a71395fd0e3038e06114185bd` |
| Parent gate | [Readiness R1](PROD_Q_R1_R7_Q_D03_A_INFRASTRUCTURE_RATIFICATION_READINESS_R1.md) |
| Direction | [Conditional infrastructure direction](PROD_Q_R1_R7_Q_D03_A_CONDITIONAL_INFRASTRUCTURE_DIRECTION_OPERATOR_DECISION.md) — unchanged |
| Design source | [Trusted anchor proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) — not copied |
| Production delta | **0** |
| Provisioning | **not authorized** |
| FRZ promotion | **0** |
| Test execution | P1–P6 acceptance: `pytest = 0`. §P7 assessment regression: see readiness §19. Cloud calls = **0** |
| Dependent gate disposition | **INDEPENDENTLY ACCEPTED — SCOPED READINESS ONLY** at `7d8b42e45b5dfe2dd08074f5d0da54419074a906` (§P7). Cursor reconciliation: **READY FOR AUDIT — CLASS-B ADMISSION DECISION RECONCILIATION**. Not infrastructure qualification. Class-B admission: **NOT AUTHORIZED — OPERATOR DECISIONS MISSING** (§P7.H). Cloud execution: **NOT AUTHORIZED**. |

This appendix prepares a later sandbox decision. It does not create accounts, resources, credentials, or adapters. It does not set a spending maximum, a retention duration, or a monthly total. A second, explicit operator authorization is required before any future provisioning phase.

Preferred pair, unchanged: Google Cloud Spanner Enterprise Plus, base multi-region `eur3`; Azure Blob Storage general-purpose v2 Standard, RA-GZRS, North Europe, immutable append-blob witness.

Semantic owner: `ProductionControlPlaneGovernance`. Sanctioned composition: `compose_history_continuity_authority`. Platform contract: `HistoryContinuityAuthority`. Current adapter: `local_deterministic`. Governed mutation authorization: `ControlPlaneMutationAuthorizationBoundary`. Sole physical CREATE gate: `GovernedProviderInvocationDispatchGate`. PIS and GCA stay independent. No second authority, composition owner, CREATE gate, PIS, or GCA is introduced.

LOCAL-01 stays **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `302e7fa74b745d10f9552df0a6eb9c702d5ce033`. PLATFORM-POLICY-CONFIG-01 stays **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `02e48d9fa61d116346da613271cc5e8b77e08cc6`. Neither is reopened.

Static symbol check at START_HEAD, scoped to the known paths:

| Symbol | Path | Result |
|---|---|---|
| `ProductionControlPlaneGovernance` | `intergrax/applications/_shared/production_control_plane_governance_wiring.py` | class present |
| `compose_history_continuity_authority` | `intergrax/applications/_shared/history_continuity_composition.py` | function present |
| `HistoryContinuityAuthority` | `intergrax/contracts/governance/history_continuity.py` | protocol present |
| `local_deterministic` | same contract, `LOCAL_DETERMINISTIC` | present |
| `HistoryContinuityRetentionPolicy` | same contract | present |
| `ControlPlaneMutationAuthorizationBoundary` | `intergrax/runtime/governance/local_history_continuity.py` | used by the local adapter |
| `GovernedProviderInvocationDispatchGate` | `applications/governed_contractor_application/host/provider_invocation_lifecycle.py` | class present |
| `ProviderInvocationStore` | `intergrax/contracts/provider_invocation_store.py` | contract present |

This check does not re-audit the accepted implementation SHAs and does not prove cloud behavior.

`STOP — ARCHITECTURE DECISION REQUIRED` was **not** reached. The selected pair still documents the required continuity primitives. The previous proposal sentence that set both Azure flags true is corrected in proposal §6.2.1. P1 records the remaining combined-hold test. It does not select another provider, edition, or region.

## P1 — Verified service feasibility

Date checked for every row: **2026-10-19**. No cloud account was authenticated.

| Fact | Documented | Unverified | Qualification consequence |
|---|---|---|---|
| Enterprise Plus includes multi-region configurations. The editions page states an availability SLA of up to 99.999% for Enterprise Plus. The configurations page states 99.999% availability for multi-region configurations. | [Editions](https://docs.cloud.google.com/spanner/docs/editions-overview); [instance configurations](https://docs.cloud.google.com/spanner/docs/instance-configurations) | A purchased SLA for this platform. Optional read-only replicas are listed for `eur3` and are outside the accepted base configuration. | Do not add `us-central1` or `us-east4` read-only replicas inside this direction. Do not treat the vendor percentage as a measured cross-cloud SLO. |
| Base `eur3` resource location is `eu`. Read-write regions: Belgium `europe-west1` (leader, two replicas) and Netherlands `europe-west4` (two replicas). Witness region: Finland `europe-north1`. | [Instance configurations](https://docs.cloud.google.com/spanner/docs/instance-configurations) | Live capacity in the operator's project. | Keep the base configuration. A different region is an architecture stop. |
| Granular instances remain supported in all editions. The editions FAQ states a minimum of 100 processing units or one node (1000 processing units). Below 1000 processing units, capacity is set in multiples of 100. 1000 processing units equal one node. | [Editions FAQ](https://docs.cloud.google.com/spanner/docs/editions-overview); [compute capacity](https://docs.cloud.google.com/spanner/docs/compute-capacity) | Whether the create API accepts 100 processing units specifically for `eur3`. No create was attempted. | Manual capacity may be planned from 100 processing units. Confirm that floor only in a later authorized create. |
| Managed autoscaler minimum is 1000 processing units or 1 node. Enabling it scales a smaller instance up to that minimum. The minimum must be at least 10% of the maximum. | [Managed autoscaler](https://docs.cloud.google.com/spanner/docs/managed-autoscaler) | Cost of that automatic scale-up for `eur3`. | Do not enable managed autoscaler in the sandbox unless the operator has accepted at least 1000 processing units. |
| Default reads are strong. Unspecified or serializable isolation provides external consistency, which the TrueTime page states is stronger than linearizability of single objects and stronger than serializability alone. Stale reads exist and are not the default. | [TrueTime and external consistency](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency) | Adapter enforcement that admission never requests a stale read. | Stale, bounded, or exact-staleness reads stay forbidden for currentness. That prohibition is untested. |
| Backups are transactionally and externally consistent at `versionTime`, replicated with the instance, and retainable for up to one year. An instance that still has backups cannot be deleted. A backup cannot outlive its instance. | [Backups](https://docs.cloud.google.com/spanner/docs/backup) | The sentence "restore always creates a new database and never overwrites the source in place" was not on the fetched backup overview. The previously used restore URL returned HTTP 404 in this pass. | D03-A21 must show a distinct database and a new authority incarnation. Restore-in-place stays forbidden by readiness §7.3. |
| Point-in-time recovery version retention must be between 1 hour and 7 days. Enterprise Plus lists 7-day PITR. | [PITR](https://docs.cloud.google.com/spanner/docs/use-pitr); [editions](https://docs.cloud.google.com/spanner/docs/editions-overview) | A PITR window longer than 7 days. | PITR cannot satisfy a longer `ANCHOR_PRIMARY_MAX_RESTORABLE_AGE` by itself. Backups, up to one year, are the longer documented mechanism. No duration is adopted. |
| Any provisioned compute is billed for a minimum of one hour, then prorated. A base multi-region node price includes its replicas. Replication and outbound reads are separate charges. Inbound and intra-region transfer have no bandwidth charge on the pricing page. | [Pricing](https://cloud.google.com/spanner/pricing) | The numeric Enterprise Plus `eur3` node-hour, storage, backup, replication, and egress rates. They were not in the static page text. | Those rates stay calculator-only. No monthly total. |
| Standard general-purpose v2 supports GZRS and RA-GZRS. The account-overview footnote says the available redundancy options also vary by region. North Europe has three availability zones and is paired with West Europe. The paired secondary is chosen by Azure and cannot be operator-selected. | [Redundancy](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy); [storage account overview](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-overview); [region list](https://learn.microsoft.com/en-us/azure/reliability/regions-list) | A create in the operator's subscription. No subscription was used. | The public retail catalog publishes Hot RA-GZRS stored-data meters for `northeurope`, which is evidence the SKU is offered for that region. Subscription eligibility remains deployment evidence. |
| RA-GZRS is synchronous zone redundancy in the primary region plus asynchronous copy to the secondary, with read access to the secondary endpoint. Geo-replication can lose writes that have not reached the secondary. Last Sync Time is the latest time at which primary writes are guaranteed to have reached the secondary. After an unplanned failover the account is left on LRS until geo-redundancy is rebuilt. | [Redundancy](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy); [reliability](https://learn.microsoft.com/en-us/azure/reliability/reliability-storage-blob); [disaster recovery](https://learn.microsoft.com/en-us/azure/storage/common/storage-disaster-recovery-guidance) | An RPO number for append blobs. Geo Priority Replication states an RPO of less than or equal to 15 minutes for **Block Blobs** only. | The secondary endpoint and Last Sync Time stay diagnostic. They do not authorize CREATE. The 15-minute figure is not used. |
| Container-level WORM is available on general-purpose v2. A time-based policy and a legal hold can be set together. Append Block is prohibited while a time-based policy and/or a legal hold is in effect, except when `allowProtectedAppendWrites` or `allowProtectedAppendWritesAll` is enabled. Existing blocks cannot be modified or deleted. The two properties are mutually exclusive. After the policy is locked, neither property can be changed, the interval cannot be shortened, and a container-level locked policy allows at most five extensions. The minimum interval is one day and the maximum is 146,000 days. For protected append blobs, effective retention is measured from the last modification time. | [Immutable storage overview](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-storage-overview); [container WORM](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-container-level-worm-policies); [immutability policy REST, api-version 2026-04-01](https://learn.microsoft.com/en-us/rest/api/storagerp/blob-containers/create-or-update-immutability-policy?view=rest-storagerp-2026-04-01) | Whether Append Block succeeds under a time-based policy with `allowProtectedAppendWrites=true` when a container legal hold has `allowProtectedAppendWritesAll` omitted or false, and whether that hold property may be true while the time-based policy keeps only `allowProtectedAppendWrites`. Proposal §6.2.1 records the test. No sandbox has run it. | The time-based policy must not set both properties. The append-blob property on that policy is the documented candidate and remains pending sandbox verification. The combined hold choice stays unresolved. If a later sandbox shows that append blobs cannot keep protected appends under both controls, stop with `STOP — ARCHITECTURE DECISION REQUIRED`. |
| Immutable storage has no extra capacity charge. It is incompatible with blob point-in-time restore and last-access tracking. Policy changes made after Last Sync Time are not guaranteed to be on the secondary until redone after failover. A container with a WORM policy must be empty before deletion. Account deletion fails while a protected container still has blobs. The fetched page also says unlocked policies do not provide delete protection; that note is truncated in the capture and is not treated as a cleanup guarantee. | Same immutable-storage pages | Whether an unlocked policy can be removed immediately in the operator's account. | Do not enable blob PITR, last-access tracking, NFS, or SFTP on the witness account. Do not assume a locked blob can be deleted on the same day. |
| Archive tier is not supported for RA-GZRS. Append blobs are copied by geo-replication together with the other blob types. | [Redundancy](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy) | Hot-tier request prices for non-hierarchical RA-GZRS. See P2. | The witness design uses Hot append blobs. Archive is not a substitute. |
| Non-billable planning surfaces used here: public documentation, the Spanner pricing page, and the unauthenticated Azure Retail Prices API. Creating a Spanner instance, database, or backup, or an Azure storage account, would be billable and was not done. | Pricing page and retail API URL in P2 | Organization policy and quota behavior inside a future project. | Planning does not authorize a create. |

The 50,000-block append-blob limit stated in the proposal was **not re-fetched** on 2026-10-19. It stays prior proposal evidence. The segment-rollover rule is unchanged and untested.

## P2 — Cost model and proposed sandbox budget

No monthly total is stated. Volumes, processing units, retention, and the spending maximum are operator inputs. A missing input is not zero.

### 1. Verified public prices

Azure Retail Prices API, unauthenticated, `BillingCurrency=USD`, `CustomerEntityType=Retail`, `armRegionName=northeurope`, retrieved 2026-10-19. These are list prices, not a quote, not tax, and not an agreement price.

| Product | SKU | Meter | Tier minimum | Unit price | Unit | Meter id | Effective |
|---|---|---|---|---|---|---|---|
| General Block Blob v2 | Hot RA-GZRS | Hot RA-GZRS Data Stored | 0 | 0.05325 USD | 1 GB-month | `877eabab-77f8-4b1d-97ad-ea5f8331cd47` | 2020-07-01 |
| General Block Blob v2 | Hot RA-GZRS | Hot RA-GZRS Data Stored | 51200 | 0.05112 USD | 1 GB-month | same | 2020-07-01 |
| General Block Blob v2 | Hot RA-GZRS | Hot RA-GZRS Data Stored | 512000 | 0.04899 USD | 1 GB-month | same | 2020-07-01 |
| Blob Features | Data Geo Priority Replication RA-GZRS | Data Geo Priority Replication RA-GZRS Data Replicated | 0 | 0.015 USD | 1 GB | `3dd80df1-1132-5bd3-ad3f-b1214a2ec204` | 2025-10-01 |

Source query: `https://prices.azure.com/api/retail/prices` filtered to North Europe, product `General Block Blob v2`, SKU `Hot RA-GZRS`, and the earlier RA-GZRS blob query that returned the geo-priority meter. Immutable storage itself adds no extra capacity charge: [immutable storage pricing note](https://learn.microsoft.com/en-us/azure/storage/blobs/immutable-storage-overview).

The same queries did **not** return `Hot RA-GZRS Write Operations`, `Hot RA-GZRS Read Operations`, or `Hot RA-GZRS Other Operations` for non-hierarchical General Block Blob v2. `Hot GZRS Write Operations` at 0.1165 USD per 10,000 was returned for a different SKU and is **not** used as the RA-GZRS rate.

### 2. Calculator-only rates

Spanner Enterprise Plus `eur3` compute, database storage, backup storage, cross-region replication, and network egress. The [pricing page](https://cloud.google.com/spanner/pricing) lists `eur3` in the multi-region selector and states the billing rules, including the one-hour compute minimum. The numeric amounts were not in the static page text on 2026-10-19.

### 3. Assumptions requiring operator input

Processing units. Hours the instance exists. Average database GiB. Backup GiB and retention. Replicated GiB. Client region and therefore egress. Transitions per test. Append, range-read, and property-read counts. Retained witness GiB. Whether ordinary RA-GZRS geo-replication traffic is included in the stored-data price or billed again. Audit-log GiB and retention days. Private Link, Private Service Connect, VPN, or peering, if required. Support plan. Tax. Discounts. Labor. The five readiness §7.5 retention inputs. The proposal §6.2.1 combined hold result. Whether a sandbox policy is locked.

### 4. Estimated ranges

No range is computed. The stored-data unit prices above cannot become a range until retained GiB and the retention interval are supplied. Spanner cannot become a range until the calculator rates and the processing-unit choice are supplied.

### 5. Unknown charges that prevent a defensible total

- Enterprise Plus `eur3` node-hour, storage, backup, replication, and egress amounts.
- Hot RA-GZRS write, read, and other operation rates for the non-hierarchical account.
- Ordinary geo-replication traffic, as distinct from the geo-priority meter.
- Google and Azure charges for the same cross-cloud read-back.
- Cloud Audit Logs, Azure Activity logs, and storage resource-log retention.
- Private networking, if chosen.
- Residual bytes that cannot be deleted while a lock or hold remains.
- Cost of rebuilding geo-redundancy after an unplanned failover leaves the account on LRS.
- Tax, support, and commitment discounts.

Formulas in readiness §8.3 stay the input structure. They are not evaluated.

### Sandbox budget proposal

The operator fills every blank. This package does not.

| Field | Value |
|---|---|
| Currency | unset |
| Hard maximum spending decision | **unset — operator approval required later** |
| Included clouds | Spanner Enterprise Plus base `eur3`; Azure GPv2 Standard Hot RA-GZRS, North Europe |
| Included activities | create, qualify D03-A01–A23, recover, delete what deletion rules allow |
| Excluded | production activation, CREATE-gate wiring, committed-use discounts, optional read-only replicas, a second region choice |
| Planned processing units | unset |
| Planned exposure window | unset |
| Who may stop spend | unset |

Controls, distinguished:

| Control | What it does | What it does not do |
|---|---|---|
| Google Cloud alerts-only budget | Sends threshold alerts. Source: [budgets](https://cloud.google.com/billing/docs/how-to/budgets), 2026-10-19. The page states that an alerts-only budget does not automatically cap usage or spending. | It is not a spending cap. A spend-cap budget exists only if supported for the service. Spanner support was not verified. |
| Azure Cost Management budget | Sends notifications. Source: [create budgets](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets), 2026-10-19. The page states that resources are not affected and consumption is not stopped. Evaluation lags by hours. | It is not a spending cap. An action group can start automation only if the operator builds and proves that automation. |
| Quotas | Can limit how many resources an account may create. Not configured here. | A quota is not a currency cap. |
| Planned runtime limits | The future authorization can name processing units, test count, and an exposure window. | A sentence in this document does not stop a running cloud resource. |
| Manual shutdown | An authorized operator deletes resources that the service allows to be deleted. | It does not delete a locked non-empty witness before retention and hold rules allow deletion. |
| Actual enforceable controls | None exist for this work. No account, budget, quota, or resource was created. | Billing alerts must not be described as the hard maximum. |

## P3 — Trust-domain and IAM preparation

Role classes only. No principal, project id, subscription id, group, or role binding is invented. A diagram is not configured independence.

```text
Spanner primary administrators
Azure witness custodians
PIS/GCA backup and restore administrators
Runtime reader
Transition writer
Provisioning principal
Recovery/reconciliation principal
Independent auditor
Break-glass, one domain at a time
```

Forbidden cross-domain entitlements, all still **unproven**:

| Holder | Forbidden |
|---|---|
| Spanner primary administrators | Azure RBAC, Entra privileged roles, witness retention, witness hold, PIS/GCA restore |
| Azure witness custodians | Spanner IAM, Spanner backup or restore, PIS/GCA restore |
| PIS/GCA backup and restore administrators | Any Spanner or witness data or admin role, including a read used to authorize CREATE |
| Runtime reader | Writes, restore, IAM, retention, other scopes |
| Transition writer | Initialize, reconcile, revoke, delete, overwrite, IAM, retention, backup, restore, the other cloud's administration |
| Provisioning principal | Ordinary transition and provider dispatch |
| Recovery/reconciliation principal | Rewriting prior rows, deleting seals, physical CREATE |
| Independent auditor | Mutation, retention changes, CREATE admission |
| Break-glass | A credential, approver, or group that is valid in both clouds |
| Any runtime identity | Static key, Shared Key, SAS, or a broad administrator role |

Negative IAM tests, none executed: D03-A22 and D03-A23, plus absence of the privilege paths in readiness §4.4. Static credentials, broad runtime administration, shared cross-domain administrators, and a service identity that can bypass `GovernedProviderInvocationDispatchGate` stay forbidden. Workload federation is not configured.

## P4 — Test plan D03-A01–A23

Canonical identities and acceptance outcomes are readiness §10. No row was executed. No new CREATE path is created. A local double is not acceptance.

Priority P0 must fail closed before any positive CREATE claim. P1 covers history mismatch and reconciliation. P2 is the positive path and the existing non-CREATE operations. Ordering does not rename the tests.

| ID | Pri | Preconditions | Infrastructure prerequisites | Action | Expected outcome | Logs / artifacts | Pass/fail oracle | Role | Cost driver | Cleanup |
|---|---|---|---|---|---|---|---|---|---|---|
| D03-A09 | P0 | One scope, one expected revision | Spanner sandbox, serializable transaction | Two concurrent compare-and-advance calls | One winner; loser gets a typed conflict or stale result | Commit records | Exactly one successor row | Transition writer | Spanner compute and storage | None beyond the sandbox |
| D03-A08 | P0 | A newer fence exists | Two processes, later Kubernetes | Stale replica presents an old fence | `DENY/FENCE_STALE`; zero provider calls | Deny event | Newer fence remains the only writer | Transition writer | Negligible if denied before cloud writes | None |
| D03-A17 | P0 | Expected append offset | Azure append blob, one protected-append flag | Two appends at the same offset | One success; loser HTTP 412 | Blob content length and 412 | One block at that offset | Transition writer | Azure write operations | Retained block if a lock or hold is active |
| D03-A18 | P0 | Append may have committed | Azure, fault-injecting proxy | Drop the success response | Same seal recovered; no second seal | Range read | Bytes equal the prepared digest | Transition writer | Read-back operations and egress | Same transition id only |
| D03-A10 | P0 | Transition left after `PREPARED`, after append, or before `COMMITTED` | Both clouds | Resolve that same transition id | CREATE denied until primary and witness match; no new id | Both records | Match, or quarantine | Recovery principal | Read-back on both clouds | Do not mint a second transition |
| D03-A19 | P0 | Segment near the close point | Azure chain | Crash around close and successor; add a malformed tail | CREATE denied until one linked successor is verified; no truncation | Chain record | Link verified, or quarantine | Recovery principal | Extra appends | Quarantine the scope; do not delete sealed blocks |
| D03-A07 | P0 | A backup or a witness prefix exists | Backup plus witness | Present the backup or prefix as current | Deny and quarantine | Mismatch record | New incarnation is not current | Recovery principal | Backup storage and restore compute | Quarantined database is not the current endpoint |
| D03-A20 | P0 | RA-GZRS account | Controlled failover | Unplanned failover exposes a prefix | Mismatch and quarantine; no automatic CREATE | Last Sync Time, prefix length | Prefix is not the head | Witness custodian and recovery principal | Failover and later re-replication | Account may be left on LRS; document that liability |
| D03-A21 | P0 | Spanner backup | GCP sandbox | Restore to a new database | New incarnation stays quarantined until the highest seal and dual control | New database id | Old endpoint remains current until a recovery seal | Spanner administrator | Restore compute and backup storage | Delete the quarantined database only through the admin path |
| D03-A22 | P0 | Runtime identities bound | Both clouds | Runtime calls init, reconcile, revoke, delete, retention, restore, or another scope | Deny and an audit event | Cloud audit | No mutation | Independent auditor | Audit retention | None |
| D03-A23 | P0 | Separate admin identities | Real tenants, not a mock | Each administrator attempts the other domain, including break-glass | No effective entitlement and no impersonation chain | IAM export and failed call | Export plus failed call | Independent auditor | Export and denied calls | Remove temporary test identities |
| D03-A13 | P0 | Two scopes or tenants | Both anchors | Present another environment or tenant | `DENY/SCOPE_MISMATCH`, `TENANT_MISMATCH`, or `ENVIRONMENT_MISMATCH` | Deny | Foreign seal is not the head | Runtime reader | One denied read | None |
| D03-A14 | P0 | Healthy head | Partition either cloud | Make primary or witness unreadable | `DENY/AUTHORITY_UNAVAILABLE`; no cached allow | Outage and deny | Zero provider calls | Runtime reader | None if denied | Restore connectivity |
| D03-A16 | P0 | Strict routes present | Current gate, later the sandbox | Attempt provider CREATE | No route except the existing dispatch gate | Reachability record | No second CREATE path | Independent auditor | None for the static check | None |
| D03-A02 | P1 | PIS or GCA empty or unproven | Resolver plus stores | CREATE admission | `DENY/UNPROVEN_INITIALIZATION`; zero provider calls | Deny | Zero execute callback | Runtime reader | None | None |
| D03-A03 | P1 | Older PIS media | PIS restore drill | Admit CREATE | `DENY/PIS_INCARNATION_MISMATCH` or stale watermark | Deny | External head unchanged | PIS/GCA restore administrator | PIS restore cost, outside this pair | Restore the drill copy, not the anchor |
| D03-A04 | P1 | Older GCA media | GCA restore drill | Admit CREATE | `DENY/GCA_INCARNATION_MISMATCH` or stale watermark | Deny | External head unchanged | PIS/GCA restore administrator | GCA restore cost | Same |
| D03-A05 | P1 | PIS and GCA agree with each other and disagree with the anchor | Both stores | Admit CREATE | Deny against the higher external head | Deny | Local agreement is not sufficient | Runtime reader | Reads | None |
| D03-A06 | P1 | Pod-local proof below the witness | Restarted process | Admit CREATE | `DENY/CONTINUITY_REVISION_STALE` or `FENCE_STALE` | Deny | Local proof is below the witness | Runtime reader | None | None |
| D03-A11 | P1 | Legacy PIS-only CREATE | Contract path | Admit without anchor evidence | `DENY/LEGACY_UNRESOLVED` | Deny | No heuristic match | Runtime reader | None | None |
| D03-A12 | P1 | One exact scope and approved evidence | Governed workflow | Reconcile | Eligible only for that scope, a new revision or epoch, and a sealed record | Dual-controlled seal | Prior rows not rewritten | Recovery principal | One sealed transition | None |
| D03-A01 | P2 | PIS, GCA, head, and highest seal agree | Future integrated sandbox | Request a CREATE verdict | `ELIGIBLE` at D03-A verdict level only | Verdict | Binds scope, revision, fence, and both incarnations; no provider execute from this predicate | Runtime reader | One strong read and one witness read | Verdict expires; it is not physical CREATE |
| D03-A15 | P2 | Existing accept and cancel paths | Current behavior | Exercise `submit_quote_acceptance` and `cancel_work` | No CREATE-only history predicate | Current admission results | Existing semantics remain | Independent auditor | None for the current inspection | None |

Physical CREATE still requires D03-B and D03-C. D03-A01 does not authorize it.

## P5 — Sandbox lifecycle and cleanup

This document is not the provisioning authorization. The future phase starts only after a second explicit operator authorization that names the spending maximum, the operators, the exposure window, and the five §7.5 inputs still required before a production duration. G1 below is that permission. It is not granted here.

Future inventory, not created:

1. One GCP project and one Spanner instance, Enterprise Plus, base `eur3`, manual processing units chosen by the operator, one database, backups only if a test requires them.
2. One Azure subscription, resource group, and general-purpose v2 Standard account, Hot, RA-GZRS, primary North Europe, shared key disabled, blob PITR and last-access tracking disabled.
3. One witness container, one unlocked immutability policy with exactly one protected-append flag, and a hold only if the operator separately accepts the deletion consequence.
4. Audit export destinations with a retention choice written down before use.
5. Private networking only if G5 selects it.
6. No production identity, no CREATE-gate change, and no adapter implementation in this preparation.

Deployment order: record the identity classes; set alerts and quotas knowing they are not caps; confirm the Spanner edition and `eur3` before any backup; confirm the Azure redundancy and region before any blob; prove an unlocked protected append; lock only after the operator accepts the residual interval; run P0 before P2; do not wire CREATE.

Evidence: vendor operation ids, IAM deny events, blob range reads, Spanner commit timestamps, and a resource list at start and end. No secrets in the evidence.

Maximum authorized exposure window: **unset**.

Shutdown: stop tests; delete the Spanner database and then backups, because an instance with backups cannot be deleted; delete unlocked Azure objects only where the service allows; do not treat a locked append blob as immediately deletable. Effective retention runs from the last append. A hold blocks deletion until cleared by the witness custodians under dual control. A locked interval cannot be shortened.

Residual cost: stored witness GiB at the verified Hot RA-GZRS rates until deletion succeeds, plus any Spanner hour already started, plus logs and any geo-replication meter. The amount is unknown until GiB and time are known.

Proof of removal is a later resource listing that shows the project and subscription empty, or a written list of what remains and why it cannot yet be deleted. That proof does not exist today.

## P6 — Outstanding operator decision sheet

Unknowns are preserved. No default is substituted.

| Id | Decision | Evidence available | Missing input | Responsible authority | Requested disposition | If rejected |
|---|---|---|---|---|---|---|
| G1 | Permission to provision, test, and incur cost | This preparation and the conditional direction. No resource exists. | Explicit authorization, named operators, and the hard maximum in P2 | Operator | Grant a later sandbox authorization, or refuse it | No cloud test. D03-A stays blocked. |
| G2 | Actual independent administrative domains | Role classes in P3 and readiness §4.3. OP-TRUST-07 remains accepted residual risk. | Expanded IAM, Entra, Kubernetes, federation, and break-glass exports | Operator | Confirm non-overlapping roots, or record that the graph fails | Independence is unproven. Do not claim it. |
| G3 | IAM and federation | Vendor pages say separate IAM and workload federation exist. | Custom roles, audiences, token lifetimes, Shared Key disabled, negative tests | Operator | Approve the role classes for a later binding | No federation and no static-credential substitute. |
| G4 | OD-D03-A-06 retention | Readiness §7.5 rule. Five inputs remain unknown. Azure documents a 1-day minimum, a 146,000-day maximum, last-append effective retention, and mutually exclusive protected-append properties on the time-based policy. Proposal §6.2.1 names the append-blob property and leaves the combined hold test unresolved. | `PIS_MAX_RESTORABLE_AGE`, `GCA_MAX_RESTORABLE_AGE`, `ANCHOR_PRIMARY_MAX_RESTORABLE_AGE`, `INVESTIGATION_WINDOW`, `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM` or an explicit declaration that none applies, and the §6.2.1 combined hold result | Operator | Supply the five inputs. Run the hold test only under a later sandbox authorization. Do not adopt ten years here. | No production duration. Do not weaken LOCAL-01. |
| G5 | OD-D03-A-07 budget, capacity, connectivity, and targets | Meter structure, one-hour minimum, and the Azure stored-data prices in P2. Readiness §8.4 targets stay unadopted. | Calculator quotes, processing units, volumes, private-path choice, and an accept or reject for each §8.4 target | Operator | Record quotes and dispositions without weakening fail-closed behavior | No budget acceptance. Do not change edition or region to obtain a price. |
| G6 | Sandbox duration, scope, operators, and cleanup | P5 deletion limits | Window, scope, named operators, cleanup approvers, and whether the sandbox policy will be locked | Operator | Authorize a bounded sandbox or refuse it | No sandbox. |
| G7 | Dependency-order preservation | Roadmap order and readiness §15 graph | None for the order itself | Operator | Keep LOCAL-01 and PLATFORM-POLICY-CONFIG-01 closed, keep D03-B and D03-C blocked, and keep cloud qualification after this preparation | Reordering or a second CREATE path is an architecture stop. |

## Ownership and dependency graph

Unchanged:

```text
OD-D03-A-01 class, ratified
  -> OD-D03-A-02 conditional preferred pair, not full ratification
       -> LOCAL-01, closed scoped
       -> readiness policy consolidation
            -> PLATFORM-POLICY-CONFIG-01, closed scoped
                 -> this preparation
                      -> future sandbox, not authorized
D03-B and D03-C remain independent and BLOCKED
```

`ProductionControlPlaneGovernance` remains the semantic owner. `compose_history_continuity_authority` remains the sanctioned composition. `GovernedProviderInvocationDispatchGate` remains the only physical CREATE gate. PIS remains the invocation store. GCA remains the execution owner. The witness is not a second PIS, GCA, or current-head owner.

## ECAP Change Delta Audit

| Step | Result |
|---|---|
| A1 | This appendix, a readiness pointer and §18, and one roadmap ledger sentence. No production symbol. |
| A2 | Documentation links only. No import, composition wiring, or configuration key. |
| A3 | No production caller. |
| A4 | Owners unchanged. |
| A5 | Production-effective reachability: **no**. |
| A6 | Families below. No status change. |
| A7 | Historical PASS is not extended. LOCAL-01 and PLATFORM-POLICY-CONFIG-01 are not reopened. |
| A8 | No regression gate. `pytest = 0`. |

### Enterprise audit matrix

| Dimension | State | Classification |
|---|---|---|
| Boundaries | No code change. Future adapter boundary is undesigned in code. | **IN-SCOPE BLOCKER** for a future implementation |
| Communication | No new runtime channel. | **NOT APPLICABLE — WITH EVIDENCE** for this delta |
| Composition | Unchanged. | **NOT APPLICABLE — WITH EVIDENCE** for this delta |
| Ownership | Symbols confirmed at START_HEAD. | **IN-SCOPE BLOCKER** until an implementation is audited |
| Contracts | No contract added. | **IN-SCOPE BLOCKER** for implementation |
| Replaceability | `RPL-A` through `RPL-H` not proven. | **TRACKED FREEZE DEBT** |
| Fail-closed behavior | Required, not executed. | **IN-SCOPE BLOCKER** |
| Tenant isolation | No new tenant path. | Documentation: **NOT APPLICABLE — WITH EVIDENCE**. Future mechanism: **IN-SCOPE BLOCKER** |
| Regression | No test run. | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |

### Applicable FRZ criteria

Checklist status is the current checklist column. This package does not promote any row. `RPL-A` through `RPL-H` are not proven.

Evidence phrases: **documented vendor capability**, **operator-accepted policy**, **missing deployment evidence**, **unexecuted qualification test**, **historical PASS not extended**.

| ID | Checklist | Evidence for this preparation |
|---|---|---|
| FRZ-OWN-01 | OPEN | Operator-accepted policy that one semantic owner remains. Missing deployment evidence. |
| FRZ-OWN-02 | OPEN | Existing contract symbol only. Missing deployment evidence for a cloud adapter. |
| FRZ-OWN-03 | OPEN | Existing composition symbol only. Missing deployment evidence. |
| FRZ-OWN-04 | OPEN | No shadow authority added. Missing deployment evidence that none appears later. |
| FRZ-OWN-05 | OPEN | No duplicated mechanism added. Missing deployment evidence. |
| FRZ-BND-01 | OPEN | Layer model unchanged. Missing deployment evidence for future adapters. |
| FRZ-BND-02 | OPEN | No new dependency. Missing deployment evidence. |
| FRZ-BND-03 | OPEN | No reverse import added. Missing deployment evidence. |
| FRZ-BND-04 | OPEN | No private leakage added. Missing deployment evidence. |
| FRZ-BND-05 | OPEN | Composition owner unchanged. Missing deployment evidence. |
| FRZ-BND-06 | OPEN | No transport adapter added. Missing deployment evidence. |
| FRZ-CTR-01 | OPEN | Consumers unchanged. Missing deployment evidence. |
| FRZ-CTR-02 | OPEN | No provider leakage added. Missing deployment evidence. |
| FRZ-CTR-03 | OPEN | No pseudo-contract added. Missing deployment evidence. |
| FRZ-CTR-04 | OPEN | No new contract classification. Missing deployment evidence. |
| FRZ-CTR-05 | OPEN | Contract responsibility unchanged. Missing deployment evidence. |
| FRZ-CTR-06 | OPEN | No vendor type added to a platform contract. Missing deployment evidence. |
| FRZ-TYP-01 | OPEN | No new semantic type. Missing deployment evidence. |
| FRZ-TYP-02 | OPEN | No new `Any` boundary. Missing deployment evidence. |
| FRZ-TYP-03 | OPEN | No new generic seam. Missing deployment evidence. |
| FRZ-TYP-04 | OPEN | No reflection dispatch added. Missing deployment evidence. |
| FRZ-TYP-05 | OPEN | No string dispatch added. Missing deployment evidence. |
| FRZ-TYP-06 | OPEN | No type-ignore added. Missing deployment evidence. |
| FRZ-PLG-01 | OPEN | No plugin surface added. Missing deployment evidence. |
| FRZ-PLG-02 | OPEN | No external implementation added. Missing deployment evidence. |
| FRZ-PLG-03 | OPEN | Discovery owner unchanged. Missing deployment evidence. |
| FRZ-PLG-04 | OPEN | Selection owner unchanged. Missing deployment evidence. |
| FRZ-PLG-05 | OPEN | No new activation path. Missing deployment evidence. |
| FRZ-PLG-06 | OPEN | No plugin package added. Missing deployment evidence. |
| FRZ-PLG-07 | OPEN | No self-expanding extension added. Missing deployment evidence. |
| FRZ-PLG-08 | OPEN | No dynamic registration added. Missing deployment evidence. |
| FRZ-RPL-01 | OPEN | Not proven. Unexecuted qualification test. |
| FRZ-RPL-02 | OPEN | Not proven. Unexecuted qualification test. |
| FRZ-RPL-03 | OPEN | Not proven. Unexecuted qualification test. |
| FRZ-RPL-04 | OPEN | Not proven. Unexecuted qualification test. |
| FRZ-CFG-01 | PASS | Historical PASS not extended. No production vendor literal added. |
| FRZ-CFG-02 | PASS | Historical PASS not extended. Missing prices were not defaulted. |
| FRZ-CFG-03 | PASS | Historical PASS not extended. |
| FRZ-CFG-04 | PASS | Historical PASS not extended. |
| FRZ-CFG-05 | PASS | Historical PASS not extended. No endpoint or secret added. |
| FRZ-CFG-06 | PASS | Historical PASS not extended. |
| FRZ-CFG-07 | PASS | Historical PASS not extended. |
| FRZ-CFG-08 | PASS | Historical PASS not extended. |
| FRZ-TEN-01 | PASS | Historical PASS not extended. |
| FRZ-TEN-02 | PASS | Historical PASS not extended. |
| FRZ-TEN-03 | PASS | Historical PASS not extended. |
| FRZ-TEN-04 | PASS | Historical PASS not extended. |
| FRZ-TEN-05 | PASS | Historical PASS not extended. |
| FRZ-TEN-06 | PASS | Historical PASS not extended. |
| FRZ-TEN-07 | PASS | Historical PASS not extended. |
| FRZ-TEN-08 | PASS | Historical PASS not extended. Restore rules do not re-prove tenant identity. |
| FRZ-TEN-09 | PASS | Historical PASS not extended. |
| FRZ-TEN-10 | PASS | Historical PASS not extended. |
| FRZ-TEN-11 | PASS | Historical PASS not extended. |
| FRZ-TEN-12 | PASS | Historical PASS not extended. Unexecuted qualification test for this mechanism. |
| FRZ-REL-01 | OPEN | Not exercised. |
| FRZ-REL-02 | OPEN | Not exercised. |
| FRZ-REL-03 | OPEN | Same-transition rule cited. Unexecuted qualification test. |
| FRZ-REL-04 | OPEN | Not exercised. |
| FRZ-REL-05 | OPEN | Unknown-append rule cited. Unexecuted qualification test. |
| FRZ-REL-06 | OPEN | Not exercised. |
| FRZ-REL-07 | OPEN | No automatic resumption cited. Unexecuted qualification test. |
| FRZ-REL-08 | OPEN | Not exercised. |
| FRZ-REL-09 | OPEN | Not exercised. |
| FRZ-REL-10 | OPEN | Not exercised. |
| FRZ-REL-11 | OPEN | Not exercised. |
| FRZ-SEC-01 | OPEN | Tenant denial specified as D03-A13. Unexecuted qualification test. |
| FRZ-SEC-02 | OPEN | No secret reference added. Missing deployment evidence. |
| FRZ-SEC-03 | OPEN | No secret added. Missing deployment evidence. |
| FRZ-SEC-04 | OPEN | Bypass remains forbidden. Unexecuted qualification test. |
| FRZ-SEC-05 | OPEN | Sandbox not created. Missing deployment evidence. |
| FRZ-SEC-06 | OPEN | Provider trust not qualified. Missing deployment evidence. |
| FRZ-SEC-07 | OPEN | Ten years is not adopted as a legal finding. |
| FRZ-SEC-08 | OPEN | No provider data path added. Missing deployment evidence. |
| FRZ-SEC-09 | OPEN | Retention responsibility remains unresolved. |
| FRZ-SEC-10 | OPEN | Vendor residency and encryption controls are documented vendor capability only. Missing deployment evidence. |
| FRZ-PRD-01 | OPEN | This document is not production qualification. |
| FRZ-PRD-02 | OPEN | No provider qualification executed. |
| FRZ-PRD-03 | OPEN | No silent duration or price default. |
| FRZ-PRD-04 | OPEN | No startup path added. |
| FRZ-PRD-05 | OPEN | Cleanup limits documented. Unexecuted qualification test. |
| FRZ-PRD-06 | OPEN | Quarantine rules cited. Unexecuted qualification test. |
| FRZ-PRD-07 | OPEN | Local SQLite is not a production duration. |
| FRZ-PRD-08 | OPEN | No second CREATE path added. Unexecuted qualification test. |
| FRZ-PRD-09 | OPEN | Not exercised. |
| FRZ-PRD-10 | OPEN | Not exercised. |
| FRZ-PRD-11 | OPEN | Capacity is an unset operator input. |
| FRZ-PRD-12 | OPEN | Recovery procedure is required and not written. |

### Findings

| Finding | Class |
|---|---|
| Preparation is not provisioning authority and not infrastructure ratification | **IN-SCOPE BLOCKER** for parent closure |
| No IAM export and no deployed independence | **IN-SCOPE BLOCKER** |
| Five §7.5 retention inputs remain unknown. The mutually exclusive time-based properties are not a substitute duration. The combined hold test remains unrun. | **IN-SCOPE BLOCKER** |
| Spanner unit prices and Azure Hot RA-GZRS operation prices are unknown, so no total and no spending maximum exist | **IN-SCOPE BLOCKER** |
| D03-A01–A23 were not executed | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** |
| Global `FRZ-*` rows stay at the checklist status. Historical PASS is not extended. `RPL-A` through `RPL-H` are not proven. | **TRACKED FREEZE DEBT** |
| RA-GZRS failover can expose a prefix. The preserved rule is quarantine. | **IN-SCOPE BLOCKER** if a design resumes CREATE from that prefix |
| Proposal §6.2 previously set both protected-append flags. §6.2.1 removes that request. The combined hold test is not run. | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED**. Not an architecture stop unless that test disproves protected appends under both controls |

No finding is self-certified as enterprise compliance. D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, and global `FRZ-*` stay open. Dependent infrastructure qualification is not authorized.

P1–P6 at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f` are **INDEPENDENTLY ACCEPTED — SCOPED PREPARATION ONLY**. Cursor status of this acceptance-recording delta: **READY FOR AUDIT**. It does not close the D03-A infrastructure gate.

## P7 — Dependent cloud-qualification admission disposition

| Field | Value |
|---|---|
| Assessed at | Package authorship baseline `8443d41494de9db43ae323db746139bd8fe3919a`; operator acceptance evidence `7d8b42e45b5dfe2dd08074f5d0da54419074a906` |
| Cursor disposition | **READY FOR AUDIT — CLASS-B ADMISSION DECISION RECONCILIATION** |
| Operator acceptance (2026-10-19) | **INDEPENDENTLY ACCEPTED — SCOPED READINESS ONLY** at `7d8b42e45b5dfe2dd08074f5d0da54419074a906`. Exact-SHA GitHub review of §P7 documentation only. Not cloud authorization or infrastructure qualification. |
| Class-B admission (authorization class B — cloud execution) | **NOT AUTHORIZED — OPERATOR DECISIONS MISSING** (G1–G6 ungranted; OD-D03-A-06 and OD-D03-A-07 unresolved; no attributable operator grant with scope, budget, resources, and expiry). A later **separate** explicit operator authorization may be recorded only after those inputs; that record is not implied by this acceptance. |
| Not | CLOSED. Not CLOUD-QUALIFIED. Not provisioning authority. |
| Reused evidence | P1–P6 at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`; readiness R1 §4–§11 and §10; [OFFLINE-PROVIDER-IMPL-01](PROD_Q_R1_R7_Q_D03_A_OFFLINE_PROVIDER_IMPL_01.md) closed-world inventory at `a7ecc5065a0343e768584e818c844d3f96b5f533`; conditional direction unchanged. |
| Production delta | **0** |
| Cloud execution | **NOT AUTHORIZED** |
| new FRZ PASS | **0** |

This section completes the bounded admission assessment for the dependent gate named in readiness §15.4. It does not open a sandbox, adopt durations or prices, or promote any `FRZ-*` row.

### P7.0 Closed-world implementation inventory (START_HEAD)

Semantic and composition authority — unchanged owners; production callers of `compose_history_continuity_authority` remain **none** (callers: D03-A unit tests only). Full symbol table: [OFFLINE-PROVIDER-IMPL-01 § Closed-world inventory](PROD_Q_R1_R7_Q_D03_A_OFFLINE_PROVIDER_IMPL_01.md). Static re-check at START_HEAD:

| Symbol | Path | Authority / notes |
|---|---|---|
| `HistoryContinuityAuthority` | `intergrax/contracts/governance/history_continuity.py` | Contract. Semantic owner: `ProductionControlPlaneGovernance`. |
| `HistoryContinuityActivationPolicy`, `HistoryContinuityProviderSelection`, `HistoryContinuityRetentionPolicy` | same | LOCAL-01 / PLATFORM-POLICY-CONFIG-01 activation; offline selection is separate and fail-closed. |
| `ProductionControlPlaneGovernance` | `intergrax/applications/_shared/production_control_plane_governance_wiring.py` | Sole governance semantic owner for continuity. |
| `compose_history_continuity_authority` | `intergrax/applications/_shared/history_continuity_composition.py` | Sole sanctioned composition. Injects only in-memory primary/witness in production path. |
| `ControlPlaneMutationAuthorizationBoundary` | `intergrax/runtime/governance/control_plane_mutation_authorization.py` | Five continuity capabilities; no second boundary. |
| `GovernedProviderInvocationDispatchGate` | `applications/governed_contractor_application/host/provider_invocation_lifecycle.py` | Sole physical CREATE gate; not on continuity graph. |
| `SpannerPrimaryRegister`, `PrimaryDatabasePort`, `PrimaryTransactionPort` | `intergrax/runtime/governance/continuity_providers/primary_register.py` | Offline authority + tests; not admitted transports/SDK bridges. |
| `AzureAppendWitness`, `WitnessAppendPort` | `intergrax/runtime/governance/continuity_providers/witness_log.py` | Same. |
| `SpannerSdkPrimaryBridge`, `AzureSdkAppendBridge`, `PrimaryWitnessEndpoint` | `spanner_sdk.py`, `azure_sdk.py` | Test-only; `open_*` refuses before SDK/network. |
| `OfflineSpannerAzureHistoryContinuity` | `intergrax/runtime/governance/offline_history_continuity.py` | Offline bundle only; `HistoryContinuityCloudQualification` must stay `UNQUALIFIED`. |

Provider replaceability and administrative trust domains are **policy and offline contract evidence only** until G1–G6 and D03-A01–A23 execute under a later sandbox authorization.

### P7.A Trust-domain and IAM qualification matrix

Builds on readiness §4.3–§4.4 and P3. Maps the six admission domains. A different cloud provider alone is **not** administrative independence.

| Domain | Intended owner | Permitted operations | Forbidden operations | Credential / identity boundary | Delete, rollback, or overwrite evidence? | Required independent negative proof | Qualification status |
|---|---|---|---|---|---|---|---|
| 1. Spanner primary administration | GCP platform team / database administrators under non-overlapping org roots | Instance, database, backup, restore to **new** database; IAM inside Spanner project | Azure RBAC, witness retention/hold, PIS/GCA restore, runtime transition writes | Human or break-glass in GCP only; no Shared Key on witness | Can restore primary into new incarnation; restore-in-place forbidden by §7.3 | D03-A23 cross-domain deny; §4.4 path exports; D03-A21 quarantine | Policy **ACCEPTED**; graph **BLOCKED** — G2 |
| 2. Azure witness administration | Azure custodians under dual control, separate subscription root | Container immutability, legal hold, witness data-plane RBAC | Spanner IAM/restore, PIS/GCA restore, GCP org admin | Entra workload roles separate from GCP; Shared Key disabled | Cannot modify sealed blocks; hold/lock changes are auditable | D03-A23; retention bypass attempt (D03-A07 family); §6.2.1 hold test | Policy **ACCEPTED**; configured WORM **BLOCKED** — G4/G6 |
| 3. Workload runtime identity | Kubernetes federated reader / transition writer per environment | Short-lived federated read and scoped transition write per binding | Init, reconcile, revoke, delete, retention, restore, IAM, other cloud admin | WIF / Entra federation; no static admin keys | Append forward seal only within granted transition scope (OP-TRUST-07 residual) | D03-A22; D03-A16 bypass | Policy **ACCEPTED**; bindings **BLOCKED** — G3 |
| 4. Deployment / IaC authority | CI/CD through sanctioned path only | Deploy reviewed configuration | Standing data-plane write on both clouds; retention changes; mint both runtime identities | Pipeline principal per cloud, not combined | Can replace endpoints if dual control fails — §4.4 path | Config deployer cannot change both trusted endpoints without dual control | Policy **ACCEPTED**; proof **BLOCKED** — G2 |
| 5. Operator authorization | Human operator / governance | Record decisions G1–G7; approve sandbox spend cap; supply §7.5 inputs | Cannot substitute for IAM exports or cloud test results | Out of band from runtime | N/A | Signed decision records vs execution evidence | OD-D03-A-06/07 **UNRESOLVED** |
| 6. Backup, restore, retention override | PIS/GCA backup admins (third domain) + separated retention approvers for witness | PIS/GCA backup/restore per runbook; witness retention extension only under dual control | Any Spanner or Azure anchor role; shorten locked retention; clear operational hold without decommission | Distinct from primary and witness data admins | Restore older PIS/GCA must deny CREATE (D03-A03–A05) | D03-A03–A05, D03-A07; decommission procedure | Policy **ACCEPTED**; drills **BLOCKED** — class B |

### P7.B OD-D03-A-06 — operator decision block (single round)

Canonical rule: readiness §7.5.3. Local non-weakening: `HistoryContinuityRetentionPolicy` (`retention_seconds` ≤ `ceiling_seconds`, `legal_hold` true; no lowering duration, ceiling, or hold) — **DOCUMENTED**, not a production day count.

**Operator must supply (no defaults):**

| Input | Requires explicit operator/legal input? | Technical lower bound derivable without operator? |
|---|---|---|
| `PIS_MAX_RESTORABLE_AGE` | Yes — production PIS restore window | Must be ≥ 0 once declared; local SQLite is **not** a source |
| `GCA_MAX_RESTORABLE_AGE` | Yes | Same |
| `ANCHOR_PRIMARY_MAX_RESTORABLE_AGE` | Yes | Spanner backup ≤ 1 year and PITR ≤ 7 days are **vendor capability bounds**, not adopted policy |
| `INVESTIGATION_WINDOW` | Yes | Must be added to `longest_restorable_age` in the inequality |
| `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM` | Yes, or explicit declaration that none applies with attributable source | If declared, `required_time_based_retention` must be ≥ that minimum |
| Proposal §6.2.1 combined hold | Yes — disposition after sandbox | Mutually exclusive time-based flags documented in P1 |

**Safety relationships (invariants, not numbers):** `required_time_based_retention > max(PIS, GCA, ANCHOR_PRIMARY) + INVESTIGATION_WINDOW`; if legal minimum applies, also `≥ REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM`. Operational preservation hold stays for active environment life. Weakening any LOCAL-01 / PLATFORM-POLICY check is forbidden.

**Weakens rollback protection if:** duration below the inequality; time-based expiry without hold while environment active; treating vendor default as production policy; resuming CREATE from geo-replicated prefix or restored backup without quarantine (§7.3).

**Requires real Azure validation:** locked time-based policy with `allowProtectedAppendWrites=true` only; optional legal-purpose hold; protected append under hold (§6.2.1 test); effective retention from last append; deletion blocked while locked.

**Restoration windows vs continuity:** PIS/GCA restore windows must not exceed witness-readable chain needed to detect stale head; primary restore always new incarnation until witness match + dual-controlled recovery seal.

**If unresolved:** OD-D03-A-06 stays open; parent D03-A stays **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**; no production duration in code or config.

### P7.C OD-D03-A-07 — capacity, budget, and SLO (bounded model)

**Already evidenced (documentation):** `eur3` topology, Enterprise Plus edition, RA-GZRS North Europe, meter structure, one-hour Spanner minimum, Azure Hot RA-GZRS stored-data list prices in P2 (dated 2026-10-19), safety invariants §8.1 and Authorization RPO 0 meaning, formulas §8.3.

**Missing inputs (typed, unresolved):** peak writes/sec; historical growth; `processing_units`; transitions/month; `retained_gib`; cluster region; private-path choice; Spanner calculator rates; Azure RA-GZRS write/read/other rates; audit GiB; hard spending maximum (P2 sandbox budget).

**Relationships (not evaluated):** readiness §8.3 formulas; §8.4 targets remain **OPERATOR CONFIRMATION REQUIRED** — achieved SLO **cannot** be claimed before measurement.

**Recovery-related capacity:** DR exercises add restore compute, backup storage, witness read-back, and labor — unpriced until G5 accepts scope.

**Consequence if unresolved:** no budget acceptance, no sandbox G1 grant, no capacity admission; fail-closed behavior unchanged.

### P7.D Recovery, adversarial qualification, and G1–G7

Canonical test catalogue: readiness §10 and P4. Supplemental mapping columns (all rows **unexecuted** for cloud acceptance):

| Gate / area | Invariant examined | Environment | Offline partial proof? | Class-B cloud required? | Evidence stage |
|---|---|---|---|---|---|
| G1 | Provisioning permission and spend cap | Operator record | N/A | No — decision only | Operator before sandbox |
| G2 | Administrative independence | Real IAM exports | No | Yes — graph inspection | Sandbox prep + D03-A23 |
| G3 | Federation and deny runtime admin | Dual cloud IAM | Policy tests only (PLATFORM-POLICY) | Yes — D03-A22 | Sandbox |
| G4 | Retention duration and hold | Azure account | Local non-weakening only | Yes — lock/hold/append | Sandbox after OD-D03-A-06 inputs |
| G5 | Cost/capacity/SLO disposition | Calculator + workload | No | Partial — measured latency needs cloud | Operator + sandbox |
| G6 | Sandbox window and cleanup | Both clouds | No | Yes | Operator |
| G7 | Dependency-order preservation | Roadmap + readiness §15 graph | LOCAL-01 / POLICY / OFFLINE closed scoped | No — order decision only | Operator confirms order; no reordering |
| D03-A09 | Concurrent CAS | Spanner serializable | In-memory lock ≠ acceptance | Yes | Sandbox P0 |
| D03-A08 | Fencing competing writers | Multi-process / K8s | Offline fencing unit tests | Yes for Spanner | Sandbox |
| D03-A10, A18 | Unknown transaction/append outcome | Both + fault injection | Resolver logic partial | Yes | Sandbox P0 |
| D03-A17 | Witness append CAS | Azure | In-memory witness tests | Yes | Sandbox P0 |
| D03-A19–A20 | Rollback, prefix, segment tail | Azure DR | Offline corrupt-chain tests | Yes | Sandbox P0 |
| D03-A07, A21 | Backup restore stale state | GCP | No | Yes | Sandbox P0 |
| D03-A13 | Tenant/environment boundary | Integrated | LOCAL/offline scope tests partial | Yes | Sandbox |
| D03-A16 | Fail-closed CREATE admission | Code + integration | Static gate inspection partial | Yes integrated | IMPL + sandbox |
| D03-A01 | Positive verdict | Full stack | No | Yes | After P0 pass |

### P7.E Consolidated admission verdict

| Prerequisite | Existing evidence | Missing evidence / decision | Owner | Admission result | Next action |
|---|---|---|---|---|---|
| OD-D03-A-01 class | Ratified | — | Operator | Ready (evidence) | None |
| OD-D03-A-02 preferred pair | Conditional direction | Full ratification | Operator | Blocked | Complete class-B qualification |
| OD-D03-A-03–05 policy | Readiness §11 accepted policy | IAM graph, negative tests | Operator / sandbox | Ready after operator decision on graph only for policy; proof blocked | G2–G3 + D03-A22–A23 |
| OD-D03-A-06 retention | §7.5 rule + local non-weakening | Five inputs + §6.2.1 test | Operator | Ready after operator decision only | Single block §P7.B |
| OD-D03-A-07 cost/SLO | P1–P2 meters, §8.3–§8.4 | Quotes, capacity, achieved SLO | Operator | Ready after operator decision only | G5 |
| OD-D03-A-08 direction | Accepted | RPL-A–H execution | Later qualification | Architectural direction ready; implementation proof blocked | Offline closed; cloud deferred |
| LOCAL-01 / POLICY-CONFIG / OFFLINE-IMPL | Closed scoped SHAs | Cloud qualification | Independent audit | Ready (scoped offline) | Do not reopen |
| G1–G6 | P6 sheet | All decisions | Operator | Blocked on authorization | Grant or refuse sandbox explicitly |
| D03-A01–A23 | §10 + P4 specs | Execution artifacts | Sandbox qualification | Blocked — class B | G1 then P0→P2 order |
| FRZ global promotion | Checklist | D03-specific proof | Freeze process | Blocked | new FRZ PASS = 0 |
| D03-B / D03-C | Roadmap blocked | Independent stages | — | Blocked | Out of scope |
| Cloud execution | Conditional direction | G1 + operator | Operator | **Blocked — unavailable authorization** | No provisioning |
| Class-B admission (cloud execution class) | P7 package + conditional direction | G1–G6, OD-D03-A-06/07, IAM exports, D03-A01–A23 execution | Operator | **NOT AUTHORIZED — OPERATOR DECISIONS MISSING** | Separate explicit sandbox authorization if operator grants G1 |
| Parent D03-A ratification | This package | All above | Operator | **IN-SCOPE BLOCKER** | Stay CONDITIONAL |

### P7.H Operator acceptance reconciliation (accepted evidence `7d8b42e45b5dfe2dd08074f5d0da54419074a906`)

| Field | Value |
|---|---|
| Review disposition | **ACCEPTED — SCOPED READINESS ONLY** (completeness and consistency of §P7; not cloud authorization; not infrastructure qualification) |
| Production delta at accepted SHA | **0** |
| Parent D03-A | **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** — unchanged |
| Global `FRZ-*` | **new FRZ PASS = 0** |
| Canonical G1–G7 view | [P6 decision sheet](#p6--outstanding-operator-decision-sheet); per-row disposition in §P7.D |
| Allowed next stage | Independent audit of this reconciliation delta; then operator G1–G7 decisions and, if granted, a **separate** explicit sandbox / class-B authorization record |

### P7.F ECAP delta (this P7 documentation pass)

| Step | Result |
|---|---|
| A1 | Preparation §P7, readiness §15.4/§19, roadmap ledger sentences. No production symbol. |
| A2 | Links only. |
| A3 | No production caller. |
| A4 | Owners unchanged. |
| A5 | Production-effective reachability: **no**. |
| A6 | No `FRZ-*` promotion. |
| A7 | Historical PASS not extended. |
| A8 | Regression: see readiness §19 test command; not cloud evidence. |

### P7.G ECAP §2.B readiness (parent) — blocked on class B

| Element | State |
|---|---|
| B1 Contract ownership | Blocked — cloud adapter not qualified |
| B2 Layer direction | Offline evidence only; class B unproven |
| B3 Single semantic/composition owner | Documented + offline tests; cloud path unproven |
| B4 Typed boundaries | Offline PASS; live vendor boundary unproven |
| B5 Provider replaceability | RPL-A–H unexecuted — **TRACKED FREEZE DEBT** |
| B6 Roadmap / gate order | Preserved; dependent gate assessment **READY FOR AUDIT**, not CLOSED |

Cursor status: **READY FOR AUDIT — CLASS-B ADMISSION DECISION RECONCILIATION**. Parent remains **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**. Cloud execution remains **NOT AUTHORIZED**.
