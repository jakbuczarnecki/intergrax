# D03-A — Infrastructure ratification readiness R1 (OD-D03-A-03–08)

| Field | Value |
|---|---|
| Status | **READY FOR AUDIT — D03-A RATIFICATION READINESS — OPERATOR DECISIONS PENDING** |
| Date | 2026-10-18 |
| Branch | `development` |
| Inspected baseline | `0ba7d5bfd5324fe66d5cd990ff2dafba2ee4d39a` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN/BLOCKED |
| Stage | `D03-A / OD-D03-A-03–08` |
| Design source | [Trusted anchor backend and trust domain proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) |
| Preferred-pair record | [Conditional infrastructure direction](PROD_Q_R1_R7_Q_D03_A_CONDITIONAL_INFRASTRUCTURE_DIRECTION_OPERATOR_DECISION.md) |
| Class decision | [OD-D03-A-01 ELCR-IIRW](PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md) — unchanged |
| Production delta | **0** |
| FRZ promotion | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |

This record consolidates the unresolved operator decisions OD-D03-A-03 through OD-D03-A-08. It does not copy the technical proposal, does not ratify any of those decisions, and does not authorize implementation, provisioning, credentials, purchases, migrations, new contracts, or runtime ports.

OD-D03-A-01 stays ratified as a mechanism class only. OD-D03-A-02 stays a conditional preferred service choice: Cloud Spanner Enterprise Plus base configuration `eur3`, plus Azure Blob Storage general-purpose v2 Standard, RA-GZRS, primary region North Europe, immutable append-blob witness. No other technology is selected. OD-D03-A-09 and OD-D03-A-10 receive no disposition.

A recommendation in this document is not an operator decision.

## 1. Current-HEAD safety

| Check | Result |
|---|---|
| Branch | `development` (`refs/heads/development`) |
| HEAD | `0ba7d5bfd5324fe66d5cd990ff2dafba2ee4d39a` |
| Remote | `origin` `https://github.com/jakbuczarnecki/intergrax.git`, tracking `origin/development` |
| Pre-audited baseline ancestry | The inspected HEAD **is** the pre-audited baseline. No intervening diff. |
| Working tree before this change | Clean. No parallel-session edits observed. |

No reset, cherry-pick, rebase, force-push, branch, or worktree was used.

## 2. Ownership verification

The intended structure remains a design target, not implementation evidence:

```text
ProductionControlPlaneGovernance
    |
    v
Production lifecycle continuity authority
    |
    v
Vendor-neutral typed ELCR contracts
    |
    +-- Cloud Spanner primary adapter
    |
    +-- Azure immutable witness adapter
    |
    v
Existing GovernedProviderInvocationDispatchGate
    |
    v
Existing provider execution route
```

Closed-world inventory in the proposal §3.2 was not repeated. At the inspected HEAD these assumptions still hold:

| Assumption | Verification at inspected HEAD | Evidence state |
|---|---|---|
| `ProductionControlPlaneGovernance` is the governance wiring record | Class in `intergrax/applications/_shared/production_control_plane_governance_wiring.py` | **DOCUMENTED** |
| `open_production_process_lifecycle` is the production lifecycle opener | Function in `intergrax/applications/_shared/production_process_lifecycle.py` | **DOCUMENTED** |
| One production `GovernedProviderInvocationDispatchGate` class | Class in `applications/governed_contractor_application/host/provider_invocation_lifecycle.py`. Interface methods with the same name exist on the dispatch contract and the reliability port; they are not a second production gate class. | **DOCUMENTED** for symbol existence. Full caller reachability was not re-audited. |
| Stable CREATE identity remains tenant- and principal-scoped | `derive_governed_external_work_create_provider_invocation_id` in `intergrax/contracts/governed_external_work_provider_invocation_identity.py` | **DOCUMENTED** |
| `ProviderInvocationStore` remains the invocation-store contract | Abstract contract in `intergrax/contracts/provider_invocation_store.py` | **DOCUMENTED** |
| No continuity owner, Spanner adapter, or Azure witness adapter exists in production | Proposal §3.1 finding; this task added no production symbol | **DOCUMENTED** for this delta. Absence beyond the known symbols was not re-inventoried. |

No evidence in this bounded check contradicts one semantic continuity owner, one sanctioned composition owner, the existing sole CREATE dispatch gate, distinct PIS and GCA ownership, or Governance separate from Execution. Vendor-specific semantics stay out of platform contracts. This check does not prove a future implementation will preserve that model.

PIS remains the invocation intent/outcome owner. GCA remains the execution/GER/F02 owner. The witness is not a second history, execution, admission, or recovery authority.

## 3. How to read each decision

| Evidence state | Meaning |
|---|---|
| `DOCUMENTED` | A canonical repository record or a vendor page inspected for this package states the fact. |
| `OPERATOR CONFIRMATION REQUIRED` | The fact is an organizational or policy choice. It is not true until the operator records it. |
| `EXECUTION EVIDENCE REQUIRED` | The property can be shown only by configuration inspection or a controlled test. It has not been run. |
| `BLOCKED` | A mandatory D03-A condition is unmet. |
| `NOT APPLICABLE — WITH EVIDENCE` | The property does not apply to this delta, and the reason is recorded. |

Assumptions are never written as guarantees. No decision below is ratified.

## 4. OD-D03-A-03 — Trust-domain independence

### 4.1 Required guarantee

No single principal, group, CI identity, federation administrator, emergency principal, or inherited organization role can modify both the primary register and the witness, restore both histories, impersonate both runtime identities, replace both trusted resource endpoints, weaken witness retention, or suppress or rewrite the audit evidence of both domains.

PIS/GCA backup and restore authority is a third domain. It cannot mutate the anchor or the witness.

### 4.2 Recommended decision

Adopt the separation-of-authority policy in §4.3 as the acceptance policy. Do not treat it as ratified, and do not treat it as evidence that such administrators exist.

Accept **OP-TRUST-07** as residual risk unless the operator requires resistance to compromise of a legitimate transition writer. Storage immutability does not stop that writer from appending a forward seal or causing denial of service. If that resistance is required:

`STOP — ARCHITECTURE DECISION REQUIRED`

A separately trusted transition service is not selected here. No alternate backend is selected.

### 4.3 Proposed separation model

These are role classes. They are not verified principals, project IDs, tenant IDs, or group names.

| Class | May hold | Must not hold |
|---|---|---|
| GCP Spanner administrators | Spanner instance, database, backup, and restore administration inside the platform database project | Azure RBAC, Entra privileged roles, PIS/GCA restore, witness retention, witness legal hold |
| Azure witness administrators | Witness subscription, storage account, container immutability, legal hold, and witness data-plane RBAC, under dual control | Spanner IAM, Spanner backup/restore, PIS/GCA restore, GCP organization administration |
| PIS/GCA backup and restore administrators | Backup and restore of PIS and GCA media under their runbook | Any Spanner or Azure anchor/witness role, including read of currentness used to authorize CREATE |
| Kubernetes workload identities | Short-lived federated reader or transition-writer data access, bound to issuer, namespace, service account, and environment | Initialize, reconcile, revoke, IAM, retention, backup, restore, delete, or the other cloud's administration |
| CI/CD principals | Build and deploy of reviewed configuration through the sanctioned path | Standing data-plane write on both clouds, retention changes, break-glass, or the right to mint both runtime identities |
| Federation and identity administrators | Administration of one provider's workload-identity trust | Administration of the other provider's trust, plus data-plane write on either history |
| Emergency access | Hardware-protected, monitored, two-person access inside one domain | A reusable credential, approver, or group that is valid in both domains |
| Audit-log administration | Read and export administration of one domain's audit sink, retained outside that domain's data-admin control | Delete, stop, or rewrite both clouds' audit exports; mutate either history |
| Organization and inherited privilege | None, unless an expanded export shows the inherited role cannot reach the other domain | GCP organization/folder owner or IAM admin inherited onto Spanner together with Entra Global Administrator, Privileged Role Administrator, or Azure management-group Owner |

Two subscriptions or projects under one shared super-administrator do not satisfy independence. Simultaneous compromise of both identity roots remains residual risk and is not prevented.

### 4.4 Privilege paths that must be shown absent

| Path | Collapsing effect | Evidence state |
|---|---|---|
| One human or group with Spanner admin or restore and Azure storage/RBAC admin | Modify or restore both primary and witness | **OPERATOR CONFIRMATION REQUIRED** |
| Spanner restore plus an Azure role that can unlock immutability, clear a legal hold, or enable a shared key | Restore the primary and weaken witness retention | **OPERATOR CONFIRMATION REQUIRED** |
| Kubernetes identity that can mint tokens for both federated service accounts, where both clouds trust that cluster issuer | Impersonate both runtime identities | **OPERATOR CONFIRMATION REQUIRED** |
| One federation administrator, or one CI principal, trusted by both GCP Workload Identity Federation and Entra | Impersonate both runtime identities or replace effective trust | **OPERATOR CONFIRMATION REQUIRED** |
| One configuration deployer who can change both resource identifiers and both issuer roots without dual control | Replace both trusted endpoints | **OPERATOR CONFIRMATION REQUIRED** |
| One break-glass principal or one person who can approve both domains | Emergency path into both histories | **OPERATOR CONFIRMATION REQUIRED** |
| One audit administrator, or a shared sink whose writers include either data admin | Suppress or rewrite audit evidence | **OPERATOR CONFIRMATION REQUIRED** |
| Unexpanded GCP organization/folder IAM or Azure management-group inheritance | Hidden joint authority | **OPERATOR CONFIRMATION REQUIRED** |
| PIS/GCA restore operator who also holds either anchor role | Restore application history and move the continuity head | **OPERATOR CONFIRMATION REQUIRED** |

No identity graph was inspected. Independence is not claimed.

### 4.5 Accidental restore and active compromise

| Threat | Protection the selected pair can support | Protection it does not provide |
|---|---|---|
| Accidental or stale restore of PIS, GCA, a Spanner backup, or an Azure geo-replicated prefix | Mismatch with the other independent history denies CREATE. A Spanner restore enters a new authority incarnation and stays quarantined. Locked append-only witness blocks overwrite of sealed blocks. | This package does not prove the IAM split or the configured immutability policy. |
| Active compromise or impersonation of the legitimate transition writer (OP-TRUST-07) | An invalid chain fails closed. Immutability preserves already sealed blocks. | The writer can attempt unauthorized forward seals and can deny service. Immutability does not prevent an authorized append. |
| Simultaneous compromise of both identity roots | None claimed | Outside the guarantee. Incident quarantine and a separate operator decision. |

### 4.6 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Proposal §5.3, §6.3, §9.0 and §18. Vendor pages document that separate IAM systems exist. They do not document this organization's administrators. |
| Remaining assumption | OP-TRUST-01, OP-TRUST-02, OP-TRUST-03, and OP-TRUST-07. Separate clouds are assumed to be placeable under non-overlapping roots. That placement is not shown. |
| Exact missing evidence | Expanded GCP organization, folder, and project IAM; Azure management-group, subscription, resource-group, and storage RBAC; Entra privileged-role assignments; Kubernetes RBAC and token-creation rights; both federation trusts; CI federation subjects; break-glass roster and approvers; audit-sink administrators and retention locks. |
| Responsible authority | Operator. A later authorized identity-graph review may collect the exports. This task does not. |
| Acceptance criteria | A dated export, with inheritance expanded, shows none of the collapsing paths in §4.4. Break-glass is distinct, hardware-protected, and two-person in each domain. Shared Key and SAS are disabled. |
| Rejection criteria | Any shared human, group, CI, federation admin, or break-glass path; any unexpanded inherited owner; any inability to enumerate a path; any requirement to resist transition-writer compromise without a new architecture decision. |
| Blocker classification | **IN-SCOPE BLOCKER** |
| Operator decision required | Accept this separation policy and OP-TRUST-07, or reject it. Rejection of OP-TRUST-07 is `STOP — ARCHITECTURE DECISION REQUIRED`. |
| Evidence state | **OPERATOR CONFIRMATION REQUIRED**. Vendor IAM primitives: **DOCUMENTED**. Configured graph: **BLOCKED**. |

## 5. OD-D03-A-04 — History scope and tenant isolation

### 5.1 Required guarantee

The continuity history is environment-scoped. A shared environment anchor does not authorize one tenant to read or write another tenant's PIS, GCA, or continuity evidence.

Tenant-level independent histories stay inactive until PIS and GCA have proven tenant-specific recovery and cross-tenant isolation for this mechanism.

Missing or conflicting environment identity, and missing or conflicting tenant identity where a tenant is required, denies. Restore must preserve the original environment and any proven tenant identity. Kubernetes restore must not become a new authority namespace.

### 5.2 Recommended decision

Keep the proposal's conservative environment-level `EnvironmentHistoryScopeIdentity` as the only scope eligible for a future implementation. Do not activate per-tenant anchors.

### 5.3 Future acceptance requirements

These requirements apply to a later authorized implementation. None is met now.

| Requirement | Acceptance rule | Evidence state |
|---|---|---|
| Environment identity | Non-empty environment identity is part of every head, transition, and witness seal. Mismatch denies. | **EXECUTION EVIDENCE REQUIRED** |
| Tenant identity | Where the caller has a tenant, the proven tenant identity is bound into the scope record. It is not optional padding. | **EXECUTION EVIDENCE REQUIRED** |
| Resource namespace | Primary and witness resources are selected only from sanctioned configuration for that environment. | **EXECUTION EVIDENCE REQUIRED** |
| Scoped authorization | Reader and writer tokens are limited to that environment namespace. | **EXECUTION EVIDENCE REQUIRED** |
| No cross-environment or cross-tenant access | A seal, head, or backup from another environment or tenant cannot satisfy admission. | **EXECUTION EVIDENCE REQUIRED** |
| Restore and identity preservation | Restored PIS, GCA, primary, and witness keep the original environment and tenant. A new Spanner database receives a new authority incarnation and stays quarantined. | **EXECUTION EVIDENCE REQUIRED** |
| Deny on missing or conflicting identity | Empty, default, or conflicting environment or tenant identity denies. It does not widen to a global scope. | **EXECUTION EVIDENCE REQUIRED** |
| Authority namespace after Kubernetes restore | Restored etcd, Lease, ConfigMap, or CRD state is not authority. Stale fences deny. Workload identity cannot initialize a new namespace. | **EXECUTION EVIDENCE REQUIRED** |

### 5.4 Tenant Isolation Audit

```text
documentation-only change verdict: N/A — WITH EVIDENCE
future infrastructure/implementation isolation: UNPROVEN
canonical tenant identity introduced by this delta: none
tenant owner: unchanged
propagation path added: none
state, provider, credential, evidence, and recovery isolation exercised: no
cross-tenant path added: none
fail-closed behavior: required later; not implemented
adversarial evidence: none
```

Historical TENANT-X acceptance does not certify a future Spanner or Azure deployment. An environment-level anchor is not permission to access other tenants. `FRZ-TEN-*` stays at checklist status and is not promoted.

### 5.5 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Proposal §7.1 and OD-D03-A-04 text. Checklist `FRZ-TEN-01` through `FRZ-TEN-12` are historical **PASS** for the platform scope accepted at TENANT-X, not for this unbuilt mechanism. |
| Remaining assumption | One environment has one continuity history, and tenant isolation continues to be enforced by existing PIS/GCA/governance paths plus future scope binding. That binding does not exist yet. |
| Exact missing evidence | Implementation of scope binding; adversarial cross-environment and cross-tenant tests; restore tests showing identity preservation; proof that a Kubernetes restore cannot mint an authority namespace. |
| Responsible authority | Operator for the scope decision. A later authorized D03-A qualification owns the tests. D03-B and D03-C remain separate gates for global PIS and cross-store continuity. |
| Acceptance criteria | Operator explicitly accepts environment-level scope and the prohibition on tenant-level histories until the future acceptance table is proven. |
| Rejection criteria | Any request to create per-tenant anchors now, or any design that treats the shared environment head as cross-tenant authority. |
| Blocker classification | **IN-SCOPE BLOCKER** for the unratified scope decision and the unproven future isolation. This documentation delta itself is **NOT APPLICABLE — WITH EVIDENCE** for runtime tenant isolation. |
| Operator decision required | Accept environment-level scope as the conservative default, or reject it. Silence is not acceptance. |
| Evidence state | Design constraint: **DOCUMENTED**. Operator acceptance: **OPERATOR CONFIRMATION REQUIRED**. Runtime isolation: **EXECUTION EVIDENCE REQUIRED**. |

## 6. OD-D03-A-05 — IAM and workload authorization

### 6.1 Required guarantee

Runtime readers, transition writers, provisioning, reconciliation, security revocation, backup/restore, witness custodians, and independent auditors have distinct minimum privileges. Both clouds use independent short-lived workload federation. Static cloud credentials, runtime Shared Key, SAS, broad administrator permission, and unrestricted cross-scope access are forbidden.

### 6.2 Recommended decision

Use proposal §6.3 as the privilege baseline. Ratify it only after the operator accepts the minimums and forbids below, and only after negative access tests exist. Do not invent project IDs, principals, or role resource names.

If resistance to compromise of `AnchorTransitionWriter` is required beyond OP-TRUST-07, do not add a second service inside this decision:

`STOP — ARCHITECTURE DECISION REQUIRED`

### 6.3 Minimum privileges and forbidden capabilities

| Principal class | Minimum | Forbidden |
|---|---|---|
| Runtime reader | Strong read of the scoped primary head; witness property and range read | Writes, restore, IAM, retention, other environments |
| Transition writer | Scoped primary `SELECT`/`INSERT`/`UPDATE` on head and transition rows; witness read plus append | Initialize, reconcile, revoke, delete, overwrite, IAM, retention, backup, restore, cross-scope access |
| Provisioning | Create genesis rows and the deterministic genesis append under a governed workflow | Ordinary transition and provider dispatch |
| Reconciliation | Create a quarantined incarnation and append one recovery seal; read approved reconciliation evidence | Rewrite or delete prior rows; physical provider dispatch |
| Security revocation | Compare-and-set to quarantined/revoked and append the matching seal | Reactivation or history rewrite |
| Backup and restore | Spanner backup/restore only for Spanner administrators; PIS/GCA backup/restore only for PIS/GCA administrators | Each is forbidden from the other domain and from the witness |
| Witness custodians | Retention, legal hold, account lock, and witness RBAC under dual control | Spanner administration and PIS/GCA restore |
| Independent auditors | Read transition, witness, and exported audit evidence | Mutation, retention changes, and CREATE admission |

Built-in broad data-writer roles are not an acceptable substitute for these minimums.

### 6.4 What is vendor-documented and what is not

| Property | State |
|---|---|
| Google workload identity federation can exchange a Kubernetes OIDC token without a downloaded service-account key | **DOCUMENTED** — proposal §17.1, not re-tested |
| Entra workload identity federation can trust a Kubernetes issuer, including clusters outside Azure | **DOCUMENTED** — proposal §17.2, not re-tested |
| Azure custom roles can limit data actions; append requires the append data action | **DOCUMENTED** — proposal §17.2 |
| Spanner database roles and Cloud Audit Logs exist as products | **DOCUMENTED** — proposal §6.1 and §17.1 |
| Exact custom grants, token audiences, attribute conditions, and deny assignments for this platform | **EXECUTION EVIDENCE REQUIRED** |
| Negative tests that runtime identity cannot init, reconcile, revoke, delete, change retention, restore, or cross scopes | **EXECUTION EVIDENCE REQUIRED** |
| No standing JSON key, Shared Key, or SAS in the runtime path | **OPERATOR CONFIRMATION REQUIRED** and **EXECUTION EVIDENCE REQUIRED** |
| The two federations do not share an administrator | **OPERATOR CONFIRMATION REQUIRED** |

### 6.5 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Proposal §6.3, §9.1, and §17. No production IAM binding exists in this repository delta. |
| Remaining assumption | Short-lived federation can be bound to the exact cluster issuer, namespace, service account, and environment, and those tokens can be limited to the minimums above. |
| Exact missing evidence | Custom role definitions, deny tests, token lifetime and audience, proof that Shared Key and SAS are disabled, and proof that provisioning, reconciliation, revocation, restore, and audit identities are not mounted into pods. |
| Responsible authority | Operator for the privilege policy. Cloud configuration evidence belongs to a later authorized qualification, not to this document. |
| Acceptance criteria | Operator accepts §6.3 and §6.3's prohibitions. A later test shows D03-A22 and D03-A23. Federation is short-lived and independent on both clouds. |
| Rejection criteria | Any static key, Shared Key, SAS, broad administrator role on a runtime identity, or a design that needs a second trusted transition service without an architecture decision. |
| Blocker classification | **IN-SCOPE BLOCKER** |
| Operator decision required | Accept the minimum-privilege baseline, or reject it. |
| Evidence state | Policy text: **DOCUMENTED** as a proposal. Configured IAM: **BLOCKED**. |

## 7. OD-D03-A-06 — Retention and recovery

### 7.1 Purpose of witness retention

The witness exists to detect rollback for as long as an old history could still be presented as current, to keep seal evidence available through recovery, and to resist deletion. It does not by itself prove a legal retention duty.

| Purpose | What would justify a duration | State |
|---|---|---|
| Detect rollback during the active history lifetime | Longer than the longest restorable age of PIS, GCA, and primary backups, plus the operator's declared investigation window | Ages are **OPERATOR CONFIRMATION REQUIRED**. They are not in the canonical record. |
| Keep evidence available throughout recovery | The chain must remain readable for the whole recovery, including dual-controlled cutover | Mechanism shape is proposed. Duration is not measured. |
| Protect against deletion | Locked container retention plus a legal hold. Time-based expiry without a hold is a deletion path. | Primitive: **DOCUMENTED** in the proposal's vendor citations. Configured policy: **EXECUTION EVIDENCE REQUIRED**. |
| Handle retention expiry | Expiry must not delete seals while the environment is active. A hold has to cover that gap, or the operator must accept a deletion window. | **OPERATOR CONFIRMATION REQUIRED** |
| Operational and regulatory impact | Storage is retained and billed for the whole period. No statute establishing ten years is cited in the canonical sources. | Ten years is not a documented legal obligation. |
| Disposal and decommissioning | Release of the hold and retention only by dual control after the environment is formally retired | Procedure: **OPERATOR CONFIRMATION REQUIRED** |
| Witness unavailable | CREATE stays denied until the latest retained chain is proved. Spanner alone is not currentness. | Rule: **DOCUMENTED** as design. Behavior: **EXECUTION EVIDENCE REQUIRED**. |

### 7.2 Recommended decision

Do not ratify ten years. Do not treat `ELCR-ACTIVE` as already adopted.

Recommend that the operator:

1. Keep deletion protection as a locked immutable policy plus a legal hold that remains until dual-controlled decommission.
2. Set the numeric retention only after declaring backup/PITR ages and the investigation window. The number must exceed those ages.
3. Treat the proposal's ten-year figure as an unadopted candidate, not as a regulatory finding.

If the operator cannot supply those ages and still wants a number, the number is an explicit policy choice and must be labeled as such.

### 7.3 Required recovery rules

These rules are the recommended acceptance policy. They are not ratified and not tested.

| Case | Required rule |
|---|---|
| Primary rollback | Deny. Restore only into a new database with a new authority incarnation. Stay quarantined until the highest witness seal matches and a dual-controlled recovery seal is committed. No restore-in-place. |
| Witness rollback or geo-replicated prefix | Deny and quarantine. The secondary endpoint and Last Sync Time are diagnostic. They do not authorize CREATE. |
| Partial write | At most one non-terminal transition per scope. CREATE denied until that same transition resolves. |
| Crash between `PREPARED`, `WITNESS_SEALED`, and `COMMITTED` | Resume only the same transition ID and payload. Do not mint a new transition ID, decrement a head, or call the provider. |
| Unknown Azure append outcome | Strong range read at the expected offset. Exact seal proceeds. Absent seal may retry the same append. Any other content quarantines. Timeout is not success. |
| Malformed witness tail | Quarantine. No truncation and no allow. |
| Segment rollover | CREATE denied until one deterministic successor link is verified. A second successor, a gap, or data after `SEGMENT_CLOSED` quarantines. |
| Regional disaster | Spanner strong reads only. One region inside `eur3` is not a license for stale reads. Azure regional loss denies CREATE until the exact chain is proved. Asynchronous geo-replication is not zero data loss. |
| New authority incarnation | Quarantined until witness equality and a new dual-controlled seal. |
| Complete loss of witness availability | Deny. No cached allow and no CREATE from the primary alone. |

No automatic CREATE resumption from an unproven history. No claim that asynchronous regional replication provides zero data loss.

Vendor fact re-verified 2026-10-18 from Microsoft Learn, [data redundancy](https://learn.microsoft.com/en-us/azure/storage/common/storage-redundancy): geo-replication to the paired secondary is asynchronous, and a primary-region failure can lose data that has not been replicated. The same page's Geo Priority Replication statement of an RPO at or under 15 minutes is explicitly for Block Blobs. This record does not extend that figure to append blobs and does not use it as authorization RPO. The paired secondary region is assigned by Azure and is not an operator-selected region.

### 7.4 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Proposal §6.2, §8.2, §8.3, and §9.4. Azure redundancy page inspected for this package. Immutability mechanics are cited from the proposal's 2026-10-18 vendor list and were not re-fetched page-by-page here. |
| Remaining assumption | Locked WORM with protected appends, plus a hold, can be configured on the selected account type. That configuration is not present. |
| Exact missing evidence | Operator backup ages and investigation window; chosen numeric retention; legal-hold owner; proof the policy is locked; rollover and crash tests; DR test showing prefix loss does not resume CREATE. |
| Responsible authority | Operator for duration, hold, and decommission. Later authorized sandbox qualification for recovery behavior. |
| Acceptance criteria | Operator records a duration justified by declared ages, or explicitly adopts an unjustified number as policy. Operator accepts every recovery rule in §7.3, including no restore-in-place and no zero-loss claim for Azure geo-replication. |
| Rejection criteria | Ratifying ten years as if a statute required it; treating secondary reads as current; automatic CREATE after an unknown append or a regional failover. |
| Blocker classification | **IN-SCOPE BLOCKER** |
| Operator decision required | Choose the retention period and accept or reject the recovery rules. Ten years is not recommended as a legal finding. |
| Evidence state | Deletion-protection need: **DOCUMENTED** as design. Ten-year period: **OPERATOR CONFIRMATION REQUIRED**. Configured WORM and recovery drills: **EXECUTION EVIDENCE REQUIRED**. |

## 8. OD-D03-A-07 — Cost, SLO, and operating model

### 8.1 Required guarantee

Safety rules are not relaxed to meet a cost or latency target. Authorization RPO 0 means an unverified older history is never accepted. It does not mean zero physical data loss in a regional disaster.

### 8.2 Price evidence inspected 2026-10-18

No monthly cost and no workload volume are stated, because neither was verified.

| Input | What was verified | What was not verified |
|---|---|---|
| Spanner edition and `eur3` | Multi-region configurations, including `eur3`, are Enterprise Plus. `eur3` read-write regions are Belgium `europe-west1` (leader, two replicas) and Netherlands `europe-west4` (two replicas). Witness is Finland `europe-north1`. Documented multi-region availability target is 99.999%. Sources: [instance configurations](https://docs.cloud.google.com/spanner/docs/instance-configurations), [pricing](https://cloud.google.com/spanner/pricing). | The live Enterprise Plus `eur3` node-hour, storage, backup, replication, and egress amounts. The pricing page loads those amounts in a selector; they were not in the static page text. |
| Capacity | Compute is billed as node-hours, with 1000 processing units equal to one node in the published example. A 200-processing-unit example is billed as a fraction of a node. Provisioned compute has a one-hour minimum, then prorates. A base multi-region node price includes its replicas. | The proposal's statement that managed autoscaling starts at 1000 processing units and that manual capacity can start at 100 processing units was not re-opened on the autoscaler page in this pass. Treat that floor as prior proposal evidence until the operator confirms the current page. |
| Fixed versus variable | Fixed component: provisioned compute, billed even when idle, including replicas. Variable components: database storage, backup storage, cross-region replication, and applicable outbound network. Optional read-only replicas are outside the accepted base `eur3` configuration and would add compute, storage, and replication. | Contract discounts, committed-use discounts, and tax. |
| Azure Blob | Meters are capacity per GB-month, operations, data transfer, and data retrieval. Operation prices are generally per 10,000 operations. `Append Block` is a Write on Standard general-purpose v2. `Get Blob` is a Read. `Get Blob Properties` is Other on Standard general-purpose v2. Sources: [plan and manage costs](https://learn.microsoft.com/en-us/azure/storage/common/storage-plan-manage-costs), [REST operation map](https://learn.microsoft.com/en-us/azure/storage/blobs/map-rest-apis-transaction-categories). | North Europe RA-GZRS list rates. The public pricing page states that displayed prices are estimates, not quotes. |
| WORM and RA-GZRS | RA-GZRS is synchronous zone redundancy in the primary region plus asynchronous copy to the paired secondary, with read access to that secondary. Retained immutable data continues to occupy billable capacity. Geo-replication can lose an unreplicated suffix. | Whether the North Europe RA-GZRS capacity price already includes geo-replication traffic. The operator must read that from the current calculator. |
| Cross-cloud | Each transition needs a Spanner commit path and an Azure append plus read-back. Client location determines egress. | Measured bytes, request counts, and egress rates. |
| Audit retention | Cloud Audit Logs and Azure Activity plus storage resource logs are separate products and separate retention charges. | Sink size, retention days, and immutability of the sink. |
| Private connectivity | Private Service Connect, Private Link, VPN, or peering would be additional meters if chosen. | Whether the operator will require private paths. |
| Operating and recovery cost | Dual control, break-glass, sandbox subscriptions, backup storage, and DR exercises are recurring work and recurring cloud spend. | Labor rates and exercise frequency. |

Supplier list prices, once quoted by the operator from the current calculator, are prices. Anything multiplied by an undeclared volume remains an estimate. This document contains neither.

### 8.3 Cost formulas

The operator supplies every rate and every volume. Symbols are inputs, not results.

```text
spanner_compute = (processing_units / 1000) * billed_hours * eur3_enterprise_plus_node_hour_rate
spanner_storage = average_database_gib * database_gib_month_rate
spanner_backup = average_backup_gib * backup_gib_month_rate
spanner_replication = replicated_gib * replication_gib_rate
spanner_egress = egress_gib * applicable_network_rate

witness_ops = (append_writes / 10000) * write_rate
            + (range_reads / 10000) * read_rate
            + (property_and_other_ops / 10000) * other_rate
witness_capacity = retained_gib * ra_gzrs_north_europe_gib_month_rate
witness_transfer = transfer_gib * transfer_rate + retrieval_gib * retrieval_rate

audit = log_gib * sink_rate
private_path = private_link_or_psc_hours + processed_gib_rate
```

`retained_gib` grows for the whole retention period because sealed blocks are not deleted. `billed_hours` is at least one hour for each provisioned capacity change, then prorated. Optional Spanner read-only replicas are excluded unless the operator explicitly adds them.

### 8.4 Proposed targets

| Target | Purpose | Measurement | Missing data | If missed | Operator decision |
|---|---|---|---|---|---|
| Positive verdict lifetime ≤ 5 seconds | Limit how long a positive CREATE verdict may be reused | `not_after` minus issue time on the verdict. A cache must not outlive it. | Clock source, reservation design, and D03-C critical-section budget | Shorten the verdict or stop issuing it. Do not allow a stale verdict. | Accept, change, or reject the number. It is not a vendor guarantee. |
| Committed revocation visibility ≤ 5 seconds | Bound how quickly a new reservation sees a committed quarantine or revocation | Strong read of the revocation revision versus the next reservation | Region of the cluster, Spanner commit latency, and probe method | Fail closed for new CREATE. Do not use a stale positive cache. | Accept, change, or reject the number. |
| Transition p99 ≤ 500 ms | Operating latency from approved EU cluster regions for one successful seal | Server span from primary CAS start through `COMMITTED`, successful transitions only | Cluster region, private versus public path, payload size, and a measured baseline. `eur3` quorum and an Azure append both add latency. | Operator decides whether to accept the measured latency. Consistency and fail-closed rules stay. | The 500 ms figure is not defensible as a commitment until measured. |
| CREATE recovery target 4 hours | Operational target to return to a legitimate eligible verdict after a regional event | Clock from the declared event to the first eligible verdict after dual-controlled recovery | Runbook steps, approver availability, restore duration, and witness comparison time | Missing the target does not authorize restore-in-place or a secondary allow. | Accept as an operating target or replace it with a staffed target. |
| Authorization RPO 0 | Never accept an unverified older history | Count of eligible verdicts whose revision is below the highest proven seal must stay zero | None for the rule. Physical loss still requires a DR test. | Deny. Unavailability is the correct safety result. | Accept the safety meaning. Do not redefine it as zero physical loss. |

The 5-second, 500-millisecond, and 4-hour numbers are proposal targets. They are **OPERATOR CONFIRMATION REQUIRED** and **EXECUTION EVIDENCE REQUIRED**. Authorization RPO 0 is a safety rule and is recommended as policy. It is still not ratified. Spanner's 99.999% multi-region availability target is a vendor availability statement for the instance configuration. It is not a measured latency, not a cross-cloud RPO, and not a purchased SLA for this platform.

### 8.5 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Pricing and redundancy pages cited in §8.2. Proposal §9.5 for the unratified numeric targets. |
| Remaining assumption | The operator can obtain current calculator quotes and can name a processing-unit size and a private-connectivity choice. |
| Exact missing evidence | Calculator quotes for Enterprise Plus `eur3` and North Europe RA-GZRS; chosen processing units; expected transitions per month; cluster region; audit retention; discount program. |
| Responsible authority | Operator for budget and for accepting or rejecting each target. |
| Acceptance criteria | Operator records the quotes, the capacity choice, and an explicit accept or reject for each row in §8.4, without weakening fail-closed behavior. |
| Rejection criteria | Inventing a monthly total, choosing a cheaper edition or a regional primary in place of `eur3`, or allowing stale reads to hit a latency target. Any of those is outside this decision and is `STOP — ARCHITECTURE DECISION REQUIRED` if proposed as a substitute pair. |
| Blocker classification | **IN-SCOPE BLOCKER** |
| Operator decision required | Supply the missing rates and volumes, then accept or reject the targets. |
| Evidence state | Meter structure and `eur3` topology: **DOCUMENTED**. Prices, capacity, and SLO attainment: **OPERATOR CONFIRMATION REQUIRED** and **EXECUTION EVIDENCE REQUIRED**. |

## 9. OD-D03-A-08 — Contracts, composition, and replaceability

### 9.1 Required guarantee

A future implementation, if separately authorized, uses typed platform-owned identity and continuity contracts, sanctioned production configuration, one approved composition owner, vendor-specific adapters under those contracts, and the existing CREATE dispatch gate. Unknown, missing, or mismatched evidence fails closed.

### 9.2 Recommended decision

The proposal is compatible with the ownership model verified in §2. Recommend that a later implementation be held to the following constraints:

- platform contracts carry no Spanner or Azure types;
- `ProductionControlPlaneGovernance` and `open_production_process_lifecycle` remain the governance and composition side of continuity;
- `GovernedProviderInvocationDispatchGate` remains the only physical CREATE gate;
- `submit_quote_acceptance` and `cancel_work` keep their existing semantics;
- PIS and GCA are not replaced or duplicated by the anchor;
- configuration comes through the existing sanctioned configuration path;
- missing configuration fails closed.

This recommendation does not create those contracts and does not claim replaceability has been demonstrated.

### 9.3 Later replaceability evidence

ECAP `RPL-A` through `RPL-H` are required before any replaceability claim. Expected evidence, all absent now:

| Id | Later evidence | State |
|---|---|---|
| RPL-A | One neutral continuity contract; consumers do not import vendor SDKs | **EXECUTION EVIDENCE REQUIRED** |
| RPL-B | A second conforming adapter selected only through sanctioned composition | **BLOCKED** |
| RPL-C | Consumers unchanged when the adapter changes | **EXECUTION EVIDENCE REQUIRED** |
| RPL-D | Configuration selects the adapter; no hardcoded production choice and no silent fallback | **EXECUTION EVIDENCE REQUIRED** |
| RPL-E | The alternate adapter preserves the continuity guarantees | **EXECUTION EVIDENCE REQUIRED** |
| RPL-F | Failure and recovery stay fail-closed and do not bypass the dispatch gate | **EXECUTION EVIDENCE REQUIRED** |
| RPL-G | Regression gates for the qualified substitution | **EXECUTION EVIDENCE REQUIRED** |
| RPL-H | An explicit extensibility classification, not an implied portability claim | **OPERATOR CONFIRMATION REQUIRED** |

Infrastructure dependency inventory, unchanged consumers, absence of a hidden vendor choice, failure-behavior equivalence, configuration selection, and regression protection are future artifacts. Checklist `FRZ-RPL-01` through `FRZ-RPL-04` stay **OPEN**.

### 9.4 Decision record

| Field | Value |
|---|---|
| Supporting evidence | Proposal §3, §7, and §10. Symbol check in §2. ECAP §4.3 RPL-A..H. No production code was modified. |
| Remaining assumption | A later implementation can place both adapters under one composition owner without a second gate. That implementation does not exist. |
| Exact missing evidence | Contract review, import graph, configuration selection test, alternate adapter, and RPL-A..H execution evidence. |
| Responsible authority | Operator for accepting the constraint set. Implementation remains unauthorized. |
| Acceptance criteria | Operator accepts the constraints in §9.2 as binding on any later implementation. Replaceability stays unclaimed until RPL-A..H exist. |
| Rejection criteria | Vendor types in platform contracts, a second CREATE gate, a shadow PIS or GCA, or a claim that this document proves replaceability. |
| Blocker classification | **IN-SCOPE BLOCKER** for ratification and implementation. Missing RPL execution evidence is also **TRACKED FREEZE DEBT** relative to global `FRZ-RPL-*`, which this package does not close. |
| Operator decision required | Accept the composition constraints, or reject them. Acceptance still does not authorize code. |
| Evidence state | Design compatibility of the proposal: **DOCUMENTED**. Implemented replaceability: **BLOCKED**. |

## 10. Sandbox qualification plan

The rows are the existing D03-A01–A23 catalogue. No second catalogue is created. No row was executed. A local double may later exercise resolver logic. It is not proof of Spanner, Azure immutability, IAM, or regional failover.

| ID | Setup | Action or fault | Expected outcome | Observable acceptance evidence | Environment | Without paid cloud |
|---|---|---|---|---|---|---|
| D03-A01 | One environment scope; PIS and GCA incarnations match the committed head; highest witness seal matches | Request a CREATE verdict | `ELIGIBLE` at D03-A verdict level only. Physical CREATE still requires D03-B and D03-C | Verdict binds scope, revision, fence, and both incarnations. No provider execute from this predicate alone | Future integration, then Kubernetes. Real anchors need the sandbox pair | No for acceptance |
| D03-A02 | PIS or GCA empty or unproven | CREATE admission | `DENY/UNPROVEN_INITIALIZATION`. Zero provider calls | Deny code and zero execute callback | Fault injection | Resolver logic only. Not acceptance of cloud state |
| D03-A03 | PIS media restored to an older incarnation or watermark | CREATE admission | `DENY/PIS_INCARNATION_MISMATCH` or stale watermark | External head unchanged; deny recorded | Restore drill | No for acceptance |
| D03-A04 | GCA media restored to an older incarnation or watermark | CREATE admission | `DENY/GCA_INCARNATION_MISMATCH` or stale watermark | External head unchanged; deny recorded | Restore drill | No for acceptance |
| D03-A05 | Both PIS and GCA restored so they agree with each other and disagree with the anchor | CREATE admission | Deny against the higher external head and witness | Both local histories agree; verdict still denies | Restore drill | No for acceptance |
| D03-A06 | Both histories and a pod-local proof restored below the witness | CREATE admission | `DENY/CONTINUITY_REVISION_STALE` or `FENCE_STALE` | Local proof revision is below the witness; zero provider calls | Fault injection and process restart | No for acceptance |
| D03-A07 | Primary backup restored, or witness prefix presented as current | Admission and recovery | Deny and quarantine. No silent allow | New primary incarnation is not current; witness mismatch is recorded | Managed-service DR drill | No for acceptance |
| D03-A08 | Stale replica or old fencing token | Transition or CREATE | `DENY/FENCE_STALE`. Zero provider calls | Newer fence remains the only writer | Multi-process, then Kubernetes partition | No for acceptance |
| D03-A09 | Two concurrent compare-and-advance calls for the same head, including conflicting expected revisions | Spanner serializable CAS | Exactly one winner. The loser receives a typed conflict or stale result | One committed successor. The other revision is absent | Spanner sandbox | No for acceptance. A local lock is not Spanner proof |
| D03-A10 | Crash or timeout after `PREPARED`, after witness append, or before `COMMITTED` | Resolve the same transition | Deny CREATE until that transition ID matches committed primary and witness state. No new transition ID | Durable row and seal either match or the scope is quarantined | Fault injection across both clouds | No for acceptance |
| D03-A11 | Legacy PIS-only CREATE without anchor evidence | Admission | `DENY/LEGACY_UNRESOLVED`. No heuristic match and no provider call | Deny code; no execute | Contract and integration | Current-code inspection cannot prove the future predicate |
| D03-A12 | Governed reconciliation for one exact scope | Reconciliation workflow | Eligible only for the proven scope, a new revision or epoch, and a sealed record | Dual-controlled seal; prior rows not rewritten | Contract, integration, operator workflow | No for acceptance |
| D03-A13 | Evidence from another environment or tenant | Admission | `DENY/SCOPE_MISMATCH`, `TENANT_MISMATCH`, or `ENVIRONMENT_MISMATCH` | Foreign seal is not a current head | Adversarial multi-scope test | No for acceptance |
| D03-A14 | Primary or witness unreadable | Admission | `DENY/AUTHORITY_UNAVAILABLE`. No cached allow | Outage or partition plus deny; zero provider calls | Network partition | No for acceptance |
| D03-A15 | `submit_quote_acceptance` and `cancel_work` | Exercise existing paths | No CREATE-only history predicate is applied. Existing accept and cancel semantics remain | Those operations still follow their current admission and F02 results | Contract and integration regression | Current behavior can be inspected without new cloud. The future predicate is not proven |
| D03-A16 | Strict production and HTTP routes | Attempt provider CREATE | No route reaches provider CREATE except the existing dispatch gate | Static reachability plus integrated HTTP, later Kubernetes | Current gate exists. Future proof needs the implementation | Static inspection of the current gate does not prove the future path |
| D03-A17 | Two writers append the same revision at one Azure offset | Conditional append | One append succeeds. The loser receives a conflict. The exact tail identifies the winner | One committed block at that offset; HTTP 412 on the loser | Azure sandbox | No for acceptance |
| D03-A18 | Append committed and the response is lost | Exact range read | The same seal is recovered. No second seal and no new transition ID | Range contents equal the prepared digest | Proxy fault injection against Azure | No for acceptance |
| D03-A19 | Crash around `SEGMENT_CLOSED` and successor genesis; also a malformed or extra tail | Rollover and read | CREATE denied until one linked successor is verified. Malformed tail, gap, or second successor quarantines. No truncation | Chain link verified or quarantine recorded | Azure sandbox | No for acceptance |
| D03-A20 | Unplanned RA-GZRS failover exposes a prefix | Compare with Spanner | Mismatch and quarantine. No automatic CREATE | Prefix length or digest behind the primary head | Controlled Azure DR | No for acceptance |
| D03-A21 | Spanner backup restored to a new database | Recovery | New authority incarnation stays quarantined until the highest witness seal and dual control are proven | New database is not the current endpoint before the recovery seal | GCP sandbox | No for acceptance |
| D03-A22 | Runtime identity calls init, reconcile, revoke, delete, retention, restore, or another scope | IAM call | Deny in both clouds, with an auditable event | Cloud audit event and no mutation | Dual-cloud IAM test | No for acceptance |
| D03-A23 | Primary administrator attempts witness administration, and the reverse | Identity-graph test and break-glass exercise | No effective entitlement and no impersonation chain | Expanded IAM export plus a failed administrative call | Identity inspection and sandbox | Graph export needs the real tenants. Not a mock |

Real Spanner and Azure behavior requires a future explicit authorization to use controlled infrastructure. This package does not provision that infrastructure and does not claim the tests passed.

## 11. Decision readiness gate

| Decision | Proposed disposition | Evidence sufficient? | Operator action |
|---|---|---|---|
| OD-D03-A-03 | Recommend the §4.3 separation policy and accept OP-TRUST-07 as residual risk. Not ratified. | No | Produce the expanded identity graph, or reject the policy. Rejecting OP-TRUST-07 stops for an architecture decision. |
| OD-D03-A-04 | Recommend environment-level history only. Tenant-level histories stay inactive. Not ratified. | No | Accept or reject the conservative scope. |
| OD-D03-A-05 | Recommend the §6.3 minimums, dual short-lived federation, and the listed prohibitions. Not ratified. | No | Accept or reject the privilege baseline. Do not authorize a second transition service inside this decision. |
| OD-D03-A-06 | Do not adopt ten years as a legal duty. Recommend locked retention plus a hold until dual-controlled decommission, a duration set from declared backup ages, and the §7.3 recovery rules. Not ratified. | No | Declare backup ages and a duration, and accept or reject the recovery rules. |
| OD-D03-A-07 | Record formulas and vendor meter facts only. Numeric SLO targets stay unmeasured. Authorization RPO 0 is recommended as a deny rule, not as physical zero loss. Not ratified. | No | Quote current prices, choose capacity, and accept or reject each target without weakening safety. |
| OD-D03-A-08 | Recommend the §9.2 constraints for any later implementation. Replaceability is not demonstrated. Not ratified. | No | Accept or reject the constraints. Acceptance does not authorize code. |

Implementation stays unauthorized until the operator explicitly accepts the mandatory infrastructure conditions and the remaining execution evidence is either obtained or assigned to a separately authorized qualification gate. That assignment must follow canonical roadmap order. It is not created by this document.

The selected pair is not reported as incapable of the ratified ELCR-IIRW class. The unproven conditions are operator evidence and future tests. They are blockers to ratification, not a silent change of backend.

## 12. Applicable enterprise criteria

Checklist status below is the status column at the inspected HEAD. Historical **PASS** is not certification of this unbuilt infrastructure. No status is promoted. Future qualification owner, for every D03-specific gap, is a later explicitly authorized D03-A implementation qualification. This readiness package is not that owner. D03-B and D03-C remain separate blocked stages.

| Family | Current status | Relevance | Existing evidence | Missing D03-specific evidence | Future owner |
|---|---|---|---|---|---|
| `FRZ-OWN-01`..`05` | **OPEN**. Wording: exactly one semantic owner; one canonical contract per responsibility; one sanctioned composition owner; no shadow authority; no duplicated semantic mechanism | Continuity must not become a second PIS, GCA, gate, or recovery authority | Current symbols in §2 and proposal §3 | Ratified owner model and implementation reachability | Later authorized D03-A qualification |
| `FRZ-BND-01`..`06` | **OPEN**. Wording: canonical layer model documented; allowed dependency directions explicit; reverse dependencies = 0; private implementation leakage = 0; cross-layer construction only through sanctioned composition; transport adapters do not own runtime semantics | Future adapters must sit behind Tier-0 contracts | Existing layer directions are historical platform evidence, not an adapter import graph | Contract home, import proof, and adapter boundary gates | Later authorized D03-A qualification |
| `FRZ-CTR-01`..`06` | **OPEN**. Wording: consumers depend on contracts; concrete provider leakage = 0; no pseudo-contract dicts; public/internal contracts classified; contract responsibility narrow/cohesive; contracts do not encode concrete vendor/runtime implementation | Neutral continuity contract only if later authorized | Conceptual operations in the proposal. No new contract in this delta | Code-level contract review | Later authorized D03-A qualification |
| `FRZ-TYP-01`..`06` | **OPEN**. Wording: semantic boundaries strongly typed; `Any` at semantic boundaries = 0 unless an evidence-backed transport reason exists; generic object semantic seams = 0; reflection-based semantic dispatch = 0; string-dispatch substitute for a typed contract = 0; type-ignore or cast cannot mask an architecture mismatch | Future records and deny reasons | Proposal types are conceptual | Implemented models and static gates | Later authorized D03-A qualification |
| `FRZ-PLG-01`..`07` | **OPEN**. Wording: extensible mechanisms expose platform contracts; external structural implementation works without a core patch; provider discovery has an explicit owner; provider selection has an explicit owner; activation/admission explicit; plugin package is not capability authority; runtime extensions cannot self-expand authority | Adapter must not self-activate | Composition precedent only | Selection contract and negative reachability | Later authorized D03-A qualification |
| `FRZ-RPL-01`..`04` | **OPEN**. Wording: representative default provider replaceable; custom implementation proof exists; replaceability is not monkeypatch-only; vendor/backend switch does not modify consumers | Required before any portability claim | None for this pair | RPL-A..H in §9.3 | Later authorized D03-A qualification; global RPL remains freeze debt |
| `FRZ-CFG-01`..`08` | **PASS**. Wording is the checklist text, including: selection comes from an explicit typed configuration contract; missing configuration fails closed; configured state is not effective state; one configuration contract and one composition owner; endpoints and secrets stay in approved boundaries; no alternate factory; deterministic precedence; regression gates against hardcoded selection | A future anchor must use sanctioned configuration | Historical PASS for the current platform. No anchor configuration exists | Proof the future pair is selected only through that path | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-SEC-01` | **OPEN**. Wording: tenant isolation proven | Cross-environment and cross-tenant denial | Historical scoped evidence in the checklist notes. Not this mechanism | D03-A13 and the identity graph | Later authorized D03-A qualification |
| `FRZ-SEC-04` | **OPEN**. Wording: production auth bypass = 0 | Runtime must not bypass IAM or the dispatch gate | Current gate symbol. Bypass of the future anchor is untested | D03-A16 and D03-A22 | Later authorized D03-A qualification |
| `FRZ-TEN-01`..`12` | **PASS**. Wording includes canonical tenant identity where tenant scope applies; end-to-end propagation; no silent widening; blocked cross-tenant state; tenant-preserving provider and credential bindings; tenant-preserving evidence; restore preserves tenant; legal cross-tenant action is explicit; missing tenant does not become global; plugins cannot widen tenant; adversarial tests exist for the accepted platform scope | Future anchor must not weaken these rules. Environment scope does not waive them | TENANT-X historical PASS | D03-A13, restore identity, and workload scope tests | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-REL-03` | **OPEN**. Wording: idempotency ownership explicit | One transition ID and one non-terminal transition per scope | Proposal §8.2 design | Crash and duplicate tests | Later authorized D03-A qualification |
| `FRZ-REL-05` | **OPEN**. Wording: external operation termination semantics explicit | Unknown append and unknown commit must not become success | Proposal resolution rules | D03-A10 and D03-A18 | Later authorized D03-A qualification |
| `FRZ-STA-01`..`08` | **PASS**. Including exactly one truth owner per state family; no duplicate semantic state stores; explicit transaction boundaries; tenant isolation; stale state rejected; configured, effective, and persisted state distinguished; durable policy artifacts where required; checkpoints are not identity authority | Anchor would be a new state family and must have one owner | Historical PASS. No anchor state family exists | Implementation and stale-state tests | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-REC-01`..`10` | **PASS**. Including deterministic crash recovery; restart preserves authority; resume does not mint identity incorrectly; replay cannot create divergent truth; explicit fork semantics; partial persistence fails closed; external uncertainty modeled; explicit backup/restore semantics; restore preserves identity, authority, tenant isolation, and exactly-one truth; corrupt state cannot become canonical truth | §7.3 is the future rule set | Historical PASS. No anchor recovery drill | D03-A03 through D03-A07, D03-A10, D03-A19 through D03-A21 | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-GOV-01`..`10` | **PASS**. Including Governance is not Execution; proposal, permission, and execution are distinct; child authority is a subset; downstream scope narrows only; fresh authorization; missing human approval is not approval; required evidence before work; policy and profile revisions are attributable; extensions cannot expand authority; control-plane mutation is governed | Continuity evidence must be governance evidence, not execution | Historical PASS and the proposed producer/consumer split | Fresh proof on the future path | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-EXE-01`..`07` | **PASS**. Including exactly one legal execution authority; meaningful work enters the canonical boundary; Nexus remains private implementation; an agent is not Execution; child work is admitted correctly; canonical execution identity; no alternate scheduler or worker authority | The existing gate must remain the only CREATE execution gate | Historical PASS and one production gate class | D03-A16 after implementation | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-TRC-01`..`12` | **PASS**. Including reconstructable causal chain; parent-child causality; attributable tool, provider, model, side effect, policy, and profile; restart continuity; terminal outcome linked to evidence; configured versus effective provenance; transport identity is not runtime identity | Transitions must be attributable without heuristic joins | Historical PASS. No anchor trace | Redacted correlation across primary, witness, and dispatch | Later authorized D03-A qualification. PASS is not extended |
| `FRZ-OBS-01`..`07` | **OPEN**. Including a canonical event spine; observability does not mint execution truth; diagnostics interpret facts only; correlation IDs propagate; terminal state and side effects are observable; diagnostic provenance is reconstructable | Audit logs must not become the current head | Canonical observability split in platform docs | Sink schema, loss behavior, and proof that logs cannot authorize CREATE | Later authorized D03-A qualification |
| `FRZ-PRD-08` | **OPEN**. Wording: production-only bypass count = 0 | No second production CREATE path | Proposal route statement. Not re-proven by a new reachability suite in this task | Closed-world route proof on the implementation SHA | Later authorized D03-A qualification |
| `FRZ-REG-02` | **OPEN**. Wording: every corrected blocker has a regression gate | Each D03 blocker needs a future gate | Specifications in §10 only | Executed gates tied to an implementation SHA | Later authorized D03-A qualification |
| `FRZ-REG-03` | **OPEN**. Wording: negative tests prove gates detect violations | Rollback, stale fence, unavailable anchor, and bypass need negative tests | Specifications in §10 only | Executed negative tests | Later authorized D03-A qualification |

`FRZ-REL-01`, `FRZ-REL-02`, `FRZ-REL-04`, and `FRZ-REL-06` through `FRZ-REL-11` are also **OPEN**. They are not given a new D03 disposition here. They are not promoted.

### 12.1 Exact checklist wording

The family table above uses these checklist acceptance sentences. Status is unchanged.

| ID | Acceptance wording | Status |
|---|---|---|
| FRZ-CFG-01 | every configurable provider/vendor/model/backend selection is derived from an explicit typed platform configuration/profile contract, not a hard-coded production implementation choice | PASS |
| FRZ-CFG-02 | missing or invalid required configuration cannot silently select a vendor, model, backend or provider; the mechanism fails closed or remains explicitly disabled | PASS |
| FRZ-CFG-03 | configured state ≠ effective state is mechanically enforced: installed/available/configured capability does not become effective without the sanctioned activation/composition path | PASS |
| FRZ-CFG-04 | each configurable concern has exactly one canonical configuration contract and exactly one sanctioned composition/selection owner | PASS |
| FRZ-CFG-05 | provider/vendor/model/backend-specific endpoints, deployments, regions, credentials or secret references are supplied through approved configuration/secret boundaries and are not embedded as production semantic choices in core code | PASS |
| FRZ-CFG-06 | an unconfigured provider/vendor/plugin/backend cannot be reached through an alternate factory, fallback, direct constructor or compatibility path | PASS |
| FRZ-CFG-07 | effective configuration resolution is deterministic and traceable across all supported configuration scopes; precedence is explicit and no hidden ambient state silently overrides the sanctioned configuration | PASS |
| FRZ-CFG-08 | regression gates mechanically prevent reintroduction of hard-coded provider/vendor/model/backend selection, silent fallback and activation without required configuration | PASS |
| FRZ-TEN-01 | canonical tenant identity where tenant scope applies | PASS |
| FRZ-TEN-02 | end-to-end tenant propagation across relevant semantic and runtime boundaries | PASS |
| FRZ-TEN-03 | child/downstream operations cannot silently switch or widen tenant scope (`child_tenant_scope ⊆ parent_tenant_scope`) | PASS |
| FRZ-TEN-04 | cross-tenant state/cache/persistence/memory/index access blocked without explicit governed cross-tenant semantics | PASS |
| FRZ-TEN-05 | provider selection, configuration, profiles and effective bindings preserve tenant scope | PASS |
| FRZ-TEN-06 | tenant-specific credentials/secrets references cannot resolve or reuse under another tenant context | PASS |
| FRZ-TEN-07 | evidence, events, traces and diagnostics preserve tenant identity; cannot establish truth for another tenant | PASS |
| FRZ-TEN-08 | checkpoint, retry, recovery, replay and restore preserve original tenant identity | PASS |
| FRZ-TEN-09 | legal cross-tenant actions require explicit typed contract + explicit authority + explicit policy + auditable evidence; no implicit admin/global bypass | PASS |
| FRZ-TEN-10 | missing/invalid tenant must not silently become global, shared, default tenant, system tenant or provider-global scope where tenant identity is required | PASS |
| FRZ-TEN-11 | runtime extensions/providers/plugins cannot mint, rewrite or widen tenant scope beyond explicitly supplied authority | PASS |
| FRZ-TEN-12 | executable adversarial tests prove tenant isolation across execution, state, provider/config, evidence and recovery critical paths | PASS |
| FRZ-STA-01 | exactly one truth owner per state family | PASS |
| FRZ-STA-02 | no duplicate semantic state stores | PASS |
| FRZ-STA-03 | atomic/transaction boundaries explicit | PASS |
| FRZ-STA-04 | tenant isolation | PASS |
| FRZ-STA-05 | stale state rejected | PASS |
| FRZ-STA-06 | configured/effective/persisted state distinction | PASS |
| FRZ-STA-07 | policy artifacts durable where required | PASS |
| FRZ-STA-08 | checkpoints are state not identity authority | PASS |
| FRZ-REC-01 | crash recovery deterministic | PASS |
| FRZ-REC-02 | restart recovery preserves authority | PASS |
| FRZ-REC-03 | resume does not mint new identity incorrectly | PASS |
| FRZ-REC-04 | replay cannot create divergent truth | PASS |
| FRZ-REC-05 | fork semantics explicit | PASS |
| FRZ-REC-06 | partial persistence fails closed where required | PASS |
| FRZ-REC-07 | external operation uncertainty modeled | PASS |
| FRZ-REC-08 | backup/restore responsibility and supported semantics explicit | PASS |
| FRZ-REC-09 | restore preserves canonical identity, authority, tenant isolation and exactly-one truth | PASS |
| FRZ-REC-10 | partial/corrupt durable state is detected and cannot silently become accepted canonical truth | PASS |
| FRZ-GOV-01 | Governance != Execution | PASS |
| FRZ-GOV-02 | proposal != permission != execution | PASS |
| FRZ-GOV-03 | child_authority subset parent_authority | PASS |
| FRZ-GOV-04 | downstream scope narrowing only | PASS |
| FRZ-GOV-05 | fresh authorization before meaningful effects | PASS |
| FRZ-GOV-06 | missing HITL never approval | PASS |
| FRZ-GOV-07 | required evidence before work | PASS |
| FRZ-GOV-08 | policy/profile revision attributable | PASS |
| FRZ-GOV-09 | runtime extensions cannot expand authority | PASS |
| FRZ-GOV-10 | control-plane mutation governed | PASS |
| FRZ-EXE-01 | exactly one legal platform execution authority | PASS |
| FRZ-EXE-02 | all meaningful executable work enters canonical boundary | PASS |
| FRZ-EXE-03 | Nexus remains private implementation | PASS |
| FRZ-EXE-04 | Agent != Execution | PASS |
| FRZ-EXE-05 | child work admitted correctly | PASS |
| FRZ-EXE-06 | execution identity canonical | PASS |
| FRZ-EXE-07 | no alternate scheduler/worker execution authority | PASS |
| FRZ-TRC-01 | execution causal chain reconstructable | PASS |
| FRZ-TRC-02 | parent-child causality reconstructable | PASS |
| FRZ-TRC-03 | tool invocation attributable | PASS |
| FRZ-TRC-04 | provider invocation attributable | PASS |
| FRZ-TRC-05 | model call/context decision attributable | PASS |
| FRZ-TRC-06 | side effect attributable to authorization | PASS |
| FRZ-TRC-07 | policy revision attributable | PASS |
| FRZ-TRC-08 | profile revision attributable | PASS |
| FRZ-TRC-09 | restart/resume preserves trace continuity | PASS |
| FRZ-TRC-10 | terminal outcome linked to causal evidence | PASS |
| FRZ-TRC-11 | configured/effective provenance reconstructable | PASS |
| FRZ-TRC-12 | transport identity != runtime identity and mapping attributable | PASS |
| FRZ-OBS-01 | canonical event/evidence spine | OPEN |
| FRZ-OBS-02 | Observability does not mint execution truth | OPEN |
| FRZ-OBS-03 | Diagnostics interprets facts only | OPEN |
| FRZ-OBS-04 | correlation IDs propagated | OPEN |
| FRZ-OBS-05 | terminal state observable | OPEN |
| FRZ-OBS-06 | side-effect paths observable | OPEN |
| FRZ-OBS-07 | diagnostic provenance reconstructable | OPEN |
| FRZ-REL-01 | timeout ownership explicit | OPEN |
| FRZ-REL-02 | retry ownership explicit | OPEN |
| FRZ-REL-04 | cancellation propagates correctly | OPEN |
| FRZ-REL-06 | provider failure typed | OPEN |
| FRZ-REL-07 | no hidden infinite retry/fallback | OPEN |
| FRZ-REL-08 | bounded concurrency and work-admission semantics explicit | OPEN |
| FRZ-REL-09 | queue/backpressure and overload behavior explicit | OPEN |
| FRZ-REL-10 | resource saturation/exhaustion has deterministic reject/degrade/fail-closed behavior | OPEN |
| FRZ-REL-11 | provider throttling/rate-limit handling cannot create unbounded retry/fallback or bypass | OPEN |

`FRZ-OWN-*`, `FRZ-BND-*`, `FRZ-CTR-*`, `FRZ-TYP-*`, `FRZ-PLG-*`, `FRZ-RPL-*`, `FRZ-SEC-01`, `FRZ-SEC-04`, `FRZ-REL-03`, `FRZ-REL-05`, `FRZ-PRD-08`, `FRZ-REG-02`, and `FRZ-REG-03` use the exact wording already quoted in the family table. All of those rows are **OPEN**.

## 13. ECAP Change Delta Audit

| Step | Result |
|---|---|
| A1 | Changed qualification markdown and one roadmap ledger reference. No public code symbol. |
| A2 | Documentation links only. No import, composition wiring, or configuration key. |
| A3 | No production caller or downstream consumer. |
| A4 | Semantic owners unchanged: governance wiring, PIS, GCA, and the existing dispatch gate. No new composition owner. |
| A5 | Production-effective reachability: **no**. |
| A6 | Families in §12. No `FRZ-*` status change. |
| A7 | Historical PASS remains scoped to its accepted SHA. It does not cover this mechanism. |
| A8 | No regression gate executed. Future catalogue remains D03-A01–A23. `pytest = 0`. |

This delta does not certify the platform or the freeze.

### 13.1 Enterprise audit matrix

| Dimension | Finding | State | Classification |
|---|---|---|---|
| Boundaries | No layer or import change. Future adapter boundary is unspecified in code. | Design only | **IN-SCOPE BLOCKER** for a future implementation; not introduced by this delta |
| Communication | No new runtime channel. Cross-cloud calls are undesigned in code. | Not applicable to this delta | **NOT APPLICABLE — WITH EVIDENCE** for the markdown delta. Future path: **EXECUTION EVIDENCE REQUIRED** |
| Composition | No composition wiring. | Unchanged | **NOT APPLICABLE — WITH EVIDENCE** for this delta |
| Ownership | §2 symbols still match the single-owner model. | Document check only | **IN-SCOPE BLOCKER** until a ratified implementation is audited |
| Contracts | No contract added. | Unchanged | **IN-SCOPE BLOCKER** for OD-D03-A-08 ratification |
| Abstraction | Vendor pair remains outside platform contracts in the proposal. | Design only | **IN-SCOPE BLOCKER** until code exists |
| Typing | No typed model added. | Unchanged | **TRACKED FREEZE DEBT** (`FRZ-TYP-*` stays OPEN) |
| Pluginability | No provider registration. | Unchanged | **TRACKED FREEZE DEBT** (`FRZ-PLG-*` stays OPEN) |
| Replaceability | RPL-A..H absent. | Not demonstrated | **TRACKED FREEZE DEBT** for global RPL; **IN-SCOPE BLOCKER** for any portability claim |
| Bypass resistance | No new path. Future bypass is untested. | Current delta adds none | **IN-SCOPE BLOCKER** for implementation ratification |
| Governance | Governance does not gain a durable continuity store in this delta. | Unchanged | **NOT APPLICABLE — WITH EVIDENCE** for this delta |
| Execution authority | Production gate class remains the one class found. Callers were not re-inventoried. | Symbol check | **IN-SCOPE BLOCKER** for a future second path; not observed in this delta |
| Evidence and traceability | This document is the evidence. It is not runtime trace. | Documentation | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** for future audit exports |
| Fail-closed behavior | Required by the accepted direction and by §7.3. Not implemented. | Design | **IN-SCOPE BLOCKER** |
| Regression protection | No test run. | `pytest = 0` | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** and still an **IN-SCOPE BLOCKER** because the D03-A conditions are unmet |
| Tenant isolation | Documentation verdict in §5.4. Future isolation unproven. | Split verdict | Documentation: **NOT APPLICABLE — WITH EVIDENCE**. Future mechanism: **IN-SCOPE BLOCKER** |

### 13.2 Unresolved findings

| Finding | Classification |
|---|---|
| OD-D03-A-03 through OD-D03-A-08 are not ratified | **IN-SCOPE BLOCKER** |
| No inspected IAM or administrator graph | **IN-SCOPE BLOCKER** |
| Ten-year retention is not justified by a cited duty, and backup ages are unknown | **IN-SCOPE BLOCKER** |
| Prices, capacity, and SLO measurements are absent | **IN-SCOPE BLOCKER** |
| D03-A01–A23 have not been executed; managed-service behavior needs a future authorized sandbox | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED**. The unmet acceptance condition remains an **IN-SCOPE BLOCKER** |
| OP-TRUST-07 writer compromise is outside immutability | **IN-SCOPE BLOCKER** until the operator accepts the residual risk or requests an architecture decision |
| Global `FRZ-*` rows listed OPEN, and historical PASS rows, are not certified for this mechanism. `R7-Q`, `PROD-Q`, and `IMPL-01` stay open | **TRACKED FREEZE DEBT** |
| RPL-A..H and `FRZ-RPL-*` are not demonstrated | **TRACKED FREEZE DEBT** |
| D03-B global PIS and D03-C cross-store continuity remain unmet | **IN-SCOPE BLOCKER** as existing independent stages. This package does not open a new child task for them |
| RA-GZRS unplanned failover can expose a prefix | **IN-SCOPE BLOCKER** for any design that would resume CREATE from that prefix. The recommended rule is quarantine |

No finding is self-certified as enterprise compliance.

## 14. Recommended roadmap position

Do not close D03-A. Do not change the conditional preferred-pair status. Do not start D03-B or D03-C. Do not authorize implementation.

Recommended status of this artifact only:

**READY FOR AUDIT — D03-A RATIFICATION READINESS — OPERATOR DECISIONS PENDING**

The next operator actions are the six decisions in §11. After an independent exact-SHA audit, the operator may accept or reject each decision. Acceptance of a decision still does not provision cloud resources or modify production code.

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
