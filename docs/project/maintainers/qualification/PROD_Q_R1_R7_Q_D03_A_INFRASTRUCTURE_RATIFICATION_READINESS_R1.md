# D03-A — Infrastructure ratification readiness R1 (OD-D03-A-03–08)

| Field | Value |
|---|---|
| Status | **READY FOR AUDIT — D03-A POLICY CONSOLIDATION — ACCEPTED POLICY IS NOT INFRASTRUCTURE PROOF** |
| Date | 2026-10-18 |
| Branch | `development` |
| Inspected baseline | `0ba7d5bfd5324fe66d5cd990ff2dafba2ee4d39a` |
| Policy consolidation baseline | `2ba07f207c9b8ff93788c447886710c60f63ab86` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN/BLOCKED |
| Stage | `D03-A / OD-D03-A-03–08` |
| Design source | [Trusted anchor backend and trust domain proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) |
| Preferred-pair record | [Conditional infrastructure direction](PROD_Q_R1_R7_Q_D03_A_CONDITIONAL_INFRASTRUCTURE_DIRECTION_OPERATOR_DECISION.md) |
| Class decision | [OD-D03-A-01 ELCR-IIRW](PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md) — unchanged |
| Production delta | **0** |
| FRZ promotion | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |
| OD-D03-A-06 inventory | Prepared against `12092a111b6c64fce2ea4a7aa45e9728ce787649`. Rule and classifications are in §7.5. No duration is adopted. The decision stays unresolved. |
| Qualification preparation | [Appendix](PROD_Q_R1_R7_Q_D03_A_INFRASTRUCTURE_QUALIFICATION_PREPARATION.md). **INDEPENDENTLY ACCEPTED — SCOPED PREPARATION ONLY** at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`. Audit basis: exact-SHA GitHub review of that documentation delta; no independent cloud-test execution; no vendor-runtime qualification; no cloud resources created; no approval to incur cloud costs. Not infrastructure proof. Not provisioning authority. |

This record consolidates operator decisions OD-D03-A-03 through OD-D03-A-08. It does not copy the technical proposal. Accepted policy and accepted architectural direction are not proof of cloud infrastructure, not ratification of a backend, and not authorization to implement, provision, purchase, migrate, add a contract, or open a runtime port. The qualification preparation appendix is the decision package for a later, separately authorized sandbox. It does not change §11.

OD-D03-A-01 stays ratified as a mechanism class only. OD-D03-A-02 stays a conditional preferred service choice: Cloud Spanner Enterprise Plus base configuration `eur3`, plus Azure Blob Storage general-purpose v2 Standard, RA-GZRS, primary region North Europe, immutable append-blob witness. No other technology is selected. OD-D03-A-09 and OD-D03-A-10 receive no disposition.

D03-A-LOCAL-01 stays **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `302e7fa74b745d10f9552df0a6eb9c702d5ce033`. This consolidation does not reopen it and does not change that implementation. Section 2 is the pre-LOCAL-01 inspection at `0ba7d5bfd5324fe66d5cd990ff2dafba2ee4d39a`. It is not a new inventory.

A recommendation the operator has not accepted remains a recommendation. The dispositions in §11 are the operator decisions for this consolidation.

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
| `ACCEPTED POLICY` | The operator accepted the policy. It is not cloud infrastructure, IAM, or durability proof. |
| `ACCEPTED ARCHITECTURAL DIRECTION` | The operator accepted the direction for a later implementation. It does not authorize that implementation and does not prove replaceability. |
| `OPERATOR CONFIRMATION REQUIRED` | The fact is an organizational or policy choice. It is not true until the operator records it. |
| `EXECUTION EVIDENCE REQUIRED` | The property can be shown only by configuration inspection or a controlled test. It has not been run. |
| `BLOCKED` | A mandatory D03-A condition is unmet. |
| `NOT APPLICABLE — WITH EVIDENCE` | The property does not apply to this delta, and the reason is recorded. |

Assumptions are never written as guarantees. Accepted policy is not infrastructure ratification.

## 4. OD-D03-A-03 — Trust-domain independence

### 4.1 Required guarantee

No single principal, group, CI identity, federation administrator, emergency principal, or inherited organization role can modify both the primary register and the witness, restore both histories, impersonate both runtime identities, replace both trusted resource endpoints, weaken witness retention, or suppress or rewrite the audit evidence of both domains.

PIS/GCA backup and restore authority is a third domain. It cannot mutate the anchor or the witness.

### 4.2 Recommended decision

The operator has accepted the §4.3 separation policy and OP-TRUST-07 as residual risk. That acceptance is not evidence that such administrators exist and is not proof of real administrative independence.

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
| Operator decision required | Recorded. Rejection of OP-TRUST-07 would have been `STOP — ARCHITECTURE DECISION REQUIRED`. That rejection was not made. |
| Evidence state | Policy: **ACCEPTED POLICY**. OP-TRUST-07: accepted residual risk, conditional on later proof of real administrative independence. Vendor IAM primitives: **DOCUMENTED**. Configured graph: **BLOCKED**. Not cloud infrastructure proof. |

## 5. OD-D03-A-04 — History scope and tenant isolation

### 5.1 Required guarantee

The continuity history is environment-scoped. A shared environment anchor does not authorize one tenant to read or write another tenant's PIS, GCA, or continuity evidence.

Tenant-level independent histories stay inactive until PIS and GCA have proven tenant-specific recovery and cross-tenant isolation for this mechanism.

Missing or conflicting environment identity, and missing or conflicting tenant identity where a tenant is required, denies. Restore must preserve the original environment and any proven tenant identity. Kubernetes restore must not become a new authority namespace.

### 5.2 Recommended decision

The operator has accepted environment-scoped history as the only active scope, with mandatory tenant isolation. Tenant-specific anchors stay inactive. That acceptance is not runtime isolation proof.

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
| Operator decision required | Recorded. |
| Evidence state | Policy: **ACCEPTED POLICY** — environment-scoped history, mandatory tenant isolation, tenant-specific anchors inactive. Runtime isolation for this mechanism: **EXECUTION EVIDENCE REQUIRED**. Not cloud infrastructure proof. |

## 6. OD-D03-A-05 — IAM and workload authorization

### 6.1 Required guarantee

Runtime readers, transition writers, provisioning, reconciliation, security revocation, backup/restore, witness custodians, and independent auditors have distinct minimum privileges. Both clouds use independent short-lived workload federation. Static cloud credentials, runtime Shared Key, SAS, broad administrator permission, and unrestricted cross-scope access are forbidden.

### 6.2 Recommended decision

The operator has accepted §6.3 as the privilege baseline, including separated operational capabilities, short-lived workload federation, and the prohibition of ordinary runtime administrator credentials. Actual IAM remains unproven. Do not invent project IDs, principals, or role resource names.

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
| Operator decision required | Recorded. A second trusted transition service was not requested. |
| Evidence state | Policy: **ACCEPTED POLICY** — minimum privilege, separated operational capabilities, short-lived workload federation, and prohibition of ordinary runtime administrator credentials. Actual IAM: **BLOCKED** / **EXECUTION EVIDENCE REQUIRED**. Not cloud infrastructure proof. |

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

The operator did not adopt a production duration. Non-weakening configurable retention is preserved. Ten years is not adopted. Do not treat `ELCR-ACTIVE` as already adopted.

Recommend that the operator:

1. Keep deletion protection as a locked time-based immutable policy plus a separate operational preservation hold that remains until dual-controlled decommission. That hold is not evidence of a legal duty. A legal-purpose hold is a further control, used only when a legal or contractual duty is declared.
2. Set the numeric retention only from the §7.5 rule, after every required input is declared. The number must exceed the longest applicable restorable age plus the investigation window, and it must meet any applicable legal or contractual minimum. No input in §7.5 is known, so no number is adopted.
3. Treat the proposal's ten-year figure as an unadopted candidate, not as a regulatory finding.

If the operator cannot supply those ages and still wants a number, the number is an explicit policy choice and must be labeled as such.

### 7.3 Required recovery rules

The operator preserves these safety and recovery invariants. They are not tested, and they do not set a production duration, price, capacity, or achieved SLO.

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
| Exact missing evidence | The five named inputs in §7.5. No duration is adopted. Also still missing: legal-hold owner; proof the policy is locked; rollover and crash tests; DR test showing prefix loss does not resume CREATE. |
| Responsible authority | Operator for duration, hold, and decommission. Later authorized sandbox qualification for recovery behavior. |
| Acceptance criteria | Operator records a duration justified by the §7.5 rule, or explicitly adopts an unjustified number as policy. Operator accepts every recovery rule in §7.3, including no restore-in-place and no zero-loss claim for Azure geo-replication. |
| Rejection criteria | Ratifying ten years as if a statute required it; treating secondary reads as current; automatic CREATE after an unknown append or a regional failover. |
| Blocker classification | **IN-SCOPE BLOCKER** |
| Operator decision required | Production duration remains unresolved. No duration is adopted here, including ten years. |
| Evidence state | Non-weakening configurable retention: **preserved**. LOCAL-01 already rejects lowering the duration, lowering the ceiling, or clearing the hold. Production duration: **OPERATOR CONFIRMATION REQUIRED**. Configured WORM and recovery drills: **EXECUTION EVIDENCE REQUIRED**. Not cloud infrastructure proof. |

### 7.5 Retention decision model

This subsection is the OD-D03-A-06 decision inventory. It does not resolve the decision, does not adopt a duration, and does not authorize provisioning or implementation.

#### 7.5.1 Closed-world inventory

Sources inspected for values: this record §4.3, §7.1–§7.4, and §8.2; the proposal §5.3, §6.1–§6.3, §8.3, §9.4, §16.1–§16.2, and §17; the conditional-direction record; LOCAL-01; PLATFORM-POLICY-CONFIG-01; `HistoryContinuityRetentionPolicy` in `intergrax/contracts/governance/history_continuity.py`. Local SQLite, tests, fixtures, sample configuration, and provider defaults are not duration sources.

| Topic | What the sources contain | Class |
|---|---|---|
| `PIS_MAX_RESTORABLE_AGE` | No production age. Proposal §16.1 records the current per-process SQLite PIS as not a global production history. §7.3 and proposal §8.3 require deny on PIS rollback and give no window. | **OPERATOR INPUT REQUIRED**. Local SQLite is **NOT APPLICABLE — WITH EVIDENCE** as a production duration. |
| `GCA_MAX_RESTORABLE_AGE` | No production age. The same recovery rules name GCA rollback and give no window. | **OPERATOR INPUT REQUIRED** |
| `ANCHOR_PRIMARY_MAX_RESTORABLE_AGE` | No production age. Proposal §6.1 and §17.1 cite Spanner backup and PITR capability and record no window. | **OPERATOR INPUT REQUIRED**. An unstated vendor default is not a policy. |
| Primary backup and PITR | Restore capability is cited. Restore-in-place is forbidden by §7.3. No backup retention and no PITR window is recorded. | Rule: **DOCUMENTED CURRENT POLICY**. Duration: **OPERATOR INPUT REQUIRED**. Vendor pages without a recorded window: **HISTORICAL / NOT VERIFIED** as a value. |
| `INVESTIGATION_WINDOW` | §7.1 names the window and states that it is not in the canonical record. | **OPERATOR INPUT REQUIRED** |
| `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM` | No statute and no contract is cited. §7.1 states that ten years is not a documented legal obligation. | **OPERATOR INPUT REQUIRED**. Absence is not zero and is not a declaration that no duty exists. |
| Ten-year witness retention and `ELCR-ACTIVE` | Proposal OP-TRUST-04 and §16.2 ask the operator to ratify them. §7.2 does not adopt them. | **HISTORICAL / NOT VERIFIED** |
| Local retention mechanism | `retention_seconds` must sit inside `ceiling_seconds`. `legal_hold` must be true. No permission can lower the duration, lower the ceiling, or clear the hold. The contract text says this is not a production period. PLATFORM-POLICY-CONFIG-01 states that the ceiling is not legal retention or WORM. | **DOCUMENTED CURRENT POLICY** for the non-weakening local mechanism. Not a production duration. |
| Environment retirement and recovery | §7.1 requires release of hold and retention only by dual control after formal retirement. §7.3 states the recovery rules. No retirement procedure and no named approvers are recorded. | Recovery rules: **DOCUMENTED CURRENT POLICY**, untested. Retirement procedure: **OPERATOR INPUT REQUIRED**. |
| Security investigation window | Same unset window as `INVESTIGATION_WINDOW`. | **OPERATOR INPUT REQUIRED** |
| Audit-sink retention | §8.2 treats cloud audit retention as a separate unquantified charge. It is not the witness duration. | **OPERATOR INPUT REQUIRED**, and not a substitute for witness retention. |
| Backup, restore, and decommission authorities | §4.3 records role classes only: Spanner administrators, Azure witness administrators under dual control, and a separate PIS/GCA backup role that must not hold anchor roles. §7.4 names the operator for duration, hold, and decommission. No person, group, or account is named. | Role separation: **DOCUMENTED CURRENT POLICY** as accepted policy, not as verified principals. Named operators: **OPERATOR INPUT REQUIRED**. |

#### 7.5.2 Controls that are not interchangeable

| Control | Meaning in this decision | Binding evidence |
|---|---|---|
| Time-based immutable retention | A locked duration. After it ends, deletion becomes possible. | Not adopted. Azure locked container retention is a cited vendor primitive, not a configured policy. |
| Operational preservation hold | Stops time-based expiry from removing evidence while the environment is still active. | Local `legal_hold=True` is this operational flag. It is not WORM and not a legal finding. |
| Dual-controlled environment decommission | The only stated path for releasing that preservation hold, after formal retirement, by separated authorities. | Required by §7.1. The procedure is not written and no approvers are named. |
| Legal-purpose hold | A hold for an actual legal or contractual matter. | Not declared. It is not created by naming the operational flag `legal_hold`, and not created by the unadopted label `ELCR-ACTIVE`. |

The selected pair is not reported as incapable of locked time-based retention plus a separate hold. Proposal §17.2 cites Azure container immutability for locked retention, protected appends, and legal holds. Those pages were not re-fetched for this inventory, and no configured account exists. That is missing execution evidence, not a reason to choose a weaker store. If a later authorized sandbox shows that append blobs cannot keep protected appends under both a locked time-based policy and a distinct hold, stop: `STOP — ARCHITECTURE DECISION REQUIRED`. Do not unlock retention, clear the local hold, shorten witness history, or change provider inside this gate.

#### 7.5.3 Calculation rule

```text
longest_restorable_age = max(
    PIS_MAX_RESTORABLE_AGE,
    GCA_MAX_RESTORABLE_AGE,
    ANCHOR_PRIMARY_MAX_RESTORABLE_AGE,
)
required_time_based_retention > longest_restorable_age + INVESTIGATION_WINDOW
```

When `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM` is declared and applicable, `required_time_based_retention` must also be greater than or equal to that minimum. A missing input is not zero. The rule is not evaluated. No duration is proposed for approval.

The operational preservation hold remains in force for the life of the active environment even after a future duration is chosen. Decommission does not become legal-hold release. A declared legal duty adds a legal-purpose hold; it does not rename the operational flag.

#### 7.5.4 Mandatory recovery and deletion guarantees

§7.3 stays in force, including all of the following:

- Primary rollback denies and quarantines. Restore goes to a new database and a new authority incarnation. There is no restore-in-place.
- Witness rollback, including a geo-replicated prefix, denies and quarantines. Prefix loss does not resume CREATE.
- An unknown append outcome requires an exact read-back. Timeout is not success.
- A partial transition resolves only that same transition identity.
- A missing witness does not allow CREATE.
- Recovery does not silently restore an older authority incarnation and does not bypass the latest verified witness.
- Retention expiry does not silently remove required evidence while the environment remains active.

These guarantees are accepted design rules. They are not drilled.

#### 7.5.5 Operator decision, future evidence, and status

Smallest missing set, all required before any number:

1. `PIS_MAX_RESTORABLE_AGE`
2. `GCA_MAX_RESTORABLE_AGE`
3. `ANCHOR_PRIMARY_MAX_RESTORABLE_AGE`
4. `INVESTIGATION_WINDOW`
5. `REQUIRED_LEGAL_OR_CONTRACTUAL_MINIMUM`, or an explicit operator declaration that no legal or contractual minimum applies, with the source of that declaration.

Future infrastructure qualification, not opened here: locked policy proof; a configured operational preservation hold that is labeled separately from any legal-purpose hold; a dual-controlled decommission record; crash, rollover, and unknown-append tests; a disaster-recovery test that prefix loss does not resume CREATE and that an older incarnation does not become current.

OD-D03-A-06 stays unresolved. The parent conditional direction stays **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**. D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, and every global `FRZ-*` stay open. New `FRZ-*` PASS = 0. No architecture stop is reached: the gap is missing operator inputs and missing cloud proof, and the accepted local non-weakening checks stay unchanged.

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
| Operator decision required | Prices, capacity, and achieved SLO remain unresolved. No production numeric parameter is adopted. |
| Evidence state | Safety and recovery invariants: **preserved**. They are not weakened and they are not measured here. Meter structure and `eur3` topology: **DOCUMENTED**. Prices, capacity, and achieved SLO: **OPERATOR CONFIRMATION REQUIRED** and **EXECUTION EVIDENCE REQUIRED**. The 5-second, 500-millisecond, and 4-hour figures stay unadopted. Not cloud infrastructure proof. |

## 9. OD-D03-A-08 — Contracts, composition, and replaceability

### 9.1 Required guarantee

A future implementation, if separately authorized, uses typed platform-owned identity and continuity contracts, sanctioned production configuration, one approved composition owner, vendor-specific adapters under those contracts, and the existing CREATE dispatch gate. Unknown, missing, or mismatched evidence fails closed.

### 9.2 Recommended decision

The operator has accepted the following constraints as architectural direction. Acceptance does not authorize code and does not demonstrate replaceability:

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
| Operator decision required | Recorded. Acceptance does not authorize code. |
| Evidence state | Direction: **ACCEPTED ARCHITECTURAL DIRECTION** — vendor-neutral typed contracts, one sanctioned governance/composition path, controlled configuration, structural replaceability, no duplicate PIS/GCA, no new CREATE gate, fail-closed defaults. Implemented replaceability: **BLOCKED**. RPL-A..H: **EXECUTION EVIDENCE REQUIRED**. Not cloud infrastructure proof. |

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

| Decision | Disposition | Infrastructure proof? | Still required |
|---|---|---|---|
| OD-D03-A-03 | **ACCEPTED POLICY.** Independent administrative trust domains. OP-TRUST-07 is accepted residual risk, conditional on later proof of real administrative independence. | No | Expanded identity graph. Rejection of OP-TRUST-07 was not made. |
| OD-D03-A-04 | **ACCEPTED POLICY.** Environment-scoped history with mandatory tenant isolation. Tenant-specific anchors stay inactive. | No | Execution evidence of isolation for this mechanism. |
| OD-D03-A-05 | **ACCEPTED POLICY.** Minimum privilege, separated operational capabilities, short-lived workload federation, and prohibition of ordinary runtime administrator credentials. | No | Actual IAM. No second transition service is authorized. |
| OD-D03-A-06 | Configurable retention with non-weakening protection is **preserved**. Production duration is **unresolved**. §7.5 classifies all five duration inputs as operator-required and adopts no number. Ten years is not adopted. | No | The five §7.5 inputs, then a duration computed from that rule, or an explicit labeled policy choice. Neither is recorded here. |
| OD-D03-A-07 | Safety and recovery invariants are **preserved**. Prices, capacity, and achieved SLO are **unresolved**. No production numeric parameter is adopted. | No | Current quotes, a capacity choice, and an explicit accept or reject of each §8.4 target without weakening safety. |
| OD-D03-A-08 | **ACCEPTED ARCHITECTURAL DIRECTION.** Vendor-neutral typed contracts, one sanctioned governance/composition path, controlled configuration, structural replaceability, no duplicate PIS/GCA, no new CREATE gate, fail-closed defaults. | No | Code, configuration binding, and RPL-A..H. This acceptance does not authorize that work. |

Accepted policy does not authorize implementation. Remaining execution evidence stays on a later, separately authorized qualification gate and must follow canonical roadmap order. This document does not open that gate. Cloud IAM, vendor integration, costs, disaster recovery, and production SLOs stay on that dependent gate.

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
| Contracts | No contract added. | Unchanged | OD-D03-A-08 direction accepted. Implementation and replaceability remain an **IN-SCOPE BLOCKER** |
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
| OD-D03-A-03, OD-D03-A-04, and OD-D03-A-05 are accepted policy only. OD-D03-A-08 is an accepted architectural direction only. None is cloud infrastructure proof | **IN-SCOPE BLOCKER** for infrastructure ratification and parent closure. The policy choice itself is no longer waiting |
| No inspected IAM or administrator graph. Real administrative independence is unproven | **IN-SCOPE BLOCKER** |
| OD-D03-A-06 production duration is unresolved. §7.5 leaves all five duration inputs operator-required and adopts no number. Ten years is not adopted. Non-weakening retention stays in force | **IN-SCOPE BLOCKER** for a production duration. Not a license to weaken LOCAL-01 retention checks |
| OD-D03-A-07 prices, capacity, and achieved SLO are absent. Safety and recovery invariants stay in force. No production numeric parameter is adopted | **IN-SCOPE BLOCKER** for budget and SLO acceptance |
| D03-A01–A23 have not been executed; managed-service behavior needs a future authorized sandbox | **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED**. The unmet acceptance condition remains an **IN-SCOPE BLOCKER** |
| OP-TRUST-07 is an accepted residual risk, conditional on later proof of real administrative independence | **MISSING EXECUTION EVIDENCE** for that proof. Not an architecture stop |
| Global `FRZ-*` rows listed OPEN, and historical PASS rows, are not certified for this mechanism. `R7-Q`, `PROD-Q`, and `IMPL-01` stay open | **TRACKED FREEZE DEBT** |
| RPL-A..H and `FRZ-RPL-*` are not demonstrated. Accepted structural replaceability is direction only | **TRACKED FREEZE DEBT** |
| D03-B global PIS and D03-C cross-store continuity remain unmet | **IN-SCOPE BLOCKER** as existing independent stages. This package does not open a new child task for them |
| RA-GZRS unplanned failover can expose a prefix | **IN-SCOPE BLOCKER** for any design that would resume CREATE from that prefix. The preserved rule is quarantine |
| `PROD-Q-R1-R7-Q-D03-A-PLATFORM-POLICY-CONFIG-01` was later accepted within child scope | **N/A — WITH EVIDENCE** for this readiness delta. The child is **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `02e48d9fa61d116346da613271cc5e8b77e08cc6`. Independent static review accepted that scoped configuration. Implementer tests were not independently rerun. That acceptance is not infrastructure proof |

No finding is self-certified as enterprise compliance.

## 14. Roadmap position

Do not close D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, or any global `FRZ-*`. Do not change OD-D03-A-01. Do not expand OD-D03-A-02 beyond the conditional preferred pair. Do not reopen D03-A-LOCAL-01. This readiness artifact does not itself authorize cloud integration or production activation.

Status of this artifact only:

**READY FOR AUDIT — D03-A POLICY CONSOLIDATION — ACCEPTED POLICY IS NOT INFRASTRUCTURE PROOF**

§15 names the platform-policy child. That child is **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `02e48d9fa61d116346da613271cc5e8b77e08cc6` and does not close this readiness gate. Cloud IAM, vendor integration, costs, disaster recovery, and production SLOs remain on the dependent qualification gate already identified by the conditional direction: trust-domain graph, retention duration, cost and budget, recovery objectives, and sandbox qualification.

## 15. Platform policy configuration child

| Field | Value |
|---|---|
| Proposed name | `PROD-Q-R1-R7-Q-D03-A-PLATFORM-POLICY-CONFIG-01` |
| Status | **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at `02e48d9fa61d116346da613271cc5e8b77e08cc6`. Independent static review only. Not cloud infrastructure proof. Does not close the parent. [Evidence](PROD_Q_R1_R7_Q_D03_A_PLATFORM_POLICY_CONFIG_01.md) |
| Parent | `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION` — remains OPEN / BLOCKED |
| Predecessor | This readiness consolidation, after independent exact-SHA audit. D03-A-LOCAL-01 stays closed at `302e7fa74b745d10f9552df0a6eb9c702d5ce033` and is not modified |
| Not a parent of | D03-B, D03-C, cloud IAM, vendor adapters, cost, disaster recovery, or production SLO |

Dependency graph, existing order preserved:

```text
OD-D03-A-01 class (ratified, unchanged)
  -> OD-D03-A-02 conditional preferred pair (unchanged, not full ratification)
       -> D03-A-LOCAL-01 (closed; do not modify)
       -> this policy consolidation (READY FOR AUDIT; not parent closure)
            -> PLATFORM-POLICY-CONFIG-01 (CLOSED — SCOPED / INDEPENDENTLY ACCEPTED at `02e48d9fa61d116346da613271cc5e8b77e08cc6`; not infrastructure proof)
                 -> dependent qualification gate (not opened):
                    trust-domain graph, IAM, vendor integration,
                    OD-D03-A-06 production duration,
                    OD-D03-A-07 prices, capacity, achieved SLO,
                    disaster-recovery drills, sandbox rows that need cloud accounts
D03-B and D03-C stay independent and BLOCKED
```

The child is the smallest platform slice that uses accepted OD-D03-A-03, OD-D03-A-04, OD-D03-A-05, and OD-D03-A-08 without a GCP or Azure account. It does not skip the cloud gate. It does not satisfy that gate.

### 15.1 Owners, files, and contracts

| Role | Binding |
|---|---|
| Semantic owner | `ProductionControlPlaneGovernance` |
| Composition | `compose_history_continuity_authority` in `intergrax/applications/_shared/history_continuity_composition.py`. No second composition owner |
| Contract | Additive typed configuration on `intergrax/contracts/governance/history_continuity.py` only. No Spanner or Azure types |
| Preserved local behavior | `intergrax/runtime/governance/local_history_continuity.py`. Accepted LOCAL-01 semantics stay, including non-weakening retention |
| Not callers | `GovernedProviderInvocationDispatchGate`, `open_production_process_lifecycle` |
| Unchanged stores | `ProviderInvocationStore` and GCA. No duplicate PIS or GCA |

Operator authorization 2026-10-18 implemented this shape on the existing contract and composition. This section does not add cloud proof. If that work had needed a second composition owner, a new CREATE gate, a duplicate PIS or GCA, or a change to accepted LOCAL-01 semantics: `STOP — ARCHITECTURE DECISION REQUIRED`. That stop was not reached.

### 15.2 Expected solution shape

Typed configuration is the only activation input to the existing composition function. Missing or invalid configuration returns the existing deny-only authority. The configuration cannot enable a tenant-specific anchor, cannot supply an ordinary runtime administrator credential, and cannot collapse the existing distinct grants into one administrator grant. Primary and witness administrative identifiers must both be present and distinct; equality or absence denies. That check is a configuration invariant, not an IAM export and not proof of administrative independence. Short-lived federation is not implemented; a static runtime credential is not accepted as a substitute. No vendor SDK, no production lifecycle call, and no CREATE call. No production retention duration, price, capacity, or SLO literal.

### 15.3 Exit criteria

All of the following, on the future implementation SHA, under a separate authorization:

1. Missing, disabled, or non-local configuration still returns deny-only authority.
2. A configuration that activates a tenant-specific anchor is rejected. Environment scope and the existing tenant-match rule stay in force.
3. Ordinary runtime administrator credentials cannot be selected. Existing grants `WRITE`, `INITIALIZE`, `RECONCILE`, `INVALIDATE`, and `RECOVER` stay distinct.
4. One configured principal cannot hold both primary-register administration and witness administration. No cloud IAM call is made, and the result is not claimed as administrative-independence proof.
5. Retention still cannot be lowered, the ceiling still cannot be lowered, and the hold still cannot be cleared. No production duration number is introduced.
6. No new CREATE gate, no call from the existing CREATE gate, no duplicate PIS or GCA, and no vendor type in the platform contract.
7. Contract-level tests cover the deny cases above. They are not Spanner, Azure, IAM, disaster-recovery, cost, or SLO evidence.
8. D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, and global `FRZ-*` stay open. RPL-A..H stay unclaimed. New `FRZ-*` PASS = 0.

Applicable `FRZ-*` for that future child, statuses unchanged by this record: `FRZ-OWN-01`..`05`, `FRZ-BND-01`..`06`, `FRZ-CTR-01`..`06`, `FRZ-TYP-01`..`06`, `FRZ-PLG-01`..`07`, `FRZ-CFG-01`..`08` (historical PASS, not extended), `FRZ-TEN-01`..`12` (historical PASS, not extended), `FRZ-SEC-01`, `FRZ-SEC-04`, `FRZ-RPL-01`..`04`.

### 15.4 Dependent qualification gate

Not opened. It keeps cloud IAM and the real administrative-independence graph, vendor-adapter integration, OD-D03-A-06 production duration, OD-D03-A-07 prices, capacity, and achieved SLO, and disaster-recovery drills, including sandbox rows that require GCP or Azure accounts. D03-A01–A23 stay unexecuted.

## 16. ECAP delta for this consolidation

| Step | Result |
|---|---|
| A1 | Qualification markdown and the roadmap ledger sentences for this gate, the conditional-direction pointer, and the LOCAL-01 next-stage pointer. No production symbol. |
| A2 | No import, composition wiring, or configuration key. |
| A3 | No production caller. |
| A4 | Owners unchanged. LOCAL-01 composition is not edited. |
| A5 | Production-effective reachability: **no**. |
| A6 | Families in §12. No `FRZ-*` status change. |
| A7 | Historical PASS is not extended. LOCAL-01 acceptance is not reopened. |
| A8 | No regression gate. `pytest = 0`. |

| Finding ID | Class | FRZ | Severity | Label |
|---|---|---|---|---|
| D03-A-POL-01 | IN_SCOPE_BLOCKER | parent D03-A | Parent stays blocked | Accepted policy is not infrastructure proof |
| D03-A-POL-02 | MISSING_EXECUTION_EVIDENCE | FRZ-SEC-04 | Blocks cloud ratification | IAM and administrative independence are unproven |
| D03-A-POL-03 | IN_SCOPE_BLOCKER | OD-D03-A-06 / OD-D03-A-07 | Blocks production numbers | Duration, prices, capacity, and achieved SLO are unresolved. §7.5 adopts no retention number |
| D03-A-POL-04 | TRACKED_FREEZE_DEBT | FRZ-RPL-* | No portability claim | Structural replaceability is direction only |
| D03-A-POL-05 | N/A_WITH_EVIDENCE | n/a | n/a | Platform-policy child is later CLOSED — SCOPED at `02e48d9fa61d116346da613271cc5e8b77e08cc6`. That acceptance is not infrastructure proof. This readiness delta adds no code |

No finding is self-certified as enterprise compliance. Cursor status of this documentation commit is at most **READY FOR AUDIT**.

## 17. ECAP delta for the OD-D03-A-06 retention inventory

Prepared against `12092a111b6c64fce2ea4a7aa45e9728ce787649`. Changed files are this qualification record and one roadmap ledger sentence. No production symbol, import, composition wiring, configuration key, or caller. Production-effective reachability: **no**. Semantic owner remains `ProductionControlPlaneGovernance`. Composition remains `compose_history_continuity_authority`. Architecture and ownership delta: **0**. `pytest = 0`. New global `FRZ-*` PASS = **0**. Checklist statuses below are the current checklist column. Historical PASS is not extended. Documentation does not qualify runtime behavior.

| Step | Result |
|---|---|
| A1 | This markdown record and the roadmap sentence for this readiness row. No public code symbol. |
| A2 | No import, composition wiring, or configuration key. |
| A3 | No production caller. |
| A4 | Owners unchanged. LOCAL-01 and PLATFORM-POLICY-CONFIG-01 mechanisms are not edited. |
| A5 | Production-effective reachability: **no**. |
| A6 | Per-ID table below. No status change. |
| A7 | `FRZ-CFG-*` and `FRZ-TEN-*` historical PASS stay scoped to their accepted platforms. They do not cover this retention decision. |
| A8 | No regression gate. `pytest = 0`. |

Applicability: **CITED** means §7.5 states an obligation and does not satisfy it. **NOT-EXERCISED** means this delta does not reach that criterion. **PASS-NOT-EXTENDED** means the checklist PASS remains historical. Evidence level for every row: **DOC-ONLY**.

| ID | Checklist | Applicability | Evidence level |
|---|---|---|---|
| `FRZ-REL-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-03` | OPEN | CITED. Same-transition resolution is restated and not proven. | DOC-ONLY |
| `FRZ-REL-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-05` | OPEN | CITED. Unknown append requires exact read-back. Not proven. | DOC-ONLY |
| `FRZ-REL-06` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-07` | OPEN | CITED. No automatic resumption after prefix loss. Not proven. | DOC-ONLY |
| `FRZ-REL-08` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-09` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-10` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-REL-11` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-03` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-05` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-06` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-07` | OPEN | CITED. Ten years is not treated as a legal finding. The claim stays bounded and OPEN. | DOC-ONLY |
| `FRZ-SEC-08` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-SEC-09` | OPEN | CITED. Retention and deletion responsibility is stated as unresolved. | DOC-ONLY |
| `FRZ-SEC-10` | OPEN | CITED. Cloud retention configuration remains infrastructure evidence, not present. | DOC-ONLY |
| `FRZ-PRD-01` | OPEN | CITED. Local retention checks are not production qualification. | DOC-ONLY |
| `FRZ-PRD-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-03` | OPEN | CITED. No silent default duration is introduced. Fail-closed behavior is not newly qualified. | DOC-ONLY |
| `FRZ-PRD-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-05` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-06` | OPEN | CITED. Quarantine rules are restated and not qualified. | DOC-ONLY |
| `FRZ-PRD-07` | OPEN | CITED. Local SQLite is excluded as a production duration source. | DOC-ONLY |
| `FRZ-PRD-08` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-09` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-10` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-11` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-PRD-12` | OPEN | CITED. Recovery and decommission procedure is required and not written. | DOC-ONLY |
| `FRZ-OWN-01` | OPEN | NOT-EXERCISED. Owner unchanged. | DOC-ONLY |
| `FRZ-OWN-02` | OPEN | NOT-EXERCISED. No new contract. | DOC-ONLY |
| `FRZ-OWN-03` | OPEN | NOT-EXERCISED. Composition unchanged. | DOC-ONLY |
| `FRZ-OWN-04` | OPEN | NOT-EXERCISED. No shadow authority. | DOC-ONLY |
| `FRZ-OWN-05` | OPEN | NOT-EXERCISED. No duplicated mechanism. | DOC-ONLY |
| `FRZ-CTR-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CTR-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CTR-03` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CTR-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CTR-05` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CTR-06` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-03` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-05` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-TYP-06` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-03` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-04` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-05` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-BND-06` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-CFG-01` | PASS | PASS-NOT-EXTENDED. No production duration literal. | DOC-ONLY |
| `FRZ-CFG-02` | PASS | PASS-NOT-EXTENDED. Missing inputs stay unresolved rather than defaulted. | DOC-ONLY |
| `FRZ-CFG-03` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-CFG-04` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-CFG-05` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-CFG-06` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-CFG-07` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-CFG-08` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-01` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-02` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-03` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-04` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-05` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-06` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-07` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-08` | PASS | PASS-NOT-EXTENDED. Restore rules do not waive tenant identity and do not re-prove it. | DOC-ONLY |
| `FRZ-TEN-09` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-10` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-11` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-TEN-12` | PASS | PASS-NOT-EXTENDED | DOC-ONLY |
| `FRZ-RPL-01` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-RPL-02` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-RPL-03` | OPEN | NOT-EXERCISED | DOC-ONLY |
| `FRZ-RPL-04` | OPEN | NOT-EXERCISED | DOC-ONLY |

| Finding ID | Class | Label |
|---|---|---|
| D03-A-RET-01 | IN-SCOPE BLOCKER | All five §7.5 duration inputs are unknown. OD-D03-A-06 is not resolved. Parent D03-A stays blocked. |
| D03-A-RET-02 | TRACKED FREEZE DEBT | Global `FRZ-*` rows stay at the checklist status above. `FRZ-CFG-*` and `FRZ-TEN-*` PASS are not extended. IMPL-01, R7-Q, and PROD-Q stay open. |
| D03-A-RET-03 | ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED | Locked retention, the preservation hold, decommission, and §7.3 recovery drills have no infrastructure execution. |

No infrastructure qualification and no production execution occurred. No finding is self-certified as enterprise compliance. Cursor status of this documentation change is at most **READY FOR AUDIT**.

## 18. Qualification preparation appendix

Checked 2026-10-19 on `development` at `343cc5627f6c1e2a71395fd0e3038e06114185bd` before the appendix was added. The working tree was clean. The appendix is [PROD_Q_R1_R7_Q_D03_A_INFRASTRUCTURE_QUALIFICATION_PREPARATION.md](PROD_Q_R1_R7_Q_D03_A_INFRASTRUCTURE_QUALIFICATION_PREPARATION.md). It holds the vendor feasibility record, the unset sandbox budget, the trust-domain preparation, the unexecuted D03-A01–A23 specifications, the sandbox lifecycle, and the operator decision sheet. It does not adopt a retention duration, a capacity, a price total, or a spending maximum. It does not authorize provisioning. D03-A, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q, and global `FRZ-*` stay open. `RPL-A` through `RPL-H` stay unproven.

Independent review accepted that appendix as **INDEPENDENTLY ACCEPTED — SCOPED PREPARATION ONLY** at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`. The review was an exact-SHA documentation delta. It did not execute cloud tests, qualify vendor runtime behavior, create cloud resources, or approve cloud cost. This readiness status stays **READY FOR AUDIT — D03-A POLICY CONSOLIDATION — ACCEPTED POLICY IS NOT INFRASTRUCTURE PROOF**. The conditional-direction parent stays **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**.

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
