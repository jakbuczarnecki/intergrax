# D03-A — Conditional infrastructure direction — operator decision

| Field | Value |
|---|---|
| Status | **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** |
| Date | 2026-10-18 |
| Authorization amendment | 2026-10-19 — two classes: offline implementation authorized subject to independent audit; cloud execution not authorized |
| Branch | `development` |
| Audited proposal baseline | `f67f4e61c80f8b408c273fc8c8423c31a86cb152` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN/BLOCKED |
| Source design | [D03-A trusted anchor backend and trust domain proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) |
| Unchanged class decision | [OD-D03-A-01](PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md) |
| Production delta | **0** |
| FRZ promotion | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |
| Offline implementation | **AUTHORIZED — SUBJECT TO INDEPENDENT AUDIT**. Not implemented by this record |
| Cloud execution | **NOT AUTHORIZED** |
| Implementation child | `PROD-Q-R1-R7-Q-D03-A-OFFLINE-PROVIDER-IMPL-01` — **READY FOR AUDIT — OFFLINE CODE ONLY**. Not closed. Not cloud-qualified. Implementation baseline `a7ecc5065a0343e768584e818c844d3f96b5f533`. Evidence: [dossier](PROD_Q_R1_R7_Q_D03_A_OFFLINE_PROVIDER_IMPL_01.md) |

## Decision boundary

This record conditionally accepts a preferred technology direction after the audited proposal baseline. It is not unconditional ratification of OD-D03-A-02–08. It does not amend OD-D03-A-01. Cloud execution stays unauthorized: no provisioning, subscription purchase, account creation, live credentials, applied deployment, or runtime activation. Offline implementation of the selected adapters is a separate authorization class, recorded below, and is subject to independent audit of this amendment. That class does not certify the design, does not grant cloud execution, and does not close D03-A.

OD-D03-A-01 remains the authority that ELCR-IIRW is the ratified mechanism class and that no service was selected by that class decision. This later record accepts the audited service pair as the preferred direction only. The proposal remains the technical design source; its matrices are not copied here.

## Accepted

1. **Preferred service pair.** OD-D03-A-02 is accepted only as a preferred service choice: Google Cloud Spanner Enterprise Plus, base multi-region configuration `eur3`, as the primary register; Azure Blob Storage general-purpose v2 Standard, RA-GZRS, primary region North Europe, immutable append-blob design, as the witness.
2. **ELCR-IIRW** remains the already ratified mechanism class under OD-D03-A-01.
3. **Vendor-neutral platform contracts** remain mandatory. Service API types and credentials stay outside platform contracts.
4. **Fail-closed behavior** and **independent continuity protection** are non-negotiable.
5. **D03-B** and **D03-C** remain independent mandatory stages and stay blocked.

## Not yet ratified

The preferred pair does not ratify:

- actual project, subscription, tenant and administrator identities;
- independence of IAM, backup, restore and emergency-access domains;
- the exact retention period and legal-hold policy;
- concrete credentials, permissions and workload identity configuration;
- actual budget, capacity, SLO and operating costs;
- adapter correctness, region-failover behavior and disaster-recovery evidence;
- cloud execution, live infrastructure activation, and production qualification. Offline implementation is a separate authorization class below. It is not ratification of these items.

**OD-D03-A-03 through OD-D03-A-08 are not ratified.** A proposal option is not accepted by silence. OD-D03-A-09 and OD-D03-A-10, left open by OD-D03-A-01, receive no disposition here.

## Next gate

These items are outstanding decision evidence for cloud execution and full ratification. They are prerequisites to authorization class B, not to offline implementation, and they are not new roadmap stages.

1. **Trust domains:** the actual administrator, service-account, federation and emergency-access graph, with no hidden common authority.
2. **Retention:** justification of the retention period and legal-hold requirements.
3. **Cost:** realistic infrastructure sizing, recurring operating costs and budget approval.
4. **Availability:** confirmation of acceptable recovery objectives and of the consequences of cross-cloud outages.
5. **Qualification:** sandbox evidence requirements covering atomic updates, immutable witness append, uncertain outcomes, corruption, crash recovery and rollback.

If obtaining this evidence requires actual cloud resources, credentials, procurement, a new trusted transition service or an architectural change:

**STOP — OPERATOR DECISION / AUTHORIZATION REQUIRED**

That stop applies to class B evidence. It does not withdraw class A. Class A still stops if the work needs a new owner, a second CREATE gate, or a different backend.

## Status preserved

- Technology direction: accepted conditionally.
- Infrastructure: not fully ratified.
- D03-A: not closed. Final acceptance stays blocked. Offline provider child: **READY FOR AUDIT — OFFLINE CODE ONLY**. Not closed. Not cloud-qualified.
- D03-B and D03-C: blocked.
- Parent `IMPL-01`, `R7-Q`, `PROD-Q` and the global enterprise freeze: not closed.
- No `FRZ-*` status is promoted.

## Offline implementation authorization amendment

This amendment separates two activities that the earlier text of this record treated as one unopened gate. It is the later explicit authority anticipated by the proposal's statement that OD-D03-A-02–08 ratification does not itself authorize code, and it is the separate named grant required by the closed LOCAL-01 record before further implementation. It does not amend OD-D03-A-01, Decision-03, LOCAL-01, PLATFORM-POLICY-CONFIG-01, or preparation P1–P6. It does not create a second roadmap.

Parent identity remains `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION`. Parent status remains **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**. The preferred service pair and the accepted architectural direction are unchanged.

### Authorization class A — offline implementation

**AUTHORIZED — SUBJECT TO INDEPENDENT AUDIT.**

After independent acceptance of this amendment, pending work inside this parent may include:

- production-grade Python implementation of the selected cloud adapters;
- vendor SDK usage strictly inside provider implementation boundaries;
- adapter-specific typed configuration;
- integration through the existing platform contract;
- sanctioned dependency injection and provider selection;
- provider-specific error mapping into platform-defined typed outcomes;
- deterministic offline contract tests;
- fault injection without cloud resources;
- negative and adversarial local tests;
- static infrastructure-as-code templates, without applying them;
- IAM role definitions and federation configuration templates;
- deployment templates, without executing them;
- observability, diagnostics, and evidence hooks;
- documentation required to operate and qualify the implementation.

This class permits implementation inside the existing architectural direction. It does not certify that design, does not grant execution authority, and does not override a later architecture blocker. If the accepted direction cannot be met without a new owner, a second composition path, a second CREATE gate, or a different backend, stop with `STOP — ARCHITECTURE DECISION REQUIRED`.

The child that carries this scope is `PROD-Q-R1-R7-Q-D03-A-OFFLINE-PROVIDER-IMPL-01`. Its status after the offline implementation is **READY FOR AUDIT — OFFLINE CODE ONLY**. Not closed. Not cloud-qualified. Writing that code before independent acceptance of this amendment was not authorized. This amendment does not itself write that code. The implementation commit does not close this parent. The consolidated evidence dossier is [PROD_Q_R1_R7_Q_D03_A_OFFLINE_PROVIDER_IMPL_01.md](PROD_Q_R1_R7_Q_D03_A_OFFLINE_PROVIDER_IMPL_01.md). Implementation baseline `a7ecc5065a0343e768584e818c844d3f96b5f533`. A static review of that SHA accepted local offline behavior and architecture boundaries only. This decision still does not close the parent or the child.

### Authorization class B — cloud execution

**NOT AUTHORIZED.**

Still prohibited:

- creating or activating GCP or Azure accounts;
- provisioning cloud resources;
- incurring cloud expenditure;
- authenticating against real cloud control planes;
- configuring live IAM, RBAC, Entra, or federation;
- applying infrastructure-as-code;
- executing real Spanner or Azure test cases;
- locking real retention policies;
- running real fault injection or disaster recovery;
- enabling the cloud adapter on any production CREATE path;
- claiming production SLO, cloud durability, or recovery certification.

Preparation G1–G6 remain prerequisites to this class. This amendment grants none of them. It invents no budget, account ID, administrator, retention duration, or legal obligation.

### Binding design constraints

These existing responsibilities stay in force. No second owner is created.

| Responsibility | Existing owner |
|---|---|
| Semantic governance | `ProductionControlPlaneGovernance` |
| Sanctioned composition | `compose_history_continuity_authority` |
| Platform contract | `HistoryContinuityAuthority` |
| Activation policy | `HistoryContinuityActivationPolicy` |
| Mutation authorization | `ControlPlaneMutationAuthorizationBoundary` |
| Physical CREATE gate | `GovernedProviderInvocationDispatchGate` |
| Local provider | `local_deterministic` |

Eventual adapters stay behind `HistoryContinuityAuthority`. There is no second semantic governance owner, no second composition owner, no second CREATE gate, and no alternate PIS or GCA. Platform semantic contracts take no direct cloud SDK dependency. Platform consumers take no vendor-specific types. Semantic boundaries take no `Any`, generic `object`, reflective contract, stringly typed command, or dict-based pseudo-contract. No automatic provider fallback may downgrade safety. Every unsafe or unqualified activation attempt fails closed. `local_deterministic` remains the local provider. This amendment does not replace it and does not enable a cloud adapter on a production CREATE path.

### Future solution shape — not implemented here

#### Spanner primary register

Production-target primary register: Enterprise Plus, `eur3`; strong current reads; transactional CAS; revision monotonicity; fencing; idempotent transition identity; conservative handling of unknown commit results; explicit bounded recovery; no rollback acceptance; tenant and environment isolation.

#### Azure witness

Production-target immutable append history: GPv2 Standard Hot RA-GZRS; North Europe; conditional append-position enforcement; exact read-back after uncertain outcomes; predecessor and revision digest continuity; safe segment rollover; locked, non-weakening time-based retention; independent preservation hold; fail-closed geo-failover recovery; no positive CREATE authorization from a secondary or a stale prefix.

Corrected proposal §6.2.1 Azure protected-append configuration is a candidate pending real vendor verification. It is not a locked production policy and it adopts no duration. OD-D03-A-06 stays unresolved.

#### Integration

Vendor-neutral platform contracts. Explicit provider selection. Typed configuration. Strict lifecycle boundaries. Existing governance-backed authorization. No physical CREATE activation before a separately authorized later integration gate.

#### Qualification

Offline unit, contract, and adversarial tests may provide code-level evidence. They must never be labeled actual Spanner, Azure, or independent administrative-domain proof. Real qualification remains D03-A01–A23 at a future authorized cloud gate. RPL-A–H remain unqualified for the actual cloud implementations. Historical `FRZ-CFG-*` and `FRZ-TEN-*` PASS is not evidence for the new cloud adapter.

### Offline provider implementation child

| Field | Value |
|---|---|
| ID | `PROD-Q-R1-R7-Q-D03-A-OFFLINE-PROVIDER-IMPL-01` |
| Status | **READY FOR AUDIT — OFFLINE CODE ONLY**. Not closed. Not cloud-qualified |
| Parent | This record. Parent final acceptance stays blocked |
| Starts when | Independent audit accepts this authorization amendment |
| In scope | The class A list, the binding constraints, and the future solution shape above |
| Out of scope | Class B; D03-B; D03-C; production CREATE activation; closure of IMPL-01, R7-Q, PROD-Q, or any global `FRZ-*` |
| Evidence label | Offline code evidence only. Not cloud qualification |
| Implementation baseline | `a7ecc5065a0343e768584e818c844d3f96b5f533`. Lineage and the closed-world inventory are in the dossier, not in this authorization |
| Formal closure | Not available from this record. Independent audit of the reconciliation commit is still required |

No further discovery or decision round is required to identify this child. A new architecture blocker, if one appears during that work, stops that work. It does not widen this authorization.

### Controlling-record reconciliation

| Record | Effect of this amendment |
|---|---|
| This decision | Earlier blanket refusal of implementation is narrowed to cloud execution and unqualified production activation. Class A is the offline grant. |
| Readiness R1 | OD-D03-A-03–08 dispositions stay as recorded. Sentences that said acceptance authorizes no code are narrowed there to cloud execution and uncertified acceptance. Readiness does not grant class A. |
| Roadmap §5 | Parent, readiness, and the new child row record the same split. LOCAL-01 and PLATFORM-POLICY-CONFIG-01 stay closed in their previous scopes. |
| OD-D03-A-01 | Unchanged. Guardrail 6 still means that class ratification alone authorizes no implementation. This successor grant does not amend it. |
| LOCAL-01 | Not reopened. Its statement that acceptance of OD-D03-A-03–08 does not authorize production code remains true. This child is the separate named grant that record required. |
| PLATFORM-POLICY-CONFIG-01 | Not reopened. Its closure does not authorize cloud qualification. |
| Preparation P1–P6 | Not rewritten. The appendix still does not authorize provisioning. P5's statement that the preparation contains no adapter implementation describes that appendix. G1–G6 stay ungranted and remain the cloud-execution prerequisites. G7 order stays. |
| Proposal | Not rewritten. Unanimous OD ratification still does not itself authorize code. This amendment is the later offline authority. It is not the provisioning authority the proposal also required. The proposal NO-GO against starting implementation before explicit authorization is met for class A only after independent acceptance of this amendment. |
| Decision-03 | Unchanged. That decision still authorizes no backend and no global PIS. This grant is the separate offline implementation authority for adapters behind the existing continuity contract. It does not authorize a production store, a Kubernetes resource, or global PIS. |
| Freeze checklist | No `FRZ-*` status changes. |

### Applicable FRZ — status unchanged

Reviewed for the future offline child. No new PASS. This amendment does not certify any family.

| Family | Checklist status | Future evidence owner |
|---|---|---|
| `FRZ-OWN-01`..`05` | **OPEN** | Offline child for owner/import evidence. Not certified here |
| `FRZ-BND-01`..`06` | **OPEN** | Offline child for adapter boundary evidence. Not certified here |
| `FRZ-CTR-01`..`06` | **OPEN** | Offline child for contract evidence. Not certified here |
| `FRZ-TYP-01`..`06` | **OPEN** | Offline child for semantic typing evidence. Not certified here |
| `FRZ-PLG-01`..`08` | **OPEN** | Offline child for selection and fail-closed activation evidence. Not certified here |
| `FRZ-RPL-01`..`04` | **OPEN** | Actual cloud replaceability stays unqualified. RPL-A–H are not claimed |
| `FRZ-CFG-01`..`08` | **PASS** | Historical CONFIG-X PASS. Not evidence for the cloud adapter |
| `FRZ-TEN-01`..`12` | **PASS** | Historical TENANT-X PASS. Not evidence for the cloud adapter |
| `FRZ-REL-01`..`11` | **OPEN** | Not promoted. Real recovery proof stays on the cloud gate |
| `FRZ-SEC-01`..`10` | **OPEN** | Not promoted. Live IAM proof stays on the cloud gate |
| `FRZ-PRD-01`..`12` | **OPEN** | Not promoted. This record is not production qualification |

D03-B and D03-C stay blocked. IMPL-01, R7-Q, and PROD-Q stay blocked. No global FRZ is promoted.

## ECAP documentation Change Delta — original conditional acceptance

The table below is the 2026-10-18 documentation delta. It is not the 2026-10-19 authorization amendment.

| Step | Result |
|---|---|
| A1 | Qualification markdown only: this record, a roadmap ledger reference, and a freeze-checklist evidence link. No production symbols. |
| A2 | No imports, composition wiring or configuration keys. |
| A3 | No production callers or downstream consumers. |
| A4 | Ownership unchanged. The witness is not a second PIS, GCA or current-head owner. |
| A5 | Production-effective reachability: **no**. |
| A6 | Applicable `FRZ-*` families stay at their current statuses. `FRZ-SEC-01` remains **OPEN**. |
| A7 | Historical certifications are unchanged and do not extend to this unbuilt mechanism. |
| A8 | No regression gate selected. `pytest = 0`. |

This delta does not change layer boundaries, add execution or governance authority, duplicate a PIS/GCA/anchor owner, place vendor types in platform contracts, open an alternate CREATE path, promote any `FRZ-*` status, or claim current-HEAD enterprise certification.

### Tenant Isolation Audit

```text
tenant scope applicable: NO for this documentation delta
future infrastructure and production integration: TENANT/ENVIRONMENT RELEVANT — UNPROVEN
canonical tenant identity: none introduced
tenant owner: unchanged; no new owner
propagation path: none
state isolation: not exercised
provider/config isolation: not exercised
evidence/trace isolation: not exercised
async/recovery continuity: not exercised
cross-tenant path: none added
fail-closed behavior: required by the accepted direction; not implemented
adversarial evidence: none; documentation is not runtime certification
result: N/A — WITH EVIDENCE
```

The diff is documentation only. No runtime tenant identity, storage, credential or configuration path is added. Historical TENANT-X acceptance does not certify future Spanner or Azure configuration, nor production integration. Environment and tenant separation for that future integration remains an explicit unproven obligation. `FRZ-TEN-*` statuses are not promoted.

## ECAP documentation Change Delta — offline authorization amendment

This table is the 2026-10-19 decision-recording delta. It is not production qualification and it is not cloud qualification.

| Step | Result |
|---|---|
| A1 | Documentation files only: this decision, the readiness record, and the roadmap evidence ledger. Public code symbols = 0. |
| A2 | Decision and roadmap references only. No production import, composition wiring, or configuration key changes. |
| A3 | No new production caller. |
| A4 | Owners and sanctioned composition unchanged: `ProductionControlPlaneGovernance`; `compose_history_continuity_authority`. |
| A5 | Production-effective reachability: **no**. |
| A6 | Applicable families reviewed: `FRZ-OWN-01`..`05`, `FRZ-BND-01`..`06`, `FRZ-CTR-01`..`06`, `FRZ-TYP-01`..`06`, `FRZ-PLG-01`..`08`, `FRZ-RPL-01`..`04`, `FRZ-CFG-01`..`08`, `FRZ-TEN-01`..`12`, `FRZ-REL-01`..`11`, `FRZ-SEC-01`..`10`, `FRZ-PRD-01`..`12`. Statuses unchanged. No `SYS-INV-*` text change. No new guard. |
| A7 | LOCAL-01 at `302e7fa74b745d10f9552df0a6eb9c702d5ce033`, PLATFORM-POLICY-CONFIG-01 at `02e48d9fa61d116346da613271cc5e8b77e08cc6`, preparation P1–P6 at `4bbd786ffbd82000ef58b330854a6cce78cd2d9f`, and the Azure configuration correction at `cd6fe5240335b9f158be6071620027d1a2c1ba2a` stay scoped to their previous acceptances. Historical CFG and TEN PASS is not extended. |
| A8 | Documentation checks only. No pytest. `pytest = 0`. |

### Tenant Isolation Audit — authorization amendment

```text
tenant scope applicable: NO for this documentation delta
future offline adapters and cloud integration: TENANT/ENVIRONMENT RELEVANT — UNPROVEN
canonical tenant identity: none introduced
tenant owner: unchanged; no new owner
propagation path: none
state isolation: not exercised
provider/config isolation: not exercised
evidence/trace isolation: not exercised
async/recovery continuity: not exercised
cross-tenant path: none added
fail-closed behavior: required by class A; not implemented in this delta
adversarial evidence: none; documentation is not runtime certification
result: N/A — WITH EVIDENCE
```

This amendment adds no runtime tenant identity, storage, credential, or configuration path. It does not enable a cloud adapter. Historical TENANT-X acceptance does not certify a future Spanner or Azure adapter.
