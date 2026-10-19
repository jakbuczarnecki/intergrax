# D03-A — Conditional infrastructure direction — operator decision

| Field | Value |
|---|---|
| Status | **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** |
| Date | 2026-10-18 |
| Branch | `development` |
| Audited proposal baseline | `f67f4e61c80f8b408c273fc8c8423c31a86cb152` |
| Parent | `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01` — OPEN/BLOCKED |
| Source design | [D03-A trusted anchor backend and trust domain proposal](PROD_Q_R1_R7_Q_D03_A_TRUSTED_ANCHOR_BACKEND_TRUST_DOMAIN_PROPOSAL.md) |
| Unchanged class decision | [OD-D03-A-01](PROD_Q_R1_R7_Q_D03_A_ELCR_IIRW_MECHANISM_CLASS_OPERATOR_RATIFICATION.md) |
| Production delta | **0** |
| FRZ promotion | **0** |
| Test execution | `pytest = 0`; `uv run = 0` |

## Decision boundary

This record conditionally accepts a preferred technology direction after the audited proposal baseline. It is not unconditional ratification of OD-D03-A-02–08. It does not amend OD-D03-A-01. It authorizes no implementation, provisioning, subscription purchase, account creation, credentials, deployment configuration or runtime activation.

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
- infrastructure activation and implementation authorization.

**OD-D03-A-03 through OD-D03-A-08 are not ratified.** A proposal option is not accepted by silence. OD-D03-A-09 and OD-D03-A-10, left open by OD-D03-A-01, receive no disposition here.

## Next gate

These items are outstanding decision evidence for this direction. They are prerequisites, not new roadmap stages.

1. **Trust domains:** the actual administrator, service-account, federation and emergency-access graph, with no hidden common authority.
2. **Retention:** justification of the retention period and legal-hold requirements.
3. **Cost:** realistic infrastructure sizing, recurring operating costs and budget approval.
4. **Availability:** confirmation of acceptable recovery objectives and of the consequences of cross-cloud outages.
5. **Qualification:** sandbox evidence requirements covering atomic updates, immutable witness append, uncertain outcomes, corruption, crash recovery and rollback.

If obtaining this evidence requires actual cloud resources, credentials, procurement, a new trusted transition service or an architectural change:

**STOP — OPERATOR DECISION / AUTHORIZATION REQUIRED**

## Status preserved

- Technology direction: accepted conditionally.
- Infrastructure: not fully ratified.
- D03-A: not implemented and not closed.
- D03-B and D03-C: blocked.
- Parent `IMPL-01`, `R7-Q`, `PROD-Q` and the global enterprise freeze: not closed.
- No `FRZ-*` status is promoted.

## ECAP documentation Change Delta

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
