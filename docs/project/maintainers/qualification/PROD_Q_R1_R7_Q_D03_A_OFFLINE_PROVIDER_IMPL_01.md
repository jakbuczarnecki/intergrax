# D03-A-OFFLINE-PROVIDER-IMPL-01

| Field | Value |
|---|---|
| Status | **READY FOR AUDIT — OFFLINE CODE ONLY** |
| Parent | `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION` — **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** |
| Cloud execution | **NOT AUTHORIZED** |
| Closure | Not requested. Not self-certified. |
| Evidence label | Offline code and deterministic tests. Not Spanner, Azure, IAM, or administrative-domain proof. |

The offline provider pair sits behind the existing `HistoryContinuityAuthority`. `compose_history_continuity_authority` remains the only composition function. `ProductionControlPlaneGovernance` remains the semantic owner. `local_deterministic` remains the local provider. `OFFLINE_EXTERNAL_PAIR` is a typed selection. Recognizing it does not admit cloud activation.

Vendor client calls live in `intergrax/runtime/governance/continuity_providers/spanner_sdk.py` and `azure_sdk.py`. Both refuse before a network call. The optional `continuity-cloud` extra names the SDK packages. This child does not install them and does not open a session.

No cloud IaC, IAM, or deployment template was added. The repository has no Terraform or Bicep convention for this pair. Docker compose files under `infra/docker/` are not that convention.

Protected append and the combined hold stay `unqualified`. No production retention duration is set. D03-B and D03-C are not implemented. No `FRZ-*` status is promoted. D03-A01–A23 and RPL cloud execution remain future work.

## Parent-scoped audit

Closure of the child is not recommended. Closure of the parent is not available. OD-D03-A-06, OD-D03-A-07, live IAM, disaster recovery, and real vendor qualification are still absent. B1–B6 of a parent closure audit are not satisfied.

LOCAL-01 and PLATFORM-POLICY-CONFIG-01 stay in their previously accepted scopes.
