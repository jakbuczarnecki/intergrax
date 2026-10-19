# D03-A-OFFLINE-PROVIDER-IMPL-01

| Field | Value |
|---|---|
| Status | **READY FOR AUDIT — OFFLINE CODE ONLY** |
| Parent | `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION` — **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** |
| Cloud execution | **NOT AUTHORIZED** |
| Closure | Not requested. Not self-certified. |
| Evidence label | Offline code and deterministic tests. Not Spanner, Azure, IAM, or administrative-domain proof. |

The offline provider pair sits behind the existing `HistoryContinuityAuthority`. `compose_history_continuity_authority` remains the only composition function. `ProductionControlPlaneGovernance` remains the semantic owner. `local_deterministic` remains the local provider. `OFFLINE_EXTERNAL_PAIR` is a typed selection. Recognizing it does not admit cloud activation.

Vendor call shapes live in `intergrax/runtime/governance/continuity_providers/spanner_sdk.py` and `azure_sdk.py`. Each Spanner callback invocation keeps its own mutation evidence; an aborted attempt cannot satisfy a later commit or an unknown-outcome read-back, and a multi-mutation read-back must match every row written by that attempt. Azure positive witness decisions require `PrimaryWitnessEndpoint` from `bind_primary_witness_endpoint` at the injected directory, then an exact blob binding. Hostname substring matching and a caller boolean are not admission. `open_spanner_sdk_database` and `open_azure_append_client` always refuse before a client import or a network call. There is no switch that enables cloud execution. `SpannerPrimaryTransport` and `AzureAppendTransport` execute the documented SDK argument shapes offline. `SpannerSdkPrimaryBridge` and `AzureSdkAppendBridge` translate those same shapes onto an injected session. Construction does not acquire a credential, open a session, or contact a network. `compose_history_continuity_authority` still admits only `InMemoryPrimaryDatabase` and `InMemoryAppendWitness`. Continuity rules stay in `SpannerPrimaryRegister` and `AzureAppendWitness`. That execution is not a live session and does not qualify Spanner or Azure. The optional `continuity-cloud` extra names the SDK packages. This child does not install them.

No cloud IaC, IAM, or deployment template was added. The repository has no Terraform or Bicep convention for this pair. Docker compose files under `infra/docker/` are not that convention.

Protected append and the combined hold stay `unqualified`. No production retention duration is set. D03-B and D03-C are not implemented. No `FRZ-*` status is promoted. D03-A01–A23 and RPL cloud execution remain future work.

## Parent-scoped audit

Closure of the child is not recommended. Closure of the parent is not available. OD-D03-A-06, OD-D03-A-07, live IAM, disaster recovery, and real vendor qualification are still absent. B1–B6 of a parent closure audit are not satisfied.

LOCAL-01 and PLATFORM-POLICY-CONFIG-01 stay in their previously accepted scopes.
