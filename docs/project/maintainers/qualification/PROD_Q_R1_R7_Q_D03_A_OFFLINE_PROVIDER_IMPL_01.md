# D03-A-OFFLINE-PROVIDER-IMPL-01

Scoped acceptance dossier for the offline provider child. This file is the detailed evidence source. Roadmap §5 is the index. The conditional-direction record remains the class-A / class-B authorization. The freeze checklist remains the owner of each `FRZ-*` status.

| Field | Value |
|---|---|
| Status | **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** |
| Formal closure | Independent operator acceptance recorded 2026-10-19 (offline-only; not cloud-qualified; does not close parent D03-A). |
| Parent | `PROD-Q-R1-R7-Q-D03-A-CONDITIONAL-INFRASTRUCTURE-DIRECTION` — **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED** |
| Cloud execution | **NOT AUTHORIZED** |
| Implementation baseline | `a7ecc5065a0343e768584e818c844d3f96b5f533` |
| Evidence reconciliation baseline | `468caa1327dab5b1e60576a9edd71ba19c957003` |
| Prior formal closure record | `54f51c4dd81e0040aab0a8a469779c5ca3509964` |
| Independently reviewed formal closure record | `771a83c6b39f37120e1d5852233cb31ff37ff02b` |
| Parent closure | Not available |
| Evidence label | Offline code and deterministic tests. Not Spanner durability, Azure retention, IAM, disaster recovery, or production qualification. |
| FRZ promotion | **0**. No checklist status changes. |

Independent operator acceptance (2026-10-19) closes this child as **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED**. Method: static exact-SHA GitHub review (operator session; **pytest not rerun** by the operator). Scope: local offline behavior and architecture boundaries only. This transcription records the operator verdict; Cursor does not autonomously accept the child.

## Formal operator acceptance record (2026-10-19)

| Anchor | SHA | Independent static review scope |
|---|---|---|
| Implementation baseline | `a7ecc5065a0343e768584e818c844d3f96b5f533` | Offline provider modules, composition admission, typed ports, SDK bridges, fail-closed openers, LOCAL-01 / PLATFORM-POLICY-CONFIG-01 non-regression boundaries |
| Evidence reconciliation | `468caa1327dab5b1e60576a9edd71ba19c957003` | Canonical dossier, roadmap §5 index, conditional-direction pointers, readiness graph pointer, freeze checklist non-promotion note |
| Prior formal closure record | `54f51c4dd81e0040aab0a8a469779c5ca3509964` | Formal closure preparation and index alignment (historical; superseded for current-state closure by operator review below) |
| Independently reviewed formal closure record | `771a83c6b39f37120e1d5852233cb31ff37ff02b` | Formal closure record reviewed before operator promotion to **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** |

**Implementation and test evidence provenance:** deterministic unit tests listed under [Test evidence](#test-evidence); implementer-reported gates preserved as non-independent (**57** pytest PASS, Ruff PASS, Pyright **0** errors at the implementation baseline; not independently rerun by the operator); reconciliation regression commands in this dossier are session safety gates, not the GitHub static reviews.

**Operator acceptance conclusion (scoped offline only):** **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** at the implementation baseline. **Excluded:** real Spanner/Azure qualification, IAM independence, production retention duration (OD-D03-A-06), cost/capacity/SLO (OD-D03-A-07), cloud execution, production CREATE qualification, parent D03-A ratification, D03-B, D03-C, IMPL-01, R7-Q, PROD-Q advancement, and any new global `FRZ-*` **PASS** (**new FRZ PASS = 0**).

**Ownership and composition (unchanged):** `HistoryContinuityAuthority` contract; sole semantic owner `ProductionControlPlaneGovernance`; sole sanctioned composition `compose_history_continuity_authority`; mutation boundary `ControlPlaneMutationAuthorizationBoundary`; physical CREATE gate remains `GovernedProviderInvocationDispatchGate` only (not on the continuity graph).

## Implementation lineage

Earlier commits are lineage. They are not a substitute for the current baseline.

| Order | SHA | Subject |
|---|---|---|
| 1 | `3b4726e88fbffdf0048c8337eb087e7b6d01c23d` | add the offline continuity pair behind the existing authority |
| 2 | `7c3261fcf922aa2bd87cd6267f97d25e6b3b5f00` | type offline continuity admission and vendor transports |
| 3 | `dc115a6f0084635bbb63d42bf4a6f154a385ae19` | translate continuity operations onto injected SDK sessions |
| 4 | `a7ecc5065a0343e768584e818c844d3f96b5f533` | keep Spanner retries and Azure witness admission fail-closed |

The current implementation baseline is the fourth SHA. Files in that range:

- `intergrax/contracts/governance/history_continuity.py`
- `intergrax/applications/_shared/history_continuity_composition.py`
- `intergrax/runtime/governance/offline_history_continuity.py`
- `intergrax/runtime/governance/continuity_providers/` (`primary_register.py`, `witness_log.py`, `spanner_sdk.py`, `azure_sdk.py`, `credentials.py`, `errors.py`, `targets.py`, `__init__.py`)
- `tests/unit/governance/test_d03_a_local_01_elcr_iirw.py`
- `tests/unit/governance/test_d03_a_platform_policy_config_01.py`
- `tests/unit/governance/test_d03_a_offline_provider_impl_01.py`
- `tests/unit/governance/test_d03_a_offline_sdk_bridge_01.py`
- `pyproject.toml` and `uv.lock` (`continuity-cloud` names `google-cloud-spanner` and `azure-storage-blob`; naming the extra does not open a client)

No Terraform, Bicep, IAM, or deployment template is in that range. `infra/docker/` is not a cloud convention for this pair.

Accepted predecessors, unchanged by this child:

| Child | Accepted SHA | Status |
|---|---|---|
| D03-A LOCAL-01 | `302e7fa74b745d10f9552df0a6eb9c702d5ce033` | **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** |
| D03-A PLATFORM-POLICY-CONFIG-01 | `02e48d9fa61d116346da613271cc5e8b77e08cc6` | **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** |

## Closed-world inventory

Verified by inspection of `a7ecc5065a0343e768584e818c844d3f96b5f533`.

| Surface | Where it lives | Current-HEAD callers |
|---|---|---|
| `HistoryContinuityAuthority` | `intergrax/contracts/governance/history_continuity.py` | Implemented by `LocalDeterministicHistoryContinuity`, `InactiveHistoryContinuityAuthority`, and `OfflineSpannerAzureHistoryContinuity`. Returned only by `compose_history_continuity_authority`. |
| `HistoryContinuityActivationPolicy` | same contract | Existing LOCAL-01 / PLATFORM-POLICY-CONFIG-01 activation. Offline selection is a different type. |
| `HistoryContinuityProviderSelection` | same contract | Argument of `compose_history_continuity_authority`. `OFFLINE_EXTERNAL_PAIR` alone does not admit execution. |
| `ProductionControlPlaneGovernance` | `intergrax/applications/_shared/production_control_plane_governance_wiring.py` | Required argument of the one composition function. One class. |
| `compose_history_continuity_authority` | `intergrax/applications/_shared/history_continuity_composition.py` | One definition. Production callers: none. Callers are the four D03-A unit modules listed above. |
| `ControlPlaneMutationAuthorizationBoundary` | `intergrax/runtime/governance/control_plane_mutation_authorization.py` | Passed through from `governance.mutation_authorization_boundary`. No second boundary is constructed for continuity. |
| `GovernedProviderInvocationDispatchGate` | `applications/governed_contractor_application/host/provider_invocation_lifecycle.py` | Not called by continuity composition or by the continuity providers. Continuity modules do not reference it. |
| `SpannerPrimaryRegister` | `continuity_providers/primary_register.py` | Used by `OfflineSpannerAzureHistoryContinuity` and by offline unit tests. Rules stay here. |
| `AzureAppendWitness` | `continuity_providers/witness_log.py` | Same split: authority and unit tests. |
| `PrimaryDatabasePort`, `PrimaryTransactionPort` | `primary_register.py` | Implemented in-process by `InMemoryPrimaryDatabase`, `SpannerPrimaryTransport`, and `SpannerSdkPrimaryBridge`. |
| `WitnessAppendPort` | `witness_log.py` | Implemented in-process by `InMemoryAppendWitness`, `AzureAppendTransport`, and `AzureSdkAppendBridge`. |
| `InMemoryPrimaryDatabase`, `InMemoryAppendWitness` | provider modules | The only concrete types `compose_history_continuity_authority` will inject. |
| `SpannerPrimaryTransport`, `AzureAppendTransport` | `spanner_sdk.py`, `azure_sdk.py` | Unit tests. Composition rejects them. |
| `SpannerSdkPrimaryBridge`, `AzureSdkAppendBridge` | same modules | Unit tests over an injected session. Composition does not accept them. Construction does not open a network client. |
| `PrimaryWitnessEndpoint` | `azure_sdk.py` | Built only by `bind_primary_witness_endpoint`. A caller boolean is not this type. |
| `open_spanner_sdk_database`, `open_azure_append_client` | SDK modules | Always raise `OfflineProviderRefusal` before an SDK import or a network call. |

Facts checked before this documentation change:

1. No second Governance owner of history continuity.
2. No second sanctioned composition owner. The offline class rejects construction without the composition token.
3. No new CREATE dispatch path.
4. No runtime cloud SDK activation. `cloud_execution_authorized()` returns `False` in both SDK modules. There is no switch that turns it on.
5. No PIS or GCA duplication. `HistoryBinding` carries existing PIS and GCA watermarks. It does not create `ProviderInvocationStore` or a GCA.
6. No vendor types in `intergrax/contracts/governance/history_continuity.py`.
7. No local-provider regression. A call with no provider selection and no offline bundle still uses `_compose_local`. LOCAL-01 tests passed in the reconciliation regression below.
8. No proof of actual cloud operation is claimed. `HistoryContinuityCloudQualification` on an admitted selection must be `UNQUALIFIED`.

`applications/` imports this composition only from `history_continuity_composition.py`. Production lifecycle modules do not.

## Composition graph

```text
ProductionControlPlaneGovernance
  compose_history_continuity_authority
    no selection and no bundle
      -> local admission -> LocalDeterministicHistoryContinuity
      -> otherwise InactiveHistoryContinuityAuthority
    selection or bundle present
      -> deny unless selection, exact in-memory bundle, and qualification-absent credentials all hold
      -> open_offline_external_history_continuity
           -> OfflineSpannerAzureHistoryContinuity
                SpannerPrimaryRegister over bundle.primary
                AzureAppendWitness over bundle.witness
                existing ControlPlaneMutationAuthorizationBoundary
SpannerPrimaryTransport / AzureAppendTransport / SDK bridges
  not admitted by compose
open_spanner_sdk_database / open_azure_append_client
  always refuse
GovernedProviderInvocationDispatchGate
  not on this graph
```

## Offline behavior demonstrated

The offline authority can initialize and advance a scoped head, compare-and-swap under a fence, seal a witness, and recover inside a bounded prefix when the injected in-memory ports agree. Rollback, a corrupt witness, a missing prefix, a stale fence, a scope mismatch, and a direct witness append for another tenant do not promote a head. Retention cannot be weakened. Protected append and the combined hold stay `unqualified`. No production retention duration is set.

`local_deterministic` remains the local provider. `OFFLINE_EXTERNAL_PAIR` is a typed selection. Recognizing the enum does not admit cloud activation.

## Transport replaceability

`test_vendor_transport_substitution_preserves_register_and_witness` drives `SpannerPrimaryRegister` and `AzureAppendWitness` once on the in-memory ports and once on `SpannerPrimaryTransport` and `AzureAppendTransport`. The register and witness results match. The vendor transport records SQL and append-blob argument shapes. `google.cloud` and `azure.storage` are not imported.

`test_closed_world_admission_rejects_marker_and_vendor_transport` shows that those same transports are refused by `compose_history_continuity_authority`. Replaceability of the port is not admission to the sanctioned composition.

`SpannerSdkPrimaryBridge` and `AzureSdkAppendBridge` translate the same shapes onto an injected session. `test_sdk_bridges_stay_offline_and_outside_composition` keeps them outside composition and outside the platform contract.

This is offline structural replaceability. It is not RPL-A–H and it is not a live vendor switch.

## Spanner retry and unknown-commit safety

`SpannerSdkPrimaryBridge.run` gives each callback invocation its own `_WrittenImage`. The next invocation clears that evidence before the register runs again. An aborted attempt cannot satisfy a later commit.

On `ProviderFault.OUTCOME_UNKNOWN`, `_confirm_committed` returns success only when the image is intact and a strong read-back matches every head row and every transition row written by that attempt. A partial image, a missing row, or a mismatched row stays `UNKNOWN`. A read-back failure stays `UNKNOWN`. The bridge does not treat an abandoned attempt as a commit.

Covering tests: `test_spanner_unknown_commit_is_success_only_after_strong_read_back`, `test_spanner_retry_discards_abandoned_attempt_evidence`, `test_spanner_retry_unknown_does_not_promote_abandoned_or_partial_evidence`, `test_spanner_idempotent_retry_does_not_create_a_second_successor`.

The injected session is not Google Cloud Spanner. These tests do not show Spanner linearizability, durability, or regional failover.

## Azure endpoint binding and uncertain-append safety

`bind_primary_witness_endpoint` accepts one unambiguous `https` primary blob origin. Secondary account hosts, malformed URLs, userinfo, query strings, and non-primary shapes are rejected. Hostname substring matching is not used.

`AzureSdkAppendBridge` admits a blob only when the directory exposes a `PrimaryWitnessEndpoint` created by that binder and the blob URL is that account, that origin, and that blob name. A caller boolean is not admission.

An uncertain append becomes `COMMITTED` only when a read-back returns the same bytes and the blob is still bound to that primary endpoint. A short or failed read-back stays `UNKNOWN`. An equal-length payload that is not the written block is a conflict, not a commit.

Covering tests: `test_azure_primary_binding_rejects_unqualified_endpoints`, `test_azure_bridge_separates_conflict_timeout_and_read_back`, `test_azure_witness_rejects_bad_bytes_rollover_and_secondary_reads`, `test_azure_predecessor_mismatch_stays_a_witness_decision`.

These tests do not show Azure immutable retention, RA-GZRS failover, or a locked legal hold.

## Local admission restrictions

Offline execution requires all of the following. Any miss returns `InactiveHistoryContinuityAuthority` or refuses the opener:

- local activation and the offline bundle are mutually exclusive;
- `history_continuity_offline_provider_execution_permitted` is true (`OFFLINE_EXTERNAL_PAIR`, `OFFLINE_CONTRACT`, `UNQUALIFIED`, distinct administrators, workload-federation binding shape);
- the bundle object is exactly `OfflineProviderBundle`;
- `primary` is exactly `InMemoryPrimaryDatabase` and `witness` is exactly `InMemoryAppendWitness`;
- `segment_block_limit >= 3` and `recovery_block_bound >= 1`;
- `immutability_period_days` is unset, `protected_append` and `combined_hold` are `unqualified`, and `processing_units` is unset;
- credential evaluation yields no token and `LiveCredentialRefusal.QUALIFICATION_ABSENT`;
- the governance principal tenant matches the resolved environment tenant.

`open_spanner_sdk_database` and `open_azure_append_client` still refuse after that evaluation. They do not import a client.

## Test evidence

### Independent audit of the implementation baseline

The operator's previous independent GitHub review accepted `a7ecc5065a0343e768584e818c844d3f96b5f533` for local offline behavior and architecture boundaries only. This session could not read the GitHub review body (`gh` is not authenticated). The disposition above is the operator statement that commissioned this reconciliation. It is not a new audit and it is not execution evidence.

That review did not accept cloud infrastructure correctness, real SDK execution, or production qualification.

### Implementer-reported results

The prior implementer report, preserved here and not treated as independent execution evidence, stated:

- 57 targeted tests passed;
- Ruff PASS;
- Pyright 0 errors;
- AST import-direction PASS.

No separate AST import-direction command is recorded in this dossier. Source assertions inside the unit tests check that the contract and the SDK modules do not import `google.cloud` or `azure.storage`, and `tests/unit/agents/test_vendor_import_guard_b24.py` checks that `agents/` does not import those prefixes.

### Reconciliation regression at the implementation baseline

This documentation session ran the following commands against the implementation tree at `a7ecc5065a0343e768584e818c844d3f96b5f533`, before the documentation commit. They are a reconciliation regression. They are not the independent GitHub audit.

```text
uv run pytest -p no:xdist tests/unit/governance/test_d03_a_local_01_elcr_iirw.py tests/unit/governance/test_d03_a_platform_policy_config_01.py tests/unit/governance/test_d03_a_offline_provider_impl_01.py tests/unit/governance/test_d03_a_offline_sdk_bridge_01.py tests/unit/agents/test_vendor_import_guard_b24.py --tb=short -q
```

Result: **57 passed** in 0.77s (formal closure record session, 2026-10-19). Prior reconciliation run: 0.92s. Exit code 0. No live cloud access.

Collected tests: LOCAL-01 18, PLATFORM-POLICY-CONFIG-01 11, offline provider 14, SDK bridge 13, vendor import guard 1.

```text
uv run ruff check <composition, history_continuity contract, continuity_providers, offline_history_continuity, and the five test modules above>
```

Result: **All checks passed.**

```text
uv run pyright intergrax/applications/_shared/history_continuity_composition.py intergrax/contracts/governance/history_continuity.py intergrax/runtime/governance/continuity_providers intergrax/runtime/governance/offline_history_continuity.py
```

Result: **0 errors, 0 warnings, 0 informations.**

## Independent audit disposition and limitations

Accepted by the stated static review, and re-checked on the baseline before this dossier:

- one Governance semantic owner and one sanctioned composition;
- vendor-neutral platform contract;
- offline register, witness, transport, and injected-session bridge behavior described above;
- fail-closed admission, including refusal of live client openers;
- no CREATE-gate integration.

Not accepted, and not claimed here:

- actual Spanner durability or linearizability;
- real Azure immutable-retention behavior;
- IAM independence;
- production SLO;
- cross-cloud disaster recovery;
- production CREATE qualification;
- full D03-A parent acceptance;
- any new `FRZ-*` PASS.

## Known future cloud qualification gaps

Still required before class B, and not opened by this child:

- trust-domain graph and live IAM independence;
- OD-D03-A-06 production retention duration and legal hold;
- OD-D03-A-07 prices, capacity, and achieved SLO;
- real Spanner atomic commit, unknown-commit, and failover evidence;
- real Azure protected append, immutable retention, and geo-recovery evidence;
- D03-A01–A23 and RPL-A–H on live vendors;
- G1–G6 preparation for cloud execution.

D03-B, D03-C, IMPL-01, R7-Q, and PROD-Q stay at their current blocked or non-closed canonical statuses. This child does not advance them.

## Parent closure is not available

ECAP §2.B is a parent-closure audit. The parent remains **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**. OD-D03-A-06, OD-D03-A-07, live IAM, disaster recovery, and real vendor qualification are absent. Parent B1–B6 are not satisfied. Closing this child, if a later independent audit does that, still would not close the parent.

## ECAP §2.A for this operator closure promotion delta

This table covers the operator closure promotion documentation delta only. It transcribes the independent operator verdict; it is not Cursor autonomous acceptance, and it does not change the historical ECAP tables on the conditional-direction record.

| Step | Result |
|---|---|
| A1 | Documentation only: operator **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** transcription in this dossier, roadmap §5 index rows, conditional-direction current-state pointers, readiness graph pointer, checklist scoped non-promotion note. Public code symbols changed = 0. |
| A2 | Cross-references only. No production import, composition wiring, or configuration key changes. |
| A3 | No new production caller. `compose_history_continuity_authority` production callers remain none. |
| A4 | Semantic owner `ProductionControlPlaneGovernance`. Sanctioned composition `compose_history_continuity_authority`. Unchanged. |
| A5 | Production-effective reachability of this delta: **no**. |
| A6 | Applicable IDs are the mapping below. Statuses unchanged. No `SYS-INV-*` text change. No new guard. |
| A7 | LOCAL-01 at `302e7fa74b745d10f9552df0a6eb9c702d5ce033` and PLATFORM-POLICY-CONFIG-01 at `02e48d9fa61d116346da613271cc5e8b77e08cc6` stay in their accepted scopes. Historical CFG, TEN, EXE, GOV, STA, and REC PASS is not extended to a live Spanner or Azure adapter. |
| A8 | Gates for this promotion delta: `git diff --check` on the documentation commit; markdown relative-link integrity on changed qualification paths; SHA ancestry checks for listed anchors; consistency scan across current-state references; no production-code delta. Session reconciliation pytest (57 passed), scoped Ruff (pass), scoped Pyright (0 errors) at unchanged implementation baseline — not operator-independent execution. No cloud gate. |

### Tenant Isolation Audit

```text
tenant scope applicable: NO for this documentation delta
offline continuity code: TENANT/ENVIRONMENT RELEVANT — offline tests only
canonical tenant identity: none introduced by this delta
tenant owner: unchanged; no new owner
propagation path: none added by this delta
state isolation: not exercised by this delta
provider/config isolation: not exercised by this delta
evidence/trace isolation: not exercised by this delta
async/recovery continuity: not exercised by this delta
cross-tenant path: none added
fail-closed behavior: already required by the offline implementation; not changed here
adversarial evidence: none in this documentation delta
result: N/A — WITH EVIDENCE
```

Historical `FRZ-TEN-*` PASS is not evidence for a future Spanner or Azure configuration.

## ECAP §2.B readiness for this child

This is a readiness assessment for the child. It is not a parent-closure audit and it is not a decision to close the child.

| Step | Readiness |
|---|---|
| B1 | Closed-world inventory above is complete for the surfaces named in the reconciliation task. |
| B2 | Applicable `FRZ-*` IDs are mapped below to current checklist statuses. No promotion. |
| B3 | Positive offline evidence exists: local and offline initialize/advance, CAS, witness seal, transport substitution, injected-session translation. |
| B4 | Adversarial offline evidence exists: deny-by-default admission, forged composition inputs, unknown commit, abandoned retry, uncertain append, secondary and unbound witness URLs, tenant and scope mismatch, rollback and corrupt witness. |
| B5 | Execution bundle at the implementation SHA is split. Implementer-reported results are not independent execution. The previous GitHub acceptance was a static review. This session's pytest, Ruff, and Pyright are a reconciliation regression, not that independent audit. |
| B6 | Roadmap §5 and this dossier record **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** per independent operator verdict. Cursor promotion commit is **READY FOR AUDIT — OPERATOR CLOSURE RECORDING** until GitHub audit of that delta. |

Scoped B1–B6 evidence assembled at the implementation baseline remains referenced above and is not rewritten by this promotion delta.

## Applicable FRZ mapping

Statuses are copied from `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` at the documentation baseline. This table does not change them. Evidence is what `a7ecc5065a0343e768584e818c844d3f96b5f533` actually shows. "Scoped offline" is not a PASS. Historical PASS is not new evidence for a live cloud adapter.

Future qualification owner, for every row: not this child. Cloud rows wait on class B (`NOT AUTHORIZED`) and D03-A01–A23. Global PASS remains with the checklist and the stage that owns that family.

| ID | Criterion | Canonical status | Evidence at the implementation SHA | Scope | Future owner |
|---|---|---|---|---|---|
| FRZ-OWN-01 | every major concern has exactly one semantic owner | OPEN | `ProductionControlPlaneGovernance` is the only continuity semantic owner. One class. | Scoped offline | Checklist stays OPEN |
| FRZ-OWN-02 | one canonical contract per responsibility | OPEN | `HistoryContinuityAuthority` is the contract. Local, inactive, and offline authorities use it. | Scoped offline | Checklist stays OPEN |
| FRZ-OWN-03 | one sanctioned composition owner | OPEN | One `compose_history_continuity_authority`. Production callers: none. | Scoped offline | Checklist stays OPEN |
| FRZ-OWN-04 | no shadow authority | OPEN | Enum recognition is not admission. Vendor transports and SDK bridges are rejected by composition. | Scoped offline | Checklist stays OPEN |
| FRZ-OWN-05 | no duplicated semantic mechanism | OPEN | Register and witness own the rules. PIS and GCA are not reimplemented. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-01 | canonical layer model documented | OPEN | No layer change. Composition stays in `applications/_shared` and imports `intergrax`. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-02 | allowed dependency directions explicit | OPEN | Same direction as accepted LOCAL-01 composition. Continuity modules do not import `applications`. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-03 | reverse dependencies = 0 | OPEN | No new reverse edge from this child. Not a repo-wide rescan. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-04 | private implementation leakage = 0 | OPEN | Vendor call shapes stay in `continuity_providers`. The contract has no vendor types. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-05 | cross-layer construction only sanctioned composition | OPEN | Offline authority construction requires the composition token. | Scoped offline | Checklist stays OPEN |
| FRZ-BND-06 | transport adapters do not own runtime semantics | OPEN | CAS, fencing, and witness rules stay in the register and witness. Transports translate. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-01 | consumers depend on contracts | OPEN | Callers of the authority depend on `HistoryContinuityAuthority`. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-02 | concrete provider leakage = 0 | OPEN | Composition admits only exact in-memory ports. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-03 | no pseudo-contract dicts | OPEN | The platform contract is typed. SDK argument messages inside the provider are not that contract. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-04 | public/internal contracts classified | OPEN | Contract module and provider modules are separate. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-05 | contract responsibility narrow/cohesive | OPEN | `PrimaryDatabasePort`, `PrimaryTransactionPort`, and `WitnessAppendPort` stay narrow. | Scoped offline | Checklist stays OPEN |
| FRZ-CTR-06 | contracts do not encode concrete vendor/runtime implementation | OPEN | `history_continuity.py` has no Spanner or Azure types. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-01 | semantic boundaries strongly typed | OPEN | Activation, selection, targets, and outcomes are typed models and enums. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-02 | Any at semantic boundaries = 0 unless evidence-backed transport reason | OPEN | No `Any` on the contract or in `continuity_providers`. Not a repo-wide audit. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-03 | generic object semantic seams = 0 | OPEN | No object-typed semantic contract. The admission helper takes `object` only to reject every value that is not the exact in-memory type. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-04 | reflection-based semantic dispatch = 0 | OPEN | Openers do not use `importlib`. Tests assert the SDK modules do not name vendor import paths. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-05 | string-dispatch substitute for typed contract = 0 | OPEN | Adapter selection is `HistoryContinuityAdapterKind`. | Scoped offline | Checklist stays OPEN |
| FRZ-TYP-06 | type-ignore/cast cannot mask architecture mismatch | OPEN | Implementation modules have no `type: ignore` and no `cast`. Tests use `cast` only for inputs composition rejects. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-01 | extensible mechanisms expose platform contracts | OPEN | Extension surface is the typed ports, not a vendor SDK type. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-02 | external structural implementation works without core patch | OPEN | Transports and bridges satisfy the ports without patching the register or witness. Composition still refuses them. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-03 | provider discovery has explicit owner | OPEN | No discovery registry was added. Selection is an explicit composition argument. | Not exercised as discovery | Checklist stays OPEN |
| FRZ-PLG-04 | provider selection has explicit owner | OPEN | Selection owner is the existing composition plus `HistoryContinuityProviderSelection`. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-05 | activation/admission explicit | OPEN | Missing, mixed, unqualified, and wrong-type inputs return the deny-only authority. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-06 | plugin package != capability authority | OPEN | `continuity-cloud` names SDK packages. Both openers still refuse. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-07 | runtime extensions cannot self-expand authority | OPEN | A transport substitute or a caller boolean cannot admit itself. | Scoped offline | Checklist stays OPEN |
| FRZ-PLG-08 | dynamic registration reversible/scoped when supported | OPEN | This child does not add dynamic registration. | Not exercised | Checklist stays OPEN |
| FRZ-RPL-01 | representative default provider replaceable | OPEN | In-memory and offline vendor transports match for the register and the witness. | Scoped offline; cloud deferred | Class B / RPL-A–H, not authorized |
| FRZ-RPL-02 | custom implementation proof exists | OPEN | SDK bridges run on injected sessions in unit tests. | Scoped offline; not a live SDK | Class B, not authorized |
| FRZ-RPL-03 | replaceability not monkeypatch-only | OPEN | Substitution goes through the ports. | Scoped offline | Checklist stays OPEN |
| FRZ-RPL-04 | vendor/backend switch does not modify consumers | OPEN | Register, witness, and `HistoryContinuityAuthority` consumers take no vendor types. | Scoped offline; cloud deferred | Class B / RPL-A–H, not authorized |
| FRZ-EXE-01 | exactly one legal platform execution authority | PASS | No new execution authority. The CREATE gate is not called. | Historical PASS, not extended | Existing EXE certification; class B not authorized |
| FRZ-EXE-02 | all meaningful executable work enters canonical boundary | PASS | Continuity is not on the production CREATE path. | Historical PASS, not extended | Existing EXE certification |
| FRZ-EXE-03 | Nexus remains private implementation | PASS | This child does not touch Nexus. | Historical PASS, not extended | Existing EXE certification |
| FRZ-EXE-04 | Agent != Execution | PASS | No agent execution surface added. | Historical PASS, not extended | Existing EXE certification |
| FRZ-EXE-05 | child work admitted correctly | PASS | Not re-certified. | Historical PASS, not extended | Existing EXE certification |
| FRZ-EXE-06 | execution identity canonical | PASS | Not re-certified. | Historical PASS, not extended | Existing EXE certification |
| FRZ-EXE-07 | no alternate scheduler/worker execution authority | PASS | No scheduler added. | Historical PASS, not extended | Existing EXE certification |
| FRZ-GOV-01 | Governance != Execution | PASS | Continuity governance is not an execution authority. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-02 | proposal != permission != execution | PASS | Not re-certified. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-03 | child_authority subset parent_authority | PASS | Not re-certified. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-04 | downstream scope narrowing only | PASS | Not re-certified. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-05 | fresh authorization before meaningful effects | PASS | Offline mutations use the existing authorization boundary. Not a new GOV proof. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-06 | missing HITL never approval | PASS | Not exercised here. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-07 | required evidence before work | PASS | Not re-certified. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-08 | policy/profile revision attributable | PASS | Not re-certified. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-09 | runtime extensions cannot expand authority | PASS | Offline admission fails closed. Not a new GOV proof. | Historical PASS, not extended | Existing GOV certification |
| FRZ-GOV-10 | control-plane mutation governed | PASS | Continuity uses `ControlPlaneMutationAuthorizationBoundary` already owned by governance. | Historical PASS, not extended | Existing GOV certification |
| FRZ-STA-01 | exactly one truth owner per state family | PASS | The in-memory continuity register is not a STATE-X certified family and is not production-wired. | Historical PASS, not extended | STATE-X owner unchanged; class B not authorized |
| FRZ-STA-02 | no duplicate semantic state stores | PASS | No second PIS or GCA. The offline register is not claimed as those stores. | Historical PASS, not extended | STATE-X owner unchanged |
| FRZ-STA-03 | atomic/transaction boundaries explicit | PASS | Offline CAS is in-process. Not STATE-X and not Spanner atomicity. | Historical PASS, not extended | Class B, not authorized |
| FRZ-STA-04 | tenant isolation | PASS | Offline scope checks exist. They do not extend STATE-X. | Historical PASS, not extended | TENANT-X / STATE-X unchanged |
| FRZ-STA-05 | stale state rejected | PASS | Stale fence and revision conflicts are offline tests. Not a new STA PASS. | Historical PASS, not extended | STATE-X owner unchanged |
| FRZ-STA-06 | configured/effective/persisted state distinction | PASS | Selection is not cloud qualification. Not re-certified. | Historical PASS, not extended | CONFIG-X / STATE-X unchanged |
| FRZ-STA-07 | policy artifacts durable where required | PASS | No durable production policy artifact added. | Historical PASS, not extended | Class B, not authorized |
| FRZ-STA-08 | checkpoints are state not identity authority | PASS | Not exercised as a checkpoint authority. | Historical PASS, not extended | STATE-X owner unchanged |
| FRZ-REC-01 | crash recovery deterministic | PASS | Offline crash and recovery tests exist. They are not STATE-X and not cloud crash recovery. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-02 | restart recovery preserves authority | PASS | Offline recovery does not mint an incarnation. Not a process-restart proof. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-03 | resume does not mint new identity incorrectly | PASS | Not a cloud resume proof. | Historical PASS, not extended | STATE-X owner unchanged |
| FRZ-REC-04 | replay cannot create divergent truth | PASS | Idempotent retry does not create a second successor offline. Not cloud replay. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-05 | fork semantics explicit | PASS | Fork is not exercised by this child. | Historical PASS, not extended | STATE-X owner unchanged |
| FRZ-REC-06 | partial persistence fails closed where required | PASS | Partial and abandoned attempts stay unknown or conflict offline. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-07 | external operation uncertainty modeled | PASS | Unknown commit and uncertain append stay unknown without an exact read-back. | Historical PASS, not extended; live vendors deferred | Class B, not authorized |
| FRZ-REC-08 | backup/restore responsibility and supported semantics explicit | PASS | No Spanner or Azure backup/restore evidence. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-09 | restore preserves canonical identity, authority, tenant isolation and exactly-one truth | PASS | No cloud restore. | Historical PASS, not extended | Class B, not authorized |
| FRZ-REC-10 | partial/corrupt durable state is detected and cannot silently become accepted canonical truth | PASS | Corrupt and missing witness prefixes do not promote a head offline. Not durable cloud state. | Historical PASS, not extended | Class B, not authorized |
| FRZ-CFG-01 | every configurable provider/vendor/model/backend selection is derived from an explicit typed platform configuration/profile contract, not a hard-coded production implementation choice | PASS | Typed selection exists. Historical CONFIG-X PASS is not cloud-adapter evidence. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-02 | missing or invalid required configuration cannot silently select a vendor, model, backend or provider; the mechanism fails closed or remains explicitly disabled | PASS | Bad offline selection fails closed. Historical PASS is not extended. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-03 | configured state ≠ effective state is mechanically enforced: installed/available/configured capability does not become effective without the sanctioned activation/composition path | PASS | Named SDK extra and enum recognition are not admission. Historical PASS is not extended. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-04 | each configurable concern has exactly one canonical configuration contract and exactly one sanctioned composition/selection owner | PASS | Existing composition remains the only owner. Historical PASS is not extended. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-05 | provider/vendor/model/backend-specific endpoints, deployments, regions, credentials or secret references are supplied through approved configuration/secret boundaries and are not embedded as production semantic choices in core code | PASS | No production endpoint or credential is configured. Historical PASS is not extended. | Historical PASS, not extended | Class B, not authorized |
| FRZ-CFG-06 | an unconfigured provider/vendor/plugin/backend cannot be reached through an alternate factory, fallback, direct constructor or compatibility path | PASS | Direct offline construction without the composition token fails. Historical PASS is not extended. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-07 | effective configuration resolution is deterministic and traceable across all supported configuration scopes; precedence is explicit and no hidden ambient state silently overrides the sanctioned configuration | PASS | Not re-certified. | Historical PASS, not extended | CONFIG-X owner unchanged |
| FRZ-CFG-08 | regression gates mechanically prevent reintroduction of hard-coded provider/vendor/model/backend selection, silent fallback and activation without required configuration | PASS | Existing tests guard this child. Historical PASS is not extended. | Historical PASS, not extended | CONFIG-X / QUAL-X unchanged |
| FRZ-TEN-01 | canonical tenant identity where tenant scope applies | PASS | Offline path uses the existing principal tenant. Historical TENANT-X PASS is not extended. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-02 | end-to-end tenant propagation across relevant semantic and runtime boundaries | PASS | Not a live provider path. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-03 | child/downstream operations cannot silently switch or widen tenant scope | PASS | Not re-certified. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-04 | cross-tenant state/cache/persistence/memory/index access blocked without explicit governed cross-tenant semantics | PASS | Offline reads and appends for another tenant do not succeed. Not a new TEN PASS. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-05 | provider selection, configuration, profiles and effective bindings preserve tenant scope | PASS | Tenant mismatch denies composition. Not cloud binding proof. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-06 | tenant-specific credentials/secrets references cannot resolve or reuse under another tenant context | PASS | Live credentials are refused. Not a secret-reuse proof across tenants. | Historical PASS, not extended | Class B, not authorized |
| FRZ-TEN-07 | evidence, events, traces and diagnostics preserve tenant identity | PASS | Not a trace certification. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-08 | checkpoint, retry, recovery, replay and restore preserve original tenant identity | PASS | Offline retry and recovery tests are scoped. Not cloud restore. | Historical PASS, not extended | Class B, not authorized |
| FRZ-TEN-09 | legal cross-tenant actions require explicit typed contract + explicit authority + explicit policy + auditable evidence | PASS | No cross-tenant grant added. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-10 | missing/invalid tenant must not silently become global, shared, default tenant, system tenant or provider-global scope | PASS | Failed resolution denies. Not re-certified. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-11 | runtime extensions/providers/plugins cannot mint, rewrite or widen tenant scope | PASS | Provider code does not widen tenant scope in the offline tests. Not a new TEN PASS. | Historical PASS, not extended | TENANT-X owner unchanged |
| FRZ-TEN-12 | executable adversarial tests prove tenant isolation across execution, state, provider/config, evidence and recovery critical paths | PASS | Offline negatives exist. They do not replace TENANT-X and they do not cover live vendors. | Historical PASS, not extended | Class B, not authorized |
| FRZ-SEC-01 | tenant isolation proven | OPEN | Offline tenant denial only. | Scoped offline; production deferred | PROD-Q / class B, not authorized |
| FRZ-SEC-02 | secrets references / late resolution | OPEN | Openers refuse before a client or network call. Not production secret resolution. | Scoped offline; cloud deferred | Class B, not authorized |
| FRZ-SEC-03 | raw secrets do not cross unrelated layers | OPEN | The platform contract carries no vendor credential type. Not a repo-wide secret audit. | Scoped offline | PROD-Q, not this child |
| FRZ-SEC-04 | production auth bypass = 0 | OPEN | No production auth path added. | Not exercised | PROD-Q, not this child |
| FRZ-SEC-05 | sandbox/isolation boundaries qualified | OPEN | No sandbox qualification. | Not exercised | PROD-Q, not this child |
| FRZ-SEC-06 | provider/plugin trust qualified | OPEN | No live provider trust evidence. | Cloud deferred | Class B, not authorized |
| FRZ-SEC-07 | security claim bounded by explicit threat model/evidence | OPEN | No new threat model. | Not exercised | PROD-Q, not this child |
| FRZ-SEC-08 | sensitive/tenant data propagation across platform/provider boundaries is explicit and qualified | OPEN | Tenant fields exist on records. Live propagation is not qualified. | Cloud deferred | Class B, not authorized |
| FRZ-SEC-09 | retention / deletion / redaction responsibility is explicit for platform-owned durable or sensitive data | OPEN | Protected append and combined hold stay `unqualified`. No duration. | Cloud deferred | Class B / OD-D03-A-06, not authorized |
| FRZ-SEC-10 | platform vs infrastructure responsibility for encryption/data residency/provider exposure is explicit and evidence-backed | OPEN | No residency or encryption evidence. | Cloud deferred | Class B, not authorized |
| FRZ-REL-01 | timeout ownership explicit | OPEN | No production timeout owner added. | Not exercised | PROD-Q, not this child |
| FRZ-REL-02 | retry ownership explicit | OPEN | Abandoned Spanner callback evidence is discarded. Not a production retry policy. | Scoped offline | Class B, not authorized |
| FRZ-REL-03 | idempotency ownership explicit | OPEN | Idempotent prepare does not create a second successor offline. | Scoped offline | PROD-Q, not this child |
| FRZ-REL-04 | cancellation propagates correctly | OPEN | Cancellation is not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-REL-05 | external operation termination semantics explicit | OPEN | Unknown commit and uncertain append stay unknown without an exact read-back. | Scoped offline; live vendors deferred | Class B, not authorized |
| FRZ-REL-06 | provider failure typed | OPEN | `ProviderFault` is the typed failure. | Scoped offline | PROD-Q, not this child |
| FRZ-REL-07 | no hidden infinite retry/fallback | OPEN | The bridge does not run its own unbounded retry loop. Not a platform fallback audit. | Scoped offline | PROD-Q, not this child |
| FRZ-REL-08 | bounded concurrency and work-admission semantics explicit | OPEN | Not production admission. | Not exercised | PROD-Q, not this child |
| FRZ-REL-09 | queue/backpressure and overload behavior explicit | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-REL-10 | resource saturation/exhaustion has deterministic reject/degrade/fail-closed behavior | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-REL-11 | provider throttling/rate-limit handling cannot create unbounded retry/fallback or bypass | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-01 | production qualification independent of lab/harness | OPEN | Offline tests are not production qualification. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-02 | provider/plugin qualification enforced | OPEN | Live Spanner and Azure are not qualified. | Cloud deferred | Class B, not authorized |
| FRZ-PRD-03 | unsupported production configuration fails closed | OPEN | Offline admission fails closed. Not a production configuration proof. | Scoped offline | PROD-Q, not this child |
| FRZ-PRD-04 | startup lifecycle qualified | OPEN | Not on the production lifecycle. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-05 | shutdown/resource cleanup qualified | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-06 | degraded behavior explicit | OPEN | Not a production degraded mode. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-07 | strict vs lab semantics separated | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-08 | production-only bypass count = 0 | OPEN | No production bypass added. Not a census. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-09 | production health/readiness/degraded-state semantics qualified | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-10 | terminal and critical production failures are operator-visible and actionable | OPEN | Not exercised. | Not exercised | PROD-Q, not this child |
| FRZ-PRD-11 | production capacity/overload behavior is qualified and bounded | OPEN | No capacity number. OD-D03-A-07 stays unresolved. | Cloud deferred | Class B, not authorized |
| FRZ-PRD-12 | critical operational recovery responsibility/procedure is explicit | OPEN | No cloud recovery procedure. | Cloud deferred | Class B, not authorized |

## Status preserved

- Child: **CLOSED — SCOPED / INDEPENDENTLY ACCEPTED** (offline-only; independent operator verdict recorded 2026-10-19).
- Parent: **CONDITIONAL DIRECTION ACCEPTED — FULL RATIFICATION BLOCKED**.
- Cloud execution: **NOT AUTHORIZED**.
- D03-B and D03-C: blocked.
- IMPL-01, R7-Q, and PROD-Q: not advanced.
- Global `FRZ-*`: unchanged. **new FRZ PASS = 0**.

**Cursor execution disposition:** **READY FOR AUDIT — OPERATOR CLOSURE RECORDING** (documentation delta only; does not substitute for independent GitHub audit of the promotion commit).
