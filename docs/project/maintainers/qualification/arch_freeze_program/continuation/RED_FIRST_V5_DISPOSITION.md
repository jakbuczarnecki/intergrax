# RED_FIRST V5 — UNREPRODUCIBLE disposition (phase-check 3)

**Worktree / branch:** `D:/Projekty/_wt/intergrax/af-v5-p1-red` · `af-v5/p1-red-first`  
**Evidence integration head:** `5935fa4f0cd3596852b3a6500f9d4e1d613a53df`  
**Final-probes remeasurement (2026-11-26, `af-v5/final-probes`):** PLG-02, PLG-07, RPL-04 and REL-07 prefix pytest exit 0; rows stay `RED_FIRST_UNREPRODUCIBLE`.
**ENT-iso remeasurement (2026-11-26, `af-v5/ent-iso` @ `2a6e197ca`):** prefix `754ee000675d71c1734656702beb0aef8ba58fd8`; probes in `tests/unit/architecture/test_nt_ent_iso_red_first_probes.py`.  
**PLG-marker remeasurement (2026-11-26, `af-v5/plg-marker`):** probes in `tests/unit/applications/test_nt_plg_red_first_probes.py`, `tests/unit/architecture/test_nt_prd_red_first_probes.py`, `tests/unit/architecture/test_nt_rpl_red_first_probes.py`.
**Prefix-API remeasurement (2026-11-26, `af-v5/prefix-api`):** probes in `tests/unit/architecture/test_nt_rel_red_first_probes.py`, `tests/unit/applications/test_nt_ten_red_first_probes.py`.
**Canonical counts (rw_plan.v1.json after prefix-api pass):** **33** `RED_AT_PREFIX` · **6** `RED_FIRST_UNREPRODUCIBLE`  
**Machine result:** [`RED_FIRST_V5_RESULT.json`](RED_FIRST_V5_RESULT.json)

## Phase-check 3 status

`verify_rw_plan_phase_check3.py` treats **RED_FIRST_EVIDENCE** as **INCOMPLETE** while any RW row remains **RED_FIRST_UNREPRODUCIBLE**. This disposition does **not** close phase-check 3. No operator row is marked **DECIDED** here; each **T-AF2-RED-CLASSES** recommendation is planning input only.

## Cause-class legend

| Class | Meaning |
| --- | --- |
| `nested_ent_gate_exit4` | Nested ENT audit wrapper reported exit 4 / “Gate check failed”, not an FRZ-matrix assertion on the nominated mechanism. |
| `historical_self_test` | Governance self-test or script surface absent at the recorded prefix (`AttributeError` on `_self_test_cases`). |
| `optional_dep_probe_green` | Optional dependency or harness missing at collection, or a dedicated red-first probe already **exit 0** at the prefix. |
| `pytest_marker_config` | Pytest configuration lacks a required custom marker (`ent_semantic_gate`). |
| `prefix_already_green` | Nominated negative path **passed** at the prefix (inventory/count already satisfied). |
| `missing_wiring_module` | `ModuleNotFoundError` for a wiring module unrelated to the matrix concern (`wiring_posture`). |
| `import_not_mechanism` | Collection/import failure for a symbol that encodes fail-closed behavior **later** than the prefix tree; no behavioral assertion naming the FRZ mechanism at the prefix. |
| `subprocess_timeout` | Nested subprocess hit `TimeoutExpired` (300s); not a failed product assertion. |

## ENT-iso promotions (prefix `754ee000675d71c1734656702beb0aef8ba58fd8`)

| `frz_id` | Prior cause | ENT-iso probe | Prefix assertion (mechanism) | Status |
| --- | --- | --- | --- | --- |
| `FRZ-CFG-02` | `nested_ent_gate_exit4` | `test_nt_cfg_02_missing_embedding_config_must_not_silently_select_vendor` | `NT-CFG-02 missing or invalid required configuration must not silently select a vendor/backend when embedding provider is unset` | **RED_AT_PREFIX** |
| `FRZ-CFG-08` | `nested_ent_gate_exit4` | `test_nt_cfg_08_embedding_profile_must_not_hardcode_provider_getenv_default` | `NT-CFG-08 silent fallback: embedding profile must not carry a hard-coded provider constant` | **RED_AT_PREFIX** |
| `FRZ-OWN-02` | `nested_ent_gate_exit4` | `test_nt_own_02_application_artifact_ref_has_one_canonical_definition` | `NT-OWN-02 duplicate canonical contract` (dual `ApplicationArtifactRef` definition) | **RED_AT_PREFIX** |
| `FRZ-OWN-03` | `nested_ent_gate_exit4` | `test_nt_own_03_one_sanctioned_llm_adapter_composition_owner` | `NT-OWN-03 multiple composition owners construct LLM adapters` | **RED_AT_PREFIX** |


## PLG-marker promotions (prefixes `22993254b13a…`, `2b12af1dc502…`, `754ee000675d…`)

| `frz_id` | Prior cause | Probe | Prefix assertion (mechanism) | Status |
| --- | --- | --- | --- | --- |
| `FRZ-PLG-01` | `optional_dep_probe_green` | `test_nt_plg_01_evaluation_wiring_exposes_tenant_scoped_registry_contract` | evaluation wiring lacks `TenantScopedOnlineEvaluationRegistry` at `22993254b13a` | **RED_AT_PREFIX** |
| `FRZ-PLG-06` | `optional_dep_probe_green` | `test_nt_plg_06_strict_manifest_gate_must_not_downgrade_product_to_lab` | `_gate_wiring_environment` downgrades PRODUCT hosts at `2b12af1dc502` | **RED_AT_PREFIX** |
| `FRZ-PRD-03` | `pytest_marker_config` | `test_nt_prd_03_unsupported_queue_backend_must_not_fall_back_to_eager_celery` | `fall back to eager Celery` present at `754ee000` | **RED_AT_PREFIX** |

## Prefix-API promotions (prefixes `ec7f1e27c4a0…`, `a01b7e415797…`, `0cee41a88274…`)

| `frz_id` | Prior cause | Probe | Prefix assertion (mechanism) | Status |
| --- | --- | --- | --- | --- |
| `FRZ-REL-09` | `import_not_mechanism` | `test_nt_rel_09_kafka_rabbitmq_must_not_be_silently_rewritten_to_eager_celery` | `NT-REL-09 … Kafka/RabbitMQ … must not be silently rewritten to in-process eager Celery` | **RED_AT_PREFIX** |
| `FRZ-REL-10` | `import_not_mechanism` | `test_nt_rel_10_in_memory_kubernetes_scaler_must_not_substitute_when_cluster_url_unset` | `NT-REL-10 … in-memory Kubernetes scaler must not substitute when INTERGRAX_KUBERNETES_URL is unset` | **RED_AT_PREFIX** |
| `FRZ-TEN-07` | `import_not_mechanism` | `test_nt_ten_07_mcp_client_asserted_tenant_identity_must_fail_closed` | `NT-TEN-07 MCP client-asserted tenant identity must fail closed` | **RED_AT_PREFIX** |

## Row dispositions (6 × RED_FIRST_UNREPRODUCIBLE)

| `frz_id` | Logged cause (summary) | Cause class | Why this is not positive red-first evidence | T-AF2-RED-CLASSES (recommendation) | Operator decision |
| --- | --- | --- | --- | --- | --- |
| `FRZ-PLG-02` | `register_plugins` explicit path | `prefix_already_bounded` | pytest exit 0 @ `22993254b13a`: prefix already accepts explicit external classes via `register_plugins` / `load_plugin_types`. | **FIX** | not DECIDED |
| `FRZ-PLG-07` | `capability_payload_to_tool_ids` + `ToolAccessPolicy` | `prefix_already_bounded` | pytest exit 0 @ `22993254b13a`: prefix already refuses tool ids outside the declared registry contract. | **FIX** | not DECIDED |
| `FRZ-RPL-04` | `MessageConsumer` worker binding | `prefix_already_bounded` | pytest exit 0 @ `754ee000675d`: queue worker call sites already bind `MessageConsumer`, not concrete vendor classes. | **FIX** | not DECIDED |
| `FRZ-FRZ-10` | `AttributeError`: `verify_arch_freeze_plan_traceability` has no `_self_test_cases` | `historical_self_test` | Post-freeze policy activation is a later W5/W8 node; the logged break is a missing historical self-test hook, not matrix mechanism proof. | **FIX** (re-home policy probe when W5/W8 owns FRZ-10) | not DECIDED |
| `FRZ-PRD-08` | P0 inventory test exit 0 at prefix (bypass count already 0) | `prefix_already_green` | Prefix tree already satisfied the bypass-count invariant; no red mechanism to replay. | **OUT** (recommendation: stays **OPEN** as historical non-evidence; no OUT operator citation) | not DECIDED |
| `FRZ-REL-07` | `RetryEngine.decide` budget | `prefix_already_bounded` | pytest exit 0 @ `66664e2bffa1`: `RetryEngine.decide` already returns `should_retry=False` at declared `max_retries`. | **FIX** | not DECIDED |

## Candidates not promoted

TEN-01 and TEN-10 promoted in the `af-v5/short-rel` review. PLG-02, PLG-07, RPL-04 and REL-07 stay unpromoted because the recorded prefixes already satisfy those mechanisms. PRD-08 and FRZ-10 stay unpromoted. Residual UNREPRODUCIBLE count **6**.

## Explicit non-effects

This document does **not**:

- set **SCOPE**, **G2**, **R**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE CLOSED**;
- weaken `scripts/governance/verify_rw_plan_phase_check3.py`;
- mark any **T-AF2-RED-CLASSES** row as operator-**DECIDED**.

**Recorded:** 2026-11-26 · red-first V5 disposition pass on `af-v5/p1-red-first` and prefix-api remeasurement on `af-v5/prefix-api`.
