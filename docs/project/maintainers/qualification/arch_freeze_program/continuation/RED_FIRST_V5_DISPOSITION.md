# RED_FIRST V5 — UNREPRODUCIBLE disposition (phase-check 3)

**Worktree / branch:** `D:/Projekty/_wt/intergrax/af-v5-p1-red` · `af-v5/p1-red-first`  
**Evidence integration head:** `5935fa4f0cd3596852b3a6500f9d4e1d613a53df`  
**ENT-iso remeasurement (2026-11-26, `af-v5/ent-iso` @ `2a6e197ca`):** prefix `754ee000675d71c1734656702beb0aef8ba58fd8`; probes in `tests/unit/architecture/test_nt_ent_iso_red_first_probes.py`.  
**PLG-marker remeasurement (2026-11-26, `af-v5/plg-marker`):** probes in `tests/unit/applications/test_nt_plg_red_first_probes.py`, `tests/unit/architecture/test_nt_prd_red_first_probes.py`, `tests/unit/architecture/test_nt_rpl_red_first_probes.py`.
**Canonical counts (rw_plan.v1.json after PLG-marker pass):** **28** `RED_AT_PREFIX` · **11** `RED_FIRST_UNREPRODUCIBLE`  
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

## Row dispositions (11 × RED_FIRST_UNREPRODUCIBLE)

| `frz_id` | Logged cause (summary) | Cause class | Why this is not positive red-first evidence | T-AF2-RED-CLASSES (recommendation) | Operator decision |
| --- | --- | --- | --- | --- | --- |
| `FRZ-PLG-02` | Tier-3 `llm_resolver` local `create_adapter()` | `import_not_mechanism` | Probe reused the OWN-03 delegation surface. It does not show an external structural implementation that works without a core patch. | **FIX** | not DECIDED |
| `FRZ-PLG-07` | `agent_runtime_governance_wiring.py` absent at `22993254b13a` | `import_not_mechanism` | A missing file is not a self-expansion assertion. | **FIX** | not DECIDED |
| `FRZ-RPL-04` | embedding resolver `default_embedding_provider_slug()` | `prefix_already_green` | Probe repeated CFG-02 on the same prefix. It does not show a vendor switch modifying consumers. | **FIX** | not DECIDED |
| `FRZ-FRZ-10` | `AttributeError`: `verify_arch_freeze_plan_traceability` has no `_self_test_cases` | `historical_self_test` | Post-freeze policy activation is a later W5/W8 node; the logged break is a missing historical self-test hook, not matrix mechanism proof. | **FIX** (re-home policy probe when W5/W8 owns FRZ-10) | not DECIDED |
| `FRZ-PRD-08` | P0 inventory test exit 0 at prefix (bypass count already 0) | `prefix_already_green` | Prefix tree already satisfied the bypass-count invariant; no red mechanism to replay. | **OUT** (recommendation: stays **OPEN** as historical non-evidence; no OUT operator citation) | not DECIDED |
| `FRZ-REL-07` | `ModuleNotFoundError: wiring_posture` (pytest exit 3) | `missing_wiring_module` | Broken import path on prefix, not “hidden infinite retry/fallback” behavior. | **FIX** (wire negative test to an existing REL mechanism probe) | not DECIDED |
| `FRZ-REL-09` | `ImportError: UnsupportedQueueBackendError` from `production_queue_resolver` | `import_not_mechanism` | Matrix concern is kafka/rabbitmq silently rewritten to eager Celery; prefix lacks the typed refusal API, so collection fails before any “broker refused / not rewritten” assertion. Integrated head passes `test_kafka_never_rewritten_to_celery` with `UnsupportedQueueBackendError` — **not** replayed as red-first at prefix in this campaign. | **FIX** (prefix-capable behavioral probe naming silent Celery rewrite) | not DECIDED |
| `FRZ-REL-10` | `ImportError: CapacityQualification` from `production_capacity_wiring` | `import_not_mechanism` | Concern is in-memory K8s scaler when cluster URL unset; prefix lacks `CapacityQualification` / fail-closed wiring surface used by integrated `test_fc2_capacity_fail_closed.py`. Import ≠ saturation fail-closed assertion at prefix. | **FIX** | not DECIDED |
| `FRZ-TEN-01` | `subprocess.TimeoutExpired` after 300s | `subprocess_timeout` | Timeout in nested pytest launcher, not a tenant-identity assertion failure. | **ENV** (harness duration / runner) unless a shorter single-node probe is bound | not DECIDED |
| `FRZ-TEN-07` | `ImportError: McpIdentityCompositionError` from `mcp_harness_principal` | `import_not_mechanism` | Matrix concern is MCP client-asserted tenant identity fail-closed; prefix tree has no `McpIdentityCompositionError` / `assert_mcp_identity_composition` — collection fails before behavioral identity-composition assertion. | **FIX** | not DECIDED |
| `FRZ-TEN-10` | `subprocess.TimeoutExpired` after 300s (same nested path as TEN-01) | `subprocess_timeout` | Same timeout class as TEN-01; not missing-tenant silent-global assertion proof. | **ENV** (harness duration / runner) | not DECIDED |

## Candidates not promoted (REL-09 / REL-10 / TEN-07)

Grok binding: these three may become **RED_AT_PREFIX** only with a **one-node** probe that **fails at the recorded prefix** with an assertion **naming the FRZ mechanism** (not import/marker/timeout/nested ENT exit 4) and **passes** on integration head `5935fa4f`. Current V5 logs show **import-only** failures at prefix; `test_fc2_queue_worker_fail_closed.py` and integrated tenant MCP tests were **not** present or not overlayable without the later typed APIs. **No conversion** for REL/TEN rows in the ENT-iso pass. PLG-02, PLG-07 and RPL-04 stay unpromoted after review. Residual UNREPRODUCIBLE count **11**.

## Explicit non-effects

This document does **not**:

- set **SCOPE**, **G2**, **R**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE CLOSED**;
- weaken `scripts/governance/verify_rw_plan_phase_check3.py`;
- mark any **T-AF2-RED-CLASSES** row as operator-**DECIDED**.

**Recorded:** 2026-11-26 · red-first V5 disposition pass on `af-v5/p1-red-first`.
