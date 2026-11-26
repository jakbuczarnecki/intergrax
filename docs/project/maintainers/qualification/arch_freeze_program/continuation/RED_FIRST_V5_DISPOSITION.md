# RED_FIRST V5 — UNREPRODUCIBLE disposition (phase-check 3)

**Worktree / branch:** `D:/Projekty/_wt/intergrax/af-v5-p1-red` · `af-v5/p1-red-first`  
**Evidence integration head:** `5935fa4f0cd3596852b3a6500f9d4e1d613a53df`  
**Canonical counts (rw_plan.v1.json):** **21** `RED_AT_PREFIX` · **18** `RED_FIRST_UNREPRODUCIBLE`  
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

## Row dispositions (18 × RED_FIRST_UNREPRODUCIBLE)

| `frz_id` | Logged cause (summary) | Cause class | Why this is not positive red-first evidence | T-AF2-RED-CLASSES (recommendation) | Operator decision |
| --- | --- | --- | --- | --- | --- |
| `FRZ-CFG-02` | Nested ENT-A-VBN-02 exit 4, “Gate check failed” on EBH-6 gate | `nested_ent_gate_exit4` | Failure is the ENT nested gate harness, not a pytest assertion that invalid/missing config silently selects a vendor/backend. | **FIX** (later single-node probe can encode CFG fail-closed without ENT nesting) | not DECIDED |
| `FRZ-CFG-08` | Nested VBN-SHADOW-SELECTION exit 4 on config_x gate | `nested_ent_gate_exit4` | Same nested-gate artefact; not a regression-gate assertion on hard-coded selection/fallback. | **FIX** | not DECIDED |
| `FRZ-FRZ-10` | `AttributeError`: `verify_arch_freeze_plan_traceability` has no `_self_test_cases` | `historical_self_test` | Post-freeze policy activation is a later W5/W8 node; the logged break is a missing historical self-test hook, not matrix mechanism proof. | **FIX** (re-home policy probe when W5/W8 owns FRZ-10) | not DECIDED |
| `FRZ-OWN-02` | Nested ENT-A-OWN-02 exit 4 on EBH-3 gate | `nested_ent_gate_exit4` | Nested audit exit, not “one canonical contract per responsibility” asserted in an isolated negative test. | **FIX** | not DECIDED |
| `FRZ-OWN-03` | Nested ENT-A-OWN-03 exit 4 on EBH-3 gate | `nested_ent_gate_exit4` | Same as OWN-02 for composition-owner concern. | **FIX** | not DECIDED |
| `FRZ-PLG-01` | `langchain_ollama` missing; `test_nt_plg_01` exit 0 at prefix | `optional_dep_probe_green` | Optional dep / collection noise; dedicated probe green at prefix does not prove PLG contract exposure red-first. | **ENV** (optional extra) + **FIX** (probe that fails on mechanism without optional SDK) | not DECIDED |
| `FRZ-PLG-02` | `testing_support.pytest_temp_plugin` missing; probe exit 0 | `optional_dep_probe_green` | Harness module absent at collection; probe already passes at prefix. | **FIX** (stable harness path or ship helpers at prefix replay) | not DECIDED |
| `FRZ-PLG-06` | `tests.qualification.governance.gr11.catalog` missing | `optional_dep_probe_green` | Qualification catalog module not on prefix tree; import error ≠ plugin-authority mechanism. | **FIX** | not DECIDED |
| `FRZ-PLG-07` | `pytest_temp_plugin` missing; probe exit 0 | `optional_dep_probe_green` | Same pattern as PLG-02. | **FIX** | not DECIDED |
| `FRZ-PRD-03` | `'ent_semantic_gate' not found in markers` | `pytest_marker_config` | Pytest ini marker registration, not “unsupported production configuration fails closed”. | **FIX** (marker + semantic gate probe) | not DECIDED |
| `FRZ-PRD-08` | P0 inventory test exit 0 at prefix (bypass count already 0) | `prefix_already_green` | Prefix tree already satisfied the bypass-count invariant; no red mechanism to replay. | **OUT** (recommendation: stays **OPEN** as historical non-evidence; no OUT operator citation) | not DECIDED |
| `FRZ-REL-07` | `ModuleNotFoundError: wiring_posture` (pytest exit 3) | `missing_wiring_module` | Broken import path on prefix, not “hidden infinite retry/fallback” behavior. | **FIX** (wire negative test to an existing REL mechanism probe) | not DECIDED |
| `FRZ-REL-09` | `ImportError: UnsupportedQueueBackendError` from `production_queue_resolver` | `import_not_mechanism` | Matrix concern is kafka/rabbitmq silently rewritten to eager Celery; prefix lacks the typed refusal API, so collection fails before any “broker refused / not rewritten” assertion. Integrated head passes `test_kafka_never_rewritten_to_celery` with `UnsupportedQueueBackendError` — **not** replayed as red-first at prefix in this campaign. | **FIX** (prefix-capable behavioral probe naming silent Celery rewrite) | not DECIDED |
| `FRZ-REL-10` | `ImportError: CapacityQualification` from `production_capacity_wiring` | `import_not_mechanism` | Concern is in-memory K8s scaler when cluster URL unset; prefix lacks `CapacityQualification` / fail-closed wiring surface used by integrated `test_fc2_capacity_fail_closed.py`. Import ≠ saturation fail-closed assertion at prefix. | **FIX** | not DECIDED |
| `FRZ-RPL-04` | `'ent_semantic_gate' not found in markers` | `pytest_marker_config` | Marker configuration gap, not vendor/backend switch consumer isolation. | **FIX** | not DECIDED |
| `FRZ-TEN-01` | `subprocess.TimeoutExpired` after 300s | `subprocess_timeout` | Timeout in nested pytest launcher, not a tenant-identity assertion failure. | **ENV** (harness duration / runner) unless a shorter single-node probe is bound | not DECIDED |
| `FRZ-TEN-07` | `ImportError: McpIdentityCompositionError` from `mcp_harness_principal` | `import_not_mechanism` | Matrix concern is MCP client-asserted tenant identity fail-closed; prefix tree has no `McpIdentityCompositionError` / `assert_mcp_identity_composition` — collection fails before behavioral identity-composition assertion. | **FIX** | not DECIDED |
| `FRZ-TEN-10` | `subprocess.TimeoutExpired` after 300s (same nested path as TEN-01) | `subprocess_timeout` | Same timeout class as TEN-01; not missing-tenant silent-global assertion proof. | **ENV** (harness duration / runner) | not DECIDED |

## Candidates not promoted (REL-09 / REL-10 / TEN-07)

Grok binding: these three may become **RED_AT_PREFIX** only with a **one-node** probe that **fails at the recorded prefix** with an assertion **naming the FRZ mechanism** (not import/marker/timeout/nested ENT exit 4) and **passes** on integration head `5935fa4f`. Current V5 logs show **import-only** failures at prefix; `test_fc2_queue_worker_fail_closed.py` and integrated tenant MCP tests were **not** present or not overlayable without the later typed APIs. **No conversion** in this commit — counts remain **21 / 18**.

## Explicit non-effects

This document does **not**:

- set **SCOPE**, **G2**, **R**, **FCS**, **M3**, **ENTRY**, or **ARCH-FREEZE CLOSED**;
- weaken `scripts/governance/verify_rw_plan_phase_check3.py`;
- mark any **T-AF2-RED-CLASSES** row as operator-**DECIDED**.

**Recorded:** 2026-11-26 · red-first V5 disposition pass on `af-v5/p1-red-first`.