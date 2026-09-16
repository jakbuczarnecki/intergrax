# HARDENING-9 — NPSC-5F Protected Drift Requalification

**Task:** `HARDENING_9_NPSC5F_PROTECTED_DRIFT_REQUALIFICATION`  
**Status:** `REQUALIFIED / RE-FROZEN / PASS`  
**Clean baseline SHA (pre-re-freeze):** `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` (`origin/development` at audit)  
**Evidence Plane prior sentinel:** `52b9dc41ed7dd83e5516d852f1ef7295cc0b10af` (OBS-ASOF-REBASE-R1)  
**R2 prior post-qualified sentinel:** `7a3569c64e892588992635c9cee10c264a9fc200` (EE-FINAL-02)

## Executive summary

Three NPSC-5F protected-drift sentinels failed on integrated `development` (audit tip `a2b33ba96`) because **OBS-CONTRACT-BOUNDARY-1** / **R1** / **R2** moved neutral Evidence Plane types into `intergrax/contracts/` and reduced runtime modules to contract-preserving shims. Architecture intent: **platform consumes contracts, not concrete runtime DTOs** (Diagnostics and other tiers must not import observability reconstruction implementations).

**Classification:** `QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED` for all drift items.  
**UNKNOWN:** `0` · **REAL_ARCHITECTURE_REGRESSION:** `0` · **UNQUALIFIED DRIFT:** `0`

**Production code changed in this requalification task:** NO — sentinel baseline + qualification evidence only.

## Failing gates (repro @ pre-re-freeze `origin/development`)

| Gate | Module |
|------|--------|
| `test_r2_final_no_unqualified_protected_drift_since_qualified_baseline` | `testing_support/npsc5f_r2_protected_drift.py` |
| `test_final_no_breaking_protected_drift_since_baseline` | `testing_support/npsc5f_final_evidence_plane_drift.py` |
| `test_npsc5f_final_protected_drift` | same Evidence Plane sentinel |

## Baseline authority (single SSOT)

| Sentinel | Owner module | Who may advance | Condition |
|----------|--------------|-----------------|-----------|
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `testing_support/npsc5f_final_evidence_plane_drift.py` | Maintainer qualification + matrix PASS | Documented qualified drift; no BREAKING unclassified production change |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `testing_support/npsc5f_r2_protected_drift.py` | R2 Final / enterprise reconciliation | R2 journal ordering invariants preserved |

Drift detection: `git diff --name-only {baseline}..origin/development` → path classifier (no content-hash in gate; git path change since baseline).

## Introducing commits

| SHA | Message |
|-----|---------|
| `20af8b8e6af2e901fe7b7811dadf979ade13d63c` | OBS-CONTRACT-BOUNDARY-1: make execution reconstruction contract-owned |
| `6a5a804e6c10f04d12d7a65768fedce3eec542f1` | OBS-CONTRACT-BOUNDARY-1-R1: close positioned runtime evidence boundary |
| `ba3fa440d64273c733f29cfaa6e96c4af373180d` | OBS-CONTRACT-BOUNDARY-1-R2: make runtime event contract deterministic |

No protected Evidence Plane path changes after `ba3fa440d` through `a2b33ba96` (MP-4R7 governance commit is applications-only).

## Protected drift ledger

| Protected artifact | Baseline blob @ `52b9dc41` | Current blob @ `a2b33ba96` | Introducing commit | Qualification | Classification | Action |
|--------------------|----------------------------|----------------------------|--------------------|---------------|----------------|--------|
| `intergrax/contracts/runtime_event.py` | ABSENT (new contract surface) | `78e722053a119de429682fb99728e89318c97169` | `6a5a804e6` | This doc + boundary tests | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/events/event_taxonomy.py` | `aafc42afe2fa13a6956a4d5e0d89e6c9a0b8f2cd` | `e0be503bce11a0990d61b901a33694fdc02ffe80` | `6a5a804e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/events/event_catalog.py` | (see `52b9dc41` tree) | (see `a2b33ba96` tree) | `ba3fa440d` | This doc + contract boundary tests | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/contracts/runtime_event_type.py` | ABSENT | (new @ `ba3fa440d`) | `ba3fa440d` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/contracts/spine_event_metadata.py` | ABSENT | (new @ `ba3fa440d`) | `ba3fa440d` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/events/execution_position.py` | `9496b0d105cdc7b35c49730a7e98f813d3b2c915` | `2e08c331bd0fb1e3bbb548c589e7a6c381529cc0` | `20af8b8e6` | This doc + R2 gates | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze R2 + Final |
| `intergrax/runtime/events/runtime_event.py` | `2934e999fde5244db59e38769ec66161f63516ce` | `b5ab60c8cc5534c859c079f9b8ded8da5c2bc0da` | `6a5a804e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/events/w3c_trace_context.py` | `2198b1058d081dcb4b39ab54b63da2ffff39ff4e` | `a229eabd34d662e6ebf0d89024f2e621e6d85713` | `6a5a804e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/observability/causal_evidence.py` | `4cd375beacca329e9fe1602a6c6530d4a94f4c98` | `fe7d5840cd61477cb361d3e9d56115ddc9501032` | `20af8b8e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/observability/reconstruction/__init__.py` | `32fe36d1f753cb0534393e31d9efc09cfb25b117` | `9c51075c280cd104a0faebd4d0d4cf7b2e1f5565` | `20af8b8e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/observability/reconstruction/execution_lineage_reconstruction.py` | `95e35a45cd54ef40dbdefbd6a89dd103ae921805` | `a0820224ce2fbc2b26d14cb450390b6c39bd9c22` | `20af8b8e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |
| `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` | `abb8f31f5b6c6911cffa72f44fa9a5740ee26951` | `b70320b4e4bdc90ce409cb730bb152a3a90358a9` | `20af8b8e6` | This doc | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | Re-freeze |

R2-only drift from `7a3569c64`: `execution_position.py` only (same row).

## Code vs evidence drift

| Question | Answer |
|----------|--------|
| Production behavior change? | **No** semantic regression — types lifted to contracts; runtime shims re-export; reconstruction readers typed on contract ports |
| Evidence-only change? | **No** — production paths changed; sentinels were stale |
| Contract/interface change? | **Yes** — additive contract modules (`runtime_event`, `platform_causal_evidence`, reconstruction models); consumers updated to contracts |
| Formatting-only? | **No** |

## Architecture / pluginability / authority / boundaries

| Audit | Result |
|-------|--------|
| Contract boundaries | **PASS** — diagnostics depend on `intergrax.contracts.*` reconstruction types |
| Pluginability (`CONTRACT → configured implementation`) | **PASS** — no new hardcoded providers; no hidden fallback |
| Duplicate authority | **0** |
| Layer boundaries | **PASS** — `test_hardening_3_layer_boundary_gate` updated in introducing commits |
| Evidence Plane ownership (no execution control) | **PASS** — read models only |

**Tests (introducing commits):** `test_execution_reconstruction_contract_boundary.py`, `test_runtime_event_contract_boundary.py`.

## Re-freeze (controlled)

| Sentinel | Old | New | Reason |
|----------|-----|-----|--------|
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `52b9dc41…` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | Last integrated `development` with qualified OBS-CONTRACT-BOUNDARY drift |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `7a3569c64…` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | `execution_position.py` shim only; journal ordering unchanged |
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `ed780d47…` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | `runtime_event.py` shim; durable commit path unchanged |

`R4_POST_QUALIFIED_BASELINE_SHA` remains `52b9dc41…` — `observability/reconstruction/**` explicitly not R4 file-frozen; no R4 drift detected.

## Regression evidence (post re-freeze)

Record in commit CI / local run:

- Three protected-drift tests: **PASS**
- `test_npsc5f_final_mandatory_regression_matrix_passes`: **PASS** (exact counts in run log)
- RC-01 / RC-02 / RC-03 / H6 / Root Admission / Production→testing_support: unchanged closure state

## Test weakening

`SKIP` / `XFAIL` / assertion removal / protected-path removal / hash weakening / allowlist expansion for unclassified drift: **0**

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu znajdującego się aktualnie na GitHub.

Post-push: verify introducing commits `20af8b8e6`, `6a5a804e6` on `development`, sentinel SHAs match this document, matrix PASS, no production workaround in re-freeze commit.
