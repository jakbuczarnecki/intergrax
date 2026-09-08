# NPSC-4.2-F1 — Tier-3 Factory Plugin Bootstrap Completion

**Verdict:** PASS  
**Date:** 2026-09-08  
**Branch:** `development`

## 1. Failure detected by NPSC-4.2-F

NPSC-4.2-F found seven Tier-3 factories still calling `bootstrap_nexus_platform(nexus_loop, …)` after `nexus_loop` compatibility resolution was removed. Each factory imported `bootstrap_harness_host_platform` but invoked the retired direct Nexus bootstrap path with an undefined `nexus_loop` symbol, causing `NameError` at factory entry.

## 2. Affected factories (seven)

| Factory | Application |
|---|---|
| `applications/legal_application/host/factory.py` | legal_application |
| `applications/governed_contractor_application/host/factory.py` | governed_contractor_application |
| `applications/dispute_sim_application/host/factory.py` | dispute_sim_application |
| `applications/intergrax_assistant_application/host/factory.py` | intergrax_assistant_application |
| `applications/attestation_demo/host/factory.py` | attestation_demo |
| `applications/poc_template_application/host/factory.py` | poc_template_application |
| `applications/lab_application/host/factory.py` | lab_application |

Discovery confirmed **no additional** Tier-3 `host/factory.py` consumers of `bootstrap_nexus_platform`. Reference factories (`local_workspace_application`, `research_application`) were already correct.

## 3. Root cause

Incomplete NPSC-4.2 migration: harness host runtime composition removed raw `nexus_loop` exposure, but seven factories were not updated to the canonical typed harness composition API. Imports were partially migrated; bootstrap calls were not.

## 4. Migration table

| Factory | Old platform bootstrap | New platform bootstrap | Application plugin migration | `nexus_loop` removed | Trace suppression removed |
|---|---|---|---|---|---|
| legal_application | `bootstrap_nexus_platform(nexus_loop, trace_store=…)` | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |
| governed_contractor_application | same | `bootstrap_harness_host_platform(runtime)` | unchanged (`bootstrap_harness_host_application_plugins`) | YES | YES |
| dispute_sim_application | same | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |
| intergrax_assistant_application | same | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |
| attestation_demo | same | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |
| poc_template_application | same | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |
| lab_application | same (variable `plugin_bootstrap`) | `bootstrap_harness_host_platform(runtime)` | N/A | YES | YES |

## 5. Before / after bootstrap architecture

**Before (broken):**

```text
Tier-3 Factory
      |
      v
bootstrap_nexus_platform(nexus_loop, trace_store=...)   # nexus_loop undefined
```

**After (canonical):**

```text
Tier-3 Factory
      |
      v
HarnessHostRuntime
      |
      +--> bootstrap_harness_host_platform(runtime)
      |
      +--> bootstrap_harness_host_application_plugins(runtime, plugins)   # where applicable
```

No raw `NexusLoop`, no `_orchestration_backend` access, no legacy resolver.

## 6. Plugin parity proof

- Default platform plugins: registered via `bootstrap_harness_host_platform(runtime)` which delegates to composition-owned `bootstrap_nexus_platform` internally.
- Application-specific plugins: `governed_contractor_application` observability export path unchanged — still calls `bootstrap_harness_host_application_plugins(runtime, [export_plugin])` and extends `platform.shutdown_callbacks`.
- Shutdown callbacks: all seven factories retain `attach_plugin_shutdown(app, platform.shutdown_callbacks)` (lab: `plugin_bootstrap.shutdown_callbacks`).

## 7. Startup proof

| Application | Result | Evidence |
|---|---|---|
| legal_application | PASS | `applications/legal_application/tests/host/test_legal_host_app.py` |
| governed_contractor_application | PASS | `applications/governed_contractor_application/tests/host/` (40 passed) |
| dispute_sim_application | PASS | `applications/dispute_sim_application/tests/host/test_dispute_sim_host_smoke.py` |
| intergrax_assistant_application | PASS | `applications/intergrax_assistant_application/tests/host/test_intergrax_assistant_host_smoke.py` |
| attestation_demo | PASS | Factory constructs; `test_attestation_demo_smoke` health/route tests pass |
| poc_template_application | PASS | `tests/unit/applications/test_poc_template_application.py` |
| lab_application | PASS | `tests/unit/applications/test_lab_harness_api_key_required.py` |

No `NameError`, no undefined `nexus_loop`, no raw Nexus bootstrap on factory entry.

## 8. Expanded harness regression

Exact NPSC-4.2-F expanded suite (5 files, 44 tests):

| Metric | Count |
|---|---|
| passed | 29 |
| failed | 15 |
| skipped | 0 |
| deselected | 0 |

Failure count **unchanged** from NPSC-4.2-F baseline. All 15 failures classified **PREEXISTING_UNRELATED** (EffectiveProfileRevisionError in test helpers, LKW projection fixture gaps, debug integration `run_id` contract, diagnostic readiness prerequisites). **0 new NPSC-4.2 bootstrap migration failures.**

## 9. Frozen execution gate results

| Gate | Result |
|---|---|
| NPSC-3C | PASS |
| NPSC-3E | PASS |
| NPSC-3F | PASS |
| NPSC-3G | PASS |
| NPSC-4 | PASS |
| NPSC-4.1 | PASS |
| NPSC-4.2 | PASS (incl. new `test_npsc42f1_tier3_host_factories_use_canonical_harness_bootstrap`) |
| NPSC-4.2-R1 | PASS |
| UE-10R1–R4 | PASS |
| UE-10R41 | KNOWN_PREEXISTING_FAIL (unchanged) |
| UE-11GP | PASS |

Execution + interaction: **674 passed**. Core harness/composition: **38 passed**.

## 10. Remaining debt

- UE-10R41 local-import hygiene (pre-existing, out of F1 scope).
- Tier-3 `trace_store` `# type: ignore[arg-type]` on `local_workspace_application` and `research_application` factories (not in F1 seven-factory set).
- Expanded harness/composition 15 failures (test-fixture / diagnostic prerequisites — not bootstrap migration).
- Failed NPSC-4.2-F certification document remains untracked historical evidence.

## 11. Final F1 verdict

**NPSC-4.2-F1: PASS** — seven broken Tier-3 factories migrated to canonical harness bootstrap; static gate strengthened; factory startup restored; no new regression attributable to NPSC-4.2 bootstrap migration.

**Next:** Rerun NPSC-4.2-F final clean certification.
