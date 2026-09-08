# NPSC-4.2-R1 — Typed Harness Composition Contract Correction

**Verdict:** PASS  
**Date:** 2026-09-08  
**Branch:** `development`

## 1. Root cause

`bootstrap_harness_host_platform()` resolves:

```text
resolved_trace = trace_store or runtime.observability.trace_store
```

and passes the result to `bootstrap_nexus_platform(trace_store=...)`, which requires `RunTraceReader | None`.

`NexusObservabilityStores.trace_store` was declared as `RunTraceWriter`, which does not guarantee read capability. The mismatch was suppressed with `# type: ignore[arg-type]` in `harness_host_composition.py`.

## 2. Previous incorrect type relation

| Surface | Declared type | Semantic need |
|---|---|---|
| `NexusObservabilityStores.trace_store` | `RunTraceWriter` | read + write |
| `bootstrap_nexus_platform(trace_store=...)` | `RunTraceReader \| None` | read for plugins/export |
| `bootstrap_harness_host_platform()` fallback | `RunTraceWriter` via observability | reader for plugin bootstrap |

## 3. Selected canonical contract

**Solution:** introduce `RunTraceStore(RunTraceWriter, RunTraceReader)` in `intergrax/runtime/nexus/tracing/persistence_models.py` and type `NexusObservabilityStores.trace_store` as `RunTraceStore`.

**Contract owner:** `intergrax/runtime/nexus/tracing/persistence_models.py`

**New abstraction:** YES — `RunTraceStore` combines existing reader/writer ABCs without duplicating methods.

**Why unavoidable:** every canonical observability-wired store (`InMemoryRunTraceStore`, `SQLiteRunTraceStore`, SQLite opener paths) already implements both capabilities; the observability contract under-stated that guarantee.

## 4. No-suppression proof

- `# type: ignore[arg-type]` removed from `bootstrap_harness_host_platform`
- no `cast`, `Any`, or reflection workaround introduced
- gate: `test_npsc42_harness_host_platform_bootstrap_has_no_trace_type_suppression`

## 5. Orchestration-backend hermeticity proof

Gate: `test_npsc42_orchestration_backend_access_confined_to_allowlist`

| Path | Reason |
|---|---|
| `intergrax/applications/_shared/harness_host_composition.py` | composition owner — platform plugin bootstrap only |
| `applications/local_workspace_application/model_runtime_proof/runtime.py` | internal model-runtime proof composition root |
| `scripts/maintenance/check_harness_security_wiring.py` | maintenance assembly verification |
| `scripts/maintenance/check_harness_reliability_wiring.py` | maintenance assembly verification |

Tier-3 host factories: **0** direct `_orchestration_backend` access.

## 6. Regression evidence

- `tests/unit/runtime/nexus/tracing/test_run_trace_store_contract.py`
- `tests/unit/runtime/architecture/test_npsc4_2_residual_compatibility_gate.py`
- focused harness/observability/platform bootstrap tests (session run log)
- frozen NPSC / UE gates (session run log)

## 7. Remaining unrelated debt

- `open_trace_store_from_profile()` still returns `Any` (pre-existing integration-profile seam)
- `NexusLoop.trace_store` remains `RunTraceWriter | None` with runtime `isinstance(RunTraceReader)` probe (orchestration write path; unchanged by R1)
- UE-10R41 local-import issue: classify separately if still present
