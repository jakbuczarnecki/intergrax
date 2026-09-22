# OBS/DIAG frozen change policy

OBS/DIAG enterprise core is **frozen** after FINAL-3 exact-SHA qualification. Certified code remains at `c5acfc17ecea4d0e4b03405eabf735799c020ff8`.

## Frozen

- Canonical **authority** and **ownership** (one `DiagnosticOrchestrator`, one `ProblemLifecycleEngine`, one default `ExecutionReconstructor` construction spine).
- Public **contract surfaces** listed in `testing_support/obs_diag_freeze/manifest.py`.
- Certified **invariants** (`ObsDiagFrozenInvariantId`).

## Still pluginable

- Vendor adapters, persistence implementations, and grouping strategies that implement existing contracts without new authority paths.

## Change control

- **SAFE_EXTENSION** — new provider/adapter/strategy behind an existing contract; freeze guard stays green.
- **REQUALIFICATION_REQUIRED** — contract, ownership, bypass, or identity/reconstruction semantics change; guard fails closed (`ObsDiagRequalificationSignal`).

Regression entrypoint: `tests/unit/runtime/architecture/test_obs_diag_frozen_boundary_change_control_guard.py` (`pytest -m obs_diag_freeze`).
