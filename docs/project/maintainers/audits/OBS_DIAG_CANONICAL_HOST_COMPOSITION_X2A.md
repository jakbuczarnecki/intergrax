# OBS-DIAG-X2A — Canonical Host Diagnostic Composition Ownership

> **Maintainer audit / qualification evidence — not architecture SSOT.**
> **CURRENT AUTHORITY:** [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)

| Field | Value |
| ----- | ----- |
| **Program** | OBS-DIAG-X2A |
| **Branch** | `development` |
| **Verdict** | `PASS — CANONICAL HOST DIAGNOSTIC COMPOSITION OWNERSHIP CLOSED` |

## Target flow

```text
build_harness_host_runtime
  → diagnostic_composition_overrides (typed, optional)
  → materialize_host_diagnostic_read_dependencies (once)
  → HarnessHostRuntime.host_diagnostic_dependencies
      ├─ wire_terminal_execution_diagnostics (write)
      ├─ resolve_host_diagnostic_read_dependencies (read)
      ├─ build_diagnostic_scope_discovery_service (scope)
      └─ close_harness_host_runtime (host-owned persistence close)
```

## Proofs

| Proof | Module |
| ----- | ------ |
| Default factory attached + write/read identity | `tests/unit/applications/_shared/test_obs_diag_x2a_canonical_host_diagnostic_composition.py` |
| `resolve_diagnostic_persistence_composition` count == 1 per host build | same |
| Custom persistence / reconstruction / grouping via factory | same |
| Borrowed not closed / host shutdown closes host-owned | same |
| Strict required missing prerequisites fail-closed | same |
| X2 composition regressions | `test_obs_diag_x2_diagnostic_composition_pluginability.py` |

## Predecessor

Contract-driven composition (X2):
[`OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md`](OBS_DIAG_DIAGNOSTIC_COMPOSITION_PLUGINABILITY_X2.md).
