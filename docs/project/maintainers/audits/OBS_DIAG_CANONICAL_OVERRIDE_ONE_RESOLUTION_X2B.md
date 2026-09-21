# OBS-DIAG-X2B — Canonical Override One-Resolution Enforcement

> **Maintainer audit / qualification evidence — not architecture SSOT.**
> **CURRENT AUTHORITY:** [`DIAGNOSTICS.md`](../../architecture/DIAGNOSTICS.md)

| Field | Value |
| ----- | ----- |
| **Program** | OBS-DIAG-X2B |
| **Branch** | `development` |
| **Verdict** | `PASS — CANONICAL DIAGNOSTIC OVERRIDES PRESERVE ONE-RESOLUTION` |

## Defect closed

Host-bound helpers previously skipped `HarnessHostRuntime.host_diagnostic_dependencies`
when callers passed explicit `overrides=` (including resolved env-wiring overrides),
causing a second `resolve_diagnostic_persistence_composition` for custom/partial hosts.

## Invariant

```text
host configuration frozen at construction
→ runtime.host_diagnostic_dependencies authoritative
→ host-bound consumers never re-materialize persistence
→ conflicting post-build overrides → DiagnosticCompositionError (fail closed)
```

## Proofs

| Proof | Module |
| ----- | ------ |
| Resolution count == 1 (default, full, reader-only, grouping-only, partial) | `tests/unit/applications/_shared/test_obs_diag_x2b_canonical_override_one_resolution.py` |
| Write/read persistence identity + post-build conflict fail-closed | same |
| Borrowed vs host-created close semantics after multi-consumer exercise | same |
| X2A default-host proofs (unchanged) | `test_obs_diag_x2a_canonical_host_diagnostic_composition.py` |

## Predecessor

X2A canonical host ownership:
[`OBS_DIAG_CANONICAL_HOST_COMPOSITION_X2A.md`](OBS_DIAG_CANONICAL_HOST_COMPOSITION_X2A.md).
