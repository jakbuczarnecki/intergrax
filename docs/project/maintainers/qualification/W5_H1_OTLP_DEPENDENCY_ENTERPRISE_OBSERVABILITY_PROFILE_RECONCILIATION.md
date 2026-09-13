# W5-H1 — OTLP Dependency & Enterprise Observability Profile Reconciliation

**Task:** W5-H1 — OTLP Dependency & Enterprise Observability Profile Reconciliation  
**Status:** QUALIFIED (pending commit SHA in session report)

## Root cause

W5-G (`test_enterprise_scale_resilience_w5_g_profile_activation.py`) is a frozen `gate` test run under canonical `uv run pytest`, but enterprise profile wiring constructed `CollectorTransport` → `OtlpTransport` with **module-level** `opentelemetry.*` imports while OpenTelemetry lived only in `dev-ci-rag`, not in `dependency-groups.test` or `dev-ci`. Test contract and dependency contract diverged.

## Dependency ownership

| Dependency | Group / extra | Required by | Owner |
|------------|---------------|-------------|-------|
| `opentelemetry-api` | `observability-otlp`, `test`, `dev-ci` | OTLP adapters, W5 gate suite | Optional capability |
| `opentelemetry-sdk` | same | `OtlpTransport` | Optional capability |
| `opentelemetry-exporter-otlp-proto-http` | same | HTTP OTLP export | Optional capability |
| `opentelemetry-exporter-otlp-proto-grpc` | same | gRPC OTLP export | Optional capability |

Core `[project].dependencies` remain free of OpenTelemetry.

## Optional capability model

- **Strategy A:** canonical certification (`uv run pytest`, `dev-ci` CI profile) installs OTLP packages via explicit declaration.
- Adapters use lazy SDK import + `require_otlp_observability_dependency_profile()` at composition and transport construction.
- Configured `DISTRIBUTED_OTLP` / `OTLP` without SDK → `ConfigurationError` (no silent fallback to NOOP).

## Test profile semantics

- Gate tests must not use `pytest.importorskip` for OTLP.
- Wiring contract (W5-G) and SDK adapter contract (W5-E/F) remain separate test modules.
- Packaging test asserts `observability-otlp` extra matches `test` group OTLP specs and `dev-ci` superset.

## Shutdown semantics

`close()` → `flush()` → `force_flush()` → `shutdown()`; second `close()` is a no-op (no duplicate flush/shutdown).

## Export failure semantics

Collector/OTLP export failures map to `OtlpTransportError` / metrics only — execution plane handlers still run (`test_export_failure_isolated_from_execution_plane`).

## No execution coupling

Mandatory evidence persistence fail-closed semantics are unchanged (EE-B1.1 out of scope). External OTLP export does not fail Execution Engine.

## Package / dependency matrix

Install for production OTLP: `uv sync --extra observability-otlp`.  
Install for default local gates: `uv sync` (includes `test` group per `[tool.uv] default-groups`).

## Test evidence

- `tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_g_profile_activation.py`
- `tests/unit/runtime/observability/test_w5_h1_otlp_dependency_contract.py`
- `tests/unit/runtime/observability/exporters/test_otlp_transport_adapter.py`
- Full observability unit tree + NPSC-5F / frozen matrix per session report

## Evidence plane drift (NPSC-5F Final)

Evidence-plane freeze baseline remains EE-FINAL-02 re-freeze `7a3569c64e892588992635c9cee10c264a9fc200`. W5-H1 and post-freeze observability integration drifts are tri-classified `QUALIFIED_COMPATIBLE` via qualification path prefixes in `testing_support/npsc5f_final_evidence_plane_drift.py` (see `W5_H1_FIX1_NPSC5F_FREEZE_BASELINE_PROVENANCE_REPAIR.md` for orphan SHA repair).

## Final verdict

OTLP dependency ownership is explicit; core runtime has no SDK coupling; missing configured capability fails with typed configuration error; W5-H1 qualification gates per session report.
