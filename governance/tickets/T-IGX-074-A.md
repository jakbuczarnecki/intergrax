# T-IGX-074-A — EC-V3-W02 guarded OTLP and sandbox composition (A4-032a, A4-032b)

## Scope

Enterprise Campaign V3 wave **EC-V3-W02** (OD-EC-W02-01, OD-EC-W02-02): close EQOA-A4-032a (silent OTLP event-delivery degradation) and A4-032b / WP2-OBS-02 (implicit sandbox root under guarded composition). Decisions **DP-24**, **Q4=A**, **Q5=A**.

Primary write set:

- `intergrax/applications/_shared/runtime_event_delivery_wiring.py`
- `intergrax/applications/_shared/sandbox_wiring.py`
- `tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py`
- `tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py`

Conditional fixture-only compatibility: `strict_product_manifest_ci_gates.py`, `package_wiring.py`, `check_sandbox_policy_wiring.py`, `task_control_product_host_test_support.py`, `test_reference_capability_bundle.py`, `test_local_workspace_environment_profile.py` as required for explicit test-owned configuration.

## Approved

Approved: operator @ 2026-11-24T12:00:00+01:00

Authorization: OD-EC-W02-01, OD-EC-W02-02 (EC-V3-W02 corrective package).

## Test contract

```text
uv run pytest tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py -q
uv run pytest tests/unit/applications/test_reference_capability_bundle.py applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/scripts/test_check_application_production_gates.py -q
uv run pytest tests/unit/runtime/observability/test_otlp_exporter.py tests/unit/runtime/observability/test_otlp_http_transport.py tests/unit/runtime/observability/exporters/test_otlp_transport_adapter.py tests/unit/runtime/integrations/test_otlp_observability_integration.py -q
uv run pytest tests/unit/runtime/sandbox/test_sandbox_session.py tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py tests/unit/tools/providers/sandbox/test_sandbox_exec.py -q
uv run python scripts/maintenance/check_sandbox_policy_wiring.py
uv run python scripts/gates/check_application_production_gates.py
```

## Out of scope

T-IGX-043-B historical scope; EC-V3-W01 / PR #804; FRZ-SEC-06 qualification; IIAC; ARCH-FREEZE train merge; production default changes; CI/workflow/hook/pyproject changes.

## Lifecycle state

in_progress