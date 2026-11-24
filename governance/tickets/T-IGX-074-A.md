# T-IGX-074-A — EC-V3-W02 guarded OTLP and sandbox composition (A4-032a, A4-032b)

## Scope

Enterprise Campaign V3 wave EC-V3-W02 (OD-EC-W02-01, OD-EC-W02-02): corrective implementation for EQOA-A4-032a (silent OTLP event-delivery degradation) and A4-032b / WP2-OBS-02 (implicit sandbox root under guarded composition). Decisions DP-24, Q4=A, Q5=A. No finding closure or certification is asserted.

Exact candidate write set (seven paths, from RECOVERY-02):

- `applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py`
- `intergrax/applications/_shared/runtime_event_delivery_wiring.py`
- `intergrax/applications/_shared/sandbox_wiring.py`
- `scripts/maintenance/check_sandbox_policy_wiring.py`
- `tests/unit/applications/task_control_product_host_test_support.py`
- `tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py`
- `tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py`

Only explicit fixture configuration is permitted in existing test and maintenance paths; existing gate assertions and meaning remain unchanged. Preserve PRODUCT or STRICT fail-closed behavior, existing non-guarded behavior, actual tool catalog membership, explicit root precedence and OTLP settings precedence.

## Approved

Approved: operator @ 2026-11-24T12:00:00+01:00

Authorization: OD-EC-W02-01, OD-EC-W02-02 (EC-V3-W02 corrective package). The approval line is transcribed from the Operator-supplied RECOVERY-02 ticket; its exact timestamp is not independently verified. Explicit approval of both decisions was given in the campaign conversation. This registration is by the external architecture authority, not the implementer.

## Test contract

```text
uv run pytest tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py -q
uv run pytest tests/unit/applications/test_reference_capability_bundle.py applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/scripts/test_check_application_production_gates.py -q
uv run pytest tests/unit/runtime/observability/test_otlp_exporter.py tests/unit/runtime/observability/test_otlp_http_transport.py tests/unit/runtime/observability/exporters/test_otlp_transport_adapter.py tests/unit/runtime/integrations/test_otlp_observability_integration.py -q
uv run pytest tests/unit/runtime/sandbox/test_sandbox_session.py tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py tests/unit/tools/providers/sandbox/test_sandbox_exec.py -q
uv run python scripts/maintenance/check_sandbox_policy_wiring.py
uv run python scripts/gates/check_application_production_gates.py
```

Run sequentially under the existing pytest mutex. Repeat on the actual train candidate; compare any failures on its exact base with equivalent dependencies and configuration. Preserve prior reported red-first evidence separately; do not count collection errors as behavioral evidence. Run Ruff, scoped typing, git diff --check and igx_gov ticket. Publication is draft-only, not acceptance.

## Recovery and publication

Original local source 5ff7b9abd479eb903b9e43e1d9cf8f99dc42cae5, immediate parent 6495498f7f8046117528e40243e78e116d9595d8. Recover only the seven-path patch on the authorized train base containing this ticket. Keep the original candidate unchanged. Do not import development history. Candidate contains no governance files. External authority registers the per-wave publication record; implementer uses only the canonical publisher.

## Out of scope

T-IGX-043-B historical scope; EC-V3-W01 / PR #804; FRZ-SEC-06 qualification; IIAC; ARCH-FREEZE train merge; production default changes; CI/workflow/hook/pyproject changes; unrelated historical regression repair; autonomous integration or certification.

## Lifecycle state

in_progress
