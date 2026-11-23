# T-IGX-063-A — EQO Wave 05 test remediation (Nexus HITL, XD-02 revoke, HARNESS-01 classification)

## Scope

Minimal qualification/integration test fixes from EQO Wave 05 packages P2 (W05-06, W05-07) and P3 (W05-08). No product behavior or security contract changes.

- W05-06: inject `LoggingNotificationAdapter` in `test_nexus_loop_governance.py` where long-running HITL paths exercise `LongRunningCoordinator.notify_hitl_pause` (fail-closed notifier unchanged in `coordinator.py`).
- W05-07: align `ArtifactAccessContext.run_id` with default artifact `run_id` in XD-02 revoke scenario (`make_context` optional `run_id`, test uses `run-1`).
- W05-08: closed-world row for `intergrax/applications/_shared/memory_provider_admission.py` (`HOST_EXECUTION_COMPOSITION` via existing `_shared` rule).

## Approved

Approved: operator @ 2026-11-23 (EQO Wave 05 sequential remediation campaign authorization).

## Write set

```text
tests/integration/runtime/test_nexus_loop_governance.py
tests/qualification/pdc_x/d8/_d8_support.py
tests/qualification/pdc_x/xd/test_xd_02_agent_context.py
tests/qualification/harness_01/nexus_import_inventory.py
governance/tickets/T-IGX-063-A.md
```

## Test contract

```text
pytest tests/integration/runtime/test_nexus_loop_governance.py tests/qualification/pdc_x/xd/test_xd_02_agent_context.py::test_xd02_revoked_agent_loses_context_access_and_the_denial_is_audited tests/qualification/harness_01/test_harness_01_gates.py::test_harness_01_higher_layer_nexus_imports_are_classified -o cache_dir=<scratch> --basetemp <scratch>
```

## Lifecycle state

`approved`