# ec-v6-r3-ec02-closure

Continuation of EC-V6. Does not replace `agent-env/plans/ec-v6-ec02-closure.md`.
Integration branch `factory/ec-v6-r3-ec02-closure/int` from development `de0a621f8` (ancestor `5935fa4f0`; only docs/ticket annex `ARCH-FREEZE V5`, no product delta).
No push. P02–P10 are orchestrated on this candidate after P01; factory receive requires a commit, so qualify-only and authority-blocked packages are not fake commits.

- [x] P01 Recover guarded sandbox, OTLP, and package-gate lifecycle
  - deps: -
  - agent: igx-implementer
  - ticket: governance/tickets/T-IGX-074-A.md
  - write: intergrax/applications/_shared/sandbox_wiring.py, intergrax/applications/_shared/package_wiring.py, intergrax/applications/_shared/strict_product_manifest_ci_gates.py, intergrax/applications/_shared/runtime_event_delivery_wiring.py, scripts/maintenance/check_sandbox_policy_wiring.py, tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py, tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py
  - tests: uv run pytest tests/unit/applications/test_ec_v3_w02_sandbox_fail_closed.py tests/unit/applications/test_ec_v3_w02_otlp_fail_closed.py tests/unit/applications/test_fc2_sandbox_composition_fail_closed.py tests/unit/applications/test_fc2_otlp_composition_fail_closed.py -q
