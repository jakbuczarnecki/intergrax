# T-IGX-083-A — EC-W3 host contract settlement 10 (EQO-W3 B-1 / EC-W3-10)

## Scope

PRE_G2_HOST_CONTRACT_SETTLEMENT EC-W3-HOST-CONTRACT-SETTLEMENT-10: notification catalog composition root (Package A), P2C guarded sandbox qualification (Package B), ProofReceipt v2 consumer migration (Package C), GCA strict smoke disposition (Package D), BND-05 owner handoff witness (Package E), evidence audit settlement and EQO-RV-01 (Packages F–G), targeted regressions (Package H), publication under `eqo_closure/runs/pre_g2_host_contract_settlement_10_<sha8>/`.

## Plan digest

Plan digest: n/a (EC-W3-HOST-CONTRACT-SETTLEMENT-10 operator batch; EQO-W3 Campaign B-1; EQO-CL-01 / EQO-RV-01 / EQO-X-06)

## Approved

Approved: operator-delegate @ 2026-11-27T17:10:00+01:00

Delegation: EC-W3-HOST-CONTRACT-SETTLEMENT-10 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-10 HOST CONTRACT SETTLEMENT)

## Write set

```text
applications/governed_contractor_application/host/environment_profile.py
applications/governed_contractor_application/host/factory.py
applications/governed_contractor_application/host/offline_demo.py
applications/governed_contractor_application/tests/host/test_governed_contractor_host_smoke.py
applications/governed_contractor_application/tests/host/test_cli_verification_hardening.py
tests/unit/applications/governed_contractor/test_prod_q_r1_dur_02_gca_composition_binding.py
tests/unit/applications/governed_contractor/test_prod_q_r1_r4_production_startup_path.py
tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_host_contract_settlement_10_*/
governance/tickets/T-IGX-083-A.md
```

## Test contract

Packages A–H regressions per EC-W3-10; `uv run ruff check` on write set; `git diff --check`; `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

COMPAT-X, ROADMAP-REPLAY-X, ARCH-FREEZE product paths, intergrax/** receipt writer changes, CI/hooks/lockfile, formal G2/EQO_CLOSED.

## Lifecycle state

in_progress