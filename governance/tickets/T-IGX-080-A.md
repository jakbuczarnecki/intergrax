# T-IGX-080-A — EC-W3 host policy corrective (EQO-CL-01 / EC-W3-07)

## Scope

Enterprise Campaign EQO pre-G2 host policy corrective execution EC-W3-HOST-POLICY-CORRECTIVE-07: reference host policy fragment declarations in environment presets, symmetric HITL requirement enforcement in skill bridge wiring, regression and evidence under `eqo_closure/runs/pre_g2_host_policy_corrective_07_<sha8>/`. Production authorization for Package D reference declarations is delegated from approved [`T-IGX-076-A`](T-IGX-076-A.md) (ARCH-FREEZE Package D).

## Plan digest

Plan digest: n/a (EC-W3-HOST-POLICY-CORRECTIVE-07 EQO-CL-01 corrective batch; canonical Package D: [`governance/tickets/plans/PLAN_T-IGX-076-A.md`](plans/PLAN_T-IGX-076-A.md))

## Approved

Approved: operator-delegate @ 2026-11-27T11:46:00+01:00

Delegation: EC-W3-HOST-POLICY-CORRECTIVE-07 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-07); production presets and skill bridge changes under T-IGX-076-A Package D

## Write set

```text
intergrax/applications/contracts/environment_profile/presets.py
intergrax/applications/_shared/skill_bridge_wiring.py
tests/unit/applications/test_eqo_reference_host_policy_fragments.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_host_policy_corrective_07_*/**
governance/tickets/T-IGX-080-A.md
```

## Test contract

```text
uv run ruff check intergrax/applications/contracts/environment_profile/presets.py intergrax/applications/_shared/skill_bridge_wiring.py tests/unit/applications/test_eqo_reference_host_policy_fragments.py
uv run pytest tests/unit/applications/_shared/test_fc1_harness_strict_persistence_wiring.py tests/unit/applications/test_mcp_server_p4_lifecycle.py tests/unit/applications/test_eqo_reference_host_policy_fragments.py tests/unit/applications/test_skill_bridge_wiring.py tests/unit/applications/test_arch_freeze_skill_bridge_package_d.py tests/unit/applications/test_governance_permission_preset.py tests/unit/skills/test_p1_10_execution_binding.py -q
git diff --check
uv run python -m scripts.governance.igx_gov ticket
```

Five-group mapped regression per DES_PREREG and EC-W3-07 Package F.

## Out of scope

T-IGX-075-D test-only scope, EDER/FRZ register edits, EQO closure promotion, remaining 29-file qualification queue execution, skill manifest changes, Legal production domain policy changes.

## Lifecycle state

in_progress