# T-IGX-016-B — IIAC ACT-01 activation gate correction

## Scope

Fail-closed correction of `verify_iiac_policy_activation.py` and the supporting activation gate, adversarial tests, and byte-preserving publication of existing Session B evidence. Policy status stays `NOT_IN_FORCE`.

Specification: [`IIAC_ACT01_CORRECTIVE_SPEC.md`](../../docs/project/maintainers/qualification/IIAC_ACT01_CORRECTIVE_SPEC.md)

## Plan digest

Plan digest: fc495625f1c8b907461f5639f82e72d42cbf9992d14adc83c6e0209be09d7f74

Source: `docs/project/maintainers/qualification/IIAC_ACT01_CORRECTIVE_SPEC.md`

## Approved

Approved: operator @ 2026-11-05T16:58:00+01:00 (IIAC-ACT-01 corrective implementation; not policy activation)

## Test contract

```powershell
uv run pytest tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q
uv run python scripts/governance/verify_iiac_policy_activation.py
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
uv run python scripts/governance/verify_iiac_policy_activation.py --expect-not-in-force
uv run python scripts/governance/verify_iiac_g0_inventory_completeness.py
uv run python scripts/governance/verify_od_ent_02_compensating_control.py
```

## Out of scope

- Setting `IIAC_POLICY_ACTIVATION_RECORD.json` to `IN_FORCE`
- Fabricated operator signature or signed `IIAC_OD_ACTIVATION_*.md`
- Self-issued PASS, G0 regeneration, FRZ mass PASS, EVOL-X / ENT-AUDIT-X / ARCH-FREEZE closure
- Docker, vendor, or production configuration changes

## Lifecycle state

quality_gate

## Checkpoint

IIAC pytest: 45 passed, 1 skipped (symlink creation unavailable; ACT-N14 traversal rejected). Blob digest test passed after the evidence commit. Canonical gate: default and `--require-in-force` exit 1 (`ACT-N01`); `--expect-not-in-force` exit 0. G0 completeness PASS. Parent baseline `5d6b44b05f90ecf420d45d01659ea64565b52848`. Gate logic `1d0722723687fb93d0a538161ca4cce376b3ecef`. Byte-correct evidence `2a6545f1d551814a0ba8896be79576c580353771`.
