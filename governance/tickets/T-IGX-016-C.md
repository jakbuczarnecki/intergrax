# T-IGX-016-C — IIAC governance certification (Session C certifier)

## Scope

Publish durable IIAC-GOV-1 governance certification evidence after independent acceptance of F01–F03 re-audit and ACT-01 re-audit. Policy status stays `NOT_IN_FORCE` until operator `IIAC_OD_ACTIVATION_*.md`.

## Plan digest

Source: operator mission IIAC governance final certification @ `ae467c8c4dd9c87521c1c60a7e5e75224d3214a0`

## Test contract

```powershell
uv run python scripts/governance/verify_iiac_audit_manifest.py docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/certification/iiac_audit_session_manifest.IIAC-GOV-1.json --repo .
uv run python scripts/governance/verify_iiac_role_separation.py docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/certification/iiac_audit_session_manifest.IIAC-GOV-1.json --strict-product-diff --repo .
uv run python scripts/governance/verify_iiac_policy_activation.py --expect-not-in-force
```

## Out of scope

- Fabricated operator activation signature
- Phase 1 / phase 2 `IN_FORCE` without authorized operator decision
- EVOL-X / ENT-AUDIT-X / ARCH-FREEZE closure
- Product-path code changes

## Lifecycle state

activated_in_force

## Checkpoint

OD-01 decision `IIAC-OD-01-2026-11-05` @ recording `2026-11-05T18:14:38+01:00`. Phase 1 `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`, phase 2 `847e9c84a1f2440161cce5a1510b7bb7ff8fcfad`. `--require-in-force` exit 0. Disposition: `IIAC GOVERNANCE — CERTIFIED — POLICY IN_FORCE — PUBLISHED`.
