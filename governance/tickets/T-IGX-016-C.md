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

ready_for_operator_activation

## Checkpoint

ACT-01 dossier digest `c405c65faf53243a0511b0b9584a9f3abda4db08c09ee9b9266c86fe9f6b7a4e` published under `iiac_evidence/IIAC-GOV-1/certification/`. Disposition: `IIAC GOVERNANCE — CERTIFIED — ACTIVATION BLOCKED BY EXACT OPERATOR GATE`.
