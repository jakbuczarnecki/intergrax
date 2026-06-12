---
id: IJ-2026-06-11-012
date: 2026-06-11
tiers:
  - tier-3
scope: AGENT_CONTRACTS
plan_ref:
  - ACP-CLOSE-ORG-2
status: completed
commit: pending
adr: none — extends ACP-ORG-5 lab fixture to product host manifests
---

# ACP-CLOSE ORG-2 — UC-11 product host compliance golden

## Operator request

Execute next ACP-CLOSE sprint: UC-11 compliance golden eval per Tier-3 product host, beyond lab-only fixture.

## Summary

Added `product_host_org_envelope`, `ApplicationEnvironmentProfile.with_uc11_organizational_policy`, and `uc11_compliance_golden` helpers. Gate tests cover six product manifests (legal, research, dispute_sim, local_workspace, poc_template, intergrax_assistant) — merge materializes `OrganizationalPolicyContext` and kernel happy-path steps assert zero policy denials.

## Project impact

Product hosts can attach org envelopes for virtual-workforce deployments with CI-verified happy-path compliance, not only `lab_org_virtual_workforce_defaults`.

## Traceability

| Link | Target |
|------|--------|
| Architecture | `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` §39.5 · UC-11 |
| Plan | `ACP-CLOSE-ORG-2` |
| ADR | none |

## Changed artifacts

- `intergrax/applications/contracts/org_policy.py` — `product_host_org_envelope`
- `intergrax/applications/contracts/environment_profile.py` — `with_uc11_organizational_policy`
- `intergrax/applications/_shared/uc11_compliance_golden.py` — golden helpers (new)
- `tests/unit/applications/test_uc11_product_host_compliance.py` — 6 hosts × 2 tests (new)

## Verification

```bash
uv run pytest tests/unit/applications/test_uc11_product_host_compliance.py -m gate -q
```

Result: 12 passed.

## Risks and follow-ups

- Product manifests do not enable org envelope by default — hosts opt in via profile copy.
- ACP-CLOSE-LEG-4 / PAT-1 / CI-2 remain open.
