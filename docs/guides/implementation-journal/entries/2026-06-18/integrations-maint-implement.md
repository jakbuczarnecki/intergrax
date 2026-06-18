---
id: IJ-2026-06-18-034
date: 2026-06-18
tiers:
  - tier-0
scope: INTEGRATIONS
plan_ref:
  - INT-MAINT-02
  - INT-MAINT-03
  - INT-MAINT-04
status: completed
commit: pending
adr: none — manifest metadata field; probe tests; cross-ref docs
---

# INT-MAINT-02..04 — audit maintenance implementation

## Summary

Added `requires_local_container` to integration manifests and catalog entries. Parametrized P4 shell health probe unit test plus CI check scripts. Added SaaS-only slug index and honesty gate. Documented nginx/ingress ECP bridge in architecture and `intergrax/integrations/USAGE.md`.

## Changed artifacts

- `intergrax/integrations/core/manifest.py`, `contracts/base.py`
- `intergrax/integrations/_shared/saas_only_slugs.py`, `USAGE.md`
- `scripts/check_integration_p4_shell_probes.py`, `check_integration_saas_honesty.py`
- `tests/unit/integrations/providers/test_p5_m6_p4_providers.py`

## Verification

```bash
uv run pytest tests/unit/integrations/providers/test_p5_m6_p4_providers.py::test_p4_shell_health_probe -q
uv run python scripts/check_integration_p4_shell_probes.py
uv run python scripts/check_integration_saas_honesty.py
```

## Risks

SaaS slug list is curated manually — extend `SAAS_ONLY_SLUGS` when promoting new cloud-only providers.
