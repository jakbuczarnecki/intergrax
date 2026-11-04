# ARCH-FREEZE B1 — independent audit handoff dossier

**Disposition target:** verify closed-world production inventory @ exact SHA without Cursor narrative.

| Field | Value |
| --- | --- |
| AUDIT_BASELINE_SHA | `a0624e28ff77b061e83f3e10aab4c2a9083ee1b4` |
| EXECUTION_SHA (certifying target) | `a0624e28ff77b061e83f3e10aab4c2a9083ee1b4` @ HEAD |
| INVENTORY_IMPLEMENTATION_SHA | `git rev-parse HEAD:scripts/governance/arch_freeze_b1_closed_world_inventory.py` |
| Historical wave-1 EXECUTION_SHA | `18a0f68fc6c8107e6344fdb4641ecec1945dae5e` (immutable) |
| Historical planning lock | `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a` (superseded by baseline above) |

## Reproduce certifying inventory (T-B1-STRICT-QUALIFY)

```bash
git checkout a0624e28ff77b061e83f3e10aab4c2a9083ee1b4
uv run pytest tests/unit/governance/test_arch_freeze_b1_closed_world_inventory.py -m "not slow"
uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py \
  --strict-certify \
  --output-dir .tmp/session/arch-freeze-b1-cert/
```

Report-only (non-certifying):

```bash
uv run python scripts/governance/arch_freeze_b1_closed_world_inventory.py \
  --report-only \
  --output-dir .tmp/session/arch-freeze-b1-cert/
```

Mandatory dispositions: [`arch_freeze_b1_mandatory_dispositions.json`](./arch_freeze_b1_mandatory_dispositions.json)

**Not claimed:** ARCH-FREEZE CLOSED, independent B1 sign-off, L3 certifying PASS.

## Traceability

```bash
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
```
