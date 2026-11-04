# ARCH-FREEZE B1 — independent audit handoff dossier

**Disposition target:** verify closed-world production inventory @ exact SHA without Cursor narrative.

| Field | Value |
| --- | --- |
| AUDIT_BASELINE_SHA | `a0624e28ff77b061e83f3e10aab4c2a9083ee1b4` |
| EXECUTION_SHA (certifying target) | `6bdf78c626bfcb347ad3ad9e3e18b65222f1b169` |
| INVENTORY_IMPLEMENTATION_SHA | `git rev-parse HEAD:scripts/governance/arch_freeze_b1_closed_world_inventory.py` |
| Historical wave-1 EXECUTION_SHA | `18a0f68fc6c8107e6344fdb4641ecec1945dae5e` (immutable) |
| Historical planning lock | `967ef2f30c9c263fb9bb40d1b0fe77c98fc3919a` (superseded by baseline above) |

## Certifying inventory run (T-B1-STRICT-QUALIFY @ EXECUTION_SHA)

| Field | Value |
| --- | --- |
| execution_sha | `6bdf78c626bfcb347ad3ad9e3e18b65222f1b169` |
| strict_pass | `true` |
| cw_gates | all PASS |
| inventory_digest | `0ffa2676b230a6e5ec3e429608d9b63a6ada38393154762870bad70f64c75d18` |
| production_effective | `12` |
| eligible_unverified | `8340` |
| wall_time | ~6 min |
| exit_code | `0` |

Evidence output dir (local, gitignored): `.tmp/session/arch-freeze-b1-cert/`

## Reproduce certifying inventory (T-B1-STRICT-QUALIFY)

```bash
git checkout 6bdf78c626bfcb347ad3ad9e3e18b65222f1b169
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

**Not claimed:** ARCH-FREEZE CLOSED, independent B1 sign-off, L3 certifying PASS, FRZ-CMP promotion.

**Claimed (implementation wave only):** B1 strict-certify replay @ EXECUTION_SHA reproduces `strict_pass=true` and digest above; handoff package refreshed for external auditor replay.

## Traceability

```bash
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
```
