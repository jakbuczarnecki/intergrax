# ACT-N11 MCP P6 — Operator activation dossier

**Status:** CERTIFIER_PREFLIGHT_PASS — **STOP** before personal IIIC activation.

| Gate | State |
| --- | --- |
| G6 evidence | **PUBLISHED** @ `2d3eceb6c` |
| IIIC evidence | **PUBLISHED** @ `2d3eceb6c` |
| Certifier | **PASS** |
| Operator OD | **REQUESTED** |
| Activation record | **UNCHANGED** (`8fff85e3b` pin) |
| `governance-fast` | **Expected FAIL** until activation |

## Operator personal line (required)

```text
Approved IIIC policy activation IIAC-2026-11-05.1 (MCP ACT-N11 P6 rebind): operator @ ________________
```

## Post-activation verification (after operator line only)

```text
.venv/Scripts/python.exe -m scripts.governance.verify_iiac_policy_activation --require-in-force
```

Expected: exit **0** on `development` HEAD after phase-2 record update.
