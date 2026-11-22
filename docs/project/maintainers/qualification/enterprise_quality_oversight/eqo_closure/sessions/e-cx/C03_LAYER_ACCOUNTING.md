# EQO-C-03 — layer accounting (Session E)

**Source SHA:** `8bdb0038d7a8fe506b5490a0ec2acad227e6b938`  
**Registry:** `EQO_C_DRIFT.jsonl` (18 canonical layer rows)

## Override (Session E)

| `layer_id` | Mechanical diff vs A_BASELINE | `drift_class` | Blocker |
|------------|------------------------------|---------------|---------|
| TENANT-X | `NO_PRODUCT_DELTA` on train/doc pin diffs | **MISSING_SURFACE** | EQOA-A3-012 + WB02 `ENV_PROVEN` nodes (see `x/A3-012/TENANT_X_DRIFT_READ.md`) |

Mechanical zero product delta does **not** close certification-drift surface.

## Row inventory (summary)

| Class | Layers |
|-------|--------|
| `NO_PRODUCT_DELTA` | TOOL-LIFE-X, SKILL-X, DIST-SCALE-X, CONFIG-X, CTRL-X, STATE-X, DUP-X, COMPAT-X, ROADMAP-REPLAY-X |
| `TARGETED_REPLAY` | TRACE-X, MCP-SERVER-X, CODEOWNERS_DELTA |
| `FULL_REQUAL_REQUIRED` | GOV-X, EBH-5/6/7, ENT-AUDIT-X, IIAC, PDC-X |
| `MISSING_SURFACE` | TENANT-X |

No `UNKNOWN` rows. Post-FCS layers in `FULL_REQUAL_REQUIRED` remain blocked for C-05.

**Ref:** `governance/tickets/T-IGX-046-F.md`