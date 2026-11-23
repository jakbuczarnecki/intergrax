# OD-SFX-R2-CANONICAL-ADOPTION — R2 overlay adoption (Option B)

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-R2-CANONICAL-ADOPTION (maps to proposed OD-SFX-R2-02) |
| **Campaign** | IGX-SFX-R2-CANONICAL-ADOPTION-001 |
| **Ticket** | T-IGX-070-C |
| **Signature status** | **PENDING SIGNATURE** |
| **Commission source** | Operator mission IGX-SFX-R2-CANONICAL-ADOPTION-001 (2026-11-23) |

## Decision (pending operator signature)

1. **R2 overlay:** Accept R2 as a **versioned overlay** on R1 in `SFX-MASTER-PLAN.md` §18; R1 remains binding for all non-overlay text.
2. **First technical reference:** REF-1 invoice↔existing PO match in **ASSISTED** mode (HO-2 human submit); no R2-W5-03 driver in first sandbox POC.
3. **HO-2:** Human submit remains binding until independent **OD-SEF-R2** on SEF track (no delegated multi-submit).
4. **ECAP §16:** `C-ECAP-01` remains **OPEN** (blocks R2-W8-03 certification rebind until OD-ECAP-16).
5. **Autonomous driver:** Defer **R2-W5-03** until SEF authorization and OD-SEF-R2.
6. **Gates:** ARCH-FREEZE and GATE-SFX-ENTRY restrictions unchanged; **no implementation authorization**.

## Explicit non-grants

| State | Value |
| --- | --- |
| `IMPLEMENTATION_AUTHORIZED` | **NO** |
| `CANONICAL_R2_APPROVED` | **NO** until this record is signed |
| `ENTERPRISE_CERTIFICATION` | **NOT CLAIMED** |
| SEF `CON-N-01` change | **NOT AUTHORIZED** |
| Production admission | **NOT AUTHORIZED** |

## Evidence anchors (historical, preserved)

| Event | SHA |
| --- | --- |
| PR #756 merge | `67b127f3b4a919c13d5d156ef792f359943cfe3f` |
| PR #757 reviewed head | `e2b2a2b7e72f35f1e6da4dad4d850d278a4f3e5d` |
| PR #757 merge / `development` | `7ddd53266d32ce47f059456e110e414b92248453` |
| Independent audit (corrective) | `e5cd9879e` |
| Independent audit (finalization) | `360cb4887f5045ddf0f8461f038fb593a2b61292` |
| Merge-tree equivalence | Empty `git diff` SFX paths: `e2b2a2b7` → `7ddd532` |

## Enterprise quality (documentation merges)

PR #756–#757 merged while `governance-fast` and `ent-audit-fast` were red on PR #757 HEAD. GitHub ruleset **Protect development** did not require green status checks (deletion + non-fast-forward only). Upstream findings remain owned by IIAC and ENT-AUDIT-X; not retroactively marked PASS.
