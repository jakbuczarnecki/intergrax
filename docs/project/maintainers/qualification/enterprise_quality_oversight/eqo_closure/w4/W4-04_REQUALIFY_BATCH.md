# EQO-W4-04 — REQUALIFY Batch (Full B-2 Queue Re-verified on `development` @ `646e63953`)

- **Parent:** `EQO-W4-04` · **Published under:** `governance/tickets/T-IGX-090-B.md` (`OP-EC-01`, `2026-11-26T17:26:01+01:00`, `OD-W2-DOCS=A`)
- **Task:** Link the five `REQUALIFY` disposition rows in `EQO_B2_DISPOSITION.jsonl` to updated Campaign C layer evidence (`EQO_C_DRIFT.jsonl` and `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` v2) and ARCH-FREEZE `PT-Q*` nodes.
- **Scope Guard:** Planning & disposition memo only — **no** product requalification executed, **no** `register.jsonl` closure, **no** FRZ promotion, **no** `EQO_CLOSED`, and **no** G6/IIAC PASS claimed.

## Pins (`AD-AF-DEVTRAIN` / `D-1`)

| Pin | SHA | Role |
| --- | --- | --- |
| `ACTIVE_TRAIN_PIN` / `DOC_PIN` | `646e63953ae1a7eb28a229322590c9cdcbb96625` | Single consolidated `development` analysis baseline (`32005c212` branch base) |
| `A_BASELINE` | `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` | EQO Campaign A baseline |
| Historical `DOC_PIN` (superseded) | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` | Prior 2026-11-21 W4-04 pin |
| Historical `TRAIN_PIN` (retired) | `28c27cb896d02e0185b9fa02047411fa152f6469` | Prior 2026-11-21 `rw-train` pin |

> **Key HEAD-Drift Finding (`DOC_PIN` `8cee3c5a9` -> `HEAD` `646e63953`):** Prior v1 `EQO_C_DRIFT.jsonl` recorded `TENANT-X` as mechanical `NO_PRODUCT_DELTA` (`drift_class=MISSING_SURFACE`). Between `DOC_PIN` and `646e63953`, **12 commits** modified `TENANT-X` code/tests (`4d339bc21`, `da5ca5b53`, `f3106110a`, `17ee001ee`, `5ab3d0c18`) and **15+ commits** modified `intergrax/platform/enterprise_audit/` (`5b100965f`, `b843b3aec`, `a59aa9075`, `eabf601f2`, `8371c57cc`, `34cdd2f66`, `6f44c23ff`, `1b89bec00`). Consequently, all 5 rows require freeze-SHA `PT-Q*` execution and remain **`BLOCKED_ON_PT_Q`** (EQO does not execute `PT-Q*` jobs).

## Batch Queue (`EQO_B2_DISPOSITION.jsonl` -> `EQO-W4-04`)

| Finding ID | Severity | Owner | `EQO_C_DRIFT.jsonl` / Matrix Line(s) | Drift @ Historical Pin -> **`HEAD = 646e63953`** | Linked ARCH-FREEZE `PT-Q*` Node(s) | Final `EQO-W4-04` Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| **`EQOA-A0-008`** | `MEDIUM` | `PDC-X` (`T-IGX-041-J` EBH-7 re-pin) | `L17` (`PDC-X`, `TARGETED_REPLAY`) + `L12` (`EBH-5/6/7`, `FULL_REQUAL_REQUIRED`) | `FULL_REQUAL_REQUIRED` -> **`FULL_REQUAL_REQUIRED`** (EBH-7 L3 replay `43738431866` failed; signed `OD-PDC-X-CLOSE.md` Decision 7 explicitly acknowledges B6 not passed & identity gate known-red) | `PT-Q06`, `PT-Q11`, `PT-Q12`, `T-AF2-REQUAL-HIST-PASS` | **`BLOCKED_ON_PT_Q`** — blocked until freeze-SHA EBH-7/PDC-X replay & independent audit execute at `G2`/`M3`. |
| **`EQOA-A2-035`** | `MEDIUM` | `PDC-X` | `L17` (`PDC-X`, `TARGETED_REPLAY`) + `L16` (`MCP-SERVER-X`, `FULL_REQUAL_REQUIRED`) + `L15` (`IIAC`, `FULL_REQUAL_REQUIRED`) | `FULL_REQUAL_REQUIRED` -> **`FULL_REQUAL_REQUIRED`** (`mcp_identity_mode.py` unchanged since `DOC_PIN`; PDC-X P8 activation `PASS` scoped carries 19 limitations in `PDC_DOMAIN_CERTIFICATION_MATRIX.md`) | `PT-Q02`, `PT-Q07`, `PT-Q11`, `T-AF2-REQUAL-HIST-PASS` | **`BLOCKED_ON_PT_Q`** — blocked until `PT-Q02` / `PT-Q07` / `PT-Q11` freeze-SHA replay & independent audit run. |
| **`EQOA-A3-013`** | `MEDIUM` | `TENANT-X` / `ARCH-FREEZE` | `L4` (`TENANT-X`, `FULL_REQUAL_REQUIRED`; `surface_blocker: EQOA-A3-012`) | `NO_PRODUCT_DELTA` (mechanical) **OVERTAKEN** -> **`FULL_REQUAL_REQUIRED`** (12 post-`DOC_PIN` commits touched `TENANT-X` code/carriers: `4d339bc21`, `da5ca5b53`, `f3106110a`, `17ee001ee`, `5ab3d0c18`; `T-AF2-RW-TENANT` open) | `PT-Q04`, `PT-Q07`, `T-AF2-RW-TENANT`, `T-B7-TEN-REVAL` | **`BLOCKED_ON_PT_Q`** — blocked until `T-AF2-RW-TENANT` (`WL-K1-C-05`) closes and independent `T-B7-TEN-REVAL` / `PT-Q04` / `PT-Q07` runs at freeze SHA. |
| **`EQOA-A3-030`** | `MEDIUM` | `ARCH-FREEZE` | `L14` (`ENT-AUDIT-X`, `FULL_REQUAL_REQUIRED`) + `L6` (`GOV-X`, `FULL_REQUAL_REQUIRED`) + `L15` (`IIAC`, `FULL_REQUAL_REQUIRED`) | `FULL_REQUAL_REQUIRED` -> **`FULL_REQUAL_REQUIRED` (widened)** (15+ post-`DOC_PIN` commits modified `intergrax/platform/enterprise_audit/`: `5b100965f`, `b843b3aec`, `a59aa9075`, `eabf601f2`, `34cdd2f66`, `1b89bec00`; TES SHAs remain historical) | `PT-S01..S04`, `PT-Q02`, `PT-Q13`, `T-AF2-REQUAL-HIST-PASS` | **`BLOCKED_ON_PT_Q`** — blocked until certifying FREEZE audit (`PT-S01..S04` / `PT-Q13`), EVOL graph re-measurement, and IIAC re-pin execute at `G2`/`M3`. |
| **`EQOA-A5-011`** | `MEDIUM` | `TENANT-X` / `ARCH-FREEZE` | `L4` (`TENANT-X`, `FULL_REQUAL_REQUIRED`; `surface_blocker: EQOA-A3-012`) | `NO_PRODUCT_DELTA` **OVERTAKEN** -> **`FULL_REQUAL_REQUIRED`** (`PT-WF` `T-IGX-043-J` published G-09 fix `e6b1d97f7` via PR #718 while E2–E4 remain open; 12 post-`DOC_PIN` TENANT-X commits supersede `5ccb72fc` delta pin) | `PT-WF`, `PT-Q04`, `PT-Q07` | **`BLOCKED_ON_PT_Q`** — blocked until `PT-WF` E2–E4 completes and `PT-Q07` replaces historical `5ccb72fc` delta gate with freeze-SHA run. |

## Rollup & Steward Action

- **EQO-W4-04 rollup:** **5 / 5** rows dispositioned as **`BLOCKED_ON_PT_Q`** with exact `EQO_C_DRIFT.jsonl` / `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` v2 line citations, post-`DOC_PIN` HEAD delta evidence (`646e63953`), and ARCH-FREEZE `PT-Q*` node ownership.
- **Zero EQO self-execution** of `PT-Q*` jobs; programme `EQO_CLOSED` is **not asserted**.

**Recorded:** 2026-11-26 · `Ref: governance/tickets/T-IGX-090-B.md` (supersedes 2026-11-21 memo under `T-IGX-046-F.md`)
