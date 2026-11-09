# Historical campaign — blocker dossier (2026-11-09 final batch)

| Field | Value |
|---|---|
| Campaign label | **`HISTORICAL_CAMPAIGN_PARTIAL_EXACT_BLOCKERS`** |
| PUBLICATION_SHA | `087bd17693db5b4b8a14cb618389ee1d6c2aaa3d` (`origin/development` target) |
| Purpose | Per operator directive §2: classify each active blocker — HIST in/out, prod impact, disposition path, owner, release gate. **No** acceptance-criteria weakening. |

## Summary table

| Blocker ID | In HIST scope? | Blocks HIST terminal? | Blocks prod / `development`? | HIST disposition | Prep done (HIST) | Owner | Gate to release |
|---|---|---|---|---|---|---|---|
| **IIAC-ACTIVATION** (historical) | Yes (P5) | No — **DONE** | Was blocking merge; now IN_FORCE | **CLOSED_SCOPED** @ P5 `8045915a` | Re-audit package `HIST-P5-2026-11-09` | Operator (signed activation) | `verify_iiac_policy_activation --require-in-force` green |
| **ENT-REACTIVATION-CONSENT** | Partial (P2 only) | Yes — prevents `CLOSED_VERIFIED` | No — HIST proceeded under Operator P2 supersession | **defer / external** | Agreement published; [`DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md`](delegations/DELEG-HIST-IIAC-P2-SUPERSESSION-2026-11-09.md) | ENT-AUDIT-X owner | ENT counter-sign or Operator supersession record |
| **RPL-01..04** | No (read-only map) | Yes | No direct prod block | **N/A WITH EVIDENCE** — matrix unchanged OPEN | Predecessor handoff + P7 notes | ARCH-FREEZE `T-B7-VBN-INDEPENDENT-ACCEPT` | External independent acceptance |
| **REG Population E** | No | Yes | No | **defer** | REG operator split recorded (DR / HIST-01 handoff) | ENT-AUDIT-X rollup | Population E qualification + audit |
| **Path B (PRD-01/02 live)** | No | Yes | No | **OPEN** per register | [`HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md`](HISTORICAL_FINAL_PATH_B_RESIDUAL_REGISTER.md) | PROD-Q / Operator | Live qualification; not HIST closure |
| **HIST-F-BND-03B** | Yes (track only) | No (expected OPEN residual) | No — known tier edge | **TRACKED** — see residual register | C4 PASS_SCOPED + matrix `p7_residual_ids` | Future ticket | Product fix + replay |
| **HIST-F-DUR-01** | Yes (track only) | No | No | **TRACKED** — defer | C3 PASS; durability out of 032-A | Future ticket | Durability semantics ticket |
| **C4 CLI identity** | Yes (boundary) | No | No — scoped test red | **PASS_SCOPED** evidence | P4 log + aud-product | Env/fixture owner (optional) | Not required for HIST partial close |
| **P1-SIGN (030-B/C, 032-A, 033-A)** | Yes | Was — **resolved** | No | **CLOSED_SCOPED** | `hist-sign-plans` + verify_plan_approval | HIST (done) | P3 executed |
| **P8 global `CLOSED_VERIFIED`** | Yes (verdict only) | N/A | No | **WITHHELD** — §J unmet | P8 audit PASS + certifier ACCEPT partial | Operator / external streams | ENT + ARCH + Path B |

## Per-blocker notes

### ENT-REACTIVATION-CONSENT

- **Out of HIST implementation**; HIST documented `PROPOSED_NO_CONSENT` and Operator-only supersession for sequencing.
- Does **not** block daily `development` after P5 activation.
- Blocks only the **legal** campaign label `HISTORICAL_CAMPAIGN_CLOSED_VERIFIED` (master plan §J).

### RPL-01..04

- WBS **BLOCKED_EXTERNAL** before HIST campaign; HIST P7 explicitly did not run ARCH-FREEZE acceptance.
- ARCH-FREEZE handoff: [`ARCH_FREEZE_HIST_CAMPAIGN_HANDOFF.md`](ARCH_FREEZE_HIST_CAMPAIGN_HANDOFF.md).

### Path B

- Rows remain OPEN by design; HIST P7 did not promote live obligations.

### Product residuals (HIST-F-*)

- Documented in [`HISTORICAL_RESIDUAL_FINDINGS_REGISTER.md`](HISTORICAL_RESIDUAL_FINDINGS_REGISTER.md).

## HIST executable obligation status

All **mandated HIST execution** items through P8 certifier closeout are **done** on the SHA ladder ending `087bd1769`. Remaining rows in this dossier are **external**, **tracked residuals**, or **verdict ceiling** — not unassigned in-scope HIST debt.
