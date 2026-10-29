# ROADMAP-REPLAY-X — FRZ PASS impact matrix (95 rows, grouped)

| Field | Value |
| --- | --- |
| **AUDITED_HEAD** | `a2ed11f0e811446ae67fdfd57b4c4997c546e10f` |
| **Prod delta** since DUP-X (`a7d83925…`) | **empty** on `intergrax/`, `agents/`, `applications/` |

**Rule:** checklist **PASS** preserved; replay supplies **IMPACT_ASSESS_L2** only — no FRZ row promotion.

| Impact group | FRZ IDs | Primary owner | Evidence @ HEAD | L2 disposition | Final @ immutable SHA |
| --- | --- | --- | --- | --- | --- |
| CONFIG-X | FRZ-CFG-01..08 | CONFIG-X | prod Δ none; `tests/qualification/config_x/` | reuse + VBN delta cross-check | **PENDING** post-commit pytest |
| COMPAT-X | FRZ-CMP-01..08 | COMPAT-X | prod Δ none | targeted compat_x replay | **PENDING** |
| CTRL-X | FRZ-CTL-01..12 | CTRL-X | prod Δ none; proof replay 44 historical | ctrl_x qual replay | **PENDING** |
| EXE / GOV | FRZ-EXE-01..07, FRZ-GOV-01..10 | GOV-X2 | prod Δ none | gov_x2 + governance qual | **PENDING** |
| TRACE-X | FRZ-TRC-01..12 | TRACE-X | prod Δ none | trace_x qual replay | **PENDING** |
| STATE / REC | FRZ-STA-01..08, FRZ-REC-01..10 | STATE-X | prod Δ none | state_x qual replay | **PENDING** |
| TENANT-X | FRZ-TEN-01..12 | TENANT-X | prod Δ none | tenant_x FINAL matrix | **PENDING** |
| HARNESS-FINAL | FRZ-HRN-01..08 | HARNESS-FINAL | prod Δ none | harness final attestation inspect | **PENDING** |

**OPEN 130 rows:** obligation = parent-stage or Path B per [`ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md`](ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md); **not** automatic REPLAY-X blockers unless BLOCKER register row exists.
