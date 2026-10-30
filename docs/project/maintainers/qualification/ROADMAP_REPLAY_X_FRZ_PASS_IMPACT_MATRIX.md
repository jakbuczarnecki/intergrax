# ROADMAP-REPLAY-X — FRZ PASS impact matrix (95 rows, grouped)

| Field | Value |
| --- | --- |
| **IMPLEMENTATION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **TEST_EXECUTION_SHA** | `22fbc0e837e819fe2a068295df5f527612dcf307` |
| **Prod delta** since DUP-X | **scoped (1 path)** · unscoped **empty** | checklist **PASS** preserved for unrelated paths; replay supplies **IMPACT_ASSESS_L2** only — no FRZ row promotion.

| Impact group | FRZ IDs | Primary owner | Evidence @ HEAD | L2 disposition | Final @ immutable SHA |
| --- | --- | --- | --- | --- | --- |
| CONFIG-X | FRZ-CFG-01..08 | CONFIG-X | prod Δ none; `tests/qualification/config_x/` | reuse + VBN delta cross-check | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-config_x.log` |
| COMPAT-X | FRZ-CMP-01..08 | COMPAT-X | prod Δ none | targeted compat_x replay | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-compat_x.log` |
| CTRL-X | FRZ-CTL-01..12 | CTRL-X | scoped prod Δ `dependency_attempt_execution_boundary.py`; L2 ctrl_x PASS @ `22fbc0e` | ctrl_x qual replay | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-ctrl_x.log` |
| EXE / GOV | FRZ-EXE-01..07, FRZ-GOV-01..10 | GOV-X2 | prod Δ none | gov_x2 + governance qual | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-gov_x2.log` |
| TRACE-X | FRZ-TRC-01..12 | TRACE-X | prod Δ none | trace_x qual replay | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-trace_x.log` |
| STATE / REC | FRZ-STA-01..08, FRZ-REC-01..10 | STATE-X | prod Δ none | state_x qual replay | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-state_x.log` |
| TENANT-X | FRZ-TEN-01..12 | TENANT-X | prod Δ none | tenant_x FINAL matrix | **IMPACT_ASSESS_L2 PASS** @ `22fbc0e` · `l2-tenant_x.log` |
| HARNESS-FINAL | FRZ-HRN-01..08 | HARNESS-FINAL | prod Δ none | harness final attestation inspect | **reuse** (prod Δ none; DUP-X attestation) |

**OPEN 130 rows:** obligation = parent-stage or Path B per [`ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md`](ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md); **not** automatic REPLAY-X blockers unless BLOCKER register row exists.
