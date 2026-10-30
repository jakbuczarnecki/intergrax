# EBH-7 — Test execution record (corrective wave)

| Field | Value |
| --- | --- |
| Baseline audited SHA | `e2467a88ed042e2d97150568f9f6b25ba85cb6b1` |
| Completion-wave execution | Working tree @ baseline + uncommitted completion-wave deltas |
| Command | `uv run pytest tests/qualification/ebh_7/ -q` |
| Result | **67 passed, 3 failed, 3 skipped** (expected OPEN behavioral gates) |
| Log | `.tmp/session/ebh-7-behavioral-wave/pytest-ebh7-full.log` |

## Expected failures (honest OPEN gates)

| Test | Blocker |
| --- | --- |
| `test_ebh7_w5_rpl_h_behavioral_all_qualified` | 51× RPL-H NOT YET @ HEAD |
| `test_ebh7_w6_tenant_e2e_all_mandatory_scenarios_qualified` | 12× tenant E2E without proof nodes |
| `test_ebh7_w7_all_mandatory_scenarios_qualified` | 18× cross-domain OPEN per semantic audit |

## L1 / L2 / L3 interpretation

| Layer | Scope | Result |
| --- | --- | --- |
| L1 | SSOT integrity, structural registries, NPSC gate | PASS |
| L2 | Executed registered proof callables (parametrized) | PASS for mapped proofs |
| L3 | All mandatory E2E/XDM + RPL-H qualified | **FAIL** (blockers above) |
