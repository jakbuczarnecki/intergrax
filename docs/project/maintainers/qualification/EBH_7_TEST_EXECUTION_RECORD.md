# EBH-7 — Test execution record (corrective wave)

| Field | Value |
| --- | --- |
| Baseline audited SHA | `59d0fe5d7faf271926f8477de102e7f26b3e1caf` |
| Corrective execution SHA | *recorded at commit* |
| Command | `uv run pytest tests/qualification/ebh_7/ -q` |
| Result | **61 passed, 3 failed, 1 skipped** (expected OPEN behavioral gates) |
| Log | `.tmp/session/ebh-7-corrective-wave/pytest.log` |

## Expected failures (honest OPEN gates)

| Test | Blocker |
| --- | --- |
| `test_ebh7_w5_rpl_h_behavioral_all_qualified` | 51× RPL-H NOT YET @ HEAD |
| `test_ebh7_w6_tenant_e2e_all_mandatory_scenarios_qualified` | 12× tenant E2E without proof nodes |
| `test_ebh7_w7_all_mandatory_scenarios_qualified` | 16× cross-domain without proof nodes |

## L1 / L2 / L3 interpretation

| Layer | Scope | Result |
| --- | --- | --- |
| L1 | SSOT integrity, structural registries, NPSC gate | PASS |
| L2 | Executed registered proof callables (parametrized) | PASS for mapped proofs |
| L3 | All mandatory E2E/XDM + RPL-H qualified | **FAIL** (blockers above) |
