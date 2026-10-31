# EBH-7 — L3 replay record

| Field | Value |
| --- | --- |
| TEST_EXECUTION_SHA | `f8c54147bd42b3b8d9032b777358fa24c0a63b01` |
| Profile | [`EBH_7_ECAP_EXECUTABLE_CHECKLIST.md`](../EBH_7_ECAP_EXECUTABLE_CHECKLIST.md) L3 block + full `tests/qualification/ebh_7/` |
| Result | **250 passed**, **8 skipped**, **1 xfailed** (`test_ebh7_remediation_rpl_normative_b6_readiness` — expected) |
| ECAP consolidated profile | **129 passed** @ same TEST_EXECUTION_SHA |
| Supplementary log | `.tmp/session/ebh-7-b6-corrective/l3-r2.log` |

RPL B6 blockers (semantic not QUALIFIED): **51** (6 BLOCKED, 45 PARTIAL including 28 dual-slug). See [`EBH_7_B6_BLOCKER_REGISTER.md`](../EBH_7_B6_BLOCKER_REGISTER.md).
