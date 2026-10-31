# EBH-7 — B6 corrective loop (Wave 7)

| Step | Action |
| --- | --- |
| 1 | Independent auditor runs [`EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md`](EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) @ `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` |
| 2 | FAIL → register finding in `tests/qualification/ebh_7/_gate_findings_register.py` |
| 3 | Map finding to Package B–F task id (EBH7-FC-*) |
| 4 | Re-run L3 @ new `TEST_EXECUTION_SHA` |
| 5 | PASS → update [`EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) (external signature) |

**Agent non-claims:** B6 PASS, OD-EBH7-08 approval.
