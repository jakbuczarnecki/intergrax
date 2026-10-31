# EBH-7 — Pre-cert independent audit worksheet (Protocol 03 B2)

| Field | Value |
| --- | --- |
| Stage | EBH-7 |
| Agent readiness | Remediation program materialized — **not** B6 PASS |
| B6 | **OPEN** — external auditor |
| OD-EBH7-08 | **OPEN** |

## Auditor checklist

1. Verify `EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md` regenerated @ audited SHA.
2. Confirm RPL normative rows — no factory-only QUALIFIED without OD scope.
3. Sample TEN/XDM EXACT scenarios — reject composed test-import proxies.
4. Replay L3 command from `EBH_7_ECAP_EXECUTABLE_CHECKLIST.md` @ `TEST_EXECUTION_SHA`.
5. Verify `ebh_7_p03_independent_replay/MANIFEST.json` hashes.
6. Record PASS/FAIL in `EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` (create on B6 PASS only).

## Corrective loop

FAIL → register finding in `tests/qualification/ebh_7/_gate_findings_register.py` → remediation waves 1–5 → re-audit.
