# EBH-6 — Protocol 03 Independent Audit Pass Record

| Field | Value |
| --- | --- |
| **Record type** | ECAP §2.B B6 — Independent exact-SHA acceptance |
| **Status** | **NOT ISSUED** — auditor / operator only |
| **Verdict** | **PENDING** |

## Required fields (complete on PASS only)

| Field | Value |
| --- | --- |
| **ACCEPTANCE_SHA** | _TBD_ |
| **TEST_EXECUTION_SHA** | _TBD_ |
| **Auditor identity** | _TBD_ |
| **Date** | _TBD_ |
| **Disposition** | PASS / REJECT |

## On REJECT

Enumerate findings with task IDs → E04 remediation → E09 re-qual → re-audit.

**Agent must not populate PASS fields.**
