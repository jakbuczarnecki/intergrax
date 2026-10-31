# EBH-7 — ECAP current HEAD audit report (C1–C5)

**Remediation program @ HEAD:** normative RPL/TEN/XDM appendix — [`EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md`](EBH_7_FINAL_AUDIT_MATRICES_APPENDIX.md).

| ECAP | Status @ remediation | Executable gate |
| --- | --- | --- |
| ECAP-C1 | CHECKLIST — mechanism registry | `test_ebh_7_mechanical_exit_gates.py` |
| ECAP-C2 | CHECKLIST — dup subset scoped | `test_ebh_7_w4_gov_dup_gates.py` |
| ECAP-C3 | **PROHIBITED unresolved 0** @ EBH-6 register scope | `test_ebh6_prohibited_unresolved_zero` |
| ECAP-C4 | **Independent audit — TEN rows in appendix** | `test_ebh_7_audit_remediation_gates.py` |
| ECAP-C5 | **Independent audit — XDM composed-proxy inventory** | appendix + W7 registry |

**RPL-H @ HEAD (normative):** see appendix — **not** 56× demonstrated replaceability.

**Agent mechanical counters:** may show 60/60 EXACT — **B6 must use appendix + normative RPL**.

**B6 / OD-EBH7-08:** **OPEN**. **TEST_EXECUTION_SHA:** pending Wave 5 replay ([`EBH_7_SHA_LADDER.md`](EBH_7_SHA_LADDER.md)).

**Non-claims:** EBH-7 CLOSED, B6 PASS, global FRZ PASS, EVOL-X entry.
