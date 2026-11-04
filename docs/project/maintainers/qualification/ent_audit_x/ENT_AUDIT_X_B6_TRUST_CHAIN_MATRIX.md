# ENT-AUDIT-X — B6 trust chain matrix (SSOT)

**Status:** **DRAFT — NOT SIGNED — NOT ACCEPTED**  
**Purpose:** Separate published technical evidence, independent technical recommendation, authorized attestation, operator approval, and final closure.  
**Authority:** [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) · [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md)

| Link | Role | SHA / artifact | Signed? | Notes |
| --- | --- | --- | --- | --- |
| L0 | W8 corrective implementation (code) | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` (= current TES) | N/A | Audited implementation @ certifying qual |
| L1 | TEST_EXECUTION_SHA | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` | N/A | Certifying qual_wave PASS @ [qualification_at_89c76eb7…](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/) |
| L2 | EVIDENCE_PUBLICATION_SHA (B5) | `300859537f12a1de7bf3d75fd685bc3a5c727183` | N/A | Strict `verify_b5_publication_tree()` @ HEAD |
| L3 | FROZEN ENT AUDIT HEAD (nightly FULL CI) | `618e234ee4dc41aa3673de5e1e81271f6039ca6c` | N/A | Run `41800553851` success — W8-004 dual-pin continuity |
| L4 | INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` | **No** | Regression gate `41804655934` success; [technical re-audit draft](./ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md) = recommendation only |
| L5 | B6_ACCEPTANCE_PUBLICATION_SHA | *unset* | **Required** | Authorized independent auditor Protocol 03 PASS record |
| L6 | ACCEPTED_BASELINE_SHA | *unset* | **Required** | Operator **OD-ENT-01** after L5 |
| L7 | FORMAL_CLOSURE_PUBLICATION_SHA | *unset* | **Required** | **OD-ENT-08** + roadmap §5 + FRZ-AUD promotion |

**Historical defect chain (immutable):** R6/R7 TES `87203c1b…` + EVID `d18f01f22` — [R7_DEFECT_ENT-W8-B6-001.md](./b6_independent_handoff/R7_DEFECT_ENT-W8-B6-001.md).

**Agent boundary:** Cursor may publish L0–L4 preparatory records only. L5–L7 require independent auditor and/or operator authority.
