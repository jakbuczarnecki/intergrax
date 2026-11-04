# ENT-AUDIT-X — W8 Protocol 03 B6 findings disposition

**Wave:** independent W8 B6 corrective implementation  
**TEST_EXECUTION_SHA:** `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91`  
**EVIDENCE_PUBLICATION_SHA:** `300859537f12a1de7bf3d75fd685bc3a5c727183`  
**Implementation commits (core):** `cc3017991…` (strict B5 + readiness), `89c76eb7b…` (tests), `300859537f12a1de7bf3d75fd685bc3a5c727183` (corrective B5 tree)

| Finding | Disposition | Evidence |
| --- | --- | --- |
| ENT-W8-B6-001 publication digest | **FIXED_WITH_EVIDENCE** | R7 tree preserved @ `d18f01f22`; byte mismatch reproduced (`4a1107f0…` ≠ `b914bd8b…`); corrective B5 @ `qualification_at_89c76eb7…`; `verify_b5_publication_tree()` → `()` |
| ENT-W8-B6-002 incomplete B5 verifier | **FIXED_WITH_EVIDENCE** | `intergrax/platform/enterprise_audit/validators.py::verify_b5_publication_tree`; `tests/qualification/ent_audit_x/unit/test_b5_publication_verifier.py` |
| ENT-W8-B6-003 readiness false PASS | **FIXED_WITH_EVIDENCE** | `mechanical_readiness_check.py` evidence_integrity → strict B5 verify; corrupt digest → FAIL; repaired tree → PASS |
| ENT-W8-B6-004 current-SHA CI | **ACCEPTED (dual-pin)** / **EXTERNAL_BLOCKED** (required checks @ `development`) | FAST + regression @ `0e538269…` (runs `41804655934`, `41798742568`); nightly FULL @ `618e234…` run `41800553851` success — branch protection still OD-ENT-02 + admin |
| ENT-W8-B6-005 branch protection | **EXTERNAL_BLOCKED** | OD-ENT-02 + GitHub admin; workflows exist |
| ENT-W8-B6-006 stale handoff README | **FIXED_WITH_EVIDENCE** | `b6_independent_handoff/README.md` pins current TES/EVID + R7 defect link |

## ENT-W8-B6-001 root cause (R7)

Publication script @ `d18f01f22` wrote `publication_index.v1.json` with `qual_wave_report.sha256` from on-disk bytes at index time, but the committed `qual_wave_report.json` in the same publication commit no longer matched that digest (non-atomic post-index mutation or index built against a different qual_wave source than the committed file). **Fix:** hash-after-final-write + `verify_b5_publication_tree()` gate in `publish_b5_evidence_bundle.py`; new immutable B5 @ `300859537…`; R7 commit preserved as defect record.

## Related gaps (not downgraded)

| Topic | Disposition |
| --- | --- |
| Mode B sandbox / integration evidence | **VERIFIED_NO_DEFECT** @ TES via `test_sandbox_isolation` + B5 outcomes; no new vendor integration |
| Windows launcher qual scope | **PARTIAL** — GAP-RC-LAUNCHER-* remain PARTIAL; BAT/SH non-certifying FAST only |
| Operator forge/replay tests | **EXTERNAL_BLOCKED** — OD-ENT-01 operator_records |
| Matrix C 75/75 accounting | **FIXED_WITH_EVIDENCE** — C075-S2 reconcile @ `wbs_reconciliation_lib.py` + master plan Matrix C/D |

**Agent boundary:** no B6 PASS, no OD, no FRZ CLOSED, no accepted baseline promotion.
