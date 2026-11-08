# ENT-AUDIT-X — Protocol 03 B6 formal acceptance package (draft)

> **Status overlay (2026-11-08, `T-IGX-024-A`):** closure authority for ENT-AUDIT-X is now IIAC (`IIAC-2026-11-05.1`, IN_FORCE, activation `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`). Statements below that assume a human Protocol 03 auditor, a named external auditor, or "ECAP-P4 NOT IN FORCE" are historical and **superseded**; this text is preserved unchanged. Current status: [ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md](ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md); this document is kept as the template for the IIAC B6 acceptance record (auditor and certifier roles replace the human auditor signature).

**Status:** **DRAFT — NOT SIGNED — NOT ACCEPTED**  
**Stage:** ENT-AUDIT-X · Protocol 03 · B6  
**Prepared for:** Authorized independent auditor (mandate per ECAP) + operator review boundary  
**Do not treat as:** B6 PASS, OD approval, FRZ CLOSED, ENT stage CLOSED, ARCH-FREEZE

---

## 1. Executive audit summary

W8 corrective wave published certifying qualification @ TEST_EXECUTION_SHA `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` and B5 @ EVIDENCE_PUBLICATION_SHA `300859537f12a1de7bf3d75fd685bc3a5c727183`. Independent **technical** re-audit @ `0e538269a7188075003b9ff2c7cce2545d8f1ccb` recommends formal B6 eligibility. **Formal attestation unset** (`B6_ACCEPTANCE_PUBLICATION_SHA`).

## 2. Stage identity

| Field | Value |
| --- | --- |
| Stage ID | ENT-AUDIT-X |
| Protocol | ECAP Protocol 03 (Mode A + B) |
| Execution SSOT | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) |
| Trust chain | [ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md](./ENT_AUDIT_X_B6_TRUST_CHAIN_MATRIX.md) |

## 3. Canonical authority references

- [PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md](../../plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) — stage order, ENT-AUDIT-X, §5 ledger
- [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) — FRZ-AUD-01..10
- [ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md](../ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) — B1–B6, auditor independence
- [SYSTEM_INVARIANTS.md](../../../technical/guides/SYSTEM_INVARIANTS.md)

## 4. Exact SHA lineage

See trust chain matrix. **Active qualification:** TES `89c76eb7…` → EVID `30085953…` → technical re-audit `0e538269…` → *formal B6 unset*.

## 5. Technical re-audit report

[ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md](./ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md) — recommendation only.

## 6. Mode A results

- B5 `publication_index.v1.json` digest chain @ `qualification_at_89c76eb7…`
- `mechanical_readiness_check.py` → `evidence_integrity` OK @ HEAD
- Artifact digests in `l3-cli-full/` tied to qual_wave @ TES

## 7. Mode B results

- Regression gate @ `0e538269…` — run `41804655934` success
- ENT nightly FULL @ `618e234…` — run `41800553851` success (W8-004 dual-pin)
- PR/FAST historical runs in [ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md](./ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md) CI table

## 8. W8 findings disposition

[ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md](./ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md) — 001–003 fixed; 004 dual-pin accepted technically; 005 external; 006 fixed; 007 doc drift closed @ publication.

## 9. B5 digest verification

- Verifier: `intergrax/platform/enterprise_audit/validators.py::verify_b5_publication_tree`
- Tests: `tests/qualification/ent_audit_x/unit/test_b5_publication_verifier.py`
- R7 defect preserved: [R7_DEFECT_ENT-W8-B6-001.md](./b6_independent_handoff/R7_DEFECT_ENT-W8-B6-001.md)

## 10. CI proof

Dossier CI tables + run IDs above. Auditor must re-verify job conclusions in GitHub Actions (not agent assertions alone).

## 11. Qualification evidence

| Artifact | Path |
| --- | --- |
| qual_wave | [qual_wave_report.json](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/qual_wave_report.json) |
| FULL report | [enterprise_audit_report.json](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/l3-cli-full/enterprise_audit_report.json) |
| Outcomes 85/85 | [enterprise_audit_outcomes.json](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/l3-cli-full/enterprise_audit_outcomes.json) |

## 12. Matrix C reconciliation

Master plan Matrix C @ TES; C075-S2 WBS reconcile; mechanical `validate_post_implementation_master_plan.py` + WBS gates.

## 13. GAP-RC / GAP-B register

- GAP-RC: [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md) — critical rows CLOSED @ W8 B5
- GAP-B: dossier external blockers — EVOL, CI enforcement, ENT B6, roadmap dependency

## 14. Remaining risks

- Branch protection not enforced (W8-005 / OD-ENT-02)
- EVOL formal B6 + FRZ-EVOL open — **BLK-ROADMAP-ENT-DEP**
- Launcher scope PARTIAL (non-blocking historical debt per disposition)
- Operator forge/replay tests EXTERNAL_BLOCKED

## 15. Scope boundaries

In scope: ENT-AUDIT-X runner, qual harness, workflows, B5 custody. Out of scope: EVOL implementation re-run, production config, Docker migration, agent signing OD/FRZ.

## 16. EVOL predecessor dependencies

ENT formal closure requires operator reconciliation of EVOL formal status — see [ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md](../ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md). **OD-ENT-00** gates **BLK-ROADMAP-ENT-DEP**.

## 17. Formal auditor mandate requirements

- Independence from implementer (ECAP)
- Verify evidence @ TES/B5/re-audit SHA independently
- Issue signed Protocol 03 record with exact evidence hashes
- Publish `B6_ACCEPTANCE_PUBLICATION_SHA` on `development` (append-only ladder)

## 18. Draft B6 acceptance record

```
Protocol 03 — ENT-AUDIT-X B6
Disposition: [ ACCEPT | REJECT | ACCEPT WITH CONDITIONS ]
TEST_EXECUTION_SHA: 89c76eb7b53a0a3ce6f75ab395cf4a2802638a91
EVIDENCE_PUBLICATION_SHA: 300859537f12a1de7bf3d75fd685bc3a5c727183
AUDIT_EXECUTION_SHA: <auditor verified HEAD>
B6_ACCEPTANCE_PUBLICATION_SHA: <commit publishing signed record>
Independent auditor: <name, mandate ref>
Date:
Signature:
Limitations: <scope text>
```

## 19. Required signatory fields

As §18. **Empty until auditor signs.**

## 20. Publication and post-acceptance instructions

After signed B6:

1. Operator **OD-ENT-01** → `ACCEPTED_BASELINE_SHA`
2. **OD-ENT-02** + GitHub admin → required checks / branch protection
3. Normal FREEZE qualification path per PL-08
4. **OD-ENT-08** + FRZ-AUD-01..10 + roadmap §5 → `FORMAL_CLOSURE_PUBLICATION_SHA`
5. ARCH-FREEZE handoff per [ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md](./ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md)

---

**Cross-reference:** Operator decisions — [ENT_EVOL_OPERATOR_SIGNATURE_PACKET_DRAFT.md](../ENT_EVOL_OPERATOR_SIGNATURE_PACKET_DRAFT.md)
