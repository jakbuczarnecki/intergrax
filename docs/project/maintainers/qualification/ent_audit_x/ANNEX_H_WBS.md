# ANNEX H — Implementation WBS (reconciled P-11)

**PLAN_BASELINE_SHA:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Execution WBS (C001–C075):** [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) Matrix C + §14 DAG (75 TaskID). **C075-S2** @ HEAD: **343** pytest collect-only in `tests/qualification/ent_audit_x`; **26** TaskID QUALIFIED @ TES; CI green @ HEAD still OPEN.

## Batch summary (verified totals)

| Batch | Tasks | ID range |
| --- | ---: | --- |
| B0 Planning | 9 | ENT-T-0001..0009 |
| B1 Contracts | 10 | ENT-T-0101..0110 |
| B2 Runner | 12 | ENT-T-0201..0212 |
| B3 Registry/safety | 14 | ENT-T-0301..0314 |
| B4 Area adapters | 63 | ENT-T-0401..0463 |
| B5 TEN adapters | 13 | ENT-T-0501..0513 |
| B6 VBN adapters | 9 | ENT-T-0601..0609 |
| B7 Baseline/SHA | 10 | ENT-T-0701..0710 |
| B8 Reporting | 8 | ENT-T-0801..0808 |
| B9 CLI/modes | 10 | ENT-T-0901..0910 |
| B10 CI | 8 | ENT-T-1001..1008 |
| B11 Qualification | 97 | ENT-T-1101..1112 + ENT-T-1113..1197 |
| B12 Closure | 10 | ENT-T-1201..1210 |
| **Total** | **273** | unique IDs |

## B0 — Planning
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0001 | ENT | `docs/project/maintainers/qualification/ent_audit_x/` | - | manual review | `.tmp/proof/ent-audit-x/` | first publication @ 6328f0cf |
| ENT-T-0002 | ENT | `ENT_AUDIT_X_SHA_LADDER.md` | 0001 | manual | `.tmp/proof/ent-audit-x/` | baseline + publication SHAs |
| ENT-T-0003 | ENT | `Annex C/D/B/F/G/G2/H` | 0001 | checklist §4 | `.tmp/proof/ent-audit-x/` | 85 checks mapped |
| ENT-T-0004 | ENT | `roadmap §5` | 0001 | n/a | `.tmp/proof/ent-audit-x/` | no premature CLOSED |
| ENT-T-0005 | ENT | `EVOL handoff digest` | 0001 | diff | `.tmp/proof/ent-audit-x/` | TES/FINAL pins |
| ENT-T-0006 | ENT | `OD-ENT register in master plan` | 0001 | n/a | `.tmp/proof/ent-audit-x/` | decisions OPEN listed |
| ENT-T-0007 | ENT | `SSOT pointer` | 0001 | n/a | `.tmp/proof/ent-audit-x/` | single master plan |
| ENT-T-0008 | ENT | `completeness self-audit` | 0003 | n/a | `.tmp/proof/ent-audit-x/` | 13/13 criteria |
| ENT-T-0009 | ENT | `P-11/P-12/P-13 reconciliation` | 6328f0cf | mechanical validation | `.tmp/proof/ent-audit-x/` | WBS+SHA+G2 aligned |

## B1 — Contracts
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0101 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | AuditCheck contract |
| ENT-T-0102 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | Finding contract |
| ENT-T-0103 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | Severity/Category enums |
| ENT-T-0104 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | manifest schema |
| ENT-T-0105 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | report schema |
| ENT-T-0106 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | coverage schema |
| ENT-T-0107 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | baseline schema |
| ENT-T-0108 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | contract versioning policy |
| ENT-T-0109 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | tier import boundary test |
| ENT-T-0110 | ENT | `intergrax/platform/enterprise_audit/contracts/` | B0 | unit/pyright | `.tmp/proof/ent-audit-x/` | FrzMapping types |

## B2 — Runner
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0201 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | EnterpriseAuditRunner |
| ENT-T-0202 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | dependency scheduling |
| ENT-T-0203 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | check isolation |
| ENT-T-0204 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | timeouts |
| ENT-T-0205 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | cancellation |
| ENT-T-0206 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | bounded parallelism |
| ENT-T-0207 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | coverage accounting |
| ENT-T-0208 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | result aggregation |
| ENT-T-0209 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | execution manifest writer |
| ENT-T-0210 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | repo identity guard |
| ENT-T-0211 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | git SHA reader |
| ENT-T-0212 | ENT | `intergrax/platform/enterprise_audit/runner.py` | B1 | integration/unit | `.tmp/proof/ent-audit-x/` | registry DI |

## B3 — Registry + safety
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0301 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | check manifest SSOT |
| ENT-T-0302 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | CheckExecutionSandbox |
| ENT-T-0303 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | argv allowlist |
| ENT-T-0304 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | cwd constraint |
| ENT-T-0305 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | env redaction filter |
| ENT-T-0306 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | PytestGateCheck |
| ENT-T-0307 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | ScriptGateCheck |
| ENT-T-0308 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | mutating policy |
| ENT-T-0309 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | worktree isolation hook |
| ENT-T-0310 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | evidence capture |
| ENT-T-0311 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | dedupe engine |
| ENT-T-0312 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | manifest loader |
| ENT-T-0313 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | check authorization |
| ENT-T-0314 | ENT | `intergrax/platform/enterprise_audit/` | B2 | adversarial/unit | `.tmp/proof/ent-audit-x/` | duplicate run guard |

## B4 — Area adapters (63)
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0401 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-bnd-01.py` | B3 | L2 gate replay (PG-010) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-BND-01 |
| ENT-T-0402 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-bnd-02.py` | B3 | L2 gate replay (PG-010b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-BND-02 |
| ENT-T-0403 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-bnd-03.py` | B3 | L2 gate replay (PG-001c) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-BND-03 |
| ENT-T-0404 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-own-01.py` | B3 | L2 gate replay (PG-003b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OWN-01 |
| ENT-T-0405 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-own-02.py` | B3 | L2 gate replay (PG-003b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OWN-02 |
| ENT-T-0406 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-own-03.py` | B3 | L2 gate replay (PG-003b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OWN-03 |
| ENT-T-0407 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ctr-01.py` | B3 | L2 gate replay (PG-011) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CTR-01 |
| ENT-T-0408 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ctr-02.py` | B3 | L2 gate replay (PG-011) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CTR-02 |
| ENT-T-0409 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ctr-03.py` | B3 | L2 gate replay (PG-011) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CTR-03 |
| ENT-T-0410 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-typ-01.py` | B3 | L2 gate replay (PG-011b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TYP-01 |
| ENT-T-0411 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-typ-02.py` | B3 | L2 gate replay (PG-011) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TYP-02 |
| ENT-T-0412 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-typ-03.py` | B3 | L2 gate replay (PG-011) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TYP-03 |
| ENT-T-0413 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-dup-01.py` | B3 | L2 gate replay (PG-005) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DUP-01 |
| ENT-T-0414 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-dup-02.py` | B3 | L2 gate replay (PG-005) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DUP-02 |
| ENT-T-0415 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-dup-03.py` | B3 | L2 gate replay (PG-005) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DUP-03 |
| ENT-T-0416 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-plg-01.py` | B3 | L2 gate replay (PG-016) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PLG-01 |
| ENT-T-0417 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-plg-02.py` | B3 | L2 gate replay (PG-016) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PLG-02 |
| ENT-T-0418 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-plg-03.py` | B3 | L2 gate replay (PG-016) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PLG-03 |
| ENT-T-0419 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-vbn-01.py` | B3 | L2 gate replay (PG-003) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-VBN-01 |
| ENT-T-0420 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-vbn-02.py` | B3 | L2 gate replay (PG-004) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-VBN-02 |
| ENT-T-0421 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-vbn-03.py` | B3 | L2 gate replay (PG-007) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-VBN-03 |
| ENT-T-0422 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-gov-01.py` | B3 | L2 gate replay (PG-014) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-GOV-01 |
| ENT-T-0423 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-gov-02.py` | B3 | L2 gate replay (PG-014) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-GOV-02 |
| ENT-T-0424 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-gov-03.py` | B3 | L2 gate replay (PG-014) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-GOV-03 |
| ENT-T-0425 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-exe-01.py` | B3 | L2 gate replay (PG-012) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EXE-01 |
| ENT-T-0426 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-exe-02.py` | B3 | L2 gate replay (PG-012) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EXE-02 |
| ENT-T-0427 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-exe-03.py` | B3 | L2 gate replay (PG-012) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EXE-03 |
| ENT-T-0428 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-obs-01.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OBS-01 |
| ENT-T-0429 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-obs-02.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OBS-02 |
| ENT-T-0430 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-obs-03.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-OBS-03 |
| ENT-T-0431 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sta-01.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-STA-01 |
| ENT-T-0432 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sta-02.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-STA-02 |
| ENT-T-0433 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sta-03.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-STA-03 |
| ENT-T-0434 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sec-01.py` | B3 | L2 gate replay (PG-001b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-SEC-01 |
| ENT-T-0435 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sec-02.py` | B3 | L2 gate replay (PG-001b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-SEC-02 |
| ENT-T-0436 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-sec-03.py` | B3 | L2 gate replay (TO_IMPLEMENT_ENT_REPORT_REDACTION) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-SEC-03 |
| ENT-T-0437 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rel-01.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REL-01 |
| ENT-T-0438 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rel-02.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REL-02 |
| ENT-T-0439 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rel-03.py` | B3 | L2 gate replay (PG-h01) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REL-03 |
| ENT-T-0440 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rec-01.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REC-01 |
| ENT-T-0441 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rec-02.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REC-02 |
| ENT-T-0442 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-rec-03.py` | B3 | L2 gate replay (PG-015) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REC-03 |
| ENT-T-0443 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-prd-01.py` | B3 | L2 gate replay (PG-001) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PRD-01 |
| ENT-T-0444 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-prd-02.py` | B3 | L2 gate replay (PG-001) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PRD-02 |
| ENT-T-0445 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-prd-03.py` | B3 | L2 gate replay (TO_IMPLEMENT_LIMITATIONS_FLAG) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-PRD-03 |
| ENT-T-0446 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-cmp-01.py` | B3 | L2 gate replay (PG-019) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CMP-01 |
| ENT-T-0447 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-cmp-02.py` | B3 | L2 gate replay (PG-019) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CMP-02 |
| ENT-T-0448 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-cmp-03.py` | B3 | L2 gate replay (PG-019) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-CMP-03 |
| ENT-T-0449 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-reg-01.py` | B3 | L2 gate replay (PG-006b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REG-01 |
| ENT-T-0450 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-reg-02.py` | B3 | L2 gate replay (PG-020) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REG-02 |
| ENT-T-0451 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-reg-03.py` | B3 | L2 gate replay (PG-001) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-REG-03 |
| ENT-T-0452 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-doc-01.py` | B3 | L2 gate replay (PG-006) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DOC-01 |
| ENT-T-0453 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-doc-02.py` | B3 | L2 gate replay (TO_IMPLEMENT_MANIFEST_DIFF) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DOC-02 |
| ENT-T-0454 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-doc-03.py` | B3 | L2 gate replay (PG-006b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-DOC-03 |
| ENT-T-0455 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ten-01.py` | B3 | L2 gate replay (PG-009) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TEN-01 |
| ENT-T-0456 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ten-02.py` | B3 | L2 gate replay (PG-009b) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TEN-02 |
| ENT-T-0457 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-ten-03.py` | B3 | L2 gate replay (TO_IMPLEMENT_TEN_SCHEDULER) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-TEN-03 |
| ENT-T-0458 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-evol-01.py` | B3 | L2 gate replay (PG-008) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EVOL-01 |
| ENT-T-0459 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-evol-02.py` | B3 | L2 gate replay (TO_IMPLEMENT_EVOL_FAULT) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EVOL-02 |
| ENT-T-0460 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-evol-03.py` | B3 | L2 gate replay (PG-020) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-EVOL-03 |
| ENT-T-0461 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-inv-01.py` | B3 | L2 gate replay (PG-020) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-INV-01 |
| ENT-T-0462 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-inv-02.py` | B3 | L2 gate replay (TO_IMPLEMENT_COVERAGE_AUDIT) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-INV-02 |
| ENT-T-0463 | ENT | `intergrax/platform/enterprise_audit/checks/ent-a-inv-03.py` | B3 | L2 gate replay (TO_IMPLEMENT_FALSE_PASS) | `.tmp/proof/ent-audit-x/` | adapter for ENT-A-INV-03 |

## B5 — TEN adapters (13)
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0501 | ENT | `intergrax/platform/enterprise_audit/ten/ten-identity.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-IDENTITY |
| ENT-T-0502 | ENT | `intergrax/platform/enterprise_audit/ten/ten-propagation.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-PROPAGATION |
| ENT-T-0503 | ENT | `intergrax/platform/enterprise_audit/ten/ten-authority.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-AUTHORITY |
| ENT-T-0504 | ENT | `intergrax/platform/enterprise_audit/ten/ten-state.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-STATE |
| ENT-T-0505 | ENT | `intergrax/platform/enterprise_audit/ten/ten-provider.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-PROVIDER |
| ENT-T-0506 | ENT | `intergrax/platform/enterprise_audit/ten/ten-config.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-CONFIG |
| ENT-T-0507 | ENT | `intergrax/platform/enterprise_audit/ten/ten-credentials.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-CREDENTIALS |
| ENT-T-0508 | ENT | `intergrax/platform/enterprise_audit/ten/ten-evidence.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-EVIDENCE |
| ENT-T-0509 | ENT | `intergrax/platform/enterprise_audit/ten/ten-trace.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-TRACE |
| ENT-T-0510 | ENT | `intergrax/platform/enterprise_audit/ten/ten-recovery.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-RECOVERY |
| ENT-T-0511 | ENT | `intergrax/platform/enterprise_audit/ten/ten-async.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-ASYNC |
| ENT-T-0512 | ENT | `intergrax/platform/enterprise_audit/ten/ten-cross-tenant.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-CROSS-TENANT |
| ENT-T-0513 | ENT | `intergrax/platform/enterprise_audit/ten/ten-fail-closed.py` | B3 | TEN gate replay | `.tmp/proof/ent-audit-x/` | adapter for TEN-FAIL-CLOSED |

## B6 — VBN adapters (9)
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0601 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-inventory-drift.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-INVENTORY-DRIFT |
| ENT-T-0602 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-core-hardcoding.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-CORE-HARDCODING |
| ENT-T-0603 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-shadow-selection.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-SHADOW-SELECTION |
| ENT-T-0604 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-silent-fallback.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-SILENT-FALLBACK |
| ENT-T-0605 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-contract-leakage.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-CONTRACT-LEAKAGE |
| ENT-T-0606 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-sqlite-effective.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-SQLITE-EFFECTIVE |
| ENT-T-0607 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-obs-ops-coupling.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-OBS-OPS-COUPLING |
| ENT-T-0608 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-replaceability-matrix.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-REPLACEABILITY-MATRIX |
| ENT-T-0609 | ENT | `intergrax/platform/enterprise_audit/vbn/vbn-reachability.py` | B3 | VBN gate replay | `.tmp/proof/ent-audit-x/` | adapter for VBN-REACHABILITY |

## B7 — Baseline + SHA
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0701 | ENT | `ENTERPRISE_AUDIT_BASELINE.json schema` | B2 | validator | `.tmp/proof/ent-audit-x/` | baseline store |
| ENT-T-0702 | ENT | `AuditDriftEngine` | 0701 | unit drift classes | `.tmp/proof/ent-audit-x/` | AUD-07 |
| ENT-T-0703 | ENT | `non-ancestor baseline guard` | 0701 | unit | `.tmp/proof/ent-audit-x/` | exit 2 |
| ENT-T-0704 | ENT | `artifact digest tamper detect` | 0701 | adversarial | `.tmp/proof/ent-audit-x/` | exit 3 |
| ENT-T-0705 | ENT | `qualification_sha vs audit_execution_sha` | 0701 | unit | `.tmp/proof/ent-audit-x/` | doc-safe publish |
| ENT-T-0706 | ENT | `SHA ladder maintenance` | 0705 | manual/script | `.tmp/proof/ent-audit-x/` | lineage append-only |
| ENT-T-0707 | ENT | `EVOL D10 digest reconcile` | B0 | integration | `.tmp/proof/ent-audit-x/` | handoff match |
| ENT-T-0708 | ENT | `baseline promotion command` | 0702 | CLI test | `.tmp/proof/ent-audit-x/` | reject failed FREEZE |
| ENT-T-0709 | ENT | `OD-ENT-01 proposal doc` | 0708 | n/a | `.tmp/proof/ent-audit-x/` | operator baseline |
| ENT-T-0710 | ENT | `requal trigger policy doc` | 0708 | n/a | `.tmp/proof/ent-audit-x/` | post-freeze gov link |

## B8 — Reporting
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0801 | ENT | `JSON report builder` | B2 | schema test | `.tmp/proof/ent-audit-x/` | AUD-04 |
| ENT-T-0802 | ENT | `Markdown report renderer` | 0801 | snapshot test | `.tmp/proof/ent-audit-x/` | AUD-04 |
| ENT-T-0803 | ENT | `FRZ mapping section` | 0801 | unit | `.tmp/proof/ent-audit-x/` | AUD-03 |
| ENT-T-0804 | ENT | `QUALIFICATION_LIMITATIONS section` | 0801 | unit | `.tmp/proof/ent-audit-x/` | Path B |
| ENT-T-0805 | ENT | `ownership attribution` | 0801 | unit | `.tmp/proof/ent-audit-x/` | actionable findings |
| ENT-T-0806 | ENT | `CI artifact layout spec` | 0801 | doc | `.tmp/proof/ent-audit-x/` | B10 input |
| ENT-T-0807 | ENT | `golden sample reports` | 0801 | jsonschema | `.tmp/proof/ent-audit-x/` | regression |
| ENT-T-0808 | ENT | `coverage_matrix export` | 0801 | unit | `.tmp/proof/ent-audit-x/` | AUD-03 |

## B9 — CLI + modes
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-0901 | ENT | `scripts/enterprise_audit/run_enterprise_audit.py` | B7,B8 | CLI smoke | `.tmp/proof/ent-audit-x/` | entrypoint |
| ENT-T-0902 | ENT | `FAST mode policy` | 0901 | unit | `.tmp/proof/ent-audit-x/` | partial labeled |
| ENT-T-0903 | ENT | `FULL mode policy` | 0901 | integration | `.tmp/proof/ent-audit-x/` | full registry |
| ENT-T-0904 | ENT | `FREEZE mode policy` | 0901 | L3 | `.tmp/proof/ent-audit-x/` | maximal set |
| ENT-T-0905 | ENT | `CHANGED-SINCE policy` | 0901 | integration | `.tmp/proof/ent-audit-x/` | EVOL selection |
| ENT-T-0906 | ENT | `exit code 0-3 mapping` | 0901 | contract tests | `.tmp/proof/ent-audit-x/` | AUD-08 |
| ENT-T-0907 | ENT | `resource budgets per mode` | 0901 | unit | `.tmp/proof/ent-audit-x/` | timeouts |
| ENT-T-0908 | ENT | `human summary output` | 0901 | snapshot | `.tmp/proof/ent-audit-x/` | MD section |
| ENT-T-0909 | ENT | `machine JSON output path` | 0901 | integration | `.tmp/proof/ent-audit-x/` | AUD-04 |
| ENT-T-0910 | ENT | `mode coverage semantics` | 0902-0905 | negative tests | `.tmp/proof/ent-audit-x/` | no false FULL |

## B10 — CI
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-1001 | ENT | `.github/workflows/enterprise-audit.yml` | B9 | workflow run | `.tmp/proof/ent-audit-x/` | jobs exist |
| ENT-T-1002 | ENT | `PR job enterprise-audit-fast` | 1001 | CI | `.tmp/proof/ent-audit-x/` | stable name |
| ENT-T-1003 | ENT | `nightly FULL schedule` | 1001 | CI | `.tmp/proof/ent-audit-x/` | artifacts |
| ENT-T-1004 | ENT | `workflow_dispatch FREEZE` | 1001 | manual | `.tmp/proof/ent-audit-x/` | freeze candidate |
| ENT-T-1005 | ENT | `artifact retention policy` | 1001 | doc | `.tmp/proof/ent-audit-x/` | SHA-bound |
| ENT-T-1006 | ENT | `path filters CHANGED-SINCE` | 1001 | PR test | `.tmp/proof/ent-audit-x/` | diff scoped |
| ENT-T-1007 | ENT | `OD-ENT-02 branch protection runbook` | 1001 | n/a | `.tmp/proof/ent-audit-x/` | exact GitHub steps |
| ENT-T-1008 | ENT | `FREEZE reports unprotected branch` | 1007 | test | `.tmp/proof/ent-audit-x/` | FRZ-AUD-10 honest |

## B11 — Qualification (97)
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-1101 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | unit conftest |
| ENT-T-1102 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | schema validation suite |
| ENT-T-1103 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | registry ordering tests |
| ENT-T-1104 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | drift semantics tests |
| ENT-T-1105 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | exit code contract tests |
| ENT-T-1106 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | sandbox injection adversarial |
| ENT-T-1107 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | L2 runner smoke |
| ENT-T-1108 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | L2 CLI smoke |
| ENT-T-1109 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | L3 FREEZE harness |
| ENT-T-1110 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | adversarial false PASS |
| ENT-T-1111 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | adversarial silent skip |
| ENT-T-1112 | ENT | `tests/qualification/ent_audit_x/` | B9 | pytest | `.tmp/proof/ent-audit-x/` | adversarial tamper |
| ENT-T-1113 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_bnd_01.py` | ENT-T-0401 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-BND-01 |
| ENT-T-1114 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_bnd_02.py` | ENT-T-0402 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-BND-02 |
| ENT-T-1115 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_bnd_03.py` | ENT-T-0403 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-BND-03 |
| ENT-T-1116 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_own_01.py` | ENT-T-0404 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OWN-01 |
| ENT-T-1117 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_own_02.py` | ENT-T-0405 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OWN-02 |
| ENT-T-1118 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_own_03.py` | ENT-T-0406 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OWN-03 |
| ENT-T-1119 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ctr_01.py` | ENT-T-0407 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CTR-01 |
| ENT-T-1120 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ctr_02.py` | ENT-T-0408 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CTR-02 |
| ENT-T-1121 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ctr_03.py` | ENT-T-0409 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CTR-03 |
| ENT-T-1122 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_typ_01.py` | ENT-T-0410 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TYP-01 |
| ENT-T-1123 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_typ_02.py` | ENT-T-0411 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TYP-02 |
| ENT-T-1124 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_typ_03.py` | ENT-T-0412 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TYP-03 |
| ENT-T-1125 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_dup_01.py` | ENT-T-0413 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DUP-01 |
| ENT-T-1126 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_dup_02.py` | ENT-T-0414 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DUP-02 |
| ENT-T-1127 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_dup_03.py` | ENT-T-0415 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DUP-03 |
| ENT-T-1128 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_plg_01.py` | ENT-T-0416 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PLG-01 |
| ENT-T-1129 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_plg_02.py` | ENT-T-0417 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PLG-02 |
| ENT-T-1130 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_plg_03.py` | ENT-T-0418 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PLG-03 |
| ENT-T-1131 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_01.py` | ENT-T-0419 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-VBN-01 |
| ENT-T-1132 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_02.py` | ENT-T-0420 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-VBN-02 |
| ENT-T-1133 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_03.py` | ENT-T-0421 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-VBN-03 |
| ENT-T-1134 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_gov_01.py` | ENT-T-0422 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-GOV-01 |
| ENT-T-1135 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_gov_02.py` | ENT-T-0423 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-GOV-02 |
| ENT-T-1136 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_gov_03.py` | ENT-T-0424 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-GOV-03 |
| ENT-T-1137 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_exe_01.py` | ENT-T-0425 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EXE-01 |
| ENT-T-1138 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_exe_02.py` | ENT-T-0426 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EXE-02 |
| ENT-T-1139 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_exe_03.py` | ENT-T-0427 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EXE-03 |
| ENT-T-1140 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_obs_01.py` | ENT-T-0428 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OBS-01 |
| ENT-T-1141 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_obs_02.py` | ENT-T-0429 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OBS-02 |
| ENT-T-1142 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_obs_03.py` | ENT-T-0430 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-OBS-03 |
| ENT-T-1143 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sta_01.py` | ENT-T-0431 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-STA-01 |
| ENT-T-1144 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sta_02.py` | ENT-T-0432 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-STA-02 |
| ENT-T-1145 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sta_03.py` | ENT-T-0433 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-STA-03 |
| ENT-T-1146 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sec_01.py` | ENT-T-0434 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-SEC-01 |
| ENT-T-1147 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sec_02.py` | ENT-T-0435 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-SEC-02 |
| ENT-T-1148 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_sec_03.py` | ENT-T-0436 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-SEC-03 |
| ENT-T-1149 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rel_01.py` | ENT-T-0437 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REL-01 |
| ENT-T-1150 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rel_02.py` | ENT-T-0438 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REL-02 |
| ENT-T-1151 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rel_03.py` | ENT-T-0439 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REL-03 |
| ENT-T-1152 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rec_01.py` | ENT-T-0440 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REC-01 |
| ENT-T-1153 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rec_02.py` | ENT-T-0441 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REC-02 |
| ENT-T-1154 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_rec_03.py` | ENT-T-0442 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REC-03 |
| ENT-T-1155 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_prd_01.py` | ENT-T-0443 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PRD-01 |
| ENT-T-1156 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_prd_02.py` | ENT-T-0444 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PRD-02 |
| ENT-T-1157 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_prd_03.py` | ENT-T-0445 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-PRD-03 |
| ENT-T-1158 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_cmp_01.py` | ENT-T-0446 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CMP-01 |
| ENT-T-1159 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_cmp_02.py` | ENT-T-0447 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CMP-02 |
| ENT-T-1160 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_cmp_03.py` | ENT-T-0448 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-CMP-03 |
| ENT-T-1161 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_reg_01.py` | ENT-T-0449 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REG-01 |
| ENT-T-1162 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_reg_02.py` | ENT-T-0450 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REG-02 |
| ENT-T-1163 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_reg_03.py` | ENT-T-0451 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-REG-03 |
| ENT-T-1164 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_doc_01.py` | ENT-T-0452 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DOC-01 |
| ENT-T-1165 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_doc_02.py` | ENT-T-0453 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DOC-02 |
| ENT-T-1166 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_doc_03.py` | ENT-T-0454 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-DOC-03 |
| ENT-T-1167 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_01.py` | ENT-T-0455 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TEN-01 |
| ENT-T-1168 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_02.py` | ENT-T-0456 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TEN-02 |
| ENT-T-1169 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_03.py` | ENT-T-0457 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-TEN-03 |
| ENT-T-1170 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_evol_01.py` | ENT-T-0458 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EVOL-01 |
| ENT-T-1171 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_evol_02.py` | ENT-T-0459 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EVOL-02 |
| ENT-T-1172 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_evol_03.py` | ENT-T-0460 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-EVOL-03 |
| ENT-T-1173 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_inv_01.py` | ENT-T-0461 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-INV-01 |
| ENT-T-1174 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_inv_02.py` | ENT-T-0462 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-INV-02 |
| ENT-T-1175 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ent_a_inv_03.py` | ENT-T-0463 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for ENT-A-INV-03 |
| ENT-T-1176 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_identity.py` | ENT-T-0501 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-IDENTITY |
| ENT-T-1177 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_propagation.py` | ENT-T-0502 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-PROPAGATION |
| ENT-T-1178 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_authority.py` | ENT-T-0503 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-AUTHORITY |
| ENT-T-1179 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_state.py` | ENT-T-0504 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-STATE |
| ENT-T-1180 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_provider.py` | ENT-T-0505 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-PROVIDER |
| ENT-T-1181 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_config.py` | ENT-T-0506 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-CONFIG |
| ENT-T-1182 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_credentials.py` | ENT-T-0507 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-CREDENTIALS |
| ENT-T-1183 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_evidence.py` | ENT-T-0508 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-EVIDENCE |
| ENT-T-1184 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_trace.py` | ENT-T-0509 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-TRACE |
| ENT-T-1185 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_recovery.py` | ENT-T-0510 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-RECOVERY |
| ENT-T-1186 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_async.py` | ENT-T-0511 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-ASYNC |
| ENT-T-1187 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_cross_tenant.py` | ENT-T-0512 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-CROSS-TENANT |
| ENT-T-1188 | ENT | `tests/qualification/ent_audit_x/acceptance/test_ten_fail_closed.py` | ENT-T-0513 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for TEN-FAIL-CLOSED |
| ENT-T-1189 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_inventory_drift.py` | ENT-T-0601 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-INVENTORY-DRIFT |
| ENT-T-1190 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_core_hardcoding.py` | ENT-T-0602 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-CORE-HARDCODING |
| ENT-T-1191 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_shadow_selection.py` | ENT-T-0603 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-SHADOW-SELECTION |
| ENT-T-1192 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_silent_fallback.py` | ENT-T-0604 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-SILENT-FALLBACK |
| ENT-T-1193 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_contract_leakage.py` | ENT-T-0605 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-CONTRACT-LEAKAGE |
| ENT-T-1194 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_sqlite_effective.py` | ENT-T-0606 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-SQLITE-EFFECTIVE |
| ENT-T-1195 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_obs_ops_coupling.py` | ENT-T-0607 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-OBS-OPS-COUPLING |
| ENT-T-1196 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_replaceability_matrix.py` | ENT-T-0608 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-REPLACEABILITY-MATRIX |
| ENT-T-1197 | ENT | `tests/qualification/ent_audit_x/acceptance/test_vbn_reachability.py` | ENT-T-0609 | acceptance pytest | `.tmp/proof/ent-audit-x/` | qual for VBN-REACHABILITY |

## B12 — Closure
| ID | Owner | Path | Dependencies | Tests | Evidence | Acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| ENT-T-1201 | ENT | `ENT_AUDIT_X_PROTOCOL_03 dossier` | B11 | auditor checklist | `.tmp/proof/ent-audit-x/` | B6 ready |
| ENT-T-1202 | ENT | `B6 evidence bundle layout` | B11 | manifest | `.tmp/proof/ent-audit-x/` | exact SHA |
| ENT-T-1203 | ENT | `corrective batch playbook` | 1201 | doc | `.tmp/proof/ent-audit-x/` | REJECT path |
| ENT-T-1204 | ENT | `FRZ-AUD promotion package` | 1201 | checklist | `.tmp/proof/ent-audit-x/` | W12 gate |
| ENT-T-1205 | ENT | `OD-ENT-08 operator record` | 1204 | signed doc | `.tmp/proof/ent-audit-x/` | closure |
| ENT-T-1206 | ENT | `roadmap §5 ledger row` | 1205 | git | `.tmp/proof/ent-audit-x/` | ENT CLOSED text |
| ENT-T-1207 | ENT | `ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md` | 1206 | review | `.tmp/proof/ent-audit-x/` | handoff |
| ENT-T-1208 | ENT | `remote HEAD verification` | 1206 | git fetch | `.tmp/proof/ent-audit-x/` | matches publish SHA |
| ENT-T-1209 | ENT | `independent B6 acceptance record` | 1201 | Protocol 03 | `.tmp/proof/ent-audit-x/` | external sign-off |
| ENT-T-1210 | ENT | `stage CLOSED declaration` | 1209,1205 | operator | `.tmp/proof/ent-audit-x/` | not implementer |

## GAP → WBS traceability

- GAP-C-01..14 → B1–B3, B7, B9, B10, B11, B12 tasks listed above.
- GAP-COV-* (85) → B4/B5/B6 adapter + B11 qual acceptance pair per Annex G2.
- GAP-P-* → ENT-T-0303, 1007, 1008, 0707, etc.