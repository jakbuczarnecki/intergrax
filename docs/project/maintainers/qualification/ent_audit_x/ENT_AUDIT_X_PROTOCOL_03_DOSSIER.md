# ENT-AUDIT-X — Protocol 03 independent audit dossier (draft)

**Status:** W8 B6 CORRECTIVE WAVE — **TECHNICAL RE-AUDIT COMPLETE @ `0e538269…` (RECOMMENDATION ONLY)** — **FORMAL B6 ATTESTATION PENDING** — no signed B6 PASS / no OD approved / no FRZ CLOSED

| Field | Value |
| --- | --- |
| Execution SSOT | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) |
| W8 findings disposition | [ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md](./ENT_AUDIT_X_W8_B6_FINDINGS_DISPOSITION.md) |
| Audited IMPLEMENTATION_SHA (IIAC loop-1 corrective code) | `62d1c615855a3eaf569777a10e342aacc889a128` |
| **TEST_EXECUTION_SHA (current qual)** | `62d1c615855a3eaf569777a10e342aacc889a128` |
| Historical TEST_EXECUTION_SHA (W8 corrective) | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| Historical TEST_EXECUTION_SHA (R6/R7) | `87203c1b6ce9786dce0523cca29b556106b6bc8d` — R7 B5 fails `verify_b5_publication_tree()` (ENT-W8-B6-001) |
| Qual wave report (current) | [qualification_at_62d1c615…/qual_wave_report.json](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/qual_wave_report.json) |
| B5 handoff tree (current) | [qualification_at_62d1c615…/](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/) |
| B5 handoff tree (W8 corrective, historical) | [qualification_at_89c76eb7…/](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/) |
| B5 handoff tree (R7 defect preserved) | [qualification_at_87203c1b…/](./b6_independent_handoff/qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d/) + [R7_DEFECT_ENT-W8-B6-001.md](./b6_independent_handoff/R7_DEFECT_ENT-W8-B6-001.md) |
| **EVIDENCE_PUBLICATION_SHA (current)** | `163ef0c03b3cbead556499ee81cb99d640665dce` |
| Historical EVIDENCE_PUBLICATION_SHA (W8 corrective) | `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| Historical EVIDENCE_PUBLICATION_SHA (R7) | `d18f01f2291328a073dcbdfbe05489faf98013b7` |
| INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` — regression gate green; technical recommendation only ([report draft](./ENT_AUDIT_X_INDEPENDENT_TECHNICAL_REAUDIT_REPORT_DRAFT.md)) |
| B6_ACCEPTANCE_PUBLICATION_SHA | *unset — authorized independent auditor signs Protocol 03* |
| ACCEPTED_BASELINE_SHA | *operator OD-ENT-01 after formal B6* |
| Formal B6 package | [ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md](./ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md) (**DRAFT — NOT SIGNED**) |

## Scope

- Tier-0 package `intergrax/platform/enterprise_audit/`
- CLI `scripts/enterprise_audit/run_enterprise_audit.py`
- Qualification tree `tests/qualification/ent_audit_x/` (collect-only reconciled @ TES)
- Workflows `.github/workflows/enterprise-audit*.yml`

## Evidence pointers

- Corrective implementation plan: [ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md)
- SHA ladder: [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md)
- Gap register: [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md)
- CI runbook: [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md)

## CI qualification runs

**W8 @ `2eddd42230da93f5f26dd648840c3c8af629f7ba`** (covers W8 code; HEAD `5ceed24c…` is doc-only post-TES):

| workflow | run_id | head_sha | conclusion | url |
| --- | --- | --- | --- | --- |
| enterprise-audit.yml | 41797446254 | `2eddd42230da93f5f26dd648840c3c8af629f7ba` | *verify @ re-audit* | https://github.com/jakbuczarnecki/intergrax/actions/runs/41797446254 |
| enterprise-audit-nightly.yml | 42491342540 | `62d1c615855a3eaf569777a10e342aacc889a128` | success | https://github.com/jakbuczarnecki/intergrax/actions/runs/42491342540 |
| Regression gate | 41797446087 | `2eddd42230da93f5f26dd648840c3c8af629f7ba` | success | https://github.com/jakbuczarnecki/intergrax/actions/runs/41797446087 |

**CI @ INDEPENDENT_TECHNICAL_REAUDIT_SHA (`0e538269…`):** Regression gate run `41804655934` → **success**. **ENT nightly FULL @ FROZEN ENT AUDIT HEAD:** run `41800553851` @ `618e234ee4dc41aa3673de5e1e81271f6039ca6c` → **success** (W8-004 dual-pin continuity — operator/admin boundary for required-check enforcement remains OD-ENT-02).

Historical (pre-W8):

| workflow | run_id | head_sha | conclusion | artifact | url |
| --- | --- | --- | --- | --- | --- |
| enterprise-audit.yml (push) | 41779404835 | `1bee524218d9185175f3e24c6e6f205565ff0569` | success | FAST + L1/L2/L3 qual | https://github.com/jakbuczarnecki/intergrax/actions/runs/41779404835 |
| enterprise-audit-nightly.yml | 41779405128 | `1bee524218d9185175f3e24c6e6f205565ff0569` | success (FULL dispatch @ HEAD) | nightly bundle | https://github.com/jakbuczarnecki/intergrax/actions/runs/41779405128 |
| enterprise-audit.yml (PR @ prior HEAD) | 41710223602 | `d3d9b62da165ce98236ad7ece51a0f4b750a67d2` | success | ent-audit-pr gate + FAST | https://github.com/jakbuczarnecki/intergrax/actions/runs/41710223602 |

Historical (pre-HEAD — audit context only):

| workflow | run_id | head_sha | conclusion | notes |
| --- | --- | --- | --- | --- |
| enterprise-audit.yml | 41647490846 | `b8bb88c9db1060153ec0a6a29bbb79ba16e00078` | failure | shallow checkout before `fetch-depth: 0` fix |
| enterprise-audit-nightly.yml | 41618283515 | `5a1bed2fd464a006725d9bd076dd2149dfa75790` | failure | schedule on `main` |

## Certifying qualification @ TEST_EXECUTION_SHA

| Stage | Evidence |
| --- | --- |
| L1/L2/L3 + semantic 85/85 + ADV31 | [qual_wave_report.json](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/qual_wave_report.json) status `PASS`, `qualification_sha=62d1c615855a3eaf569777a10e342aacc889a128` |
| Certifying FULL 85/85 | [enterprise_audit_report.json](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/l3-cli-full/enterprise_audit_report.json) → `checks_executed=85`, `coverage_label=FULL` |
| VBN D01–D31 | [vbn_domains.v1.json](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/l3-cli-full/vbn_domains.v1.json) → `domains_accounted=31` @ TES digest pin |
| FRZ map | [frz_coverage.v1.json](./b6_independent_handoff/qualification_at_62d1c615855a3eaf569777a10e342aacc889a128/l3-cli-full/frz_coverage.v1.json) → `row_count=85` |

## External blockers (GAP-B)

| GAP-B-ID | Owner | Status | Notes |
| --- | --- | --- | --- |
| BLK-EVOL-B6 | Operator / EVOL auditor | OPEN | EVOL Protocol 03 B6 still pending |
| BLK-EVOL-FRZ | Operator | OPEN | FRZ-EVOL after EVOL B6 |
| BLK-EVOL-HANDOFF | Operator | OPEN | W7-06 handoff pin |
| BLK-CI-ENFORCE | OD-ENT-02 + GitHub admin | OPEN | Branch protection 404; workflows exist |
| BLK-ROADMAP-ENT-DEP | OD-ENT-00 | OPEN | EVOL-X formal closure decision |
| BLK-ENT-B6 | Independent Protocol 03 | OPEN | Awaiting B5 + external audit |

## Mechanical preflight

```bash
uv run python scripts/enterprise_audit/mechanical_readiness_check.py
```

## Operator boundaries (F-07)

- Baseline **promotion** requires operator OD-ENT-01 record (`promote-baseline` CLI).
- Agent/bootstrap may write **candidate** baseline only.
- Independent B6 issues PASS/FAIL; this dossier does **not** substitute external attestation (**no pass declared by agent**).
