# ENT-AUDIT-X — Protocol 03 independent audit dossier (draft)

**Status:** W8 B6 CORRECTIVE WAVE — **READY FOR INDEPENDENT RE-AUDIT** — no B6 PASS / no OD approved / no FRZ CLOSED

| Field | Value |
| --- | --- |
| Execution SSOT | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) |
| Audited IMPLEMENTATION_SHA (W8 corrective code) | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| **TEST_EXECUTION_SHA (current qual)** | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| Historical TEST_EXECUTION_SHA (R6/R7) | `87203c1b6ce9786dce0523cca29b556106b6bc8d` — R7 B5 fails `verify_b5_publication_tree()` (ENT-W8-B6-001) |
| Qual wave report (current) | [qualification_at_89c76eb7…/qual_wave_report.json](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/qual_wave_report.json) |
| B5 handoff tree (current) | [qualification_at_89c76eb7…/](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/) |
| B5 handoff tree (R7 defect preserved) | [qualification_at_87203c1b…/](./b6_independent_handoff/qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d/) + [R7_DEFECT_ENT-W8-B6-001.md](./b6_independent_handoff/R7_DEFECT_ENT-W8-B6-001.md) |
| **EVIDENCE_PUBLICATION_SHA (current)** | `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| Historical EVIDENCE_PUBLICATION_SHA (R7) | `d18f01f2291328a073dcbdfbe05489faf98013b7` |
| B6_ACCEPTANCE_PUBLICATION_SHA | *external auditor — agent does not sign* |
| ACCEPTED_BASELINE_SHA | *operator OD-ENT-01 after B6* |

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

Latest green @ `1bee524218d9185175f3e24c6e6f205565ff0569` (historical failures retained below — **not** HEAD proof):

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
| L1/L2/L3 + semantic 85/85 + ADV31 | `qual_wave_report.json` status `PASS`, `qualification_sha=87203c1…` |
| Certifying FULL 85/85 | `b6_independent_handoff/.../l3-cli-full/enterprise_audit_report.json` → `checks_executed=85`, `coverage_label=FULL` |
| VBN D01–D31 | `vbn_domains.v1.json` → `domains_accounted=31` @ TES digest pin |
| FRZ map | `frz_coverage.v1.json` → `row_count=85` |

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
