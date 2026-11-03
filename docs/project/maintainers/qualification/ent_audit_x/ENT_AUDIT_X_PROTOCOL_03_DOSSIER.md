# ENT-AUDIT-X — Protocol 03 independent audit dossier (draft)

**Status:** TECHNICAL PARTIAL — **no PASS declared by agent** (see master plan disposition)

| Field | Value |
| --- | --- |
| Execution SSOT | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) |
| Audited IMPLEMENTATION_SHA | `894b0e1aff071d19636d23bea406b95dcffc79e1` |
| Plan mechanical validator | `scripts/enterprise_audit/validate_post_implementation_master_plan.py` (exit 0 required before B6 dossier fill) |
| TEST_EXECUTION_SHA | `b8bb88c9db1060153ec0a6a29bbb79ba16e00078` |
| Qual wave report (local) | `.tmp/session/ent-audit-x-certifying-final/qual_wave_report.json` |
| B5 handoff tree (tracked) | [b6_independent_handoff/qualification_at_b8bb88c9db1060153ec0a6a29bbb79ba16e00078/](./b6_independent_handoff/qualification_at_b8bb88c9db1060153ec0a6a29bbb79ba16e00078/) |
| EVIDENCE_CANDIDATE_SHA | `894b0e1aff071d19636d23bea406b95dcffc79e1` |
| EVIDENCE_PUBLICATION_SHA | `894b0e1aff071d19636d23bea406b95dcffc79e1` |
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

Pinned GitHub Actions evidence (honest conclusions — **not** claimed green until post-fix re-run):

| workflow | run_id | head_sha | conclusion | artifact | url |
| --- | --- | --- | --- | --- | --- |
| enterprise-audit.yml | 41647490846 | `b8bb88c9db1060153ec0a6a29bbb79ba16e00078` | failure | ent-audit-pr-383 (warn: empty — pytest failed before CLI) | https://github.com/jakbuczarnecki/intergrax/actions/runs/41647490846 |
| enterprise-audit-nightly.yml | 41618283515 | `5a1bed2fd464a006725d9bd076dd2149dfa75790` | failure | nightly bundle (see run log) | https://github.com/jakbuczarnecki/intergrax/actions/runs/41618283515 |

**PR failure root cause @ TES:** shallow checkout — `test_changed_since_git.py` requires `HEAD~1` (fixed in workflow `fetch-depth: 0` @ post-TES commit). **Re-run required** after push for OD-ENT-02 enforceability evidence.

## Certifying qualification @ TEST_EXECUTION_SHA

| Stage | Evidence |
| --- | --- |
| L1/L2/L3 + semantic 85/85 + ADV31 | `qual_wave_report.json` status `PASS`, `qualification_sha=b8bb88c9…` |
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
