# PLAN T-IGX-036-C — PDC-P2 requirements and gap audit

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md) §C–E and §G "PDC-P2". Depends on T-IGX-036-B (P1 inventory). No product code.

## Deliverables

1. **PDC-P2-01..08:** the final requirements matrix `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, rendered to `.md` by `scripts/qualification/pdc_x_requirements.py`.
   - Stable IDs `PDC-Dn-Rnnn`.
   - The 23 mandated fields per row: REQUIREMENT_ID, DOMAIN_ID, REQUIREMENT, SEMANTIC_OWNER, IMPLEMENTATION_PATH, PUBLIC_CONTRACT, CURRENT_IMPLEMENTATION_STATUS, EXISTING_CERTIFICATION, EXISTING_ACCEPTANCE_SHA, CURRENT_HEAD_DELTA, MANDATORY_GAP, CORRECTIVE_TASK, TEST_CASE, FRZ_IDS, SECURITY_IMPACT, TENANT_IMPACT, DEPENDENCIES, CERTIFICATION_STATUS, AUDITOR_ID, AUDITED_SHA, EVIDENCE_PATH, ARCH_FREEZE_BLOCKER, FINAL_DISPOSITION.
   - Every preliminary `TRACKED_UNKNOWN` is resolved against code. Classes are those of master plan §7.
   - No row may be PASS or certified; `CERTIFICATION_STATUS` stays `NOT_CERTIFIED` until P7.
2. **PDC-P2-09:** reuse-delta evidence. For each row classified `ALREADY_CERTIFIED_REUSABLE`, record the accepted SHA → current HEAD diff over the row's implementation paths, plus a replay of the existing qualification suite for the domain. Each replay is a single bounded `uv run pytest` per domain, logged under `.tmp/proof/pdc-x-p2/<Dn>/`. A non-empty delta, or a replay failure that also fails at baseline, is recorded as such and is never hidden.
3. **PDC-P2-10:** FRZ crosswalk `PDC_FRZ_COVERAGE_MATRIX.md`. It uses existing FRZ IDs only, under Option A of OD-PDC-FRZ-01, which is drafted and still pending signature. A mechanical check confirms 225 criteria in 26 families.
4. **PDC-P2-11:** gap register `PDC_GAP_REGISTER.md`. One entry per non-reuse requirement, with owner lane, closure path, duplication risk and regression risk.
5. A unit test `tests/unit/governance/test_pdc_x_requirements.py` that checks the matrix schema:
   - all 23 fields are present;
   - IDs are unique;
   - no row is `TRACKED_UNKNOWN`, PASS or certified;
   - every non-reuse row has a corrective task;
   - every FRZ ID exists in the checklist.

## Acceptance

- Every mandatory requirement has a verifiable status and a closure path.
- No requirement is marked PASS without evidence.
- The checklist count is unchanged.

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_requirements.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; FRZ status changes; signing OD-PDC-FRZ-01; ADRs (P3); certification.
