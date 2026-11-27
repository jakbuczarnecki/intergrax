# PLAN T-IGX-090-B — Wave 2 Round 1 Writer 2: W2-B2 Docs, EDER E-01..E-04 (DP-02-Safe), and EQO Re-baseline Sync

**Ticket:** `governance/tickets/T-IGX-090-B.md`
**Authority:** Operator approval `OP-EC-01` (`2026-11-26T17:26:01+01:00`, decisions `OD-EQO-09=A` and `OD-W2-DOCS=A`).
**Analysis baseline:** `origin/development` @ `646e63953ae1a7eb28a229322590c9cdcbb96625` (branch cut from `development` @ `32005c212ed4debca91f607ee30b3abfab79ce6c`, ancestor check `646e63953 <= HEAD` verified).

## Objective

Execute Wave 2 Round 1 Writer 2 documentation and governance bookkeeping strictly disjoint from Writer 1 (`W2-B1`/`W2-B3`), `AF-CLOSE` K0.3/K4 root files, and all product/test/script/CI trees:

1. **Batch `W2-B2` (45 B-2 doc-fix rows across 4 files):** Record PDC-X P2-snapshot supersession and per-row P9 disposition authority in `PDC_INDEX.md` (preserving byte-locked `PDC_GAP_REGISTER.md` untouched) and reconcile the 3 underlying domain doc files (`AGENT_DISTRIBUTION.md`, `architecture/BACKGROUND_TASKS.md`, `plans/BACKGROUND_TASKS.md`).
2. **`EDER E-01..E-04` (DP-02-safe under `OD-EQO-09=A`):**
   - `E-01`: Reconcile `reconciliation.json` so `rows_total = 1073`, `generated_at_head = 646e63953ae1a7eb28a229322590c9cdcbb96625`, and all 12 downstream breakdowns match live post-apply `register.jsonl`.
   - `E-02`: Apply DP-02-safe R1–R4 updates to `register.jsonl` (6 substantive R1 severity copies, 5 substantive R2 track mappings — holding the 325/305 `EQOA-A7-034` meta-links under DP-02 — plus R3 `current_head_sha`/`current_head_status`/`current_head_evidence` on all 1073 rows and R4 duplicate flags without mechanical `SUPERSEDED`).
   - `E-03`: Confirm `owner_status = "CONFIRMED"` for the 821 `SINGLE_PROPOSED` + 120 gap owner mappings (`15 MISSING`, `14 UNMAPPED`, `75 SHARED_UNRESOLVED`, `12 ROLE_NOT_OWNER`, `4 GENERIC_NOT_AN_OWNER`) per `E03_OWNER_BATCH_TABLE.md`, keeping all 132 `OPERATOR_ONLY` rows unchanged and `accountable_person = null` on all 1073 rows.
   - `E-04`: Bind `independent_audit_ref` on all 174 `needs_independent_audit = "yes"` rows per `E04_AUDIT_BINDING_MAP.json` (zero one-off audits; `re_audit_required_on_freeze_sha = true`; no PASS or certification claimed).
3. **`EQO` Re-baseline Sync (`eqo_closure/`):** Align `EQO_PIN_BIND.json` to `AD-AF-DEVTRAIN` `D-1` (retiring `rw-train` and orphaned `9f40f851f`), refresh `EQO_W2_TRAIN_VERIFICATION.md` (13/13 tasks on `646e63953`), refresh `EQO_C_DRIFT.jsonl` & `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` (18/18 layers, `0 UNKNOWN`), update `w4/W4-04_REQUALIFY_BATCH.md` and `w4/W4-05_NONE_JUSTIFICATION.md`, publish `x/EQO_X_HEAD_REBASELINE_SUMMARY.md` (`X-01..08` + `KR-01..28`), and sync `orchestration/EQO_ORCHESTRATION_STATE.json` while keeping `eqo_terminal_status: "EQO_NOT_CLOSED_EXACT_BLOCKERS"`.

## Scope boundary

| In scope | Out of scope |
| --- | --- |
| 4 `W2-B2` documentation files (`PDC_INDEX.md`, `plans/AGENT_DISTRIBUTION.md`, `architecture/BACKGROUND_TASKS.md`, `plans/BACKGROUND_TASKS.md`) | Byte-locked `PDC_GAP_REGISTER.md`, immutable `PDC_P7_D1..D8_AUDIT_REPORT.md`, `PDC_INDEPENDENT_AUDIT.md` |
| `enterprise_debt_register/reconciliation.json` and `register.jsonl` (`E-01..E-04`, DP-02-safe) | EDER `E-05` (`OD-EQO-10`), `lifecycle_state = CLOSED`, new `EXCEPTION_ACCEPTED`, fabricated `accountable_person` |
| 8 `eqo_closure/` re-baseline and summary files listed in Write Set | Product code (`intergrax/`, `agents/`, `applications/`), `scripts/`, `.github/`, `tests/` |
| Ticket `T-IGX-090-B.md` and plan `PLAN_T-IGX-090-B.md` | Writer 1 files (`W2-B1` 13 files, `W2-B3` 6 files), `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`, `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`, `arch_freeze_program/` |

## Exact write set (closed — 16 files)

1. `governance/tickets/plans/PLAN_T-IGX-090-B.md`
2. `governance/tickets/T-IGX-090-B.md`
3. `docs/project/maintainers/qualification/platform_domain_completeness/PDC_INDEX.md`
4. `docs/project/maintainers/plans/AGENT_DISTRIBUTION.md`
5. `docs/project/architecture/BACKGROUND_TASKS.md`
6. `docs/project/maintainers/plans/BACKGROUND_TASKS.md`
7. `docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json`
8. `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl`
9. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_PIN_BIND.json`
10. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_W2_TRAIN_VERIFICATION.md`
11. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_C_DRIFT.jsonl`
12. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl`
13. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/w4/W4-04_REQUALIFY_BATCH.md`
14. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/w4/W4-05_NONE_JUSTIFICATION.md`
15. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/x/EQO_X_HEAD_REBASELINE_SUMMARY.md`
16. `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/orchestration/EQO_ORCHESTRATION_STATE.json`

## Acceptance & Non-Claims

- Every JSON and JSONL file in the write set parses cleanly; `register.jsonl` has exactly `1073` valid JSON lines validating against `register_schema.json`.
- `register.jsonl` invariants: `0` rows set to `CLOSED`, `0` new `EXCEPTION_ACCEPTED`, `0` non-null `accountable_person`, `941` `CONFIRMED` + `132` `OPERATOR_ONLY` (`0` remaining owner-gap rows), `682` `UNRATED` (DP-02-safe), `423` `UNCLASSIFIED` (DP-02-safe), `174` `needs_independent_audit="yes"` rows carrying `independent_audit_ref`.
- `EQO_C_DRIFT.jsonl` and `EQO_C_HISTORICAL_CERTIFICATION_REQUALIFICATION_MATRIX.jsonl` each contain `18` valid JSON lines (`0 UNKNOWN`, `1 NO_PRODUCT_DELTA`, `5 TARGETED_REPLAY`, `12 FULL_REQUAL_REQUIRED`).
- `EQO_ORCHESTRATION_STATE.json` retains `"eqo_terminal_status": "EQO_NOT_CLOSED_EXACT_BLOCKERS"`.
- `PDC_GAP_REGISTER.md` is untouched and `scripts/qualification/pdc_x_requirements.py --check` exits `0`.
- `scripts/docs/check_docs_domain_pairs.py` exits `0`.
- `python -m scripts.governance.verify_plan_approval governance/tickets/T-IGX-090-B.md` exits `0`.
- Non-claims preserved: no `CLOSED`, `IIAC PASS`, `FRZ PASS`, `C01-C25 FULLY_MET`, `G6 PASS`, `EQO_CLOSED`, `G2`, `FCS`, or `M3` claimed.

## Test contract (serial, deterministic, no pytest)

```text
python -m scripts.governance.verify_plan_approval governance/tickets/T-IGX-090-B.md
python scripts/qualification/pdc_x_requirements.py --check
python scripts/docs/check_docs_domain_pairs.py
# + deterministic JSON/JSONL schema & EDER 1073-row invariant validation script
```

## Rollback

Revert the single commit on `ec-w2/b2-eder-eqo` touching only the 16 documentation/governance files listed in the closed write set.
