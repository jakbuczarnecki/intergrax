# INTEGRAx-QUALIFICATION-OBS-RECONSTRUCTION-REFERENCE-ALIGNMENT

## Metadata

| Field | Value |
|-------|-------|
| Task ID | `INTEGRAx-QUALIFICATION-OBS-RECONSTRUCTION-REFERENCE-ALIGNMENT` |
| Migration | OBS-RECONSTRUCTION-1 |
| Classification | Class A — qualification reference path alignment |
| Baseline audit HEAD (pre-commit) | `a4e989469253cef07eadc982843abc1f36dc0b0a` |

## Session Scope

Wyrównanie legacy qualification reference surfaces (OBS-TRACE-1, NPSC-5F R2/R3 Final `_MANDATORY_SUITES`) do canonical catalog SSOT i aktualnego ownera factual reconstruction w `observability/reconstruction`. Bez zmian produkcji, bez zmian semantyki `mandatory_sources.py`.

## Repository State

| Item | Value |
|------|-------|
| Branch | `development` |
| HEAD (pre-task) | `a4e989469253cef07eadc982843abc1f36dc0b0a` |
| `origin/development` | `a4e989469253cef07eadc982843abc1f36dc0b0a` |
| Ahead/behind | 0 / 0 |
| Tracked WIP in scope | none |
| Untracked WIP | unrelated qualification doc only |

## OBS-RECONSTRUCTION-1 Ownership

Factual execution reconstruction production owner po migracji: `intergrax/runtime/observability/reconstruction/`.

## Canonical Production Owner

`intergrax/runtime/observability/reconstruction/execution_reconstruction.py` — istnieje; jedyny `execution_reconstruction.py` w `intergrax/`.

## Canonical Qualification Owner

`tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py` — istnieje.

Legacy `tests/unit/runtime/diagnostics/test_execution_reconstruction.py` — **nie istnieje**.

## Canonical Catalog SSOT

`testing_support/execution_qualification/catalog/mandatory_sources.py`: `NPSC5F_R2_FINAL_MANDATORY` i `NPSC5F_R3_FINAL_MANDATORY` wskazują `tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py`. **Bez zmian w tym tasku.**

## OBS-TRACE-1 Stale Reference

`_RECONSTRUCTION_MODULES` wskazywał `intergrax/runtime/diagnostics/execution_reconstruction.py` → `FileNotFoundError` w gate import scan.

## R2 Final Stale Reference

`_MANDATORY_SUITES` label `Execution reconstruction` wskazywał diagnostics test path.

## R3 Final Stale Reference

Analogicznie do R2.

## Selected Correction

1. OBS-TRACE-1: jeden path w `_RECONSTRUCTION_MODULES`.
2. R2/R3 Final: jedna ścieżka suite pod label `Execution reconstruction`.

## Architecture Boundary Assessment

Brak naruszenia warstw Tier-0; alignment wyłącznie w qualification tests.

## Production Changes

**NONE**

## Qualification Reference Changes

Trzy pliki testów (patrz Changed Files). Dokumentacja tasku.

## Drift Guard Verification

`tests/unit/testing_support/execution_qualification/catalog/test_orchestrator_mandatory_reference_drift.py` — R1/R2/R3 (wynik w sekcji Findings po run).

## OBS-TRACE-1 Verification

Targeted gate + pełny plik `test_obs_trace_1_qualification.py` (wynik w Findings).

## Runtime Observability Regression

`tests/unit/runtime/observability/` (wynik w Findings).

## Qualification Regression

`tests/unit/testing_support/execution_qualification/` (wynik w Findings).

## Embedded Harness Safety

`test_mandatory_frozen_suite_passes` — nie modyfikowane; pełny mandatory matrix nie uruchamiany w tym tasku.

## Changed Files

- `tests/unit/runtime/observability/test_obs_trace_1_qualification.py`
- `tests/unit/runtime/architecture/test_npsc5f_r2_final_journal_completeness_ordering.py`
- `tests/unit/runtime/architecture/test_npsc5f_r3_final_governed_evidence_export.py`
- `docs/project/maintainers/qualification/INTEGRAX_QUALIFICATION_OBS_RECONSTRUCTION_REFERENCE_ALIGNMENT.md`

## Static Quality

Ruff check/format, pyright, `git diff --check` na changed tests (wynik w Findings).

## Findings

- Stale diagnostics paths potwierdzone na HEAD `a4e98946`; canonical catalog już poprawny.
- Po korekcie: `r2._MANDATORY_SUITES == NPSC5F_R2_FINAL_MANDATORY` i `r3._MANDATORY_SUITES == NPSC5F_R3_FINAL_MANDATORY` (exact-match).
- Pytest: drift R1/R2/R3 PASS; OBS-TRACE-1 gate + plik PASS; `execution_qualification/` 721 passed (7 skipped live perf); `runtime/observability/` w tej samej sesji PASS.
- R2/R3 targeted: predecessor SHA + protected drift PASS (bez `test_mandatory_frozen_suite_passes`).
- Ruff check + format PASS na changed tests; pyright na całych plikach R3 — błędy linii 211/456 **pre-existing**, poza diffem path alignment.
- Follow-up (poza scope): `tests/system/functional_diagnostics_h1/inventory.py` nadal wskazuje legacy diagnostics test owner.

## Decision

Wyrównać legacy surfaces do istniejącego canonical SSOT; nie odwracać kierunku authority.

## Final Verdict

```text
OBS RECONSTRUCTION QUALIFICATION REFERENCE ALIGNMENT = PASS
```
