# HARDENING-9 — NPSC-5F Protected Drift Requalification V2

**Task:** `HARDENING_9_NPSC5F_PROTECTED_DRIFT_REQUALIFICATION_V2`  
**Status:** `REQUALIFIED / RE-FROZEN / PASS`  
**Audited protected-change tip:** `4e92d14c58f02200518786c59c9128b3068e4bc5`  
**Re-freeze / `origin/development` SHA:** `22c4793da4ba751fff6c93f780a7f6848650a5d9` (recertified: MP-4R7 docs-only after OBS commits)  
**Prior sentinel (V1 / OBS-CONTRACT-BOUNDARY-1):** `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1`

## Executive summary

After V1 re-freeze @ `a2b33ba96`, three OBS follow-up commits on integrated `development` changed R1/R2/Final Evidence Plane protected surfaces without advancing sentinels. All items are **contract-first, behavior-preserving** evolution (DTO lifts + bounded process-local history); **no** layer-boundary regression, **no** duplicate authority, **no** gate weakening.

**Classification:** `QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED` for every drift path.  
**UNKNOWN:** `0` · **UNQUALIFIED_PROTECTED_DRIFT:** `0` · **REAL_ARCHITECTURE_REGRESSION:** `0`

**Production code changed in this requalification task:** NO — sentinel baselines + qualification evidence only.

## Baseline re-freeze

| Sentinel | Old | New |
|----------|-----|-----|
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `a2b33ba965c57cd3c812720f7b5f84b40b2b32f1` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` |

`NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA` / `NPSC5F_R3_H1_QUALIFICATION_RECORD_SHA`: **intentionally not advanced** (separate integrated-head pin task).

## Pre-fix drift (repro @ `a2b33ba..origin/development`)

| Family | Protected paths |
|--------|-----------------|
| R1 | `intergrax/runtime/events/event_bus.py`, `intergrax/runtime/events/persistence_contract.py` |
| R2 | `intergrax/runtime/events/persistence_contract.py` |
| Final Evidence Plane | `event_bus.py`, `persistence_contract.py`, `runtime_event_history.py`, `export_attributes.py`, `functional_validation_evidence.py`, `problem_signal.py` |

Protected drift trio before re-freeze: **0/3 PASS**.

## Introducing commits (protected production)

| SHA | Message | Qualification evidence |
|-----|---------|------------------------|
| `78f4350c80085f2706c0acd92dfafd122340e8c1` | OBS-CONTRACT-BOUNDARY-2: platform problem signal contract-owned | `tests/unit/contracts/test_platform_problem_signal_contract_boundary.py`; `OBSERVABILITY.md` / `DIAGNOSTICS.md` |
| `1d2936c4cddc1b152bcb91c4cc71aaa723e949a1` | OBS-EVIDENCE-PERSISTENCE-CONTRACT-CLEANUP | `tests/unit/contracts/test_evidence_persistence_port_contract_boundary.py`, `test_task_runtime_event_runs_contract.py` |
| `550227883070849b87dbfdb4a64f818050c5d1c6` | OBS-RUNTIME-HISTORY-BOUNDS | `tests/unit/runtime/events/test_runtime_event_history_bounds.py`; `intergrax/contracts/runtime_event_history.py` |

No further protected `intergrax/runtime/events/` or `intergrax/runtime/observability/` path changes from `550227883` through `22c4793da` (`22c4793da` is docs-only MP-4R7 sync).

## Protected drift ledger

| Family | Protected path | Old baseline | Introducing commit | Classification | Qualification sufficient |
|--------|----------------|--------------|--------------------|----------------|------------------------|
| R1 | `event_bus.py` | `a2b33ba96` | `550227883` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES (history bounds; durable commit before delivery preserved) |
| R1 | `persistence_contract.py` | `a2b33ba96` | `1d2936c4c` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES (port contract boundary tests) |
| R2 | `persistence_contract.py` | `a2b33ba96` | `1d2936c4c` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES (journal ordering / AsOf semantics unchanged) |
| Final | `event_bus.py` | `a2b33ba96` | `550227883` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES |
| Final | `persistence_contract.py` | `a2b33ba96` | `1d2936c4c` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES |
| Final | `runtime_event_history.py` | `a2b33ba96` | `550227883` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES (new module + contract policy) |
| Final | `export_attributes.py` | `a2b33ba96` | `78f4350c8` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES (shim → `application_observability_attributes` contract) |
| Final | `functional_validation_evidence.py` | `a2b33ba96` | `78f4350c8` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES |
| Final | `problem_signal.py` | `a2b33ba96` | `78f4350c8` | QUALIFIED_CHANGE_BASELINE_NOT_REFRESHED | YES |

## Architecture proofs (audit summary)

| Audit | Result |
|-------|--------|
| Contract-first (consumers → contracts) | **PASS** |
| Persistence port / adapter / failure isolation | **PASS** (`EvidencePersistencePort`; no vendor in core) |
| Event bus durable evidence before delivery | **PASS** (no alternate side channel introduced) |
| R1 durability / tenant integrity | **PASS** (R1 final gate suite) |
| R2 journal ordering | **PASS** (R2 final gate suite) |
| Evidence Plane read-only / no execution control | **PASS** |
| Pluginability / hidden fallback / service locator | **0** new regressions |
| Layer boundaries (`test_hardening_3_layer_boundary_gate`) | **PASS** |
| EE-A2 / EE-A2-H2 | **PASS** |
| RC-01 / RC-02 / RC-03 | **CLOSED** (ancestry + prior qualification; no new RC drift) |

## Regression evidence (post re-freeze)

Record in commit CI / local run:

- Protected drift trio: **3/3 PASS**
- `test_npsc5f_final_mandatory_regression_matrix_passes`: run log (H1 integrated-head pin may remain expected blocker)

## Test weakening

`SKIP` / `XFAIL` / assertion removal / protected-path removal / allowlist expansion: **0**

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu znajdującego się aktualnie na GitHub.

Post-push: verify sentinel SHAs, introducing commits `78f4350c8`, `1d2936c4c`, `550227883` on `development`, protected trio **3/3 PASS**, no production workaround in re-freeze commit.
