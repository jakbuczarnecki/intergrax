# PT-WF — E2, E3, E4 (Campaign B-1) — status tor `development`

**SSOT plan:** [`governance/tickets/plans/PLAN_T-IGX-043-J.md`](../../../../../../governance/tickets/plans/PLAN_T-IGX-043-J.md) §7.1  
**Ticket:** `T-IGX-043-J` (`plan_approved`; G-09 product **done**)

## Podsumowanie

| Warstwa | Ścieżki (plan §3) | Node ids | Status fala 5 |
|---------|-------------------|----------|---------------|
| **E2** | `intergrax/runtime/execution/runtime.py` | `NT-WF-NEG-E2-01` — **provisional anchor** (poniżej) | **BLOCKED** (brak potwierdzenia B-1) |
| **E3** | `intergrax/compat/langchain/documents.py` | `NT-WF-NEG-E3-01` — **provisional anchor** (poniżej) | **BLOCKED** |
| **E4** | `experiments/store.py`, `debug/router.py`, `experiments/workflow.py` | `NT-WF-NEG-E4-01` — **provisional anchor** (poniżej) | **BLOCKED** |

**G-09** (osobny wiersz PT-WF): **DONE** na `development` — testy `NT-WF-G09-01..03` green (`test_harness_evaluation_wiring.py`).

## wf-exec fala (operator UNBLOCK, pin `e69e1ff58`)

Plan **nie** zezwala na product slice E2–E4 bez wyniku Campaign B-1 (`PLAN_T-IGX-043-J` §8.2; prerequisite §2 `SCOPE` / PT-B01). Ta fala **tylko dokumentuje** kandydatów węzłów pytest zbieranych na pinie — **nie** zastępuje architect rev 2 ani nie oznacza slice green.

### Provisional anchors (do potwierdzenia po B-1)

| Node id | Slice | Proposed pytest node id (pin `e69e1ff58`) | Uzasadnienie |
|---------|-------|-------------------------------------------|--------------|
| NT-WF-NEG-E2-01 | E2 | *(brak negatywnego — TBD)*; regression anchor: `tests/unit/runtime/execution/test_execution_runtime.py::test_inference_root_runtime_binds_identity_authority_budget` | MEP §F E2 dotyczy ścieżki `runtime.py`; B-1 musi wskazać fail-closed violation zanim powstanie red-first negatyw |
| NT-WF-NEG-E3-01 | E3 | `tests/unit/compat/langchain/test_documents.py::test_from_langchain_rejects_missing_tenant` | fail-closed na seam `documents.py` (tenant w metadata transport) |
| NT-WF-NEG-E4-01 | E4 | `tests/unit/experiments/test_experiment_workflow.py::test_experiment_session_echo_workflow` | F-D5-07 (`PLAN_T-IGX-041-E`): governance identity przy `ExperimentSession.run` |

**E4 powiązane pliki (plan §3):** `intergrax/experiments/workflow.py` (primary), `intergrax/experiments/store.py`, `intergrax/debug/router.py` — fix cross-file dopiero po B-1 + plan rev 2.

## Dlaczego BLOCKED (product)

1. MEP-AF2 §F E2–E4 są **Campaign B-1 conditional** — negatywne węzły wymagają wyniku B-1 zanim architect przypisze pełne ścieżki pytest.
2. W planie §7.1 wiersze E2–E4 pozostają **BLOCKED**; prowisoryczne anchory powyżej to **nie** zamknięcie slice.
3. Owner-repair batch (`tests/unit/applications/test_wf_fail_closed.py`) nadal **not started** (SCOPE / FRZ owners).

## Następny krok (poza tą falą)

Po B-1: plan rev 2 z bindingiem `NT-WF-NEG-*` → red-first / green product na ścieżkach §3.

## STOP

Bez merge `rw-train`, bez FRZ promotion, bez twierdzenia SCOPE closure (RW-PLAN-AUDIT **BLOCKED**).