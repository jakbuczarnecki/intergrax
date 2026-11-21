# PT-WF — E2, E3, E4 (Campaign B-1) — status tor `development`

**SSOT plan:** [`governance/tickets/plans/PLAN_T-IGX-043-J.md`](../../../../../../governance/tickets/plans/PLAN_T-IGX-043-J.md) §7.1  
**Ticket:** `T-IGX-043-J` (`plan_approved`; G-09 product **done**)

## Podsumowanie

| Warstwa | Ścieżki (plan §3) | Node ids | Status fala 5 |
|---------|-------------------|----------|---------------|
| **E2** | `intergrax/runtime/execution/runtime.py` | `NT-WF-NEG-E2-01` — **TBD** po B-1 | **BLOCKED** |
| **E3** | `intergrax/compat/langchain/documents.py` | `NT-WF-NEG-E3-01` — **TBD** | **BLOCKED** |
| **E4** | `experiments/store.py`, `debug/router.py`, `experiments/workflow.py` | `NT-WF-NEG-E4-01` — **TBD** | **BLOCKED** |

**G-09** (osobny wiersz PT-WF): **DONE** na `development` — testy `NT-WF-G09-01..03` green (`test_harness_evaluation_wiring.py`).

## Dlaczego BLOCKED

1. MEP-AF2 §F E2–E4 są **Campaign B-1 conditional** — negatywne węzły wymagają wyniku B-1 zanim architect przypisze pełne ścieżki pytest (`tests/unit/runtime/execution/`, `tests/unit/compat/langchain/`, `tests/unit/experiments/`).
2. W planie §7.1 wszystkie trzy wiersze mają jawny stan **BLOCKED** i katalogi testów oznaczone **TBD after B-1**.
3. Fala 5 **nie** implementuje minimal slice — brak gotowych node ids; wdrożenie byłoby spekulatywne (poza mandate autonomii).

## Następny krok (poza tą falą)

Po B-1: plan rev 2 z node ids → red-first w `tests/unit/applications/test_wf_fail_closed.py` (`NT-WF-OWN-*`) lub dedykowanych pakietach per E2/E3/E4.

## STOP

Bez merge `rw-train`, bez FRZ promotion, bez twierdzenia SCOPE closure (RW-PLAN-AUDIT **BLOCKED**).