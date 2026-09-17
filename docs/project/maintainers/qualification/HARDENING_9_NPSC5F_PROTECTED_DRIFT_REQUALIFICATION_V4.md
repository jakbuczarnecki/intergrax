# HARDENING-9 — NPSC-5F Protected Drift Requalification V4

**Task:** `HARDENING_9_NPSC5F_PROTECTED_DRIFT_REQUALIFICATION_V4`  
**Status:** `REQUALIFIED / RE-FROZEN / PASS`  
**START_HEAD (`origin/development` at session open):** `a522cfcb630afe24bb368c86571233a9493b8b61`  
**FINAL_QUALIFIED_HEAD (`origin/development` at re-freeze):** `33576b80521dda7dfc0e5895c943f91dfebffa94`  
**HEAD movement after open:** `a4730b391` (docs), `33576b805` (enterprise-reliability) — **OUTSIDE** R1/Final protected production surfaces; included in re-freeze pin only.
**Prior sentinel (V3):** `6b3744f356c6e725fc124843f1da4a8fd5417120`

## Executive summary

After V3 @ `6b3744f3`, integrated `development` gained **OBS-RUNTIME-HISTORY-BOUNDS-R3** (`39265e263`) and a follow-on **platform-owned buffer validation** adjustment on the same file (`82d7989bf`). Custom `history_buffer` injection was removed; **PlatformOwnedRuntimeEventHistoryBuffer** is the sole retained-history owner; **RuntimeEventHistoryStrategy** is a contract observer over immutable bounded snapshots. R1 durable evidence path (`EvidencePersistencePort.append` before history append / delivery) is unchanged in ordering.

**Classification:** `QUALIFIED_OBS_EVOLUTION` (R1 + Final Evidence Plane production paths).  
**PROTECTED_BREAKING_DRIFT:** `0` · **UNCERTAIN:** `0`

**Production code in this requalification commit:** NO — sentinel baselines + qualification record only.

## Baseline re-freeze (V4)

| Sentinel | Old | New |
|----------|-----|-----|
| `R1_POST_R2_QUALIFIED_BASELINE_SHA` | `6b3744f356c6e725fc124843f1da4a8fd5417120` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |
| `NPSC5F_FINAL_EVIDENCE_PLANE_BASELINE_SHA` | `6b3744f356c6e725fc124843f1da4a8fd5417120` | `33576b80521dda7dfc0e5895c943f91dfebffa94` |
| `R2_POST_QUALIFIED_BASELINE_SHA` | `22c4793da4ba751fff6c93f780a7f6848650a5d9` | **UNCHANGED** |
| `NPSC5F_R3_H1_QUALIFIED_BASELINE_SHA` | `ad1a1e57fc70529aedcbfa27808fffdbfe5fdd14` | **UNCHANGED** (H1 integrated-head pin — out of V4 scope) |

## Qualified drift commits (`6b3744f3..33576b80`)

| Commit | Protected owner | Chronione ścieżki produkcyjne | Klasyfikacja | Wynik |
|--------|-----------------|-------------------------------|--------------|-------|
| `39265e263` | R1 + Final | `event_bus.py` | PROTECTED_NON_BREAKING_DRIFT | QUALIFIED — `history_strategy` DI; durable path preserved |
| `39265e263` | Final | `runtime_event_history.py`, `runtime_event_history_validation.py`, `runtime_event_metric_scope.py` | PROTECTED_NON_BREAKING_DRIFT | QUALIFIED — platform-owned retention + strategy observer |
| `39265e263` | Supporting | `intergrax/contracts/runtime_event_history.py` | SUPPORTING_CHANGE | Contract documents strategy; not Final BREAKING sentinel |
| `39265e263` | Outside | `nexus_loop.py`, `task_finisher.py` | OUTSIDE_PROTECTED_SURFACE | Final classifier `intergrax/runtime/nexus/` unrelated |
| `82d7989bf` | Final | `runtime_event_history_validation.py` | PROTECTED_NON_BREAKING_DRIFT | QUALIFIED — `PlatformOwnedRuntimeEventHistoryBuffer` composition probe |
| `a522cfcb6` | Outside | Nexus context only | OUTSIDE_PROTECTED_SURFACE | No R1/Final production drift |
| `a4730b391` | Outside | docs only | OUTSIDE_PROTECTED_SURFACE | — |
| `33576b805` | Outside | `provider_invocation_recovery.py` | OUTSIDE_PROTECTED_SURFACE | Enterprise reliability; not R1/Final |

Commits `5bf824c98` … `f17a6df80` (docs, enterprise-reliability, governance, memory, inspection elsewhere): **OUTSIDE_PROTECTED_SURFACE** for R1/Final ownership (no `event_bus` / Evidence Plane BREAKING paths).

## Protected path ledger

| Path | Introducing | Classification |
|------|-------------|----------------|
| `intergrax/runtime/events/event_bus.py` | `39265e263` | QUALIFIED_OBS_EVOLUTION |
| `intergrax/runtime/events/runtime_event_history.py` | `39265e263` | QUALIFIED_OBS_EVOLUTION |
| `intergrax/runtime/events/runtime_event_history_validation.py` | `39265e263`, `82d7989bf` | QUALIFIED_OBS_EVOLUTION |
| `intergrax/runtime/events/runtime_event_metric_scope.py` | `39265e263` | QUALIFIED_OBS_EVOLUTION (scoped counters; non-evidence) |
| `intergrax/contracts/runtime_event_history.py` | `39265e263` | SUPPORTING contract lift |

## Architecture invariants (@ FINAL_QUALIFIED_HEAD)

| Invariant | Wynik |
|-----------|-------|
| Durable evidence authority preserved | **PASS** |
| Commit-before-delivery semantics preserved | **PASS** (`_commit_durable_evidence` persists before sink/handlers) |
| Runtime history remains non-canonical | **PASS** (contract + OBSERVABILITY.md) |
| Platform owns bounded retention | **PASS** (`PlatformOwnedRuntimeEventHistoryBuffer`, injection removed) |
| Plugins operate through contracts | **PASS** (`RuntimeEventHistoryStrategy` protocol) |
| No layer-boundary violation | **PASS** |
| No bypass flow | **PASS** |
| No private API access | **PASS** (audit scope) |
| No `getattr` / `setattr` workaround | **PASS** (`event_bus.py`) |

**Observer failure:** `on_history_window` exceptions propagate after successful `EvidencePersistencePort.append` and platform deque update; they do not roll back durable evidence. Documented diagnostic observer role (OBSERVABILITY.md); fail-fast on misconfigured strategy.

## Tests / gates (V4 session)

See operator report section E for commands and exit codes.

## GitHub audit note

> Wprowadzone zmiany muszą zostać zaudytowane na podstawie kodu z GitHub przed uznaniem zadania za ostatecznie zamknięte.
