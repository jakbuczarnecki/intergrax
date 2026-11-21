# W7 — PT-VBN prep dossier (**prep only, not independent VBN ACCEPT**)

**Data:** 2026-11-21 (fala MEP W7, worktree `af-dev-mep-w7`, base `d043a21c67cb57475a4f36a2baff7020718ec7cb`)  
**Węzeł WP-12:** `PT-VBN` (= `T-B7-VBN-INDEPENDENT-ACCEPT`)  
**Werdykt wykonania:** **DRAFT / NOT STARTED** — **FIND-VBN-PARTIAL** nadal BLOCKER (EDER-D-0010; A7-005). Brak 7 werdyktów per-artefakt @ M3.

## Aliasy i kolejność (MEP vs DAG)

| Id | Znaczenie |
|----|-----------|
| `PT-VBN` | Niezależna akceptacja dossier VBN @ evidence SHA = M3 |
| `T-AF2-VBN-DOSSIER` / `T-B7-VBN-HEAD-DOSSIER` | `ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md` — W5 **final** przed lockiem |
| `T-B7-ECAP-2C` | Pełna platforma C1–C5 @ M3; audytor ≠ PT-Q02 — **między PT-Q13 a PT-VBN** (MEP; reguły POST_REGEN) |

Twarde minimum DAG: `M3`, `PT-Q13` (+ `FCS`, `DS-5`). Pełna ścieżka MEP: `PT-Q13` → `T-B7-ECAP-2C` → `PT-VBN` → certifier chain → `ENTRY`.

## Stan dowodów @ pin kontynuacji

| Obszar | Stan | SSOT |
|--------|------|------|
| Dossier draft W2 | istnieje | [`ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md`](../../ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md) |
| Dossier final W5 | **OPEN** | ticket `T-IGX-052-C` (nie ten commit) |
| B2 partial @ baseline | FIND-VBN-PARTIAL | [`ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md`](../../ARCH_FREEZE_B2_VBN_CURRENT_HEAD_EVIDENCE.md) |
| RM499 M10.1–M10.7 | **OPEN** | [`RM499_ENTRY_MAP.md`](../RM499_ENTRY_MAP.md) |

## Checklist wykonania PT-VBN (przyszła sesja certifier)

| Krok | Wymaganie MEP §H / N5 |
|------|------------------------|
| VBN-1 | Nowa sesja **independent auditor** (bez kontekstu implementacji) |
| VBN-2 | 7 werdyktów per-artefakt (§2.1–2.7) @ SHA = M3 |
| VBN-3 | Niezależna ratyfikacja każdego fixed substrate §2.4 |
| VBN-4 | Replay 12 kontroli VBN z `ENT_AUDIT_X_SEMANTIC=1` |
| VBN-5 | Podpis §2.7: `prohibited_unresolved = 0`, `promised_unqualified = 0` |
| VBN-6 | Zamknięcie FIND-VBN-PARTIAL / EDER-D-0010 (nie w prep) |

Wejścia tylko (EBH-5, ENT VBN 12/12) — nie zastępują kroku VBN-4.

## Prerequisites — tabela prep

| Id | | |
|----|---|---|
| `PT-Q13` | ENTERPRISE FULL @ M3 | [ ] BLOCKED |
| `T-B7-ECAP-2C` | Residual qualification accept | [ ] BLOCKED |
| `T-B7-LOCK-FREEZE-SHA` | = M3 | [ ] BLOCKED |
| `T-AF2-VBN-DOSSIER` final | Przed lock / przed PT-VBN | [ ] W5 OPEN |

## Werdykt prep PT-VBN

| Warstwa | Werdykt |
|---------|---------|
| Prep dokumentacja W7 | **DRAFT** — scope i kolejność zapisane |
| PT-VBN / FIND-VBN-PARTIAL | **NOT STARTED** — brak niezależnej akceptacji |

Żadnego „VBN ACCEPTED”, „7/7 PASS” ani promocji FRZ w tym deliverable.