# WP12 W3 — PT ticket prep on `development` (MEP W3 @ `d043a21c6`)

**Banner:** Prep only — **nie** implementacja produktu, **nie** G2 / RW milestone PASS, **nie** merge train.  
**Pin:** `d043a21c67cb57475a4f36a2baff7020718ec7cb`  
**Worktree:** `af-dev-mep-w3`  
**SSOT węzłów:** [`WP12_DAG.md`](../WP12_DAG.md) sekcja W3; macierz wykonawcza: continuation plan A.4 (cite-only, bez edycji plan file).

## Legenda statusu

| Status | Znaczenie na pinie MEP W3 |
|--------|---------------------------|
| MERGED_TRAIN | Na `arch-freeze/rw-train`; requal @ PT-Q02 |
| ON_DEV | Ticket + commity na torze primary (poza pinem lub po rebase) |
| PLAN_OK | Plan + `Approved:` w repo; product **STOP** bez OD / SCOPE |
| DRAFT | Scaffold ticket (ten batch) |
| BLOCKED | Zależność operatora / audytu / inny ticket |

## Mapa PT-* → T-IGX (W3, actionable vs blocked)

| WP12 | Ticket | Lifecycle @ prep | Następny krok (tor dev) | Train cherry-pick |
|------|--------|------------------|-------------------------|-------------------|
| PT-WA / PT-WB | 043-G, 043-H (+051-I train) | MERGED_TRAIN | Requal only | **STOP** |
| PT-FC4 | 043-D | MERGED_TRAIN | Requal | **STOP** |
| PT-FC2 | 043-B (+B3/B4 plan) | MERGED_TRAIN | B3/B4 `draft` plans — product po 043-B | **STOP** |
| PT-FC2-E | 043-E | MERGED_TRAIN | Część 2 sandbox wiring | **STOP** |
| PT-FC1 | 043-A | plan_approved | Review #680 / digest — nie nowa implementacja | **STOP** |
| PT-FC3 | 043-C | plan_approved | Product po FC-2 posture + green gate discipline | **STOP** |
| PT-WD | 043-I (+043-M) | 043-I train; 043-M in_progress | 043-M Part E **BLOCKED** (RW-PLAN) | **STOP** |
| PT-WF | 043-J | plan_approved / partial | G-09 done off-pin; E2–E4 **BLOCKED** B-1 | **STOP** |
| PT-WF follow-on | **043-N** | **plan_pending** (hardened R1) | Po merge FC-1 authority (#680); plan digest recorded | **STOP** |
| PT-E05 | 044-F | MERGED_TRAIN | IIAC requal @ Q02 | **STOP** |
| T-AF2-RW-TRUST | 051-C | implementing / merged off-pin | Trust slice merged fala 5 @ ref `e61e5136a` | **STOP** |
| T-AF2-RW-SUPPLY A/B | **051-N** | **plan_pending** (hardened R1) | Import review @ `d5d54710`; plan `PLAN_T-IGX-051-N.md` on `development` | **STOP** |
| T-AF2-RW-TENANT | **051-A** | **in_progress** (Part A @ `af-dev-051-tenant`) | OD-AF-AUTO-04 **Approved** 2026-11-21 UNBLOCK; plan `PLAN_T-IGX-051-A.md` | product wave |
| T-AF2-RW-GATES | 051-F | brak ticketu | Alloc `T-IGX-051-F` — poza tym batch | — |
| T-AF2-RW-TYP | 051-G | brak ticketu | Po TYP-MEASURE / OD-AF-11 | — |
| PT-T01 | brak | — | Alloc ticket; NT-01..16 red-first | — |
| PT-D08 | brak | — | 2 product-path docs | — |
| RW (milestone) | — | **OPEN** | Wszystkie twarde PT-W* + PT-FC* + PT-T01 | **STOP** |

## Serializacja (`harness_host_runtime.py`)

1. **043-A** (FC-1) review / merge #680  
2. **043-N** (FC-1 residual)  
3. **043-J** WF (G-09; E2–E4 warunkowe)  
4. **051-A** TENANT — tylko po mandacie operatora i bez kolizji z powyższymi

## Scaffold dostarczony w MEP W3

| Plik | WP12 node |
|------|-----------|
| `governance/tickets/T-IGX-043-N.md` | PT-FC1 follow-on |
| `governance/tickets/T-IGX-051-N.md` | T-AF2-RW-SUPPLY |
| `governance/tickets/T-IGX-051-A.md` | T-AF2-RW-TENANT |

**Runda 1 (autonomia @ `c207e6ad3`, fala TICKET-SCAFFOLD):** utwardzono **043-N** i **051-N** (`plan_pending`, plany kanoniczne + digest); **051-A** pominięty (D5 BLOCKED).

## STOP (fala MEP W3)

Cherry-pick / train merge, product paths bez ticketu + `Approved:`, TENANT bez OD-AF-AUTO-04/05, claim G2/FCS/RC-2 CLOSED.