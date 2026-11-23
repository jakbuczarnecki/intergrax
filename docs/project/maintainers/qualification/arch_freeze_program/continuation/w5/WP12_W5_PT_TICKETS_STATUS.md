# WP-12 W5 — status ticketów PT-* (exec @ `e69e1ff58`)

**Banner:** Mapa lifecycle **tylko dokumentacja** — **nie** FCS, **nie** G2, **nie** EBH-7 pin, **nie** implementacja product path.  
**Pin:** `e69e1ff5819b98c316377bcadf749f61fe00abc2` (`e69e1ff58`)  
**Worktree:** `af-dev-w5-exec`  
**SSOT węzłów:** [`WP12_DAG.md`](../../WP12_DAG.md) sekcja W5; prep per node: `PT-*_PREP.md` w tym katalogu.

## Legenda statusu

| Status | Znaczenie @ ten pin |
|--------|---------------------|
| **IN_PROGRESS** | Trwa wyłącznie mapowanie / routing ticketów (ten tor exec); **nie** oznacza startu węzła produktowego |
| NOT STARTED | Brak mandatu + twardy przodek (zwykle **FCS**) |
| BLOCKED | Zależność operatora / audytu / ticket niespełniony |
| ON_DEV | Ticket W2 zamknięty na `development`; final W5 (**052-***) jeszcze **nie** scaffolded |
| ABSENT | Brak pliku `governance/tickets/T-IGX-052-*` dla finalu MEP |
| DRAFT | Scaffold W6 (`052-F`..`H`) — poza zakresem węzłów W5 PT-* |

## Tor exec W5

| Warstwa | Werdykt |
|---------|---------|
| Mapa statusów ticketów (ten plik + `PT-*_PREP.md`) | **IN PROGRESS** @ `e69e1ff58` |
| Wykonanie węzłów WP-12 W5 (product/docs cert) | **NOT STARTED** — **nie FCS** @ pin |

## MEP W5 finals (`T-IGX-052-A`..`E`) — cite only

| Plan node | Ticket (reserved) | Powiązanie PT-* | Lifecycle @ pin | Product W5 final |
|-----------|-------------------|-----------------|-----------------|------------------|
| T-AF2-ENT-BASELINE-PREFREEZE | `T-IGX-052-A` | PT-S06 (po S06) | **ABSENT** | NOT STARTED |
| T-AF2-MANIFESTS (G2 tree) | `T-IGX-052-B` | (poza dossier PT-*) | **ABSENT**; W2 [`T-IGX-050-G`](../../../../../../../governance/tickets/T-IGX-050-G.md) on tree | NOT STARTED |
| T-AF2-VBN-DOSSIER | `T-IGX-052-C` | PT-VBN @ W7 | **ABSENT**; W2 [`T-IGX-050-H`](../../../../../../../governance/tickets/T-IGX-050-H.md) | NOT STARTED |
| T-AF2-ECAP-REVISION | `T-IGX-052-D` | PT-D09/D10 / AUD-DOC | **ABSENT**; W2 [`T-IGX-050-I`](../../../../../../../governance/tickets/T-IGX-050-I.md) | NOT STARTED |
| T-AF2-ECAP-EXEC (DP-54) | `T-IGX-052-E` | — | **ABSENT** | NOT STARTED |

## Mapa PT-* → ticket / AR (W5 only)

| WP12 | Kind | Ticket / mandat | Lifecycle @ pin | Węzeł programowy | Train cherry-pick |
|------|------|-----------------|-----------------|------------------|-------------------|
| PT-Q04 | TEST | AR-34; ticket TBD pre-pin | NOT STARTED | NOT STARTED | **STOP** (D2) |
| PT-PB | TEST | Operator (DP-60 CONFLICT); ticket TBD | NOT STARTED | NOT STARTED | **STOP** |
| PT-D09 | DOC | AR-19 | NOT STARTED | NOT STARTED | **STOP** |
| PT-D10 | DOC | PT-FRZ-RR + AR | NOT STARTED | NOT STARTED | **STOP** |
| PT-AUD-DOC | AUD | AR-12-03; sesja niezależna | NOT STARTED | NOT STARTED | **STOP** |
| PT-D11 | DOC | render @ FCS trail | NOT STARTED | NOT STARTED | **STOP** |
| PT-S05 | SET | AR-49 (C9) | NOT STARTED | NOT STARTED | **STOP** |
| PT-S06 | SET | AR-52; final **052-A** po S06 | NOT STARTED | NOT STARTED | **STOP** |
| PT-AUD-ENFA | AUD | AR-25; po S03..S07 | NOT STARTED | NOT STARTED | **STOP** |

## Twarde prerequisites (fakt @ pin — bez claimu FCS)

| Id | Stan | Dowód |
|----|------|-------|
| FCS / G2 | **BLOCKED** | [`RC3_EXIT_PREP_CHECKLIST.md`](../RC3_EXIT_PREP_CHECKLIST.md) — J4/J5 |
| RC-3 exit prep | **IN PROGRESS** | Ten sam checklist @ `c207e6ad3` lineage |
| W2 draft finals | ON_DEV (050-G/H/I) | Pliki ticketów na `development` @ merge-base pin |

## STOP

Product paths bez `Approved:`, cherry-pick train, Stage 4/5 LIVE, jakikolwiek język **FCS osiągnięty** lub **G2 PASS**. Zgodnie z **OD-AF-AUTO-05**.