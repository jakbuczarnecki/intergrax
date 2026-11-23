# WP12 W5 — PT and MEP ticket prep on `development` (MEP W5 @ `e69e1ff58`)

**Banner:** Execution batch **governance + scoped product** — **nie** FCS, **nie** EBH-7 pin, **nie** G2 PASS, **nie** FC-3 bundle (260) / trust bundle (80).  
**Pin:** `e69e1ff5819b98c316377bcadf749f61fe00abc2` (`e69e1ff58`)  
**Worktree:** `af-dev-w5-product`  
**SSOT węzłów:** [`WP12_DAG.md`](../WP12_DAG.md) sekcja W5; prep: [`W5_PREP_DOSSIER.md`](W5_PREP_DOSSIER.md).

## Legenda statusu

| Status | Znaczenie @ `e69e1ff58` |
|--------|-------------------------|
| DRAFT | Ticket + plan scaffold (ten batch) |
| BLOCKED | Twardy przodek (FCS, DS-*, audyt) |
| PLAN_OK | `Approved:` + implementacja dozwolona |
| IN_PROGRESS | Aktywny worktree (product po FCS / AR) |

## MEP W5 finals (`T-IGX-052-A`..`E`)

| MEP node | Ticket | W2 draft | Lifecycle @ batch | Następny krok |
|----------|--------|----------|-------------------|---------------|
| T-AF2-ENT-BASELINE-PREFREEZE | **052-A** | ENT chain @ G2 | `draft` | Po FCS + PT-S06; product empty @ G2 |
| T-AF2-MANIFESTS (final) | **052-B** | T-IGX-050-G drafts | `draft` | Regenerate manifests z drzewa G2; drift gate |
| T-AF2-VBN-DOSSIER (final) | **052-C** | T-IGX-050-H | `draft` | Final dossier + SQLite annex; **nie** VBN-ACCEPT |
| T-AF2-ECAP-REVISION (final) | **052-D** | T-IGX-050-I proposal | `draft` | ECAP / checklist / roadmap — Operator paths |
| T-AF2-ECAP-EXEC (DP-54) | **052-E** | brak | `draft` | **BLOCKED** external; scaffold only |

## Mapa PT-* → T-IGX (W5)

| WP12 | Ticket | Kind | Lifecycle @ batch | Hard blocker | Test contract (ticket) |
|------|--------|------|-------------------|--------------|-------------------------|
| PT-Q04 | **053-A** | TEST | `draft` | FCS, DS-6, AR-34 | Scoped qual gates only after `Approved:`; **no** FC-3 260 |
| PT-PB | **053-C** | TEST | `draft` | FCS; DP-60 Operator | RQ-1 scoped; gate-meaning **not** delegated |
| PT-D09 | **053-D** | DOCS | `draft` | FCS; PT-D03/04 | `check_docs_domain_pairs`; **no** certification claims |
| PT-D10 | **053-E** | DOCS | `draft` | FCS; PT-FRZ-RR | docs pair + PT-AUD-DOC handoff |
| PT-AUD-DOC | **053-F** | AUD | `draft` | PT-D09, PT-D10 | Independent session only; **no** implementer PASS |
| PT-D11 | **053-G** | DOCS | `draft` | D09, D10, AUD-DOC | Generator `render_ebh7_f10_normative_matrix.py` |
| PT-S05 | **053-H** | SET | `draft` | PT-S04 | `igx_gov ticket` push-fast |
| PT-S06 | **053-I** | SET | `draft` | PT-S04, PT-G01 | `igx_gov ticket` push-fast |
| PT-AUD-ENFA | **053-J** | AUD | `draft` | PT-S03..S07, AUD-ENF12 | G6 session; prep cites only |

**Note:** `T-IGX-053-B` reserved (AGENT-OPS-X W1). Letters **052-F**..**H** = MEP W6 prep (osobny worktree).

## MEP poza batch (cite only)

| Node | Ticket | Status |
|------|--------|--------|
| T-B7-PFP-ACTIVATE | unallocated | Post-FCS policy docs; **nie** ten batch |

## Scaffold dostarczony (MEP W5 execution @ `e69e1ff58`)

| Artefakt | Pliki |
|----------|-------|
| Macierz | ten plik |
| MEP finals | `governance/tickets/T-IGX-052-A`..`E.md` + `plans/PLAN_T-IGX-052-*.md` |
| PT rows | `governance/tickets/T-IGX-053-A,C`..`J.md` + plans |

## STOP

FCS claim, pin EBH-7, M3, `ARCH_FREEZE CLOSED`, heavy regression (FC-3 **260**, trust **80**, dev-ci-rag bundle), product bez `Approved:`.