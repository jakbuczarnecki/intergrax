# W6 prep dossier (requalification + campaigns + M3 entry) — **nie M3, nie W6 CLOSED**

**Banner:** Dokumentacja wejścia w falę **W6** MEP-AF2 (rekwalifikacja na drzewie po **FCS**). **Nie** oznacza M3, **nie** `FREEZE_CERTIFICATION_SHA`, **nie** EBH-7 pin, **nie** wykonania kampanii B/C ani werdyktów PT-Q05..Q12.

**Data:** 2026-11-21 (MEP W6, worktree `af-dev-mep-w6`)  
**Pin development:** `d043a21c67cb57475a4f36a2baff7020718ec7cb` (`d043a21c6`)  
**Źródło ladder:** [`MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) — rungi `FCS` → … → `M3`; W6 exit = **M3 recorded** (PT-Q11).

## Werdykt (prep only)

| Warstwa | Werdykt |
|---------|---------|
| MEP W6 dokumentacja prep | **IN PROGRESS** @ `d043a21c6` |
| Program W6 (kampanie + requal + pin) | **DRAFT / NOT STARTED** — wymaga **FCS** i domknięcia W5 |

## Twarde prerequisites (program — nie spełnione @ pin)

| Id | Prerequisite | Stan @ `d043a21c6` |
|----|--------------|---------------------|
| W6-P0 | Rung **FCS**; product tree ≡ G2 | **BLOCKED** — brak G2/FCS na torze |
| W6-P1 | W5 pre-pin (`PT-D09/D10`, manifests final, ECAP final, …) | **OPEN** |
| W6-P2 | Operator pre-pin packet (OD-AF-17 m.in.) | **OPEN** |
| W6-P3 | AR-57 / AR-08 authorizations dla kampanii i audytów | cite WP-12; **nie** wykonano |

## W6 nodes (WP-12) — mapa prep

| Bucket | WP-12 nodes | Checklist |
|--------|-------------|-----------|
| Campaign B omega | `PT-B1W`, `PT-B2W`, `PT-BA`, `PT-D13` | [`W6_CAMPAIGN_PREP_CHECKLIST.md`](W6_CAMPAIGN_PREP_CHECKLIST.md) |
| Campaign C | `PT-C01` | tamże §C |
| Family requal | `PT-Q05`..`Q07`, `Q08`, `Q09`, `Q10` (opc.), `Q12` | [`W6_REQUAL_PREP_CHECKLIST.md`](W6_REQUAL_PREP_CHECKLIST.md) |
| M3 pin (last W6 event) | `PT-Q11` (EBH-7 final + B6) | [`M3_PREP_CHECKLIST.md`](M3_PREP_CHECKLIST.md) — **nie pin** |

## Ticket scaffolds (governance — draft only)

| Ticket | Node scope | Status |
|--------|------------|--------|
| `T-IGX-052-F` | Campaign B omega prep (`PT-B1W`, `B2W`, `BA`, `D13`) | `draft` |
| `T-IGX-052-G` | Campaign C + family requal prep (`PT-C01`, `Q05`..`Q12`) | `draft` |
| `T-IGX-052-H` | M3 pin prep scaffold (`PT-Q11`); **no freeze** | `draft` |

Letters `T-IGX-052-A`..`E` remain reserved for W5 (`MEP-AF2_v2.1` Appendix Z / EDER-N-0004).

## Powiązane artefakty (tor development)

| Artefakt | Rola |
|----------|------|
| [`RC3_PREP_DOSSIER.md`](RC3_PREP_DOSSIER.md) | RC-3 entry prep (fala 6) |
| [`G2_CHECKLIST_DRAFT.md`](G2_CHECKLIST_DRAFT.md) | G2 draft — **nie PASS** |
| [`CHECKPOINT.md`](CHECKPOINT.md) | G4/G6 rows |
| [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) | OD-AF-AUTO-05 — brak fałszywych claimów |

Zgodnie z **OD-AF-AUTO-05**: żadnego języka M3 PASS, `LOCK`, `ARCH_FREEZE_CLOSED`, ani wykonanej kampanii B omega.