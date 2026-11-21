# Train ↔ development gap (MEP W3 — pin `d043a21c6`)

**Snapshot:** 2026-11-21 (worktree `af-dev-mep-w3`, fala MEP W3; serial merge W4–W8 na `development` **poza** pinem liczników poniżej)  
**Disposition:** **MERGED / replay OPEN (QUEUED / STOP)** — PR **#699** **MERGED** @ `dbaebfa8`; replay `governance-fast` na train pin **OPEN**. **Operator STOP:** bez green `governance-fast` — **brak** kolejnego cherry-pick / merge train.

## Piny (MEP W3)

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` (MEP W3) | `d043a21c67cb57475a4f36a2baff7020718ec7cb` | `d043a21c6` |
| TOR C snapshot (historyczny) | `e61e5136aa0617eea69b97ceedd8da38a4d0e796` | `e61e5136a` |
| `origin/arch-freeze/rw-train` | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| Merge-base dev ↔ train | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |
| Slice tip (cherry-pick na PR) | `1cec68222751a128feab00e7b80063a09983296e` | `1cec68222` |

## Liczniki divergence (od merge-base `70d73f0f0`)

| Kierunek | @ `d043a21c6` | @ `e61e5136a` (referencja) | Znaczenie |
|----------|---------------|---------------------------|-----------|
| `rw-train` … not in `development` | **217** | **217** | Historia train / EQO — poza merge mandate |
| `development` … not in `rw-train` | **0** | **286** | Remediacja na torze primary — cherry-pick / cite |

Polecenie SSOT: `git rev-list --count 70d73f0f0..<dev-pin>`; train-only: `git rev-list --count 70d73f0f0..dbaebfa8`.

Na pinie MEP W3 tor `development` **nie wyprzedza** merge-base — późniejsze commity remediacji (do `e61e5136a` i dalsze MEP W4–W8 na serial `af-dev-merge`) są **poza** tym pinem i nie zmieniają faktu merge #699 na trainie.

Product na `development` musi **cherry-pick/rebase** z trainu tam, gdzie slice już istnieje — **nie** pełny merge `rw-train` bez mandatu operatora (`CHECKPOINT.md`, OD-AF-AUTO-01).

## Kolejka slice (SSOT narracji)

[`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) — seria 1–4 **done**; merge **#699** **MERGED**. **Następny krok:** replay CI @ train HEAD — **nie** nowy slice bez green gate.

### Bramka CI (następny krok)

| Workflow | Stan | Uwagi |
|----------|------|-------|
| `governance-fast` @ merge **#699** / train HEAD `dbaebfa8` | **FAILURE (ACT-N11)** | RC2-B2 / replay — **OPEN**; product paths @ train HEAD |
| `ci-smoke` / `rag-gate` | **red** @ merge PR | zapisany stan checków merge |
| `governance-fast` @ `development` | **BLOCKED** | [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) — stale pin `969cbd964` vs drzewo (RB-2 re-pin poza autonomią docs) |

**Werdykt:** cherry-pick seria — **on train**; formalny green `governance-fast` @ train — **STOP** (nie RC-2 B2–B3 PASS).

### Proponowane slice (product + tests)

| Kolejność | Commit (skrót) | Slice | Ląduje na train? | Pozostaje dev-only? |
|-----------|----------------|-------|------------------|---------------------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` | Tak (jeśli brak na train) | — |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta (#688) | **Tak** — w merge **#699** | — |
| 3 | `aac4b083b` / `e6b1d97f7` | PT-WF G-09 | **Tak** | — |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust + guardrail e2e | **Tak** | trust slice 3 DEFER na `development` |

**Wyłączone z train PR:** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy `continuation/*` (w tym ten plik) — cite only na trainie.

### Commity dev-only (klasa, nie pełna lista @ `e61e5136a`)

| Klasa | Przykłady na torze `development` | Train |
|-------|-----------------------------------|-------|
| EQO AUD wave-2 / matrix docs | seria EQO post-merge-base | **Nie** |
| ARCH-FREEZE continuation | RC-2 / RC-3 / MEP prep (`continuation/*`) | **Nie** — cite only |
| Product remediacja poza slice | commity po `1cec68222` na `development` | **Częściowo** — tylko jeśli w kolejce cherry-pick |

## Wykonanie (następny operator)

1. **Replay** `governance-fast` (i ewentualnie `ci-smoke` / `rag-gate`) na pinie train `dbaebfa8`.
2. RB-2 re-pin IIAC poza autonomią docs — prerequisite dla zielonego ACT-N11 na `development`.
3. RC-2 exit B2–B3 — [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) (**PARTIAL**; B3 = merge fact, B2 = green replay).
4. Kolejne slice dev-only — nowa kolejka w `TRAIN_CHERRY_PICK_STATUS.md`, nie merge całego `development`.

## Stan MEP W3

Worktree **`af-dev-mep-w3`** @ `d043a21c6`; aktualizacja SSOT gap + cherry status + WP12 prep matrix — **bez** claim G2, FCS, green `governance-fast`, **bez** train merge w tej fali.
