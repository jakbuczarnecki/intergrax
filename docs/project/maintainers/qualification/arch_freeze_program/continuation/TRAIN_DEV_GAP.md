# Train ↔ development gap (TOR C — po merge #699; fala 6)

**Snapshot:** 2026-11-21  
**Disposition:** **MERGED / replay OPEN (QUEUED / STOP)** — PR **#699** (`af-dev-train-sync` → `rw-train`) **MERGED** @ `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2`; replay `governance-fast` (oraz `ci-smoke` / `rag-gate`) na train pin pozostaje **OPEN** (ACT-N11 **red** na merge — patrz `TRAIN_CHERRY_PICK_STATUS.md`). Merge mandate OD-AF-AUTO-01 **nie** spełniony na zielonym checku.

## Piny

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` (fetch) | `f551abb495ad2316a4b5ca0a90dd9aef2a4f28d9` | `f551abb49` |
| Pin sesji G4 / fala 6 parallel | `dc083f6eadf0f7dbe501fbcf93659e2dd158b825` | `dc083f6ea` |
| `origin/arch-freeze/rw-train` | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` | `dbaebfa8` |
| Merge-base dev ↔ train | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |
| Slice tip (cherry-pick na PR) | `1cec68222751a128feab00e7b80063a09983296e` | `1cec68222` |

## Liczniki divergence (od merge-base `70d73f0f0`)

| Kierunek | Commity | Znaczenie |
|----------|---------|-----------|
| `rw-train` … not in `development` | **217** | Historia train / EQO — poza merge mandate |
| `development` … not in `rw-train` | **241** | Remediacja product + docs na torze primary — **do cherry-pick** / cite |

Polecenie SSOT: `git log --oneline 70d73f0f0..dc083f6ea` (dev-only); `git log --oneline 70d73f0f0..dbaebfa8` (train-only).

Product na `development` musi **cherry-pick/rebase** z trainu tam, gdzie slice już istnieje na trainie — **nie** pełny merge `rw-train` bez mandatu operatora (`CHECKPOINT.md`, OD-AF-AUTO-01).

## Kolejka slice (SSOT narracji)

[`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) — cherry-pick seria 1–4 **done**; merge **#699** **MERGED** 2026-11-21. Poniżej landing + bramka CI po merge **#699**.

### Bramka CI (następny krok)

| Workflow | Stan | Uwagi |
|----------|------|-------|
| `governance-fast` @ merge **#699** / train HEAD `dbaebfa8` | **FAILURE (ACT-N11)** | RC2-B2 / replay — **OPEN**; product paths @ train HEAD |
| `ci-smoke` / `rag-gate` | **red** @ merge PR | zapisany stan checków merge |
| `governance-fast` @ `development` | **BLOCKED** | [`IIIC_GOVERNANCE_FAST_BLOCKED.md`](IIIC_GOVERNANCE_FAST_BLOCKED.md) — stale pin `969cbd964` vs drzewo (RB-2 re-pin poza autonomią docs) |

**Werdykt:** cherry-pick seria 1–4 — **on train**; formalny green `governance-fast` @ train — **STOP** (nie RC-2 B2–B3 PASS).

### Proponowane slice (product + tests)

| Kolejność | Commit (skrót) | Slice | Ląduje na train? | Pozostaje dev-only? |
|-----------|----------------|-------|------------------|---------------------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` | Tak (jeśli brak na train) | — |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta (#688) | **Tak** — w merge **#699** | — |
| 3 | `aac4b083b` / `e6b1d97f7` | PT-WF G-09 | **Tak** | — |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust + guardrail e2e | **Tak** | trust slice 3 DEFER na `development` |

**Wyłączone z train PR:** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy `continuation/*` (w tym ten plik) — cite only na trainie.

### Commity dev-only (klasa, nie pełna lista 241)

| Klasa | Przykłady na torze `development` | Train |
|-------|-----------------------------------|-------|
| EQO AUD wave-2 / matrix docs | seria EQO post-merge-base | **Nie** |
| ARCH-FREEZE continuation | RC-2 / RC-3 prep (`continuation/*`) | **Nie** — cite only |
| Product remediacja poza slice | commity po `1cec68222` na `development` | **Częściowo** — tylko jeśli w kolejce cherry-pick |

## Wykonanie (następny operator)

1. **Replay** `governance-fast` (i ewentualnie `ci-smoke` / `rag-gate`) na pinie train `dbaebfa8`.
2. RB-2 re-pin IIIC poza autonomią docs — prerequisite dla zielonego ACT-N11 na `development`.
3. RC-2 exit B2–B3 — [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) (**PARTIAL**; B3 = merge fact, B2 = green replay).
4. Kolejne slice dev-only — nowa kolejka w `TRAIN_CHERRY_PICK_STATUS.md`, nie merge całego `development`.

## Stan fala 6 (TOR C)

Worktree **`af-dev-rc3-prep`** @ `dc083f6ea`; docs odzwierciedlają merge **#699** i luki 241/217 — **bez** claim G2, FCS ani green governance-fast.

## Stan G4 (fala 5–6; RC-1)

Worktree **`af-dev-rc1-docs`** @ pin sesji `dc083f6ea`; dokumentacja odzwierciedla merge **#699** bez claim RC-2 CLOSED.
