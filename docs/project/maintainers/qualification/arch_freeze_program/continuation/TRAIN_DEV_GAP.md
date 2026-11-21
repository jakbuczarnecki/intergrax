# Train ↔ development gap (TOR C fala 6)

**Snapshot:** 2026-11-21  
**Disposition:** **IN FLIGHT / STOP merge** — PR **#699** (`af-dev-train-sync` → `rw-train`) **open**; merge **STOP** do green `governance-fast` (ACT-N11 **red** na HEAD `1cec68222`).

## Piny

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` | `e0d75042f2c2c72d2b8eb7df3a52e12509a00bac` | `e0d75042f` |
| `origin/arch-freeze/rw-train` | `c443865e9129300a7bbb71abf65dee246f747794` | `c443865e9` |
| Merge-base | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |
| HEAD PR train slice | `1cec68222751a128feab00e7b80063a09983296e` | `1cec68222` |

## Liczniki divergence

| Kierunek | Commity | Znaczenie |
|----------|---------|-----------|
| `rw-train` … not in `development` | **205** | Historia train / EQO — poza merge mandate |
| `development` … not in `rw-train` | **194** | Remediacja product + docs na torze primary — **do cherry-pick** |

Product na `development` musi **cherry-pick/rebase** z trainu tam, gdzie slice już istnieje na trainie — nie merge `rw-train` bez mandatu operatora (`CHECKPOINT.md`, OD-AF-AUTO-01).

## Kolejka slice (z `TRAIN_CHERRY_PICK_STATUS.md`)

SSOT narracji slice — poniżej stan landingu + PR **#699**.

### Bramka CI

| Workflow | Stan | Uwagi |
|----------|------|-------|
| `governance-fast` @ PR **#699** | **FAILURE (ACT-N11)** | product paths; merge **STOP** (OD-AF-AUTO-01) |
| `governance-fast` @ `development` | poza tym batch docs | [`IIIC_GOVERNANCE_FAST_BLOCKED.md`](IIIC_GOVERNANCE_FAST_BLOCKED.md) — RB-2 re-pin |

**Werdykt:** cherry-pick na `af-dev-train-sync` **done**; merge train — **STOP** do **green** `governance-fast` na `1cec68222`.

### Proponowane slice (product + tests)

| Kolejność | Commit (skrót) | Slice | Ląduje na train? | Pozostaje dev-only? |
|-----------|----------------|-------|------------------|---------------------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` | Tak (jeśli brak na train) | — |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta (#688) | **Tak** — na `#699` | — |
| 3 | `aac4b083b` / `e6b1d97f7` | PT-WF G-09 | **Tak** — na `#699` | — |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust + guardrail e2e | **Tak** — na `#699` | trust slice 3 DEFER na `development` |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy qualification (`continuation/*`) — ten commit docs **nie** idzie na train.

### Commity dev-only (klasa, nie pełna lista 194)

| Klasa | Przykłady na torze `development` | Train |
|-------|-----------------------------------|-------|
| EQO AUD wave-2 / matrix docs | `569254840`, `cd73ad88e`, … | **Nie** — osobny tor EQO |
| ARCH-FREEZE continuation | RC-2 / RC-3 prep (`continuation/*`) | **Nie** — cite only |
| Product remediacja FC-3/G-09/trust | już na `af-dev-train-sync` | **Tak** — PR **#699** |

Pełna lista 194 commitów: `git log --oneline 70d73f0f0..e0d75042f` (operator).

## Wykonanie (gdy governance-fast green na `#699`)

1. Branch `af-dev-train-sync` @ `1cec68222` — **gotowy**.
2. Cherry-pick seria 1–4 — **done** na branchu PR.
3. Targeted pytest — replay na HEAD PR po green CI.
4. Merge PR **#699** → `arch-freeze/rw-train` **tylko** przy **green** `governance-fast`.

## Stan fala 6

Worktree **`af-dev-train-sync`**; PR **#699** **open**, checks **red** (`governance-fast` ACT-N11). Operator: merge train po green; docs TOR C tylko odzwierciedlają stan (bez bypass).
