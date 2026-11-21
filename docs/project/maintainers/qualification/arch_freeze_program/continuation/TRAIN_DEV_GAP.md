# Train ↔ development gap (TOR C fala 6)

**Snapshot:** 2026-11-21  
**Disposition:** **QUEUED / STOP** — merge train i PR `development` → `rw-train` **nie** wykonane do green `governance-fast`.

## Piny

| Ref | SHA (40) | Krótki |
|-----|----------|--------|
| `origin/development` | `4889fc85907d3da3eb4bd75d1d0c8f2475e09ee1` | `4889fc859` |
| `origin/arch-freeze/rw-train` | `c443865e9129300a7bbb71abf65dee246f747794` | `c443865e9` |
| Merge-base | `70d73f0f08325a39c6b2fd0e3b00f4cab7e83033` | `70d73f0f0` |

## Liczniki divergence

| Kierunek | Commity | Znaczenie |
|----------|---------|-----------|
| `rw-train` … not in `development` | **205** | Historia train / EQO — poza merge mandate |
| `development` … not in `rw-train` | **185** | Remediacja product + docs na torze primary — **do cherry-pick** |

Product na `development` musi **cherry-pick/rebase** z trainu tam, gdzie slice już istnieje na trainie — nie merge `rw-train` bez mandatu operatora (`CHECKPOINT.md`, OD-AF-AUTO-01).

## Kolejka slice (z `TRAIN_CHERRY_PICK_STATUS.md`)

Narracja SSOT fala 5 — **bez zmiany semantyki** w tym pliku; poniżej kopia robocza + mapowanie landingu.

### Bramka CI

| Workflow | Stan fala 6 | Uwagi |
|----------|-------------|-------|
| `governance-fast` | **BLOCKED** | [`IIAC_GOVERNANCE_FAST_BLOCKED.md`](IIAC_GOVERNANCE_FAST_BLOCKED.md) — RB-2 re-pin, nie docs bypass |
| Regression gate (fala 5 pin) | `ci-smoke` remediated @ `2100a85f9`; governance-fast FAIL expected | run `44880866642` (historyczny) |

**Werdykt:** PR/cherry-pick merge train — **STOP** (RC2-B2 / OD-AF-AUTO-01). Kolejka = gotowość, nie merge.

### Proponowane slice (product + tests)

| Kolejność | Commit (skrót) | Slice | Ląduje na train? | Pozostaje dev-only? |
|-----------|----------------|-------|------------------|---------------------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` | Tak (jeśli brak na train) | — |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta (#688) | **Tak** (seria A–F) | tylko do zsynchronizowania |
| 3 | `aac4b083b` | PT-WF G-09 | **Tak** | — |
| 4 | `a667e6dd7`, `7b1818f5f` | trust slice 1–2 + guardrail e2e | **Tak** | trust slice 3 DEFER |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy qualification (`continuation/*`) — dokumentacja toru `development` **nie** idzie na train w tym batchu.

### Commity dev-only (klasa, nie pełna lista 185)

| Klasa | Przykłady na HEAD `development` | Train |
|-------|-----------------------------------|-------|
| EQO AUD wave-2 docs | `cd73ad88e`, `d57c5e62c`, … | **Nie** — osobny tor EQO |
| ARCH-FREEZE continuation fala 5 | `61516a7cf` (historyczny), RC-2 artifacts | **Nie** — cite only |
| Product remediacja FC-3/G-09/trust | `7b1818f5f`, `413fb9186`, … | **Tak** — w kolejce slice |

Pełna lista 185 commitów: `git log --oneline 70d73f0f0..4889fc859` (operator / następna fala train).

## Wykonanie (gdy governance-fast green)

1. Worktree od `origin/arch-freeze/rw-train`, branch np. `af-dev-train-slice6`.
2. Cherry-pick seria 1–4 (konflikty tylko w scope slice).
3. Targeted pytest: G-09 harness + `test_trust_fail_closed.py` + FC-3 bundle subset.
4. `gh pr create` → base `arch-freeze/rw-train`; merge **tylko** przy green `governance-fast`.

## Stan fala 6

Branch train slice **nie** utworzony. Operator: ponowić po green `governance-fast` na HEAD `development` po RB-2 / IIAC remediation (poza autonomią docs TOR C).