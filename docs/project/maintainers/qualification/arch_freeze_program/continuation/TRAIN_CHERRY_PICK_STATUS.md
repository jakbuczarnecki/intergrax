# development → `arch-freeze/rw-train` — kolejka slice (fala 5)

**Baza train (po landingu):** `origin/arch-freeze/rw-train` @ `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (merge PR **#699**, 2026-11-21)  
**Baza train (przed slice):** `c443865e9129300a7bbb71abf65dee246f747794`  
**Baza development (tor primary):** `origin/development` — slice product @ `1cec68222`; tor poszedł dalej (pełny cherry-pick **nie** wykonany)  
**Norma bramki (OD-AF-AUTO-01 / KR-08):** `governance-fast` **green** na HEAD PR przed merge (ACT-N11 na product paths **nie** jest bypassem).

## Snapshot CI / PR (2026-11-21)

| Element | Wartość |
|---------|---------|
| PR | **#699** — `af-dev-train-sync` → `arch-freeze/rw-train` — **MERGED** |
| Merge commit | `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (`dbaebfa8`) |
| Slice tip (cherry-pick) | `1cec68222751a128feab00e7b80063a09983296e` (`1cec68222`) |
| `governance-fast` @ merge | **FAILURE** — **ACT-N11** (product paths po ostatnim audytowanym SHA) |
| `ci-smoke` @ merge | **red** (zapisany stan checków PR) |
| `rag-gate` @ merge | **red** (zapisany stan checków PR) |

**Fakt operatora / automation (2026-11-21):** merge PR **#699** wykonany przy **czerwonym** `governance-fast`, `ci-smoke` i `rag-gate` na HEAD PR. To rejestr zdarzenia — **nie** oznacza spełnienia OD-AF-AUTO-01 ani zamknięcia RC-2.

**Werdykt dokumentacyjny:** cherry-pick slice **MERGED** na train; **replay** `governance-fast` na pinie train i RC2-B2 / RC2-B3 — **OPEN** (RC-2 **PARTIAL**, nie CLOSED).

## Proponowane slice (product + tests, topologicznie)

| Kolejność | Commit (skrót) | Slice | Uwagi |
|-----------|----------------|-------|-------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` (jeśli brak na train) | zależność FC-3 |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta względem train (#688) | seria części A–F + §8.6 |
| 3 | `aac4b083b` | PT-WF **G-09** | `NT-WF-G09-*` |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust slice 1–2 + guardrail e2e + fix NexusLoop | **na train** (#699) |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy bez product.

## Wykonanie

1. Worktree od `origin/arch-freeze/rw-train`, branch **`af-dev-train-sync`** — **done** (`c443865e9..1cec68222`).
2. Cherry-pick seria FC-3 / G-09 / trust — **done**.
3. Targeted pytest: G-09 harness + `test_trust_fail_closed.py` + FC-3 bundle subset — operator / CI PR.
4. PR **#699** → base `arch-freeze/rw-train` — **MERGED** @ `dbaebfa8` (2026-11-21); CI na merge: **red** (`governance-fast` / `ci-smoke` / `rag-gate`).

## Stan fali 5 (po merge)

Train HEAD **`dbaebfa8`**; slice 11 commitów na `arch-freeze/rw-train`. `development` **nie** zsynchronizowany w całości (~setki commitów poza slice — patrz `TRAIN_DEV_GAP.md`). RB-2 / green `governance-fast` na train pin — **następny krok**, nie claim w tej aktualizacji.
