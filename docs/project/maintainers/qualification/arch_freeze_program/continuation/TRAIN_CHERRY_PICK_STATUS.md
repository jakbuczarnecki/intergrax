# development → `arch-freeze/rw-train` — kolejka slice (fala 5)

**Baza train:** `origin/arch-freeze/rw-train` @ `c443865e9`  
**Baza development (tor primary):** `origin/development` @ `e0d75042f` (pin po fali 4 trust; PR train slice osobno)  
**Bramka merge (OD-AF-AUTO-01 / KR-08):** `governance-fast` **musi być green** na HEAD PR przed merge (wyjątek ACT-N11 na product paths **nie** upoważnia merge przy czerwonym checku — merge **tylko** przy green).

## Snapshot CI / PR (2026-11-21)

| Element | Wartość |
|---------|---------|
| PR | **#699** — `af-dev-train-sync` → `arch-freeze/rw-train` |
| HEAD PR | `1cec68222751a128feab00e7b80063a09983296e` (`1cec68222`) |
| `governance-fast` | **FAILURE** — podpis **ACT-N11** (product paths po ostatnim audytowanym SHA) |
| Merge train | **STOP** — dopiero przy **green** `governance-fast` na HEAD `#699` |

**Werdykt:** cherry-pick **wykonany** na branchu; merge train — **STOP** (RC2-B2 / OD-AF-AUTO-01).

## Proponowane slice (product + tests, topologicznie)

| Kolejność | Commit (skrót) | Slice | Uwagi |
|-----------|----------------|-------|-------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` (jeśli brak na train) | zależność FC-3 |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta względem train (#688) | seria części A–F + §8.6 |
| 3 | `aac4b083b` | PT-WF **G-09** | `NT-WF-G09-*` |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust slice 1–2 + guardrail e2e + fix NexusLoop | na `#699` |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy bez product.

## Wykonanie

1. Worktree od `origin/arch-freeze/rw-train`, branch **`af-dev-train-sync`** — **done** (10 commitów `c443865e9..1cec68222`).
2. Cherry-pick seria FC-3 / G-09 / trust — **done**.
3. Targeted pytest: G-09 harness + `test_trust_fail_closed.py` + FC-3 bundle subset — operator / CI PR.
4. PR **#699** → base `arch-freeze/rw-train` — **open**; merge **tylko** przy green `governance-fast`.

## Stan fali 5

Branch **`af-dev-train-sync`** @ `1cec68222`; PR **#699** oczekuje na **green** `governance-fast`. Docs-only na `development` nie zmienia tej bramki.
