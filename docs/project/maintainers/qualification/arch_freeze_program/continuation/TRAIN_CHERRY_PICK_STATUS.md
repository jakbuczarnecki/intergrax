# development → `arch-freeze/rw-train` — kolejka slice (fala 5)

**Baza train:** `origin/arch-freeze/rw-train` @ `c443865e9`  
**Baza development:** HEAD (pre-push fala 5) @ `c6ce2a364` (pin fala 4) + commit fali 5  
**Bramka merge (OD-AF-AUTO-01 / KR-08):** `governance-fast` **musi być green** na proponowanym HEAD (wyjątek ACT-N11 tylko na product paths — tu **nie** stosowany jako bypass docs-only).

## Snapshot CI (2026-11-21)

| Workflow | Run | Stan fala 5 |
|----------|-----|-------------|
| Regression gate @ `development` push `7b1818f5f` | `44880866642` | `governance-fast` **in_progress**; `ci-smoke` **failure** |

**Werdykt fala 5:** PR/cherry-pick merge train — **STOP** (RC2-B2). Kolejka poniżej jest **gotowością**, nie merge.

## Proponowane slice (product + tests, topologicznie)

| Kolejność | Commit (skrót) | Slice | Uwagi |
|-----------|----------------|-------|-------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` (jeśli brak na train) | zależność FC-3 |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta względem train (#688) | seria części A–F + §8.6 |
| 3 | `aac4b083b` | PT-WF **G-09** | `NT-WF-G09-*` |
| 4 | `a667e6dd7`, `7b1818f5f` | trust slice 1–2 + guardrail e2e | `NT-TRUST-04/05`, FC-3 guardrail |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy bez product.

## Wykonanie (gdy RC2-B2 green)

1. Worktree od `origin/arch-freeze/rw-train`, branch `af-dev-train-slice5`.
2. Cherry-pick seria 1–4 (rozwiązać konflikty tylko w scope slice).
3. Targeted pytest: G-09 harness + `test_trust_fail_closed.py` + FC-3 bundle subset.
4. `gh pr create` → base `arch-freeze/rw-train`; merge dopiero przy green `governance-fast`.

## Stan fala 5

Branch **nie** utworzony (bramka CI). Operator: ponowić po zielonym `governance-fast` na `development` po push fali 5.