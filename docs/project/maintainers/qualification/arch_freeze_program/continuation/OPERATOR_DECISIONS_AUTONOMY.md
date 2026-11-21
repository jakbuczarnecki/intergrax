# ARCH-FREEZE — decyzje operatora (ciągła autonomia)

**Data:** 2026-11-21  
**Sesja:** full_auto, tor primary `development`, merge serial przez `af-dev-merge`  
**Plan SSOT (read-only):** upload `arch-freeze_continuation_9eb1c56f.plan-L1-L490-0.md`  
**Baza development (pin sesji):** `origin/development` @ `3497faa58` (FC3 A–F + §8.5/8.6)

## Tryb i tor

| Parametr | Wartość |
|----------|---------|
| Tryb | `full_auto` — brak pytań mid-flight |
| Tor primary | `development` — commity + merge serial `af-dev-merge` |
| Tor train | PR/cherry-pick gdy stabilne; **nie** merge `rw-train` bez mandatu |
| Worktrees | `af-dev-pt-fc3`, `af-dev-pt-wf`, `af-dev-rw-trust`, `af-dev-rc1-docs`, `af-dev-merge` |

## Decyzje wiążące (2026-11-21)

### OD-AF-AUTO-01 — KR-08 / governance-fast na train

**TAK** (wąsko): kolejne **product** PR na `arch-freeze/rw-train` przy podpisie błędu **ACT-N11** na ścieżkach produktu zmienionych po ostatnim audytowanym SHA.

- **Nie** dotyczy samych docs.
- **#688:** merge do train **tylko przy zielonym** `governance-fast` (nie przy czerwonym).
- Nie wyłącza checków, nie zmienia rulesetu.

### OD-AF-AUTO-02 — Regression 78 (FC-3)

**TAK** — pełny batch 78 węzłów z planu `T-IGX-043-C` / `regression_nodes.txt` na `development`, fala 1.

### OD-AF-AUTO-03 — T-IGX-043-J + T-IGX-051-C

**Approved** po weryfikacji digestu planu — implementacja product równolegle (worktrees G2/G3).

- Digest planu weryfikowany lokalnie w tej sesji; linia `Approved:` w ticketach po dopasowaniu SHA planu do pliku kanonicznego.

### OD-AF-AUTO-04 — T-IGX-051-A (TENANT)

**Approved** (OPERATOR UNBLOCK wave 2026-11-21) — implementacja product na `af-dev-051-tenant` od pinu `e69e1ff58`; bez FCS/G2/CLOSED (OD-AF-AUTO-05).

### OD-AF-AUTO-05 — FCS / G2 / CLOSED

**Nigdy nie ogłaszać** w sesji autonomii — wyłącznie dossier + **STOP** przy próbie.

### OD-AF-AUTO-06 — RW-PLAN-AUDIT

**BLOCKED** bez symulacji niezależnej sesji audytora (bez PASS w repo).

### OD-AF-AUTO-07 — T-IGX-043-M

**Fala 2** — równolegle z WF gdy FC3 stabilny; worktree `af-dev-043m` jeśli brak kolizji.

## Stan PR / CI (snapshot)

| PR | Branch | Stan | Uwagi |
|----|--------|------|-------|
| #688 | `af2/pt-fc3` → `arch-freeze/rw-train` | MERGED (historycznie) | Ostatni run: `governance-fast` **FAILURE** — przy ponownym merge **STOP** do zielonego checku (OD-AF-AUTO-01) |

## Out of scope / STOP

- `051-A`, `FCS`, freeze `G2`, `ARCH_FREEZE_CLOSED`, `REVIEW_PASS` planu kontynuacji.
- Merge train przy czerwonym `governance-fast` bez wyjątku ACT-N11 na product paths (tu: nie stosować jako bypass).

## Recovery

Przy timeout sesji: `CONTINUATION_AUTONOMY_CHECKPOINT.md` w tym katalogu + wskaźnik następnej fali.