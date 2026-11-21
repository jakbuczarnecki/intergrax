# TRAIN CI replay playbook — Runda 1 (D1=A, D2=STOP)

**Tor:** `arch-freeze/rw-train` — **tylko monitoring / dokumentacja**; **bez** merge train, **bez** cherry-pick (SKIP).  
**Decyzje operatora:** **D1=A** (po green CI — replay + dokumentacja RC-2 B2) · **D2=STOP** (kolejny merge train zabroniony do green wszystkich required checks).  
**Docs pin:** `origin/development` @ `c207e6ad3c01221e1e5867adaa21e0e5b038adc4` (`c207e6ad3`)  
**Train pin (merge #699):** `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (`dbaebfa8`)

## SSOT i powiązania

| Artefakt | Rola |
|----------|------|
| [`TRAIN_CHERRY_PICK_STATUS.md`](TRAIN_CHERRY_PICK_STATUS.md) | Kolejka slice, snapshot CI @ merge #699 |
| [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) | RC2-B2 pozostaje `[ ]` do **green** replay |
| [`TRAIN_DEV_GAP.md`](TRAIN_DEV_GAP.md) | Gap development ↔ train |
| [`OPERATOR_DECISIONS_AUTONOMY.md`](OPERATOR_DECISIONS_AUTONOMY.md) | OD-AF-AUTO-01 / KR-08 |

## Wymagane checki @ train HEAD `dbaebfa8`

| Job | Workflow (repo) | Stan @ merge PR #699 | Runda 1 |
|-----|-----------------|----------------------|---------|
| `governance-fast` | `.github/workflows/unit-tests.yml` | **FAILURE (ACT-N11)** | Replay **OPEN** |
| `ci-smoke` | `.github/workflows/unit-tests.yml` | **red** | Replay **OPEN** |
| `rag-gate` | `.github/workflows/rag-guard.yml` | **red** | Replay **OPEN** |

**Nie** traktować merge #699 przy czerwonym CI jako spełnienia OD-AF-AUTO-01 ani RC2-B2 DONE.

## Kroki operatora — replay (Runda 1, bez mutacji train)

1. **Potwierdź pin train:**  
   `git rev-parse origin/arch-freeze/rw-train` → oczekiwane `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2`.
2. **Lista runów na branchu train** (read-only):  
   `gh run list --branch arch-freeze/rw-train --limit 20`  
   Filtruj workflow **Regression gate** (`unit-tests.yml`) oraz **rag-guard**.
3. **Dla każdego jobu** (`governance-fast`, `ci-smoke`, `rag-gate`) na HEAD = `dbaebfa8`:  
   - `gh run view <run-id> --log-failed` (diagnoza; opcjonalnie sesja `ci-investigator` na **jeden** znany fail ACT-N11).  
   - Zapisz run id + conclusion w notatce operatora (poza repo lub w kolejnym pinie `TRAIN_CHERRY_PICK_STATUS` **tylko** po replay, bez claimu PASS).
4. **Przypadek green (wszystkie trzy joby success @ `dbaebfa8`):**  
   - Zaktualizuj `RC2_EXIT_CHECKLIST.md` RC2-B2 → `[x]` **tylko** z cytatem run id.  
   - Sygnał operatora: start **Rundy 2** (`TRAIN-REPLAY` / opcjonalnie CHERRY tylko przy **D1=B** i **D2=ALLOW**).  
   - **Ta fala (Runda 1):** nie wykonuj kroku 4 — CI nadal red @ pinie historycznym merge.
5. **Przypadek nadal red:**  
   - Pozostaw RC2-B2 `[ ]`, werdykt RC-2 **PARTIAL**.  
   - **STOP** cherry-pick i train merge (**D2=STOP**).  
   - Kolejna product sync — wyłącznie Runda 2 po explicit green.

## Explicit STOP (Runda 1)

- **NO** nowy PR `development` → `rw-train`.  
- **NO** cherry-pick kolejnych slice z tabeli w `TRAIN_CHERRY_PICK_STATUS.md` (SKIP).  
- **NO** oznaczenie RC-2 CLOSED, G2, FCS, ARCH_FREEZE CLOSED.

## Werdykt playbook (Runda 1 @ docs `c207e6ad3`)

**OPEN** — procedura replay udokumentowana; formalny green @ train pin **nie** zadeklarowany w tej fali.