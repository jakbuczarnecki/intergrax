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
---

# TRAIN CI replay — Runda 2 (D1=A, D2=STOP)

**Tor:** `arch-freeze/rw-train` @ merge **#699** — **tylko monitoring / dokumentacja**; **bez** merge train, **bez** cherry-pick (SKIP).  
**Worktree:** `af-dev-train-replay` · branch od `origin/development` @ `4e911556b4669dd1f9bf7efc9e18b73ded508648` (`4e911556b`).  
**Decyzje operatora:** **D1=A** · **D2=STOP** (bez kolejnego train merge do green wszystkich trzech jobów).

## Kroki wykonane (Runda 2, read-only `gh`)

1. **Pin train:** `git rev-parse origin/arch-freeze/rw-train` → `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (**OK**).
2. **Runy na branchu train @ HEAD:** `gh run list --branch arch-freeze/rw-train` → **brak** runów (merge commit nie uruchomił osobnego push workflow na branchu).
3. **Checki @ train pin (SSOT merge PR #699, HEAD slice `1cec68222`):** `gh pr view 699 --json statusCheckRollup` + commit status API @ `dbaebfa8` (`state: pending`, `total_count: 0` — brak osobnych statusów na merge SHA; cytujemy runy z PR **#699**).

| Job | Workflow | Run id | Conclusion @ 2026-11-21 |
|-----|----------|--------|-------------------------|
| `governance-fast` | Regression gate (`unit-tests.yml`) | [44883349765](https://github.com/jakbuczarnecki/intergrax/actions/runs/44883349765) | **FAILURE** (ACT-N11) |
| `ci-smoke` | Regression gate (`unit-tests.yml`) | [44883349765](https://github.com/jakbuczarnecki/intergrax/actions/runs/44883349765) | **FAILURE** |
| `rag-gate` | RAG guard (`rag-guard.yml`) | [44883349736](https://github.com/jakbuczarnecki/intergrax/actions/runs/44883349736) | **FAILURE** |

4. **Green gate:** **nie** — RC2-B2 pozostaje `[ ]` (aktualizacja dowodu w `RC2_EXIT_CHECKLIST.md`, **bez** `[x]`).
5. **STOP:** **D2=STOP** — brak PR `development` → `rw-train`, brak cherry-pick kolejnych slice.

## Werdykt playbook (Runda 2 @ docs `4e911556b`)

**OPEN (red)** — formalny replay z `gh` potwierdza **FAILURE** na wszystkich trzech wymaganych jobach @ train pin; RC-2 **PARTIAL**. **Do not** apply this row to train pin `e9bebe0ac` (merges **#811** / **#812**).

---

# TRAIN CI replay — Runda 3 (PT-Q02 re-audit, read-only `gh`)

**Tor:** `origin/arch-freeze/rw-train` @ `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a` (`e9bebe0ac`, merge PR **#812**, 2026-11-24).  
**Source:** `gh pr checks` + `gh pr view --json statusCheckRollup` at merge time (not prose from Rounds 1–2).  
**Session logs:** `.tmp/session/arch-freeze/pt-q02_gh_pr_checks_811.log`, `pt-q02_gh_pr_checks_812.log`.

| Job | PR / merge SHA | Run (job URL) | Conclusion @ 2026-11-24 |
|-----|----------------|---------------|-------------------------|
| `ci-smoke` | **#811** → `876af7e5d2670ec566e70142d37e57e8e3cfba53` | [45468072190 / job 138708234393](https://github.com/jakbuczarnecki/intergrax/actions/runs/45468072190/job/138708234393) | **SUCCESS** |
| `ci-smoke` | **#812** → `e9bebe0aca6f6e80d4f1aa7d52d51292133e8e4a` (train tip) | [45471125619 / job 138719037983](https://github.com/jakbuczarnecki/intergrax/actions/runs/45471125619/job/138719037983) | **SUCCESS** |
| `rag-gate` | **#811** → `876af7e5d2670ec566e70142d37e57e8e3cfba53` | [45468072184 / job 138708232459](https://github.com/jakbuczarnecki/intergrax/actions/runs/45468072184/job/138708232459) | **SUCCESS** |
| `rag-gate` | **#812** (HEAD `2c8556a23`) | — | **not executed** (`gh pr checks 812` — no job; path filter) |
| `governance-fast` | **#812** @ train tip | [45471125619 / job 138719037514](https://github.com/jakbuczarnecki/intergrax/actions/runs/45471125619/job/138719037514) | **FAILURE** (ACT-N11 / KR-08 — expected; no remediation in this wave) |

**Verdict @ `e9bebe0ac`:** `ci-smoke` **green**; `rag-gate` **green** on the last merge-chain run (**#811**), **not red** on tip; `governance-fast` **red** only under ACT-N11. RC2-B2 for historical pin `dbaebfa8` remains **OPEN** (red on #699); do **not** mark `[x]` without operator-mandated green `governance-fast`.

