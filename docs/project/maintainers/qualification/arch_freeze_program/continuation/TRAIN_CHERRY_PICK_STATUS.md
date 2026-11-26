> **RETIRED 2026-11-26 (AD-AF-DEVTRAIN; unsigned draft).** Under decision OD-AF-CLOSE-D1, `development` is the train: no `arch-freeze/rw-train` cherry-pick or merge queue is open. The text below is a historical record, kept so that links stay valid. See [`../MEP-AF2_v2.1.md`](../MEP-AF2_v2.1.md) Appendix V6.

## Development tor — Runda 1–2 (D2=STOP)

Historical: before Runda 3 on development, **D2=STOP** blocked cherry-pick and train merge without green CI; see Runda 1–2 tables below on the train side for replay facts.

# development → `arch-freeze/rw-train` — kolejka slice (MEP W3 prep)

**Baza train (po landingu):** `origin/arch-freeze/rw-train` @ `dbaebfa8cd13a6ba8bfa6eef2cdf581216ddb3d2` (merge PR **#699**, 2026-11-21)  
**Baza train (przed slice):** `c443865e9129300a7bbb71abf65dee246f747794`  
**Baza development (docs pin Runda 1):** `origin/development` @ `c207e6ad3c01221e1e5867adaa21e0e5b038adc4` (`c207e6ad3`)  
**Baza development (docs pin Runda 2):** `origin/development` @ `4e911556b4669dd1f9bf7efc9e18b73ded508648` (`4e911556b`)  
**Baza development (docs pin Runda 3):** e69e1ff5819b98c316377bcadf749f61fe00abc2 (e69e1ff58)  
**Baza development (MEP W3 pin historyczny):** `d043a21c67cb57475a4f36a2baff7020718ec7cb` (`d043a21c6`)  
**Norma bramki (OD-AF-AUTO-01 / KR-08):** governance-fast **green** na HEAD PR przed merge (ACT-N11 na product paths **nie** jest bypassem). **Operator cherry-pick:** **D2=ALLOW** (2026-11-21). **Operator train merge:** bez green **STOP**; cherry-pick na branch **nie** oznacza green CI.

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

**Werdykt dokumentacyjny (MEP W3):** cherry-pick seria 1–4 **MERGED** na train (#699); **replay** `governance-fast` / `ci-smoke` / `rag-gate` na pinie train i RC2-B2 / RC2-B3 — **OPEN** (RC-2 **PARTIAL**, nie CLOSED). **Runda 3:** kontrolowany cherry-pick post-`1cec68222` @ `e69e1ff58` — **D2=ALLOW**; procedura replay: [`TRAIN_CI_REPLAY_PLAYBOOK.md`](TRAIN_CI_REPLAY_PLAYBOOK.md).

## Runda 1 — autonomia (D1=A, D2=STOP, SKIP cherry-pick)

| Parametr | Wartość |
|----------|---------|
| Worktree | `af-dev-train-ci-doc` |
| Zakres | Playbook replay + piny w tym pliku i `RC2_EXIT_CHECKLIST.md` |
| Cherry-pick kolejnych slice | **SKIP** — tabela „Proponowane slice” bez wykonania |
| Train merge | **STOP** — bez PR `development` → `rw-train` |

## Runda 2 — replay CI @ train pin (D1=A, D2=STOP, `gh`)

| Parametr | Wartość |
|----------|---------|
| Worktree | `af-dev-train-replay` |
| Źródło checków | PR **#699** `statusCheckRollup` (merge @ `dbaebfa8`; brak osobnych runów na `gh run list --branch arch-freeze/rw-train`) |
| `governance-fast` | run **44883349765** — **FAILURE** |
| `ci-smoke` | run **44883349765** — **FAILURE** |
| `rag-gate` | run **44883349736** — **FAILURE** |
| Cherry-pick / train merge | **SKIP** / **STOP** (**D2=STOP**) |
| RC2-B2 | **`[ ]` OPEN** — brak green; dowód w [`RC2_EXIT_CHECKLIST.md`](RC2_EXIT_CHECKLIST.md) |

## Runda 3 — cherry-pick post-#699 (D2=ALLOW, docs pin `e69e1ff58`)

| Parametr | Wartość |
|----------|---------|
| Worktree | `af-dev-train-sync` |
| Baza branch | `origin/arch-freeze/rw-train` @ `dbaebfa8` → tip cherry `90faeee5c` (slice 5) |
| Slice 1 backfill `60aa5282c` | **SKIP** — treść już na train (pusty cherry-pick @ `dbaebfa8`) |
| Slice 5 `2100a85f9` | **done** → `90faeee5c` |
| `governance-fast` / `ci-smoke` / `rag-gate` @ tip branch | **brak claimu green** — ostatni zapisany stan CI: run **#699** (jak Runda 2); replay na `90faeee5c` — **OPEN** |
| Train merge PR | **STOP** do green required checks |

### Kolejka slice 5+ (product + tests, @ pin `e69e1ff58`)

| Kolejność | Commit (skrót) | Slice | Stan |
|-----------|----------------|-------|------|
| 5 | `2100a85f9` | ci-smoke DF5 / G-09+FC-3 selectors | **done** (`90faeee5c`) |
| 6 | `4be976212` | rag-gate vector_store fixtures | **QUEUED** |
| 7 | `e69e1ff58` | ci-smoke G-09 tenant registry (tip pin) | **QUEUED** |
| 8 | `bab8114d8` | PT-WF G-09 harness follow-up | **QUEUED** |
| 9 | `f04b85443` | trust slice 3 (NT-TRUST-06) | **QUEUED** |
| 10 | `f08aba348` … `548f14b5a` | T-IGX-043-M wave-diff 1–3 | **QUEUED** |

**Wyłączone z train PR (Runda 3):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy bez product.

## Proponowane slice (product + tests, topologicznie)

| Kolejność | Commit (skrót) | Slice | Uwagi |
|-----------|----------------|-------|-------|
| 1 | `60aa5282c` | FC-2 `wiring_posture` (jeśli brak na train) | treść na train (#699); backfill **SKIP** |
| 2 | `d00d0e29b` … `413fb9186` | FC-3 delta względem train (#688) | seria części A–F + §8.6 |
| 3 | `aac4b083b` | PT-WF **G-09** | `NT-WF-G09-*` |
| 4 | `a667e6dd7`, `7b1818f5f`, `1cec68222` | trust slice 1–2 + guardrail e2e + fix NexusLoop | **na train** (#699) |

**Wyłączone z train PR (ta fala):** `agent-env/*`, `docs/eqo_closure/*`, czyste checkpointy bez product.

## Wykonanie (historyczne)

1. Worktree od `origin/arch-freeze/rw-train`, branch **`af-dev-train-sync`** — **done** (`c443865e9..1cec68222`).
2. Cherry-pick seria FC-3 / G-09 / trust — **done**.
3. Targeted pytest: G-09 harness + `test_trust_fail_closed.py` + FC-3 bundle subset — operator / CI PR.
4. PR **#699** → base `arch-freeze/rw-train` — **MERGED** @ `dbaebfa8` (2026-11-21); CI na merge: **red** (`governance-fast` / `ci-smoke` / `rag-gate`).

## Stan MEP W3 / Runda 1 (prep @ `c207e6ad3` docs)

Train merge pin **`dbaebfa8`**; branch `af-dev-train-sync` tip **`90faeee5c`** (+1 slice 5). RC-2 / green `governance-fast` (+ `ci-smoke`, `rag-gate`) — **Runda 2** @ `4e911556b` nadal jedyny zapisany run (**red**); tip `90faeee5c` **nie** wpisany jako green; **nie** RC2-B2 PASS.
