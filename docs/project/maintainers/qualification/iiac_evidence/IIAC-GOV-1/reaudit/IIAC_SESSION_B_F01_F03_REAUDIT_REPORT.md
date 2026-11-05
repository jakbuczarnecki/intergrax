# IIAC Session B — niezależny re-audyt korekty F01–F03

**Rola:** `independent-quality-auditor`  
**Etap:** `IIAC-GOV-1`  
**Wersja polityki:** `IIAC-2026-11-05.1`  
**Gałąź:** `development` (`refs/heads/development`)  
**HEAD / baseline re-audytu:** `5d6b44b05f90ecf420d45d01659ea64565b52848`  
**Drzewo robocze:** czyste (`git status --porcelain` puste po testach)  
**Status polityki:** `NOT_IN_FORCE`  
**Werdykt:** **PASS — F01–F03 INDEPENDENTLY VERIFIED**  
**Pole `verdict` w manifeście:** `PASS`  
**Aktywacja `IN_FORCE`:** **nie wydana** (sesja audytora nie jest sesją certyfikującą)

Historyczny werdykt Session B pozostaje bez zmian w `.tmp/session/IIAC-GOV-B/`: **PARTIAL / EXACT BLOCKERS** @ `2ae1014116d353bc7742985e62f2bccc2bfac798`.

## Odpowiedź

Korekta F01–F03 usuwa ustalenia pierwotnego Session B i nie osłabia kontroli G1/G3. Polityka IIAC nie jest zdatna do aktywacji w tej sesji: brak linii operatora w `IIAC_OD_ACTIVATION_*.md`, brak decyzji certyfikującej, a linia podpisu handoff jest **PENDING**.

## Granice SHA

| Etykieta | SHA | Przodek HEAD | Commit |
|---|---|---|---|
| Historyczna implementacja governance | `2ae1014116d353bc7742985e62f2bccc2bfac798` | tak | Implement IIAC governance reform Session A (G0-G3). |
| Źródło inwentarza (korekta kodu) | `afe9e8abeabdbab410bf562ebed6f4a309a91057` | tak | Fix IIAC G0 inventory completeness (Session B F01–F03). |
| Publikacja macierzy G0 v2 | `9bd8559b36948aefd8134232b89b7738d6dd5bbc` | tak | Publish IIAC G0 matrix v2 @ inventory source afe9e8abe. |
| Zapis SHA w tickecie | `7d6061e48c6f84bbf1b05e34a28d4fd495602ae8` | tak | Record CORRECTIVE_IMPLEMENTATION_SHA for IIAC G0 F01 batch. |
| Baseline re-audytu = HEAD | `5d6b44b05f90ecf420d45d01659ea64565b52848` | tak (równy HEAD) | Align CORRECTIVE_IMPLEMENTATION_SHA with pushed development HEAD. |

Łańcuch `merge-base --is-ancestor`: `2ae10141` → `afe9e8ab` → `9bd8559b` → `7d6061e` → `5d6b44b`. Checkout pinów nie był wykonywany.

`7d6061e48` nie jest HEAD. Różnica `7d6061e48..5d6b44b05` obejmuje wyłącznie:

- `governance/tickets/T-IGX-016-A.md`
- `docs/project/maintainers/qualification/IIAC_GOVERNANCE_IMPLEMENTATION_SESSION_A_TERMINAL_REPORT.md`

Ticket nadal zapisuje `CORRECTIVE_IMPLEMENTATION_SHA=7d6061e48` (rodzic commita, który tę wartość wpisał). Blob macierzy na `9bd8559b` i na HEAD jest ten sam: `438ee5b97ad4dfa0cdcb2ef718216d63c9390908`. Blob `scripts/governance/iiac_g0_inventory.py` na `afe9e8ab` i na HEAD: `b1ca559fafd03fb4c09e7e7ea4f346758a4e5ea4` (równy `generator_sha` w JSON).

`inventory_publication_sha` w JSON macierzy jest `null` (generator nie wpisuje self-ref). Publikacja `9bd8559b` jest potwierdzona historią gita i identycznością bloba, nie polem wewnątrz macierzy.

## Ustalenia pierwotne (zachowane)

| ID | Sev. historyczna | Przyczyna | Ocena re-audytu |
|---|---|---|---|
| **IIAC-GOV-B-F01** | BLOCKER | Macierz G0 z `eab65e0…`, bez regeneracji po artefaktach Session A; brak protokołu IIAC, `verify_iiac_*`, `T-IGX-016-A.md` | **zamknięte** — pełna enumeracja @ `afe9e8ab`, publikacja @ `9bd8559b` |
| **IIAC-GOV-B-F02** | LOW | Kontrakt testów używał globu `test_iiac_*.py` | **zamknięte** — jawne ścieżki w tickecie i instrukcji Session B; pytest w PowerShell wykonał oba moduły |
| **IIAC-GOV-B-F03** | INFO | Był tylko szablon handoff, bez wypełnionego rekordu | **rekord jest**; linia operatora **PENDING** — warunek przed aktywacją, nie bloker tej weryfikacji |

## F01 — niezależna enumeracja

Weryfikator kompletności importuje `enumerate_governed_paths_at_ref` z tego samego modułu co generator. Ta sesja **nie** oparła werdyktu na tym współdzieleniu.

Niezależny skrypt `.tmp/session/IIAC-GOV-B-REAUDIT/independent_f01_enum.py` (bez importu `iiac_g0_inventory`) zrobił `git ls-tree -r --name-only` i zastosował opublikowane reguły sufiksów oraz wykluczeń.

| Źródło | Ścieżki governowane |
|---|---|
| `afe9e8ab` (inventory source) | **17510** |
| `9bd8559b` | **17510** |
| `7d6061e` | **17510** |
| `5d6b44b` / HEAD | **17510** |
| `2ae10141` (historia) | **17505** |

Macierz `iiac-authority-impact.v2`:

| Pole | Deklaracja | Pomiar niezależny |
|---|---|---|
| `covered_path_count` | 17510 | 17510 unikalnych `path` |
| `entry_count` | 18388 | 18388 obiektów |
| `governed_path_count` | 17510 | 17510 |
| Brakujące względem źródła | — | **0** |
| Nadmiarowe względem źródła | — | **0** |
| Zduplikowane pary (kategoria, ścieżka) | — | **0** |
| SHA-256 pliku macierzy | — | `c76fe01e94773d2c8893dc347e33331060b67bd9436a3da102dc257e07cc6f23` |

Zbiór ścieżek HEAD minus zbiór @ `afe9e8ab`: **0**. Korekta nie zostawia plików governowanych powstałych po źródle inwentarza poza macierzą.

Pięć ścieżek, których nie było w drzewie `2ae10141`, a są w macierzy v2:

- `docs/project/maintainers/qualification/IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md`
- `scripts/governance/iiac_g0_inventory.py`
- `scripts/governance/verify_iiac_g0_inventory_completeness.py`
- `tests/fixtures/iiac/stale_g0_matrix_f01.json`
- `tests/unit/governance/test_iiac_g0_inventory.py`

Pierwotnie brakujące ścieżki IIAC (protokół, schemat manifestu, przykład, instrukcje B/C/D, szablon i wypełniony handoff, rekord aktywacji, macierz migracji, kontrakt publikacji, OD-ENT-02, `T-IGX-016-A.md`, `verify_iiac_*`, `iiac_common.py`, generator, testy negatywne, `.cursor/rules/intergrax-independent-audit.mdc`) są w niezależnym zbiorze i w macierzy. Lista obowiązkowa: **0 braków**.

W zbiorze źródłowym są też m.in. reguły `.cursor/rules/*.mdc` (11 plików + `README.md`), `schemas/iiac_audit_session_manifest.v1.schema.json` oraz tickety `governance/tickets/T-IGX-*.md` włącznie z `T-IGX-016-A.md`.

### Klasyfikacja kategorii 1

Jedyny wpis `NORMATIVE_CATEGORY_REQUIREMENTS`: protokół IIAC musi mieć kategorię **1 lub 2**.

Kategorie protokołu w macierzy: **2, 3, 4, 5, 6, 7**. Kategorii 1 nie ma, bo reguła `preserve_only` zdejmuje kategorię 1, gdy tekst trafia w kategorie 3, 4 lub 5 (`fail-closed`, `PROD-Q`, język prawny). Wymaganie normatywne jest spełnione przez kategorię 2. Kategorie 3, 4 i 5 są obecne — autorytet runtime, PROD-Q i prawny pozostaje oznaczony jako PRESERVE.

Rozkład kategorii w całej macierzy: 1:392, 2:15264, 3:1001, 4:372, 5:698, 6:209, 7:452.

Wszystkie 392 ścieżki kategorii 1 istnieją w drzewie `afe9e8ab` (0 nieobecnych). Wzorce sumują się do 392:

| Wzorzec | Liczba |
|---|---|
| `OD-[A-Z]+-\d+\|FRZ-[A-Z]+-\d+` | 119 |
| Protocol 03 / human auditor / mandatory external human | 126 |
| `\bB6\b\|OD-EVOL-11\|OD-ENT-03` | 145 |
| `CLOSED — INDEPENDENTLY\|stage CLOSED` | 2 |

Ścieżki z kategorią 1 oraz 3, 4 albo 5 jednocześnie: **0 / 0 / 0**.

### Wykluczenia

Pod prefiksami autorytetu (qualification, plans, `scripts/governance`, `governance/`, `.cursor/rules/`, `.github/`) poza zbiorem governowanym są dokładnie dwa obiekty:

- `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` — wykluczenie wyjścia generatora
- `.github/repo-management/sync-github-metadata.bat` — sufiks `.bat` spoza listy; nie jest normą IIAC

Fałszywego wykluczenia ścieżek IIAC, schematów JSON, reguł Cursor ani ticketów nie stwierdzono.

### Stary inwentarz F01

`tests/fixtures/iiac/stale_g0_matrix_f01.json`: `verify_iiac_g0_inventory_completeness.py` zwraca **FAIL** (17482 brakujących ścieżek, w tym protokół IIAC i `verify_iiac_*`).

`--expect-head eab65e010f5a1abfc413bb05419435bdf6b1810b` na macierzy v2: **FAIL** (`inventory_source_sha afe9e8ab…` nie jest przodkiem tego HEAD).

Właściwość kontrolna, obecnie bez skutku na tym SHA: domyślny `--expect-head` akceptuje przodka, nie tylko równość z HEAD. Zbiory governowane od `afe9e8ab` do HEAD są równe, więc ta akceptacja nie ukrywa dryfu plików.

## F02 — pytest w PowerShell

Polecenie z ticketa i zaktualizowanej instrukcji Session B:

```powershell
uv run pytest tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q
```

Wynik: **22 passed in 77.64s**. Glob `test_iiac_*.py` w instrukcji Session B został zastąpiony tymi dwiema ścieżkami (`git diff 2ae10141..HEAD` na pliku instrukcji).

## F03 — handoff

Rekord: `docs/project/maintainers/qualification/IIAC_SESSION_HANDOFF_RECORD_2026_11_05_SESSION_A_B.md`.

| Identyfikator | Wartość |
|---|---|
| `implementation_session_id` | `IIAC-GOV-A-2026-11-05` |
| `auditor_session_id` (rekord historyczny) | `IIAC-GOV-B-2026-11-05-independent` |
| `auditor_session_id` (ta sesja re-audytu) | `IIAC-GOV-B-2026-11-05-reaudit-f01f03` |
| `certifier_session_id` | `IIAC-GOV-CERT-PENDING` |

Identyfikatory implementera, audytora i certyfikującego są różne. Rejestr ról zawiera `independent-quality-auditor` jako osobną sesję. Ta sesja nie jest sesją implementera korekty.

Linia `Approved handoff: operator @ ________________` jest **PENDING**. Nie została podpisana w tej sesji.

Klasyfikacja według protokołu IIAC i planu reformy §4.2 oraz §15:

**Wymagany warunek przed aktywacją**, nie bloker certyfikacji samej korekty F01–F03 i nie podpisana zgoda operatora.

- §15 stawia werdykt Session B **przed** linią operatora w `IIAC_OD_ACTIVATION_*.md`.
- Pierwotne F03 miało wagę INFO i właściciela „operator”.
- N-02 w `verify_iiac_audit_manifest.py --require-handoff-template` sprawdza istnienie **szablonu**, nie podpisu linii operatora.
- Poziom `process_enforced` oznacza rekord handoff i znacznik czasu operatora, bez roszczenia dowodu kryptograficznego sesji.

## Regresja i kontrole negatywne

Zakres diff `2ae10141..HEAD` w `intergrax/`, `agents/`, `applications/`: **pusty**.

Zmienione kontrole governance względem pinu historycznego: generator macierzy, nowy `iiac_g0_inventory.py`, nowy weryfikator kompletności. Protokół IIAC, rekord aktywacji, `verify_iiac_policy_activation.py`, `verify_iiac_role_separation.py`, `verify_iiac_audit_manifest.py`, `verify_od_ent_02_compensating_control.py` i `.cursor/rules/intergrax-independent-audit.mdc` nie wchodzą w ten diff. Zmiana macierzy migracji to jedno zdanie o schemacie v2; zdanie o ręcznym przeglądzie kategorii 1 przed aktywacją zostało.

| Polecenie | Wynik |
|---|---|
| pytest (jak wyżej) | **22 passed**, exit zgodny z sukcesem |
| `verify_iiac_g0_inventory_completeness.py` | **PASS** (17510 @ `afe9e8ab`) |
| `verify_iiac_policy_activation.py --expect-not-in-force` | **NOT_IN_FORCE** |
| `verify_od_ent_02_compensating_control.py` | **PASS (package draft)** |
| `verify_iiac_audit_manifest.py` na manifeście przykładowym | **PASS** |
| `verify_iiac_role_separation.py` na manifeście przykładowym | **PASS** |
| kompletność na `stale_g0_matrix_f01.json` | **FAIL** (oczekiwane) |
| kompletność `--expect-head eab65e010f5a…` | **FAIL** (oczekiwane, N-04 / stale SHA) |

`IIAC_POLICY_ACTIVATION_RECORD.json`: `status = NOT_IN_FORCE`, `session_b_verdict = null`, `operator_activation_line = null`. Brak plików `*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md` (test N-09 w zestawie 22).

## Pozostałe warunki (nie są otwartymi blokerami F01–F03)

1. Podpis operatora na handoff — **PENDING** (warunek przed aktywacją).
2. Osobna sesja certyfikująca i linia w `IIAC_OD_ACTIVATION_*.md` plus aktualizacja JSON — wymagane przez §15 przed `IN_FORCE`.
3. `certification_publication_sha` w manifeście tego re-audytu jest SHA replay (`5d6b44b05…`), nie commitem publikacji certyfikującej.
4. Dowody leżą w `.tmp/session/IIAC-GOV-B-REAUDIT/` i nie są commitem gita.

## Zdatność do aktywacji

**Brak zdatności do `IN_FORCE`.** Manifest re-audytu można przekazać osobnej sesji certyfikacji / governance. Ta sesja nie ustawia aktywacji i nie wydaje decyzji certyfikującego.

## Handoff

Manifest: `.tmp/session/IIAC-GOV-B-REAUDIT/iiac_audit_session_manifest.IIAC-GOV-1.json`  
Rejestr ustaleń: `.tmp/session/IIAC-GOV-B-REAUDIT/IIAC-GOV-1_reaudit_findings.json`  
Enumeracja: `.tmp/session/IIAC-GOV-B-REAUDIT/independent_f01_report.json`
