# DIST-SCALE-X — L9 audytor: checklista G-06 (Protocol 03 / ECAP Parent Closure)

| Pole | Wartość |
| --- | --- |
| **Zakres** | Jeden audyt L9 parent **P5–P7** (nie osobne audyty P5/P6) |
| **Protokół** | G-06 = [ECAP §2.B Parent Closure Audit](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) (kroki B1–B6) + klasyfikacja usterek [ECAP §13](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) |
| **Handoff** | [DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md](DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md) |
| **SAC** | `SAC-DIST-SCALE-X-v1` — bez amendment (OD-03) |

**Uwaga sesji:** soak WL-7 **4 h** (`14400 s`) może być w toku poza tym audytem (logi operatora: `.tmp/session/dist-scale-x-wl7-4h/`). G-06 **nie wymaga** ukończenia soak przy zaakceptowanym deferralu; po zakończeniu soak audytor **nie** automatycznie zamyka parent — aktualizacja wierszy 1/11/16 to osobna kwalifikacja.

---

## 1. Cel i zakres

- Potwierdzić spójność implementacji, manifestów i dokumentów kwalifikacyjnych @ **AUDIT_BASELINE_SHA** (GitHub, nie wyłącznie lokalne `.tmp/`).
- Zweryfikować matrycę wyjścia L9 (16 wierszy) vs cytowane **EXECUTION_SHA** i disposycje (**PASS**, **ALREADY PROVEN**, **PARTIAL**, **BLOCKED**).
- **Poza zakresem G-06 (chyba że handoff zmieniony):** pełny 4 h soak WL-7 jako warunek **AUDIT PASS (scoped)** — patrz [DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md](DIST_SCALE_X_WL7_PHYSICAL_DEFERRAL_RECORD.md).
- **Zabronione w werdykcie G-06:** parent **CLOSED**, **INDEPENDENTLY ACCEPTED**, globalny **FRZ PASS**, **EBH-5**.

---

## 2. Baseline SHA — jak ustalić

1. Odczytaj **`AUDIT_BASELINE_SHA`** z tabeli w [DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md](DIST_SCALE_X_L9_INDEPENDENT_AUDIT_HANDOFF.md) (np. `a093bfa7a…` lub nowszy, jeśli handoff zaktualizował agent równoległy).
2. Na GitHubie: porównaj pliki i commity @ ten SHA (catalog, provenance, matryca parent).
3. **Operator (lokalnie):** zostań na gałęzi (`development`); **nie** używaj detached checkout do „dopasowania” HEAD do pinu. Sprawdź zawartość pinu:

   ```bash
   git fetch origin development
   git merge-base --is-ancestor <AUDIT_BASELINE_SHA> HEAD
   ```

4. W raporcie audytora zapisz **dwa** SHA: **AUDIT_BASELINE_SHA** (dowody) oraz **EXECUTION_SHA (P5/P6 physical)** z handoff/parent (`9ac74ca3849013e729dbc8c893f64427ef588f94` jeśli bez zmian).

---

## 3. Matryca wyjścia L9 (16 wierszy) — skrót

Źródło pełne: [DIST_SCALE_X_PARENT_QUALIFICATION_EVIDENCE.md](DIST_SCALE_X_PARENT_QUALIFICATION_EVIDENCE.md) § L9 exit matrix.

| ID | Co sprawdzić | Gdzie dowód |
| --- | --- | --- |
| 1 | Spec WL-1..7 + harness WL-7; dispo **PARTIAL** jeśli 4 h pending | [DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md](DIST_SCALE_X_WORKLOAD_WL1_WL7_SPECS.md); [DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md); [DIST_SCALE_X_M_EVIDENCE.md](DIST_SCALE_X_M_EVIDENCE.md) |
| 2 | Adaptery MB-1..5; reuse P2–P5 + slice P6 | Provenance P5/P6; wcześniejsze stage records P2–P5 |
| 3 | Multi-worker N≥2 | P2 reconciliation / cited SHA `ee5a8371…` |
| 4 | Overload | P2/P3 SHAs w matrycy parent |
| 5 | Retry storm WL-6 | Cited `e6020efc3`; Rabbit M evidence |
| 6 | Hot key WL-3 | Cited `8656fb7d8` |
| 7 | Crash/restart P4 G/H | [DIST_SCALE_X_G_EVIDENCE.md](DIST_SCALE_X_G_EVIDENCE.md); SHA `f0c49fd3…` |
| 8 | Partial failure P4 F + P5 I | P4 + [DIST_SCALE_X_P5_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P5_PHYSICAL_PROVENANCE.md) |
| 9 | Tenant stress WL-2 K | Cited `8656fb7d8` |
| 10 | Resource exhaustion L | P5 L + P3 reuse @ `9ac74ca…` |
| 11 | Soak M WL-7 **14400 s** | [DIST_SCALE_X_M_EVIDENCE.md](DIST_SCALE_X_M_EVIDENCE.md); manifest P6; **BLOCKED** do `soak_complete_envelope=1` |
| 12 | Chaos N P6 | [DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md](DIST_SCALE_X_P6_PHYSICAL_PROVENANCE.md); matrix 4/4 |
| 13 | Zero invariant violations | Manifesty F02 / metryki per run |
| 14 | Capacity envelope | [DIST_SCALE_X_CAPACITY_ENVELOPE.md](DIST_SCALE_X_CAPACITY_ENVELOPE.md) |
| 15 | Failure envelope | [DIST_SCALE_X_FAILURE_ENVELOPE.md](DIST_SCALE_X_FAILURE_ENVELOPE.md) |
| 16 | Blockers = 0 | Spójność z wierszem 11 — **BLOCKED** dopóki row 11 open |

Indeks bundle: parent § Proof bundle index; catalog: `tests/qualification/dist_scale_x/catalog.py`.

---

## 4. Fizyczne vs ALREADY PROVEN

| Disposition | Oczekiwanie audytora |
| --- | --- |
| **PASS** / **PARTIAL** (P5/P6 @ EXECUTION_SHA) | Manifest F02 pod `.tmp/proof/dist-scale-x/<run_id>/` **lub** opublikowany run_id + metryki w provenance/M evidence; **EXECUTION_SHA** musi się zgadzać z parent/handoff. |
| **ALREADY PROVEN** | Audytor weryfikuje **cytowany SHA** i dokument zamknięcia stage (P2–P4 reconciliation) — **nie** wymaga ponownego runu fizycznego w G-06, o ile brak sprzeczności z AUDIT_BASELINE_SHA. |
| **ENVELOPE_ONLY** (14–15) | Dokumenty P7 — bez twierdzenia pełnego soak. |
| **BLOCKED** (11, 16) | Jawny status; brak udawania PASS na wymiar M. |

Mock-only jako dowód wymiaru **M** przy deklaracji pełnego soak → **AUDIT FAIL**.

---

## 5. WL-7: `soak_complete_envelope=1` @ 14400 s

- Envelope: [DIST_SCALE_X_P6_ENVELOPE_RECORD.md](DIST_SCALE_X_P6_ENVELOPE_RECORD.md) — `DIST_SCALE_X_WL7_DURATION_SECONDS=14400`.
- W manifeście runu P6 (klasa F02, run_id typu `wl07-soak-*`): pole metryk **`soak_complete_envelope=1`**, `duration_s` ≥ **14400**, `errors=0` (patrz [DIST_SCALE_X_M_EVIDENCE.md](DIST_SCALE_X_M_EVIDENCE.md), [DIST_SCALE_X_FAILURE_ENVELOPE.md](DIST_SCALE_X_FAILURE_ENVELOPE.md)).
- Stan deferral (harness 120 s, `soak_complete_envelope=0`): akceptowalny dla **G-06 scoped PASS**; **nie** zamyka wierszy 11/16 ani G-07 parent closure bez osobnego operator run.

---

## 6. SAC-DIST-SCALE-X-v1

- Porównaj [DIST_SCALE_X_SAC_PROPOSAL.md](DIST_SCALE_X_SAC_PROPOSAL.md) / OD-03 z dossierem @ AUDIT_BASELINE_SHA.
- **Brak** niedokumentowanej zmiany zakresu, progów lub exit criteria (silent amendment → **AUDIT FAIL**).

---

## 7. FRZ / ECAP

- Agent implementacyjny **nie** nadaje globalnego **FRZ PASS**.
- Audytor: spójność deklaracji parent z ECAP (tylko dowody w scope DIST-SCALE-X); ewentualne odwołania FRZ jako **scoped / NOT PROVEN @ HEAD** — bez promocji checklisty freeze bez pełnego ENT-AUDIT-X.
- Ustalenia klasyfikuj wg [ECAP §13](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md).

---

## 8. PASS / FAIL (G-06) i wpływ na G-07

| Werdykt G-06 | Warunki |
| --- | --- |
| **AUDIT PASS (scoped)** | B1–B6 spełnione @ AUDIT_BASELINE_SHA; wiersze 1–10, 12–15 zgodne z dokumentacją; deferral WL-7 udokumentowany; brak sprzeczności SAC/SHA. |
| **AUDIT FAIL** | Sprzeczność matrycy vs manifest, brak ownera dowodu, mock-only M, nieudokumentowany drift SAC, brak pliku @ cytowanym SHA. |

**G-07 (operator):** zablokowane przy **AUDIT FAIL**; przy **PASS (scoped)** G-07 **nie** oznacza automatycznie parent **CLOSED** — wymaga osobnej decyzji operatora; wiersze **11/16** pozostają otwarte do fizycznego soak (chyba że wcześniej zamknięte kwalifikacją).

---

## 9. Deliverable audytora

1. **Werdykt:** `AUDIT PASS (scoped)` lub `AUDIT FAIL` (bez G-06 PASS jako zamknięcia parent).
2. **Lista findings** (ID, klasa ECAP §13, SHA, ścieżka, minimalna korekta).
3. **Potwierdzone SHA:** AUDIT_BASELINE_SHA + EXECUTION_SHA (P5/P6) użyte w weryfikacji.
4. Opcjonalnie: szablon [ECAP §14](ENTERPRISE_CONTINUOUS_ARCHITECTURE_ASSURANCE_PROTOCOL.md) z `Audit mode: PARENT`.

Zapis wyniku: issue/PR komentarz lub rekord kwalifikacyjny wskazany przez operatora — **bez** zmiany statusu parent na CLOSED w tym kroku.
