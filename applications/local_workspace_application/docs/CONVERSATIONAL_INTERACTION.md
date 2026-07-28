# Conversational Interaction — LKW

**Status:** `LKW-CONVERSATIONAL-INTERACTION-1A` — planner contract implemented; execution not yet wired.

LKW ma działać jak inteligentny współpracownik, a nie jak terminal komend. Użytkownik pisze naturalnie — po polsku, z literówkami, w jednej wiadomości mieszając pliki, URL-e, lokalne ścieżki i wskazanie workspace — a system rozumie intencję i przygotowuje bezpieczny plan działań.

---

## 1. Dlaczego nie terminal komend

Ścisłe komendy tekstowe (`/sources`, `/workspace use 2`) są przydatne jako techniczny fallback, ale nie są docelowym UX-em. Prawdziwy użytkownik mówi:

```text
dołącz informacje o cennikach ze strony https://www.cenniki.pl
oraz dorzuć moją kopię lokalną cenników z c:\moje dokumenty\cenniki.xls
a to wszystko do workspace "magazyn"
```

System musi zrozumieć **całą** wiadomość i przygotować plan wielu działań ze wspólnym celem workspace — bez wymuszania kolejności słów jako kolejności wykonania.

---

## 2. LLM planner vs deterministyczny executor

```text
naturalna wiadomość użytkownika
        ↓
provider-neutralny LLM planner        ← to zadanie (1A)
        ↓
ustrukturyzowany ConversationInteractionPlan
        ↓
deterministyczna walidacja planu
        ↓
przyszły resolver referencji           ← zadanie 1B
        ↓
przyszły executor capabilities
```

**Planner** interpretuje język naturalny i zwraca typowany plan JSON. **Nie wykonuje** żadnych operacji, nie wywołuje endpointów LKW, nie zmienia workspace.

**Executor** (przyszły) wykona plan po rozwiązaniu referencji, autoryzacji i ewentualnej polityce potwierdzeń.

---

## 3. Wiele działań w jednej wiadomości

Jedna wiadomość może zawierać:

- kilka źródeł wiedzy (załączniki, URL-e, lokalne referencje);
- operacje na workspace (lista, utworzenie, aktywacja, usunięcie);
- zapytanie Ask;
- prośbę o doprecyzowanie (clarification).

Kolejność słów użytkownika **nie jest** kolejnością wykonania. Logiczna kolejność wynika z `depends_on` w planie oraz z przyszłego resolvera referencji.

---

## 4. Pliki, URL-e i ścieżki jednocześnie

Przykład z dokumentacji zadania:

| Źródło | Akcja planu | Cel |
|--------|-------------|-----|
| `https://www.cenniki.pl` | `knowledge.add_web_urls` | workspace `"magazyn"` |
| `c:\moje dokumenty\cenniki.xls` | `knowledge.add_local_references` | workspace `"magazyn"` |

Obie akcje mają ten sam target workspace. **`workspace.activate` nie powstaje** — użytkownik wskazał workspace jako cel operacji, nie jako aktywny kontekst.

URL w zwykłym pytaniu (`co sądzisz o https://example.com?`) planowany jest jako `workspace.ask`, chyba że użytkownik wyraźnie prosi o dodanie strony do wiedzy.

---

## 5. Kierunek działań: źródło → workspace target

Każda akcja zależna od workspace ma jawny `WorkspaceReference`:

| Rodzaj | Znaczenie |
|--------|-----------|
| `active` | użyj obecnie aktywnego workspace |
| `name` | użytkownik podał nazwę (także z literówką) |
| `ordinal` | numer z widocznej listy |
| `created_by_action` | workspace utworzony wcześniejszą akcją `workspace.create` |

Planner **nie rozwiązuje** nazw do `workspace_id` — to robi przyszły resolver deterministycznie.

---

## 6. Workspace jako cel ≠ workspace aktywny

```text
dodaj to do workspace "magazyn"     → target operacji, bez workspace.activate

przełącz mnie na workspace magazyn  → workspace.activate (jawna intencja)
ustaw magazyn jako aktywny
od teraz pracujmy w magazynie
```

Workspace będący celem ingestion lub Ask **nie zmienia** aktywnego workspace użytkownika.

---

## 7. Provider-neutralność LLM

`ConversationInteractionPlanner` przyjmuje gotowy `LLMAdapter` przez dependency injection. Nie importuje Ollama ani innego providera; nie czyta konfiguracji providera. Ten sam kontrakt planowania działa dla Slacka, HTTP, MCP i przyszłych frontendów.

---

## 8. Structured output i walidacja

1. Adapter musi obsługiwać `supports_structured_output()` — w przeciwnym razie fail-closed.
2. Wynik to `ConversationInteractionPlan` (Pydantic v2, `extra="forbid"`).
3. Deterministyczna walidacja `validate_plan_against_request()` sprawdza m.in.:
   - attachment IDs tylko z requestu;
   - URL-e i ścieżki jako **dokładne** zasoby użytkownika — nie wystarczy substring wiadomości;
   - `evidence_quotes` jako fragmenty rzeczywistej wypowiedzi;
   - graf `depends_on` acykliczny i spójny.
4. Przy błędnym JSON / schemacie / walidacji — **jedna** kontrolowana próba naprawy; brak trzeciej próby lokalnie.

### Dokładne dopasowanie URL-i i ścieżek

Walidacja **nie** sprawdza jedynie, czy wartość jest substringiem wiadomości. Wymaga dokładnej tożsamości URL-a lub odgraniczonej pełnej ścieżki lokalnej.

Przykład: jeśli użytkownik podał `https://example.com/private?token=abc`, plan nie może zaakceptować skróconego `https://example.com` — krótszy fragment większego URL-a jest odrzucany. Końcowa interpunkcja zdania (np. kropka po URL-u) może zostać bezpiecznie pominięta przy ekstrakcji kandydatów z wiadomości.

Dla ścieżek Windows porównanie jest case-insensitive, ale nadal wymaga pełnej, odgraniczonej ścieżki — skrócony katalog lub nazwa pliku bez rozszerzenia są odrzucane.

---

## 9. Ochrona przed wymyślonymi obiektami

Model LLM **nie może**:

- wymyślać attachment IDs, URL-i ani ścieżek;
- zgadywać `candidate_id` (tylko `name` / `ordinal` jako referencja użytkownika);
- twierdzić, że akcja została wykonana.

Przy niejednoznaczności zwraca `ConversationClarification` zamiast zgadywać.

---

## 10. Przyszły przepływ end-to-end

```text
planner
→ resolver (workspace name → workspace_id, ścieżka → candidate/upload)
→ authorization
→ confirmation policy (np. delete)
→ executor
→ LKW capabilities (intake, Ask, workspace ops)
→ aggregate response
```

W zadaniu **1A** wykonanie **nie jest podłączone**. Obecne dokładne komendy Slack pozostają tymczasowym fallbackiem.

---

## 11. Lokalizacja kodu

| Moduł | Rola |
|-------|------|
| `conversation/interaction_models.py` | modele requestu, akcji, planu, walidacja strukturalna |
| `conversation/interaction_prompt.py` | prompt systemowy i bezpieczny kontekst JSON |
| `conversation/interaction_planner.py` | `ConversationInteractionPlanner`, błędy, walidacja względem requestu |

Powiązana roadmapa: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
