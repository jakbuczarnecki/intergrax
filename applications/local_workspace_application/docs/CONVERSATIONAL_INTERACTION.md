# Conversational Interaction — LKW

**Status:** `LKW-CONVERSATIONAL-INTERACTION-1A` — planner contract implemented (`plan_version = "2"`); execution not yet wired.

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

## 2. Przepływ planu v2

```text
naturalna wiadomość użytkownika
        ↓
provider-neutralny LLM planner        ← to zadanie (1A)
        ↓
ConversationInteractionPlan v2
        ↓
extracted objects + grounded evidence spans
        ↓
deterministyczna walidacja planu
        ↓
przyszły resolver referencji           ← zadanie 1B
        ↓
przyszły executor capabilities
```

**Planner** interpretuje język naturalny i zwraca typowany plan JSON (`plan_version = "2"`). **Nie wykonuje** żadnych operacji, nie wywołuje endpointów LKW, nie zmienia workspace.

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

## 4. Obiekty vs akcje (plan v2)

Plan v2 rozdziela **wydobycie obiektu** od **decyzji, co z nim zrobić**.

### Obiekty (`objects`)

| Typ | Model | Pola |
|-----|-------|------|
| URL | `WebUrlExtractedObject` | `object_type: web_url`, `value`, `evidence` |
| ścieżka lokalna | `LocalFileReferenceExtractedObject` | `object_type: local_file_reference`, `reference_kind`, `value`, `evidence` |

### Akcja routingu źródeł

| Pole | Wartość |
|------|---------|
| model | `KnowledgeAddSourcesPlannedAction` |
| `action_type` | `knowledge.add_sources` |
| `source_object_ids` | lista `object_id` z sekcji `objects` |

Przykład (opisowy):

```yaml
objects:
  - object_id: obj-url-1
    object_type: web_url
    value: https://www.cenniki.pl
    evidence: { source: message_text, start: …, end: …, text: … }

  - object_id: obj-file-1
    object_type: local_file_reference
    reference_kind: file
    value: C:\moje dokumenty\cenniki.xls
    evidence: { source: message_text, start: …, end: …, text: … }

actions:
  - action_type: knowledge.add_sources
    workspace: { kind: name, value: magazyn }
    source_object_ids: [obj-url-1, obj-file-1]
```

Obie wartości trafiają do tego samego target workspace. **`workspace.activate` nie powstaje** — użytkownik wskazał workspace jako cel operacji, nie jako aktywny kontekst.

URL w zwykłym pytaniu (`co sądzisz o https://example.com?`) planowany jest jako `workspace.ask`, chyba że użytkownik wyraźnie prosi o dodanie strony do wiedzy.

---

## 5. Model evidence

Każdy wyekstrahowany obiekt ma `MessageTextEvidenceSpan`:

| Reguła | Wartość |
|--------|---------|
| `evidence.source` | `"message_text"` |
| `start` | zero-based |
| `end` | exclusive |
| `evidence.text` | musi równać się `message_text[start:end]` |
| `object.value` | musi równać się `evidence.text` |

Offsety odnoszą się do **decoded** `message_text` z requestu. Wartość **nie jest** trimowana, normalizowana, lowercasowana ani przepisywana.

Indeksy `start` i `end` muszą być dokładnie typu `int` (nie string, float ani bool).

Walidacja zakresu (`end <= len(message_text)`) odbywa się w `validate_plan_against_request()`, nie w modelu Pydantic.

---

## 6. Granica LLM / determinizm

**LLM** klasyfikuje fragment wiadomości jako `web_url` albo `local_file_reference` i podaje evidence span.

**Warstwa deterministyczna** nie rozpoznaje składni URL-a ani ścieżki. Sprawdza wyłącznie, czy wartość obiektu pochodzi dokładnie ze wskazanego evidence span w `message_text`.

Nie ma deterministycznych parserów URL-i, heurystyk ścieżek Windows, normalizacji ani budowania kandydatów URL.

---

## 7. Kierunek działań: źródło → workspace target

Każda akcja zależna od workspace ma jawny `WorkspaceReference`:

| Rodzaj | Znaczenie |
|--------|-----------|
| `active` | użyj obecnie aktywnego workspace |
| `name` | użytkownik podał nazwę (także z literówką) |
| `ordinal` | numer z widocznej listy |
| `created_by_action` | workspace utworzony wcześniejszą akcją `workspace.create` |

Planner **nie rozwiązuje** nazw do `workspace_id` — to robi przyszły resolver deterministycznie.

---

## 8. Workspace jako cel ≠ workspace aktywny

```text
dodaj https://example.com/docs do workspace magazyn
→ knowledge.add_sources z targetem name=magazyn
→ bez workspace.activate
```

Przykład z kontekstem: aktywny workspace to `finanse` (`ws-1`), a `magazyn` (`ws-2`) jest nieaktywny. Operacja kieruje źródło do `magazyn` bez aktywacji.

```text
przełącz mnie na workspace magazyn  → workspace.activate (jawna intencja)
ustaw magazyn jako aktywny
od teraz pracujmy w magazynie
```

Workspace będący celem ingestion lub Ask **nie zmienia** aktywnego workspace użytkownika.

---

## 9. Provider-neutralność LLM

`ConversationInteractionPlanner` przyjmuje gotowy `LLMAdapter` przez dependency injection. Nie importuje Ollama ani innego providera; nie czyta konfiguracji providera. Ten sam kontrakt planowania działa dla Slacka, HTTP, MCP i przyszłych frontendów.

---

## 10. Structured output i walidacja

1. Adapter musi obsługiwać `supports_structured_output()` — w przeciwnym razie fail-closed.
2. Wynik to `ConversationInteractionPlan` (Pydantic v2, `extra="forbid"`, `plan_version = "2"`).
3. Deterministyczna walidacja `validate_plan_against_request()` sprawdza m.in.:
   - unikalne `object_id` w planie;
   - akcje odwołują się tylko do istniejących `object_id`;
   - każdy obiekt musi być użyty przez `knowledge.add_sources` (unused object → odrzucenie);
   - nieznany `source_object_id` → odrzucenie;
   - `obj.value == evidence.text` oraz `evidence.text == message_text[start:end]`;
   - attachment IDs tylko z requestu;
   - `evidence_quotes` jako fragmenty rzeczywistej wypowiedzi;
   - graf `depends_on` acykliczny i spójny.
4. Przy błędnym JSON / schemacie / walidacji — **jedna** kontrolowana próba naprawy; brak trzeciej próby lokalnie.

---

## 11. Ochrona przed wymyślonymi obiektami

Model LLM **nie może**:

- wymyślać attachment IDs, URL-i ani ścieżek;
- zgadywać `candidate_id` (tylko `name` / `ordinal` jako referencja użytkownika);
- twierdzić, że akcja została wykonana;
- emitować nieużywanych obiektów w sekcji `objects`.

Przy niejednoznaczności zwraca `ConversationClarification` zamiast zgadywać.

---

## 12. Przyszły przepływ end-to-end

```text
planner (v2)
→ resolver (workspace name → workspace_id, ścieżka → candidate/upload)
→ authorization
→ confirmation policy (np. delete)
→ validated executor
→ LKW capabilities (intake, workspace ops, Ask)
→ Knowledge Query Orchestrator (gdy akcja to workspace.ask)
→ Hybrid Ask (indexed + live evidence)
→ aggregate response
```

### 12.1 Conversation Interaction Planner vs Knowledge Query Orchestrator

| Warstwa | Odpowiedzialność |
|---------|------------------|
| **Conversation Interaction Planner** | Interpretacja intencji produktowej użytkownika: utworzenie workspace, dodanie źródła, połączenie zasobu, pytanie Ask, inspekcja operacji. Nie generuje Graph API, JQL, DAX, SQL ani wywołań MCP. |
| **Knowledge Query Orchestrator** | Pozyskanie dowodów dla autoryzowanego pytania Ask: RAG, live capabilities, hybrid, clarification. Model może zaproponować plan dowodów; runtime deterministycznie waliduje każde wywołanie. |

Natural-language intencje połączenia i źródeł (np. „podłącz projekt Jira i SharePoint Orion”) trafiają do planera jako akcje produktowe; wykonanie przechodzi przez resolver, autoryzację i allowlistowane capabilities — nie przez bezpośrednie wywołania providerów ze Slacka.

**Hybrid Ask** łączy indexed RAG evidence z autoryzowanymi live provider results w jeden zestaw **Evidence Items** z ujednoliconą proweniencją.

Slack pozostaje jednym z wielu frontendów. Większy blok produktowy: `LKW-CONVERSATIONAL-FRONTEND-1`. Wewnętrzne slice'y: **CONV-1B** (resolver + executor), **CONV-1C** (Slack mixed-message cutover). Architektura: [`KNOWLEDGE_ACCESS_ARCHITECTURE.md`](KNOWLEDGE_ACCESS_ARCHITECTURE.md).

W zadaniu **1A** wykonanie **nie jest podłączone**. Planner rozpoznaje intencję dodania źródła URL (`web_url` → `knowledge.add_sources` w kontrakcie planu). Backend LKW udostępnia już capability `WEB_URL` (`POST …/knowledge/web-urls`), ale planner nie jest jeszcze podłączony do tego API. Resolving workspace references, validated execution i Slack natural-language cutover nadal należą do **CONV-1B** / **CONV-1C** (nie do osobnej komendy URL). Obecne dokładne komendy Slack pozostają tymczasowym fallbackiem.

---

## 13. Lokalizacja kodu

| Moduł | Rola |
|-------|------|
| `conversation/interaction_models.py` | modele requestu, obiektów, akcji, planu v2, walidacja strukturalna |
| `conversation/interaction_prompt.py` | prompt systemowy i bezpieczny kontekst JSON |
| `conversation/interaction_planner.py` | `ConversationInteractionPlanner`, błędy, walidacja względem requestu |

Powiązana roadmapa: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## 14. Deterministic context resolution before planner use

Future planner, resolver and executor requests receive a pre-resolved **ConversationExecutionContext** — not discretionary model choices about audience or workspace.

**Conceptual envelope:**

```text
ConversationExecutionContext
├── tenant_id
├── audience_mode
├── workspace_id
├── principal_ref
├── conversation_context_binding_id
├── activation_policy
├── canonical_thread_ref
└── allowed_product_capabilities
```

The LLM planner must **not** choose: audience mode; shared versus personal memory; conversation workspace binding; source visibility; private-to-shared data elevation; allowed product capabilities.

**Ingress validation (before planner):** `binding.audience_mode` must match `ingress.observed_audience`; `UNKNOWN` fails closed. At most one `ACTIVE` binding per semantic identity (`tenant_id` + `conversation_connection_ref` + `opaque_conversation_ref`).

**PERSONAL** conversation: observed `PERSONAL` + principal match → workspace via `FIXED_WORKSPACE` or durable `PERSONAL_SELECTION` → thread memory partition → permitted evidence.

**SHARED** conversation: observed `SHARED` → fixed shared workspace (`FIXED_WORKSPACE` only) → `READ_ONLY_ASK` capability boundary → `SHARED_ALLOWED` evidence only. Caller's private active workspace, DM workspace selection, personal memory and private sources are ignored. Missing shared binding fails closed.

A natural-language workspace reference may target an operation but must not silently replace the workspace bound to a shared conversation.

Canonical contract: [`CONVERSATION_CONTEXT_ARCHITECTURE.md`](CONVERSATION_CONTEXT_ARCHITECTURE.md).
