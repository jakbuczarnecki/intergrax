---
name: Plan implementacji fundamentu
overview: "Wykonawczy plan ARCH-FREEZE. Produkcja odmawia magazynu, którego katalog nie oznacza jako stabilny. Praca poza procesem ma trwały wiersz i ogrodzenie dzierżawy; broker tylko budzi workera. Skill nie osłabia polityk nadrzędnych. Nieznany skutek nie uruchamia kolejnej operacji zewnętrznej."
todos: []
isProject: false
---

# Plan implementacji fundamentu Integrax

Plan jest do wykonania. Nie projektuje drugiego silnika, rejestru, tożsamości ani katalogu dostawców. Fabryka Scenariuszy nie wchodzi w zakres.

Decyzje podjęte na kodzie, nie do ponownego rozstrzygania przez wykonawcę:

- Milvus i Weaviate nie dostają adaptera. Źródłem prawdy jest istniejący katalog integracji: `IntegrationManifest.status`. Stabilne magazyny wektorowe w tym katalogu to Qdrant, PgVector i Chroma. Pozostałe wpisy wektorowe są beta. Ścisły profil, który wskaże wpis inny niż stabilny, nie składa się. Nie powstaje druga lista slugów. Otwarty klient nie zapisuje wiedzy do pamięci procesu. Odmowa mieści się w zamrożeniu architektury: nieobsługiwana konfiguracja produkcyjna ma padać. Pełna implementacja i żywa kwalifikacja dostawcy są poza tym zakresem.
- Praca poza procesem używa kontraktu `TaskQueue`. Autorytatywny stan zadania jest trwałym wierszem plus księgą idempotencji. Compare-and-set rozstrzyga wyścig zapisu i nie dowodzi, że wiersz przeżyje proces. Trwałość i współdzielenie wynikają z topologii magazynu i z ponownego otwarcia w drugim procesie. Ogrodzenie starego workera jest istniejącą dzierżawą księgi (`fence`, `StaleClaimError`), nie nową dzierżawą. Celery jest budzeniem, nie stanem. Rdzeń `execute_logical_task` nie zna Celery. Aplikacja i worker otwierają własne połączenia.
- Skutek zewnętrzny ma już stany: niezacząty, mógł się zacząć, nieznany, zakończony, nieudany. Księga zapisuje „mógł się zacząć” przed wywołaniem. Nieznany wynik nie uruchamia kompensacji ani powtórzenia. Najpierw istniejące uzgodnienie bez mutacji, potem decyzja człowieka albo autoryzacja. Kompensacja rusza dopiero przy rozstrzygnięciu, które jej wymaga, i przy ostatecznym dowodzie.
- Trwałość, atomowość, współdzielenie i izolacja klienta są czterema osobnymi właściwościami. Żadna nie wynika z interfejsu `ConditionalDocumentStore` ani z deklaracji dostawcy. Aplikacja bez wymaganej pamięci trwałej zostaje bezstanowa.
- Wznowienie skilla czyta wersję i skrót treści spięte przy starcie. Polityka platformy, aplikacji i klienta jest nadrzędna. Skill może tylko zaostrzyć ograniczenie, które już jest w katalogu hosta.
- MCP w postawie ścisłej używa istniejącej odmowy. Tryb laboratoryjny na loopbacku zostaje, gdy postawa nie jest ścisła.

Kolejność i równoległość są w sekcji organizacji. W jednym drzewie: E, A, C, D, B, F, G. Jeden przebieg pytest naraz.

## Pakiet A. Produkcyjny magazyn wiedzy nie udaje bazy

**Problem.** Wskazanie Milvusa otwiera klienta i zapisuje wiedzę do `InMemoryVectorStore`. Weaviate po błędzie zapisu przełącza się na tę samą pamięć i połyka błąd kasowania. Klient widzi sukces, którego baza nie ma. Retencja nie usuwa wektorów klienta.

**Po zmianie.** Profil produkcyjny z magazynem spoza Qdrant, PgVector i Chroma nie składa aplikacji. Operacja na Milvusie lub Weaviate z podanym klientem podnosi błąd i nic nie zapisuje w pamięci procesu. Jawny magazyn w pamięci działa tylko poza postawą ścisłą i tylko gdy profil go wprost wybrał. Qdrant, PgVector i Chroma zostają bez zmian.

**Rozwiązanie.** Odmowa przy składaniu czyta `IntegrationManifest.status` z katalogu (`catalog_snapshot` / `iter_entries`). Wpis `vector_store` o statusie innym niż `STABLE` w postawie ścisłej nie jest otwierany. Qdrant, PgVector i Chroma są w katalogu `STABLE`. Milvus, Weaviate, LanceDB, Typesense, Pinecone, Vespa i InMemory są `BETA`. Deklarowane zdolności `CONNECT`/`READ`/`WRITE` na kontrakcie Milvusa nie są zezwoleniem produkcyjnym. Tabela w kanonie wiedzy opisuje to samo rozróżnienie i nie jest kopiowana do kodu jako druga lista. Usunąć cichą atrapę, żeby bezpośrednie skonstruowanie klasy też nie zapisywało słownika. Jawny magazyn w pamięci zostaje tylko poza postawą ścisłą i tylko gdy profil go wprost wybrał.

**Komponenty.**

- Skład profilu integracji, ten sam resolver, który dziś wybiera magazyn: przed otwarciem odczytuje status wpisu katalogu. Nie dodawać rejestru dojrzałości.
- `intergrax/integrations/providers/vector_store/milvus/rag_store.py`, klasa `MilvusVectorStore`: `add_records`, `query`, `delete`, `count` nie wołają `_inner`. Przy kliencie albo bez niego podnoszą `IntegrationConfigurationError`.
- `intergrax/integrations/providers/vector_store/milvus/opens.py`, `open_milvus_rag_store`: nie owija klienta atrapą.
- `intergrax/integrations/providers/vector_store/weaviate/rag_store.py`: usunąć przejście `_native = False` i dalszy zapis do `_inner`. Wyjątek klienta oraz nieobsługiwane kasowanie zakresu wychodzą jako błąd. `except Exception: pass` w `delete` znika.

**Zadania.**

1. W ścisłym składzie odmówić wpisu wektorowego, którego status w katalogu nie jest `STABLE`. Slug bierze się z wpisu, nie z tablicy wpisanej w planie.
2. Przerwać ścieżki zapisu Milvusa i Weaviate do atrapy. Pozostałych wpisów beta nie otwierać w tej postawie, więc ich atrap nie rozbudowywać.
3. Zostawić wstrzyknięty `store` w otwarciu Milvusa tylko jako jawny dubel testowy, nigdy jako skutek otwarcia klienta.

**Testy.** Ścisły profil i każdy slug wektorowy o statusie beta z żywego katalogu pada przy składaniu i nie tworzy `InMemoryVectorStore`. Slug stabilny przechodzi dotychczasową ścieżką. `MilvusVectorStore.add_records` z klientem-dublem nie zostawia rekordu w pamięci. Błąd wstawienia Weaviate nie wywołuje `InMemoryVectorStore.add_records`. `delete` zakresu, którego Weaviate nie obsługuje, nie kasuje atrapy. Istniejące testy Qdrant nie zmieniają zachowania.

**Koniec.** Produkcja nie wybiera niestabilnego magazynu i żaden otwarty klient nie udaje zapisu w pamięci procesu. Stabilne magazyny działają jak przed zmianą.

## Pakiet B. Zadanie żyje poza procesem aplikacji

**Problem.** Host produkcyjny umie tylko odmówić workera `memory://` albo, w laboratorium, wykonać pracę w tym samym procesie. Osobna konfiguracja Celery przy pustej zmiennej wybrałaby Redis na localhost. Aplikacja i worker nie mają wspólnego trwałego rekordu zadania, jeśli ktoś przekaże im ten sam obiekt w pamięci. Po restarcie API praca znika.

**Po zmianie.** Zadanie przyjęte w trybie produkcyjnym jest wierszem, który drugi proces odczytuje po śmierci pierwszego. Restart aplikacji i restart workera nie kasują tego wiersza. Przejęcie jest atomowe: jeden worker dostaje pracę, drugi nie. Po wygaśnięciu dzierżawy nowy worker dostaje nowe ogrodzenie, a stary nie zapisuje już stanu i nie woła systemu zewnętrznego. Komunikat Celery tylko budzi workera. Gdy zapis wiersza się nie uda, komunikat nie wychodzi i praca nie leci w procesie aplikacji. Gdy komunikat nie wyjdzie, wiersz zostaje do podjęcia. Potwierdzenie brokera idzie dopiero po trwałym zapisie końca. Niedostępny magazyn albo broker nie włącza wykonania w procesie. Tożsamość klienta jedzie w wierszu. Dostawa, która się z nim nie zgadza, nie uruchamia handlera.

**Rozwiązanie.** Nie pisać nowej kolejki ani drugiej dzierżawy. Autorytetem stanu jest wiersz `DocumentStoreTaskQueue` oraz księga idempotencji. `claim_pending` już przechodzi `PENDING` → `RUNNING` przez `replace_if_match`. To jest atomowość jednego zapisu. Trwałość jest osobno: topologia magazynu musi spełniać host (`DURABLE_SINGLE_HOST` albo `SHARED_MULTI_HOST`) i drugi proces musi wiersz zobaczyć. Magazyn tylko w pamięci procesu, nawet z compare-and-set, pada przy workerze poza procesem. Ogrodzenie jest już w księdze: właściciel, dzierżawa, `fence`, `StaleClaimError`. Wiersz zadania dziś tego ogrodzenia nie niesie, a `recover_interrupted_running` wprost nie odróżnia martwego workera od żywego. To samo ogrodzenie księgi trzeba zapisać w wierszu przy przejęciu i wymagać go przy każdym zakończeniu. Celery zostaje budzeniem przy jawnym URL. `execute_logical_task` nie zna brokera. `admit_background_execution_reentry` zostaje bramą ponownej dostawy.

**Komponenty.**

- `intergrax/applications/_shared/queue_worker_wiring.py`, `wire_optional_queue_execution`.
- `intergrax/applications/_shared/production_queue_resolver.py`, `resolve_production_queue_backend`.
- `intergrax/queueing/providers/document_store/document_store_task_queue.py`, `DocumentStoreTaskQueue`.
- `intergrax/queueing/worker/execution.py`, `execute_logical_task`.
- `intergrax/queueing/worker/dispatcher.py`, `register_dispatcher_task`, tylko jako adapter Celery do tego samego rdzenia.
- `intergrax/integrations/providers/message_bus/celery/config.py`, `CeleryIntegrationConfig.from_env`.
- Fabryki, które już wołają `wire_optional_queue_execution`: `applications/legal_application/host/factory.py`, `applications/dispute_sim_application/host/factory.py`. Ten sam skład ma dostać host LKW, jeśli włącza kolejkę. Nie dodać trzeciej pętli w aplikacji.

**Zadania.**

1. W postawie ścisłej `INLINE` zostaje tylko wtedy, gdy profil wprost nie żąda workera. `CELERY` wymaga niepustego `INTERGRAX_CELERY_BROKER_URL`. Brak zmiennej jest błędem konfiguracji, nie `redis://localhost`. `memory://` i `task_always_eager` w postawie ścisłej zostają odmową, jak dziś.
2. Backend kolejki w postawie ścisłej to magazyn, którego topologia spełnia host i który drugi proces potrafi otworzyć. Atomowe przejęcie wymaga `replace_if_match`. Brak compare-and-set przy więcej niż jednym workerze pada. Samo compare-and-set na magazynie w pamięci procesu też pada. Składanie woła `DocumentStoreTaskQueue`, nie klasę Mongo ani Redis.
3. Przy udanym przejęciu w tym samym compare-and-set zapisać w wierszu identyfikator ogrodzenia z księgi. `complete`, błąd i odzyskanie `RUNNING` wołają księgę z tym ogrodzeniem. `StaleClaimError` kończy starego workera bez zapisu stanu i bez wywołania zewnętrznego. `recover_interrupted_running` nie zamyka wiersza, którego dzierżawa jeszcze trwa. Po wygaśnięciu nowe przejęcie dostaje nowe ogrodzenie. Nie dodawać typu dzierżawy obok `LeaseOwnership`.
4. Kolejność przyjęcia: najpierw trwały wiersz, potem publikacja budzenia. Zapis odrzucony: brak publikacji i brak pracy w procesie. Publikacja odrzucona: wiersz zostaje `PENDING` i konsument magazynu może go wziąć. Potwierdzenie komunikatu dopiero po trwałym zapisie końca. Zapis końca odrzucony: brak potwierdzenia, ponowna dostawa. Ponowna dostawa przy już zapisanym końcu tylko potwierdza komunikat. Broker niedostępny przy przyjęciu nie włącza eager.
5. Worker dokumentu i worker Celery wołają `execute_logical_task`. Przy przejęciu porównać `tenant_id` i `TaskExecutionIdentity` wiersza z dostawą. Rozjazd kończy próbę bez handlera i bez potwierdzenia sukcesu.
6. Kafka zostaje odmową `BROKER_WIRING_NOT_BUILT`. RabbitMQ jest URL-em AMQP adaptera Celery, nie drugim konsumentem. `MessageConsumer` zostaje transportem.
7. Księga, checkpoint i pin skilla są otwierane przez worker własnym klientem tej samej konfiguracji.

**Testy.** Dwa procesy systemu operacyjnego, jeden plik magazynu, który przeżywa śmierć pierwszego procesu. Nie dwa klienty w jednym interpreterze. Pierwszy zapisuje zadanie i ginie. Drugi widzi ten sam wiersz i tę samą tożsamość. Dwa procesy wołają przejęcie naraz: handler startuje raz. Po wygaśnięciu dzierżawy stary proces woła zakończenie i dostaje `StaleClaimError`, a licznik skutku zewnętrznego nie rośnie. Zapis wiersza odrzucony nie publikuje i nie uruchamia pracy. Publikacja odrzucona zostawia wiersz do podjęcia przez drugi proces. Koniec zapisany, potwierdzenie nieposłane: ponowna dostawa nie woła handlera. Pusty URL Celery w postawie ścisłej pada. `test_fc2_queue_worker_fail_closed.py` zostaje zielony. Rozjazd tenanta nie woła handlera.

**Koniec.** Zadanie da się przyjąć, zabić proces API, zabić workera i podjąć z wiersza widzianego przez inny proces, z tą samą tożsamością, bez dwóch żywych wykonań po wygaśnięciu dzierżawy i bez utraty wiersza, gdy broker nie przyjął komunikatu.

## Pakiet C. Sesja i pamięć: trwałość z właściwości, izolacja bez przepisywania

**Problem.** Admisja sesji uznaje trwałość po typie klasy albo po fladze zadeklarowanej przez wtyczkę. `DocumentStoreSessionStorage` chowa magazyn, więc dowodu restartu nie widać. Osobny błąd projektu: interfejs warunkowego zapisu nie jest trwałością. Trwałość, atomowość, współdzielenie i izolacja klienta są niezależne. Aplikacja bezstanowa musi dalej móc nie mieć sesji trwałej.

**Po zmianie.** Ścisła aplikacja z wymaganą pamięcią trwałą startuje tylko wtedy, gdy magazyn sesji ma dowód ponownego otwarcia po restarcie. Gdy profil wymaga równoczesnych zapisów, magazyn musi dodatkowo umieć warunkowy zapis. Gdy profil wymaga wielu procesów, topologia musi być współdzielona. Brak którejkolwiek z wymaganych właściwości przerywa skład. Aplikacja bez wymaganej pamięci trwałej składa sesję w pamięci. Izolacja klienta, retencja i kasowanie zostają mechanizmami, które już są.

**Rozwiązanie.** Cztery bramki, każda tylko dla zastosowania, które jej potrzebuje.

- Trwałość: `MemoryProviderDurabilityEvidence` rodzaju `RESTART_REOPEN` dla konkretnego magazynu. Nie `isinstance`, nie `memory_provider_durability() == DURABLE`, nie `ConditionalDocumentStore`.
- Atomowość: `replace_if_match`, gdy to zastosowanie ma równoczesnych pisarzy. Nie jest dowodem trwałości.
- Współdzielenie: `PersistenceTopology.SHARED_MULTI_HOST`, gdy zadanie ma być widoczne w więcej niż jednym procesie. Trwały plik na jednym hoście zostaje legalny dla aplikacji jednohostowej.
- Izolacja: istniejące `MemoryControlAccessDenied` i `task_namespace_for_tenant`. Nie przepisywać.

Nie pisać nowego magazynu sesji, pamięci długoterminowej ani epizodycznej.

**Komponenty.**

- `intergrax/applications/_shared/memory_provider_admission.py`, `_session_storage_is_durable`, `validate_organization_and_session_admission`.
- `intergrax/runtime/nexus/session/document_store_session_storage.py`, `DocumentStoreSessionStorage`.
- `intergrax/memory/contracts/provider_durability_evidence.py`, `MemoryProviderDurabilityProofKind.RESTART_REOPEN`.
- `intergrax/applications/_shared/memory_wiring.py`, gałąź Mongo i gałąź braku magazynu.

**Zadania.**

1. Udostępnić admisji magazyn sesji dokumentu, tylko do odczytu, żeby dało się sprawdzić dowód i topologię. Klucze `session:{id}` zostają.
2. Sesja wymagana jako trwała przechodzi wyłącznie przy dowodzie `RESTART_REOPEN`. Deklaracja `DURABLE` i sam warunkowy interfejs w postawie ścisłej padają. Skrót `SQLiteSessionStorage` zostaje tylko tam, gdzie istniejący test już otwiera ten sam plik po restarcie.
3. Gdy profil nie wymaga trwałej pamięci użytkownika, admisja dalej wraca bez błędu dla sesji w pamięci.
4. Pamięć długoterminowa i profil organizacji używają tej samej bramki dowodu, nie metody `memory_provider_durability`. Istniejące magazyny SQLite i dokumentu zostają. Brak dowodu przy włączonym profilu przerywa skład. Mongo bez tej bramki dalej pada. Nie budować nowego magazynu.
5. Pamięć epizodyczna: jeśli profil ją włącza, a zdolność nie jest podłączona, odczyt pada. Nie zwracać pustego sukcesu i nie budować nowego indeksu. Niezgodny tenant indeksu tur zostaje odmową, która już jest.
6. Retencja i kasowanie zostają przy `retention_sweep` i `MemoryControlPlane.forget`. Nie zmieniać ich kodu. W pakiecie G potwierdzić, że zamiatanie nie rusza obcego tenanta.
7. Równoczesny zapis tego samego profilu przy wielu procesach wymaga atomowości i współdzielenia. Magazyn, który ich nie ma, w takim profilu pada. Aplikacja jednohostowa bez równoczesnych pisarzy nie dostaje tego wymogu.

**Testy.** `test_mem_audit5a_production_provider_admission.py` zostaje zielony. Sesja z dowodem restartu w postawie ścisłej przechodzi. Sesja na warunkowym magazynie bez dowodu restartu pada. Sesja w pamięci bez wymogu trwałości przechodzi. Włączona pamięć epizodyczna bez implementacji pada. Obcy tenant na płaszczyźnie pamięci, jeśli test już jest, zostaje uruchomiony i nie jest pisany drugi raz, gdy jest zielony.

**Koniec.** Ścisła aplikacja nie startuje na pamięci, której trwałość jest tylko nazwą albo interfejsem zapisu. Aplikacja bezstanowa nadal startuje. Izolacja, retencja i kasowanie nie dostają nowego kodu.

## Pakiet D. Instrukcja i polityka skilla są wersją z chwili startu

**Problem.** Resolver zbiera identyfikatory promptów i fragmentów polityki, ale przebieg ich nie czyta. Mostek `skill_prompt_metadata` i `merge_skill_policy_fragments` woła tylko własny test, a fragment polityki wkłada jako słownik `source` i `id`. Wznowienie, które rozwiązałoby sam identyfikator, wzięłoby dzisiejszą treść stabilną, nie treść z chwili pinu. Skill nie może sam dodać narzędzia, i ta odmowa już jest. Nie wolno jej osłabić.

**Po zmianie.** Dwa hosty z tą samą paczką dostają te same instrukcje i te same reguły. Dwa skille z tym samym identyfikatorem promptu i różną wersją nie składają się. Dwa fragmenty, które ustawiają tę samą regułę na różne wartości, nie składają się. Skill nie włącza narzędzia spoza profilu hosta, nie wyłącza obowiązku decyzji człowieka i nie zamienia odmowy platformy, aplikacji albo klienta na zgodę. Wznowienie ładuje prompt w wersji zapisanej w pinie i porównuje skrót. Inna treść pod tym identyfikatorem przerywa wznowienie.

**Rozwiązanie.** Rejestr promptów już umie `resolve(prompt_id, pin)`. Pin wykonania już trzyma wersję skilla. Rozszerzyć ten sam rekord o wersję i skrót każdego promptu oraz o skrót każdego fragmentu polityki. Składanie idzie w stałej kolejności: reguły platformy, potem aplikacji, potem klienta, potem fragmenty skilli. Fragment musi istnieć w katalogu hosta. Skill może tylko dodać ograniczenie co najmniej tak ostre jak reguła nadrzędna. Poluzowanie przerywa wiązanie. Przy skutku ubocznym zostaje istniejące rozstrzygnięcie konserwatywne w `runtime_policy_engine`: odmowa wygrywa ze zmianą. Nie dodawać silnika polityk.

**Komponenty.**

- `intergrax/applications/_shared/skill_bridge_wiring.py`, `skill_prompt_metadata`, `merge_skill_policy_fragments`.
- `intergrax/applications/_shared/environment_wiring.py`, miejsce, gdzie powstają `prompt_registry` i paczka polityk.
- `intergrax/prompts/registry/yaml_registry.py`, `YamlPromptRegistry.resolve`.
- `intergrax/prompts/registry/pin_config.py`, `PromptPinConfig`.
- `intergrax/skills/execution_binding.py` i `execution_pinning_persistence.py`, zapis i odczyt pinu.
- `intergrax/skills/registry/tool_requirements.py`, `assert_skill_tool_requirements_satisfied`. Zostaje bez poluzowania.
- `intergrax/runtime/policy/declarative_enforcer.py`, `DeclarativePolicyEnforcer`. Dostaje reguły z katalogu, nie ze stubu.

**Zadania.**

1. Przy wiązaniu paczki, dla każdego `prompt_instruction_id`, wywołać `resolve` bez polegania na późniejszej zmianie stabilnej wersji. Zapisać w pinie identyfikator, numer wersji i skrót treści. Brak promptu przerywa wiązanie.
2. Przy wznowieniu zbudować `PromptPinConfig` z pinu i wołać `resolve` z tym pinem. Skrót inny niż zapisany przerywa wznowienie.
3. Fragment polityki musi istnieć w katalogu hosta. Do pinu wchodzi skrót rozwiązanej definicji. Przy wznowieniu definicja jest użyta tylko przy zgodnym skrócie. Stub `{"source","id"}` znika.
4. Konflikt skilli: ten sam prompt, różna wersja, błąd składu. Ta sama reguła, różna wartość, błąd składu. Kolejność resolvera nie jest ostatnim wygranym.
5. Konflikt z polityką nadrzędną: próba wyłączenia `require_human_on_critical`, dodania narzędzia, poszerzenia tenanta albo zamiany odmowy na zgodę przerywa wiązanie. Zaostrzenie, które jest w katalogu, przechodzi.
6. Narzędzia: istniejąca asercja wobec `ToolProfile`. Skill nie dopisuje identyfikatora narzędzia.

**Testy.** Nieznany prompt przerywa wiązanie. Dwie wersje jednego promptu w jednej paczce przerywają skład. Po zmianie treści w rejestrze pod tym samym identyfikatorem wznowienie pada i nie dostaje nowej treści. Fragment spoza katalogu pada. Dwa fragmenty z tą samą regułą i inną wartością padają. Skill, który zdejmuje obowiązek decyzji człowieka albo zamienia odmowę na zgodę, pada i przebieg nie dostaje szerszego uprawnienia. Test wymagania narzędzia spoza profilu zostaje czerwony dla naruszenia. Istniejące testy pinowania wersji skilla zostają zielone.

**Koniec.** Przebieg i jego wznowienie widzą tę samą instrukcję i tę samą regułę. Skill nie osłabia polityki platformy, aplikacji ani klienta i nie poszerza narzędzi.

## Pakiet E. Ścisły host nie składa lokalnego MCP

**Problem.** Domyślny tryb `HARNESS_BOUND` już odrzuca tenanta od klienta. Testy `test_ten_07a` i `test_nt_ten_07` są zielone. Odmowa trybu `LOCAL_DEV_EXPLICIT` działa tylko, gdy ktoś poda `fails_closed=True`. Skład hosta tej flagi nie bierze z postawy. Przebieg `run_agent` jej nie dostaje, więc skład i wykonanie mogą się rozjechać.

**Po zmianie.** Każdy serwer MCP złożony przy ścisłej postawie hosta odmawia trybu lokalnego w momencie składania. Ta sama flaga obowiązuje przy `run_agent` i przy narzędziach katalogu. Laboratorium bez ścisłej postawy nadal może złożyć tryb lokalny, i tylko dla loopbacku, jak dziś.

**Rozwiązanie.** Jedna flaga z `wiring_fails_closed(env)` wchodzi do istniejącego `assert_mcp_identity_composition` i do `resolve_mcp_run_agent_identity`. Nowego principalu nie ma.

**Komponenty.**

- `intergrax/applications/_shared/wiring_posture.py`, `wiring_fails_closed`.
- `intergrax/applications/_shared/mcp_harness_principal.py`, `assert_mcp_identity_composition`, `resolve_mcp_run_agent_identity`.
- `intergrax/applications/_shared/mcp_application_server.py`, `compose_application_mcp_server`.
- `intergrax/applications/_shared/mcp_nexus_server.py`, `build_nexus_mcp_server`, `execute_mcp_agent_task_with_resolved_identity`.
- `intergrax/applications/_shared/mcp_server_engine.py`, `compose_mcp_server_from_definition` i narzędzie `run_agent`.
- `intergrax/scaffold/canonical_host_templates.py`, `render_canonical_mcp_server_py`.
- `applications/local_workspace_application/mcp/server.py`, `build_local_workspace_mcp_server`.
- Każda fabryka hosta, która woła te funkcje. Szukanie wywołań `build_nexus_mcp_server` i `compose_application_mcp_server` jest częścią zadania, nie decyzją do podjęcia później.

**Zadania.**

1. Zapis flagi na obiekcie zależności serwera, ustawianej wyłącznie z postawy hosta.
2. `compose_mcp_server_from_definition` woła `assert_mcp_identity_composition` z tą flagą, zanim zarejestruje narzędzia.
3. `run_agent` i wywołanie narzędzia katalogu przekazują tę samą flagę do rozwiązania tożsamości.
4. Szablon scaffoldu i serwer LKW nie mają własnego domyślnego `False`, gdy host jest ścisły. Poza postawą ścisłą domyślny tryb zostaje `HARNESS_BOUND`.
5. Nie zmieniać `HarnessAuthState` ani sposobu weryfikacji klucza.

**Testy.** Ścisły profil i `LOCAL_DEV_EXPLICIT` podnoszą `McpIdentityCompositionError` na `compose_application_mcp_server`, na `build_nexus_mcp_server` i na `compose_mcp_server_from_definition`. `test_ten_07a` i `test_ten_07d` zostają: domyślny tryb odmawia cudzego tenanta, tryb lokalny bez flagi działa tylko poza postawą ścisłą i na loopbacku.

**Koniec.** Nie ma punktu składania ani wykonania MCP, który przy ścisłym hoście przyjmie tenanta od klienta.

## Pakiet F. Skutek zewnętrzny po śmierci procesu

**Problem.** Samo podłączenie księgi do workera nie wystarcza, jeśli księga żyje w pamięci procesu albo jeśli skutek idzie obok invokera. W invokerze kolejność już jest właściwa: `before_external_effect`, potem `admit_external_effect_may_have_started`, potem wywołanie, potem zapis zakończenia. Punkt kontrolny `AFTER_BACKEND_BEFORE_EFFECT_COMMIT` przy utracie procesu oznacza skutek jako mogły się zacząć i nie powtarza go w ciemno. Luka: worker poza procesem musi widzieć ten zapis po śmierci API, a handler zadania nie może wywołać systemu zewnętrznego z pominięciem invokera.

**Po zmianie.** Platforma rozróżnia pięć sytuacji i tylko te przejścia są legalne.

- Nie rozpoczęto: w księdze nie ma bitu „mógł się zacząć”. Odzyskanie może zdjąć roszczenie i pozwolić na nowe wywołanie.
- Rozpoczęto, wynik nieznany: bit jest zapisany, a zakończenia nie ma, albo status to nieznany. Drugi proces nie woła systemu zewnętrznego. Dalszy krok to uzgodnienie albo decyzja uprawnionego użytkownika.
- Wykonano: księga ma wynik zakończony. Ponowna dostawa odtwarza wynik i nie woła systemu.
- Niepowodzenie, skutek na pewno nie wszedł: roszczenie jest zdejmowane, wywołanie wolno powtórzyć w granicach istniejącego budżetu.
- Kompensacja albo decyzja człowieka: kompensacja jest kolejną operacją zewnętrzną. Ze stanu nieznanego nie startuje sama. Najpierw istniejące uzgodnienie, które nie powtarza mutacji. Gdy wynik nadal nieznany, zostaje eskalacja do uprawnionego użytkownika albo stan bez ruchu. Kompensacja wykonuje się dopiero, gdy rozstrzygnięcie ma akcję wymagającą kompensacji, dowód jest ostateczny, a decyzja jest autoryzowana. Niewystarczający dowód zostaje odroczeniem, nie wywołaniem wtyczki.

Awaria po udanym wywołaniu, a przed zapisem potwierdzenia, zostaje w stanie „wynik nieznany”. Drugi proces nie wykonuje wywołania ponownie i nie uruchamia kompensacji.

**Rozwiązanie.** Nie zmieniać automatu księgi. `IdempotencyStore.claim`, `admit_external_effect_may_have_started_with_claim`, `complete_with_claim`, `mark_uncertain_with_claim` i `abandon_pre_effect_with_claim` zostają. Worker z pakietu B otwiera `SQLiteIdempotencyStore` albo `RedisIdempotencyStore` zgodnie z topologią hosta. Magazyn w pamięci przy workerze poza procesem pada przy składaniu. Skutek narzędzia zostaje w `RuntimeToolInvoker.invoke`.

**Komponenty.**

- `intergrax/runtime/nexus/tools/invoker.py`, `RuntimeToolInvoker.invoke`, `_execute_external_effect`.
- `intergrax/runtime/tools/idempotency_pre_effect_coordinator.py`, `IdempotencyPreEffectCoordinator`.
- `intergrax/runtime/tools/sqlite_idempotency_store.py`, `SQLiteIdempotencyStore`.
- `intergrax/distributed/providers/redis_idempotency_store.py`, `RedisIdempotencyStore`, tylko gdy topologia hosta już go wybiera. Nie dodawać nowego dostawcy.
- `intergrax/queueing/worker/execution.py`, `execute_logical_task`.
- `intergrax/applications/_shared/reliability_wiring.py`, `wire_application_reliability`.
- `intergrax/runtime/enterprise_reliability/provider_invocation_recovery.py` i kompensacja. Wołane ze stanu nieznanego, nie zastępowane.

**Zadania.**

1. Uruchomić istniejące testy księgi skutku, w szczególności `test_idempotency_durable_effect_admission_r5_9_r3.py`. Zielony wynik oznacza, że automatu się nie przepisuje.
2. Skład workera z pakietu B odrzuca księgę, której `persistence_topology` nie spełnia topologii hosta. Reguła `topology_satisfies` już jest w `reliability_assembly_resolver.py`.
3. Handler zadania ze skutkiem zewnętrznym woła `RuntimeToolInvoker`. Dodać asercję w rejestrze handlerów albo w istniejącym kontrakcie narzędzia: skutek uboczny bez invokera nie ma ścieżki wykonania w workerze. Nie budować równoległego wywołania HTTP w handlerze.
4. Test dwóch procesów na jednym pliku SQLite: pierwszy przechodzi admit, udaje powrót systemu i ginie przed `complete_with_claim`. Drugi otwiera ten sam plik, dostaje stan nieznany i licznik wywołań dubla systemu zostaje bez wzrostu.
5. Osobny przypadek: śmierć przed admit, bit nieustawiony, drugi proces zdejmuje roszczenie i wolno mu wywołać raz.
6. Przypadek zakończony: drugi proces dostaje zapisany wynik i nie woła systemu.
7. W `DefaultFailClosedProviderInvocationRecoveryPolicy.decide` nie wybierać `allowed[0]`, gdy ta akcja mutuje, a wynik jest nieznany. Kolejność zostaje: uzgodnienie, potem eskalacja do człowieka, w przeciwnym razie brak akcji. `execute_external_effect_compensation` nie jest wołane ze stanu nieznanego bez ostatecznego dowodu i autoryzowanej decyzji. `evaluate_resolution_disposition` już odracza niewystarczający dowód. Tej funkcji nie osłabiać.
8. Nie dodawać rejestru kompensacji.

**Testy.** Wymienione wyżej oraz istniejący test invokera z `SimulatedHostProcessLostError` i punktem `AFTER_BACKEND_BEFORE_EFFECT_COMMIT`. Regresja: wygaśnięcie dzierżawy bez bitu nie jest traktowane jak sukces. Stan nieznany bez ostatecznego dowodu nie zwiększa licznika kompensacji ani powtórzenia. Po ostatecznym dowodzie sukcesu i autoryzowanej decyzji kompensacja może wejść w istniejącą wtyczkę. Bez tej decyzji zostaje eskalacją.

**Koniec.** Drugi proces po śmierci pierwszego nie powtarza skutku, którego potwierdzenie nie zostało zapisane, nie gubi skutku, który na pewno nie wystartował, i nie uruchamia drugiej operacji zewnętrznej tylko dlatego, że wynik pierwszej jest nieznany.

## Pakiet G. Potwierdzenie całego fundamentu

**Problem.** Pakiety A–F domykają luki, które kod potwierdził. Zamrożenie architektury obejmuje też mechanizmy, które już działają, oraz obowiązek sprawdzenia, że całość nadal się składa. Historyczny rejestr czerwonych i otwarty status w dokumencie kwalifikacji nie są listą prac. Brakujący dowód na tym drzewie jest.

**Po zmianie.** Każde obowiązkowe wymaganie zamrożenia ma dokładnie jedno rozliczenie: działa i jest potwierdzone przebiegiem, ma zadanie w pakiecie A–F, ma wyłącznie weryfikację, albo jest wyłączeniem dopuszczonym przez kanon. Jeden przebieg istniejącej bramki obejmuje drzewo po A–F. Czerwony wynik, który jest rzeczywistym brakiem fundamentu, wraca do właściciela mechanizmu i jest naprawiany tym mechanizmem. Czerwony wynik, który jest tylko brakiem niezależnego wpisu akceptacji, zostaje w aneksie i nie dostaje nowego kodu.

**Nie implementować ponownie, gdy przebieg jest zielony.**

- Domyślna tożsamość MCP odmawia tenanta od klienta: `test_rw_tenant_mcp_client_asserted_tenant.py`, `test_nt_ten_07_mcp_client_asserted_tenant_identity_must_fail_closed`. Ostatni wspólny przebieg z kolejką, pinem skilli i admisją pamięci: 73 passed.
- Pin wersji skilla: `test_skill_x_batch_c_versioning_provenance_recovery.py`.
- Admisja pamięci produkcyjnej: `test_mem_audit5a_production_provider_admission.py`.
- Odmowa Kafki, cichego eager i eager w produkcji: `test_fc2_queue_worker_fail_closed.py`.
- Skład hosta LKW bez kompozycji: `applications/local_workspace_application/tests/host/test_lkw_production_host_bootstrap.py`.
- Izolacja pamięci: `MemoryControlAccessDenied` przy obcym tenancie i odmowa niezgodnego tenanta indeksu tur. Jeśli testu nie ma, dopisać jeden. Kodu płaszczyzny nie zmieniać, gdy jest zielony. Retencja zostaje przy `retention_sweep`.
- Obserwowalność: istniejący test bramy transportu. Nie dodawać toru zdarzeń. Jeśli przejęcie i koniec zadania z pakietu B nie emitują na istniejącej magistrali, dodać emisję tym samym `RuntimeEventBus`, z identyfikatorem zadania i tenantem. To jedyna dopuszczalna zmiana obserwowalności.
- Granice warstw, jeden właściciel wykonania, kontrakty bez drugiej kolejki i drugiej tożsamości: pakiety A–F nie dodają tych mechanizmów. Bramka importów, która już jest w repozytorium, jest uruchamiana. Czerwony import odwrotny jest naprawiany w miejscu naruszenia, bez nowej warstwy.

**Zadania.**

1. Po A–F jeden przebieg istniejącej bramki. Jeśli kolekcja pada, bo brakuje grupy zależności już zapisanej w projekcie, dołożyć instalację tej grupy albo wyłączyć moduł w sposób, który nie liczy się jako zaliczenie. Nie pisać nowego runnera.
2. Testy, które mutują globalny rejestr i przechodzą solo, a padają w zestawie, naprawić istniejącym sprzątaniem rejestru.
3. Dla każdego wiersza wymagań w aneksie wykonać przypisane potwierdzenie albo zostawić wyłączenie. Nie otwierać historycznego rejestru, którego ten przebieg nie powtórzy.
4. Jeżeli bramka pokaże brak fundamentu spoza A–F, naprawić go istniejącym mechanizmem i dopisać ten brak do aneksu jako zadanie, nie jako nowy pakiet produktowy.

**Koniec.** Bramka na tym drzewie ma zapisany wynik. Aneks nie ma wiersza bez rozliczenia. Pakiety A–F są zielone w tym przebiegu albo w przebiegu bezpośrednio przed nim, jeden po drugim, bez równoległych runnerów.

## Zależności i organizacja pracy

Pliki się nie krzyżują między A, C, D i E. B i F dzielą wykonanie workera, więc idą jednym ciągiem u jednego wykonawcy.

- A: magazyny wektorowe i resolver profilu integracji.
- C: admisja pamięci, sesja dokumentu, wiązanie pamięci.
- D: mostek skilli, rejestr promptów, pin, egzekutor polityk.
- E: składanie MCP, postawa hosta, szablon i serwer lokalnego obszaru roboczego.
- B, potem F: kolejka, dispatcher, fabryki hostów, księga i odzyskanie wywołania.
- G na końcu, na drzewie po złożeniu reszty.

W osobnych drzewach A, C, D i E mogą iść równolegle. B startuje, gdy te cztery nie edytują już jego plików. F nie startuje przed B. W jednym drzewie kolejność: E, A, C, D, B, F, G.

Jeden przebieg pytest naraz na maszynie, także gdy drzewa są osobne. Wykonawca pakietu uruchamia tylko testy swojego pakietu, gdy żaden inny przebieg nie działa. Pełna bramka jest raz, w G.

Agent badawczy nie jest potrzebny. Jeden wykonawca na pakiet, z listą plików i testów z tego pakietu. Przegląd czyta diff wyłącznie pod kryterium końca pakietu. Nie dzielić B i F.

## Aneks. Rozliczenie wymagań i kwalifikacja

Główny plan jest w języku produktu. Tu każde obowiązkowe wymaganie zamrożenia architektury ma jedno rozliczenie. Identyfikatory służą śledzeniu. Nie organizują pracy. Źródło listy: lista akceptacji zamrożenia. Wiersz architektury musi zostać potwierdzony albo mieć dopuszczone wyłączenie. Wiersze żywej kwalifikacji zostają otwarte z mocy kanonu i ten plan ich nie zamyka. Wykonawca nie wpisuje zaliczenia do listy akceptacji. Po G oddaje skrót drzewa i wynik przebiegu. Niezależna akceptacja zostaje poza tym planem.

**Zadanie implementacyjne.**

- Nieobsługiwana konfiguracja produkcyjna magazynu wiedzy: pakiet A, status katalogu, nie nowa lista. Rodzina kwalifikacji produkcji w części architektonicznej odmowy.
- Trwałość, atomowe przejęcie, ogrodzenie po dzierżawie, rozjazd brokera i wiersza, awaria zapisu, publikacji i potwierdzenia: pakiet B.
- Pięć stanów skutku i zakaz drugiej operacji zewnętrznej przy wyniku nieznanym: pakiet F.
- Trwałość sesji z dowodu, osobno atomowość i współdzielenie, odmowa niepodłączonej pamięci epizodycznej: pakiet C.
- Instrukcja i polityka z chwili startu, brak osłabienia polityk nadrzędnych: pakiet D.
- Składanie i wykonanie MCP przy ścisłym hoście: pakiet E.

**Tylko weryfikacja, bez nowego kodu, gdy przebieg jest zielony.**

- Izolacja pamięci klienta, przestrzeń nazw zadania, retencja i kasowanie przez istniejącą płaszczyznę.
- Domyślna odmowa tenanta MCP, pin wersji skilla, odmowa nietrwałej pamięci, odmowa Kafki i eager, skład hosta bez kompozycji.
- Granice warstw i jeden właściciel wykonania, kolejki, tożsamości i katalogu. Bramka importów już w repozytorium.
- Ciągłość śladu przy restarcie, jeśli istniejący test tej rodziny jest zielony. Historyczne zaliczenie nie jest dowodem tego drzewa, dopóki G go nie powtórzy.
- Skaler bez adresu klastra już pada. Nie przepisywać.
- Eksport obserwowalności istniejącą bramą transportu. Emisja przejęcia zadania tylko na istniejącej magistrali, i tylko gdy jej nie ma.

**Wyłączenie dopuszczone przez kanon.**

- Żywa kwalifikacja produkcji, kont dostawców i profilu operatorskiego: pierwsze dwa wiersze tej rodziny oraz żywe obowiązki w wierszach konfiguracji i bezpieczeństwa, które lista zostawia otwarte do bramki żywej. Architektoniczna odmowa złej konfiguracji zostaje w pakiecie A i B.
- Szyfrowanie, rezydencja danych i ekspozycja u dostawcy, gdy właścicielem nie jest platforma: wyłączenie z dowodem, nie implementacja.
- Fabryka Scenariuszy, Studio, dopuszczenie scenariusza, pamięć proceduralna jako nowy magazyn, organizacja na Mongo bez bramki dowodu, sandboksy usług, Kafka jako osobny konsument, pełne SDK Milvusa i Weaviate.
- Reszty protokołu pamięci opisane w kanonie jako nieoddane i bez publicznej kwalifikacji: nie budować nowego podsystemu. Włączenie zdolności bez implementacji pada w pakiecie C.

**Kwalifikacja po zielonym G.**

1. Zapisać skrót drzewa i wynik jednego przebiegu bramki oraz testów pakietów, w tym dwóch procesów systemu dla kolejki i księgi.
2. Nie zmieniać statusów listy akceptacji i nie ogłaszać zamknięcia zamrożenia.
3. Przekazać skrót i wynik do niezależnej akceptacji. Akceptacja, scalenie i ustawienia zdalne zostają poza wykonawcą.

## Co platforma będzie umiała

Aplikacja wielu klientów wystartuje tylko z tożsamością, której klient sam nie ustala, i z pamięcią, której trwałość da się wykazać osobno od atomowości zapisu i osobno od współdzielenia między procesami. Zadanie przyjęte od użytkownika przetrwa restart aplikacji i workera, nie wykona się dwa razy naraz po wygaśnięciu dzierżawy i nie zniknie, gdy broker nie przyjmie komunikatu. Wywołanie systemu zewnętrznego, które mogło się udać, nie zostanie powtórzone ani skompensowane tylko dlatego, że potwierdzenie nie zostało zapisane. Skill niesie tę samą instrukcję przy wznowieniu i nie osłabia reguł platformy, aplikacji ani klienta. Magazyn, którego katalog nie oznacza jako stabilny, nie udaje działania na słowniku w procesie.

Na tym da się budować produkty, w tym Fabrykę Scenariuszy, bez dokładania drugiego wykonania, drugiej kolejki i drugiej tożsamości. Fabryka nadal nie jest częścią tego planu. Żywy dostawca u klienta nadal wymaga osobnej kwalifikacji środowiska, i ten plan jej nie zastępuje.
