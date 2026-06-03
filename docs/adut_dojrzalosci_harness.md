Audyt dojrzałości Harness Intergrax (bez agentów biznesowych)
Zakres: Tier-0 platforma, Tier-1 Nexus, Tier-3 lab_application / scaffold, katalogi Integration → Tool → Skill, środowisko harness (Phase L–V, P-Ext). Poza zakresem: K.1/K.2, głęboka logika legal / problem_radar / aplikacje produktowe — traktowane jako konsumenci harnessu, nie miara OS.

Źródła: implementacja (intergrax/, applications/lab_application/), intergrax_runtime_architecture.md, IDEAL_HARNESS_AI_ARCHITECTURE.md, INTERGRAX_IMPLEMENTATION_PLAN.md, HARNESS_ENVIRONMENT.md, EXTENSION_AUTHOR_GUIDE.md.

1. Werdykt w skrócie
Wymiar	Ocena	Komentarz
Deklaracja planu
Harness completion Done
Q→V + P-Ext + W-OPS zamknięte (kod); gate 469 (full regression)
IDEAL L0–L4 (§12)
L2+ → L3 (kod/kontrakty)
Pełne L3 operacyjne i L4 adaptacyjne — nie (2 cykle release + SLO/incydenty)
Agent OS (Appendix A)
L1 certyfikowany
20/20 — ścieżka tworzenia agenta + lab + trace (agenty referencyjne)
Zgodność z IDEAL North Star
Wysoka na policy-first, trace, composability; średnia na identity enterprise i pełną odporność produkcyjną
Jednozdaniowy werdykt: Intergrax jest dojrzałym laboratorium / scalable harness (L2+) z szerokim pokryciem kontraktów L3/L4 w kodzie i CI, ale nie jest jeszcze operacyjnym Production Harness OS (L3) w sensie IDEAL (SLO, dwa stabilne cykle release, pełna tożsamość multi-tenant w ścieżce wykonania).

2. Model relacji (zgodnie z IDEAL §0.2)
Harness product
Out of audit scope
reference only
deferred
deferred
echo / signoff_probe / research mock
Tier-0 catalogs + adapters
Tier-1 Nexus UAEP
Tier-3 lab_application
Business agents K.1/K.2
Product applications
Harness = Nexus + platforma + wiring aplikacji — zgodne z canon §5.3 i IDEAL §17 (profil LLM, modality, skills, policy, context, memory, tools).

3. Macierz IDEAL (§12.2) — ocena 0–4 na obszar
Skala: 0 = L0 Fragmented · 1 = L1 MVP · 2 = L2 Scalable · 3 = L3 Production · 4 = L4 Adaptive

Obszar IDEAL	Poziom	Dowody w implementacji	Luki vs IDEAL
Interface / API
3
POST /v1/lab/run, /debug/*, interaction intake, MCP (couple_fastapi_with_mcp), UnifiedTaskRunner
Brak pełnego API produktowego multi-tenant; streaming/SLA queue jako produkt
Identity / Trust
2
INTERGRAX_HARNESS_API_KEY, middleware, require_harness_api_key (U-Sec)
RBAC/ABAC per tenant w runtime, delegacja, signing — głównie kontrakty V-SEC, nie pełna ścieżka prod
Policy / Governance
3
PolicyEngine, RuntimePolicyBundle, ToolAccessPolicy, HITL, interrupt handler, bridge do RuntimeConfig (U/R)
Adaptive policy learning = kontrakty (adaptive_governance.py), nie pętla produkcyjna
Orchestration
3
NexusLoop, GraphExecutor, delegation (R-Delegate), long-running scheduler, RetryCoordinator / RetryEngine
Brak centralnego schedulera SLA/kolejki jak w IDEAL §3.4; fan-out ograniczony testami
LLM / Cognition
3
llm_adapters, CatalogToolPlanner, ToolPlanningService bez importu tools_agent (U-Typ.2)
ToolsAgent nadal istnieje (legacy, zamrożony); routing modeli po koszcie/ryzyku — podstawowy
Tools / Skills / Integrations
3
P-Ext Done: ~99 slugów, 13/13 ToolPlugin, 3/3 SkillPlugin, EP fixture, bootstrap_catalogs, resolve_typed
Nie każdy slug = prod-hardened; requires_skills bez adopcji w shipped manifests
Memory / RAG
2–3
MemoryView, task memory, RAG tools, Graph-RAG kontrakty (V-KG)
Graph RAG jako first-class reasoning w prod — głównie architektura + testy kontraktowe
Context engineering
3
ContextBudgetPolicy, CONTEXT_* events (R-Context), V-CE scoring/dedup
Context regression w CI — benchmarki, nie ciągły monitoring prod
Reliability / Runtime
2
Checkpoints, resume HITL, run-level retry (max_run_retries), long-running
Idempotency + integration circuit breaker — **Done** (W-OPS.1–2); compensation paths — częściowe
Observability / Ops
3
Trace store, runtime events, debug API, OTEL profile (noop w CI), metrics routes
SLO/alerting/on-call runbooks — poza gate; L3 operacyjny pending w planie
Registries
3
6 rejestrów IDEAL §19 (agent/tool/skill/integration + prompt + eval w V)
Evaluation registry = trendy/artefakty, nie pełny online scoring
Prompt engineering
3
YamlPromptRegistry, V-PE composition/regression kontrakty
Prompt Lab / pełna governance UI — nie
Security / data
2–3
V-SEC adversarial baseline w maturity_gate_evidence, sandbox opt-in (U-Sec.3)
Tenant isolation zweryfikowana w harness baseline, nie end-to-end prod
Cost / resources
3
RunBudget, V-COST quota/deny/forecast kontrakty
Optymalizacja kosztów L4 — symulacje, nie closed-loop prod
Developer experience
3
intergrax.scaffold (agent/app/skill/integration/tool), lab, extension guide
Replay Lab / Agent simulator — częściowo (debug trace), bez pełnego DX IDEAL §22
Testing / quality
3
pytest -m gate 486, check_plugin_catalog 19, check_harness_no_getattr, phase_v guards
Chaos/multi-agent simulation — ograniczone; brak pełnej piramidy IDEAL §10
Obszary krytyczne IDEAL (§12.3): Policy 3, Reliability 2, Observability 3 → minimum = 2 → formalna reguła IDEAL („wszystkie krytyczne ≥ 3 dla L3 release”) nie jest spełniona mimo że phase_v_closeout_gate.py zwraca l3_passed: True.

Uwaga metodologiczna: gate V-V6 używa złagodzonych progów metryk (np. modularity_score_min=0.20 w collect_harness_governance_signals) i scenariuszy syntetycznych — to dowód gotowości kontraktowej CI, nie dowód dwóch stabilnych cykli release z SLO (plan §2173–2180).

4. Warstwy IDEAL (§3) vs implementacja
Warstwa IDEAL	Stan harness	Pliki / mechanizmy
3.1 Interface
Silny w lab
lab_application/serving, interactions/router, debug app
3.2 Identity
Lab-only
harness_auth.py — opcjonalny klucz
3.3 Policy + HITL
Silny
policy_engine.py, interrupts/handler.py, governance bridge
3.4 Orchestration
Silny w OS, słabszy w ops
nexus_loop.py, graph_executor, long_running/
3.5 Cognition + modality
MVP+
llm_adapters, Phase W-ML, MODALITY.md, tools vision.* / speech.*
3.6 Capability
Najsilniejszy filar
P-Ext, ToolRuntime, SkillResolver, IntegrationProfile + typed accessors
3.7 Memory / MCP
Lab-strong
task memory, RAG bundles, MCP export
3.8 Reliability
Średni
checkpoint + retry; brak pełnego IDEAL §8.3 SLO
3.9 Observability
Lab-strong, prod-ops pending
trace/events/OTEL
5. Cztery tiery (canon §5.1) — bez agentów biznesowych
Tier	Dojrzałość	Uzasadnienie
Tier-0
Wysoka (L3 kod)
Katalogi pluginów zamknięte; silne typowanie (IntegrationBinding, protokoły); conflict policy
Tier-1
Wysoka (L2+ / ~99% §42)
UAEP, hooks, tool gateway, decomposition po Q+; zero getattr w ścieżkach harness audit
Tier-2 (referencyjny)
Poza oceną produktu
echo, signoff_probe, research mock = sondy harnessu, nie KPI biznesu
Tier-3
Wysoka (lab GA)
lab_harness_preset(), lifespan (bez on_event), auth/MCP/sandbox świadomie domyślne
Anti-patterns IDEAL §13 — status:

Anti-pattern	Status
Monolityczny agent-god
Mitigowany — UAEP + skill_ids
Tool bez policy/audit
Mitigowany — ToolRuntime, policy resolution
Brak timeout/retry
Częściowo — retry run-level; nie wszędzie bounded
Brak izolacji subagentów
Mitigowany — DelegationSpec + namespace
Observability „best effort”
Słabsze w prod ops — w lab obowiązkowe
Registry bypass
Mitigowany — P-Ext + capability graph guard
Context ad hoc
Mitigowany — ContextBudget + events
6. Fazy planu — mapowanie na dojrzałość
Faza	Wpływ na harness	Stan
L
Agent OS scaffold + lab
Done — Appendix A 20/20
Q / Q+
Jakość, getattr, monolity
Done
R
Skill Library, context, delegation, policy bundle
Done (MVP)
S / T / U
Środowisko lab, preset, security, typowanie
Done
V / V-V6
Kontrakty L3/L4, capability graph, eval/security/cost
Done (kod); ops L3/L4 pending
P-Ext
Plugin catalogs EP
Done (61/61)
W-ML
Modality plane
Done (wg planu)
K / Band 3
Agenty biznesowe
Deferred — słusznie poza tym audytem
7. Dowody ilościowe (stan 2026-06-02)
Metryka	Wartość
Gate tests
486 passed (pytest -m gate)
Plugin smoke
19 passed (check_plugin_catalog.py)
Integracje (full preset)
≥ 95 slugów (test MIN_FULL_INTEGRATIONS)
Tool bundles
≥ 13, skill bundles ≥ 3, skill_ids ≥ 9
CI harness audits
check_harness_no_getattr, check_tools_agent_*, phase_v_*_gate
Closeout syntetyczny
l3_passed / l4_passed True (kontrakty)
8. Luki i dług (priorytetyzacja)
P0 — blokuje operacyjne L3 (IDEAL §12.3)
Reliability produkcyjna — idempotency side-effectów, circuit breakers na integracjach, jawne SLO + incident budget (plan: 2 release cycles).
Identity / Trust w ścieżce wykonania — ponad opcjonalnym API key: tenancy, scope per run, audit impersonation.
Rozróżnienie „L3 CI” vs „L3 ops” — komunikacja release board; nie traktować maturity_gate_evidence jako zamiennika PRR/SLO.
P1 — Band 2 (on demand, zgodnie z planem)
R-Skill — rozszerzenie harness.* i adopcja requires_skills.
M.6 — nowe slugi integracji z pełnym health + testami kontraktowymi.
W-ML ops — opcjonalne skalowanie workerów Celery (Band 2b).
Online evaluation — shadow registry (`online_evaluation_observations.json`) **Done**; A/B produkcyjne — on demand.
P2 — higiena
ToolsAgent — ścieżka legacy zamrożona; długoterminowo usunięcie po migracji ostatnich konsumentów.
load_callable w applications/_shared/wiring.py — dynamiczny import poza audytem getattr (niski risk, ale luźniejsze niż reszta harnessu).
Metryki architektury — compute_architecture_metrics report-only; progi informacyjne do czasu hard enforcement.
9. Zgodność z zasadami IDEAL (§2 North Star)
Zasada	Ocena
Policy-first
Silna — ToolRuntime + PolicyEngine + bundle
Composable-by-default
Silna — tiery, plugin catalogs, scaffold
Trace-everything
Silna w lab; ops dashboards — słabsze
Safe-failure
Średnia — klasyfikacja błędów, retry; pełna taksonomia IDEAL §8.1 częściowa
Deterministic-enough
Średnia — replay/debug; brak pełnego replay lab
Human-governed autonomy
Silna — HITL, interrupts, signoff flow
Progressive extensibility
Silna — P-Ext, EP, extension guide
10. Rekomendacje (kolejne kroki harness — bez K.1/K.2)

**Źródło prawdy implementacji:** [INTERGRAX_IMPLEMENTATION_PLAN.md](INTERGRAX_IMPLEMENTATION_PLAN.md) — **Phase W-OPS** + **§6.2w** (kolejność PR).

§6.1 maintenance — utrzymać gate **469** (full regression) + audyty CI po każdym PR harness.
**W-OPS (kod)** — **Done** (W-OPS.1–15): idempotency, circuit breaker, SLO, shadow eval + file registry, lab stack health, harness skills.
**Operacyjne L3** — `record_harness_release_cycle.py` × 2 cykle → `phase_w_ops_evidence.py --enforce`.
**On demand** — nowe slugi M.6, Celery scale-out (W-OPS.12), Band 3 (K.1/K.2) po decyzji §6.3.
11. Podsumowanie dla stakeholderów
Intergrax spełnia definicję Harness AI z canonu i IDEAL jako Agent OS do szybkiego tworzenia i testowania agentów (Appendix A, lab, katalogi, Nexus §42). To jest najlepiej dowiedziona warstwa platformy w historii planu (Q→V + P-Ext).

Nie jest jeszcze w pełni „Production Harness OS” (IDEAL L3 operacyjny) ani „Adaptive Agent OS” (L4) — mimo że kontrakty L3/L4 przechodzą w CI. Główny dystans to Reliability w produkcji, Identity multi-tenant, ciągła ewaluacja online i dowód operacyjny (SLO + 2 release cycles), nie brak modułów w repozytorium.

Agenci biznesowi nie obniżają oceny harnessu — są świadomie odłożeni i nie powinny być kryterium „czy OS jest gotowy”; do tego służą agenty referencyjne i gate agent_os.