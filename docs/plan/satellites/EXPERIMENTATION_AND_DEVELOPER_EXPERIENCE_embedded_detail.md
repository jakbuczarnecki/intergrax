# EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE — embedded detail

**Parent hub:** [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §10, §22 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Planned** — incremental after IDEAL-L3 W2 closeout

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-26.1 | §26 CI | Architecture-boundary chaos job in weekly CI | P2 | **Done** |
| AUDIT-IDEAL-26.2 | §26 CI | Simulation tests for multi-agent contention | P2 | **Done** |
| AUDIT-IDEAL-27.1 | §27 DX | Trace Explorer interactive UI (beyond lab APIs) | P2 | **Done** |
| AUDIT-IDEAL-27.2 | §27 DX | Replay environment HTTP API on product hosts | P1 | **Done** |
| AUDIT-IDEAL-27.3 | §27 DX | Agent simulator on product hosts (not CLI-only) | P2 | **Done** |
| AUDIT-IDEAL-27.4 | §27 DX | Visual builder / graph editor (Phase 2 UI) | P3 | **Done** |
| AUDIT-IDEAL-30.2 | §30 Ops | Real deploy SLO window evidence (shared OBSERVABILITY) | P1 | **Done** |
| AUDIT-IDEAL-30.3 | §30 Ops | On-call ownership model for production components | P2 | **Done** |
| AUDIT-IDEAL-6.7 | §6 LLM (shared) | Developer `USAGE.md` + startup validation | P2 | **Partial** — [USAGE.md](../../intergrax/llm_adapters/USAGE.md) Done; `validate_runtime` [M-LLM-X.7.2](plan/LLM_ADAPTERS.md) |

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR → update this table + master register → gate green.

---

### 6.2w Phase W-OPS execution order (Band 2d — complete 2026-06-06)

**Status:** **Done** · register: [Phase W-OPS](plan/PLATFORM_FOUNDATION.md)

Work **one W-OPS ID per PR**; after each step update the W-OPS table + paydown log; keep §6.1 scripts green.

| Order | ID | Deliverable | Priority | IDEAL gap |
|-------|-----|-------------|----------|-----------|
| 1 | W-OPS.1 | Side-effect tool idempotency keys + dedup | **Critical** | Reliability §8.3 |
| 2 | W-OPS.2 | Integration circuit breaker (`_shared`) | **Critical** | Reliability §8.2 |
| 3 | W-OPS.3 | Long-running / checkpoint / retry gate tests | High | Reliability §8.3 |
| 4 | W-OPS.6 | `tenant_id` on TaskEnvelope → trace/events | High | Identity §3.2 |
| 5 | W-OPS.7 | Mandatory harness API key (staging profile) | High | Identity §3.2 |
| 6 | W-OPS.4 | SLO catalog + incident budget + runbooks | **Critical** | Observability §11 |
| 7 | W-OPS.5 | L3-ops evidence (2 release cycles) | **Critical** | §12.3 vs V-V6 CI |
| 8 | W-OPS.8 | `harness.*` platform skill packs | Medium | Capability §3.6 |
| 9 | W-OPS.9 | `requires_skills` shipped demo | Medium | Registries §19 |
| 10 | W-OPS.10 | Harness lab stable stack health (catalog slugs) | Medium | Capability §3.6 |
| 11 | W-OPS.11 | Online/shadow evaluation registry writes | Medium | Evaluation §18 |
| 12 | W-OPS.12 | W-ML Celery Tier-3 scale-out (optional) | Low | Modality §3.5.1 |
| 13 | W-OPS.13 | ToolsAgent removal roadmap | Low | Cognition hygiene |
| 14 | W-OPS.14 | Typed wiring (no `load_callable`) | Low | DX §22 |
| 15 | W-OPS.15 | Architecture metrics threshold enforcement | Low | §21.6 |

**Wave P0 (orders 1–7)** must be **Done** before declaring **operational IDEAL L3**. **Wave P1/P2** run in parallel with P0 when owners differ.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new product applications, Problem Radar wave 2+.### 6.2x Phase H-APP execution order (Band 2e — complete 2026-06-03)

**Status:** **Done** · canonical register: [Phase H-APP — Master deliverables register](#h-app--master-deliverables-register-all-43-tasks) · audit narrative: [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) §7.

Work **one H-APP ID per PR**; after each step update the H-APP master table + paydown log; keep §6.1 scripts green.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| H0 | H-APP.0.1–H-APP.0.5 | 5 | Terminology, CI guards, `poc_template` getattr fix, manifest conformance |
| H1 | H-APP.1.1–H-APP.1.8 | 8 | `ApplicationEnvironmentProfile`, unified wiring, runtime bridge, LLM resolver |
| H2 | H-APP.2.1–H-APP.2.8 | 8 | Identity, policy DSL, execution modes, V-SEC per application |
| H3 | H-APP.3.1–H-APP.3.6 | 6 | Orchestration profile, graph spec, Nexus factory, shadow/sandbox |
| H4 | H-APP.4.1–H-APP.4.8 | 8 | Context, memory, reliability, observability profiles |
| H5 | H-APP.5.1–H-APP.5.5 | 5 | Migrate lab/legal/research/poc/docker_verify + scaffold |
| H6 | H-APP.6.1–H-APP.6.3 | 3 | Operational L3 sign-off (release cycles + CI + audit §4) |
| **Total** | | **43** | |

**Suggested PR order (same as Phase H-APP paydown):** H-APP.0.3 → H-APP.1.1–H-APP.1.4 → H-APP.1.5–H-APP.1.8 → H-APP.3.4–H-APP.3.5 → H-APP.2.1–H-APP.2.8 → H-APP.4.1–H-APP.4.8 → H-APP.3.1–H-APP.3.3 → H-APP.5.1–H-APP.5.5 → H-APP.0.1–H-APP.0.5 → H-APP.6.1–H-APP.6.3.

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new **product** Tier-3 apps, Problem Radar wave 2+, marketplace UI, catalog hot-reload.

---

### 6.2y Phase DX execution order (Band 2f — mostly done)

**Status:** **Done** (2026-06-02) · **47/47 Done** · canonical register: [Phase DX — Master deliverables register](#dx--master-deliverables-register-all-47-tasks).

Work **one DX ID per PR**; after each step update the DX master table + paydown log; keep §6.1 scripts green. **Start with DX1 (scaffold/H-APP alignment)** before DX2 facades — otherwise new authors copy broken `factory.py` patterns.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| DX0 | DX-0.1–DX-0.4 | 4 | LangGraph mapping, responsibility matrix, progressive disclosure |
| DX1 | DX-1.1–DX-1.6 | 6 | **P0** — scaffold + poc/legal/research factories on H-APP path only |
| DX2 | DX-2.1–DX-2.6 | 6 | `HarnessApplication`, `AgentGraph`, `IntergraxAgent` + `@step` |
| DX3 | DX-3.1–DX-3.6 | 6 | `--minimal` stack, `intergrax run`, `doctor`, TTFRun acceptance |
| DX4 | DX-4.1–DX-4.4 | 4 | Integration presets + picker + gate tests |
| DX5 | DX-5.1–DX-5.8 | 8 | Host hooks, YAML loader, logging, event catalog, policy rule plugins |
| DX6 | DX-6.1–DX-6.5 | 5 | Tier-2 hygiene, external `intergrax init` template |
| DX7 | DX-7.1–DX-7.5 | 5 | JSON Schema + spec versioning + UI feed (Phase 2 prep) |
| DX8 | DX-8.1–DX-8.3 | 3 | `doctor --ci`, DX metrics artifact, scaffold alignment script |
| **Total** | | **47** | |

**Suggested PR order:** DX-1.1 → DX-1.2 → DX-1.3 → DX-1.6 → DX-8.3 → DX-2.1 → DX-2.2 → DX-2.3 → DX-2.5 → DX-3.1 → DX-3.2 → DX-3.5 → DX-3.6 → DX-4.1 → DX-4.4 → DX-1.4–DX-1.5 → DX-2.4 → DX-2.6 → DX-3.3–DX-3.4 → DX-5.1–DX-5.2 → DX-6.1–DX-6.2 → DX-4.2–DX-4.3 → DX-5.3–DX-5.8 → DX-6.3–DX-6.5 → DX-7.1–DX-7.5 → DX-8.1–DX-8.2 → DX-0.1–DX-0.4.

**Success gate for Phase DX full closeout:** All rows **Done** or **Won't fix**; DX-3.5 + DX-8.1 green in CI; DX-3.6 quickstart validated; DX-7.1 schemas under `build/harness_specs/`. **Core path (DX1–DX2, DX3.2–3.3, DX8.3) already meets harness authoring needs.**

**Explicitly out of NOW:** K.1, K.2, visual environment builder UI, new product Tier-3 apps, Problem Radar wave 2+.### 6.2z Phase AA execution order (Band 2g — mostly done)

**Status:** **Mostly Done** (2026-06-02) · platform **Done** · domain **Deferred** · canonical register: [Phase AA — Master deliverables register](#aa--master-deliverables-register-all-tasks).

Work **one AA ID per PR/session**; after each step update the AA master table + paydown log + conformance matrix; keep §6.1 scripts green. **Legal:** follow **hard reset** policy (AA-LEG.0.1) — no incremental preservation of legacy pipeline code.

| Wave | IDs | Count | Focus |
|------|-----|-------|--------|
| AA0 | AA-0.1, AA-0.2, AA-S0.1–AA-S0.6, AA-LG.1, AA-APP.0.1–AA-APP.0.3 | 12 | Scaffold checklist, tier guards, deploy triad standard |
| AA1 | AA-D0.1–AA-D0.7 | 7 | README, guides, TIER3_READINESS, USAGE |
| AA2 | AA-LEG.0.2–AA-LEG.3.1 | 12 | **Legal agent hard reset** |
| AA3 | AA-LEGAPP.1–AA-LEGAPP.8 | 8 | `legal_application` + deploy triad |
| AA4 | AA-ECHO.1–AA-ECHO.5 | 5 | Reference echo agent |
| AA5 | AA-SIG.1–AA-SIG.3 | 3 | Signoff probe |
| AA6 | AA-PR.1–AA-PR.5 | 5 | Problem radar (docs/hygiene; frozen feature) |
| AA7 | AA-ORG.1–AA-ORG.5 | 5 | Organization worker |
| AA8 | AA-RES.1–AA-RES.6 | 6 | Research agents |
| AA9 | AA-LABAG.1–AA-LABAG.2 | 2 | Lab mocks |
| AA10 | AA-LABAPP.1–AA-LABAPP.7 | 7 | Lab application host |
| AA11 | AA-POC.1–AA-POC.5 | 5 | POC template (canonical shell) |
| AA12 | AA-RESAPP.1–AA-RESAPP.6 | 6 | Research application host |
| **Total** | | **83** | |

**Suggested PR order:** AA-S0.2 → AA-S0.5 → AA-APP.0.1 → AA-APP.0.3 → AA-POC.1 → AA-POC.2 → AA-LABAPP.2 → AA-ECHO.2 → AA-LEG.0.3 → AA-LEG.1.1 → AA-LEG.1.2 → AA-LEG.1.3 → AA-LEG.2.1 → AA-LEG.2.2 → … → AA-LEGAPP.1–AA-LEGAPP.6 → AA-D0.1 → AA-D0.3–AA-D0.5 → AA-RESAPP.* → AA-LABAPP.1 → AA-APP.0.2 → remaining ARCHITECTURE.md rows.

**Per-application deploy triad gate (AA-APP.0.2):** for each of `lab_application`, `legal_application`, `local_workspace_application`, `poc_template_application`, `research_application` assert:

1. `docker/Dockerfile` + `docker-compose.yml` + `build-docker.sh` / `.bat`
2. `BUILD_AND_DEPLOY.md` present and matches scaffold generator output (or documented drift)
3. `ARCHITECTURE.md` § **Dependencies** lists required `pyproject.toml` extras (e.g. `harness-author`, provider-specific `llm-*`, `dev-ci` for tests)

**Doc pair gate (AA-D0.6):** for each listed Tier-2 agent and Tier-3 application assert `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md` exist and cross-link. Gate: `tests/unit/applications/test_agent_app_doc_pair.py`.

**Success gate for Phase AA platform closeout:** **Met** (2026-06-02) — conformance matrix **OK**; legal tree = scaffold; `lab_application` on `build_harness_host_runtime`; AA-APP.0.2 green; gate **533**. **Full AA register closeout** additionally requires Band 3 domain rows **Done** or explicitly **Deferred** (current policy: **Deferred**).

**Explicitly out of NOW:** K.1/K.2 implementation, Legal **live LLM** E2E (Band 3), new product hosts beyond the four listed, Legal UAEP step port (AA-LEG.2.2+) unless product reprioritizes §6.3.

---

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages/`) | P0 |
| 2 | P-Ext.0.6 | EP discovery tests (all three groups) | P0 |
| 3 | P-Ext.1.6 | Integration EP test via fixture | P0 |
| 4 | P-Ext.1.10 | Tier-3 `integration_wiring` → `bootstrap_catalogs()` | P0 |
| 5 | P-Ext.2.9–2.11 | External tool example + unit + EP tests | P0 |
| 6 | P-Ext.3.6–3.8 | External skill example + unit + EP tests | P0 |
| 7 | P-Ext.0.7 | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | P1 |
| 8 | P-Ext.4.3, 4.5, 1.8 | Conflict policy + CI smoke (incl. integration counts) | P1 |
| 9 | P-Ext.1.5, 1.7, 5.5–5.6 | Slug/docs cleanup + author guide matrix | P2 |
| 10 | P-Ext.2.12, 3.9–3.11 | Tool/skill lazy bootstrap, scaffold plugin template, importer docs | P2 |
| 11 | P-Ext.1.3a, 1.4, 1.9, 1.11–1.12 | Typed resolve expansion, health API, integration wiring helper | P3 |
| 12 | P-Ext.5.1, 3.10, 3.12 | Scaffold CLI (all three catalogs) + harness `requires_skills` demo | P3 |

Full task register: [Appendix I](plan/PLATFORM_FOUNDATION.md).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2ag Phase M.6 P6 execution order (Band 2ac — Done)

**Status:** **Done** (2026-06-02) · register: [M.6 P6](#m6-p6--harness-integration-expansion-planned) · queue: [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done)

```text
Wave H-INT-0 (categories):  M-P6-CAT.1 → M-P6-CAT.2 → M-P6-CAT.3 → M-P6-CAT.4 → M-P6-CAT.5 → M-P6-CAT.6 → M-P6-CAT.7 → M-P6-CAT.8 → M-P6-CAT.9
Wave H-INT-10 (security):   M-P6.1 → M-P6.2 → M-P6.3 → M-P6.4
Wave H-INT-11 (sandbox):    M-P6.5 → M-P6.6 → M-P6.7
Wave H-INT-12 (identity):   M-P6.8 → M-P6.9 → M-P6.10
Wave H-INT-13 (gitops CI):  M-P6.11 → M-P6.12 → M-P6.13
Wave H-INT-14 (speech):     M-P6.14 → M-P6.15
Wave H-INT-15 (enterprise): M-P6.16 → M-P6.17 → M-P6.18 → M-P6.19
Wave H-INT-16 (data/wf):    M-P6.20 → M-P6.21 → M-P6.22 → M-P6.23 → M-P6.24
Wave H-INT-17 (reserve):    M-P6.25 → M-P6.26 → M-P6.27 → M-P6.28 → M-P6.29 → M-P6.30 → M-P6.31 → M-P6.32
Wave PRE (presets):         M-P6-PRE.1  (after H-INT-10 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P5 **Done**; M-P5.FU wiring **Done**; Phase SEC closeout **Done** (V-SEC patterns for `security_scanner`).  
**Parallelism:** H-INT-10 unblocks STABLE promote gate; H-INT-11 unblocks cloud `sandbox.exec`; H-INT-12 unblocks multi-tenant hosts; H-INT-14 unifies speech catalog.  
**Closeout target:** catalog **167** slugs; optional `HARNESS_M6_P6_PROBE_SLUGS` + four Tier-3 presets; gate green.

---

### Phase P-Ext — Plugin Catalogs (Integrations, Tools, Skills)

**Status:** **Done** (2026-06-02) — MVP + production closure (Appendix I).  
**Prerequisites:** Phases **M** (Integration Library), **O** (Tool Library), **R** (Skill Library MVP) **Done**; open integration slug model (no closed `IntegrationSlug` enum in registry) **Done**.  
**Goal:** Make all three Tier-0 catalogs **plugin-native** and aligned with market patterns (hexagonal adapters, MCP-style tools, capability packs) — including **pip-installable** extensions without editing Intergrax core.  
**Tracker:** **Appendix I** (task-level status). **Author guide:** [`guides/EXTENSION_AUTHOR_GUIDE.md`](guides/EXTENSION_AUTHOR_GUIDE.md).

**Delivered (2026-06-02):** `load_plugins` + `bootstrap_catalogs()` · three plugin protocols · lazy presets/bundle ids · EP fixture package · `warn_override` conflict policy · scaffold CLI · integrations **manifest+factory** (**135** full) + `IntegrationPlugin` for externals · tools **13/13** `ToolPlugin` · skills **3/3** `SkillPlugin` · `resolve_typed` (6 categories) · health API · `CatalogSnapshot` · expanded `check_plugin_catalog.py` · canon §7.1.5.1 + author guide.

**Principle:** Integration → Tool → Skill → Agent (unchanged) · explicit first-party bootstrap + optional entry points · one P-Ext.* ID per PR · gate green.

**Production-path reality (do not confuse with MVP):**

| Layer | Shipped catalog | External extension | Runtime materialization |
|-------|-----------------|--------------------|-------------------------|
| **Integrations** | **135** slugs (`preset="full"`) / **12** core (`preset="core"`) via `register_from_manifest` + `create_*` — **0** shipped `register.py` use `register_integration_plugin` | `IntegrationPlugin` + EP `intergrax.integrations` | `IntegrationProfile.resolve(category, config=…)` → backend instance |
| **Tools** | **13** bundles / **~29** `tool_id` — **13/13** via `ToolPlugin` (`shipped_plugins.py`) | `ToolPlugin` + EP `intergrax.tools` | `bootstrap_catalogs` → `build_registry_from_profile(ToolProfile, ctx)` → `ToolRegistry` → `RuntimeToolInvoker` / MCP |
| **Skills** | **3** bundles / **8** `skill_id` — **3/3** via `SkillPlugin` (`harness`×6, `legal`×1, `research`×1) | `SkillPlugin` + EP `intergrax.skills` | `build_registry_from_profile(SkillProfile)` → `SkillRegistry` → `SkillResolver` → `allowed_tools` |

**Out of scope for Phase P-Ext:**

- Online plugin marketplace UI / central registry service
- Runtime hot-reload of catalogs without process restart
- Skill as executable workflow graph (LangGraph pack) — separate initiative
- Replacing `ToolWiringContext` with a generic DI framework
- Migrating all **135** shipped integrations to `IntegrationPlugin` classes (optional long-term; manifest path remains supported)

#### P-Ext.0 — Shared plugin foundation

**Goal:** One plugin loader and one Tier-3 bootstrap entry point.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.0.1 | **`load_plugins(group, …)`** — entry point discovery | **Done** | **Critical** | `intergrax/core/plugins/discovery.py` | Idempotent; `on_conflict=error\|skip` |
| P-Ext.0.2 | **Plugin errors** — `PluginConflictError`, `PluginLoadError` | **Done** | High | `intergrax/core/plugins/errors.py` | Unit tests |
| P-Ext.0.3 | **`bootstrap_catalogs()`** — unified Tier-3 composition | **Done** | **Critical** | `intergrax/core/catalog_bootstrap.py` | tool/skill wiring + idempotent shipped |
| P-Ext.0.4 | **`docs/guides/EXTENSION_AUTHOR_GUIDE.md`** | **Done** | High | `docs/` | pip package walkthrough |
| P-Ext.0.5 | **Fixture pip package** in tests | **Done** | High | `tests/fixtures/plugin_packages/` | editable install; registers integration + tool + skill |
| P-Ext.0.6 | **EP discovery tests** via fixture (all three groups) | **Done** | High | `tests/unit/core/plugins/` | `bootstrap_catalogs(discover_entry_points=True)` loads fixture |
| P-Ext.0.7 | **`INTERGRAX_DISCOVER_PLUGINS`** env + Tier-3 wiring | **Done** | Medium | `catalog_bootstrap.py`, `applications/_shared/platform_wiring.py` | lab opt-in; default `false` in prod hosts |

**DoD:** Fixture package registers via entry point; discovery unit tests green.

**Entry point groups (canonical names):**

```toml
[project.entry-points."intergrax.integrations"]
[project.entry-points."intergrax.tools"]
[project.entry-points."intergrax.skills"]
```

---

#### P-Ext.1 — Integrations: plugin closure

**Baseline:** `IntegrationManifest`, `IntegrationPlugin`, `register_from_manifest`, per-provider `manifest.py` (open slug catalog).

**Audit snapshot (2026-06-02 — integrations only; counts synced post M.6 P5 closeout):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | `bootstrap_core` **12** slugs + `bootstrap_extended` **~123** → **135** full; all `register.py` call `register_from_manifest(MANIFEST, create_*)` | **Yes** — primary harness path |
| **`IntegrationPlugin` shipped** | **0/135** providers register via `register_integration_plugin` in shipped code | N/A — external / explicit only |
| **Reference plugin class** | `SqliteIntegrationPlugin` in `sqlite/plugin.py`; `register.py` still uses manifest path | Doc pattern only (P-Ext.1.12) |
| **External example** | `integrations/examples/custom_memory_kv/` + `test_external_plugin.py` (explicit register) | **Yes** API; EP not tested |
| **`IntegrationProfile.resolve`** | Manifest, plugin class, slug `str`, or pre-built instance via `IntegrationBinding` | **Yes** — Tier-3 prod |
| **`resolve_typed.py`** | Six typed helpers incl. vector_store, notification_channel, object_storage | **Done** |
| **`IntegrationSlug` enum** | **0** references in `intergrax/**/*.py` and provider `USAGE.md`; legacy mention only in plan + migration scripts | **Done** (P-Ext.1.5) |
| **Tier-3 bootstrap** | `integration_wiring` / `tool_wiring` / `skill_wiring` → `bootstrap_catalogs()` + lazy bundle ids | **Done** |
| **Entry points** | Fixture pip package + EP tests; `INTERGRAX_DISCOVER_PLUGINS` for lab | **Done** |
| **`on_conflict`** | `bootstrap_catalogs(on_conflict=…)` — `error`, `skip`, `override`, `warn_override` for catalog slugs + EP names | **Done** (P-Ext.4.3) |
| **Health API** | `integrations/registry/health.py` — `ping_integration` / `integration_registered` | **Done** |
| **Unit tests** | Per-provider tests + `test_profile` + `test_external_plugin` + lazy `preset="core"` in `test_lazy_catalog_bootstrap` | **Strong**; no full-count assertion in CI |

**Verdict:** Shipped integrations are **production-ready** on the **manifest + factory** path. `IntegrationPlugin` is **production-ready for third-party** extensions; parity with tools (all shipped as plugin classes) is **explicitly out of scope**.

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.1.1 | Wire **`intergrax.integrations`** entry points in `bootstrap_catalogs()` | **Done** | **Critical** | `catalog_bootstrap.py` | `discover_entry_points=True` |
| P-Ext.1.2 | Split **`register_default_integrations()`** → core + optional | **Done** | High | `integrations/registry/bootstrap_core.py` | `preset="core"` (12) \| `"full"` (135) |
| P-Ext.1.3 | **Typed resolve** helpers (top categories) | **Done** | Medium | `integrations/registry/resolve_typed.py` | 3 categories today |
| P-Ext.1.3a | Expand **`resolve_typed`** + unit tests | **Done** | Medium | `resolve_typed.py`, `tests/unit/integrations/test_resolve_typed.py` | +`vector_store`, `notification_channel`, `object_storage`; used in lab docs |
| P-Ext.1.4 | **Health check** API per slug (optional) | **Done** | Low | `integrations/registry/health.py` | `ping(slug) -> bool` smoke helper |
| P-Ext.1.5 | Remove **`IntegrationSlug`** from docs/scripts | **Done** | Medium | `**/USAGE.md`, `README.md`, `scripts/`, `docs/guides/AGENT_CREATION_GUIDE.md` | `intergrax/**/*.py` already clean |
| P-Ext.1.6 | **EP integration test** via fixture | **Done** | High | `tests/unit/integrations/` | `discover_entry_points=True` loads fixture slug |
| P-Ext.1.7 | **Dual-model docs** — manifest+factory vs `IntegrationPlugin` | **Done** | Medium | `architecture/INTEGRATIONS.md`, `guides/EXTENSION_AUTHOR_GUIDE.md` | decision table + when to migrate |
| P-Ext.1.8 | **CI smoke** — integration slug counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | `core` ≥12, `full` ≥95 (or exact snapshot) |
| P-Ext.1.9 | **`test_resolve_typed.py`** | **Done** | Low | `tests/unit/integrations/` | type errors on wrong contract |
| P-Ext.1.10 | **Tier-3** lab/poc use `bootstrap_catalogs(integration_preset=…)` | **Done** | High | `applications/*/host/integration_wiring.py` | replace bare `register_default_integrations()` |
| P-Ext.1.11 | **`applications/_shared/integration_wiring.py`** helper | **Done** | Medium | `applications/_shared/` | mirror `tool_wiring` — bootstrap + profile factory |
| P-Ext.1.12 | **`SqliteIntegrationPlugin`** — document or wire one shipped slug | **Done** | Low | `sqlite/register.py` or `architecture/INTEGRATIONS.md` | either `register_integration_plugin` in sqlite **or** “reference only” in docs |

**DoD:** 364+ integration unit tests green; external integration via entry point **and** via pip entry point (fixture); Tier-3 hosts use unified `bootstrap_catalogs()` for integrations.

---

#### P-Ext.2 — Tools: ToolPlugin + MCP export

**Baseline:** `ToolContract`, `ToolBundleEntry`, `ToolProfile`, `ToolWiringContext`, `RuntimeToolInvoker`.

**Audit snapshot (2026-06-02 — tools only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **13/13** bundles on `ToolPlugin` via `shipped_plugins.py` + `define_tool_plugin` | **Yes** — full plugin parity |
| **Tool count** | **~29** `tool_id` across bundles (RAG, websearch, jira, sandbox, vision, speech, …) | **Yes** |
| **Legacy register path** | No shipped bundle bypasses `register_tool_plugin`; `register_from_tool_manifest` is internal only | **Yes** |
| **External example** | `intergrax/tools/examples/` + `test_external_tool_plugin.py` | **Yes** |
| **EP `intergrax.tools`** | Fixture package + EP discovery tests (P-Ext.0.5 / 2.11) | **Yes** |
| **Tier-3 wiring** | `tool_wiring.build_application_tool_wiring` → `bootstrap_catalogs(register_shipped=True)` | **Yes** |
| **Lazy catalog** | `tool_wiring` passes `tool_bundle_ids` from `ToolProfile` | **Done** |
| **Runtime materialization** | Two-phase: catalog → `ToolWiringContext` + integrations → `ToolRegistry` handlers | **Yes** |
| **MCP / standalone LLM** | `export_mcp_tools`, `ToolsAgent`, `RuntimeToolInvoker` trace | **Yes** — strongest market path |
| **Unit tests** | Per-bundle tests + `test_external_tool_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped tools are **production-ready** on **`ToolPlugin`**; P-Ext.2 closure complete (external example, EP test, lazy `tool_wiring`).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.2.1 | **`ToolPlugin` Protocol** | **Done** | **Critical** | `intergrax/tools/core/plugin.py` | `tool_bundle_manifest()`, `register_tools(registry, ctx)` |
| P-Ext.2.2 | **`ToolManifest`** (bundle metadata) | **Done** | **Critical** | `intergrax/tools/core/manifest.py` | bundle_id, tool_ids, status |
| P-Ext.2.3 | **`register_tool_plugin()`** | **Done** | **Critical** | `intergrax/tools/registry/plugin_register.py` | Mirror integrations |
| P-Ext.2.4 | **Pilot migration** — RAG bundle → `ToolPlugin` | **Done** | High | `tools/providers/rag/` | Pattern for other bundles |
| P-Ext.2.5 | Entry point group **`intergrax.tools`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.2.6 | **`export_mcp_tools(registry)`** | **Done** | High | `intergrax/tools/exporters/mcp.py` | alias of `to_mcp_tools` |
| P-Ext.2.7 | **`ToolContract.version`** field (semver) | **Done** | Medium | `tools/core/contracts.py` | Default `1.0.0` |
| P-Ext.2.8 | **Migrate all shipped tool bundles** → `ToolPlugin` | **Done** | High | `tools/registry/shipped_plugins.py`, `providers/*/register.py` | 13/13 bundles |
| P-Ext.2.9 | **Reference external tool** — `tools/examples/` | **Done** | High | `intergrax/tools/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.2.10 | **`test_external_tool_plugin.py`** | **Done** | High | `tests/unit/tools/` | catalog → `build_registry_from_profile` → `RuntimeToolInvoker.invoke` |
| P-Ext.2.11 | **EP tool test** via fixture | **Done** | High | `tests/unit/tools/` | depends on P-Ext.0.5 |
| P-Ext.2.12 | **`tool_wiring` lazy bootstrap** — pass `tool_bundle_ids` from profile | **Done** | Medium | `applications/_shared/tool_wiring.py` | `bootstrap_catalogs(..., tool_bundle_ids=profile.enabled_bundles)` |

**DoD:** External tool executes via `RuntimeToolInvoker` after entry-point registration (test proves it); Tier-3 `tool_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.3 — Skills: SkillPlugin

**Baseline:** `SkillManifest`, `SkillBundleEntry`, `SkillResolver`, `AgentRegistry` merge to `allowed_tools`.

**Audit snapshot (2026-06-02 — skills only):**

| Area | Finding | Prod? |
|------|---------|-------|
| **Shipped catalog** | **3/3** bundles on `SkillPlugin` via `shipped_plugins.py` + `register_default_skills()` | **Yes** — best plugin parity of Tier-0 |
| **Skill count** | **8** `skill_id`: `harness` (6), `legal` (1), `research` (1) | **Yes** |
| **Legacy `register_skill_bundle`** | Only in `plugin_register.py` + **outdated** `scaffold new-skill` output | Scaffold **not** prod (P-Ext.3.10) |
| **`register_from_skill_manifest`** | Internal helper; all shipped paths use `register_skill_plugin` | **Yes** |
| **External example** | `intergrax/skills/examples/` + external plugin tests | **Yes** |
| **EP `intergrax.skills`** | Fixture package + EP discovery tests (P-Ext.0.5 / 3.8) | **Yes** |
| **Tier-3 wiring** | `skill_wiring.build_application_skill_wiring` → `bootstrap_catalogs(register_shipped=True)` — **better than integrations** | **Yes** |
| **Lazy catalog** | `skill_wiring` passes `skill_bundle_ids` from `SkillProfile` | **Done** |
| **Runtime materialization** | Two-phase like tools: catalog bundle rows → `build_registry_from_profile` → `SkillRegistry` | **Yes** |
| **`requires_skills`** | Resolver + `test_requires_skills.py`; **0** shipped manifests use it | Feature **Done**; adoption open (P-Ext.3.12) |
| **Cursor `SKILL.md` importer** | `CursorSkillImporter` — parallel path, not `SkillPlugin` | **Yes** for import; document vs plugin (P-Ext.3.11) |
| **Agent merge** | `AgentRegistry.register(..., skill_registry=, tool_registry=)` + `test_agent_registry_skills.py` | **Yes** |
| **Unit tests** | Harness + resolver + `test_external_skill_plugin` + EP fixture | **Yes** |

**Verdict:** Shipped skills are **production-ready** on **`SkillPlugin`**; P-Ext.3 closure complete (external example, EP test, lazy `skill_wiring`, scaffold alignment).

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.3.1 | **`SkillPlugin` Protocol** | **Done** | **Critical** | `intergrax/skills/core/plugin.py` | `skill_bundle_manifest()`, `skill_manifests()`, `register_skills(registry)` |
| P-Ext.3.2 | **`register_skill_plugin()`** | **Done** | **Critical** | `intergrax/skills/registry/plugin_register.py` | Wraps `register_from_skill_manifest` |
| P-Ext.3.3 | Entry point group **`intergrax.skills`** | **Done** | High | `catalog_bootstrap.py` | opt-in `discover_entry_points` |
| P-Ext.3.4 | Migrate **`harness`** + **`research`** + **`legal`** → `SkillPlugin` | **Done** | High | `skills/providers/*/plugin.py`, `shipped_plugins.py` | **3/3** bundles |
| P-Ext.3.5 | **`requires_skills`** on `SkillManifest` + resolver DFS | **Done** | Low | `skills/resolver.py`, `test_requires_skills.py` | Cycle + unknown dep errors |
| P-Ext.3.6 | **Reference external skill** — `skills/examples/` | **Done** | High | `intergrax/skills/examples/` | mirror `integrations/examples/custom_memory_kv` |
| P-Ext.3.7 | **`test_external_skill_plugin.py`** | **Done** | High | `tests/unit/skills/` | explicit `register_skill_plugin` → `SkillResolver` → tool merge |
| P-Ext.3.8 | **EP skill test** via fixture | **Done** | High | `tests/unit/skills/` | depends on P-Ext.0.5 |
| P-Ext.3.9 | **`skill_wiring` lazy bootstrap** — pass `skill_bundle_ids` from profile | **Done** | Medium | `applications/_shared/skill_wiring.py` | `bootstrap_catalogs(..., skill_bundle_ids=profile.enabled_bundles)` |
| P-Ext.3.10 | **Scaffold `new-skill`** emits `SkillPlugin` + `plugin.py` | **Done** | Medium | `intergrax/scaffold/new_skill.py` | remove legacy `register_skill_bundle` template |
| P-Ext.3.11 | **Docs: SkillPlugin vs Cursor importer** | **Done** | Medium | `architecture/SKILLS.md`, `guides/EXTENSION_AUTHOR_GUIDE.md` | when to use pip plugin vs `SKILL.md` import |
| P-Ext.3.12 | **`requires_skills` in shipped harness** (optional demo) | **Done** | Low | `skills/providers/harness/manifests.py` | one derived skill depending on `harness.tool_smoke` |

**DoD:** External skill merges `allowed_tools` on `AgentRegistry.register` (test proves it); Tier-3 `skill_wiring` supports lazy bundle bootstrap.

---

#### P-Ext.4 — Operational scale

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.4.1 | **Lazy bootstrap** — register only bundles in active `*Profile` | **Done** | High | `catalog_bootstrap.py`, bootstrap modules | `tool_bundle_ids`, `skill_bundle_ids`, `integration_preset` |
| P-Ext.4.2 | **`CatalogSnapshot` API** (read-only) | **Done** | Medium | `intergrax/core/catalog_snapshot.py` | list slugs for docs/UI |
| P-Ext.4.3 | Slug conflict policy in bootstrap | **Done** | Medium | `catalog_bootstrap.py` | `error` / `warn_override` |
| P-Ext.4.4 | CI **`check_plugin_catalog.py`** | **Done** | High | `scripts/` | smoke: shipped bundles present |
| P-Ext.4.5 | **Expand CI smoke** — all three catalog counts | **Done** | Medium | `scripts/check_plugin_catalog.py` | tools **13** bundles / ~**29** tool_id; skills **3** bundles / **8** skill_id; integrations **core≥12**, **full≥95** (see also P-Ext.1.8) |

---

#### P-Ext.5 — Docs, scaffold, canon

| # | Deliverable | Status | Priority | Location | Acceptance |
|---|-------------|--------|----------|----------|------------|
| P-Ext.5.1 | Scaffold **`new_integration` / `new_tool_bundle` / `new_skill_bundle`** | **Done** | Medium | `intergrax/scaffold/` | manifest + plugin + register |
| P-Ext.5.2 | **External plugins** sections in INTEGRATIONS/TOOLS/SKILLS | **Done** | Medium | `docs/` | Cross-link Appendix I |
| P-Ext.5.3 | **Canon §7.1.5.1** — entry points + plugin protocols | **Done** | High | `intergrax_runtime_architecture.md` | §7.1.5.1 Tier-0 Plugin Catalogs |
| P-Ext.5.4 | Remove duplicate `PLUGIN_CATALOG_PLAN.md` | **Done** | Low | — | tracking only in this plan + Appendix I |
| P-Ext.5.5 | **Prod path matrix** in author guide (integration vs tool vs skill) | **Done** | Medium | `guides/EXTENSION_AUTHOR_GUIDE.md` | two-phase tool bootstrap documented |
| P-Ext.5.6 | **Lab wiring recipe** for external plugins | **Done** | Medium | `applications/lab_application/`, `TIER3_READINESS.md` | `discover_entry_points` + profile example |

---

#### P-Ext.6 — Production closure (paydown)

**Goal:** Close gaps between **MVP** (API + shipped catalogs) and **production-ready extensibility** (tested pip install, parity across three layers, ops hooks).

| # | Deliverable | Status | Priority | Depends on | Acceptance |
|---|-------------|--------|----------|------------|------------|
| P-Ext.6.1 | **Fixture pip package** (unblocks EP tests) | **Done** | **Critical** | — | same as P-Ext.0.5 |
| P-Ext.6.2 | **External tool + skill examples + tests** | **Done** | **Critical** | 6.1 | P-Ext.2.9–2.11, P-Ext.3.6–3.8, 3.7 green |
| P-Ext.6.8 | **Skill Tier-3 + scaffold** (rollup) | **Done** | Medium | — | P-Ext.3.9–3.12, scaffold overlap P-Ext.5.1 |
| P-Ext.6.9 | **Tool Tier-3 lazy wiring** (rollup) | **Done** | Medium | — | P-Ext.2.12 (symmetric with P-Ext.3.9) |
| P-Ext.6.10 | **Tier-3 lazy wiring** (all catalogs rollup) | **Done** | Medium | — | P-Ext.2.12 + P-Ext.3.9 + optional `integration_preset` in shared helpers |
| P-Ext.6.3 | **EP discovery** in tests + lab env flag | **Done** | High | 6.1 | P-Ext.0.6–0.7, P-Ext.1.6 |
| P-Ext.6.4 | **IntegrationSlug cleanup** in docs/scripts | **Done** | Medium | — | P-Ext.1.5 |
| P-Ext.6.5 | **Scaffold** `new_tool_bundle` / `new_skill_bundle` / `new_integration` | **Done** | Medium | — | P-Ext.5.1 |
| P-Ext.6.6 | **Integration Tier-3** + typed resolve + health (rollup) | **Done** | Medium | — | P-Ext.1.3a, 1.4, 1.8–1.11 |
| P-Ext.6.7 | **Conflict policy** + expanded CI smoke | **Done** | Medium | — | P-Ext.4.3, P-Ext.4.5, P-Ext.1.8 |

**DoD (phase closure):** Appendix I has no **Planned** P0/P1 rows; external integration, tool, and skill each proven via **entry point** (fixture package), not only explicit in-process registration.

---

#### Phase P-Ext — Definition of done

**MVP (met 2026-06-02):**

1. `bootstrap_catalogs()` + three plugin protocols + lazy presets.
2. All shipped tool/skill bundles on `ToolPlugin` / `SkillPlugin`.
3. Integration example `custom_memory_kv` + `test_external_plugin.py`.
4. Canon §7.1.5.1 + `guides/EXTENSION_AUTHOR_GUIDE.md` (EN).
5. Gate: `tests/unit/core/plugins`, integrations/tools/skills plugin tests green.

**Production closure (P-Ext.6 — open):**

1. **Fixture pip package** registers integration + tool + skill without Intergrax core edits.
2. **EP discovery tests** for all three groups (`discover_entry_points=True`).
3. **External tool test** — `RuntimeToolInvoker` after EP registration.
4. **External skill test** — `allowed_tools` merge after EP registration.
5. **Tier-3** documents/env for optional discovery; default remains explicit bootstrap.
6. **Tier-3 lazy wiring** — `tool_wiring` and `skill_wiring` pass profile bundle ids to `bootstrap_catalogs()` (P-Ext.2.12, P-Ext.3.9).
7. **No central slug enum** in new code/docs (string slugs); `IntegrationSlug` removed from author-facing examples.
8. **MCP export** from active `ToolRegistry` (already met).
9. Appendix I: all P-Ext.* rows **Done** or **Won't fix** with reason.

#### Phase P-Ext — Recommended execution order

```text
MVP (Done):               P-Ext.0.1–0.4 | P-Ext.1.1–1.2 | P-Ext.2.1–2.8 | P-Ext.3.* | P-Ext.4.1–4.2,4.4 | P-Ext.5.2–5.4

Paydown Wave P1 (critical):
  P-Ext.0.5 → P-Ext.0.6 → P-Ext.1.6 → P-Ext.1.10
           → P-Ext.2.9 → P-Ext.2.10 → P-Ext.2.11
           → P-Ext.3.6 → P-Ext.3.7 → P-Ext.3.8

Paydown Wave P2 (ops + docs):
  P-Ext.0.7 → P-Ext.4.3 → P-Ext.4.5 → P-Ext.1.8 → P-Ext.1.5 → P-Ext.1.7 → P-Ext.5.5 → P-Ext.5.6
           → P-Ext.2.12 → P-Ext.3.9 → P-Ext.3.10 → P-Ext.3.11

Paydown Wave P3 (optional polish):
  P-Ext.1.3a → P-Ext.1.4 → P-Ext.5.1 → P-Ext.3.12
```

**Effort estimate:** MVP ~21–32 person-days (**spent**); paydown **~12–18** person-days incl. integration + tool + skill closure (Appendix I).

**Priority ladder:** **Band 2c** (§4.0) — harness Tier-0 extensibility; **not** Band 3 product work.

---

## 4. Priority Order

### 4.0 Implementation priority ladder (canonical)

**Read this before §6.** The plan has three bands. Implement **top to bottom**. **Never** pull items from band 3 into “next step” summaries while band 1–2 are the active policy.

| Band | What | Status (2026-06-05) | Examples |
|------|------|---------------------|----------|
| **1 — Harness platform** | Tier-0/1/3 lab wiring, security, policy, typing, legacy removal, gate audits | **Maintenance** (§4.1 **Done**; keep green) | `pytest -m gate`, `check_harness_*`, `check_legacy_modules_removed.py`, regression fixes |
| **2 — Harness architecture hardening** | Capability graph, lifecycle governance, prompt/eval/context/security/cost/metrics hardening — **no** business domain | **Done** (2026-06-05) | V-CG … V-KG, V-V6 closeout · V-REM |
| **2i — Phase V runtime remediation (V-REM)** | Close 9 Partial Phase V + EvalRunner gate gaps — runtime enforcement, not new OS features | **Done** (2026-06-05) | [Phase V-REM](#phase-v-rem--phase-v-runtime-remediation-audit-closeout) · Appendix J |
| **2b — Modality plane (optional parallel)** | Vision CV, speech, classical ML — harness Tier-0 only | **Done** | W-ML complete; optional Celery bus wiring for Tier-3 scale-out |
| **2c — Plugin catalogs (P-Ext)** | Entry points + `ToolPlugin` + `SkillPlugin` + `bootstrap_catalogs()` | **Done** (2026-06-02) | Appendix I · [guides/EXTENSION_AUTHOR_GUIDE.md](guides/EXTENSION_AUTHOR_GUIDE.md) |
| **2d — Operational L3 (W-OPS)** | Reliability, identity, SLO/ops evidence, online eval — **no** business agents | **Done** (2026-06-06) | [Phase W-OPS](#phase-w-ops--operational-harness-maturity-ideal-l3-ops) · `phase_w_ops_evidence.py` |
| **2e — Application environment (H-APP)** | `ApplicationEnvironmentProfile`, unified Tier-3 wiring, host migration — **no** business agents | **Done** (2026-06-03) | [Phase H-APP](#phase-h-app--tier-3-application-environment-full-configurability) · [`HARNESS_APPLICATION_LAYER_AUDIT.md`](HARNESS_APPLICATION_LAYER_AUDIT.md) · **§6.2x** |
| **2f — Developer authoring UX (DX)** | LangGraph-like facades, minimal scaffold, CLI run/doctor, TTFRun gates, UI spec export — **no** business agents | **Done** (2026-06-03) | [Phase DX](#phase-dx--developer-authoring-experience-fast-environment--agent-builds) · **§6.2y** |
| **2g — Agents & applications conformance (AA)** | Scaffold alignment, per-agent/app `ARCHITECTURE.md`, deploy triad, legal **scaffold** reset (domain steps → Band 3) | **Mostly Done** (2026-06-02) | [Phase AA](#phase-aa--agents--applications-conformance-scaffold-docs-deploy) · **§6.2z** · [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business) |
| **2h — Memory platform (MEM)** | H-APP→runtime bridge, durable user LTM, session SQLite, gates, hooks, memory docs — **no** business agents | **Done** (2026-06-02) | [Phase MEM](#phase-mem--memory-platform-completion) · **§6.2aa** |
| **2j — Orchestration closeout (ORCH)** | Wire `planner_kind`/`classifier_kind`, `ApplicationGraphSpec`→plan, graph concurrency cap — **no** business agents | **Done** (2026-06-05) | [Phase ORCH](#phase-orch--orchestration-control-plane-closeout) · **§6.1b** · **§6.2bb** |
| **2k — Tools/skills closeout (TS)** | Catalog→`RuntimeConfig` bridge, harness LLM wiring, `SkillResolverProtocol`, Appendix J — **no** business agents | **Done** (2026-06-02) | [Phase TS](#phase-ts--tools--skills-control-plane-closeout) · **§6.1c** · **§6.2bc** |
| **2l — Integration closeout (INT)** | `integration_runtime_bridge`, bootstrap health probes, Appendix K — **no** business agents | **Done** (2026-06-02) | [Phase INT](#phase-int--integration-control-plane-closeout) · **§6.1d** · **§6.2bd** |
| **2m — RAG closeout (RAG)** | `rag_runtime_bridge`, RAG stack on environment wire — **no** business agents | **Done** (2026-06-02) | [Phase RAG](plan/RAG.md) · **§6.1e** · **§6.2be** |
| **2n — Context engineering closeout (CTX)** | `context_runtime_bridge`, `context_wiring`, Nexus `ContextManager` wire — **no** business agents | **Done** (2026-06-02) | [Phase CTX](#phase-ctx--context-engineering-control-plane-closeout) · **§6.1f** · **§6.2bf** |
| **2o — Legacy tool plan closeout (LEG)** | `tool_ids` canonical path; gateway/engine planner migration — **no** business agents | **Done** (2026-06-02) | [Phase LEG](#phase-leg--legacy-tool-plan-boolean-closeout) · **§6.1h** |
| **2p — Prompt registry closeout (PE)** | `PromptProfile`, `prompt_runtime_bridge`, `prompt_wiring`, Appendix M — **no** business agents | **Done** (2026-06-02) | [Phase PE](#phase-pe--prompt-registry-control-plane-closeout) · **§6.1i** |
| **2q — Agent assembly closeout (AS)** | Agent contract conformance, capability/skill resolution, lifecycle state — **no** business agents | **Done** (2026-06-02) | [Phase AS](#phase-as--agent-assembly-control-plane-closeout) · **§6.1k** · **Appendix N** |
| **2r — Registry architecture closeout (REG)** | Registry snapshot, assembly resolver, host resolution CI — **no** business agents | **Done** (2026-06-02) | [Phase REG](#phase-reg--registry-architecture-control-plane-closeout) · **§6.1l** · **Appendix O** |
| **2s — Capability graph closeout (CG)** | Environment graph slice, wire-time validation, CI audit — **no** business agents | **Done** (2026-06-02) | [Phase CG](#phase-cg--capability-graph-control-plane-closeout) · **§6.1m** · **Appendix P** |
| **2t — Observability closeout (OBS)** | Profile bridge, assembly resolver, host wiring CI — **no** business agents | **Done** (2026-06-02) | [Phase OBS](#phase-obs--observability-control-plane-closeout) · **§6.1n** · **Appendix Q** |
| **2u — Reliability closeout (REL)** | Idempotency bridge, circuit breaker wire, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase REL](#phase-rel--reliability-control-plane-closeout) · **§6.1o** · **Appendix R** |
| **2v — Security closeout (SEC)** | V-SEC bridge, middleware assembly resolver, host CI — **no** business agents | **Done** (2026-06-02) | [Phase SEC](#phase-sec--security-control-plane-closeout) · **§6.1q** · **Appendix S** |
| **2w — Cost governance closeout (COST)** | Budget bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase COST](#phase-cost--cost-governance-control-plane-closeout) · **§6.1r** · **Appendix T** |
| **2x — Evaluation closeout (EVAL)** | Registry bridge, policy bundle merge, assembly resolver CI — **no** business agents | **Done** (2026-06-02) | [Phase EVAL](#phase-eval--evaluation-control-plane-closeout) · **§6.1s** · **Appendix U** |
| **2y — Adaptive Harness Intelligence (W-ADAPT)** | L4 **runtime** closed loop — SignalCollector, AdaptationEngine, ProfileVersionStore, verify/rollback — **no** business agents | **Done** (2026-06-02) — **70/70 Done** | [Phase W-ADAPT](#phase-w-adapt--adaptive-harness-intelligence-l4-runtime) · [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · **§6.1t** · **§6.2ac** · **Appendix K** |
| **2z — LLM completion envelope (M-LLM-R)** | Typed `LLMAdapterResponse` replaces `str`/`dict` adapter returns; full consumer refactor — **no** business agents | **Done** (2026-06-06) — **39/39** | [Phase M-LLM-R](#phase-m-llm-r--llm-completion-response-envelope-audit-2026-06-06) · **§6.1v** · **§6.2ad** · **Appendix L** |
| **2aa — Integration expansion (M.6 P4)** | 28 harness-ROI provider slugs (secrets, observability stack, OLAP, feature flags, prod deploy) — **no** business agents | **Done** (2026-06-02) — **28/28** | [M.6 P4 register](#m6-p4--harness-platform-expansion-done) · **§6.1w** · **§6.2ae** |
| **2ab — Integration depth (M.6 P5)** | Harden 25 beta + 8 greenfield harness slugs (metrics, CI/CD, eval, async, data plane) — **no** business agents | **Done** (2026-06-02) — **33/34** | [M.6 P5 register](#m6-p5--harness-integration-depth-done--3334) · **§6.1x** · **§6.2af** |
| **2ac — Integration expansion (M.6 P6)** | 32 harness slugs + post-catalog wiring (tools, bridges, promote gate, infra `p6`) — **no** business agents | **Done** (2026-06-02) — **32/32 + M-P6-WIRE** | [M.6 P6 register](#m6-p6--harness-integration-expansion-planned) · **§6.1y** · **§6.2ag** |
| **2ad — FAUDIT-32 remediation** | Close 32-layer audit residuals (tier gate, intake, observability taxonomy, registry depth, eval release gate) — **no** business agents | **Done** (2026-06-06) — **23/23 + §6.1ai follow-up** | [Phase FAUDIT-32](#phase-faudit-32--full-architecture-audit-closeout) · **§6.1ah** · **§6.1ai** · **Appendix M** |
| **2aj — Nexus execution depth (FLOW)** | Close `FLOW-GAP.*` (01–16) — delegation, SubtaskContract, backpressure profile, LLM planner, merge, eval, graph hardening — **no** K.1/K.2 | **Done** (2026-06-09) — **18/18 harness** (FLOW-8 product **Deferred** §6.3) | [Phase FLOW](#phase-flow--nexus-execution-depth) · **§6.1aj** · **§6.2aj** · **Appendix N (FLOW)** |
| **2ak — Critic & Verification Layer (CRIT-V)** | PEV verify depth — `CriticOrchestrator`, `eval.judge`, `eval.trajectory`, evaluator-loop, semantic offline runner — **no** business agents | **Done** | [Phase CRIT-V](#phase-crit-v--critic--verification-layer) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md) · **§6.1ak** · **§6.2ak** · canon §55 · [ADR-CRITIC-001](adr/entries/2026-06-07/ADR-CRITIC-001.md) |
| **2al — Unified Observability Spine (OBS-BUS)** | Full HOS — typed payloads, `ObservabilityEmitter`, emission coverage, extension SDK, L4 §21 — **no** business agents | **Done** | [Phase OBS-BUS](#phase-obs-bus--unified-observability-spine) · [`architecture/OBSERVABILITY.md`](architecture/OBSERVABILITY.md) · **§6.1al** · [ADR-OBS-001](adr/entries/2026-06-08/ADR-OBS-001.md) |
| **2am — Memory intelligence depth (MEM-DEPTH)** | Context Compiler, never-overflow invariant, lifecycle automation, explore delegation, entity memory — **no** business agents | **Planned** (0/26) | [Phase MEM-DEPTH](#phase-mem-depth--memory-intelligence-depth) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · **§6.2ab** |
| **3 — END OF PLAN (product)** | Business agents, new product Tier-3 apps, domain skills, Legal live E2E | **Deferred** — **[§6.3](#63-end-of-plan--deferred-product-work-only)** | K.1, K.2, `applications/<product>/`, K.6, B.15, S-Ops.4 · FLOW-8 |

**Hard rule:** Band 3 is **not** “next after harness.” It runs only after an **explicit product prioritization decision** (Appendix A for agents; separate decision for new applications). Until then, **do not** implement, extend, or schedule K.1/K.2 waves, new product hosts, or product-only E2E in implementation cadence (§6.1–§6.2).

**Policy (2026-06-07):** Harness completion in §4.1 is **Done**. Band 1 = keep gate green on every PR. Bands **2j–2ad** platform closeouts = **Done**. **Band 2aj (Phase FLOW)** = **Done** (18/18 harness; FLOW-8 product **Deferred** §6.3). **Band 2ak (Phase CRIT-V)** = **Done** (24/24). Band 3 = **frozen** unless leadership reprioritizes.

```text
BAND 1:  Harness maintenance — gate + audit scripts (§6.1) — every PR
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (2026-06-06)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (2026-06-05)
BAND 2:  Harness architecture hardening — Phase V + V-REM — DONE (2026-06-05)
BAND 2i: Phase V runtime remediation — V-REM — DONE (2026-06-05)
BAND 2d: Operational L3 — Phase W-OPS (§6.2w) — DONE
BAND 2e: Application environment — Phase H-APP (§6.2x) — DONE (43 tasks)
BAND 2f: Developer authoring UX — Phase DX (§6.2y) — DONE (47 tasks)
BAND 2g: Agents & applications conformance — Phase AA (§6.2z) — MOSTLY DONE (platform); domain → Band 3
BAND 2h: Memory platform — Phase MEM (§6.2aa) — DONE (48/48)
BAND 2j: Orchestration closeout — Phase ORCH (§6.1b) — DONE (ORCH-1 → ORCH-4)
BAND 2k: Tools/skills closeout — Phase TS (§6.1c) — DONE (TS-1 → TS-3)
BAND 2l: Integration closeout — Phase INT (§6.1d) — DONE (INT-1 → INT-2)
BAND 2m: RAG closeout — Phase RAG (§6.1e) — DONE (RAG-1)
BAND 2n: Context engineering closeout — Phase CTX (§6.1f) — DONE (CTX-1 → CTX-2)
BAND 2o: Legacy tool plan closeout — Phase LEG (§6.1h) — DONE (LEG-1 → LEG-3)
BAND 2p: Prompt registry closeout — Phase PE (§6.1i) — DONE (PE-1 → PE-3)
BAND 2q: Agent assembly closeout — Phase AS (§6.1k) — DONE (AS-1 → AS-3)
BAND 2r: Registry architecture closeout — Phase REG (§6.1l) — DONE (REG-1 → REG-3)
BAND 2s: Capability graph closeout — Phase CG (§6.1m) — DONE (CG-1 → CG-3)
BAND 2t: Observability closeout — Phase OBS (§6.1n) — DONE (OBS-1 → OBS-3)
BAND 2u: Reliability closeout — Phase REL (§6.1o) — DONE (REL-1 → REL-3)
BAND 2v: Security closeout — Phase SEC (§6.1q) — DONE (SEC-1 → SEC-3)
BAND 2w: Cost governance closeout — Phase COST (§6.1r) — DONE (COST-1 → COST-3)
BAND 2x: Evaluation closeout — Phase EVAL (§6.1s) — DONE (EVAL-1 → EVAL-3)
BAND 2y: Adaptive Harness Intelligence — Phase W-ADAPT (§6.1t) — DONE (70/70, Wave 0–7 Done)
BAND 2z: LLM completion envelope — Phase M-LLM-R (§6.1v) — DONE (39/39)
BAND 2aa: Integration expansion — Phase M.6 P4 (§6.1w) — DONE (28/28)
BAND 2ab: Integration depth — Phase M.6 P5 (§6.1x) — DONE (33/34)
BAND 2ac: Integration expansion — Phase M.6 P6 (§6.1y) — DONE (32/32 + M-P6-WIRE)
BAND 2ad: FAUDIT-32 remediation — DONE (2026-06-06)
BAND 2aj: Nexus execution depth — Phase FLOW (§6.1aj) — DONE (18/18 harness; FLOW-8 product Deferred §6.3)
BAND 2ak: Critic & Verification Layer — Phase CRIT-V (§6.1ak) — **Done** (incl. CRIT-V-FOLLOWUP)
BAND 2al: Unified Observability Spine — Phase OBS-BUS (§6.1al) — **Done**
DONE:    Phase CLEAN — legacy module closeout (§6.1j) — 2026-06-02
BAND 3:  END OF PLAN — product agents & applications (§6.3) — DO NOT SCHEDULE AS DEFAULT NEXT

DONE:    Harness completion backlog (§4.1) — 2026-06-02
DONE:    Phase U — Harness production hardening (2026-06-01)
DONE:    Phase T — Harness cleanliness (2026-06-01)
DONE:    Phase S — Harness environment GA (2026-06-01)
DONE:    Phase Q+ — Harness Hardening (Appendix D)
DONE:    Phase R (MVP) — Harness AI alignment (Appendix E)
DONE:    Phase Q — Harness Quality (audit #1) — Waves 1–9
DONE:    Phase L, M, M-LLM, M-RAG, N, O — harness GA (functional)
DONE:    Phase K hardening K.3–K.5; Appendix B paydown (except B.15)

PARALLEL (harness-only): M.6 P6 integration expansion (§6.1y, **32 Done — closed 2026-06-02**); M.6 P5 residual `trivy` absorbed into P6 M-P6.1; legacy M.6 on-demand slugs; R-Skill catalog expansion (platform packs)

BAND 3 — END OF PLAN (see §6.3; not default “next”):
  • K.1 Problem Radar / K.2 Vendor Discovery (business agents)
  • K.6 / B.15 / S-Ops.4 — Legal live LLM E2E (product/CI)
  • New Tier-3 **product** applications (beyond lab + existing reference hosts)
  • Domain skill packs for product agents (until K.* started)
  • Problem Radar wave 2+ (`agents/problem_radar/` frozen)

RULE:    Strategy → canon → plan → code; Tier-1 via §0.6; four layers Integration → Tool → Skill → Agent
```

**Rationale:** Phases S/T/U + §4.1 delivered a production-configurable **harness**. Band 1–2 preserve and extend that platform. **Band 3 (product) is intentionally last** so business agents and new applications do not drive Tier-1 evolution (canon §52, [INTERGRAX_DEVELOPMENT_STRATEGY.md](guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)).

### 4.0a Implementation scope split (infrastructure vs business)

**Canonical rule:** Default implementation queue = **infrastructure only** (Bands 1–2g + §6.1). **Business** work runs only after explicit product prioritization — **[§6.3](#63-end-of-plan--deferred-product-work-only)**.

**Documentation rule:** This plan and [`intergrax_runtime_architecture.md`](intergrax_runtime_architecture.md) document **platform** delivery (Harness / Agent OS). They do **not** subsume `applications/<product>/IMPLEMENTATION_PLAN.md` or `agents/<name>/` product roadmaps — each business environment and business agent owns its architecture and deployment narrative.

| Layer | Bands / phases | What it includes | Default queue |
|-------|----------------|------------------|---------------|
| **Infrastructure (Intergrax Harness)** | 1, 2, 2b–2j (platform rows) | `intergrax/runtime/`, Tier-0 catalogs, H-APP, DX, MEM, ORCH, scaffold, CI audits, reference hosts | **Active** — §6.1 maintenance only |
| **Conformance shells (platform)** | 2g AA | `legal` / `legal_application` **scaffold** + deploy triad + tier hygiene (no domain UAEP steps) | **Done** (shell) |
| **Business agents & product apps** | 3, §6.3, AA-LEG.2.*, K.* | K.1/K.2, Legal UAEP steps, research/org domain tests, new `applications/<product>/`, live LLM E2E | **Deferred** — not default next |

**Module classification (repo inventory):**

| Module | Role | Queue |
|--------|------|-------|
| `agents/echo`, `agents/signoff_probe` | Harness reference Tier-2 | Infrastructure — **Done** |
| `agents/lab` | Lab mocks (not product agents) | Infrastructure — AA-LABAG.* optional |
| `applications/poc_template_application`, `applications/lab_application` | Reference Tier-3 hosts | Infrastructure — **Done** |
| `agents/legal`, `applications/legal_application` | Product shell on scaffold | Platform **Done**; domain logic **Deferred** (AA-LEG.2.2+) |
| `agents/research`, `applications/research_application` | Research prototype host | Platform **Done**; domain tests **Deferred** (AA-RES.4–5, AA-RESAPP.6) |
| `agents/organization_worker` | HITL / long-running demo | Docs **Done**; full scaffold + lab flag **Deferred** (AA-ORG.3–4) |
| `agents/problem_radar` | K.1 placeholder | **Frozen** — Band 3 (K.1) |
| New `applications/<product>/` beyond four hosts | Customer/product deploy | **Deferred** — §6.3 |

**Where to look for open work:**

| Topic | Section |
|-------|---------|
| **Canonical implementation queue (infrastructure)** | [§6.1](#61-harness-platform-maintenance-default--band-1) (**active** — maintenance) · [§6.1b](#61b-harness-implementation-queue--orchestration-closeout-closed) · [§6.1c](#61c-harness-implementation-queue--toolsskills-closeout-closed) · [§6.1d](#61d-harness-implementation-queue--integration-closeout-closed) · [§6.1e](#61e-harness-implementation-queue--rag-closeout-closed) (all closed) · [§6.1z](#61z-harness-implementation-queue-consolidated) (closed) |
| Integration catalog expansion (Done) | [M.6 P4](#m6-p4--harness-platform-expansion-done) · [§6.1w](#61w-harness-implementation-queue--integration-expansion-m6-p4-closed) — **28/28 Done** |
| Integration harness depth (Done) | [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done) — **33/34 Done** |
| Integration harness expansion | [M.6 P6](#m6-p6--harness-integration-expansion-planned) · [§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done) — **Done** (32/32 + wiring) |
| Ongoing gate + audit scripts | [§6.1](#61-harness-platform-maintenance-default--band-1) |
| Memory platform wiring (Done) | [Phase MEM](#phase-mem--memory-platform-completion) · [§6.2aa](#62aa-phase-mem-execution-order-band-2h--closed) |
| **Memory intelligence depth (closed)** | [Phase MEM-DEPTH](#phase-mem-depth--memory-intelligence-depth) · [`architecture/MEMORY.md`](architecture/MEMORY.md) · [§6.2ab](#62ab-phase-mem-depth-execution-order-band-2am--closed) |
| All business / domain work | [§6.3](#63-end-of-plan--deferred-product-work-only) · [Business backlog register](#63a-business-backlog-register-consolidated) |

### 4.1 Harness completion backlog (execution order)

Work **one ID per PR**; gate green after each step. Map fixes to Appendix G where applicable.

| Order | ID | Deliverable | Priority | Notes |
|-------|-----|-------------|----------|-------|
| 1 | U-Leg.2 | Remove or archive `intergrax/rag/answers/`; migrate tests to `RetrievalService` | **Done** | `intergrax/legacy/rag_answers/`; import guard |
| 2 | U-Leg.1 | Freeze `ToolsAgent.run` — docs + `check_tools_agent_run.py` | **Done** | Deprecation + CI audit |
| 3 | U-Leg.3 | Sunset legacy plan booleans (`from_legacy`, `uses_legacy_booleans_only`) | **Done** | Warnings + `check_legacy_tool_plan_booleans.py` |
| 4 | U-Typ.4 | `profile.slug_for_category` + sandbox `session_id` typing | **Done** | No getattr on integration profile |
| 5 | U-Arch.2 | Typed `LabIntegrationWiring` — sqlite bundle types | **Done** | Removed `# type: ignore` on lab wiring |
| 6 | U-CI.3 | CI job: `LAB_STRICT_HARNESS` + API key | **Done** | `harness-strict` workflow job |
| 7 | R-Skill.* | `harness.skill_registry` platform skill | **Done** | Harness bundle + gate test |
| 8 | U-Con.* | `ResearchAgent` / `SummaryAgent` → `HarnessReferenceAgent` | **Done** | Lab `requires_uaep` when research enabled |

**Explicitly out of NOW:** K.1, K.2, Legal product E2E, new `applications/<product>/`, Problem Radar wave 2+.



---
