# Experimentation And Developer Experience — Implementation Plan

**Architecture (1:1):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and this plan doc for the domain.

**Last updated:** 2026-06-17 — **Full Harness LC** (re-validates DX + W-OPS + AUDIT-IDEAL-26/27 closeout).

---

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



## 5. Definition of Done (Global)



1. **Contract** — Pydantic / Protocol public API

2. **Trace** — state transitions emit `TraceEvent` (+ `RuntimeEvent` where wired)

3. **Test** — unit + integration, deterministic, no network

4. **Documentation** — update this plan + [`guides/AGENT_CREATION_GUIDE.md`](guides/AGENT_CREATION_GUIDE.md) when workflow changes

5. **No regression** — `pytest tests/ -m gate` green; Echo through NexusLoop

6. **Reuse Tier-0** — extend existing modules; no parallel LLM/log/trace stacks (§5.2)
7. **Architecture governance** — for Phase V streams, update compatibility/evaluation evidence (graph impact + score deltas)
8. **Security/cost controls** — hardening changes include policy-enforced tests for deny/degrade paths
9. **No product scope creep** — harness phases MUST NOT implicitly include K.1/K.2 or new product hosts



---

---

## Phase EVAL — Evaluation control plane closeout

**Status:** **Done** (2026-06-02) — **4/4** deliverables Done (EVAL-DOC.1 + EVAL-1–3)

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §25; V-EVAL **Done**; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix U**.

**Priority ladder:** **Band 2x** (§4.0) — closed; default queue = **§6.1** maintenance.

### EVAL — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| EVAL-DOC.1 | EVAL0 | **Appendix U** — evaluation control plane closeout | **Done** | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| EVAL-1 | EVAL1 | **`EvaluationProfile`** + **`evaluation_runtime_bridge`** + **`evaluation_wiring`** | **Done** | `environment_profile.py`, `evaluation_runtime_bridge.py`, `evaluation_wiring.py`, `policy_wiring.py` | `test_harness_evaluation_wiring.py` |
| EVAL-2 | EVAL2 | **`evaluation_assembly_resolver`** — profile ↔ registry conformance | **Done** | `evaluation_assembly_resolver.py`, `harness_host_runtime.py`, `runtime_config_bridge.py`, `runtime.py` | assembly validation tests |
| EVAL-3 | EVAL3 | **Host evaluation CI** — `check_harness_evaluation_wiring.py` | **Done** | `scripts/`, CI workflow | audit script in CI |

**Explicitly excluded:** new business agents (K.1/K.2), product quality dashboards — [§6.3a](#63a-business-backlog-register-consolidated).

---

## Phase MVP-EVOL — MVP-to-product evolution layer (Band 2at — planned)

**Status:** **Done** (2026-06-09) — architecture canon §44; MVP-EVOL.1–6 implemented.

**Goal:** Deliver systematic **prototype → MVP → production** tooling: simulation harness, replay UX, KPI/satisfaction hooks, and promotion gate automation — competitive DX for product teams on Intergrax.

**Prerequisites:** Phase DX **Done**; Phase EVAL **Done**; lab host **Done**.

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| MVP-EVOL-DOC.1 | MVP0 | Canon §44 + hub cross-ref | **Done** | `docs/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` | ORCHE §58 index |
| MVP-EVOL.1 | MVP1 | **Promotion gate script** — G0–G2 CI checks (runnable, eval baseline, policy) | **Done** | `scripts/check_mvp_promotion_gates.py` | G0–G2 OK |
| MVP-EVOL.2 | MVP2 | **Agent simulator CLI** — multi-agent failure/contention scenarios | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp simulate` |
| MVP-EVOL.3 | MVP3 | **Trace replay** — reconstruct from trace store | **Done** | `intergrax/cli/mvp_evolution.py` | `intergrax mvp replay` |
| MVP-EVOL.4 | MVP4 | **Product KPI registry** — tenant-scoped metric definitions + export | **Done** | `product_kpi_registry.py` | unit tests deferred |
| MVP-EVOL.5 | MVP5 | **User satisfaction adapter** — thumbs / CSAT event schema + online eval bridge | **Done** | `user_satisfaction.py` | `test_user_satisfaction.py` |
| MVP-EVOL.6 | MVP6 | **Author guide appendix** — MVP evolution playbook | **Done** | `guides/AGENT_CREATION_GUIDE.md` Appendix X | TOC + scripts table |
| MVP-EVOL.7 | Exposure | **Tier-3 router optional** — HTTP endpoints for simulate/replay/KPI export (or document CLI-only canon) | **Done** | `mvp_evolution_routes.py` · lab `/v1/mvp/*` when `LAB_HARNESS=true` | CLI remains canonical |

**Cross-plan:** MVP-EVOL.2 ↔ ORCH CFG matrix; MVP-EVOL.5 ↔ OBS + EVAL online registry; promotion G4–G5 ↔ Phase V / W-OPS.

**Audit note (2026-06-09):** MVP-EVOL.1–6 **Done**; remaining debt is **exposure** (CLI vs product HTTP) — see [`architecture/ORCHESTRATION.md`](../architecture/ORCHESTRATION.md) §59.4.

**Explicitly excluded:** Product analytics SaaS UI; K.1/K.2 feature work.

---

## Phase DX — Developer Authoring Experience (fast environment + agent builds)

**Status:** **Done** (2026-06-02) — **47/47** deliverables **Done** in master table; gate **533+ passed**.  
**Prerequisites:** Phase **H-APP** **Done** (typed `ApplicationEnvironmentProfile`, `wire_application_environment`, `build_harness_host_runtime`). Phases **N**, **P-Ext**, **S** scaffold baseline **Done**.  
**Goal:** Make building **Tier-3 application environments** and **Tier-2 agents** trivial for Python developers — LangGraph-like mental model (state/steps → graph → run), **measurable** time-to-first-run (TTFRun), progressive disclosure (minimal → standard → production), and **UI-ready** serialized specs for Phase 2 (non-developer environment builder).  
**Priority ladder:** **Band 2f** (§4.0) — **closed for core path**; residual IDs are **infrastructure** follow-ups, not Band 3.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2y](#62y-phase-dx-execution-order-band-2f--mostly-done).

**Delivery rule:** One `DX-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green.

**Strategic split:**

| Phase | Audience | Outcome |
|-------|----------|---------|
| **DX (this phase)** | Python developers | Import contracts → define typed agents → configure environment → run HTTP/MCP in minutes |
| **Phase 2 (future — not DX)** | Business users via UI | Visual builder over same Pydantic/YAML specs (`DX-7.*` prepares artifacts only) |

**Target metrics (enforced by DX-3.5, DX-8.1):**

| Metric | Baseline (2026-06-03) | Target after DX |
|--------|----------------------|-----------------|
| **TTFRun** (scaffold → successful `POST …/run`) | ~45–90 min (docs + wiring) | **≤15 min** guided; **≤60 s** CI smoke |
| Author-edited files (hello world) | ~12–25 | **≤4** (`--minimal`) |
| Author LOC (excluding generated boilerplate) | ~200–400 | **≤120** |
| Commands to first run | 3+ | **1** (`intergrax run`) |
| Scaffold H-APP alignment | Partial (`factory.py` legacy path) | **100%** |

**LangGraph mapping (author mental model — implement in DX-0.2):**

| LangGraph | Intergrax (DX target) |
|-----------|------------------------|
| `State` fields | `AgentContract` + `RuntimeExecutionContext.metadata` |
| Node function | `@step` / `run_step` on `IntergraxAgent` |
| Conditional edges | `decide_after_step` → `AgentDecision` |
| `StateGraph.compile()` | `AgentGraph.build()` → `ApplicationGraphSpec` |
| `app.invoke()` | `HarnessApplication.serve()` / `POST /v1/…/run` |

**Out of scope (not counted in 47):** Band 3 product agents (K.1/K.2); visual graph editor UI; integration marketplace; catalog hot-reload; renaming `applications/` → `harness/` (canon §5.3.0 — **Application** = Tier-3 instance, **Harness** = platform); LangGraph skill pack import.

```text
Wave DX0 — Docs & traceability (4 tasks)
Wave DX1 — Scaffold/H-APP alignment fix (6 tasks) — P0 before facades
Wave DX2 — Authoring facades: HarnessApplication, AgentGraph, IntergraxAgent (6 tasks)
Wave DX3 — Minimal path + CLI + TTFRun gates (6 tasks)
Wave DX4 — Integration presets & picker (4 tasks)
Wave DX5 — Host hooks, YAML, observability/logging DX (8 tasks)
Wave DX6 — Tier hygiene + external projects (5 tasks)
Wave DX7 — UI engine prep: JSON Schema, spec versioning, catalog feed (5 tasks)
Wave DX8 — DX metrics & CI guards (3 tasks)
Total: 47
```

### DX — Traceability (audit gap → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| L1 | Scaffold generates legacy + H-APP wiring in parallel | DX-1.1–DX-1.2, DX-1.6, DX-8.3 |
| L2 | No minimal hello harness (1–3 files) | DX-3.1, DX-2.1–DX-2.3, DX-3.2 |
| L3 | No fluent graph API | DX-2.2, DX-7.3 |
| L4 | No `HarnessApplication` / single entry class | DX-2.1, DX-2.6, DX-5.1–DX-5.2 |
| L5 | Monorepo-only (`pythonpath`) | DX-6.3–DX-6.5 |
| L6 | Tier-2 agents import `applications/_shared` | DX-6.1–DX-6.2 |
| L7 | IntegrationProfile slot knowledge burden | DX-4.1–DX-4.3, DX-4.2 |
| L8 | No `intergrax run` / `intergrax doctor` / TTFRun metric | DX-3.2–DX-3.3, DX-3.5, DX-8.1–DX-8.2 |
| L9 | Documentation sprawl, no single 15-min path | DX-0.1–DX-0.4, DX-3.6 |
| L10 | No JSON Schema / stable spec for UI phase 2 | DX-7.1–DX-7.5 |
| H-APP.5.3 gap | `poc_template` / scaffold `factory.py` not on `build_nexus_loop_from_environment` | DX-1.1, DX-1.3 |
| §6 responsibility table | Agent vs environment concerns split | DX-0.3 |
| Progressive disclosure | minimal → standard → production | DX-0.4, DX-3.4 |
| Architecture audit rec. | Product observability preset, trace_id in logs, event catalog | DX-5.5–DX-5.7 |
| Architecture audit rec. | Policy handler plugins (extend without core PR) | DX-5.8 |
| Do not weaken tiers | Doctor/checks enforce boundaries | DX-0.3, DX-3.3, DX-6.2, DX-8.3 |

### DX — Master deliverables register (all 47 tasks)

| ID | Wave | Deliverable | Status | Priority | Location / acceptance |
|----|------|-------------|--------|----------|------------------------|
| DX-0.1 | DX0 | **Phase DX register** in this plan + doc model row (§Documentation model) | **Done** | Low | This section + §6.2y |
| DX-0.2 | DX0 | **LangGraph ↔ Intergrax mapping** table (state, nodes, edges, compile, invoke) | **Done** | High | `guides/EXTENSION_AUTHOR_GUIDE.md` §0 or `guides/AGENT_CREATION_GUIDE.md` §1 |
| DX-0.3 | DX0 | **Responsibility matrix** — what belongs in agent vs environment (single canonical table) | **Done** | High | `guides/EXTENSION_AUTHOR_GUIDE.md` §0 + cross-link canon §5.3.0 |
| DX-0.4 | DX0 | **Progressive disclosure** doc — minimal (`--minimal`) → standard scaffold → production (`expand`, Docker, MCP) | **Done** | Medium | `guides/AGENT_CREATION_GUIDE.md` Step 4E § E.0 + `applications/USAGE.md` |
| DX-1.1 | DX1 | **Scaffold `factory.py`** — build `NexusLoop` only via `build_nexus_loop_from_environment(registry, env, …)` + integration bundle from `wire_application_environment` | **Done** | **Critical** | `intergrax/scaffold/new_application.py`, `new_application_product.py` |
| DX-1.2 | DX1 | **Scaffold default output** — remove generated `integration_wiring.py` + `tool_wiring.py`; retain via `--full` flag only | **Done** | **Critical** | Scaffold CLI + README in generated app |
| DX-1.3 | DX1 | **Migrate `poc_template_application/host/factory.py`** to H-APP factory pattern (no parallel legacy wiring) | **Done** | High | Parity with `lab_application`; gate smoke |
| DX-1.4 | DX1 | **Audit + fix** `legal_application` / `research_application` factories — single env path, no duplicate tool/integration wiring | **Done** | High | Host smoke tests green |
| DX-1.5 | DX1 | **Scaffold manifest** — embed `environment: ApplicationEnvironmentProfile…` at generation (not only lazy `environment_profile.py` fallback) | **Done** | High | Generated `manifest.py` |
| DX-1.6 | DX1 | **Gate test** — scaffold output: `factory.py` must not import `host.tool_wiring` / `host.integration_wiring` unless `--full` | **Done** | High | `tests/unit/scaffold/test_scaffold_harness_alignment.py` |
| DX-2.1 | DX2 | **`HarnessApplication` facade** — `.agents()`, `.integrations()`, `.graph()`, `.mode()`, `.llm()`, `.hooks()`, `.build()`, `.serve()` | **Done** | **Critical** | `intergrax/harness/app.py` |
| DX-2.2 | DX2 | **`AgentGraph` fluent builder** — nodes, edges, default agent, `on_error(retry=…)` → `ApplicationGraphSpec` | **Done** | **Critical** | `intergrax/applications/contracts/graph_builder.py` |
| DX-2.3 | DX2 | **`IntergraxAgent` base** + **`@step` decorator** — generates UAEP `get_steps` / `run_step` wiring | **Done** | **Critical** | `intergrax/agents/authoring/` |
| DX-2.4 | DX2 | **Decision helpers** — `continue_to()`, `complete()`, `delegate_to()` wrapping `AgentDecision` | **Done** | Medium | `intergrax/agents/authoring/decisions.py` |
| DX-2.5 | DX2 | **Unit test** — minimal `HarnessApplication` + `EchoAgent`/`IntergraxAgent` runs offline (no network) | **Done** | High | `tests/unit/harness/test_harness_application_minimal.py` |
| DX-2.6 | DX2 | **Public package** `intergrax.harness` — stable imports documented in author guide | **Done** | High | `intergrax/harness/__init__.py` |
| DX-3.1 | DX3 | **`new-stack --minimal`** — ≤4 author-facing files + smoke test (no Docker/MCP by default) | **Done** | **Critical** | `intergrax/scaffold/new_stack.py`, `new_application.py` `--minimal` |
| DX-3.2 | DX3 | **`intergrax run <module>:app`** — load `.env`, uvicorn, print route + sample curl | **Done** | High | `intergrax/cli/run.py` + `scaffold/cli.py` |
| DX-3.3 | DX3 | **`intergrax doctor`** — tier import violations, manifest/env conformance, scaffold freshness hint, TTFRun estimate | **Done** | High | `intergrax/cli/doctor.py` |
| DX-3.4 | DX3 | **`intergrax scaffold expand`** — promote minimal app → standard (Docker, MCP, debug, BUILD_AND_DEPLOY) | **Done** | Medium | `intergrax/scaffold/expand_application.py` |
| DX-3.5 | DX3 | **Acceptance test** `test_minimal_stack_ttf_run` — scaffold minimal → pytest → HTTP run **≤60s** in CI | **Done** | High | `tests/acceptance/dx/test_minimal_stack_ttf_run.py` |
| DX-3.6 | DX3 | **15-minute quickstart** — single numbered path: `new-stack --minimal` → edit agent → `intergrax run` → curl | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` Step 4E § E.0 |
| DX-4.1 | DX4 | **`IntegrationProfile` presets** — `.lab_stack()`, `.legal_stack()`, `.data_stack()`, `.observability_stack()` (typed, documented slugs) | **Done** | High | `intergrax/integrations/registry/presets.py` |
| DX-4.2 | DX4 | **`intergrax integrations pick`** CLI — emit profile fragment (postgres, redis, s3, prometheus, …) for `environment_profile.py` | **Done** | Medium | `intergrax/cli/integrations_pick.py` |
| DX-4.3 | DX4 | **Preset catalog table** in `architecture/INTEGRATIONS.md` + `guides/EXTENSION_AUTHOR_GUIDE.md` | **Done** | Medium | `architecture/INTEGRATIONS.md` § Named integration presets |
| DX-4.4 | DX4 | **Gate tests** — each preset resolves with in-memory/sqlite stubs (no network) | **Done** | High | `tests/unit/integrations/test_integration_presets.py` |
| DX-5.1 | DX5 | **`ApplicationHost` Protocol/base** — override methods for environment control (intake, agent selection, finalize, error) | **Done** | High | `intergrax/harness/application_host.py` |
| DX-5.2 | DX5 | **Map host overrides → `HookPoint`** + optional `RuntimeEventBus` subscribe API on `HarnessApplication` | **Done** | High | Bridge in `intergrax/harness/hooks.py` |
| DX-5.3 | DX5 | **`HarnessApplication.from_yaml(path)`** — load `ApplicationEnvironmentProfile` + roster from `env.yaml` | **Done** | Medium | `intergrax/harness/yaml_loader.py` |
| DX-5.4 | DX5 | **Optional `agents.yaml`** — declarative `AgentBinding` list validated against importable classes | **Done** | Low | Same loader; schema test |
| DX-5.5 | DX5 | **Product scaffold observability preset** — `ObservabilityProfile` template (trace + optional read-only debug) | **Done** | Medium | `new_application_product.py` `environment_profile.py` (`otel_enabled`, debug override) |
| DX-5.6 | DX5 | **Structured log correlation** — inject `trace_id` / `run_id` in FastAPI middleware (lab + product factories) | **Done** | Medium | `intergrax/applications/_shared/logging_middleware.py` |
| DX-5.7 | DX5 | **Runtime event catalog table** — `RuntimeEventType` → emit phase → ops filter hints in canon §42 | **Done** | Low | `intergrax_runtime_architecture.md` §42.1.5; `phase_coverage.EVENT_OPS_FILTER_HINTS` |
| DX-5.8 | DX5 | **Policy rule handler plugins** — entry point group `intergrax.policy_rules` (mirror P-Ext pattern) | **Done** | Medium | `runtime/policy/rules/` + author guide § |
| DX-6.1 | DX6 | **`intergrax.agents.defaults`** — `harness_production_mode`, lab runtime config helpers (no Tier-3 import from agents) | **Done** | High | `intergrax/agents/defaults.py`; Tier-3 re-export in `runtime_defaults.py` |
| DX-6.2 | DX6 | **Fix reference agents** — `echo`, `research` (and scaffold template) must not import `applications/_shared` | **Done** | High | `agents/echo/`, `agents/research/` + `check_agent_registry_bypass` |
| DX-6.3 | DX6 | **`intergrax init <project>`** — cookiecutter: external repo, `pip install intergrax`, minimal harness layout | **Done** | High | `intergrax/scaffold/external_project/` template |
| DX-6.4 | DX6 | **CI smoke** — generated external template project pytest (fixture repo) | **Done** | Medium | `tests/integration/dx/test_external_project_template.py` |
| DX-6.5 | DX6 | **`pyproject` optional extra `[harness-author]`** — documented minimal dependency set for external apps | **Done** | Low | `pyproject.toml` + README |
| DX-7.1 | DX7 | **JSON Schema export** for `ApplicationEnvironmentProfile`, `ApplicationManifest`, `ApplicationGraphSpec` | **Done** | High | `scripts/export_harness_spec_schemas.py` → `build/harness_specs/` (CI) |
| DX-7.2 | DX7 | **`spec_version` on environment profile** + migration note in plan | **Done** | Medium | `environment_profile.py` |
| DX-7.3 | DX7 | **YAML round-trip tests** — graph + environment serialize/deserialize without loss | **Done** | High | `tests/unit/harness/test_spec_roundtrip.py` |
| DX-7.4 | DX7 | **Capability catalog JSON feed** — integrations/tools/skills slugs + labels for future UI builder | **Done** | Medium | `scripts/export_capability_catalog_feed.py` (CI) |
| DX-7.5 | DX7 | **Phase 2 UI boundary doc** — UI engine consumes DX-7 artifacts only; no parallel spec | **Done** | Low | Plan §Phase DX — UI boundary (below) |
| DX-8.1 | DX8 | **`intergrax doctor --ci`** — fail on tier violations, scaffold misalignment, TTFRun regression | **Done** | High | `.github/workflows/unit-tests.yml` |
| DX-8.2 | DX8 | **DX metrics in paydown** — record TTFRun seconds, author file count per release cycle | **Done** | Low | `scripts/record_dx_metrics.py` → `build/architecture_hardening/dx_metrics.json` |
| DX-8.3 | DX8 | **`check_scaffold_harness_alignment.py`** — CI script (complements DX-1.6 gate) | **Done** | High | `scripts/` + §6.1 maintenance list |

### DX — Explicitly deferred (not in the 47-task register)

| Topic | Reason |
|-------|--------|
| Visual graph editor / drag-and-drop UI | Phase 2 product; DX-7 only exports schemas |
| Rename `applications/` directory | Canon decision: Application = Tier-3 deployable instance |
| K.1 / K.2 business agents | Band 3 frozen (§6.3) |
| Full LangGraph runtime compatibility | Different execution model; mapping doc only (DX-0.2) |
| `phase_w_ops_evidence --verify-gate` hardening | W-OPS maintenance, not DX |

### DX — Paydown log

| Date | DX ID | Summary |
|------|-------|---------|
| 2026-06-03 | DX-1.1–DX-8.3 (core) | HarnessApplication, scaffold H-APP alignment, CLI run/doctor, presets, `check_scaffold_harness_alignment`; gate **518** |
| 2026-06-02 | Plan sync | Master table synced to codebase; **17** IDs remain **Pending** — [residual backlog](#dx--residual-backlog-infrastructure) |
| 2026-06-02 | DX residual closeout | `--minimal` stack, `expand`, doctor CI, spec export + round-trip, TTFRun acceptance, `agents.defaults.harness_production_mode`, docs quickstart; gate **533** |
| 2026-06-02 | DX-5.7 | §42.1.5 event catalog + `EVENT_OPS_FILTER_HINTS`; Phase DX **47/47 Done** |

**Suggested PR order (residual):** None — Phase DX infrastructure **Done**.

**Phase 2 UI boundary (DX-7.5):** A future visual builder must consume only versioned artifacts from `build/harness_specs/*.json` and `build/capability_catalog_feed.json` — not parallel Pydantic copies. Host behavior stays `HarnessApplication` / Tier-3 factories; UI edits serialize to the same `ApplicationEnvironmentProfile` + `ApplicationManifest` + `ApplicationGraphSpec` models validated by DX-7.1/DX-7.3.

### DX — Residual backlog (infrastructure)

**Not Band 3.** Platform DX rows **Done** (2026-06-02), including DX-5.7 (§42.1.5). No open DX IDs — see [§6.1z](#61z-harness-implementation-queue-consolidated).

---

---

## Phase AA — Agents & Applications Conformance (scaffold, docs, deploy)

**Status:** **Mostly Done** (2026-06-02) — **platform/conformance Done** (tier hygiene, ARCHITECTURE matrix, deploy triad, legal **scaffold** reset); **domain steps Deferred** (AA-LEG.2.2+); gate **534 passed**.  
**Prerequisites:** Phase **H-APP** **Done**, Phase **DX** **Mostly Done** (scaffold generators, `build_harness_host_runtime`, CLI, presets).  
**Goal:** Bring every **Tier-2** agent under `agents/` and every **Tier-3** host under `applications/` to a **documented, scaffold-aligned** state — fast authoring, full environment control (handlers, observability, policy), and **repeatable deploy** (Docker + deploy doc + `pyproject.toml` dependency contract per application). **Domain UAEP implementation is Band 3** — see [§6.3](#63-end-of-plan--deferred-product-work-only).  
**Priority ladder:** **Band 2g** (§4.0) — **platform rows closed**; only [AA residual](#aa--residual-backlog-infrastructure) + §6.1 maintenance.  
**Scope split:** [§4.0a](#40a-implementation-scope-split-infrastructure-vs-business).  
**Execution order:** [§6.2z](#62z-phase-aa-execution-order-band-2g--mostly-done).

**Delivery rule:** One `AA-*` ID per PR → update status in tables below + paydown log → `pytest -m gate` + §6.1 audit scripts green → discuss scope in session before coding.

**Policy decision (2026-06-03):** **`agents/legal/` hard reset** — remove pre-architecture implementation; regenerate from `intergrax.scaffold new-agent legal` and re-implement only against UAEP + H-APP rules. Legacy tests become **behavioral spec** input, not code to preserve. **`legal_application`** follows the same reset cadence after the agent baseline exists (product shell only).

**Inventory (in scope):**

| Tier | Slug | Role |
|------|------|------|
| Agent | `echo` | Harness reference agent |
| Agent | `lab` | Lab mock agents (not product agents) |
| Agent | `legal` | **Hard reset** → scaffold baseline |
| Agent | `organization_worker` | Long-running / HITL demo |
| Agent | `problem_radar` | K.1 prototype (frozen until Band 3) |
| Agent | `research` | Multi-agent research prototype |
| Agent | `signoff_probe` | Appendix A sign-off exercise |
| Application | `lab_application` | Universal lab / debug superset |
| Application | `legal_application` | Legal product host (reset with agent) |
| Application | `poc_template_application` | Canonical minimal Tier-3 shell |
| Application | `research_application` | Research product host |

**Per-application deploy triad (mandatory for every Tier-3 host in this phase):**

Each `applications/<app>/` MUST document and maintain:

| Piece | Path / generator | Acceptance |
|-------|------------------|------------|
| **Docker** | `docker/Dockerfile`, `docker/docker-compose.yml`, `docker/build-docker.sh`, `docker/build-docker.bat`, `docker/.dockerignore` | Image builds locally; health path matches manifest `route_prefix` |
| **Deploy doc** | `BUILD_AND_DEPLOY.md` | Generated/updated via `intergrax.applications._shared.build_deploy_doc.render_build_deploy_doc` (scaffold) or manual parity with scaffold output |
| **`ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md`** | Per agent/app directory | Cross-linked doc pair; local task queue in `IMPLEMENTATION_PLAN.md`; scaffold emits both (`intergrax/scaffold/doc_templates.py`) |
| **`pyproject.toml` deps** | Root `[project]` + `[project.optional-dependencies]` | Section in `applications/<app>/ARCHITECTURE.md`: which **core** deps apply, which **extras** (`harness-author`, `langgraph-legacy`, `llm-*`, `dev-ci`, integration extras) the host requires; no undeclared imports |

Scaffold already emits Docker + `BUILD_AND_DEPLOY.md` for **new** apps (`new-application`, `new-stack`). Phase AA **backfills and verifies** this triad on all four existing applications.

**Audit verdict (2026-06-03):**

| Area | Verdict | Gap → AA IDs |
|------|---------|----------------|
| Tier-2 structure vs canon | **OK** | Tier-3 imports removed; `legal` scaffold baseline; CI `check_agents_no_tier3_imports.py` |
| Tier-3 H-APP factory | **OK** | `build_harness_host_runtime` on lab/poc/legal/research; manifest `environment=` |
| Scaffold completeness | **OK** | Typed `can_handle`, `--reference`, deploy triad regression test |
| Documentation | **OK** | ARCHITECTURE.md matrix, guides, TIER3_READINESS — AA-D0.* **Done** |
| LangGraph independence | **Done** | `langgraph` not in core deps; `check_langgraph_not_required.py` — AA-LG.1 **Done** |
| Legal module | Reset required | AA-LEG.* + AA-LEGAPP.* |

```text
Wave AA0  — Register, scaffold checklist, LangGraph (done), deploy triad standard (5)
Wave AA1  — Platform docs meta: README, guides, TIER3_READINESS (7)
Wave AA2  — Legal agent HARD RESET (12)
Wave AA3  — legal_application reset + deploy triad (8)
Wave AA4  — echo agent (5)
Wave AA5  — signoff_probe agent (3)
Wave AA6  — problem_radar agent (5)
Wave AA7  — organization_worker agent (5)
Wave AA8  — research agent (+ summary) (6)
Wave AA9  — lab mock agents doc (2)
Wave AA10 — lab_application host (7)
Wave AA11 — poc_template_application host (5)
Wave AA12 — research_application host (6)
Total: 83 (incl. AA-LG.1 counted in AA0)
```

### AA — Traceability (audit topic → task IDs)

| Audit ref | Topic | Task IDs |
|-----------|--------|----------|
| A1 | Tier-2/3 separation (`agents` must not import `applications`) | AA-S0.2, AA-ECHO.2, AA-PR.2, AA-D0.3 |
| A2 | Scaffold agent `getattr` in `can_handle` | AA-S0.3 |
| A3 | Scaffold app missing manifest `environment=` | AA-S0.4, AA-POC.2, AA-RESAPP.2 |
| A4 | `lab_application` legacy Nexus factory | AA-LABAPP.2 |
| A5 | Legal pre-UAEP monolith | **AA-LEG.1–AA-LEG.12** (hard reset) |
| A6 | Per-agent architecture MD | AA-ECHO.1, AA-PR.1, AA-ORG.1, AA-RES.1, AA-SIG.1, AA-LEG.3 |
| A7 | Per-application architecture MD + deploy triad | AA-LABAPP.1, AA-POC.1, AA-RESAPP.1, AA-LEGAPP.1, AA-APP.0.1–AA-APP.0.3 |
| A13 | Doc pair `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` (scaffold + gate) | AA-D0.6, `intergrax/scaffold/doc_templates.py` |
| A8 | Root README completeness vs canon | AA-D0.1 |
| A9 | `AGENT_CREATION_GUIDE` / `TIER3_READINESS` stale | AA-D0.2–AA-D0.4 |
| A10 | LangGraph not required | AA-LG.1 (**Done**) |
| A11 | Docker + deploy script + pyproject per app | AA-APP.0.1–AA-APP.0.3, AA-*APP.*.4–*.6 |
| A12 | Legal application custom serving vs scaffold | AA-LEGAPP.3–AA-LEGAPP.5 |

### AA — Master deliverables register (all tasks)

#### Wave AA0 — Platform & scaffold foundation

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-0.1 | **Phase AA register** in this plan + §6.2z + doc model row | **Done** | Low | This section |
| AA-0.2 | **Scaffold ↔ H-APP checklist** table (new-agent / new-application / new-stack outputs) | **Done** | High | This section §AA scaffold matrix (below) |
| AA-S0.1 | Audit script: tier-2 must not import `applications` (extend `check_agent_registry_bypass` or sibling) | **Done** | High | `scripts/` + CI §6.1 |
| AA-S0.2 | **`new-agent`**: remove `getattr` from generated `can_handle` — typed `TaskContext` | **Done** | High | `intergrax/scaffold/new_agent.py` |
| AA-S0.3 | **`new-agent`**: optional `--reference` template (`HarnessReferenceAgent`) vs pure `Agent` | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.4 | **`new-agent`**: scaffold `contract.py` includes `skill_ids` placeholder + link architecture/SKILLS.md | **Done** | Medium | `intergrax/scaffold/new_agent.py` |
| AA-S0.5 | **`new-application`**: manifest always embeds `environment=ApplicationEnvironmentProfile…` | **Done** | High | `intergrax/scaffold/new_application.py` |
| AA-S0.6 | Document **`--full`** vs default scaffold (integration/tool wiring) | **Done** | Medium | `applications/USAGE.md` |
| AA-LG.1 | **LangGraph optional** — not in core deps; `langgraph-legacy` extra; `check_langgraph_not_required.py` | **Done** | High | `pyproject.toml`, CI |
| AA-APP.0.1 | **Deploy triad standard** — Docker + `BUILD_AND_DEPLOY.md` + pyproject extras section (canonical template) | **Done** | High | `applications/USAGE.md` §Deploy triad |
| AA-APP.0.2 | **Gate**: each existing `applications/*_application/` has `docker/`, `BUILD_AND_DEPLOY.md`, ARCHITECTURE deploy section | **Done** | High | `tests/unit/applications/test_application_deploy_triad.py` (incl. `local_workspace_application`) |
| AA-APP.0.3 | **Scaffold verify**: `new-application` output includes deploy triad (regression) | **Done** | High | `tests/unit/scaffold/test_scaffold_deploy_triad.py` |
| AA-D0.6 | **Gate**: doc pair `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` on listed agents/apps; cross-links | **Done** | High | `tests/unit/applications/test_agent_app_doc_pair.py` |

**AA scaffold matrix (generator vs H-APP target):**

| Output | `new-agent` | `new-application` (default) | `new-application --full` |
|--------|-------------|----------------------------|---------------------------|
| UAEP `Agent` + ``on_next_step` / cognitive pattern hooks` | Yes | — | — |
| `contract.py` / `capabilities.py` | Yes | — | — |
| `ARCHITECTURE.md` + `IMPLEMENTATION_PLAN.md` | Yes | Yes | Yes |
| `manifest.py` + `AgentBinding` | — | Yes | Yes |
| `host/environment_profile.py` | — | Yes | Yes |
| `host/factory.py` → `build_harness_host_runtime` | — | Yes | Yes |
| `host/integration_wiring.py` | — | No | Yes |
| `host/tool_wiring.py` | — | No | Yes |
| `host/policy/rules/` | — | Yes (`.gitkeep`) | Yes |
| `docker/*` + `BUILD_AND_DEPLOY.md` | — | Yes | Yes |
| MCP + smoke tests | — | Yes | Yes |

#### Wave AA1 — Documentation meta (canon alignment)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-D0.1 | **Root `README.md`** — `HarnessApplication`, `intergrax` CLI, `poc_template` as Tier-3 reference, LangGraph optional, agent vs app matrix | **Done** | High | `README.md` |
| AA-D0.2 | **`docs/README.md`** — Phase AA row, last updated | **Done** | Low | `docs/README.md` |
| AA-D0.3 | **`guides/AGENT_CREATION_GUIDE.md`** — DX paths (`intergrax run`, `doctor`, minimal stack); no stale Nexus-only flow | **Done** | High | `docs/guides/AGENT_CREATION_GUIDE.md` |
| AA-D0.4 | **`applications/TIER3_READINESS.md`** — `environment_profile`, `build_harness_host_runtime`; deploy triad; no mandatory `tool_wiring` for all apps | **Done** | High | `applications/TIER3_READINESS.md` |
| AA-D0.5 | **`applications/USAGE.md`** — deploy triad + pyproject extras per host | **Done** | High | `applications/USAGE.md` |
| AA-D0.6 | **`guides/EXTENSION_AUTHOR_GUIDE.md`** — LangGraph analogy only (not required) — verify post AA-LG.1 | **Done** | Low | Already partially done |
| AA-D0.7 | **Conformance index** in plan — agent/app status columns (this register) | **Done** | Low | Appendix row or §AA paydown |

#### Wave AA2 — `agents/legal` HARD RESET (decision: scaffold baseline only)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEG.0.1 | **Record hard-reset decision** in plan + remove “incremental migration” as default for legal | **Done** | Critical | This section |
| AA-LEG.0.2 | **Archive tag** `legal-legacy-pre-aa` on git (pointer for forensic diff) | **Done** | High | Tag on parent of `bbce1bd` (pre hard-reset) |
| AA-LEG.0.3 | **Extract behavioral spec** from legacy tests → `agents/legal/SPEC_FROM_LEGACY.md` (requirements only) | **Done** | High | Before delete |
| AA-LEG.1.1 | **Delete** legacy `agents/legal/` tree (pipeline, governance, custom loop, tracing dupes) | **Done** | **Critical** | PR after AA-LEG.0.3 |
| AA-LEG.1.2 | **`python -m intergrax.scaffold new-agent legal --capability legal.review`** (force clean tree) | **Done** | **Critical** | `agents/legal/` matches scaffold layout |
| AA-LEG.1.3 | **`agents/legal/ARCHITECTURE.md`** — target UAEP graph, skills, tools, config, observability hooks (design-only until steps exist) | **Done** | High | English canonical doc |
| AA-LEG.2.1 | **Register** `legal` skill bundle on contract (`skill_ids`) per architecture/SKILLS.md | **Done** | High | `contract.py` |
| AA-LEG.2.2 | **UAEP steps** — port minimal slice from spec (one step per PR) | **Deferred** | High | `steps/` |
| AA-LEG.2.3 | **Remove** custom `legal_execution_loop`, `legal_tool_runtime_bridge` patterns — use Nexus `RuntimeToolGateway` only | **Deferred** | High | No parallel runtime |
| AA-LEG.2.4 | **Agent tests** — smoke + one spec-backed test per ported step | **Deferred** | High | `agents/legal/tests/` |
| AA-LEG.2.5 | **Retire** `ROADMAP.md` / `IMPLEMENTATION_PLAN.md` / `HOST_README.md` under agent — merge into `ARCHITECTURE.md` | **Done** | Medium | Single agent doc |
| AA-LEG.3.1 | **Gate**: `legal` agent imports no `applications.*`; no `getattr` on contract | **Done** | High | CI scripts |

**Explicitly NOT in legal reset:** Live LLM E2E product proof (Band 3 — K.6 / B.15 / S-Ops.4).

#### Wave AA3 — `applications/legal_application` (reset with agent)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LEGAPP.1 | **`ARCHITECTURE.md`** — manifest, environment profile, factory, auth, observability DB paths, MCP | **Done** | High | `applications/legal_application/` |
| AA-LEGAPP.2 | **Manifest** — `environment=ApplicationEnvironmentProfile.product_defaults(…)` inline | **Done** | High | `manifest.py` |
| AA-LEGAPP.3 | **Factory/serving** — align to `poc_template` + product settings; remove redundant `runtime_bridge` if superseded by `UnifiedTaskRunner` | **Done** | High | `host/factory.py`, `serving/` |
| AA-LEGAPP.4 | **Deploy triad** — verify/update `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | See AA-APP.0.1 |
| AA-LEGAPP.5 | **`pyproject.toml` deps section** in ARCHITECTURE — `harness-author`, LLM extras, optional `langgraph-legacy` N/A | **Done** | High | ARCHITECTURE §Dependencies |
| AA-LEGAPP.6 | **Host smoke** — `legal_tests/` green on scaffolded agent only | **Deferred** | High | After AA-LEG.2.2 |
| AA-LEGAPP.7 | **`.env.example`** parity with scaffold product profile | **Done** | Medium | `.env.example` |
| AA-LEGAPP.8 | **Remove** duplicate legal test trees if consolidated | **Deferred** | Low | `legal_tests/` vs agent tests |

#### Wave AA4 — Agent `echo`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ECHO.1 | **`agents/echo/ARCHITECTURE.md`** — reference role, capabilities, skills, lab registration | **Done** | High | English |
| AA-ECHO.2 | **Remove Tier-3 imports** — inject `LabHarnessContext` from `lab_application` factory only | **Done** | **Critical** | `agents/echo/echo_agent.py` |
| AA-ECHO.3 | Align with **`HarnessReferenceAgent`** pattern documented in canon | **Done** | Medium | Code + doc |
| AA-ECHO.4 | **Tests** — import agent module without `applications` on PYTHONPATH | **Done** | High | `tests/unit/agents/` |
| AA-ECHO.5 | **README** — pointer to ARCHITECTURE only | **Done** | Low | `agents/echo/README.md` |

#### Wave AA5 — Agent `signoff_probe`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-SIG.1 | **`ARCHITECTURE.md`** — Appendix A sign-off flow, capability `signoff.probe` | **Done** | Medium | `agents/signoff_probe/` |
| AA-SIG.2 | Verify **scaffold parity** when AA-S0.2 lands (regenerate diff empty except domain) | **Done** | Low | `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-SIG.3 | **README** → ARCHITECTURE link | **Done** | Low | |

#### Wave AA6 — Agent `problem_radar`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-PR.1 | **`ARCHITECTURE.md`** — K.1 placeholder, I/O schema, policy | **Done** | Medium | Frozen until Band 3 |
| AA-PR.2 | **Remove Tier-3 imports** (same pattern as echo) | **Done** | High | `problem_radar_agent.py` |
| AA-PR.3 | **Notebook + tests** documented in ARCHITECTURE | **Done** | Low | |
| AA-PR.4 | **Status** in plan §6.3 — no feature work until K.1 reprioritized | **Done** | Low | |
| AA-PR.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA7 — Agent `organization_worker`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-ORG.1 | **`ARCHITECTURE.md`** — HITL, long-running, `org.vendor_report` | **Done** | Medium | |
| AA-ORG.2 | **Remove `testing_support` import** — fake LLM via test fixture injection | **Done** | High | `organization_worker_agent.py` |
| AA-ORG.3 | **Scaffold-align** — add `contract.py`, `capabilities.py`, `steps/` if missing | **Deferred** | Medium | |
| AA-ORG.4 | **Lab manifest flag** + integration test | **Deferred** | Medium | `lab_application/manifest.py` |
| AA-ORG.5 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA8 — Agent `research` (+ `summary_agent`)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RES.1 | **`agents/research/ARCHITECTURE.md`** — graph intent `research.pipeline`, two agents | **Done** | High | |
| AA-RES.2 | **Remove Tier-3 imports** from agents if any | **Done** | High | |
| AA-RES.3 | **`HarnessReferenceAgent`** alignment for Research/Summary | **Done** | Medium | |
| AA-RES.4 | **Skill ids** on contracts | **Deferred** | Medium | |
| AA-RES.5 | **Tests** — UAEP + graph delegation | **Deferred** | High | |
| AA-RES.6 | **README** merge into ARCHITECTURE | **Done** | Low | |

#### Wave AA9 — Agent `lab` (mocks)

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAG.1 | **`agents/lab/README.md`** — mock agents purpose, not product Tier-2 | **Done** | Low | `agents/lab/README.md` |
| AA-LABAG.2 | **(Optional)** move mocks to `testing_support/` if they are test-only | **Won't fix** | Low | Until leadership requests — mocks stay under `agents/lab/` |

#### Wave AA10 — Application `lab_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-LABAPP.1 | **`ARCHITECTURE.md`** — debug API, interaction, scheduler, manifest flags | **Done** | High | |
| AA-LABAPP.2 | **Migrate factory** to `build_harness_host_runtime` (retain rich wiring via env profile) | **Done** | **Critical** | `host/factory.py` |
| AA-LABAPP.3 | **`environment` in manifest** or documented single profile builder | **Done** | High | `manifest.py` / `_shared` |
| AA-LABAPP.4 | **Deploy triad** — verify `docker/*`, `BUILD_AND_DEPLOY.md` | **Done** | High | |
| AA-LABAPP.5 | **`pyproject.toml` deps** section in ARCHITECTURE | **Done** | High | |
| AA-LABAPP.6 | **Smoke tests** after factory migration | **Done** | High | `lab_application_tests/host/test_lab_host_smoke.py` + `tests/acceptance/agent_os/test_lab_application.py` |
| AA-LABAPP.7 | **README** → ARCHITECTURE | **Done** | Low | |

#### Wave AA11 — Application `poc_template_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-POC.1 | **`ARCHITECTURE.md`** — canonical Tier-3 lab shell (reference for new apps) | **Done** | High | |
| AA-POC.2 | **Manifest `environment=`** explicit (not only factory fallback) | **Done** | High | `manifest.py` |
| AA-POC.3 | **Deploy triad** verification | **Done** | Medium | |
| AA-POC.4 | **`pyproject.toml` deps** section | **Done** | Medium | |
| AA-POC.5 | **Link from root README** as “start here for new application” | **Done** | Medium | AA-D0.1 |

#### Wave AA12 — Application `research_application`

| ID | Deliverable | Status | Priority | Location / acceptance |
|----|-------------|--------|----------|------------------------|
| AA-RESAPP.1 | **`ARCHITECTURE.md`** — multi-agent HTTP, env vars, graph | **Done** | High | |
| AA-RESAPP.2 | **Manifest `environment=`** + `host/environment_profile.py` parity with scaffold | **Done** | High | |
| AA-RESAPP.3 | **Remove dead flags** `RESEARCH_USE_LEGACY_*` if obsolete | **Done** | Medium | `host/settings.py` |
| AA-RESAPP.4 | **Deploy triad** verification | **Done** | High | |
| AA-RESAPP.5 | **`pyproject.toml` deps** section | **Done** | High | |
| AA-RESAPP.6 | **Smoke tests** + `test_research_manifest_wiring` green | **Deferred** | High | |

### AA — Conformance matrix (living status)

| Module | Scaffold-aligned | ARCHITECTURE.md | Deploy triad | pyproject doc | Tier hygiene |
|--------|------------------|-----------------|--------------|---------------|--------------|
| `agents/echo` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/lab` | N/A (mocks) | README only | N/A | N/A | OK |
| `agents/legal` | Yes (scaffold) | **Done** | N/A | N/A | **OK** |
| `agents/organization_worker` | Partial | **Done** | N/A | N/A | **OK** |
| `agents/problem_radar` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/research` | Yes | **Done** | N/A | N/A | **OK** |
| `agents/signoff_probe` | Yes | **Done** | N/A | N/A | OK |
| `applications/lab_application` | Yes | **Done** | **OK** | **Done** | H-APP factory |
| `applications/legal_application` | Yes | **Done** | **OK** | **Done** | H-APP |
| `applications/poc_template_application` | Yes | **Done** | **OK** | **Done** | OK |
| `applications/research_application` | Yes | **Done** | **OK** | **Done** | H-APP |

### AA — Residual backlog (infrastructure)

**Platform AA rows closed (2026-06-02).** Open infrastructure work: [§6.1z](#61z-harness-implementation-queue-consolidated) **V-REM** (2026-06-05) + ongoing **§6.1** maintenance.

| ID | Deliverable | Priority | Notes |
|----|-------------|----------|-------|
| AA-LABAG.1 | `agents/lab/README.md` — mock agents, not product Tier-2 | Low | **Done** — `agents/lab/README.md` |
| AA-LABAG.2 | (Optional) move lab mocks to `testing_support/` | Low | **Won't fix** until leadership requests |
| AA-SIG.2 | Scaffold parity diff test for `signoff_probe` | Low | **Done** — `tests/unit/scaffold/test_signoff_scaffold_parity.py` |
| AA-LABAPP.6 | Lab host smoke after H-APP factory | High | **Done** — unit + acceptance coverage |
| AA-LEG.0.2 | Git tag `legal-legacy-pre-aa` | High | **Done** — annotated tag on pre-reset commit |

### AA — Explicitly deferred (business / domain — Band 3)

| Topic | Task IDs | Reason |
|-------|----------|--------|
| Legal UAEP domain steps | AA-LEG.2.2–2.4, AA-LEGAPP.6, AA-LEGAPP.8 | Business logic on scaffold — [§6.3a](#63a-business-backlog-register-consolidated) |
| Research domain | AA-RES.4, AA-RES.5, AA-RESAPP.6 | Skills + graph tests — product prototype |
| Organization worker full scaffold | AA-ORG.3, AA-ORG.4 | Demo agent + lab roster |
| Lab host extra smoke | AA-LABAPP.6 | **Done** (2026-06-02 sync) — not blocking |
| K.1 / K.2 | Phase K | Band 3 — problem_radar / vendor discovery |
| Legal live LLM E2E | K.6 / B.15 / S-Ops.4 | Band 3 — CI budget |
| New product Tier-3 beyond four hosts | §6.3 | Product decision |

### AA — Paydown log

| Date | AA ID | Summary |
|------|-------|---------|
| 2026-06-03 | AA-0.1, AA-LEG.0.1 | Phase AA registered; **legal hard reset** policy recorded |
| 2026-06-03 | AA-LG.1 | LangGraph removed from core deps; CI `check_langgraph_not_required.py` |
| 2026-06-03 | AA-S0.1–S0.2, AA-S0.5, AA-APP.0.1–0.3, AA-ECHO.2, AA-PR.*, AA-LABAPP.2, AA-POC.2, AA-RESAPP.2, AA-LEG.1–1.3 | Tier hygiene, lab harness runtime, legal hard reset, deploy triad gate; gate **521** |
| 2026-06-02 | AA-S0.3, AA-D0.*, AA-* ARCHITECTURE, AA-LABAPP.3, AA-RESAPP.3 | `--reference` scaffold, docs matrix, lab manifest environment, tier import tests; gate **526** |
| 2026-06-02 | Plan sync | §4.0a scope split, DX/AA residual backlogs, §6.3a business register, master tables synced |
| 2026-06-02 | AA sync | AA-LABAPP.6 **Done**; AA-LABAG.2 **Won't fix**; §6.1z implementation queue |
| 2026-06-02 | AA-LEG.0.2, OPS-L3.1 | Tag `legal-legacy-pre-aa`; operational L3 evidence verified |

**Suggested session order (platform — complete):**  
See [§6.1z](#61z-harness-implementation-queue-consolidated). **Do not schedule** AA-LEG.2.* / AA-RES.5 / AA-ORG.3–4 in harness cadence — use [§6.3a](#63a-business-backlog-register-consolidated) after product decision.

---

---

## Phase W-OPS — Operational Harness Maturity (IDEAL L3 ops)

**Status:** **Done** (2026-06-06) — W-OPS.1–W-OPS.15 delivered including W-OPS.10 lab stack health probes; **operational L3** sign-off still requires `W_OPS_RELEASE_CYCLES>=2` (or `build/architecture_hardening/release_cycles.json`) via `phase_w_ops_evidence.py --enforce`.  
**Source:** Harness maturity audit (2026-06-02; conversation) · [IDEAL_HARNESS_AI_ARCHITECTURE.md](guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §12.3–§12.4 · [guides/HARNESS_ENVIRONMENT.md](guides/HARNESS_ENVIRONMENT.md)  
**Prerequisites:** Phases **V**, **P-Ext**, **W-ML**, §4.1 **Done**.  
**Goal:** Close the gap between **L3 CI evidence** (`maturity_gate_evidence`, relaxed thresholds) and **L3 operational** (IDEAL critical areas Policy/Reliability/Observability ≥ 3 with release evidence).  
**Out of scope:** K.1, K.2, new product Tier-3 apps, domain/product skills (Band 3 · §6.3).

**Audit verdict (harness-only):** Intergrax is **L2+ scalable harness** with strong Tier-0 catalogs and Nexus §42; default implementation queue is **§6.1 + §6.2w**, not product agents.

#### W-OPS — Deliverables

| # | Deliverable | Status | Priority | Location / acceptance |
|---|-------------|--------|----------|------------------------|
| W-OPS.0 | Plan traceability from maturity audit | **Done** | — | This phase + §6.2w + doc model row |
| W-OPS.1 | **Side-effect idempotency** — `IdempotentToolInvoker` + `idempotency_key` on `ToolExecutionRequest` | **Done** | **Critical** | `runtime/tools/idempotent_invoker.py`; gate `test_idempotent_invoker.py` |
| W-OPS.2 | **Integration circuit breaker** — `IntegrationCircuitBreaker` in `integrations/_shared/` | **Done** | **Critical** | `IntegrationDependencyError`; `test_integration_circuit_breaker.py` |
| W-OPS.3 | **Reliability gate tests** — long-running scheduler / checkpoint in gate | **Done** | High | `test_long_running_scheduler_j4.py` (`pytest -m gate`) |
| W-OPS.4 | **SLO catalog + incident budget** — harness SLIs + runbook stubs | **Done** | **Critical** | `guides/HARNESS_ENVIRONMENT.md` § Harness SLO catalog |
| W-OPS.5 | **L3-ops evidence artifact** — distinct from V-V6 CI gate | **Done** | **Critical** | `phase_w_ops_evidence.py`; `record_harness_release_cycle.py`; `release_cycles.json` |
| W-OPS.6 | **`tenant_id` on execution path** — required on `RuntimeRequest`; trace/events scoped | **Done** | High | `runtime/nexus/engine/runtime.py`; `RuntimeState.tenant_id` |
| W-OPS.7 | **Mandatory harness auth** — stage/prod/strict require `INTERGRAX_HARNESS_API_KEY` | **Done** | High | `LabApplicationSettings.requires_harness_api_key`; `test_lab_harness_api_key_required.py` |
| W-OPS.8 | **`harness.*` skill expansion** — `harness.reliability_smoke`, `harness.policy_smoke` | **Done** | Medium | `skills/providers/harness/manifests.py` |
| W-OPS.9 | **`requires_skills` adoption** — `harness.stack_demo` | **Done** | Medium | `test_harness_requires_skills_demo.py` |
| W-OPS.10 | **Harness lab stack health** — per-slug probes + circuit breaker | **Done** | Medium | `health_check_catalog_slugs`, `harness_lab_health.py`; `test_harness_lab_health.py` |
| W-OPS.11 | **Online evaluation path** — shadow observations → evaluation trends | **Done** | Medium | `online_evaluation_trend.py`, `export_harness_shadow_eval_trend.py`; file registry + AgentEngine hook |
| W-OPS.12 | **W-ML Celery scale-out (optional)** — env-driven via `wire_modality_extras` | **Done** | Low | `INTERGRAX_MODALITY_EXECUTION=celery`; documented in HARNESS_ENVIRONMENT |
| W-OPS.13 | **ToolsAgent removal roadmap** — CI blocks new imports; module frozen | **Done** | Low | `check_tools_agent_imports.py`, `check_tools_agent_run.py` |
| W-OPS.14 | **Typed Tier-3 wiring** — `load_callable` uses module namespace (no `getattr`) | **Done** | Low | `applications/_shared/wiring.py` |
| W-OPS.15 | **Architecture metrics enforcement (phased)** — tightened V-V6 thresholds | **Done** | Low | `maturity_gate_evidence.collect_harness_governance_signals` |

#### W-OPS — Execution waves (dependency order)

```text
Wave W-OPS-0 (governance):  W-OPS.0  — Done (audit → plan)
Wave W-OPS-P0 (critical):   W-OPS.1 → W-OPS.2 → W-OPS.3 → W-OPS.4 → W-OPS.5 → W-OPS.6 → W-OPS.7
Wave W-OPS-P1 (extend):     W-OPS.8 → W-OPS.9 → W-OPS.10 → W-OPS.11 → W-OPS.12 (optional)
Wave W-OPS-P2 (hygiene):    W-OPS.13 → W-OPS.14 → W-OPS.15
```

**IDEAL §12.3 gate:** Do not declare **operational L3** until W-OPS-P0 is **Done** and W-OPS.5 records **two consecutive release cycles** within SLO/incident budget (W-OPS.4).

**Delivery rule:** One **W-OPS.\*** ID per PR → update this table + paydown log → `pytest -m gate` + harness audit scripts (§6.1).

#### W-OPS — Paydown log

| Date | W-OPS ID | Summary |
|------|----------|---------|
| 2026-06-02 | W-OPS.0 | Maturity audit → Phase W-OPS + §6.2w execution order in implementation plan |
| 2026-06-06 | W-OPS.1–W-OPS.15 | Circuit breaker, idempotency gate, SLO docs, ops evidence script, staging API key, harness skills, online eval, wiring/metrics |
| 2026-06-02 | OPS-L3.1 | `phase_w_ops_evidence.py` Windows pytest argv + shadow trend probe; `--enforce` green |
| 2026-06-02 | REG / §6.1 | `doctor --ci` green: research `ToolEnablementProfile` protocol; lab factory via `bootstrap_lab_integration_wiring` |
| 2026-06-03 | W-OPS.10–W-OPS.11 | Lab stack health by catalog slug; shadow eval wired in `AgentEngine`; CI `phase_w_ops_evidence.py`; gate **470** |
| 2026-06-03 | W-OPS.5/11 | File-backed shadow eval registry; `record_harness_release_cycle.py`; extended ops evidence checks |
| 2026-06-03 | §6.1 / N.9 | Product scaffold `legal_product()` manifest + catalog bootstrap; gate **470** |
| 2026-06-03 | W-OPS.11 | Shadow eval trend export + `--verify-gate` on release cycle recorder |
| — | — | *(append row per merged PR)* |

---

---

### Phase A — Foundation Stabilization



| # | Deliverable | Status |

|---|-------------|--------|

| A.1 | Unified run lifecycle | **Done** |

| A.2 | Task trace persistence | **Done** |

| A.3 | NexusLoop production path | **Done** |

| A.4 | EvalRunner integration (NexusEvalRunner + gate coverage) | **Done** |

| A.4.1 | NexusEvalRunner integration tests + inclusion in gate | **Done** (2026-06-05 — `tests/integration/eval/test_nexus_eval_runner.py`) |

| A.5-min | Pre-P4.2 regression gate | **Done** |

| A.5 | Full regression suite (Legal E2E, all steps) | **Deferred** |

| A.6 | Shim cleanup | **Done** | Removed `applications/legal_agent/`; docs + duplicate `legal_application/tests/` cleaned |



**A.5-min completion criteria (gate before P4.2):**



```bash

uv run pytest tests/ -m gate -q

```



| Test area | File |

|-----------|------|

| TaskLifecycle transitions | `tests/unit/runtime/task/test_task_lifecycle.py` |

| TaskTraceEmitter + RuntimeEventBus | `tests/unit/runtime/task/test_task_trace_event_bus.py` |

| trace_bridge mapping | `tests/unit/runtime/events/test_trace_bridge.py` |

| AgentEngine.run / run_with_result | `tests/integration/agents/test_agent_engine_*.py` |

| NexusLoop + Echo (lifecycle + events) | `tests/integration/runtime/test_nexus_loop_echo.py` |

| GraphExecutor sequential stub | `tests/integration/runtime/test_graph_executor_stub.py` |



**Infrastructure fixes included:** circular import (`tool_runtime` ↔ `runtime_state`), missing `RegistryToolExecutor`, `ExecutionGraph` pydantic imports, lazy pipeline imports in `tests/conftest.py`.



**Explicitly not required before P4.2:** Legal through NexusLoop, full Nexus step matrix, E2E with real LLM.



---

---

### Phase D — Observability and Experiments



**Goal:** §19, §35 — laboratory tooling (not SaaS UI).



| # | Deliverable | Status | Notes |

|---|-------------|--------|-------|

| D.0 | §42 P4.1 Event Bus wiring | **Done** | `RuntimeEventBus`, `trace_bridge`, NexusLoop |

| D.1 | Debug CLI | **Done** | `python -m intergrax.debug tasks list\|show\|trace` |

| D.2 | Minimal debug API | **Done** | FastAPI `GET /debug/tasks` on trace store |

| D.3 | Experiment registry | **Done** | SQLite registry; CLI + `GET/POST /debug/experiments` |

| D.4 | Experiment workflow API | **Done** | `intergrax/experiments/workflow.py`, `tests/unit/experiments/`; platform `notebooks/` removed (2026-06-12) |

| D.5 | Cost in trace | **Done** | `AgentExecutionResult.cost` from LLM usage / runtime stats |



---

---

### Phase E — Legal Agent Refactoring (parallel)



| # | Deliverable | Status |

|---|-------------|--------|

| E.1 | Thin sequential Legal — domain steps as UAEP `AgentStep` list | **Done** |

| E.2 | ToolRuntime via gateway (no direct Nexus step imports in bridge) | **Done** (P4.4) |

| E.3 | Governance on UAEP decision path | **Done** (P4.3) |

| E.4 | Thin dynamic Legal (`LegalDynamicPipeline` routing) | **Done** |



**E.4 delivered (2026-05-27):** `agents/legal/uaep/dynamic_steps.py` — 5 UAEP macro-steps (setup → tool plan → route → waves → finalize); `legal_execution_loop` phase functions extracted. Gate: 34 tests.



**E.1 delivered (2026-05-27):** `agents/legal/uaep/thin_steps.py` — 8 UAEP steps (setup → finalize); `LegalAnalysisPipeline` reuses same runners; dynamic mode keeps single pipeline boundary. Gate: 33 tests.



---

---

## Appendix A


---

## Appendix A — Business agents readiness checklist

Gate before Problem Radar / Vendor Discovery. Run:

```bash
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest tests/ -m gate -q
```

### Agent creation & registration

| # | Question | Status |
|---|----------|--------|
| 1 | Scaffold in minutes (`intergrax.scaffold new-agent`)? | ✅ |
| 2 | UAEP structure generated (contract, steps, tests)? | ✅ |
| 3 | First run in < 1 hour? | ✅ |
| 4 | Register via `AgentRegistry` only (no Nexus edits)? | ✅ |
| 5 | Capabilities in contract? | ✅ |

### Execution & observability

| # | Question | Status |
|---|----------|--------|
| 6 | Runs through NexusLoop / lab `/v1/lab/run`? | ✅ |
| 7 | UnifiedTaskRunner same path as HTTP? | ✅ |
| 8 | Graph sequential + parallel? | ✅ |
| 9 | Trace via `/debug/tasks/{id}`? | ✅ |
| 10 | Runtime events + checkpoints + progress? | ✅ |

### Recovery, HITL, memory, isolation

| # | Question | Status |
|---|----------|--------|
| 11 | Nexus validates output? | ✅ |
| 12 | Retry / alternate agent on validation failure? | ✅ |
| 13 | HITL pause + resume? | ✅ |
| 14 | Checkpoint recovery? | ✅ |
| 15 | Shared context in graphs? | ✅ |
| 16 | Sandbox + shadow workspace? | ✅ |

### Tooling & composition

| # | Question | Status |
|---|----------|--------|
| 17 | Canonical agent guide exists? | ✅ |
| 18 | Lab application (Tier-3)? | ✅ |
| 19 | Same agent reusable across applications? | ✅ |
| 20 | Applications contain wiring only? | ✅ |

### Go / no-go

| Criterion | Threshold | Current |
|-----------|-----------|---------|
| Checklist | ≥ 90% | **20/20** |
| Acceptance suite | 10/10 green | ✅ |
| Sign-off exercise | 1 new agent, < 1h, zero runtime edits | **Done** (`signoff_probe`) |

**Verdict:** **L1 Agent Operating System certified** (technical). **Phase S** (harness environment GA) is next; **K.1/K.2** wait until S is **Done**.

### Sign-off record

```text
Date:           2026-05-27
Agent exercise: signoff_probe
Capability:     signoff.probe
Time to first run: ~15 min (scaffold + smoke test)
Runtime files modified: none (only agents/signoff_probe/ added)
Smoke test:     agents/signoff_probe/tests — 1 passed
HTTP proof:     lab_application wiring + POST /v1/lab/run
Trace proof:    GET /debug/tasks/{id}, /trace?include_runtime=true, /events
                (test_lab_application_runs_signoff_probe_with_trace)
Acceptance suite: pass (tests/acceptance/agent_os)
Gate suite:     pass (228+ tests)
Trace:          NexusLoop smoke + HTTP debug API (SQLite trace store in lab factory)
Decision:       L1 certified — GO Phase S (harness environment), then Phase K (K.1/K.2)
```

---

---

## Appendix I


---

## Appendix I — Plugin catalog traceability (Phase P-Ext)

**Purpose:** Task-level tracker for plugin-native Integration, Tool, and Skill catalogs. **Canonical phase narrative:** [Phase P-Ext](#phase-p-ext--plugin-catalogs-integrations-tools-skills) · paydown: [P-Ext.6](#p-ext6--production-closure-paydown).

**Status:** **Done** (2026-06-02) · **MVP effort:** ~21–32 person-days · **paydown estimate:** ~8–14 person-days.

### I.1 Delivery rule

Same as §6.1: one **P-Ext.\*** ID → PR → update status in this appendix → `pytest -m gate` green. Paydown cadence: [§6.1p](#61p-phase-p-ext-paydown-band-2c--optional-parallel-with-61).

### I.2 Task register

| ID | Layer | Summary | Status | Priority |
|----|-------|---------|--------|----------|
| P-Ext.0.1 | All | `load_plugins()` / entry point discovery | **Done** | P0 |
| P-Ext.0.2 | All | `PluginConflictError`, `PluginLoadError` | **Done** | P0 |
| P-Ext.0.3 | All | `bootstrap_catalogs()` Tier-3 API | **Done** | P0 |
| P-Ext.0.4 | All | `guides/EXTENSION_AUTHOR_GUIDE.md` (EN) | **Done** | P0 |
| P-Ext.0.5 | All | Test fixture pip package | **Done** | P0 |
| P-Ext.0.6 | All | EP discovery tests (3 groups) | **Done** | P0 |
| P-Ext.0.7 | All | `INTERGRAX_DISCOVER_PLUGINS` + lab wiring | **Done** | P1 |
| P-Ext.1.1 | Integrations | Entry points `intergrax.integrations` | **Done** | P0 |
| P-Ext.1.2 | Integrations | `bootstrap_core` / optional split | **Done** | P1 |
| P-Ext.1.3 | Integrations | Typed `resolve_*` helpers (top categories) | **Done** | P2 |
| P-Ext.1.3a | Integrations | Expand `resolve_typed` + tests | **Done** | P2 |
| P-Ext.1.4 | Integrations | Health check API (optional) | **Done** | P3 |
| P-Ext.1.5 | Integrations | `IntegrationSlug` cleanup (docs/scripts) | **Done** | P2 |
| P-Ext.1.6 | Integrations | EP test via fixture | **Done** | P0 |
| P-Ext.1.7 | Integrations | Dual-model docs (manifest vs plugin) | **Done** | P2 |
| P-Ext.1.8 | Integrations | CI integration slug count smoke | **Done** | P1 |
| P-Ext.1.9 | Integrations | `test_resolve_typed.py` | **Done** | P3 |
| P-Ext.1.10 | Integrations | Tier-3 `bootstrap_catalogs` in integration_wiring | **Done** | P0 |
| P-Ext.1.11 | Integrations | `_shared/integration_wiring.py` helper | **Done** | P2 |
| P-Ext.1.12 | Integrations | `SqliteIntegrationPlugin` wire or document | **Done** | P3 |
| P-Ext.2.1 | Tools | `ToolPlugin` Protocol | **Done** | P0 |
| P-Ext.2.2 | Tools | `ToolBundleManifest` / bundle metadata | **Done** | P0 |
| P-Ext.2.3 | Tools | `register_tool_plugin()` | **Done** | P0 |
| P-Ext.2.4 | Tools | RAG bundle plugin migration (pilot) | **Done** | P1 |
| P-Ext.2.5 | Tools | Entry points `intergrax.tools` | **Done** | P1 |
| P-Ext.2.6 | Tools | MCP tool export | **Done** | P1 |
| P-Ext.2.7 | Tools | `ToolContract.version` | **Done** | P2 |
| P-Ext.2.8 | Tools | All 13 shipped bundles → `ToolPlugin` | **Done** | P1 |
| P-Ext.2.9 | Tools | `tools/examples/` reference package | **Done** | P0 |
| P-Ext.2.10 | Tools | `test_external_tool_plugin.py` | **Done** | P0 |
| P-Ext.2.11 | Tools | EP tool test via fixture | **Done** | P0 |
| P-Ext.2.12 | Tools | `tool_wiring` lazy `tool_bundle_ids` | **Done** | P2 |
| P-Ext.3.1 | Skills | `SkillPlugin` Protocol | **Done** | P1 |
| P-Ext.3.2 | Skills | `register_skill_plugin()` | **Done** | P1 |
| P-Ext.3.3 | Skills | Entry points `intergrax.skills` | **Done** | P1 |
| P-Ext.3.4 | Skills | harness + research + legal plugin migration | **Done** | P1 |
| P-Ext.3.5 | Skills | `requires_skills` (optional) | **Done** | P3 |
| P-Ext.3.6 | Skills | `skills/examples/` reference package | **Done** | P0 |
| P-Ext.3.7 | Skills | `test_external_skill_plugin.py` | **Done** | P0 |
| P-Ext.3.8 | Skills | EP skill test via fixture | **Done** | P0 |
| P-Ext.3.9 | Skills | `skill_wiring` lazy `skill_bundle_ids` | **Done** | P2 |
| P-Ext.3.10 | Skills | Scaffold `new-skill` → `SkillPlugin` | **Done** | P2 |
| P-Ext.3.11 | Skills | Docs: SkillPlugin vs Cursor importer | **Done** | P2 |
| P-Ext.3.12 | Skills | Shipped `requires_skills` demo (optional) | **Done** | P3 |
| P-Ext.4.1 | Ops | Lazy profile bootstrap | **Done** | P2 |
| P-Ext.4.2 | Ops | `CatalogSnapshot` API | **Done** | P2 |
| P-Ext.4.3 | Ops | Slug conflict policy (bootstrap) | **Done** | P2 |
| P-Ext.4.4 | Ops | `check_plugin_catalog.py` CI | **Done** | P1 |
| P-Ext.4.5 | Ops | CI smoke: tool/skill bundle counts | **Done** | P1 |
| P-Ext.5.1 | Docs | Scaffold `new_*` commands | **Done** | P2 |
| P-Ext.5.2 | Docs | INTEGRATIONS/TOOLS/SKILLS external sections | **Done** | P2 |
| P-Ext.5.3 | Docs | Canon §7.1.5.1 plugin narrative | **Done** | P1 |
| P-Ext.5.4 | Docs | remove `PLUGIN_CATALOG_PLAN.md` | **Done** | P3 |
| P-Ext.5.5 | Docs | Prod path matrix in author guide | **Done** | P2 |
| P-Ext.5.6 | Docs | Lab wiring recipe for external plugins | **Done** | P2 |
| P-Ext.6.1 | Paydown | Fixture pip package (rollup) | **Done** | P0 |
| P-Ext.6.2 | Paydown | External tool + skill examples + tests | **Done** | P0 |
| P-Ext.6.3 | Paydown | EP discovery + lab env | **Done** | P1 |
| P-Ext.6.4 | Paydown | IntegrationSlug cleanup | **Done** | P2 |
| P-Ext.6.5 | Paydown | Scaffold CLI | **Done** | P2 |
| P-Ext.6.6 | Paydown | Integration Tier-3 + typed resolve + health | **Done** | P2 |
| P-Ext.6.7 | Paydown | Conflict policy + CI smoke | **Done** | P1 |
| P-Ext.6.8 | Paydown | Skill Tier-3 + scaffold rollup | **Done** | P2 |
| P-Ext.6.9 | Paydown | Tool Tier-3 lazy wiring rollup | **Done** | P2 |
| P-Ext.6.10 | Paydown | Tier-3 lazy wiring (all catalogs) rollup | **Done** | P2 |

**Paydown summary:** 0 **Planned** · 61 **Done** · 0 **Partial** (Phase P-Ext production closure complete; rollup rows duplicate leaf IDs).

### I.3 Market alignment checklist

| Pattern | Target |
|---------|--------|
| Hexagonal adapters | `IntegrationCategory` + contracts + `IntegrationPlugin` |
| MCP tools | `ToolContract` + `export_mcp_tools` |
| Capability packs | `SkillManifest` + resolver (not LLM-invokable) |
| 12-factor config | env_prefix + `IntegrationProfile.options` |
| Plugin discovery | entry points (hybrid with explicit bootstrap) |
| Tier-3 composition root | `bootstrap_catalogs()` |

### I.4 Paydown log

| Date | P-Ext ID | Summary |
|------|----------|---------|
| 2026-06-02 | — | Phase P-Ext + Appendix I added (migrated from `PLUGIN_CATALOG_PLAN.md`) |
| 2026-06-02 | 0.1–0.4, 1.1–1.2, 2.1–2.8, 3.1–3.5, 4.1–4.2, 4.4, 5.2–5.4 | MVP: protocols, bootstrap, 13 tool + 3 skill plugins, lazy catalog, `custom_memory_kv` test |
| 2026-06-02 | — | Plan updated: **MVP Done** + **P-Ext.6 paydown** backlog (EP fixture, external tool/skill tests, ops/docs) |
| 2026-06-02 | 1.* audit | Integrations audit: 12 core / ~99 full manifest path; `resolve_typed` partial; Tier-3 integration_wiring gap; +P-Ext.1.3a, 1.8–1.12 |
| 2026-06-02 | M.6 P5 closeout | Catalog **135** full (`12` core); timeline 99→127→135; P-Ext integration counts synced |
| 2026-06-02 | 3.* audit | Skills audit: 3/3 `SkillPlugin`, 8 skill_id; Tier-3 `skill_wiring` OK; scaffold legacy; +P-Ext.3.9–3.12, 6.8 |
| 2026-06-02 | 2.* audit | Tools audit section + `tool_wiring` lazy (P-Ext.2.12); P-Ext.4.5 unified counts; +P-Ext.6.9–6.10 |
| 2026-06-02 | P-Ext paydown | Fixture EP package, external examples/tests, Tier-3 wiring, docs, CI smoke (residual: 1.5, 4.3, 5.1, 5.6) |
| 2026-06-02 | P-Ext closure | IntegrationSlug docs cleanup, `warn_override` conflict policy, scaffold CLI, lab wiring recipe |
| 2026-06-02 | P-Ext complete | Phase narrative + §6.1p synced; expanded `check_plugin_catalog.py` smoke suite |
| 2026-06-02 | §6.1 | Gate green **486**: IntegrationBinding test fixes, circular import, catalog re-bootstrap after test clears, scaffold templates |
| 2026-06-02 | TYP-06, U-Typ.4 | `IntegrationProfile` explicit binding accessors; removed `tools_agent.AgentDecision` alias |
| 2026-06-02 | W-OPS.0 | Harness maturity audit → Phase W-OPS + §6.2w in implementation plan |
| 2026-06-05 | V-REM.0.* | Plan audit → Phase V-REM + Appendix J + §6.1z queue (10 open) |
| — | — | *(append row per merged PR)* |

---

## Phase DX-LC — Full Harness Layer Completion closeout (2026-06-17)

**Status:** **Done** (2026-06-17) — re-validates DX 47/47 + W-OPS + AUDIT-IDEAL-26/27; no open P0/P1  
**Goal:** Formal Full Harness LC closeout — gate verification, journal  
**ADR:** **No ADR needed**

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| DX-LC-S1 | **Re-audit** — DX/W-OPS register | **Done** | High | No P0/P1 |
| DX-LC-S2 | **Plan/architecture sync** — Full Harness LC note | **Done** | High | Domain pair consistent |
| DX-LC-S3 | **Gate verification** | **Done** | High | DX CI scripts (trace explorer, simulator, chaos) |
| DX-LC-S4 | **Journal + progress tracker** | **Done** | High | `layer_completion_progress.json` mature |

**Deferred P2–P4:** GOV-PROD.1 dashboard · polished SaaS UI non-goal · AUDIT-IDEAL-6.7 doctor hook (LLM)

### 6.1av Harness implementation queue — Developer experience audit maintenance (planned)

**Source:** Layer 21 audit (2026-06-18) — `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · [`guides/audit/results/2026-06-18/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](guides/audit/results/2026-06-18/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Priority ladder:** **Band 1** (§6.1) — DX bundle + cross-refs; **one ID per PR**

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **DX-MAINT-01** | Cross-ref | P2 | **Planned** | AUDIT-IDEAL-6.7 doctor hook — cross-ref [`LLM-MAINT-01`](LLM_ADAPTERS.md#61av-harness-implementation-queue--llm-adapters-audit-maintenance-planned) | LLM owns implementation |
| 2 | **DX-MAINT-02** | DX | P3 | **Planned** | Expand `intergrax doctor check` — trace explorer + replay + simulator wiring subset | `intergrax doctor --ci` runs DX gates |
| 3 | **DX-MAINT-03** | Backlog | P3 | **Planned** | GOV-PROD.1 dashboard — register row with scope boundary | Not blocking L3 DX |
| 4 | **DX-MAINT-04** | Docs | P4 | **Planned** | Polished SaaS UI — explicit non-goal note in DX canon | Prevents scope creep |

**Suggested PR order:** DX-MAINT-02 → DX-MAINT-01 → DX-MAINT-03 → DX-MAINT-04.

**Note:** AUDIT-IDEAL-27.2 replay wiring **Done** (2026-06-18 revalidation).

---

*End of Experimentation and Developer Experience Implementation Plan.*
