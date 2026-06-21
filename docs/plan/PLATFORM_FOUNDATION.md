# Platform Foundation — Implementation Plan

**Architecture (1:1):** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (PLATFORM_FOUNDATION plan).

- **Implement / audit default:** §6.1 gate maintenance (default) · §6.3 deferred product only · §4.0a scope split. **On demand:** [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) · [`plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md) (re-validate closed only) · [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) (Band 2ae HEP only)
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/PLATFORM_FOUNDATION.md`](../guides/audit_slices/PLATFORM_FOUNDATION.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Not default:** K.1, K.2, Legal UAEP domain steps, new product Tier-3 apps — **[§6.3](#63-end-of-plan--deferred-product-work-only)** · **[§6.3a](#63a-business-backlog-register-consolidated)** · **[§4.0a](#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](#61z-harness-implementation-queue-consolidated) / [§6.1aa](#61aa-harness-implementation-queue-memory-platform).

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Band 2ay M.12** = **Done** — see **[§6.1an](#61an-harness-implementation-queue--llm-guardrail-integrations-closed)**. **Optional harness extension (after gate green):** **[Band 2ae Phase HEP](#61aw-phase-hep--harness-evidence-pack-band-2ae)** — runtime evidence packaging, not §6.3. **Next product work** = [§6.3](#63-end-of-plan--deferred-product-work-only) (product prioritization only).

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/check_harness_no_getattr.py
  python scripts/check_legacy_modules_removed.py
  python scripts/check_agent_skill_resolution.py
  python scripts/check_harness_registry_resolution.py
  python scripts/check_harness_capability_graph_wiring.py
  python scripts/check_legacy_tool_plan_booleans.py
  python scripts/check_trace_bridge_event_catalog.py
  python scripts/check_plugin_catalog.py
  python scripts/check_llm_adapter_typed_returns.py
  python scripts/check_agents_llm_adapter_response.py
  uv run python scripts/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/check_scaffold_harness_alignment.py
  python scripts/check_agents_no_tier3_imports.py
  python scripts/check_intergrax_no_applications_imports.py
  uv run python scripts/check_harness_prompt_golden_catalog.py
  uv run python scripts/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/phase_v_capability_graph_guard.py --enforce
  python scripts/check_agents_no_inline_prompts.py
  python scripts/check_agents_no_vendor_sdk_imports.py
  uv run python scripts/check_ideal_harness_l3_gates.py
  uv run python scripts/harness_maturity_report.py --enforce-l3-critical
```

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3.

### 6.1av Harness implementation queue — Platform Foundation audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `PLATFORM_FOUNDATION` layers 1, 2, 32 · [`../audit_results/2026-06-19/PLATFORM_FOUNDATION.md`](../audit_results/2026-06-19/PLATFORM_FOUNDATION.md) · prior: [`../audit_results/2026-06-18/PLATFORM_FOUNDATION.md`](../audit_results/2026-06-18/PLATFORM_FOUNDATION.md)  
**Priority ladder:** **Band 1** (§6.1) — doc hygiene + optional legacy cleanup; runs **in parallel** with gate maintenance

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **PF-MAINT-DOC-01** | Docs | P2 | **Done** | Remove stale M.6 P6 from audit prompt known-gaps; sync audit result file | Audit prompt + result match plan §6.1y (**Done** 32/32) |
| 2 | **PF-MAINT-DOC-02** | Docs | P2 | **Done** | Sync §6.1au + §4.0 Band 2az counter with `AUDIT_IDEAL_2026.md` | Plan shows **90/90 Done** · **0 Planned** |
| 3 | **PF-MAINT-DX-01** | Docs | P3 | **Done** | Implementer quick-start in `intergrax_runtime_architecture.md` hub | §4.0 ladder + scaffold flow linked |
| 4 | **PF-MAINT-LEG-01** | Code | P3 | **Done** | Remove `use_rag`/`use_websearch` from LLM planner schema (`EnginePlan`) | `check_legacy_tool_plan_booleans.py` green; `tool_ids` only |
| 5 | **PF-MAINT-DOC-03** | Docs | P3 | **Done** | Sync §0.5 regression gate counter with live `pytest -m gate` snapshot | Plan §0.5 shows **1498 passed** (2026-06-19) |
| 6 | **PF-MAINT-LEG-02** | Code | P3 | **Done** | Remove legacy `use_rag`/`use_websearch` shims from `ToolInvocationPlan` (`tool_runtime.py`) | Zero DeprecationWarning in gate; `tool_ids` only at runtime bridge |
| 7 | **PF-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19/` | `PLATFORM_FOUNDATION.md` + `progress.json` present |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** Phase K, §50 marketplace, new Tier-0 mechanisms — [§6.3](#63-end-of-plan--deferred-product-work-only).

### 6.1aw Phase HEP — Harness Evidence Pack (Band 2ae)

**Status:** HEP-1 **Done**; HEP-2 Trace Evidence Path **Done**; HEP-3 Evidence Posture / Scoreboard **Done**; EVID-CORE-FU-01 Selected Live Tier-0 Probes **Done** — `certify core` → `trace export` → `evidence live-core` → `evidence posture` / `evidence posture export`. EVID-CORE-FU-01 adds selected local no-network live Tier-0 probes with mock LLM/tools. It does not replace deterministic CORE certification and is not full runtime certification.  
EVID-EVAL Eval Regression Evidence **Done** — `evidence eval` writes deterministic eval evidence artifacts and optionally enriches `evidence posture` via `EVAL_REGRESSION` when the report exists. It is not a new eval framework and does not run real LLM/provider evaluation.  
**Priority ladder:** **Band 2ae** — §6.1 extension (harness evidence / runtime proof / onboarding); runs **after** gate green; **not** §6.3 product work  
**Source:** External infrastructure audit (2026-06) + operator decision B → A → C

| Wave | Scope | IDs | Status |
|------|-------|-----|--------|
| HEP-0 | Mapping & contracts | EVID-CORE-01 … EVID-CORE-03 | **Done** (2026-06-21 C1) |
| HEP-1 | Core evidence path (`intergrax certify core`) | EVID-CORE-04 … EVID-CORE-06 | **Done** (2026-06-21 C2–C3) |
| HEP-2 | Trace evidence path (`intergrax trace show` · `intergrax trace export`) | EVID-TRACE-01 … EVID-TRACE-04 | **Done** (2026-06-21 C4–C6) |
| HEP-3 | Evidence posture / scoreboard (`intergrax evidence posture`) | EVID-POSTURE-01 … EVID-POSTURE-04 | **Done** (2026-06-21 C10) |

**Explicitly excluded:** §6.3 product apps/agents · W-ADAPT L4 replacement · duplicate doctor CI gates · Band 2ad (M.7 P7 — **Done**).

**Suggested PR order:** see [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) § Implementation waves.

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

Full task register: [Appendix I](plan/satellites/PLATFORM_FOUNDATION_appendices.md).

**Out of scope for §6.1:** K.1, K.2, new `applications/<product>/`, Problem Radar wave 2+, Legal live LLM E2E — see §6.3. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](#61x-harness-implementation-queue--integration-depth-m6-p5-done)

```text
Wave H-INT-0 (categories):  M-P5-CAT.1 → M-P5-CAT.2 → M-P5-CAT.3
Wave H-INT-6 (ops/CI):      M-P5.1 → M-P5.2 → M-P5.3 → M-P5.4 → M-P5.5 → M-P5.6 → M-P5.7 → M-P5.8 → M-P5.9 → M-P5.10
Wave H-INT-7 (eval/async):  M-P5.11 → M-P5.12 → M-P5.13 → M-P5.14 → M-P5.15 → M-P5.16 → M-P5.17 → M-P5.18 → M-P5.19 → M-P5.20
Wave H-INT-8 (data lab):    M-P5.21 → M-P5.22 → M-P5.23 → M-P5.24 → M-P5.25 → M-P5.26 → M-P5.27 → M-P5.28
Wave H-INT-9 (P2 reserve):  M-P5.29 → M-P5.30 → M-P5.31 → M-P5.32 → M-P5.33 → M-P5.34
Wave PRE (presets):         M-P5-PRE.1  (after H-INT-6 P0 slugs wired)
```

**Prerequisites:** Phase M.6 P4 **Done**; M-P4.FU wiring **Done**; Phase INT closeout **Done** (health probe patterns).  
**Parallelism:** H-INT-6 unblocks W-OPS metrics + multi-CI; H-INT-7 unblocks EVAL/W-ADAPT; H-INT-8 is lab-only.  
**Closeout target:** catalog **136** slugs; `HARNESS_M6_P5_PROBE_SLUGS` + four Tier-3 presets; gate green.

### 6.3 End of plan — deferred product work only (Band 3)

**This section is the last band in the implementation plan.** Nothing here is the default “next step” after harness work.

| ID | Deliverable | Status | Gate to start |
|----|-------------|--------|----------------|
| K.1 | Problem Radar prototype | **Deferred** | Explicit product decision + [Appendix A](plan/PLATFORM_FOUNDATION.md) |
| K.2 | Vendor Discovery prototype | **Deferred** | Same as K.1 |
| K.6 / B.15 / S-Ops.4 | Legal live LLM E2E | **Deferred** | Product/CI budget decision |
| `agents/legal` UAEP domain steps | Scaffold shell **Done** (Band 2g); step port **Deferred** | **Business** | [§6.3a](#63a-business-backlog-register-consolidated) AA-LEG.2.2+ |
| Tier-3 product apps | New `applications/<product>/` beyond lab + reference hosts | **Deferred** | Product decision only — confirmed 2026-06-09; scaffold exists (Phase N **Done**) |
| Domain skills | Product agent skill packs (non-`harness.*`) | **Deferred** | With K.1 or K.2 |
| `agents/problem_radar/` | Wave 1 scaffold frozen | **Deferred** | Do not extend until K.1 reprioritized |

**When Band 3 may start:** Record the decision in this plan (date + chosen K.1 vs K.2), then follow [guides/AGENT_CREATION_GUIDE.md](guides/AGENT_CREATION_GUIDE.md). Tier-3 scaffold reference (Phase N) applies **only after** that decision — not as ongoing harness work.

**Tier-3 scaffold (for when Band 3 is approved):**

```bash
python -m intergrax.scaffold new-stack <slug> --profile lab --capability <slug>.basic
```

See [`applications/TIER3_READINESS.md`](../applications/TIER3_READINESS.md). Existing hosts (`lab_application`, `legal_application`, `research_application`, `poc_template_application`) are sufficient for **all harness** work. **Product:** [`local_workspace_application`](../applications/local_workspace_application/) — Local Knowledge Workspace (LKW) — first business environment after harness GA; see [ARCHITECTURE.md](../applications/local_workspace_application/ARCHITECTURE.md).

### 6.3a Business backlog register (consolidated)

**Single register for Band 3 and AA domain-deferred rows.** Do not duplicate in harness session summaries.

| ID | Deliverable | Module | Priority | Depends on |
|----|-------------|--------|----------|------------|
| **LKW.0** | Local Knowledge Workspace — scaffold + architecture baseline | `agents/local_{indexer,search,synthesizer}/`, `applications/local_workspace_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **LKW.1** | Wave 1 — ingest + search smoke on explicit paths | `agents/local_*/steps/` | **High** | LKW.0 |
| **LKW.2** | Multi-agent pipeline (`local.workspace.pipeline` graph) | `local_workspace_application/` + Nexus graph | High | LKW.1 |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | `intergrax/tools/providers/filesystem/` | Medium | LKW.1 |
| **LKW.4** | Background ingest queue + incremental index | Tier-0 queue + Tier-3 worker | Medium | LKW.2 |
| **LKW.5** | `LKW_DATA_HOME` + Chroma persistent local index | `local_workspace_application/host/settings.py` | High | LKW.1 |
| **LKW.6** | Local OS daemon (Win/Linux/macOS) + interaction intake on host | `local_workspace_application/` | High | LKW.1 |
| **LKW.6b** | Slack Socket Mode + slash command → Nexus (interaction surface) | Tier-3 + `slack` integration | Medium | LKW.6 |
| **LKW.7** | Background file watcher + incremental index + optional Slack notify | Tier-0 queue + Tier-3 worker | Medium | LKW.3 |
| **LKW.8** | Tray / file-picker UI (localhost HTTP/MCP client) | Product (out of harness) | Low | LKW.6 |
| **DSW.0** | Dispute Simulation Workspace — scaffold + architecture baseline | `agents/dispute_{intake,analyst,strategist,scenario}/`, `applications/dispute_sim_application/` | **High** | Product reprioritization (2026-06-07) — **Done** |
| **DSW.1** | Wave 1 — case intake + RAG ingest + timeline artifact | `agents/dispute_intake/steps/` | **High** | DSW.0 |
| **DSW.2** | Multi-agent pipeline (`dispute.pipeline` graph) | `dispute_sim_application/` + Nexus graph | High | DSW.1 · **Harness:** CFG-06 proven in `test_orchestration_cfg_simulation.py`; product wiring §6.3 |
| **DSW.3** | Analyst matrix + strategist brief domain steps | `agents/dispute_analyst/`, `agents/dispute_strategist/` | High | DSW.1 |
| **DSW.4** | Scenario variants + correspondence review + HITL | `agents/dispute_scenario/` | High | DSW.3 |
| **DSW.5** | Optional subgraph to `legal.review` for clause drill-down | Nexus graph | Medium | DSW.3 |
| **DSW.6** | Case persistence + retention policy | `dispute_sim_application/host/settings.py` | Medium | DSW.1 |
| **DSW.7** | Polish dispute eval fixtures + regression | `tests/` / agent eval | Medium | DSW.4 |
| **K.1** | Problem Radar prototype (wave 2+) | `agents/problem_radar/` | Product | Explicit reprioritization |
| **K.2** | Vendor Discovery prototype | (greenfield) | Product | K.1 decision or parallel product call |
| **AA-LEG.2.2** | Legal UAEP steps (one step per PR from `SPEC_FROM_LEGACY.md`) | `agents/legal/steps/` | High | Product/legal owner |
| **AA-LEG.2.3** | Remove any parallel legal runtime (Nexus gateway only) | `agents/legal/` | High | AA-LEG.2.2 |
| **AA-LEG.2.4** | Legal agent tests per ported step | `agents/legal/tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.6** | `legal_application` host smoke on real steps | `legal_tests/` | High | AA-LEG.2.2 |
| **AA-LEGAPP.8** | Consolidate duplicate legal test trees | `legal_tests/` vs agent tests | Low | AA-LEG.2.4 |
| **AA-RES.4** | Research skill ids on contracts | `agents/research/` | Medium | Product |
| **AA-RES.5** | Research UAEP + graph delegation tests | `agents/research/tests/` | High | Product |
| **AA-RESAPP.6** | Research application smoke + manifest wiring | `research_application_tests/` | High | AA-RES.5 |
| **AA-ORG.3** | Organization worker scaffold-align (`contract`, `steps/`) | `agents/organization_worker/` | Medium | Harness demo |
| **AA-ORG.4** | Lab manifest flag + integration test | `lab_application/manifest.py` | Medium | AA-ORG.3 |
| ~~AA-LABAPP.6~~ | ~~Extra lab host smoke~~ | — | — | **Done** (2026-06-02) — not in business queue |
| **K.6 / B.15 / S-Ops.4** | Legal full E2E with live LLM | CI / acceptance | Low | CI budget approval |
| **Tier-3 product** | New `applications/<product>/` beyond four reference hosts | `applications/` | Product | Phase N scaffold + §6.3 decision |
| **Domain skills** | Non-`harness.*` skill packs for product agents | `intergrax/skills/providers/` | Product | With K.1 or K.2 |
| **A.5** | Full Legal regression (all steps, live model) | Phase A row | Low | K.6 / B.15 |
| **Phase E** | Legal agent refactoring (parallel track) | `agents/legal/` | On demand | Product architecture |

**Not business (infrastructure — closed; see [§6.1z](#61z-harness-implementation-queue-consolidated)):** DX-5.7, AA-LEG.0.2, OPS-L3.1 **Done**; ongoing **§6.1** maintenance only.
