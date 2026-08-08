# Platform Foundation — Implementation Plan

**Architecture (1:1):** [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md)
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)
**Architecture governance:** [`architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md`](../../architecture/INTERGRAX_ARCHITECTURE_PRINCIPLES.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (PLATFORM_FOUNDATION plan).

- **Implement / audit default:** §6.1 gate maintenance (default) · §4.0a scope split. **On demand:** [`plan/satellites/PLATFORM_FOUNDATION_master_registers.md`](plan/satellites/PLATFORM_FOUNDATION_master_registers.md) · [`plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/satellites/PLATFORM_FOUNDATION_06_closed_queues.md) (re-validate closed only) · [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) (Band 2ae HEP only)
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/PLATFORM_FOUNDATION.md`](../../architecture/PLATFORM_FOUNDATION.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/PLATFORM_FOUNDATION.md`](../../technical/guides/audit_slices/PLATFORM_FOUNDATION.md).
- **Satellites:** at most **one** `plan/satellites` file per session unless RESUME cites more.

---

## 6. What to implement next

**Default answer (infrastructure):** **[§6.1](.#61-harness-platform-maintenance-default--band-1)** gate green on every PR — CRIT-V and OBS-BUS platform closeouts **Done**.

**Maintenance-only mode:** If CRIT-V paused by explicit decision, revert to §6.1 gate-only maintenance.

**Out of scope:** product/application implementation work and business backlogs — **[§6.3](.#63-end-of-platform-plan)** · **[§4.0a](.#40a-implementation-scope-split-infrastructure-vs-business)**.

**Audit basis:** Governance audit (2026-06-05) → GOV-AUDIT **Done**; orchestration audit (2026-06-05) → Phase ORCH + §6.1b; tools/skills audit (2026-06-02) → Phase TS + §6.1c; integration/RAG audit (2026-06-02) → Phase INT + RAG + §6.1d/§6.1e; context engineering audit (2026-06-02) → Phase CTX + §6.1f; prior V-REM/MEM/DX/AA closeouts in [§6.1z](.#61z-harness-implementation-queue-consolidated) / [§6.1aa](.#61aa-harness-implementation-queue-memory-platform).

### 6.1 Harness platform maintenance (default — Band 1)

§4.1 backlog is **closed**. Ongoing work = keep the harness green; **Band 2y W-ADAPT**, **Band 2z M-LLM-R**, **Band 2aa M.6 P4**, and **Band 2ab M.6 P5** are **closed**. **Band 2ac M.6 P6** = **Done** (32/32) — see **[§6.1y](.#61y-harness-implementation-queue--integration-expansion-m6-p6-done)**. **Band 2ay M.12** = **Done** — see **[§6.1an](.#61an-harness-implementation-queue--llm-guardrail-integrations-closed)**. **Optional harness extension (after gate green):** **[Band 2ae Phase HEP](.#61aw-phase-hep--harness-evidence-pack-band-2ae)** — runtime evidence packaging.

```text
Verify (every harness PR):
  uv run pytest -m gate -q
  python scripts/maintenance/check_harness_no_getattr.py
  python scripts/maintenance/check_legacy_modules_removed.py
  python scripts/maintenance/check_agent_skill_resolution.py
  python scripts/maintenance/check_harness_registry_resolution.py
  python scripts/maintenance/check_harness_capability_graph_wiring.py
  python scripts/maintenance/check_legacy_tool_plan_booleans.py
  python scripts/maintenance/check_trace_bridge_event_catalog.py
  python scripts/maintenance/check_plugin_catalog.py
  python scripts/maintenance/check_llm_adapter_typed_returns.py
  python scripts/maintenance/check_agents_llm_adapter_response.py
  uv run python scripts/release/phase_w_ops_evidence.py
  # Per release (ops):
  uv run python scripts/release/export_harness_shadow_eval_trend.py --release-id <release-id>
  uv run python scripts/release/record_harness_release_cycle.py --cycle-id <release-id> --verify-gate
  python scripts/maintenance/check_scaffold_harness_alignment.py
  python scripts/maintenance/check_agents_no_tier3_imports.py
  python scripts/maintenance/check_intergrax_no_applications_imports.py
  python scripts/check_no_upward_application_imports.py
  uv run python scripts/maintenance/check_harness_prompt_golden_catalog.py
  uv run python scripts/maintenance/check_agents_lifecycle_metadata.py
  uv run intergrax doctor --ci
  uv run python scripts/release/phase_v_closeout_gate.py --enforce --enforce-l4
  uv run python scripts/release/phase_w_adapt_closeout_gate.py --enforce-l4-runtime
  uv run python scripts/release/phase_v_capability_graph_guard.py --enforce
  python scripts/maintenance/check_agents_no_inline_prompts.py
  python scripts/maintenance/check_agents_no_vendor_sdk_imports.py
  uv run python scripts/gates/check_ideal_harness_l3_gates.py
  uv run python scripts/gates/harness_maturity_report.py --enforce-l3-critical
```

**Out of scope for §6.1:** product/application implementation work and business backlogs. This document does not track product application roadmaps.

### 6.1av Harness implementation queue — Platform Foundation audit maintenance

**Source:** Interactive layer audit (2026-06-19) — `PLATFORM_FOUNDATION` layers 1, 2, 32 · [`../audit_results/2026-06-19/PLATFORM_FOUNDATION.md`](../../../audit_results/2026-06-19/PLATFORM_FOUNDATION.md) · prior: [`../audit_results/2026-06-18/PLATFORM_FOUNDATION.md`](../../../audit_results/2026-06-18/PLATFORM_FOUNDATION.md)
**Priority ladder:** **Band 1** (§6.1) — doc hygiene + optional legacy cleanup; runs **in parallel** with gate maintenance

| Order | ID | Type | Priority | Status | Deliverable | Acceptance |
|-------|-----|------|----------|--------|-------------|------------|
| 1 | **PF-MAINT-DOC-01** | Docs | P2 | **Done** | Remove stale M.6 P6 from audit prompt known-gaps; sync audit result file | Audit prompt + result match plan §6.1y (**Done** 32/32) |
| 2 | **PF-MAINT-DOC-02** | Docs | P2 | **Done** | Sync §6.1au + §4.0 Band 2az counter with `AUDIT_IDEAL_2026.md` | Plan shows **90/90 Done** · **0 Planned** |
| 3 | **PF-MAINT-DX-01** | Docs | P3 | **Done** | Implementer quick-start in `intergrax_runtime_architecture.md` hub | §4.0 ladder + scaffold flow linked |
| 4 | **PF-MAINT-LEG-01** | Code | P3 | **Done** | Remove `use_rag`/`use_websearch` from LLM planner schema (`EnginePlan`) | `check_legacy_tool_plan_booleans.py` green; `tool_ids` only |
| 5 | **PF-MAINT-DOC-03** | Docs | P3 | **Done** | Sync §0.5 regression gate counter with live `pytest -m gate` snapshot | Plan §0.5 shows **1498 passed** (2026-06-19) |
| 6 | **PF-MAINT-LEG-02** | Code | P3 | **Done** | Remove legacy `use_rag`/`use_websearch` shims from `ToolInvocationPlan` (`tool_runtime.py`) | Zero DeprecationWarning in gate; `tool_ids` only at runtime bridge |
| 7 | **PF-MAINT-AUDIT-01** | Docs | P3 | **Done** | Persist Mode A2 audit result under `docs/audit_results/2026-06-19` | `PLATFORM_FOUNDATION.md` + `progress.json` present |

**Suggested PR order:** none — §6.1av queue closed (2026-06-19).

**Explicitly excluded:** §50 marketplace, new Tier-0 mechanisms beyond §6.1 — see [§6.3](.#63-end-of-platform-plan).

### 6.1aw Phase HEP — Harness Evidence Pack (Band 2ae)

**Status:** HEP-1 **Done**; HEP-2 Trace Evidence Path **Done**; HEP-3 Evidence Posture / Scoreboard **Done**; EVID-CORE-FU-01 Selected Live Tier-0 Probes **Done** — `certify core` → `trace export` → `evidence live-core` → `evidence posture` / `evidence posture export`. EVID-CORE-FU-01 adds selected local no-network live Tier-0 probes with mock LLM/tools. It does not replace deterministic CORE certification and is not full runtime certification.
EVID-EVAL Eval Regression Evidence **Done** — `evidence eval` writes deterministic eval evidence artifacts and optionally enriches `evidence posture` via `EVAL_REGRESSION` when the report exists. It is not a new eval framework and does not run real LLM/provider evaluation.
EVID-COST Cost Evidence **Done** — `evidence cost` writes deterministic local cost evidence artifacts and optionally enriches `evidence posture` via `COST_EVIDENCE` when the report exists. It is not a billing engine, provider pricing system, cloud cost estimator, or real LLM usage meter.
Evidence platform proof path **Done** — the canonical local proof path is now documented in architecture and README: `certify core` → `trace export` → `evidence live-core` → `evidence eval` → `evidence cost` → `evidence posture` / `posture export`.
Evidence smoke audit: **Done** — canonical local proof path verified (see `HARNESS_EVIDENCE_PACK.md` § A2 closeout). README / onboarding update after smoke audit: **Done** — operator-facing proof path in README (see `HARNESS_EVIDENCE_PACK.md` § A3 closeout). Evidence artifact sanity checker / docs checker: **Done** — `scripts/maintenance/check_evidence_artifacts.py` validates expected artifacts and README proof-path references (see `HARNESS_EVIDENCE_PACK.md` § A4 closeout). External one-page harness narrative: **Done** — `docs/project/technical/guides/INTERGRAX_HARNESS_NARRATIVE.md` (see `HARNESS_EVIDENCE_PACK.md` § A5 closeout). **Strong ROI and polished/adopter-ready ROI are closed.** No immediate HEP evidence ROI task remains; deferred evidence waves remain deferred until explicitly prioritized. The detailed task count and roadmap live in `HARNESS_EVIDENCE_PACK.md` § Evidence ROI roadmap. **Boundary:** HEP remains §6.1 harness/platform evidence extension — not product/application work.
**Priority ladder:** **Band 2ae** — §6.1 extension (harness evidence / runtime proof / onboarding); runs **after** gate green; **not** product/application work
**Source:** External infrastructure audit (2026-06) + operator decision B → A → C

| Wave | Scope | IDs | Status |
|------|-------|-----|--------|
| HEP-0 | Mapping & contracts | EVID-CORE-01 … EVID-CORE-03 | **Done** (2026-06-21 C1) |
| HEP-1 | Core evidence path (`intergrax certify core`) | EVID-CORE-04 … EVID-CORE-06 | **Done** (2026-06-21 C2–C3) |
| HEP-2 | Trace evidence path (`intergrax trace show` · `intergrax trace export`) | EVID-TRACE-01 … EVID-TRACE-04 | **Done** (2026-06-21 C4–C6) |
| HEP-3 | Evidence posture / scoreboard (`intergrax evidence posture`) | EVID-POSTURE-01 … EVID-POSTURE-04 | **Done** (2026-06-21 C10) |

**Explicitly excluded:** product/application implementation work · W-ADAPT L4 replacement · duplicate doctor CI gates · Band 2ad (M.7 P7 — **Done**).

**Suggested PR order:** see [`HARNESS_EVIDENCE_PACK.md`](HARNESS_EVIDENCE_PACK.md) § Implementation waves.

### 6.1p Phase P-Ext paydown (Band 2c — optional parallel with §6.1)

**Status:** **Done** (2026-06-02) · closure complete; extend catalogs via Appendix I + author guide.

| Order | ID | Deliverable | Priority |
|-------|-----|-------------|----------|
| 1 | P-Ext.0.5 | Fixture pip package (`tests/fixtures/plugin_packages`) | P0 |
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

**Out of scope for §6.1:** product/application implementation work and business backlogs. This document does not track product application roadmaps. **Feature queues:** Phase W-ADAPT — §6.1t; Phase M-LLM-R — §6.1v; Phase M.6 P4 — §6.1w (closed); Phase M.6 P5 — §6.1x (closed); Phase M.6 P6 — §6.1y (closed).

### 6.2af Phase M.6 P5 execution order (Band 2ab — Planned)

**Status:** **Done** (2026-06-02) · register: [M.6 P5](.#m6-p5--harness-integration-depth-done--3334) · queue: [§6.1x](.#61x-harness-implementation-queue--integration-depth-m6-p5-done)

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

### 6.3 End of platform plan

This plan is platform-only. Product applications, product agents, and business backlogs are intentionally out of scope and are not tracked here.

Platform Foundation tracks only harness/platform maintenance, platform gates, platform evidence, and reusable platform closeouts.
