# Stage 17 — Agent DX Closure

**Audit ID:** STAGE-17-AGENT-DX-CLOSURE  
**Date:** 2026-09-05  
**Branch:** `development`  
**Start HEAD:** `e9cd8cb0d52cf4123d3ecb2306ae2b067db634ff`  
**origin/development (start):** `e9cd8cb0d52cf4123d3ecb2306ae2b067db634ff`

---

## Canonical developer journey

```text
Create agent (new-agent scaffold → contract/capabilities/domain logic)
        ↓
Verify agent (await agent.run(AgentRunRequest(...)) + canonical_execution_identity_scope in tests)
        ↓
Integrate (AgentBinding / distribution metadata → Agent Distribution lifecycle → Execution)
```

Public surfaces route to `AGENT_CREATION_GUIDE`, `AGENT_DISTRIBUTION`, and `APPLICATION_RUNTIME_GRAPH_MODEL`. Tier-3 authors do not construct mutable `AgentRegistry` or direct `NexusLoop` for serving.

---

## Residual inventory from Stage 16

| ID | Stage 16 state | Stage 17 resolution |
| --- | --- | --- |
| S16-006 | Historical notebooks with `NexusLoop` | **Retained (historical)** — banner added; not linked as canonical tutorials |
| S16-007 | Legacy agent tests via `NexusLoop` | **Closed** — migrated to `agent.run()` |
| S16-008 | Appendices I–O Nexus quickstart semantics | **Closed** — terminology blocks + Appendix C/D integration rewrite |
| S16-009 | Scenario `AgentRegistry()` fallback | **Closed** — removed scenario-owned registry construction |
| S16-010 | `RuntimeContext` / `RuntimeRequest` in `intergrax.runtime.nexus.*` | **Accepted namespace debt** — no neutral public re-export; not a lifecycle bypass |

---

## S16-006 status — notebooks

**Option B (KEEP AS HISTORICAL)** for all 10 experiment notebooks under `agents/*/notebooks/`:

- `dispute_analyst`, `dispute_intake`, `dispute_scenario`, `dispute_strategist`
- `intergrax_assistant`, `legal`, `local_indexer`, `local_search`, `local_synthesizer`, `signoff_probe`

First markdown cell now states historical / non-production status and points to AGENT_CREATION_GUIDE Step 4. No canonical doc links found; no migration required.

---

## S16-007 status — legacy agent tests

**Migrated (TYPE 1 — agent unit smoke):** 12 packages

- `signoff_probe`, `dispute_analyst`, `dispute_scenario`, `dispute_intake`, `dispute_strategist`
- `intergrax_assistant`, `legal`, `local_search`, `local_synthesizer`, `local_indexer`
- `vendor_discovery`, `problem_radar`
- `external_contractor_adapter` (identity scope added to existing `agent.run` test)

Pattern: `canonical_execution_identity_scope` + `AgentRunRequest`. Removed `NexusLoop` / `AgentRegistry` from author-facing smoke tests.

**Removed from agent contract tests:** `local_search` / `local_indexer` registry-bootstrap assertions (registry skill resolution remains framework-owned).

**Retained as internal (TYPE 3):** Tier-1 `tests/unit/runtime/nexus/*`, execution integration tests, conformance gates — unchanged.

---

## S16-008 status — appendices

Bounded audit of appendices I–O: internal `NexusLoop` references retained only as orchestration control-plane documentation.

Changes:

- Terminology block added to appendices I–O (`Execution` / `Nexus` / `Agent Distribution` / `AgentRegistry` projection).
- Appendix C multi-agent example rewritten to public **Execution** boundary (no `loop.handle_task` author snippet).
- Step 4 host factory wording: registry projection → Execution (not `NexusLoop` quickstart).
- Appendix K integration sample reframed as host-factory composition (no direct `NexusLoop(...)` author wiring).

Bounded authoring gate scope unchanged (main guide Steps 1–6 + anti-patterns; appendices excluded by design).

---

## S16-009 status — scenario fallback

**Consumer audit:** `build_runtime_bundle` callers use fixture/platform composition; `is_platform_attached` fallback had no real non-platform production consumer.

**Resolution:** Removed `AgentRegistry()` construction from `platform_proofs/scenarios/ai_incident_investigation/application/scenario.py`. Non-attached compositions delegate to `build_scenario_runtime_composition(...)` only. Scenario application roster construction moved to Tier-1 `build_scenario_lab_agent_registry()` in `scenario_runtime_baseline.py` (platform-owned, not scenario-local).

---

## S16-010 status — Nexus namespace

**Option 2:** No neutral public contract/re-export for `RuntimeContext` / `RuntimeRequest`. Dependencies remain on `intergrax.runtime.nexus.*` in agent authoring scaffold.

Documented as **namespace debt only** — not a lifecycle or public-execution bypass. No broad namespace move (Stage 17 stop rule).

---

## Deleted legacy

- Scenario-local `AgentRegistry()` bootstrap in `ai_incident_investigation/application/scenario.py`
- Agent package tests constructing `NexusLoop` for smoke verification (12 files)
- Agent contract tests using `registry.register()` for tool resolution smoke (`local_search`, `local_indexer`)

---

## Migrated legacy

- 12 agent smoke tests → `agent.run(AgentRunRequest(...))`
- Scaffold `new_agent.py` generated test template → `canonical_execution_identity_scope`
- AGENT_CREATION_GUIDE Appendix C + Step 4 + Appendix K sample
- 10 historical notebooks → historical banner

---

## Explicitly retained internal/historical uses

- Historical experiment notebooks (bannered, not linked canonically)
- Tier-1 Nexus internal tests (`NexusLoop` allowed where testing orchestration)
- Appendices I–O internal orchestration reference material (with Stage 17 terminology guardrails)

---

## Conformance gates

| Gate | Result |
| --- | --- |
| `check_canonical_authoring_surface_conformance.py` | Pass (run in Stage 17 verification) |
| Agent package `-m gate` smoke (migrated packages) | 13 passed |
| Scenario architecture gates | Pass (run in Stage 17 verification) |
| Dedicated ai_incident real-package conformance | PASS with zero lifecycle exemptions |
| Aggregate initialized-scenario architecture conformance | PASS |
| indirect_prompt_injection / verified_product_identification lifecycle | PASS — zero `AGENT_LIFECYCLE_BYPASS` |
| Application lifecycle gates | Pass (run in Stage 17 verification) |
| Stage 15 architecture gate | Pass (run in Stage 17 verification) |
| Stage 15 E2E | Pass (run in Stage 17 verification) |

No new repo-wide `NexusLoop` ban gate (Tier-1 internal tests remain valid).

---

## Residual debt

1. `RuntimeContext` / `RuntimeRequest` namespace under `intergrax.runtime.nexus.*` (S16-010).
2. Historical notebook code cells still contain legacy Nexus imports (bannered; not canonical).
3. `DOCUMENTATION_MAP.md` / root `README.md` could add explicit `APPLICATION_RUNTIME_GRAPH_MODEL` link in a future docs-only pass (navigation already reaches `AGENT_CREATION_GUIDE` + `AGENT_DISTRIBUTION`).

---

## Confirmation

- No public Tier-3 Nexus quickstart in bounded authoring surfaces
- No public mutable `AgentRegistry` quickstart in bounded authoring surfaces
- No scenario-owned production lifecycle (`AgentRegistry()` removed from scenario path)
- No initialized scenario owns mutable `AgentRegistry` lifecycle
- No second lifecycle authority introduced
- Stage 15 proof suite re-verified in Stage 17 test budget
