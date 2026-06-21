# Harness Evidence Pack — Implementation Plan

**Phase:** HEP (Harness Evidence Pack)  
**Band:** 2ae  
**Hub register:** [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) §6.1aw  
**Architecture (DX owns smoke/e2e evidence):** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> **Placement:** §6.1 harness infrastructure extension — **not** §6.3 product work.  
> **Naming:** Do **not** use `IDEAL-L4-EVIDENCE` — L4 in repo is W-ADAPT closed-loop semantics (`l4_runtime_evidence.py`). Do **not** reuse Band 2ad (M.7 P7 integrations — **Done**).

**Last updated:** 2026-06-21 — HEP-1 **Done** (EVID-CORE-01…06); HEP-2 Trace Evidence Path **Done** (EVID-TRACE-01…04; C4–C6); HEP-3 Evidence Posture / Scoreboard **In progress** — EVID-POSTURE-01/02 **Done** (C8/C9); EVID-POSTURE-03/04 **Planned**.

---

## Cursor read scope (token budget)

- **Implement HEP-1 (closed):** § Mode I summary · § Certification semantics · § CORE levels · **EVID-CORE-*** rows.
- **Implement HEP-2 Trace:** § Mode I — HEP-2 · § Trace semantics · open **EVID-TRACE-*** rows only.
- **Plan HEP-3 Posture:** § Mode I — HEP-3 · § Evidence posture semantics · open **EVID-POSTURE-*** rows only.
- **Skip** HEP-2 EVID-EVAL / EVID-COST and HEP-4+ unless implementing those waves.
- **Architecture:** DX read-scope block only — smoke/e2e evidence owns list.
- **Audit slice:** [`guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md).

---

## Mode I summary (approved 2026-06-21)

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-0-1` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `improvement` |
| **Tier** | Tier-0 (`intergrax/cli/`, `intergrax/runtime/evidence/`) |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` (primary) · `PLATFORM_FOUNDATION` (hub) |

**Reason:** Mechanisms exist across IDEAL-* gates, doctor, trace, policy, eval, cost, W-OPS, and W-ADAPT — but no single **productized runtime certification path** (`intergrax certify core`). Value = packaging + executable proof + onboarding clarity.

**ADR:** No ADR needed for HEP-0 + HEP-1 (packaging existing contracts; no Nexus/UAEP/HarnessKernel semantic change).

---

## Problem statement

Intergrax is **architecture-heavy** (32/32 L3 scorecard, extensive CI gates) but **evidence-heavy adoption** is fragmented:

- Repo health (`pytest -m gate`, `intergrax doctor --ci`) ≠ live harness runtime proof.
- W-ADAPT L4 evidence measures adaptive utility improvement — not core E2E certification.
- Agent certification (`agent_certification.py`) is per-agent — not harness core.

**Goal:** One operator path — `intergrax certify core` — with JSON/Markdown report under `build/evidence/core_certification/`.

---

## Certification semantics (EVID-CORE-02)

| Mechanism | Question | Scope | Output |
|-----------|----------|-------|--------|
| `pytest -m gate` | Do tests pass? | Unit/integration matrix | pytest report |
| `intergrax doctor` / `--ci` | Is repo wiring healthy? | ~15 check scripts | PASS/FAIL per script |
| `phase_v_closeout_gate --enforce-l4` | Governance artifacts OK? | Capability graph + maturity evidence | closeout stdout |
| `check_l4_runtime_evidence.py` | W-ADAPT closed-loop OK? | 30-day utility / rollback on golden scenarios | `l4_runtime_evidence.json` |
| `check_mvp_promotion_gates.py` G0–G2 | MVP runnable baseline exists? | File existence checks | G0–G2 OK |
| **`intergrax certify core`** | Does the CORE evidence contract path execute end-to-end? | Deterministic mock evidence, no network/real LLM; not live HarnessKernel/Nexus/provider E2E | CORE-L* + JSON/Markdown certification report |

**CI placement (operator decision):** `certify core` is **operator-local** at HEP-1 launch. **Not** a default PR gate. Optional later: CORE-L1 as nightly or release preflight — not in scope for HEP-1.

---

## CORE certification levels

| Level | Scenarios | Use case |
|-------|-----------|----------|
| **CORE-L1** | `basic_run_completed`, `trace_persisted`, `tool_denied_by_policy`, `certification_report_emitted` | Quick smoke (~2 min) |
| **CORE-L2** | CORE-L1 + `high_risk_tool_hitl`, `budget_exceeded_handled`, `retry_executed`, `domain_signal_emitted` | Default `certify core` target |
| **CORE-L3** | CORE-L2 + `llm_error_classified`, `memory_read_write_recorded`, `rag_context_event_recorded`, `cost_report_generated` | Full evidence matrix |

CLI: `intergrax certify core --level L1|L2|L3` (default **L2**).

---

## Output artifacts (EVID-CORE-05)

```text
build/evidence/core_certification/
  report.json      # structured certification report
  report.md        # human-readable summary
```

Future waves may add sibling dirs (not HEP-1):

```text
build/evidence/trace/
build/evidence/eval/
build/evidence/cost/
build/evidence/posture/
```

---

## Non-goals

- **Not** §6.3 product work (K.1, K.2, LKW, Legal, new Tier-3 apps).
- **Not** W-ADAPT L4 replacement — keep `check_l4_runtime_evidence.py` separate semantics.
- **Not** duplicate doctor — do not add certify scenarios as new `check_*.py` entries in `doctor.py`.
- **Not** per-agent or business agent certification — see `agent_certification.py`, `business_agent_certification.py`.
- **Not** full Trace Explorer, Policy DSL, Extension SDK — deferred HEP-2+ (see § Future waves).
- **Not** new Tier-0 parallel subsystems — reuse event spine, reference harness, golden scenario IDs.

---

## External audit mapping (EVID-CORE-01)

Condensed matrix from external infrastructure audit → existing evidence → gap → proposed item.

| Audit # | Finding | Existing (plan / code) | Gap type | HEP item |
|---------|---------|------------------------|----------|----------|
| 1 | Core Certification Pack | IDEAL-27.3, §6.1 gates, `doctor.py`, `l4_runtime_evidence.py` | runtime proof + packaging | EVID-CORE-01…06 |
| 2 | Trace Explorer product | IDEAL-27.1 **Done**, `trace_explorer_routes.py` | UX depth | EVID-TRACE-* (HEP-2) |
| 3 | Policy-as-Product | IDEAL-5.x, `RuntimePolicyBundle` | mechanism + UX | EVID-POL-* (HEP-4) |
| 4 | Eval regression path | IDEAL-25.x, `check_eval_scenario_library.py` | packaging | EVID-EVAL-* (HEP-2) |
| 5 | Capability compatibility | capability graph gates, plugin catalog | UX | EVID-CAP-* (HEP-5) |
| 6 | Runtime replay | MVP-EVOL.3, `intergrax mvp replay` | product depth | EVID-REPLAY-* (HEP-5) |
| 7 | Cost intelligence | IDEAL-24.x, `assembly_cost.py` | UX | EVID-COST-* (HEP-2) |
| 8 | Context compiler | ContextProfile, CE-MAINT-02 | mechanism | EVID-CTX-* (HEP-5) |
| 9 | Extension SDK | P-Ext **Done**, scaffold | formal contract | EVID-EXT-* (HEP-5) |
| 10 | Security hardening pack | IDEAL-23.x, injection gates | packaging | EVID-SEC-* (HEP-5) |
| 11 | Attestation standard | ExecutionBoundaryExport (architecture) | mechanism | EVID-ATT-* (HEP-5) |
| 12 | Harness scoreboard | `harness_maturity_report.py` (static) | runtime proof | EVID-POSTURE-* (HEP-3) |

---

## Implementation register — Wave HEP-0 + HEP-1

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-CORE-01** | HEP-0 | P1 | **Done** | External audit → evidence matrix in plan | § External audit mapping (2026-06-21) |
| **EVID-CORE-02** | HEP-0 | P1 | **Done** (doc) | Certification spec: CI gate vs certify-core vs W-ADAPT L4 | § Certification semantics + § CORE levels delivered in docs; **code enums** (`CoreCertificationLevel`, etc.) ship with EVID-CORE-03 in C1 — not a separate reopen of this row |
| **EVID-CORE-03** | HEP-0 | P1 | **Done** | Runtime scenario contracts (12 scenarios) | `intergrax/runtime/evidence/` — Pydantic contracts; deterministic; no network; `validate_core_scenario_catalog()` |
| **EVID-CORE-04** | HEP-1 | P0 | **Done** | `intergrax certify core` CLI | `intergrax/cli/certify.py`; `--level L1\|L2\|L3`; exit non-zero on failure |
| **EVID-CORE-05** | HEP-1 | P1 | **Done** | JSON + Markdown certification report | `build/evidence/core_certification/report.json` + `report.md` |
| **EVID-CORE-06** | HEP-1 | P2 | **Done** | README / HARNESS_ENVIRONMENT evidence path | Quick start + HARNESS_ENVIRONMENT § Core certification; mock vs live narrative |

---

## Scenario contracts (EVID-CORE-03)

| Scenario ID | CORE min level | Reuse path |
|-------------|----------------|------------|
| `basic_run_completed` | L1 | `EchoAgent` + `AgentEngine` — `tests/integration/agents/test_agent_engine_uaep_echo.py` |
| `trace_persisted` | L1 | `SQLiteRunTraceStore` readback — IDEAL-27.1 trace store |
| `tool_denied_by_policy` | L1 | Policy deny — IDEAL-5.5 adversarial fixtures; `test_kernel_policy_pre_deny` |
| `certification_report_emitted` | L1 | EVID-CORE-05 report models (new) |
| `high_risk_tool_hitl` | L2 | IDEAL-11.3 HITL policy pattern |
| `budget_exceeded_handled` | L2 | IDEAL-24.5 quota hard-stop |
| `retry_executed` | L2 | REL retry path tests |
| `domain_signal_emitted` | L2 | `RuntimeEventBus` + event catalog |
| `llm_error_classified` | L3 | Error handling / REL classification |
| `memory_read_write_recorded` | L3 | MEM wiring events |
| `rag_context_event_recorded` | L3 | RAG/CTX `CONTEXT_ASSEMBLED` events |
| `cost_report_generated` | L3 | `assembly_cost.py` / budget policy |

Golden scenario IDs for echo baseline: `golden-echo`, `golden-policy`, `golden-routing` (from `l4_runtime_evidence.py` — reuse IDs only, not W-ADAPT utility semantics).

---

## Delivered modules (HEP-1 complete)

```text
intergrax/cli/certify.py
intergrax/runtime/evidence/
  __init__.py
  core_certification_spec.py    # EVID-CORE-02 code enums + CORE_LEVEL_SCENARIOS
  scenario_contracts.py         # EVID-CORE-03 contracts + validate_core_scenario_catalog()
  scenario_runner.py            # deterministic mock evidence runner
  certification_report.py       # JSON + Markdown report
tests/unit/runtime/evidence/
  test_core_certification_spec.py
  test_scenario_contracts.py
  test_scenario_runner.py
  test_certification_report.py
  test_certify_cli.py
```

**Docs (EVID-CORE-06):** README Quick start + `guides/HARNESS_ENVIRONMENT.md` § Core certification evidence path (HEP).

**Follow-up (EVID-CORE-FU-01):** `tests/integration/evidence/test_core_certification.py` — optional live Tier-0 runtime probes; not HEP-1 scope.

**Tier boundaries:** Tier-0 only — no `applications/` imports in evidence runner; use `reference_harness.py` / echo agent patterns.

**Optional preflight:** `certify core --with-doctor` may run doctor checks before scenarios — does not add scenarios to doctor.

---

## Implementation notes (C1 + C2)

| Artifact | Path |
|----------|------|
| CORE levels + surfaces | `intergrax/runtime/evidence/core_certification_spec.py` |
| Scenario contracts (12) | `intergrax/runtime/evidence/scenario_contracts.py` |
| Scenario runner (mock proof) | `intergrax/runtime/evidence/scenario_runner.py` |
| Report JSON/Markdown | `intergrax/runtime/evidence/certification_report.py` |
| CLI | `intergrax/cli/certify.py` · `intergrax certify core` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |
| Unit tests | `tests/unit/runtime/evidence/` (incl. `test_certify_cli.py`) |

**Verify:** `uv run pytest tests/unit/runtime/evidence -q` · `uv run intergrax certify core --level L2`

---

## Implementation waves (suggested PR order)

| PR | IDs | Scope | Status |
|----|-----|-------|--------|
| **PR1** | EVID-CORE-02 (code), EVID-CORE-03 | Spec enums + scenario contracts + unit tests | **Done** (C1) |
| **PR2** | EVID-CORE-04, EVID-CORE-05 | CLI + report generation | **Done** (C2) |
| **PR3** | EVID-CORE-06 | Docs: README, HARNESS_ENVIRONMENT evidence path | **Done** (C3) |

**Verify after PR2:** `uv run intergrax certify core --level L2` + `uv run pytest -m gate -q`

---

## Mode I — HEP-2 Trace Evidence Path

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-2-trace` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `developer_experience` · `observability` |
| **Tier** | Tier-0 / Tier-1 boundary, depending on existing trace package placement |
| **Domains** | `OBSERVABILITY` · `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` |

**Verdict:** `partial_overlap`

**Reason:** Trace routes, event spine, trace stores, and debug endpoints already exist; HEP-1 report path is delivered. The gap is a lightweight, canonical evidence timeline contract and operator path (`intergrax trace show`) — not a full Trace Explorer UI in HEP-2.

**ADR:** No ADR needed for HEP-2 Trace (packaging timeline over existing evidence artifacts; no Nexus/UAEP/HarnessKernel semantic change).

### Problem statement (HEP-2)

HEP-1 answers whether the core certification contract passed. HEP-2 must show **how** it passed.

The gap is not raw observability infrastructure. The gap is a small, canonical evidence timeline that can be rendered in CLI, Markdown, JSON, and later Trace Explorer.

**Strategic goal:** After `certify core`, the operator sees pass/fail and a report. After the trace evidence path, the operator sees a step-by-step timeline: certification started → scenario started → evidence emitted → policy/budget/HITL marker → scenario passed → report written → certification completed.

### Trace semantics

| Surface | Existing mechanism | Question | HEP-2 role |
|---------|-------------------|----------|------------|
| Runtime events | `RuntimeEvent` / `RuntimeEventBus` | What happened inside runtime? | Source input, not operator format |
| Trace store | `PersistedRunTraceEventStore` / persisted trace records | What was recorded? | Source input / future adapter |
| Trace Explorer routes | `trace_explorer_routes.py` · `/ops/trace/*` or debug trace routes | Can operators inspect traces in host? | Existing infra, not HEP-2 CLI contract |
| Debug formatters | `build_trace_payload` (`intergrax/debug/formatters.py`) | How are persisted traces shaped for UI? | Existing infra; HEP-2 does not replace |
| Replay bridge | `trace_replay_bridge.py` | How are trace events replayed? | Existing infra; not HEP-2 timeline contract |
| HEP-1 report | `build/evidence/core_certification/report.json` / `report.md` | Did certification pass? | Input artifact for timeline |
| Trace Evidence Path | planned `intergrax trace show` | What happened step-by-step? | Canonical operator timeline |

### Planned trace evidence artifacts

```text
build/evidence/trace/
  timeline.json
  timeline.md
```

HEP-2 may read `build/evidence/core_certification/report.json` as an input artifact.

### HEP-2 non-goals

- Not full Trace Explorer UI.
- Not replacement for existing trace routes (`trace_explorer_routes.py`, debug trace endpoints).
- Not new event bus or `RuntimeEventBus` wiring.
- Not new trace store (`PersistedRunTraceEventStore` remains separate).
- Not live provider tracing.
- Not W-ADAPT L4 evidence (`check_l4_runtime_evidence.py` semantics).
- Not `doctor` changes.
- Not a new default CI gate.
- Not changing `intergrax certify core` semantics.
- Not implementing EVID-CORE-FU-01 live runtime probes.

---

## Implementation register — Wave HEP-2 Trace Evidence Path

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-TRACE-01** | HEP-2 | P1 | **Done** | Trace timeline contract | Canonical timeline models for evidence runs; stable event kinds; artifact refs; no CLI yet |
| **EVID-TRACE-02** | HEP-2 | P1 | **Done** | Policy/budget/HITL facets | Timeline can represent policy decisions, budget markers, HITL markers, scenario lifecycle |
| **EVID-TRACE-03** | HEP-2 | P1 | **Done** | `intergrax trace show` CLI | Renders timeline from HEP-1 report/evidence artifacts; no UI |
| **EVID-TRACE-04** | HEP-2 | P2 | **Done** | Trace export JSON/Markdown | Writes timeline JSON/Markdown under `build/evidence/trace/` |

---

## Suggested implementation order — HEP-2

| Step | IDs | Scope |
|------|-----|-------|
| C4 | EVID-TRACE-01 | Timeline contracts only | **Done** |
| C5 | EVID-TRACE-02 | Facets for policy/budget/HITL/evidence | **Done** |
| C6 | EVID-TRACE-03/04 | CLI + export | **Done** |

---

## Implementation notes (C4 · EVID-TRACE-01)

| Artifact | Path |
|----------|------|
| Trace timeline contracts | `intergrax/runtime/evidence/trace_timeline_contracts.py` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_contracts.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_contracts.py -q`

---

## Implementation notes (C5 · EVID-TRACE-02)

| Artifact | Path |
|----------|------|
| Trace timeline facets | `intergrax/runtime/evidence/trace_timeline_facets.py` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_facets.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_facets.py -q`

---

## Implementation notes (C6 · EVID-TRACE-03/04)

| Artifact | Path |
|----------|------|
| Certification report adapter | `intergrax/runtime/evidence/trace_timeline_adapter.py` |
| Timeline CLI/export | `intergrax/runtime/evidence/trace_timeline_export.py` |
| CLI | `intergrax/cli/trace.py` · `intergrax trace show` · `intergrax trace export` |
| Unit tests | `tests/unit/runtime/evidence/test_trace_timeline_adapter.py` · `test_trace_timeline_export.py` · `test_trace_cli.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Operator path:**

```bash
uv run intergrax certify core --level L2
uv run intergrax trace show
uv run intergrax trace export
```

**Operator semantics (C6 hotfix):**

- Timeline is **report-derived only** from `build/evidence/core_certification/report.json`.
- Evidence basis matches HEP-1: `deterministic_mock` — not live runtime trace, event bus, or trace store.
- `trace show` renders to stdout; `trace export` writes `build/evidence/trace/timeline.{json,md}`.
- Policy/budget/HITL facets are descriptive markers mapped from scenario evidence refs, not runtime probes.

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_trace_timeline_adapter.py tests/unit/runtime/evidence/test_trace_timeline_export.py tests/unit/runtime/evidence/test_trace_cli.py -q`

---

## HEP-2 closeout (2026-06-21)

HEP-2 Trace Evidence Path: **Done**  
EVID-TRACE-01…04: **Done**

HEP-2 delivered the report-derived operator timeline path:

```bash
uv run intergrax certify core --level L2
uv run intergrax trace show
uv run intergrax trace export
```

`trace show` renders the timeline to stdout.
`trace export` writes `build/evidence/trace/timeline.json` and `timeline.md`.

The timeline is derived from `build/evidence/core_certification/report.json` and uses deterministic mock evidence. It is not live RuntimeEventBus, persisted trace store, or provider tracing.

The `certification_report_emitted` scenario is represented in the timeline by a `REPORT_WRITTEN` event kind — not a separate scenario lifecycle triplet.

---

## Mode I — HEP-3 Evidence Posture / Scoreboard

| Field | Value |
|-------|-------|
| **Idea label** | `harness-evidence-pack-hep-3-posture` |
| **Verdict** | `partial_overlap` |
| **Type** | `harness_capability` · `developer_experience` · `evidence_packaging` |
| **Tier** | Tier-0 CLI / evidence package, with read-only references to existing gates |
| **Domains** | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` · `PLATFORM_FOUNDATION` · `OBSERVABILITY` |

HEP-1 answers: did the core certification path pass?

HEP-2 answers: what happened step-by-step in the report-derived timeline?

HEP-3 should answer: what is the current evidence posture of the harness?

The goal is not to create new runtime proof. The goal is to aggregate existing evidence surfaces into a small operator-facing posture summary that clearly distinguishes:

- repo health,
- test gates,
- core certification,
- trace artifacts,
- deterministic mock evidence,
- deferred live runtime probes,
- W-ADAPT L4 as separate semantics.

The posture surface should make Intergrax easier to evaluate without reading the entire architecture. It should show what evidence exists, what it proves, what it does not prove, and which follow-ups are deferred.

### Evidence posture semantics

| Surface | Source | Question | HEP-3 role |
|---------|--------|----------|------------|
| Repo health | `intergrax doctor` | Is repository wiring healthy? | Input signal / optional status |
| Pytest gate | `pytest -m gate` | Does the unit/integration gate pass? | Input signal, not runtime proof |
| Core certification | `build/evidence/core_certification/report.json` | Did CORE-L* deterministic contract evidence pass? | Primary evidence source |
| Trace timeline | `build/evidence/trace/timeline.json` | Is report-derived timeline available? | Secondary evidence source |
| W-ADAPT L4 | `check_l4_runtime_evidence.py` | Is adaptive utility/rollback evidence healthy? | Separate semantics, not CORE posture |
| Live Tier-0 probes | EVID-CORE-FU-01 | Are selected runtime probes live? | Deferred / future input |

### HEP-3 non-goals

- Not a new CI gate.
- Not a replacement for `doctor`.
- Not a replacement for `pytest -m gate`.
- Not live runtime certification.
- Not W-ADAPT L4 evidence.
- Not Trace Explorer UI.
- Not policy DSL.
- Not cost engine.
- Not eval regression runner.
- Not EVID-CORE-FU-01 live Tier-0 probes.
- Not a production readiness certification for business applications.

---

## Implementation register — Wave HEP-3 Evidence Posture / Scoreboard

| ID | Wave | Priority | Status | Deliverable | Acceptance criteria |
|----|------|----------|--------|-------------|---------------------|
| **EVID-POSTURE-01** | HEP-3 | P1 | **Done** | Evidence posture contract | Pydantic/read-model contract for posture summary; no CLI yet |
| **EVID-POSTURE-02** | HEP-3 | P1 | **Done** | Evidence posture collector | Reads existing artifacts and optional command outputs; no new runtime proof |
| **EVID-POSTURE-03** | HEP-3 | P1 | **Planned** | `intergrax evidence posture` CLI | Renders current evidence posture to stdout |
| **EVID-POSTURE-04** | HEP-3 | P2 | **Planned** | Posture export JSON/Markdown | Writes posture artifacts under `build/evidence/posture/` |

---

## Suggested implementation order — HEP-3

| Step | IDs | Scope |
|------|-----|-------|
| C8 | EVID-POSTURE-01 | Posture contracts only | **Done** |
| C9 | EVID-POSTURE-02 | Artifact collector / read-only aggregation | **Done** |
| C10 | EVID-POSTURE-03/04 | CLI + export |

---

## Implementation notes (C8 · EVID-POSTURE-01)

| Artifact | Path |
|----------|------|
| Evidence posture contracts | `intergrax/runtime/evidence/evidence_posture_contracts.py` |
| Unit tests | `tests/unit/runtime/evidence/test_evidence_posture_contracts.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_evidence_posture_contracts.py -q`

---

## Implementation notes (C9 · EVID-POSTURE-02)

| Artifact | Path |
|----------|------|
| Evidence posture collector | `intergrax/runtime/evidence/evidence_posture_collector.py` |
| Unit tests | `tests/unit/runtime/evidence/test_evidence_posture_collector.py` |
| Public exports | `intergrax/runtime/evidence/__init__.py` |

**Verify:** `uv run pytest tests/unit/runtime/evidence/test_evidence_posture_collector.py -q`

---

## Planned posture artifacts

```text
build/evidence/posture/
  posture.json
  posture.md
```

HEP-3 may read:

```text
build/evidence/core_certification/report.json
build/evidence/trace/timeline.json
```

It should not require live providers, network access, runtime event bus, or trace store.

---

## Planned HEP-3 operator path

```bash
uv run intergrax doctor
uv run pytest -m gate -q
uv run intergrax certify core --level L2
uv run intergrax trace export
uv run intergrax evidence posture
```

Expected posture summary should clearly state:

- repo health: available / unknown / failed,
- gate status: available / unknown / failed,
- core certification: available / missing / failed,
- core level: CORE-L1 / CORE-L2 / CORE-L3 / unknown,
- trace timeline: available / missing,
- evidence basis: deterministic_mock,
- live Tier-0 probes: deferred,
- W-ADAPT L4: separate,
- overall posture: onboarding-ready / partial / missing-evidence.

---

## Completed waves

| Wave | IDs | Status |
|------|-----|--------|
| HEP-1 Core Certification | EVID-CORE-01 … EVID-CORE-06 | **Done** — `certify core` report path delivered |
| HEP-2 Trace Evidence Path | EVID-TRACE-01 … EVID-TRACE-04 | **Done** — `trace show` / `trace export` report-derived timeline delivered |

---

## Future waves (not approved for implementation)

| Wave | IDs | Audit priority |
|------|-----|----------------|
| HEP-2 (other) | EVID-EVAL, EVID-COST | External audit #4, #7 — not Mode I approved yet |
| HEP-3 | EVID-POSTURE-01 … EVID-POSTURE-04 | External audit #12 — **Mode I seed** (see § Mode I — HEP-3); **Planned**, not approved for implementation |
| HEP-4 | EVID-POL | External audit #3 |
| HEP-5 | EVID-CAP, EVID-REPLAY, EVID-CTX, EVID-EXT, EVID-SEC, EVID-ATT | External audit #5–11 |
| HEP-FU | **EVID-CORE-FU-01** | Replace selected mock scenarios with real Tier-0 runtime probes (post HEP-1; not mixed with CORE-L* contract delivery) |

Register future rows in this file when Mode I approves each wave.

### EVID-CORE-FU-01 (deferred follow-up — not HEP-1)

| ID | Priority | Status | Deliverable | Acceptance |
|----|----------|--------|-------------|------------|
| **EVID-CORE-FU-01** | P2 | **Deferred** | Live Tier-0 runtime probes for selected CORE scenarios | Subset of scenarios exercise real HarnessKernel/event spine with mock LLM only; no §6.3 product scope |

---

## Definition of Done (HEP-1)

1. **Contract** — Pydantic report + scenario contracts in `intergrax/runtime/evidence/`
2. **Trace** — scenarios assert `RuntimeEvent` / trace persistence where applicable
3. **Test** — integration tests deterministic, no network
4. **Documentation** — this plan + EVID-CORE-06 onboarding path
5. **No regression** — `pytest -m gate` green; certify-core does not break doctor
6. **Reuse Tier-0** — extend existing harness paths; no parallel certification stack
