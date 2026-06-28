# Observability — Implementation Plan

**Architecture (1:1):** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

**Cross-plan — Agent layer (ACP):** Dual observability planes (architecture §31) — `AgentRunTrace` on `AgentRunResult` (Plane B) and `ApplicationRunSummary` on Task completion (Plane A). Delivered in [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) **Wave 3** (`ACP-OBS-1`, `ACP-OBS-2`) and **Wave 7** redaction (`ACP-PROD-8`). Trace spine changes MUST keep step records compatible with `AgentStepRecord` tool/RAG/LLM fields.

**Cross-plan — Event catalog (OBS-EVOL-9 · P1-ARCH-02):** Layered spine + `event_kind` (architecture §4.4 · ADR-OBS-003). Developers extend via `emit_domain_signal`, not new `RuntimeEventType`. Pre-release spine consolidation before publication.

**Cross-feature — Token Optimization:** feature architecture [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md) · feature plan [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md). OBSERVABILITY owns token savings attribution, optimization receipts visibility, typed diagnostic payloads, metrics, and regression-gate reporting through the Harness Observability Spine.

**Last updated:** 2026-06-24 — **OECP** plan satellite register.

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (OBSERVABILITY plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites/) satellites on demand. **On demand (one max):** [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) (active OECP register), [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) (closed phases). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Token Optimization:** read feature pair + rows `TOKEN-OBS-1` / `TOKEN-OBS-2`; use HOS/domain-signal model, do not create private telemetry channel.
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/OBSERVABILITY.md`](../guides/audit_slices/OBSERVABILITY.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Architecture documentation (P2)

| ID | Task | Status |
|----|------|--------|
| **P2-ARCH-07** | Clarify observability event spine and event ownership | **Done** (2026-06-20) |

Architecture: [`OBSERVABILITY.md`](../architecture/OBSERVABILITY.md#observability-event-spine).

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/OBSERVABILITY_eval_control_plane.md`](plan/satellites/OBSERVABILITY_eval_control_plane.md) | **OECP** — eval control plane implementation register (active) |
| [`plan/satellites/OBSERVABILITY_audit_history.md`](plan/satellites/OBSERVABILITY_audit_history.md) | audit history (closed phases) |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

---

## Phase TOKEN-OBS — Token optimization telemetry and regression gates (Planned)

**Feature:** [`features/plan/TOKEN_OPTIMIZATION.md`](../features/plan/TOKEN_OPTIMIZATION.md)  
**Architecture:** [`features/architecture/TOKEN_OPTIMIZATION.md`](../features/architecture/TOKEN_OPTIMIZATION.md)  
**Priority:** P1 after TOKEN-UER-1; TOKEN-OBS-1 may ship before CE/MEM integrations, TOKEN-OBS-2 after first optimized source exists.  
**Delivery rule:** one `TOKEN-OBS-*` row per PR; emit through HOS or approved domain-signal path only.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **TOKEN-OBS-1** | Code | P1 | Planned | `intergrax/runtime/token_optimization/telemetry.py` with typed optimization summary payload, receipt visibility, and counters/spans for saved tokens, failures, fallbacks, source type, strategy, output profile, model/provider | No private telemetry bus; attribution includes run/step/source/model/provider/strategy/profile; redaction rules followed; compatible with unified run journal; `uv run pytest tests/unit/runtime/observability/ -q`; `uv run pytest tests/unit/runtime/token_optimization/ -q` |
| **TOKEN-OBS-2** | Test/Gate | P1 | Planned | Token-vs-quality regression benchmark runner and scripts `check_compression_receipts.py`, `check_token_regression_benchmarks.py` | CI can fail on uncontrolled token growth, missing receipts, protected-region failures, or quality regression; benchmark fixtures cover output policy, tool catalog, and context pack cases; `uv run python scripts/check_compression_receipts.py`; `uv run python scripts/check_token_regression_benchmarks.py` |

**Explicit exclusions:** no new `RuntimeEventType` unless ADR/OBS review requires it; prefer typed payload/domain-signal style consistent with OBS event ownership; no raw prompt/completion persistence in production traces.

---

## Phase OBS-EXPORT — External observability export boundary (Planned)

**Purpose:** Define the platform-level export boundary for external observability sinks such as JSONL/file, OTLP, Elasticsearch, Langfuse, Arize/Phoenix.

**Required decisions:**

- External sinks are optional subscribers/export targets, not semantic owners of Intergrax observability.
- Intergrax RuntimeEvent / trace / journal / diagnostic payloads remain the canonical source.
- Vendor SDKs must not be called directly from runtime hot paths, agents, or LKW product code.
- Exporter failure must never fail product runs.
- Raw prompts, raw documents, raw RAG chunks, raw synthesized content, secrets, and full local file paths are not exported by default.
- Redaction/export policy must run before external export.
- Default posture for local-first apps such as LKW: disabled by default, strict redaction, `export_content=false`.
- OBS-EXPORT depends on LKW.2.4 pipeline proof as a representative multi-agent workload.

**Delivery rule:** one `OBS-EXPORT-*` row per PR; export through normalized envelope only; no vendor SDK in runtime hot paths.

| ID | Type | Priority | Status | Deliverable | Acceptance |
|----|------|----------|--------|-------------|------------|
| **OBS-EXPORT-1** | Code | P2 | Planned | Normalized export envelope and exporter interface | Defines stable export envelope, exporter interface, no-op exporter, and test exporter. Uses existing spine/journal/runtime metadata as source. No vendor SDK. |
| **OBS-EXPORT-2** | Code | P2 | Planned | Redaction/export policy and failure isolation | Explicit allow/drop/hash policy for exported fields. Export timeout/failure does not fail the run. Tests prove raw content is not exported. |
| **OBS-EXPORT-3** | Code | P2 | Planned | Safe JSONL/file exporter | Writes redacted export records for representative runs. Useful as reference output before vendor adapters. |
| **OBS-EXPORT-4** | Code | P2 | Planned | First real backend adapter: OTLP or Elasticsearch | Adapter maps normalized export records to backend format without changing Intergrax event semantics. |
| **OBS-EXPORT-5** | Code | P3 | Planned | Langfuse / Arize / Phoenix adapter | Adapter consumes normalized export envelope only. No runtime/vendor coupling. No raw content by default. |

---

## Phase AUDIT-IDEAL — Ideal architecture gap register (2026-06-09)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.9, §11 · baseline **32/32 L3**  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Status:** **Done** (2026-06-09) — AUDIT-IDEAL observability rows closed

| ID | AUDIT § | Gap | Priority | Status |
|----|---------|-----|----------|--------|
| AUDIT-IDEAL-5.3 | §5 Policy | Governance health dashboard (GOV-PROD.1) | P4 | **Done** |
| AUDIT-IDEAL-21.1 | §21 Observability | Causal diagnostics beyond trace bridge (ops tooling) | P2 | **Done** |
| AUDIT-IDEAL-21.2 | §21 Observability | Quality / governance / cost health dashboard contracts | P2 | **Done** |
| AUDIT-IDEAL-21.3 | §21 Observability | Unified product observability dashboard | P4 | **Done** |
| AUDIT-IDEAL-30.2 | §30 Ops | Real deploy SLO window evidence (prod `W_OPS_RELEASE_CYCLES`) | P1 | **Done** |

**Delivery rule:** One **AUDIT-IDEAL-*** ID per PR → update this table + master register → gate green.

---
