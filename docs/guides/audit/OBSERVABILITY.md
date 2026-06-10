# Observability Spine (HOS) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) · [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md)  
**Audit map layers:** 21, 30 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with **full repository access**.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: OBSERVABILITY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Observability Spine (HOS) (`OBSERVABILITY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Observability Spine (HOS)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Harness Observability Spine**: 54+ RuntimeEventType catalog, typed DiagnosticPayload, unified journal, causal trees, redaction, extension SDK, and operator reconstructability of every run.

## Key symbols and contracts

RuntimeEvent · TraceEvent · DiagnosticPayload · TraceComponent · ObservabilityProfile · CoreLLMCallRecordedDiagV1 · ToolsSummaryDiagV1 · RagSummaryDiagV1 · ops filter hints

## Active plan phases (verify status vs code reality)

OBS-BUS 0–7 Done · ADR-OBS-001 · maintenance §6.1 only

## Known open gaps — re-validate every item (closed / still open / partial)

Residual RuntimeEventPayload evolution · product dashboards deferred §6.3a · external APM optional not mandatory

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/OBSERVABILITY.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/OBSERVABILITY.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 21, 30
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H (observability mandatory vs optional)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/runtime/events/runtime_event.py · event_bus.py · phase_coverage.py · unified_run_journal.py
intergrax/runtime/nexus/tracing/ · ObservabilityEmitter · TraceScope
intergrax/runtime/observability/payload_registry.py · persistence_conformance.py
intergrax/runtime/observability/harness_slos.py
LLM/RAG/modality metric plugins
scripts/check_observability_gates.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Single spine — no per-agent private trace SQLite DBs.
2. Every RuntimeEventType has ExecutionPhase + ops filter hint — CI catalog tests.
3. DiagnosticPayload guard rejects raw dicts where typed schema required.
4. parent_event_id via TraceScope — causal tree reconstructable.
5. AGENT_SELECTED, STEP_FAILED, TOOL_*, POLICY_* emitted on hot paths.
6. Journal export includes parser/RAG summaries where applicable.
7. redact() before persist in production_mode.
8. Extension SDK registers schema_id for custom payloads.
9. correlation_id defaults to task_id consistently.
10. persistence_conformance assert passes.
11. Multi-agent graph callbacks emit typed graph_node.v1 payloads.
12. Metrics layer third after events — not replacing journal.
13. Debug APIs documented; PII never in prod journal content fields.
14. check_harness_observability_wiring.py green for reference hosts.
15. External OTLP export optional — canonical journal always populated.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Long run 10k+ events — journal merge performance.
- Nested subagents — trace tree depth.
- Export backpressure to OTLP sink.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ObservabilityProfile · wire_nexus_observability() · PersistingTaskTraceEmitter · custom RuntimeEventBus handlers (Tier-3 plugins)

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/guides/audit/README.md` §Shared production Harness checklist:

- Architecture & modularity
- Configuration & strategy selection
- Override & customization surfaces
- Observability, tracing & logging
- Security & governance
- Reliability & error handling
- Performance & scale
- Testing & verification
- Documentation alignment

---

## 7. Production baseline comparison

Compare against: **OpenTelemetry + structured logging · Datadog/Honeycomb SLO workflows · Langfuse/LangSmith LLM trace UX**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Per-agent trace DB · raw prompt/completion in prod journal · orphan event types without phase · metrics-only observability

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/check_observability_gates.py
uv run pytest tests/unit/runtime/observability/ -q
uv run pytest tests/unit/runtime/events/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/OBSERVABILITY.md` gap rows + `docs/architecture/OBSERVABILITY.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
