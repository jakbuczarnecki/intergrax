# Observability Spine (HOS) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/OBSERVABILITY.md`](../architecture/OBSERVABILITY.md) · [`plan/OBSERVABILITY.md`](../plan/OBSERVABILITY.md)  
**Audit map layers:** 21, 30 · compact slice: [`audit_slices/OBSERVABILITY.md`](../guides/audit_slices/OBSERVABILITY.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

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

Audit the **Harness Observability Spine**: layered event catalog (spine + event_kind), typed DiagnosticPayload, unified journal, causal trees, and operator reconstructability.

## Key symbols and contracts

RuntimeEvent · event_kind · EventCategory · EventCatalog · emit_domain_signal · DiagnosticPayload · TraceComponent · ops filter hints

## Active plan phases (verify status vs code reality)

OBS-BUS 0–7 Done · OBS-EVOL-9 Planned · ADR-OBS-001 · ADR-OBS-003

## Known open gaps — re-validate every item (closed / still open / partial)

OBS-LC Done · OBS-EVOL-9 M0–M3 Done · runtime_event.v2 preview registered · product dashboards §6.3a → Phase K

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/OBSERVABILITY.md`](../guides/audit_slices/OBSERVABILITY.md) — compact slice (layers **21, 30**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/OBSERVABILITY.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/OBSERVABILITY.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/OBSERVABILITY.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
5. `@docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H (observability mandatory vs optional)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/OBSERVABILITY.md` — then inspect:

```text
intergrax/runtime/events/runtime_event.py · event_catalog.py · signals.py · event_bus.py
intergrax/runtime/nexus/tracing/ · ObservabilityEmitter · TraceScope
intergrax/runtime/events/payload_registry.py · persistence_conformance.py
scripts/maintenance/check_observability_gates.py · check_event_catalog.py
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Single spine — no per-agent private trace SQLite DBs.
2. Spine event_type frozen ~50 at publication; domain extends via event_kind.
3. Every spine RuntimeEventType has EventCatalog entry — phase + ops hint + payload.
4. Tier-2/3 use emit_domain_signal — not new RuntimeEventType.
5. DiagnosticPayload guard rejects raw dicts where typed schema required.
6. parent_event_id via TraceScope — causal tree reconstructable.
7. AGENT_SELECTED, STEP_FAILED, TOOL_*, POLICY_* emitted on hot paths.
8. Journal export includes parser/RAG summaries where applicable.
9. redact() before persist in production_mode.
10. Extension SDK registers schema_id for custom payloads.
11. correlation_id defaults to task_id consistently.
12. persistence_conformance assert passes.
13. Multi-agent graph callbacks emit typed graph_node.v1 payloads.
14. Metrics layer third after events — not replacing journal.
15. Debug APIs documented; PII never in prod journal content fields.
16. check_harness_observability_wiring.py green for reference hosts.
17. External OTLP export optional — canonical journal always populated.

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

Apply **every** section in `docs/audit/README.md` §Shared production Harness checklist:

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
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Per-agent trace DB · raw prompt/completion in prod journal · Tier-2 adding RuntimeEventType · metrics-only observability

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/maintenance/check_observability_gates.py
uv run python scripts/maintenance/check_event_catalog.py
uv run pytest tests/unit/runtime/observability/ -q
uv run pytest tests/unit/runtime/events/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- **O1 terse** checkpoint unless operator requests full report.
- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7–§8 for final write-up.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update plan/arch gap rows; **no code** unless operator requests separately.

Begin the audit now.

---END PROMPT---
