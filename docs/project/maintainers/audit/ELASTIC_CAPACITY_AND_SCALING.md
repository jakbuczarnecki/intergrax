# Elastic Capacity and Platform Scaling (ECP) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plans/ELASTIC_CAPACITY_AND_SCALING.md)  
**Audit map layers:** 30 · compact slice: [`audit_slices/ELASTIC_CAPACITY_AND_SCALING.md`](../../technical/guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md)  
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

domain: ELASTIC_CAPACITY_AND_SCALING
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Elastic Capacity and Platform Scaling (ECP) (`ELASTIC_CAPACITY_AND_SCALING`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Elastic Capacity and Platform Scaling (ECP)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **Elastic Capacity Plane**: signals, ScalingPolicy, backpressure vs autoscale distinction, queueing/workers, K8s integration path, ECP-DEPTH target modules — honest L0–L2 vs plan targets.

## Key symbols and contracts

ScalingProfile (target) · ScalingPolicy · ScalingAction · ScalingSignal · CapacitySignalCollector · ScalingProvisioner · SIG_QUEUE_DEPTH · GRAPH_BACKPRESSURE

## Active plan phases (verify status vs code reality)

ECP-DOC · ECP-DEPTH (ECP-1..8, ECP-OBS) · ADR-SCALE-001/002 · cross-ref W-OPS.4 SLIs · ORCH GRAPH_BACKPRESSURE

## Known open gaps — re-validate every item (closed / still open / partial)

ECP-LC Done · §6.1av ECP-MAINT Done · live K8s soak manual runbook · ingress slug → INT-MAINT-04

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md`](../../technical/guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md) — compact slice (layers **30**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/ELASTIC_CAPACITY_AND_SCALING.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/ELASTIC_CAPACITY_AND_SCALING.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **N/A — cross-ref OBSERVABILITY (SLIs) and ORCHESTRATION (backpressure)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/ELASTIC_CAPACITY_AND_SCALING.md` — then inspect:

```text
intergrax/queueing/ · intergrax/distributed/
integrations/providers/cloud_platform/kubernetes/
integrations/providers/message_bus/celery/
intergrax/runtime/architecture/multi_agent_contention_simulation.py
intergrax/runtime/observability/harness_slos.py
target: intergrax/runtime/capacity/ (ECP-DEPTH ECP-1..8)
docs/project/technical/adr/entries/2026-06-08/ADR-SCALE-001.md · ADR-SCALE-002.md
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. ECP control loop async outside Nexus hot path.
2. Provisioning via integrations/tools — not Nexus importing K8s SDK.
3. Backpressure (GRAPH_BACKPRESSURE) ≠ auto-scale — documented distinction.
4. Hysteresis + cooldown on scaling rules (target ECP).
5. Scale actions idempotent with NOTIFY_ONLY at max replicas.
6. Tenant isolation on capacity signals.
7. K8s HPA complementary — Intergrax rules orchestrate ceilings.
8. Agent topology scaling (dimension B) separate from worker scaling.
9. SCALE_* trace events when ECP implemented.
10. Fail-safe on provisioner error — no runaway scale-up.
11. Tier-3 owns deploy manifests (Helm/HPA in applications/*/docker/).
12. PolicyEngine/HITL on scale-up when profile requires.
13. Queue depth from intergrax/queueing/task_index.py as signal.
14. Multi_agent_contention_simulation aligns with architecture claims.
15. Honest: mark L0/L1 where ECP-DEPTH not yet implemented.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- GRAPH_BACKPRESSURE rate under max_inflight_nodes.
- Queue depth burst → worker autoscale (target ECP-5).
- Modality Celery W-OPS.12 cross-ref.
- Multi-replica Nexus vs in-process concurrency caps.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ScalingProfile on ApplicationEnvironmentProfile (target) · OrchestrationProfile.max_inflight_nodes ceiling · Helm/HPA per host

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/project/maintainers/audit/README.md` §Shared production Harness checklist:

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

Compare against: **Kubernetes HPA/VPA · Celery autoscale · nginx upstream scaling · cloud autoscaler APIs · Prometheus SLI runbooks**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Nexus hot-path synchronous provisioning · conflating backpressure with autoscale · missing cooldown · scale without tenant bounds

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/architecture/test_multi_agent_contention_simulation.py -q
uv run pytest tests/unit/queueing/ -q
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
