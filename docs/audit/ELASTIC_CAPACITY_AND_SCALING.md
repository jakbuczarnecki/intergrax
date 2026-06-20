# Elastic Capacity and Platform Scaling (ECP) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plan/ELASTIC_CAPACITY_AND_SCALING.md)  
**Audit map layers:** 30 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

## 0. Context budget (mandatory — quality without bulk loading)

Deep audit = **targeted reads + code/gate evidence**, not loading entire plan files.

### Session rules
- **One domain per chat** unless the operator explicitly batches.
- **Never** read a file >500 lines in full — grep section headers, then `Read` with offset/limit.
- **Never** re-read the same file in one session unless it changed.
- Prefer **grep with path filters** over repo-wide semantic search for known symbols.
- Run **only** scripts in section 10 — no full-suite pytest unless this prompt lists a domain slice.
- Do **not** load `docs/audit_results/` unless RESUME/bootstrap says so.
- Respect **`.cursorignore`** — excluded paths are out of scope unless the operator points to them.

### Scoped plan read (`docs/plan/{DOMAIN}.md`)
Read **only**: `## 6.` open queue rows only · gap/remediation registers tied to **Known open gaps** and **Active plan phases** · skip `(closed)`, `(complete)`, `Archived` unless re-validating a listed gap

### Scoped architecture read (`docs/architecture/{DOMAIN}.md`)
Table of contents + sections for audit-map layers **30** + registers tied to **Known open gaps**. Skip historical paydown logs unless a gap ID points there.

### Scoped guide reads
- **Prefer** [`docs/guides/audit_slices/{DOMAIN}.md`](../guides/audit_slices/{DOMAIN}.md) — compact slice for this domain (replaces bulk IDEAL + AUDIT_MAP load)
- Otherwise: `IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for layers **30** only
- `INTEGRAX_HARNESS_AUDIT_MAP.md` — layers **30** + maturity §5 only
- `SYSTEM_INVARIANTS.md` — skim invariant IDs referenced in section 3 dimensions only

---


## 1. Canonical reads (scoped — in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — **layers 30 only** (see §0)
2. `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` — **scoped sections** (see §0)
3. `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` — **scoped sections only** (see §0) — do **not** load the full file
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — **layers 30** + §5 maturity
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **N/A — cross-ref OBSERVABILITY (SLIs) and ORCHESTRATION (backpressure)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/queueing/ · intergrax/distributed/
integrations/providers/cloud_platform/kubernetes/
integrations/providers/message_bus/celery/
intergrax/runtime/architecture/multi_agent_contention_simulation.py
intergrax/runtime/observability/harness_slos.py
target: intergrax/runtime/capacity/ (ECP-DEPTH ECP-1..8)
docs/adr/entries/2026-06-08/ADR-SCALE-001.md · ADR-SCALE-002.md
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

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

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` gap rows + `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
