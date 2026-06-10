# Elastic Capacity and Platform Scaling — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ELASTIC_CAPACITY_AND_SCALING.md`](../architecture/ELASTIC_CAPACITY_AND_SCALING.md) · [`plan/ELASTIC_CAPACITY_AND_SCALING.md`](../plan/ELASTIC_CAPACITY_AND_SCALING.md)  
**Audit map layers:** 30 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with repository access.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus`).
4. Output must follow [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: ELASTIC_CAPACITY_AND_SCALING
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Elastic Capacity and Platform Scaling (`ELASTIC_CAPACITY_AND_SCALING`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Elastic Capacity and Platform Scaling** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Audit platform scaling: elastic capacity, backpressure, resource pools, contention handling, and SLO-oriented capacity governance.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` — current architecture canon
3. `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 30
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **OBSERVABILITY (SLIs), ORCHESTRATION (backpressure)** — control-plane wiring

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
capacity simulation, backpressure, scaling hooks, ECP modules
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. Backpressure signals from orchestration to intake.
2. Concurrency and queue depth limits enforced.
3. Capacity model or simulation aligned with architecture (ECP).
4. Horizontal scaling assumptions documented — state externalization.
5. Observability SLIs for saturation and queue latency.
6. Cost/capacity coupling — scale without budget blowout.

---

## 4. Workload and scale probes

Evaluate behaviour for:

Burst traffic, multi-tenant contention, regional failover.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

Capacity profiles, queue tuning per deployment, elastic worker pools at Tier-3 infra.

Confirm overrides are **wired**, not documentation-only.

---

## 6. Cross-cutting checklist (mandatory)

Apply every item in `docs/guides/audit/README.md` §Shared production Harness checklist:

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

## 7. Production comparison

Compare the implementation to **production-grade systems** in this domain (commercial and open-source). State clearly:

- What Intergrax already matches at L3 production Harness OS level
- What is L2 or below with specific gaps
- What is intentionally deferred (design boundary) vs **niedoróbka** / missing wiring

---

## 8. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5:

```text
L0 — Fragmented
L1 — Operational MVP
L2 — Scalable Harness
L3 — Production Harness OS
L4 — Adaptive Agent OS
```

Report **score before**, **target for current milestone**, evidence, and **remaining risks**.

---

## 9. Verification commands

Run applicable checks; cite results:

```bash
uv run pytest -m gate -q
uv run pytest tests/unit/<relevant>/ -q
python scripts/check_harness_no_getattr.py
# plus domain-specific scripts discovered during inspection
```

---

## 10. Output and mode rules

- Follow output format in `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 (Audit Result template).
- End with §8 Completion Summary.
- `audit-only`: **no file edits**
- `audit-and-fix`: update `docs/plan/ELASTIC_CAPACITY_AND_SCALING.md` gap rows and `docs/architecture/ELASTIC_CAPACITY_AND_SCALING.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
