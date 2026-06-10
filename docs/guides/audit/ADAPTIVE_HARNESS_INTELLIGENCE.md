# Adaptive Harness Intelligence (L4) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md) · [`plan/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../plan/ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Audit map layers:** L4 AHI · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: ADAPTIVE_HARNESS_INTELLIGENCE
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Adaptive Harness Intelligence (L4) (`ADAPTIVE_HARNESS_INTELLIGENCE`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Adaptive Harness Intelligence (L4)** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Audit L4 adaptive loops: bounded self-tuning, evaluation-driven routing, and safe automation without uncontrolled behaviour changes.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` — current architecture canon
3. `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers L4 AHI
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
adaptive routing, feedback loops, AHI modules in runtime
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. Feedback signals defined and consumed (eval scores, latency, cost, quality).
2. Adaptive routing bounded by policy — not unconstrained model/tool switching.
3. Human approval gates for adaptive policy changes.
4. Separation from L3 production paths — experimental flags clear.
5. Observability of adaptive decisions (why route changed).
6. Deferred vs implemented features explicitly marked in plan.

---

## 4. Workload and scale probes

Evaluate behaviour for:

Continuous adaptation under load, feedback delay, exploration/exploitation balance.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

AHI profiles, opt-in per Tier-3 host, kill switches.

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
- `audit-and-fix`: update `docs/plan/ADAPTIVE_HARNESS_INTELLIGENCE.md` gap rows and `docs/architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
