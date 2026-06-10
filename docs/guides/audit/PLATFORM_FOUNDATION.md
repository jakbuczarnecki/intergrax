# Platform Foundation — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) · [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md)  
**Audit map layers:** 1–2, 32 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: PLATFORM_FOUNDATION
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Platform Foundation (`PLATFORM_FOUNDATION`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Platform Foundation** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Verify Intergrax remains a Harness AI / Agent OS — durable runtime, replaceable agents — with correct four-tier model, documentation governance, and strategic alignment to the ideal architecture.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/PLATFORM_FOUNDATION.md` — current architecture canon
3. `docs/plan/PLATFORM_FOUNDATION.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 1–2, 32
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
docs/, AGENTS.md, .cursor/rules/, tier boundaries across repo
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. Harness vs agent prioritization — product logic not in Nexus; agents not hard-wiring platform internals.
2. Four-tier dependency rules enforced in code imports (`intergrax/` ↔ `agents/` ↔ `applications/`).
3. Documentation model: 21 domain pairs 1:1, hub-only root, no monolithic plan files.
4. Strategic principles in canon match implementation reality (policy-first, trace-everything, composable-by-default).
5. Gate maintenance workflow and PLATFORM_FOUNDATION ladder — plan rows match evidence.
6. Architecture governance loop — audits update paired docs, ADRs, plan registers.

---

## 4. Workload and scale probes

Evaluate behaviour for:

N/A — meta-layer; sample multiple tiers for boundary violations.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

Tier placement rules for new components; scaffold defaults; extension author boundaries.

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
- `audit-and-fix`: update `docs/plan/PLATFORM_FOUNDATION.md` gap rows and `docs/architecture/PLATFORM_FOUNDATION.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
