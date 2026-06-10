# Agent Contracts and Assembly — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Audit map layers:** 17–20, 31 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: AGENT_CONTRACTS_AND_ASSEMBLY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice, e.g. "ingest pipeline only" or "ToolRuntime policy path"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Agent Contracts and Assembly (`AGENT_CONTRACTS_AND_ASSEMBLY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Agent Contracts and Assembly** domain — architecture canon, implementation plan, source code, tests, and CI gates. Compare against production-grade systems in this problem space. Do **not** produce a shallow documentation survey.

**Mission:** Audit agent contracts, registry resolution, prompt assembly, capability graph, and agent lifecycle governance.

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state for this concern
2. `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — current architecture canon
3. `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — implementation status and gap registers
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 17–20, 31
5. `docs/guides/audit/README.md` — shared production Harness checklist (mandatory)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix M (prompt), Appendix N/O/P (assembly, registry, capability graph)** — control-plane wiring

---

## 2. Code and test paths (inspect concretely)

Search and read — do not rely on memory:

```text
intergrax/agents/, agents/, registry, prompt registry, capability graph
tests/unit/ and tests/integration/ matching the above
scripts/check_harness_*.py and scripts/check_* relevant to this domain
```

---

## 3. Domain-specific audit dimensions

Answer each with **Yes / Partial / No / Unknown** and **evidence** (file + symbol or test name):

1. AgentContract: capabilities, allowed_tools, skill_ids resolution, risk metadata.
2. Registry: host resolution, version pinning, discovery — CI wiring scripts green.
3. Prompt registry: YamlPromptRegistry, PromptProfile, layered compilation (system/task/policy/context).
4. Capability graph: agent ↔ tool ↔ skill edges, policy-aware traversal.
5. Agent assembly path — no vendor SDKs in Tier-2; composition from Tier-0 only.
6. Lifecycle: scaffold → register → evaluate → deprecate; governance documented.

---

## 4. Workload and scale probes

Evaluate behaviour for:

Large agent rosters, many capabilities, prompt template fan-out.

For each probe: describe actual code path, limits, and failure mode — not hypothetical design.

---

## 5. Tier-3 and agent override surfaces

Verify customization without forking Tier-0/Tier-1:

Per-agent manifest, host registry overrides, custom PromptRegistryProtocol impl at Tier-3.

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
- `audit-and-fix`: update `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` gap rows and `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` audit register if present; **no code changes** unless user requests separately
- Never declare the whole platform complete
- Record out-of-scope findings with suggested next domain

Begin the audit now.

---END PROMPT---
