# Agent Contracts and Assembly — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Audit map layers:** 17–20, 31 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: AGENT_CONTRACTS_AND_ASSEMBLY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Agent Contracts and Assembly (`AGENT_CONTRACTS_AND_ASSEMBLY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Agent Contracts and Assembly** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **AgentContract**, registry resolution, **Prompt Registry**, capability graph, agent lifecycle governance, and assembly paths — Tier-2 as composable workers with no vendor SDKs and full registry/traceability discipline.

## Key symbols and contracts

AgentContract · Agent (interface) · AgentExecutionResult · CapabilityDescriptor · CapabilityMatchResult · PromptMeta · YamlPromptRegistry · AgentExecutionMode

## Active plan phases (verify status vs code reality)

PE (Prompt Registry) · REG (Registry) · CG (Capability Graph) · AS + V-REM-ALG (Agent Lifecycle)

## Known open gaps — re-validate every item (closed / still open / partial)

prompt_instruction_ids bridge to ContextManager (SK-BRIDGE.1 cross-domain) · procedural memory store minimal · retired agents filter in production_mode

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 17–20, 31
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix M (prompt) · Appendix N/O/P (assembly, registry, capability graph)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/contracts/agent_contract_meta.py
intergrax/runtime/registry/agent_registry.py
intergrax/prompts/registry/ (YamlPromptRegistry)
intergrax/runtime/architecture/prompt_registry_governance.py · prompt_composition.py · prompt_policy_overlay.py
intergrax/runtime/architecture/capability_graph*.py
intergrax/runtime/architecture/agent_lifecycle_governance.py · agent_certification.py
agents/ (Tier-2 roster) · applications/_shared/prompt_wiring.py
scripts/check_agents_lifecycle_metadata.py · scripts/phase_v_capability_graph_guard.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. AgentContract has required fields per §12 — capabilities, allowed_tools, risk metadata.
2. execute() delegates to AgentEngine — not standalone HTTP/script bypass.
3. Nexus routes by capability token — not Python class name.
4. Prompt templates have ownership, version, layered compilation (system/task/policy/context).
5. Capability graph edges reflect manifest roster with lineage.
6. Deprecated/retired agents rejected in strict production_mode.
7. Registry snapshot conformance tests pass CI.
8. Agent creation checklist §45 satisfied for reference agents.
9. Evaluation registry wired for promotion evidence.
10. AgentExecutionResult is structured — not bare str.
11. Forbidden §42.41 patterns absent (vendor SDK, direct integrations).
12. Certification gates documented before production roster add.
13. skill_ids → allowed_tools resolution audited (check_agent_skill_resolution if present).
14. Host registry resolution CI green (check_harness_registry_resolution).
15. Capability graph wiring CI green (check_harness_capability_graph_wiring).

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Large agent roster with capability-based routing.
- Registry snapshot at bootstrap vs runtime mutation.
- Promotion dev→staging→prod evidence chain.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

PromptProfile · AgentRegistry.register(skill_registry, tool_registry) · Tier-3 manifest roster · wire_application_environment · external SkillImporter / Cursor SKILL.md

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

Compare against: **Enterprise agent registries · prompt governance (versioned templates) · capability-based routing (service-mesh analogy)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Hardcoded agent class routing · vendor SDK in Tier-2 · orphan prompts without registry · skipping lifecycle metadata

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/check_agents_lifecycle_metadata.py
uv run python scripts/phase_v_capability_graph_guard.py
uv run python scripts/check_agents_vendor_imports.py
uv run pytest agents/ -q --co -q 2>/dev/null | head
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` gap rows + `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
