# Platform Foundation — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) · [`plan/PLATFORM_FOUNDATION.md`](../plan/PLATFORM_FOUNDATION.md)  
**Audit map layers:** 1–2, 32 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: PLATFORM_FOUNDATION
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Platform Foundation (`PLATFORM_FOUNDATION`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Platform Foundation** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Verify Intergrax is developed as a **Harness AI / Agent OS** — the runtime is the durable product, agents are replaceable — with enforced four-tier boundaries, 22 domain-pair documentation governance, gate maintenance discipline, and strategic alignment to IDEAL_HARNESS_AI_ARCHITECTURE.

## Key symbols and contracts

Four-tier model · IntegrationProfile/ToolProfile/SkillProfile/LLMProfile · ApplicationEnvironmentProfile · ApplicationManifest · RuntimePolicyBundle · AgentContract · plugin entry points (intergrax.tools, intergrax.skills, intergrax.integrations)

## Active plan phases (verify status vs code reality)

§6.1 gate maintenance queue · Phase V architecture hardening · Phase K business agents (**deferred** — must not start silently) · §6.3 product backlog

## Known open gaps — re-validate every item (closed / still open / partial)

Phase K / §6.3 deferred product work · long-term §50 marketplace/visual builder · codecraft/ incremental · unified tool model (legacy boolean flags deprecated)

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/PLATFORM_FOUNDATION.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/PLATFORM_FOUNDATION.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 1–2, 32
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
docs/intergrax_runtime_architecture.md (hub)
docs/architecture/PLATFORM_FOUNDATION.md · docs/plan/PLATFORM_FOUNDATION.md
AGENTS.md · .cursor/rules/intergrax-iteration.mdc
scripts/check_intergrax_no_applications_imports.py
scripts/check_agents_no_tier3_imports.py
scripts/check_docs_domain_pairs.py
scripts/check_harness_no_getattr.py
scripts/phase_v_capability_graph_guard.py
intergrax/applications/reference/harness_manifest_catalog.py
Sample imports across intergrax/, agents/, applications/ for tier violations
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Harness treated as durable product — not single-agent optimization (§1 strategic frame).
2. Tier-0 (`intergrax/`) contains only universal mechanisms — no business agent logic.
3. Tier-1 Nexus domain-agnostic — no agent-specific branches in NexusLoop.
4. Tier-2 agents consume Tier-0 via policy/ToolRuntime — no vendor SDK imports.
5. Tier-3 applications compose runtime+agents+profiles — no duplicated agent pipelines.
6. Import boundaries enforced: `intergrax/` ↛ `agents/`/`applications/`; agents ↛ applications.
7. Documentation model: hub-only `docs/` root; 22 architecture↔plan pairs 1:1; no monolithic plan.
8. New capabilities reuse Tier-0 (§5.2.2) — no parallel universal mechanisms.
9. LLM calls via `llm_adapters/` — not Integration Library vendor wrappers.
10. Integrations register via manifest/`register_from_manifest` — not ad-hoc SDK in agents.
11. Gate maintenance §6.1 rows match evidence (tests, CI scripts, doc updates).
12. Scaffold (`new-agent`, `new-application`, `new-stack`) emits tier-correct artifacts + ADR folders.
13. Capability graph seeding uses `harness_manifest_catalog` — not orphan registrations.
14. `getattr`/reflection banned outside approved bridges — CI green.
15. Phase K / business agents not started without explicit operator reprioritization.
16. Architecture governance loop: audits update paired docs, ADRs, plan registers — not chat-only.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- 185+ integration slugs in catalog — stable vs beta honesty.
- Harness lab stack (sqlite, redis, qdrant, otel) as reference Tier-3 preset.
- Plugin entry-point registration at scale (tools, skills, integrations bundles).

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

IntegrationProfile presets (`lab_stack`, `legal_stack`, `research_stack`, `harness_production_stack`) · ApplicationManifest · scaffold defaults · `wire_application_environment`

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

Compare against: **Cursor/Claude Code/Codex-class agent harnesses · enterprise Agent OS platforms (policy-first, composable runtime, replaceable workers)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Declaring whole platform complete · starting Phase K silently · duplicating Tier-0 in Nexus · monolithic implementation plan files

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/check_docs_domain_pairs.py
uv run python scripts/check_intergrax_no_applications_imports.py
uv run python scripts/check_agents_no_tier3_imports.py
python scripts/check_harness_no_getattr.py
uv run pytest -m gate -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/PLATFORM_FOUNDATION.md` gap rows + `docs/architecture/PLATFORM_FOUNDATION.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
