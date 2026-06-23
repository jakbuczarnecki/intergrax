# Skill Library — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/SKILLS.md`](../architecture/SKILLS.md) · [`plan/SKILLS.md`](../plan/SKILLS.md)  
**Audit map layers:** 12 · compact slice: [`audit_slices/SKILLS.md`](../guides/audit_slices/SKILLS.md)  
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

domain: SKILLS
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Skill Library (`SKILLS`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Skill Library** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **150 skills / 42 bundles** as composable capability packs above tools: resolution, policy fragments, registration, roster consistency, and honest SK-BRIDGE gap status.

## Key symbols and contracts

SkillManifest · SkillProfile · SkillRegistry · SkillResolver/SkillResolverProtocol · ResolvedSkillPack · SkillPlugin · SkillBundleEntry

## Active plan phases (verify status vs code reality)

SK-EXP through SK-EXP5 Done · **SK-BRIDGE.1** prompt→ContextManager · **SK-BRIDGE.2** policy_fragment→bundle · SK-PRESET.1 · Phase TS-3

## Known open gaps — re-validate every item (closed / still open / partial)

prompt_instruction_ids not auto-injected to ContextManager · policy_fragment_id not merged to RuntimePolicyBundle · knowledge bundle BETA

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/SKILLS.md`](../guides/audit_slices/SKILLS.md) — compact slice (layers **12**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/SKILLS.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/SKILLS.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/SKILLS.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
5. `@docs/guides/AGENT_CREATION_GUIDE.md` **Appendix J** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/SKILLS.md` — then inspect:

```text
intergrax/skills/registry/catalog.py · bootstrap.py · resolver.py
intergrax/skills/integration/contract_resolution.py
intergrax/skills/providers/*/ · importers/cursor_skill_md.py
applications/_shared/skill_wiring.py · skill_tool_profile.py · catalog_runtime_bridge.py
intergrax/runtime/registry/agent_registry.py (skill resolution at register)
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Skills are not LLM-callable directly — tools are the invocation surface.
2. allowed_tools is output of registry resolution — not hand-maintained duplicate list.
3. Unknown skill_id fails at register time — not runtime surprise.
4. Resolved tool_ids exist in ToolRegistry.
5. requires_skills topological expansion detects cycles.
6. USAGE.md per skill/bundle where canon requires.
7. External Cursor SKILL.md import traced (SKILL_IMPORT_FAILED/SKILL_RESOLVED events).
8. Capability graph records skill edges.
9. Environment roster ⊆ skill/tool profile intersection enforced.
10. skill.resolve catalog tool works for diagnostics.
11. Bundles STABLE except knowledge (BETA labeled).
12. SK-BRIDGE.1/.2 gaps documented honestly — verify if closed since last audit.
13. SkillProfile presets (legal_skill_profile, research_skill_profile) wired at Tier-3.
14. extend_tool_profile_for_skills() pattern used — not duplicate tool lists.
15. Clear separation: skill composition vs atomic tool operation.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- 150 skills resolved for agent with deep requires_skills chain.
- Roster vs environment consistency check at host bootstrap.
- Import external SKILL.md at scale.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

SkillProfile.enabled_bundles · presets · AgentContract.skills[] · extend_tool_profile_for_skills() · import_cursor_skill_file

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

Compare against: **Cursor SKILL.md packs · CrewAI role bundles · policy fragments per capability pack**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Skills as parallel tool runtimes · silent skill ignore on unknown id · policy fragments not merged · knowledge bundle treated as STABLE

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/skills/ -q
uv run pytest tests/unit/ -q -k skill
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
