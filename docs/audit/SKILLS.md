# Skill Library — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/SKILLS.md`](../architecture/SKILLS.md) · [`plan/SKILLS.md`](../plan/SKILLS.md)  
**Audit map layers:** 12 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

Audit **149 skills / 41 bundles** as composable capability packs above tools: resolution, policy fragments, registration, roster consistency, and honest SK-BRIDGE gap status.

## Key symbols and contracts

SkillManifest · SkillProfile · SkillRegistry · SkillResolver/SkillResolverProtocol · ResolvedSkillPack · SkillPlugin · SkillBundleEntry

## Active plan phases (verify status vs code reality)

SK-EXP through SK-EXP5 Done · **SK-BRIDGE.1** prompt→ContextManager · **SK-BRIDGE.2** policy_fragment→bundle · SK-PRESET.1 · Phase TS-3

## Known open gaps — re-validate every item (closed / still open / partial)

prompt_instruction_ids not auto-injected to ContextManager · policy_fragment_id not merged to RuntimePolicyBundle · knowledge bundle BETA

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
Table of contents + sections for audit-map layers **12** + registers tied to **Known open gaps**. Skip historical paydown logs unless a gap ID points there.

### Scoped guide reads
- **Prefer** [`docs/guides/audit_slices/{DOMAIN}.md`](../guides/audit_slices/{DOMAIN}.md) — compact slice for this domain (replaces bulk IDEAL + AUDIT_MAP load)
- Otherwise: `IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for layers **12** only
- `INTEGRAX_HARNESS_AUDIT_MAP.md` — layers **12** + maturity §5 only
- `SYSTEM_INVARIANTS.md` — skim invariant IDs referenced in section 3 dimensions only

---


## 1. Canonical reads (scoped — in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — **layers 12 only** (see §0)
2. `docs/architecture/SKILLS.md` — **scoped sections** (see §0)
3. `docs/plan/SKILLS.md` — **scoped sections only** (see §0) — do **not** load the full file
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — **layers 12** + §5 maturity
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix J**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/skills/registry/catalog.py · bootstrap.py · resolver.py
intergrax/skills/integration/contract_resolution.py
intergrax/skills/providers/*/ · importers/cursor_skill_md.py
applications/_shared/skill_wiring.py · skill_tool_profile.py · catalog_runtime_bridge.py
intergrax/runtime/registry/agent_registry.py (skill resolution at register)
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

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

- 149 skills resolved for agent with deep requires_skills chain.
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

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/SKILLS.md` gap rows + `docs/architecture/SKILLS.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
