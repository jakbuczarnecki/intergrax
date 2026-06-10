# Tier-3 Application Environment — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/TIER3_APPLICATION_ENVIRONMENT.md`](../architecture/TIER3_APPLICATION_ENVIRONMENT.md) · [`plan/TIER3_APPLICATION_ENVIRONMENT.md`](../plan/TIER3_APPLICATION_ENVIRONMENT.md)  
**Audit map layers:** 3, 28 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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

domain: TIER3_APPLICATION_ENVIRONMENT
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Tier-3 Application Environment (`TIER3_APPLICATION_ENVIRONMENT`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Tier-3 Application Environment** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **deployable application hosts**: ApplicationEnvironmentProfile as composition root, all runtime bridges, catalog bootstrap, host matrix honesty, and product wiring without Nexus business logic.

## Key symbols and contracts

ApplicationEnvironmentProfile · ApplicationManifest · ApplicationBuildContext · IdentityProfile · ExecutionMode · ShadowWorkspaceProfile · SandboxProfile · ScalingProfile (ECP cross-ref) · full §22.1 sub-profiles table

## Active plan phases (verify status vs code reality)

H-APP 43 tasks Done · H-APP-WIRING · H-APP-DOC.* · CFG-* cross-ref ORCH-CONFIG

## Known open gaps — re-validate every item (closed / still open / partial)

CFG-14 LKW hybrid incomplete · MCP optional uneven · queue worker not scaffold-default · INCLUDE_INTERACTIONS/SCHEDULER adoption uneven

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 3, 28
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H (full profile map)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
applications/*/host/factory.py
intergrax/applications/contracts/environment_profile.py
applications/_shared/environment_wiring.py · nexus_factory.py · harness_host_runtime.py
applications/_shared/*_wiring.py (identity, shadow, sandbox, interaction, catalog_runtime_bridge, …)
applications/reference hosts: lab, legal, research, poc_template, LKW, …
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. ApplicationManifest declares environment_id and roster.
2. wire_application_environment() without getattr reflection.
3. Business logic only in Tier-2 agents — not Tier-3 host factory.
4. Free-text intake has classifier or explicit capability routing.
5. Posture (S1–S7) matches profile knobs §23.2.
6. All *Profile sections wired through bridges — no orphan fields.
7. bootstrap_catalogs + ToolProfile/SkillProfile/IntegrationProfile coherent.
8. Roster ⊆ skill/tool profiles (EnvironmentSkillToolConsistencyCheck).
9. IdentityProfile enforces tenant on runs.
10. Guardrail slug wired when security profile requires.
11. Task control routes mounted when INCLUDE_TASK_CONTROL.
12. Shadow/sandbox scoped per task — no global leak.
13. Docker/deploy artifacts from scaffold Phase N where claimed.
14. Host matrix §59.2 honest vs architecture claims.
15. graph_spec and OrchestrationProfile aligned per CFG case.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Cold start bootstrap all catalogs.
- Multi-host fleet profile variant drift.
- strict_multi_agent_defaults() on legal/finance hosts.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

Full ApplicationEnvironmentProfile — this layer IS the primary override surface for the platform

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

Compare against: **Reference hosts (legal_application, research_application, lab_application) · Viktor worker-in-Slack · enterprise FastAPI agent host patterns**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Business pipeline in applications/host · orphan profile fields · getattr wiring · Nexus fork per product

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/applications/ -q
uv run pytest tests/ -q -k orchestration_wiring
python scripts/check_harness_no_getattr.py
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` gap rows + `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
