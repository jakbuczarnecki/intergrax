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

Audit **deployable application hosts** (architecture §24–§51): ApplicationEnvironmentProfile as composition root, host contracts §25–§32, environment state §42, production gates §40/§46, evolution §49, platform ops §50, and author DX — without Nexus business logic or duplicate registries.

## Key symbols and contracts

ApplicationEnvironmentProfile · HostMeta · CapabilityBundle · CognitionBundle · GovernanceBundle · DomainPolicyFragments · ProfileInvariantValidator · ApplicationManifest · EnvironmentSnapshot · bundle_normalized_payload

## Active plan phases (verify status vs code reality)

H-APP Done · APP-CON/PROD/EVOL/OPS Done · APP-EVOL-8 M1 Done · **Full Harness LC** (2026-06-17)

## Known open gaps — re-validate every item (closed / still open / partial)

CFG-14 LKW hybrid incomplete · MCP optional uneven · queue worker not scaffold-default · policy_coverage health proxy (UC-A7 golden deferred) · multi-tenant registry store deferred

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/TIER3_APPLICATION_ENVIRONMENT.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/TIER3_APPLICATION_ENVIRONMENT.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 3, 28
5. `docs/guides/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/APPLICATION_CREATION_GUIDE.md`
7. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix F · Appendix H**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
applications/*/host/factory.py
intergrax/applications/contracts/environment_profile/ · application_registry.py · environment_health_score.py
intergrax/applications/_shared/environment_wiring.py · harness_host_runtime.py · environment_snapshot_wiring.py
intergrax/applications/_shared/reference_capability_bundle.py · environment_conformance.py
intergrax/applications/_shared/*_wiring.py (snapshot, migration, package, health_score, recovery, certification, …)
scripts/check_application_production_gates.py · check_environment_profile_bundle_schema.py
intergrax/cli/apps.py · envs.py · doctor_health_app.py · doctor_diff_app.py
docs/guides/APPLICATION_CREATION_GUIDE.md
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. ApplicationManifest + full profile on product hosts (§45 checklist).
2. wire_application_environment() without getattr; package closure when conformance_check.
3. Business logic only in Tier-2 agents — not Tier-3 host factory (§28).
4. Capability routing via capabilities[] not class names (§37.4).
5. ApplicationHost hooks: timeout, BLOCK on error, audit events (APP-CON-5).
6. ApplicationEnvironmentState lifecycle sync on Nexus hooks (APP-CON-3).
7. RunArtifactBundle on ApplicationRunSummary (APP-CON-6).
8. Tier-3 scenario matrix / UC-A* evidence per reference host (APP-CON-7).
9. Workspace shadow/sandbox cleanup on lifespan (APP-CON-8).
10. EnvironmentSnapshot on intake + profile_snapshot_id (APP-EVOL-1).
11. ApplicationMigration CI + typed sub-migrations (APP-EVOL-2/2b).
12. CapabilityAlias sunset routing in STRICT (APP-EVOL-3).
13. AgentCertification on STRICT roster (APP-EVOL-4).
14. ApplicationRecoveryContract + ARCHITECTURE recovery docs (APP-EVOL-5).
15. ApplicationEnvironmentDiff + doctor diff-app (APP-EVOL-6).
16. ApplicationPackage + package.json from scaffold (APP-EVOL-7).
17. APP-EVOL-8 M1: nested profile bundles + flat property shims (ADR-APP-003).
18. bundle_normalized_payload on EnvironmentSnapshot digests (APP-EVOL-8.3).
19. ProfileInvariantValidator cross-bundle checks (APP-EVOL-8).
20. check_environment_profile_bundle_schema.py (APP-EVOL-8.7).
21. STRICT capability graph deploy gate + blast radius (APP-OPS-1).
22. ApplicationOperationalOwnership on product manifests (APP-OPS-2).
23. EnvironmentHealthScore + doctor health-app (APP-OPS-3).
24. ApplicationRegistry + EnvironmentRegistry + apps/envs CLI (APP-OPS-4).
25. check_application_production_gates.py aggregates APP-PROD + APP-CON + APP-EVOL + APP-OPS.
26. Roster ⊆ skill/tool profiles (EnvironmentSkillToolConsistencyCheck).
27. IdentityProfile + budget enforcement on STRICT product hosts.
28. Deploy triad present on scaffolded standard hosts.
29. APPLICATION_CREATION_GUIDE.md aligns with §31 · §45 · §47.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Cold start bootstrap all catalogs across four product hosts.
- Registry sync + health score for full STRICT fleet.
- Pre-deploy diff between manifest versions.
- strict_multi_agent_defaults() on legal/finance hosts.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

Full ApplicationEnvironmentProfile · ApplicationManifest · OrganizationalPolicyEnvelope per tenant · registry artifacts in build/

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

Compare against: **Reference hosts (legal, research, dispute_sim, local_workspace) · enterprise FastAPI agent host · ops registry + health score on release**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **niedoróbka** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Business pipeline in applications/host · README as ops registry · getattr wiring · Nexus fork per product · skipping production gates

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/applications/ -q
uv run python scripts/check_application_production_gates.py
uv run python scripts/check_application_registry.py
uv run python scripts/check_application_health_score.py
uv run python scripts/check_environment_profile_bundle_schema.py
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
