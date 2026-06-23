# Integration Library — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/INTEGRATIONS.md`](../architecture/INTEGRATIONS.md) · [`plan/INTEGRATIONS.md`](../plan/INTEGRATIONS.md)  
**Audit map layers:** 13 · compact slice: [`audit_slices/INTEGRATIONS.md`](../guides/audit_slices/INTEGRATIONS.md)  
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

domain: INTEGRATIONS
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Integration Library (`INTEGRATIONS`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Integration Library** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Integration Library** as the sole vendor boundary: 197+ slugs, typed contracts, health probes, IntegrationProfile-driven backend selection, guardrail integrations, and CI-enforced import boundaries.

## Key symbols and contracts

IntegrationManifest · IntegrationProfile · IntegrationCategory · IntegrationPlugin · LlmGuardrailBackend · GuardrailScanResult · RelationalStore · VectorStore · MessageBus · SearchProvider

## Active plan phases (verify status vs code reality)

Phase M catalog · M.6 P5/P6/P7 Done · M.12 guardrails Done · M-P12-CAT.1 · GR-DOC

## Known open gaps — re-validate every item (closed / still open / partial)

Most slugs **beta** — stable vs beta must be honest · thin P4 shells · SaaS-only without local container · nginx/ingress slug missing (ECP cross-ref)

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/INTEGRATIONS.md`](../guides/audit_slices/INTEGRATIONS.md) — compact slice (layers **13**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/INTEGRATIONS.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/INTEGRATIONS.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/INTEGRATIONS.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
5. `@docs/guides/AGENT_CREATION_GUIDE.md` **Appendix K (integration control plane)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/INTEGRATIONS.md` — then inspect:

```text
intergrax/integrations/ (contracts/, registry/, providers/)
intergrax/integrations/registry/harness_lab_stack.py · presets.py
intergrax/integrations/_shared/p2|p3|p4|p5|p6|p7|p8/factories.py
applications/_shared/integration_wiring.py · integration_runtime_bridge.py
applications/_shared/guardrail_wiring.py
scripts/check_integration_vendor_imports.py
scripts/check_harness_guardrail_wiring.py · scripts/generate_integration_usage_docs.py
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. No vendor SDK imports in agents/ or Nexus business logic.
2. Every slug in layout/registry with conformance tests where claimed stable.
3. IntegrationProfile drives backend selection — wired through bridges, not getenv in agents.
4. llm_guardrail via middleware + IntegrationProfile — not parallel tier or agent SDK.
5. Health probes for external deps; circuit breaker registry used.
6. Secrets via SecretsStore/integration options — not committed config.
7. RAG vector stores via catalog bridges — not duplicate vector clients in agents.
8. Guardrail layering L1→L4 documented and composed (ADR-GR-001).
9. Slack/Teams/etc. are adapters — not orchestrators replacing Nexus.
10. Cloud facades do not wrap LLM providers (LLM via llm_adapters/).
11. bootstrap_application_integration_catalog() used by Tier-3 hosts.
12. Harness lab stable stack smoke tests pass.
13. New provider has USAGE.md and manifest conformance.
14. Vendor imports only in allowed modules — CI check_integration_vendor_imports green.
15. Tier-3 extend_tool_profile_for_integration() pattern followed.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- HARNESS_M6_P5/P6/P7 probe slugs and health endpoints.
- Failover between providers (where documented).
- Rate limits and bulk operations on message_bus/data slugs.
- Compose profiles: lab_stack, harness_guardrail_stack, research_web_stack.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

IntegrationProfile + presets · per-slug options in profile · IntegrationPlugin (EXTENSION_AUTHOR_GUIDE) · wire_integration_tool_context()

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

Compare against: **Large integration catalogs (LangChain-style) · harness lab stable stack · NeMo/Guardrails AI/LLM Guard/Presidio (§47)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Agent-imported vendor SDK · duplicate adapter per product · guardrail as agent code · stable label on beta-only slug

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
python scripts/check_integration_vendor_imports.py
uv run python scripts/check_harness_guardrail_wiring.py
uv run pytest tests/unit/integrations/ -q
uv run python scripts/generate_integration_usage_docs.py
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
