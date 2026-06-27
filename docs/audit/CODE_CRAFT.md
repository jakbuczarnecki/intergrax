# Ephemeral Code Craft — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/CODE_CRAFT.md`](../architecture/CODE_CRAFT.md) · [`plan/CODE_CRAFT.md`](../plan/CODE_CRAFT.md)  
**Audit map layers:** 11b · compact slice: [`audit_slices/CODE_CRAFT.md`](../guides/audit_slices/CODE_CRAFT.md)  
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with **full repository access**.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: CODE_CRAFT
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Ephemeral Code Craft (`CODE_CRAFT`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Ephemeral Code Craft** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **Ephemeral Code Craft runtime** at L3+: verify `codecraft.*` tools, `wire_application_codecraft()`, orchestrator loop, sandbox tiers, policy gates, CVL integration, ephemeral tool registry hygiene — confirm Done vs depth backlog.

## Key symbols and contracts

CodeCraftProfile · CodeCraftOrchestrator · CodeCraftSession · CraftResult · IterationRecord · StaticCodeGate · craft modes (disabled|dry_run|assist_only|supervised|autonomous) · EphemeralToolRegistry · wire_application_codecraft

## Active plan phases (verify status vs code reality)

ECC-0…ECC-6 + S7–S11 **Done** (L3+, 2026-06-13) · ADR-CODECRAFT-001

## Known open gaps — re-validate every item (closed / still open / partial)

GAP-ECC-20…23 **Closed** (ECC-MAINT-01..04) · local SandboxSession ≠ OS containment (accepted) · dedicated container runtime backend product opt-in beyond local fallback

---

## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/CODE_CRAFT.md`](../guides/audit_slices/CODE_CRAFT.md) — compact slice (layers **11b**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/CODE_CRAFT.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/CODE_CRAFT.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/plan/CODE_CRAFT.md` — hub + one `plan/satellites/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
5. `@docs/guides/AGENT_CREATION_GUIDE.md` **Appendix J (tool surfaces)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/CODE_CRAFT.md` — then inspect:

```text
intergrax/codecraft/ · intergrax/runtime/codecraft/
intergrax/runtime/sandbox/
intergrax/tools/providers/sandbox/ · intergrax/tools/providers/codecraft/
intergrax/applications/_shared/codecraft_wiring.py
intergrax/runtime/critic/ (CVL hooks)
docs/architecture/CODE_CRAFT.md · docs/plan/CODE_CRAFT.md · ADR-CODECRAFT-001
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. CodeCraft uses existing sandbox ToolRuntime path — no parallel execution stack.
2. L0 StaticCodeGate before any execute in autonomous/supervised paths.
3. Codegen LLM separated from producer/judge LLM identity (template adapter shipped; profile ref → backlog).
4. Ephemeral tools do not persist in global ToolRegistry after session.
5. CraftResult promotion typed — not stdout-only.
6. Fail-closed when codecraft_profile missing or mode=disabled.
7. CODECRAFT_* events correlated with trace_id/run_id.
8. Tier-2 invokes only codecraft.* / sandbox.* catalog tools.
9. Network egress policy enforced per sandbox tier.
10. CVL L0/L1 integrated — not parallel verification stack.
11. Modes table §6.3 respected (supervised vs autonomous).
12. Resource disposal releases craft_id / sandbox session.
13. cloud substrate (e2b/modal/daytona) via IntegrationProfile — not agent SDK.
14. max_total_exec_time_s enforced on session iteration paths.
15. Document honest L3 maturity — depth backlog only for metrics/container/codegen LLM.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- generate→gate→exec→test→CVL iteration within max_iterations.
- max_code_bytes and max_total_exec_time_s enforcement.
- Concurrent codegen sessions without registry pollution.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

ApplicationEnvironmentProfile.codecraft_profile · Task.metadata.codecraft_mode · sandbox_host_slug · codegen_llm_profile_ref · require_hitl_before_exec

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

Compare against: **Cursor ephemeral codegen · E2B/Modal sandboxes · CI codegen with semgrep/trivy gates**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Claiming ECC Planned when runtime shipped · arbitrary exec bypassing ToolRuntime · global registry pollution · local workspace labeled as OS sandbox

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
python scripts/maintenance/check_codecraft_layer.py
uv run pytest tests/unit/codecraft/ tests/unit/tools/providers/codecraft/ tests/unit/runtime/codecraft/ -q
uv run pytest tests/unit/runtime/sandbox/ -q
python scripts/maintenance/check_harness_no_getattr.py
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
