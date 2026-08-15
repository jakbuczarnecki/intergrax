# Experimentation and Developer Experience — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) · [`plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)
**Audit map layers:** 25–27, 30 · compact slice: [`audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../technical/guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md)
**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

---

## How to use

1. Open a new agent chat with the repository available, but do not perform broad repository exploration. Read only the files listed in Context budget / Canonical reads, use path-filtered grep before opening files, and do not use semantic search, subagents, or full-repo scans unless the operator explicitly approves.
2. Copy from `---BEGIN PROMPT---` through `---END PROMPT---`.
3. Edit **USER CONFIG** only (`mode`, optional `focus` slice).
4. The agent must **read code, run tests, and re-validate known gaps** — not survey documentation alone.
5. Output: [`HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md`](../HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) §7–§8.

Regenerate after architecture/plan changes: `uv run python scripts/audit/generate_domain_audit_prompts.py`

---

---BEGIN PROMPT---

# ═══ USER CONFIG ═══

domain: EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Experimentation and Developer Experience (`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Experimentation and Developer Experience** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **DX and experimentation**: scaffold, eval harness, CI architecture gates, doctor CLI, lab environment, W-OPS evidence, TTFRun <1h goal, and gate maintenance discipline.

## Key symbols and contracts

EvaluationProfile · ExperimentSession · OnlineEvaluationRegistry · maturity gate evidence · TTFRun metric · shadow workspace bindings

## Active plan phases (verify status vs code reality)

EVAL · CRIT-V cross-ref · MVP-EVOL · DX · AA · W-OPS · Phase V G5 Production PRR

## Known open gaps — re-validate every item (closed / still open / partial)

DX-LC Done · §6.1av DX-MAINT Done · GOV-PROD.1 dashboard backlog · polished SaaS UI explicit non-goal

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](../../technical/guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) — compact slice (layers **25–27, 30**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites` or `architecture/satellites`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — hub read-scope + one `architecture/satellites` satellite max
3. `docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — hub + one `plan/satellites` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **EXTENSION_AUTHOR_GUIDE** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md` — then inspect:

```text
intergrax/scaffold/
intergrax/runtime/architecture/ (eval, maturity gates, online_evaluation_registry.py)
intergrax/experiments/ · nexus_eval_runner.py
scripts/check_*.py (harness gates) · scripts/ci/test.bat
scripts/release/phase_v_closeout_gate.py · phase_w_ops_evidence.py
docs/project/technical/guides/AGENT_CREATION_GUIDE.md · HARNESS_ENVIRONMENT.md
```

Grep `tests/unit`, `tests/integration`, `tests/acceptance` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Scaffold new-agent runnable through Nexus — not standalone script only.
2. new-application emits profile+wiring+docker+ADR per Phase N.
3. intergrax doctor diagnoses lab stack accurately.
4. Gate scripts pass after harness change (mandatory verification set).
5. check_docs_domain_pairs enforces 22 pairs.
6. Eval registry trends before promotion (require_baseline_for_release).
7. Shadow workspace observe-only compare path works.
8. Acceptance agent_os suite covers OS claims.
9. Extension author guide aligned with plugin entry points.
10. Phase V PRR evidence for production readiness claims.
11. Structured output required on agent contracts per guide.
12. Trace on every decision per DX checklist.
13. Tier-0 reused in scaffold — not duplicated stubs.
14. W-OPS release cycle docs match scripts/build artifacts.
15. Single plan pair per domain — no orphan implementation docs.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Full CI gate suite runtime on developer machine.
- Parallel eval workloads in lab.
- TTFRun: idea → first Nexus run timing evidence.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

Scaffold templates · EvaluationProfile.shadow_eval_enabled · lab vs strict production defaults

---

## 6. Cross-cutting checklist (mandatory)

Apply **every** section in `docs/project/maintainers/audit/README.md` §Shared production Harness checklist:

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

Compare against: **Cursor agent iteration UX · Braintrust/prompt regression CI · platform engineering PRR culture**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Skipping gates after harness change · duplicate DX docs · eval only in notebooks not registry · false PRR without evidence

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest -m gate -q
python scripts/maintenance/check_harness_no_getattr.py
uv run python scripts/maintenance/check_observability_gates.py
uv run python scripts/audit/check_docs_domain_pairs.py
scripts/ci/test.bat unit
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
