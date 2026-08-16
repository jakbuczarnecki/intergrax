# Application Hosting — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/APPLICATION_HOSTING.md`](../architecture/APPLICATION_HOSTING.md) · [`plan/APPLICATION_HOSTING.md`](../plan/APPLICATION_HOSTING.md)  
**Audit map layers:** HOST · compact slice: [`audit_slices/APPLICATION_HOSTING.md`](../guides/audit_slices/APPLICATION_HOSTING.md)  
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

domain: APPLICATION_HOSTING
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Application Hosting (`APPLICATION_HOSTING`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Application Hosting** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **platform-owned Application Hosting subsystem**: turn any configured Tier-3 application into a continuously running, managed, observable, extensible instance — lifecycle, readiness, typed hooks/components/events, graceful shutdown, restart supervision, and OS adapters — without cognition, Nexus orchestration, or product-owned generic hosting.

## Key symbols and contracts

HostedApplicationProfile · HostedApplicationContext · HostedApplicationEngine · HostedApplicationComponent · HostedApplicationHooks · HostedApplicationSupervisor · HOST-INV-01..12

## Active plan phases (verify status vs code reality)

APP-HOST-0..2 Done · APP-HOST-W1/W2/W3 Done · APP-HOST-8A..8E LKW proof Done · APP-HOST-9A Done

## Known open gaps — re-validate every item (closed / still open / partial)

APP-HOST-3D Planned · APP-HOST-5D/5E Planned · APP-HOST-6A..6D Planned · APP-HOST-7A..7F Planned · APP-HOST-9B..9F Planned

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/APPLICATION_HOSTING.md`](../guides/audit_slices/APPLICATION_HOSTING.md) — compact slice (layers **HOST**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites/` or `architecture/satellites/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/APPLICATION_HOSTING.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/APPLICATION_HOSTING.md` — hub read-scope + one `architecture/satellites/` satellite max
3. `docs/project/maintainers/plans/APPLICATION_HOSTING.md` — hub + one `plan/satellites/` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/APPLICATION_HOSTING.md` — then inspect:

```text
intergrax/hosting/ (contracts, engine, supervisor, instance, control, runner)
intergrax/hosting/engine/ · intergrax/hosting/supervisor/ · intergrax/hosting/instance/
intergrax/applications/_shared/hosting_wiring.py
applications/local_workspace_application/hosting/  # LKW adoption/proof only — not generic engine
tests/unit/hosting/
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. HOST-INV-01: Application Hosting is platform-owned; products adopt it.
2. HOST-INV-02: Hosting never performs cognition or orchestration.
3. HOST-INV-03: Supervisor has no dependency on Task, NexusLoop, agents, tools, or product capabilities.
4. HOST-INV-04: Application code contains no standard Windows/Linux/macOS branching.
5. HOST-INV-05: HostedApplicationProfile is the primary public composition surface.
6. HOST-INV-11: ApplicationHost.on_hook and HostedApplicationHooks MUST NOT be merged.
7. HOST-INV-12: Restart creates a new instance/process lifecycle; hosted app does not exec itself.
8. No private hosting event bus — reuse Intergrax event/observability spine.
9. Required unhealthy components block readiness; optional components do not.
10. LKW is a proof workload, never the owner of generic hosting contracts.
11. Generic contracts/engine MUST NOT live under applications/local_workspace_application.
12. Existing FastAPI/Tier-3 applications continue without adopting hosting.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Single-instance rejection (INSTANCE_CONFLICT) under concurrent start.
- Supervisor restart creates new instance_id with preserved profile digest.
- LKW live request succeeds after restart (APP-HOST-8D).

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

HostedApplicationProfile · run_hosted_application(profile) · OS adapters (APP-HOST-7 target) · LKW hosted profile

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

Compare against: **OS service managers (systemd/launchd/Windows Service) · process supervisors with restart policy · readiness/health aggregation on long-running hosts**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Product-owned generic daemon framework · supervisor calling Nexus/tasks/tools · private hosting event bus · restart via os.exec · LKW proof as sole platform contract test

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/hosting/ -q
uv run pytest tests/unit/hosting/test_hosting_import_boundaries.py -q
uv run pytest tests/unit/hosting/engine/test_engine_import_boundaries.py -q
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
