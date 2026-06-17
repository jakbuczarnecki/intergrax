# Architecture Audit — Orchestration (audit-only, all domains)

**Status:** Canonical orchestrator for **Mode A** — audit + plan sync, **no implementation**  
**Bootstrap (paste):** [`bootstrap/01_audit_all_domains.txt`](bootstrap/01_audit_all_domains.txt)  
**Single domain:** [`bootstrap/02_audit_one_domain.txt`](bootstrap/02_audit_one_domain.txt)  
**Related:** [`ORCHESTRATOR.md`](ORCHESTRATOR.md) (this file) · [`LAYER_COMPLETION_MODE.md`](../LAYER_COMPLETION_MODE.md) · [`implementation-journal/`](../implementation-journal/README.md)

---

## Purpose

Run a **repeatable, evidence-backed audit** across all **22 harness domain pairs** without pasting per-domain instructions.

- **One Cursor session bootstrap** → agent iterates domain-by-domain.
- **One iteration unit** = one domain pair (full audit + result artifact + optional plan row updates).
- **No code changes** in this mode unless the operator explicitly approves a P0 doc fix.

---

## Modes (pick one bootstrap file)

| Mode | Bootstrap | Canonical doc | Code? | Plan updates? |
|------|-----------|---------------|-------|----------------|
| **A — Audit all** | `01_audit_all_domains.txt` | This file | No | Yes (backlog rows only) |
| **A1 — Audit one** | `02_audit_one_domain.txt` | This file | No | Yes |
| **B — Implement plans all** | `03_implement_plan_all_domains.txt` | [`IMPLEMENT_ORCHESTRATOR.md`](IMPLEMENT_ORCHESTRATOR.md) | Yes | Already in plan |
| **B1 — Implement one** | `04_implement_plan_one_domain.txt` | [`IMPLEMENT_ORCHESTRATOR.md`](IMPLEMENT_ORCHESTRATOR.md) | Yes | Already in plan |

Mode **B** uses the same iteration discipline as Mode A (atomic domain unit, no micro-pauses, continue without asking between domains) — see [`IMPLEMENT_ORCHESTRATOR.md`](IMPLEMENT_ORCHESTRATOR.md) §Iteration discipline. Resume across sessions when one Cursor turn does not finish all 22 pairs.
| **C — Layer closeout** | `05_closeout_all_domains.txt` | [`LAYER_COMPLETION_ORCHESTRATOR.md`](LAYER_COMPLETION_ORCHESTRATOR.md) | If P0/P1 | Yes + architecture |

---

## Domain order (22 pairs — do not reorder without reason)

1. `PLATFORM_FOUNDATION`
2. `UNIFIED_EXECUTION_RUNTIME`
3. `ORCHESTRATION`
4. `NEXUS_EXECUTION_FLOW`
5. `REASONING_AND_COGNITION`
6. `AGENT_CONTRACTS_AND_ASSEMBLY`
7. `LLM_ADAPTERS`
8. `TOOLS`
9. `CODE_CRAFT`
10. `SKILLS`
11. `INTEGRATIONS`
12. `RAG`
13. `MEMORY`
14. `CONTEXT_ENGINEERING`
15. `MODALITY`
16. `OBSERVABILITY`
17. `RELIABILITY_FAILURE_AND_HITL`
18. `CRITIC_VERIFICATION`
19. `ADAPTIVE_HARNESS_INTELLIGENCE`
20. `ELASTIC_CAPACITY_AND_SCALING`
21. `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`
22. `TIER3_APPLICATION_ENVIRONMENT`

Audit map (32 layers) → domain: [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md).

---

## Before the first iteration

```bash
uv run python scripts/init_architecture_audit_run.py --date YYYY-MM-DD --mode audit_only
```

Or let the agent create `docs/guides/audit/results/YYYY-MM-DD/` on first run.

**Skim once per session:** [`SYSTEM_INVARIANTS.md`](../SYSTEM_INVARIANTS.md).

---

## Per-domain iteration (Mode A)

For current `<DOMAIN>`:

1. **Read only:** `docs/architecture/<DOMAIN>.md` · `docs/plan/<DOMAIN>.md` · `docs/guides/audit/<DOMAIN>.md`
2. **Skim:** `SYSTEM_INVARIANTS.md` (do not load other domain pairs).
3. **Execute** the domain audit prompt (`audit/<DOMAIN>.md` — inspect code, tests, CI gates).
4. **Classify** findings P0–P4 ([`LAYER_COMPLETION_MODE.md`](../LAYER_COMPLETION_MODE.md) §Step 3).
5. **Write** `results/<run_id>/<DOMAIN>.md` from [`TEMPLATE_DOMAIN_RESULT.md`](TEMPLATE_DOMAIN_RESULT.md).
6. **Plan sync (allowed):**
   - P2–P4 → add/update rows in `docs/plan/<DOMAIN>.md` backlog register.
   - P0/P1 → add rows + set `p0_open` / `p1_open` in `progress.json`; **do not implement**.
   - **Do not** edit `docs/architecture/<DOMAIN>.md` unless operator approved contract change (note `needs_architecture_sync` in result).
7. **Update** `results/<run_id>/progress.json` — `status: completed`, verdict, counts.
8. **Checkpoint** — one short operator summary; **continue** to next domain (no mid-domain questions).

**Safety fuse:** plan ↔ architecture ↔ ideal conflict → **STOP**, report operator, set domain `blocked`.

---

## Iteration discipline

| Rule | Meaning |
|------|---------|
| **Atomic unit** | Finish the **entire** current domain before starting the next |
| **No micro-pauses** | Do not ask operator every 2 steps inside a domain |
| **Between domains** | Brief checkpoint; continue unless operator wrote `pause` / `stop` |
| **Commits** | Only when operator explicitly requests |
| **Tier boundaries** | [`AGENTS.md`](../../AGENTS.md) — never violate |

---

## Resume across Cursor sessions

1. Paste the **same** bootstrap file.
2. Add one line: `RESUME: docs/guides/audit/results/YYYY-MM-DD/progress.json`
3. Agent reads `current_domain` and continues — **do not restart from pair #1**.

---

## Run completion

When all domains in `domain_order` are `completed`:

1. Write rollup in `results/<run_id>/RUN_SUMMARY.md`.
2. Set `completed_at` and `current_domain: null` in `progress.json`.
3. Run `uv run python scripts/check_architecture_audit_run.py <run_id> --require-complete`.

---

## Verification (after implementation modes only)

Mode A audit-only: run domain-relevant gate scripts cited in `audit/<DOMAIN>.md` — record commands in result file. Full harness bundle from [`AGENTS.md`](../../AGENTS.md) is **not** required every domain.

---

## ADR

No ADR for this orchestrator unless process semantics change platform contracts.
