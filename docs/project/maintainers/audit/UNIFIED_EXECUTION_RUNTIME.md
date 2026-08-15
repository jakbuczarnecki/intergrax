# Unified Execution Runtime (UAEP) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../../architecture/UNIFIED_EXECUTION_RUNTIME.md) · [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plans/UNIFIED_EXECUTION_RUNTIME.md)
**Audit map layers:** 4–5, 8, 23–24 · compact slice: [`audit_slices/UNIFIED_EXECUTION_RUNTIME.md`](../../technical/guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md)
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

domain: UNIFIED_EXECUTION_RUNTIME
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Unified Execution Runtime (UAEP) (`UNIFIED_EXECUTION_RUNTIME`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Unified Execution Runtime (UAEP)** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit the **Agent OS execution substrate**: policy-first UAEP, typed runtime events, identity/trust propagation, security redaction, cost governance, checkpoint/pause/resume, and delegation — on **every** runtime path with no policy bypass.

## Key symbols and contracts

RuntimeEvent/RuntimeEventType · ExecutionPhase · HookPoint/HookContext · AgentDecision · ExecutionInterrupt · PauseRecord · RuntimeExecutionContext · RuntimePolicyBundle · PolicyDecision · ToolRequest/ToolResponse · AgentStep · ValidationResult · MemoryView · DelegationSpec · ApplicationSecurityProfile · GuardrailProfile

## Active plan phases (verify status vs code reality)

R-Policy Done · R-Delegate Done · V-REM-SEC · SEC · COST · GR-DOC · REL-ADV autonomy

## Known open gaps — re-validate every item (closed / still open / partial)

HTTP mid-run autonomy mostly lab-only · supervisor EscalationRouter future · middleware target layout partially evolved

---

## 0. Context budget (mandatory)

**Load first:** [`docs/project/technical/guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md`](../../technical/guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md) — compact slice (layers **4–5, 8, 23–24**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/satellites` or `architecture/satellites`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results` unless RESUME

---


## 1. Canonical reads (order)

1. **`docs/project/technical/guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md` — hub read-scope + one `architecture/satellites` satellite max
3. `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md` — hub + one `plan/satellites` satellite max
4. `docs/project/maintainers/audit/README.md` — shared production Harness checklist
5. `@docs/project/technical/guides/AGENT_CREATION_GUIDE.md` **Appendix H (governance control plane)** — on demand
**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/project/technical/guides/audit_slices/UNIFIED_EXECUTION_RUNTIME.md` — then inspect:

```text
intergrax/runtime/nexus/nexus_loop.py · unified_task_runner.py
intergrax/agents/agent_engine.py · intergrax/agents/uaep.py
intergrax/runtime/nexus/tools/tool_runtime.py
intergrax/runtime/policy/policy_engine.py
intergrax/runtime/events/ (runtime_event.py, phase_coverage.py, unified_run_journal.py)
intergrax/runtime/middleware/ · intergrax/runtime/architecture/ (prompt_security, tool_security, tenant_security, retrieval_security, cost_budget, cost_quota)
intergrax/runtime/schema/registry.py
applications/_shared/runtime_config_bridge.py · identity_wiring.py · guardrail_wiring.py
```

Grep `tests/unit`, `tests/integration`, `tests/acceptance` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. Single path: UnifiedTaskRunner → NexusLoop → AgentEngine → UAEP steps — no parallel legacy engines.
2. Every AgentStep emits STEP_* events with trace_id/run_id/tenant_id.
3. ToolRuntime.invoke emits TOOL_* events — all tool paths, including catalog dispatch.
4. PolicyEngine: pre-run, pre-plan, pre-LLM, pre-tool, post-tool, pre-output, memory writes.
5. RuntimePolicyBundle is the single policy composition object — no orphan policy dicts.
6. AgentDecision emitted **before** Nexus acts on model output.
7. Retry managed by runtime (RetryEngine) — not unbounded agent while-loops.
8. MemoryView is the agent memory interface — no direct store access from Tier-2.
9. Delegation uses DelegationSpec with scoped permissions — child cannot inherit all parent tools.
10. tenant_id on events; secrets redacted in traces (ApplicationSecurityProfile).
11. Guardrail middleware (llm_guardrail) composes via IntegrationProfile — not agent SDK.
12. Checkpoint/pause/resume uses RuntimeCheckpoint — recoverable UAEP cursor.
13. schema_version validated on runtime contracts.
14. HITL via REQUEST_HUMAN / policy — not ad-hoc Slack in agent code.
15. Cost budgets enforced (max_cost, token metering hooks).
16. Hooks (HookRegistry) do not call vendor adapters directly.
17. Forbidden: agent-specific Nexus branches; duplicate policy engines.

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- PM→UX→Legal→Validator→Human multi-agent chain (§42.43).
- Budget exhaustion mid-run (max_steps, max_cost).
- Cooperative cancel at step boundaries.
- Large delegation trees with permission scope audit.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

RuntimePolicyBundle via runtime_config_bridge · ApplicationSecurityProfile · GuardrailProfile · HookRegistry · RuntimePlugin · TaskExecutionOptions.autonomy_level

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

Compare against: **UAEP-class agent runtimes · NeMo Guardrails / Guardrails AI / LLM Guard as integration backends (§42.11.6)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Policy in docs only · LLM calls bypassing policy · context assembly bypass · untraced policy decisions

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run pytest tests/unit/runtime/ -q
python scripts/maintenance/check_harness_no_getattr.py
uv run python scripts/maintenance/check_observability_gates.py
uv run pytest -m gate -q
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
