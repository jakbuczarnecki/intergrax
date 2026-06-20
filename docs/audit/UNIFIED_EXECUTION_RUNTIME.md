# Unified Execution Runtime (UAEP) — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/UNIFIED_EXECUTION_RUNTIME.md`](../architecture/UNIFIED_EXECUTION_RUNTIME.md) · [`plan/UNIFIED_EXECUTION_RUNTIME.md`](../plan/UNIFIED_EXECUTION_RUNTIME.md)  
**Audit map layers:** 4–5, 8, 23–24 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
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
Table of contents + sections for audit-map layers **4–5, 8, 23–24** + registers tied to **Known open gaps**. Skip historical paydown logs unless a gap ID points there.

### Scoped guide reads
- `IDEAL_HARNESS_AI_ARCHITECTURE.md` — sections for layers **4–5, 8, 23–24** only
- `INTEGRAX_HARNESS_AUDIT_MAP.md` — layers **4–5, 8, 23–24** + maturity §5 only
- `SYSTEM_INVARIANTS.md` — skim invariant IDs referenced in section 3 dimensions only

---


## 1. Canonical reads (scoped — in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — **layers 4–5, 8, 23–24 only** (see §0)
2. `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` — **scoped sections** (see §0)
3. `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` — **scoped sections only** (see §0) — do **not** load the full file
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — **layers 4–5, 8, 23–24** + §5 maturity
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **Appendix H (governance control plane)**

---

## 2. Code and test paths (inspect — search repo, do not assume)

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

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

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
python scripts/check_harness_no_getattr.py
uv run python scripts/check_observability_gates.py
uv run pytest -m gate -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/UNIFIED_EXECUTION_RUNTIME.md` gap rows + `docs/architecture/UNIFIED_EXECUTION_RUNTIME.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
