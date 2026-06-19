# Agent Contracts and Assembly — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md) · [`plan/AGENT_CONTRACTS_AND_ASSEMBLY.md`](../plan/AGENT_CONTRACTS_AND_ASSEMBLY.md)  
**Audit map layers:** 17–20, 31 · ACP §21 · [`INTEGRAX_HARNESS_AUDIT_MAP.md`](../INTEGRAX_HARNESS_AUDIT_MAP.md)  
**ADR:** [`ADR-AGENT-001`](../../adr/ADR-AGENT-001.md) · [`ADR-AGENT-002`](../../adr/ADR-AGENT-002.md) · [`ADR-AGENT-003`](../../adr/ADR-AGENT-003.md)  
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

domain: AGENT_CONTRACTS_AND_ASSEMBLY
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — Agent Contracts and Assembly (`AGENT_CONTRACTS_AND_ASSEMBLY`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **Agent Contracts and Assembly** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

Audit **AgentContract**, registry resolution, **Prompt Registry**, capability graph, agent lifecycle governance, **ACP cognitive patterns**, **author run() facade** (ADR-AGENT-001/002), **step loop on_next_step** and **dual observability** (ADR-AGENT-003) — Tier-2 hooks + environment merge; Nexus remains Agent OS for Task.

## Key symbols and contracts

AgentContract · UAEPAgent · RuntimeExecutionContext · AgentDecision · CognitiveAgent · acp.state.v1 · IntergraxAgent · PromptMeta · AgentStepContext · StepOutcome · AgentRunTrace · ApplicationRunSummary

## Active plan phases (verify status vs code reality)

ACP · ACP-CLOSE · ACP-FINISH Done (2026-06-13) · PE/REG/CG/AS closed · AUDIT-IDEAL-19.1/20.1/31.1 parallel

## Known open gaps — re-validate every item (closed / still open / partial)

GAP-ACP-36/37 Closed (ACP-TOK-*) · GAP register 37 Closed · 0 Open · AUDIT-IDEAL-19.1/20.1/31.1 Planned · COST-1 RunBudget Partial

---

## 1. Canonical reads (in order)

1. `docs/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md` — target state
2. `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` — architecture canon (incl. audit registers if present)
3. `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` — implementation plan and gap IDs
4. `docs/guides/INTEGRAX_HARNESS_AUDIT_MAP.md` — layers 17–20, 31 · ACP §21
5. `docs/audit/README.md` — shared production Harness checklist (**mandatory**)
6. `docs/guides/AGENT_CREATION_GUIDE.md` **ADR-AGENT-001 · ADR-AGENT-002 · ADR-AGENT-003 · Appendix M/N/O/P · Appendix AC**

---

## 2. Code and test paths (inspect — search repo, do not assume)

```text
intergrax/contracts/agent_contract_meta.py · runtime_execution_context.py
intergrax/agents/agent_engine.py · uaep.py · uaep_protocol.py · authoring/
intergrax/agents/authoring/patterns/  [ACP]
intergrax/agents/authoring/step_loop.py · acp_run.py  [ACP-STEP Done]
intergrax/contracts/agent_run_trace.py · shared_context.py  [ACP-OBS/STATE Done]
intergrax/agents/persistence/  [ACP-PROD checkpoint · declarative tools]
intergrax/runtime/registry/agent_registry.py
intergrax/prompts/registry/ (YamlPromptRegistry)
intergrax/runtime/architecture/capability_graph*.py · agent_lifecycle_governance.py
intergrax/runtime/nexus/tools/tool_loop.py  [ACP tool loop]
agents/ (Tier-2 roster) · applications/_shared/prompt_wiring.py
scripts/check_agents_lifecycle_metadata.py · check_agents_vendor_imports.py
```

Also grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

1. AgentContract has required fields per §12 — capabilities, allowed_tools, risk metadata.
2. UAEPAgent: get_steps/run_step — AgentEngine path, not private HTTP bypass.
3. decide_after_step returns typed AgentDecision — not ad-hoc control flow.
4. Nexus routes by capability token — not Python class name.
5. ADR-AGENT-001 Accepted; architecture §21–§36 ACP + run/step canon present.
6. Three cognition planes (§23) — no private multi-agent graph inside run_step (ACP-AP-01).
7. Tool calls via RuntimeExecutionContext.invoke_tool / ToolRuntime only.
8. Agents control loop via on_next_step only — no Tier-2 RuntimeEngine/pipeline (ACP-CLOSE-LEG-5).
9. CognitiveAgent base exists or gap ACP-1 recorded.
10. Pattern classes Reflex/ReAct/PlanExecute/Decomposition/Reflection vs ACP-2..6.
11. acp.state.v1 schema and cognitive_pattern on contract (ACP-0/0b).
12. ReActAgent iteration budget aligns with TOOL-ENG-6 when both Done.
13. ReflectionAgent uses CVL critic hooks — no critic SDK in Tier-2.
14. Config split: Tier-3 profile vs agent domain — not all config in agent class (ACP-AP-03).
15. Prompt templates have ownership, version, layered compilation.
16. Capability graph edges reflect manifest roster with lineage.
17. Registry snapshot conformance tests pass CI.
18. Deprecated/retired agents rejected in strict production_mode.
19. Agent checklist §45 + ACP pattern selection (§26.1).
20. Forbidden §42.41 patterns absent (vendor SDK, direct integrations).
21. skill_ids → allowed_tools resolution audited.
22. scaffold --pattern when ACP-8 Done.
23. check_agent_pattern_conformance.py when ACP-13 Done.
24. acceptance agent_os covers UAEP path for reference agents.
25. AgentRunRequest/Result and merge_environment per §29–§30 (ACP-DX).
26. Per-agent memory_namespace and rag_collection — not global store.
27. Application metadata → environment_overrides wired in hosts.
28. on_next_step / StepOutcome author API per §32 (ACP-STEP-1).
29. execute_next_step harness-only — authors cannot override (ACP-STEP-2).
30. HarnessKernel.execute_step deterministic primitive — no agent planning §38 (ACP-STEP-2b).
31. NexusLoop vs HarnessKernel separation §38 — not nexus.run() as agent brain.
32. AgentRunTrace on AgentRunResult with tool/RAG/LLM step records §31 (ACP-OBS-1).
33. ApplicationRunSummary for Task orchestration §31 (ACP-OBS-2).
34. StepLLMRouter per-step model within LLMProfile §33 (ACP-LLM-1).
35. SharedContextView for multi-agent handoffs §34 (ACP-STATE-1).
36. Use-case catalog UC-1..10 supported without agent rewrite §35.
37. AgentRunErrorCode and TerminalReason enums per §37.4–§37.5 (ACP-CON-1).
38. state_delta JSON merge-patch + _version + resume conflict §37.2 (ACP-CON-2).
39. Side-effect mode immediate vs declarative — no mix per step §32.8 (ACP-CON-3).
40. Capability routing by token not class name §37.6 (ACP-CON-6).
41. Security guards STRICT tool/memory/RAG §37.7 (ACP-CON-7).
42. OrganizationalPolicyEnvelope constrains agents without code fork §39 (ACP-ORG).
43. PolicyVerdictRecord on steps for compliance measurement §39.5 (ACP-ORG-4).
44. Checkpoint/resume/replay semantics §40.1 (ACP-PROD-1).
45. Side-effect idempotency keys and dedupe §40.2 (ACP-PROD-2).
46. ToolExecutionProfile mutability/compensation §40.3 (ACP-PROD-3).
47. SharedContextView CAS concurrency §40.5 (ACP-PROD-5).
48. ArtifactRef typed contract §40.6 (ACP-PROD-6).
49. Agent threat model mitigations §40.7 (ACP-PROD-7).
50. Privacy/redaction on trace/memory §40.8 (ACP-PROD-8).
51. Release eval gates before production_mode §40.9 (ACP-PROD-9).
52. CI conformance matrix §40.10 (ACP-PROD-10).
53. Contract schema_version migration §40.11 (ACP-PROD-11).
54. RequestIdentity tenant_id/user_id and memory_scope user vs org §30.9 (ACP-DX-1/2).

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

- Large agent roster with capability-based routing.
- Registry snapshot at bootstrap vs runtime mutation.
- Promotion dev→staging→prod evidence chain.
- ReActAgent at max_react_iterations — FAIL vs REQUEST_HUMAN behavior.
- DecompositionAgent deep sub-question tree — budget + acp.state.v1 checkpoint.
- Same agent class in two Tier-3 hosts with different ToolProfile/LLMProfile.

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

PromptProfile · ToolProfile · LLMProfile · OrchestrationProfile · ApplicationGraphSpec · cognitive_pattern/pattern_config (ACP-0b) · AgentRegistry.register · wire_application_environment · scaffold --pattern (ACP-8)

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

Compare against: **Enterprise agent registries · LangGraph/ADK pattern libraries · Cursor-style decomposition · prompt governance · capability routing (service-mesh analogy)**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

- Hardcoded agent class routing · vendor SDK in Tier-2 · orphan prompts · skipping lifecycle · ACP-AP-01..07 (fat agent absorbs Nexus, multi-agent in run_step, secrets in agent source)

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
uv run python scripts/check_agents_lifecycle_metadata.py
uv run python scripts/phase_v_capability_graph_guard.py
uv run python scripts/check_agents_vendor_imports.py
uv run pytest tests/acceptance/agent_os -m agent_os -q
uv run pytest tests/unit/agents/ -q
uv run pytest tests/unit/agents/authoring/patterns/ -q
```

Add any domain-specific scripts you discover. If a command fails, state why.

---

## 11. Output and mode rules

- Use `HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md` §7 Audit Result template.
- End with §8 Completion Summary.
- **`audit-only`:** no file edits.
- **`audit-and-fix`:** update `docs/plan/AGENT_CONTRACTS_AND_ASSEMBLY.md` gap rows + `docs/architecture/AGENT_CONTRACTS_AND_ASSEMBLY.md` audit register; map findings to plan phase IDs; **no code** unless user requests separately.
- Out-of-scope findings → suggest next `audit/<DOMAIN>.md`.

Begin the audit now.

---END PROMPT---
