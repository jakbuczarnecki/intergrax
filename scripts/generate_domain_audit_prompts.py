# © Artur Czarnecki. All rights reserved.
"""Generate docs/audit/<DOMAIN>.md prompt files. Idempotent."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "audit"

# Each domain: canon-aligned audit instruction. Regenerate after architecture/plan contract changes.
DOMAINS: list[dict] = [
    {
        "id": "PLATFORM_FOUNDATION",
        "title": "Platform Foundation",
        "layers": "1–2, 32",
        "mission": (
            "Verify Intergrax is developed as a **Harness AI / Agent OS** — the runtime is the durable "
            "product, agents are replaceable — with enforced four-tier boundaries, 22 domain-pair documentation "
            "governance, gate maintenance discipline, and strategic alignment to IDEAL_HARNESS_AI_ARCHITECTURE."
        ),
        "code": """docs/intergrax_runtime_architecture.md (hub)
docs/architecture/PLATFORM_FOUNDATION.md · docs/plan/PLATFORM_FOUNDATION.md
AGENTS.md · .cursor/rules/intergrax-iteration.mdc
scripts/check_intergrax_no_applications_imports.py
scripts/check_agents_no_tier3_imports.py
scripts/check_docs_domain_pairs.py
scripts/check_harness_no_getattr.py
scripts/phase_v_capability_graph_guard.py
intergrax/applications/reference/harness_manifest_catalog.py
Sample imports across intergrax/, agents/, applications/ for tier violations""",
        "key_symbols": "Four-tier model · IntegrationProfile/ToolProfile/SkillProfile/LLMProfile · ApplicationEnvironmentProfile · ApplicationManifest · RuntimePolicyBundle · AgentContract · plugin entry points (intergrax.tools, intergrax.skills, intergrax.integrations)",
        "active_phases": "§6.1 gate maintenance queue · Phase V architecture hardening · Phase K business agents (**deferred** — must not start silently) · §6.3 product backlog",
        "known_gaps": "Phase K / §6.3 deferred product work · long-term §50 marketplace/visual builder · codecraft/ incremental · unified tool model (legacy boolean flags deprecated)",
        "plan_read_scope": (
            "**Hub only** (`docs/plan/PLATFORM_FOUNDATION.md`): §4 ladder · §6.1 maintenance · §6.3 deferred product · satellite index. "
            "**On demand:** [`plan/plan/PLATFORM_FOUNDATION_master_registers.md`](plan/plan/PLATFORM_FOUNDATION_master_registers.md) (gap IDs) · "
            "[`plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md`](plan/plan/PLATFORM_FOUNDATION_06_closed_queues.md) (re-validate closed items only)"
        ),
        "dimensions": [
            "Harness treated as durable product — not single-agent optimization (§1 strategic frame).",
            "Tier-0 (`intergrax/`) contains only universal mechanisms — no business agent logic.",
            "Tier-1 Nexus domain-agnostic — no agent-specific branches in NexusLoop.",
            "Tier-2 agents consume Tier-0 via policy/ToolRuntime — no vendor SDK imports.",
            "Tier-3 applications compose runtime+agents+profiles — no duplicated agent pipelines.",
            "Import boundaries enforced: `intergrax/` ↛ `agents/`/`applications/`; agents ↛ applications.",
            "Documentation model: hub-only `docs/` root; 22 architecture↔plan pairs 1:1; no monolithic plan.",
            "New capabilities reuse Tier-0 (§5.2.2) — no parallel universal mechanisms.",
            "LLM calls via `llm_adapters/` — not Integration Library vendor wrappers.",
            "Integrations register via manifest/`register_from_manifest` — not ad-hoc SDK in agents.",
            "Gate maintenance §6.1 rows match evidence (tests, CI scripts, doc updates).",
            "Scaffold (`new-agent`, `new-application`, `new-stack`) emits tier-correct artifacts + ADR folders.",
            "Capability graph seeding uses `harness_manifest_catalog` — not orphan registrations.",
            "`getattr`/reflection banned outside approved bridges — CI green.",
            "Phase K / business agents not started without explicit operator reprioritization.",
            "Architecture governance loop: audits update paired docs, ADRs, plan registers — not chat-only.",
        ],
        "scale_probes": [
            "185+ integration slugs in catalog — stable vs beta honesty.",
            "Harness lab stack (sqlite, redis, qdrant, otel) as reference Tier-3 preset.",
            "Plugin entry-point registration at scale (tools, skills, integrations bundles).",
        ],
        "overrides": "IntegrationProfile presets (`lab_stack`, `legal_stack`, `research_stack`, `harness_production_stack`) · ApplicationManifest · scaffold defaults · `wire_application_environment`",
        "ci_scripts": [
            "uv run python scripts/check_docs_domain_pairs.py",
            "uv run python scripts/check_intergrax_no_applications_imports.py",
            "uv run python scripts/check_agents_no_tier3_imports.py",
            "python scripts/check_harness_no_getattr.py",
            "uv run pytest -m gate -q",
        ],
        "production_baseline": "Cursor/Claude Code/Codex-class agent harnesses · enterprise Agent OS platforms (policy-first, composable runtime, replaceable workers)",
        "anti_patterns": "Declaring whole platform complete · starting Phase K silently · duplicating Tier-0 in Nexus · monolithic implementation plan files",
        "appendix": "N/A",
    },
    {
        "id": "UNIFIED_EXECUTION_RUNTIME",
        "title": "Unified Execution Runtime (UAEP)",
        "layers": "4–5, 8, 23–24",
        "mission": (
            "Audit the **Agent OS execution substrate**: policy-first UAEP, typed runtime events, "
            "identity/trust propagation, security redaction, cost governance, checkpoint/pause/resume, "
            "and delegation — on **every** runtime path with no policy bypass."
        ),
        "code": """intergrax/runtime/nexus/nexus_loop.py · unified_task_runner.py
intergrax/agents/agent_engine.py · intergrax/agents/uaep.py
intergrax/runtime/nexus/tools/tool_runtime.py
intergrax/runtime/policy/policy_engine.py
intergrax/runtime/events/ (runtime_event.py, phase_coverage.py, unified_run_journal.py)
intergrax/runtime/middleware/ · intergrax/runtime/architecture/ (prompt_security, tool_security, tenant_security, retrieval_security, cost_budget, cost_quota)
intergrax/runtime/schema/registry.py
applications/_shared/runtime_config_bridge.py · identity_wiring.py · guardrail_wiring.py""",
        "key_symbols": "RuntimeEvent/RuntimeEventType · ExecutionPhase · HookPoint/HookContext · AgentDecision · ExecutionInterrupt · PauseRecord · RuntimeExecutionContext · RuntimePolicyBundle · PolicyDecision · ToolRequest/ToolResponse · AgentStep · ValidationResult · MemoryView · DelegationSpec · ApplicationSecurityProfile · GuardrailProfile",
        "active_phases": "R-Policy Done · R-Delegate Done · V-REM-SEC · SEC · COST · GR-DOC · REL-ADV autonomy",
        "known_gaps": "HTTP mid-run autonomy mostly lab-only · supervisor EscalationRouter future · middleware target layout partially evolved",
        "dimensions": [
            "Single path: UnifiedTaskRunner → NexusLoop → AgentEngine → UAEP steps — no parallel legacy engines.",
            "Every AgentStep emits STEP_* events with trace_id/run_id/tenant_id.",
            "ToolRuntime.invoke emits TOOL_* events — all tool paths, including catalog dispatch.",
            "PolicyEngine: pre-run, pre-plan, pre-LLM, pre-tool, post-tool, pre-output, memory writes.",
            "RuntimePolicyBundle is the single policy composition object — no orphan policy dicts.",
            "AgentDecision emitted **before** Nexus acts on model output.",
            "Retry managed by runtime (RetryEngine) — not unbounded agent while-loops.",
            "MemoryView is the agent memory interface — no direct store access from Tier-2.",
            "Delegation uses DelegationSpec with scoped permissions — child cannot inherit all parent tools.",
            "tenant_id on events; secrets redacted in traces (ApplicationSecurityProfile).",
            "Guardrail middleware (llm_guardrail) composes via IntegrationProfile — not agent SDK.",
            "Checkpoint/pause/resume uses RuntimeCheckpoint — recoverable UAEP cursor.",
            "schema_version validated on runtime contracts.",
            "HITL via REQUEST_HUMAN / policy — not ad-hoc Slack in agent code.",
            "Cost budgets enforced (max_cost, token metering hooks).",
            "Hooks (HookRegistry) do not call vendor adapters directly.",
            "Forbidden: agent-specific Nexus branches; duplicate policy engines.",
        ],
        "scale_probes": [
            "PM→UX→Legal→Validator→Human multi-agent chain (§42.43).",
            "Budget exhaustion mid-run (max_steps, max_cost).",
            "Cooperative cancel at step boundaries.",
            "Large delegation trees with permission scope audit.",
        ],
        "overrides": "RuntimePolicyBundle via runtime_config_bridge · ApplicationSecurityProfile · GuardrailProfile · HookRegistry · RuntimePlugin · TaskExecutionOptions.autonomy_level",
        "ci_scripts": [
            "uv run pytest tests/unit/runtime/ -q",
            "python scripts/check_harness_no_getattr.py",
            "uv run python scripts/check_observability_gates.py",
            "uv run pytest -m gate -q",
        ],
        "production_baseline": "UAEP-class agent runtimes · NeMo Guardrails / Guardrails AI / LLM Guard as integration backends (§42.11.6)",
        "anti_patterns": "Policy in docs only · LLM calls bypassing policy · context assembly bypass · untraced policy decisions",
        "appendix": "Appendix H (governance control plane)",
    },
    {
        "id": "ORCHESTRATION",
        "title": "Orchestration",
        "layers": "3, 9",
        "mission": (
            "Audit **intake normalization**, **scheduling**, **ExecutionGraph** execution, parallelism, "
            "merge policies, coordination patterns, resilience layers, and CFG configuration completeness — "
            "as formal Tier-1 responsibilities, not agent-implemented orchestration."
        ),
        "code": """intergrax/runtime/nexus/orchestration/ (intake, planning, graph_runner)
intergrax/runtime/nexus/execution/graph_executor.py
intergrax/runtime/architecture/multi_agent_coordination.py (CoordinationPattern)
intergrax/runtime/nexus/orchestration_capabilities.py
intergrax/queueing/ · intergrax/distributed/
applications/_shared/task_intake.py · orchestration_wiring.py
applications/contracts/graph_builder.py (AgentGraph)
scripts/check_orchestration_config_docs.py""",
        "key_symbols": "TaskEnvelope · OrchestrationProfile · ApplicationGraphSpec · ExecutionGraph · NexusPlan/PlanStep · CoordinationPattern · MergeStrategy (concat/last_wins/structured_json) · SubtaskContract · IntentRoute",
        "active_phases": "ORCH Done · ORCH-STRAT · ORCH-CONFIG (11/11) · ORCH-5.1 swarm · ORCH-6 sync/async · H-APP-WIRING surface parity",
        "known_gaps": "CFG-14 LKW hybrid E2E deferred · active-active node redundancy L0 · QueuedNexusExecutionAdapter not scaffold-default · semantic merge ORCH-5.4 future",
        "dimensions": [
            "All tasks enter via UnifiedTaskRunner / normalized TaskEnvelope — no API bypass.",
            "ExecutionGraph has typed nodes/edges — graph not implicit in agent methods.",
            "DELEGATES_TO expands to child node (ADR-FLOW-001) — not function-call subagents.",
            "Parallel batches specify merge_strategy — deterministic merge verified.",
            "max_delegation_depth enforced.",
            "Scheduler: priority, concurrency caps, backpressure (GRAPH_BACKPRESSURE event).",
            "Three retry layers A/B/C documented and not conflated in code.",
            "CoordinationPattern explicit per graph/host (§50 catalog).",
            "classifier_kind rules|llm for free-text intake when required.",
            "graph_spec respects trigger_capabilities (ADR-FLOW-004).",
            "CFG-01–CFG-20 cases documented with honest host matrix §59.2.",
            "Tier-2 agents do not call other agents directly — Nexus delegates.",
            "Fan-out/fan-in with concurrency limits — not unbounded asyncio.gather in agents.",
            "Long-running recovery (CFG-19) and strict mode (CFG-20) paths inspected.",
            "OrchestrationProfile fields wired — no orphan CFG knobs.",
            "Sync/async/streaming postures share same Nexus core path.",
        ],
        "scale_probes": [
            "CFG simulation tests (orchestration config matrix).",
            "Deep graph + wide parallel fan-out + stuck-node recovery.",
            "Swarm CFG-17 budget envelope.",
            "GRAPH_BACKPRESSURE at max_inflight_nodes.",
        ],
        "overrides": "OrchestrationProfile (planner_kind, classifier_kind, merge_strategy, caps) · ApplicationGraphSpec · trigger_capabilities · strict_multi_agent_defaults() · apply_long_running_from_profile",
        "ci_scripts": [
            "uv run python scripts/check_orchestration_config_docs.py",
            "uv run pytest tests/unit/runtime/nexus/orchestration/ -q",
            "uv run pytest tests/acceptance/agent_os/ -q -k orchestration",
        ],
        "production_baseline": "LangGraph/CrewAI coordination · Viktor-style long-running workflows · enterprise multi-agent orchestration (IDEAL §6.4)",
        "anti_patterns": "Subtasks as plain function calls · implicit graph in agent code · missing merge policy · scheduler logic in Tier-2",
        "appendix": "Appendix I (orchestration control plane)",
    },
    {
        "id": "NEXUS_EXECUTION_FLOW",
        "title": "Nexus Execution Flow",
        "layers": "8–10",
        "mission": (
            "Audit the **end-to-end Nexus loop narrative** against NEXUS_EXECUTION_FLOW canon: "
            "step ordering, three planning planes, handoff/retry, final response composition, "
            "flow-level observability, and acceptance-scenario coverage."
        ),
        "code": """intergrax/runtime/task/unified_task_runner.py
intergrax/runtime/nexus/nexus_loop.py
intergrax/runtime/nexus/tools/tool_loop.py · plan_context_invocation.py
intergrax/agents/agent_engine.py · authoring/acp_run.py · HarnessKernel
intergrax/runtime/nexus/agent_router.py
intergrax/runtime/nexus/context/context_manager.py
intergrax/runtime/nexus/handoff/coordinator.py
intergrax/runtime/nexus/retry/retry_engine.py
intergrax/runtime/nexus/response/final_response_composer.py
applications/_shared/nexus_factory.py · graph_spec_to_plan.py
tests/acceptance/agent_os/""",
        "key_symbols": "Task/TaskLifecycle/TaskResult · SharedTaskContext · AgentContextBundle · TaskContextAssemblyOptions · RuntimeRequest · AgentHandoff · ValidationResult · ExecutionNode",
        "active_phases": "FLOW 18/18 Done · FLOW-CTL · FLOW-8 harness Done/product Deferred · H-APP-WIRING · COG-DEPTH cross-ref",
        "known_gaps": "FLOW-GAP-20 hybrid daemon LKW · UC-6 research stubs · WAITING_FOR_RESOURCES/EXPIRED reserved v1 · production-ready = Partial without strict profile + W-OPS",
        "dimensions": [
            "Three planning planes distinguished: Nexus planner / agent on_next_step / tool planner.",
            "TaskClassifier does not mutate Task.state directly.",
            "AgentRouter respects production_mode and registry constraints.",
            "Handoff uses HandoffCoordinator — traced lineage.",
            "FinalResponseComposer applies merge_strategy from orchestration.",
            "FLOW-GAP register items closed in code or explicitly deferred with risk.",
            "Cancel is cooperative at step boundaries.",
            "Trace reconstructs full 'why did run stop' narrative.",
            "DECISION_EMITTED on UAEP steps before side effects.",
            "RAG poisoning defense active on catalog rag.retrieve path (cross-check RAG domain).",
            "Reserved lifecycle states not used in production hosts.",
            "Engine planner requires llm_adapter at bootstrap — fail-fast if missing.",
            "Partial completion policy explicit when PARTIALLY_COMPLETED allowed.",
            "Evaluation/critic hooks profile-driven — not hardcoded per agent.",
            "Lab vs production matrix §1.4 respected in host configs.",
            "Acceptance scenarios UC-1–UC-9 / S1–S7 have test evidence.",
        ],
        "scale_probes": [
            "Acceptance 01–10 including mid-UAEP resume 05b.",
            "Parallel execution cap integration tests.",
            "Handoff + retry combined scenarios.",
            "Long-running loop with nested delegation.",
        ],
        "overrides": "execution_mode strict|balanced · EvaluationProfile · CriticProfile · require_human_approval · graph_spec on profile · lab trace debug routes",
        "ci_scripts": [
            "uv run pytest tests/acceptance/agent_os/ -q",
            "uv run pytest tests/unit/runtime/nexus/ -q -k 'handoff or graph_spec'",
            "python scripts/check_harness_no_getattr.py",
        ],
        "production_baseline": "Agent OS acceptance suite · reference host presets · W-OPS SLO evidence for production claims",
        "anti_patterns": "Confusing three planning planes · agent-specific Nexus branches · undocumented partial completion · flow doc/code drift",
        "appendix": "Appendix I §I.2–I.6",
    },
    {
        "id": "AGENT_CONTRACTS_AND_ASSEMBLY",
        "title": "Agent Contracts and Assembly",
        "layers": "17–20, 31 · ACP §21",
        "mission": (
            "Audit **AgentContract**, registry resolution, **Prompt Registry**, capability graph, "
            "agent lifecycle governance, **ACP cognitive patterns**, **author run() facade** "
            "(ADR-AGENT-001/002), **step loop on_next_step** and **dual observability** "
            "(ADR-AGENT-003) — Tier-2 hooks + environment merge; Nexus remains Agent OS for Task."
        ),
        "code": """intergrax/contracts/agent_contract_meta.py · runtime_execution_context.py
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
scripts/check_agents_lifecycle_metadata.py · check_agents_vendor_imports.py""",
        "key_symbols": "AgentContract · UAEPAgent · RuntimeExecutionContext · AgentDecision · CognitiveAgent · acp.state.v1 · IntergraxAgent · PromptMeta · AgentStepContext · StepOutcome · AgentRunTrace · ApplicationRunSummary",
        "active_phases": "ACP · ACP-CLOSE · ACP-FINISH Done (2026-06-13) · PE/REG/CG/AS closed · AUDIT-IDEAL-19.1/20.1/31.1 parallel",
        "known_gaps": "GAP-ACP-36/37 Closed (ACP-TOK-*) · GAP register 37 Closed · 0 Open · AUDIT-IDEAL-19.1/20.1/31.1 Planned · COST-1 RunBudget Partial",
        "dimensions": [
            "AgentContract has required fields per §12 — capabilities, allowed_tools, risk metadata.",
            "UAEPAgent: get_steps/run_step — AgentEngine path, not private HTTP bypass.",
            "decide_after_step returns typed AgentDecision — not ad-hoc control flow.",
            "Nexus routes by capability token — not Python class name.",
            "ADR-AGENT-001 Accepted; architecture §21–§36 ACP + run/step canon present.",
            "Three cognition planes (§23) — no private multi-agent graph inside run_step (ACP-AP-01).",
            "Tool calls via RuntimeExecutionContext.invoke_tool / ToolRuntime only.",
            "Agents control loop via on_next_step only — no Tier-2 RuntimeEngine/pipeline (ACP-CLOSE-LEG-5).",
            "CognitiveAgent base exists or gap ACP-1 recorded.",
            "Pattern classes Reflex/ReAct/PlanExecute/Decomposition/Reflection vs ACP-2..6.",
            "acp.state.v1 schema and cognitive_pattern on contract (ACP-0/0b).",
            "ReActAgent iteration budget aligns with TOOL-ENG-6 when both Done.",
            "ReflectionAgent uses CVL critic hooks — no critic SDK in Tier-2.",
            "Config split: Tier-3 profile vs agent domain — not all config in agent class (ACP-AP-03).",
            "Prompt templates have ownership, version, layered compilation.",
            "Capability graph edges reflect manifest roster with lineage.",
            "Registry snapshot conformance tests pass CI.",
            "Deprecated/retired agents rejected in strict production_mode.",
            "Agent checklist §45 + ACP pattern selection (§26.1).",
            "Forbidden §42.41 patterns absent (vendor SDK, direct integrations).",
            "skill_ids → allowed_tools resolution audited.",
            "scaffold --pattern when ACP-8 Done.",
            "check_agent_pattern_conformance.py when ACP-13 Done.",
            "acceptance agent_os covers UAEP path for reference agents.",
            "AgentRunRequest/Result and merge_environment per §29–§30 (ACP-DX).",
            "Per-agent memory_namespace and rag_collection — not global store.",
            "Application metadata → environment_overrides wired in hosts.",
            "on_next_step / StepOutcome author API per §32 (ACP-STEP-1).",
            "execute_next_step harness-only — authors cannot override (ACP-STEP-2).",
            "HarnessKernel.execute_step deterministic primitive — no agent planning §38 (ACP-STEP-2b).",
            "NexusLoop vs HarnessKernel separation §38 — not nexus.run() as agent brain.",
            "AgentRunTrace on AgentRunResult with tool/RAG/LLM step records §31 (ACP-OBS-1).",
            "ApplicationRunSummary for Task orchestration §31 (ACP-OBS-2).",
            "StepLLMRouter per-step model within LLMProfile §33 (ACP-LLM-1).",
            "SharedContextView for multi-agent handoffs §34 (ACP-STATE-1).",
            "Use-case catalog UC-1..10 supported without agent rewrite §35.",
            "AgentRunErrorCode and TerminalReason enums per §37.4–§37.5 (ACP-CON-1).",
            "state_delta JSON merge-patch + _version + resume conflict §37.2 (ACP-CON-2).",
            "Side-effect mode immediate vs declarative — no mix per step §32.8 (ACP-CON-3).",
            "Capability routing by token not class name §37.6 (ACP-CON-6).",
            "Security guards STRICT tool/memory/RAG §37.7 (ACP-CON-7).",
            "OrganizationalPolicyEnvelope constrains agents without code fork §39 (ACP-ORG).",
            "PolicyVerdictRecord on steps for compliance measurement §39.5 (ACP-ORG-4).",
            "Checkpoint/resume/replay semantics §40.1 (ACP-PROD-1).",
            "Side-effect idempotency keys and dedupe §40.2 (ACP-PROD-2).",
            "ToolExecutionProfile mutability/compensation §40.3 (ACP-PROD-3).",
            "SharedContextView CAS concurrency §40.5 (ACP-PROD-5).",
            "ArtifactRef typed contract §40.6 (ACP-PROD-6).",
            "Agent threat model mitigations §40.7 (ACP-PROD-7).",
            "Privacy/redaction on trace/memory §40.8 (ACP-PROD-8).",
            "Release eval gates before production_mode §40.9 (ACP-PROD-9).",
            "CI conformance matrix §40.10 (ACP-PROD-10).",
            "Contract schema_version migration §40.11 (ACP-PROD-11).",
            "RequestIdentity tenant_id/user_id and memory_scope user vs org §30.9 (ACP-DX-1/2).",
        ],
        "scale_probes": [
            "Large agent roster with capability-based routing.",
            "Registry snapshot at bootstrap vs runtime mutation.",
            "Promotion dev→staging→prod evidence chain.",
            "ReActAgent at max_react_iterations — FAIL vs REQUEST_HUMAN behavior.",
            "DecompositionAgent deep sub-question tree — budget + acp.state.v1 checkpoint.",
            "Same agent class in two Tier-3 hosts with different ToolProfile/LLMProfile.",
        ],
        "overrides": "PromptProfile · ToolProfile · LLMProfile · OrchestrationProfile · ApplicationGraphSpec · cognitive_pattern/pattern_config (ACP-0b) · AgentRegistry.register · wire_application_environment · scaffold --pattern (ACP-8)",
        "ci_scripts": [
            "uv run python scripts/check_agents_lifecycle_metadata.py",
            "uv run python scripts/phase_v_capability_graph_guard.py",
            "uv run python scripts/check_agents_vendor_imports.py",
            "uv run pytest tests/acceptance/agent_os -m agent_os -q",
            "uv run pytest tests/unit/agents/ -q",
            "uv run pytest tests/unit/agents/authoring/patterns/ -q",
        ],
        "production_baseline": "Enterprise agent registries · LangGraph/ADK pattern libraries · Cursor-style decomposition · prompt governance · capability routing (service-mesh analogy)",
        "anti_patterns": "Hardcoded agent class routing · vendor SDK in Tier-2 · orphan prompts · skipping lifecycle · ACP-AP-01..07 (fat agent absorbs Nexus, multi-agent in run_step, secrets in agent source)",
        "appendix": "ADR-AGENT-001 · ADR-AGENT-002 · ADR-AGENT-003 · Appendix M/N/O/P · Appendix AC",
        "adr": "ADR-AGENT-001 · ADR-AGENT-002 · ADR-AGENT-003",
    },
    {
        "id": "INTEGRATIONS",
        "title": "Integration Library",
        "layers": "13",
        "mission": (
            "Audit the **Integration Library** as the sole vendor boundary: 185+ slugs, typed contracts, "
            "health probes, IntegrationProfile-driven backend selection, guardrail integrations, and "
            "CI-enforced import boundaries."
        ),
        "code": """intergrax/integrations/ (contracts/, registry/, providers/)
intergrax/integrations/registry/harness_lab_stack.py · presets.py
intergrax/integrations/_shared/p2|p3|p4|p5|p6|p7|p8/factories.py
applications/_shared/integration_wiring.py · integration_runtime_bridge.py
applications/_shared/guardrail_wiring.py
scripts/check_integration_vendor_imports.py
scripts/check_harness_guardrail_wiring.py · scripts/generate_integration_usage_docs.py""",
        "key_symbols": "IntegrationManifest · IntegrationProfile · IntegrationCategory · IntegrationPlugin · LlmGuardrailBackend · GuardrailScanResult · RelationalStore · VectorStore · MessageBus · SearchProvider",
        "active_phases": "Phase M catalog · M.6 P5/P6/P7 Done · M.12 guardrails Done · M-P12-CAT.1 · GR-DOC",
        "known_gaps": "Most slugs **beta** — stable vs beta must be honest · thin P4 shells · SaaS-only without local container · nginx/ingress slug missing (ECP cross-ref)",
        "dimensions": [
            "No vendor SDK imports in agents/ or Nexus business logic.",
            "Every slug in layout/registry with conformance tests where claimed stable.",
            "IntegrationProfile drives backend selection — wired through bridges, not getenv in agents.",
            "llm_guardrail via middleware + IntegrationProfile — not parallel tier or agent SDK.",
            "Health probes for external deps; circuit breaker registry used.",
            "Secrets via SecretsStore/integration options — not committed config.",
            "RAG vector stores via catalog bridges — not duplicate vector clients in agents.",
            "Guardrail layering L1→L4 documented and composed (ADR-GR-001).",
            "Slack/Teams/etc. are adapters — not orchestrators replacing Nexus.",
            "Cloud facades do not wrap LLM providers (LLM via llm_adapters/).",
            "bootstrap_application_integration_catalog() used by Tier-3 hosts.",
            "Harness lab stable stack smoke tests pass.",
            "New provider has USAGE.md and manifest conformance.",
            "Vendor imports only in allowed modules — CI check_integration_vendor_imports green.",
            "Tier-3 extend_tool_profile_for_integration() pattern followed.",
        ],
        "scale_probes": [
            "HARNESS_M6_P5/P6/P7 probe slugs and health endpoints.",
            "Failover between providers (where documented).",
            "Rate limits and bulk operations on message_bus/data slugs.",
            "Compose profiles: lab_stack, harness_guardrail_stack, research_web_stack.",
        ],
        "overrides": "IntegrationProfile + presets · per-slug options in profile · IntegrationPlugin (EXTENSION_AUTHOR_GUIDE) · wire_integration_tool_context()",
        "ci_scripts": [
            "python scripts/check_integration_vendor_imports.py",
            "uv run python scripts/check_harness_guardrail_wiring.py",
            "uv run pytest tests/unit/integrations/ -q",
            "uv run python scripts/generate_integration_usage_docs.py",
        ],
        "production_baseline": "Large integration catalogs (LangChain-style) · harness lab stable stack · NeMo/Guardrails AI/LLM Guard/Presidio (§47)",
        "anti_patterns": "Agent-imported vendor SDK · duplicate adapter per product · guardrail as agent code · stable label on beta-only slug",
        "appendix": "Appendix K (integration control plane)",
    },
    {
        "id": "RAG",
        "title": "RAG and Retrieval Engine",
        "layers": "14",
        "mission": (
            "Deep audit of the **Tier-0 retrieval engine** vs production RAG systems: ingest, chunking, "
            "indexing, retrieval modes, query routing, resilience, security (poisoning), tenancy, "
            "observability, evaluation — and honest L2.5/L3 posture with M-RAG-DEPTH gap closure status."
        ),
        "code": """intergrax/rag/profiles/rag_profile.py
intergrax/rag/bootstrap/rag_stack_bootstrap.py · create_default_rag_stack()
intergrax/rag/ingest/ingest_pipeline.py · ParserPipeline
intergrax/rag/retrieval/retrieval_service.py
intergrax/rag/retrievers/ (hybrid, fusion, graph_rag, hierarchical, multi_query, agentic, …)
intergrax/rag/rerankers/ · intergrax/rag/vectorstore/
intergrax/rag/evaluation/golden_harness.py
intergrax/rag/tracking/ (RetrievalTrace, metrics)
applications/_shared/rag_runtime_bridge.py
intergrax/runtime/nexus/tools/plan_context_invocation.py
intergrax/tools/providers/rag/
.github/workflows/rag-guard.yml · tests/fixtures/rag_golden/""",
        "key_symbols": "RagProfile · RagStack · RetrievalService · RetrievalRequest/RetrievalResult · RetrievalTrace · IngestPipeline · QueryRouter · MetadataFilter · DualIndexStrategy · HierarchicalRetriever",
        "active_phases": "RAG-LC Done · M-RAG-DEPTH/M-RAG-GRAPH/M-RAG-BACKLOG Done · §6.1av RAG-MAINT queue",
        "known_gaps": "GAP-RAG-01..40 **Closed** (LC) · GAP-RAG-15/34 Frozen · §6.1av RAG-MAINT **Done** · M-RAG.58 → AHI Frozen index",
        "dimensions": [
            "Single canonical path: RagProfile → RetrievalService → rag.retrieve catalog tool.",
            "Agents do not call vectorstore.query or vendor SDKs directly.",
            "RagProfile fields wired — flag dead config (especially query_expansion, INTERGRAX_RAG_* env).",
            "ParserPipeline + chunking strategies (5+) used on ingest — not raw text shortcut.",
            "Retrieval modes: vector, keyword, hybrid, fusion, graph, rerank, agentic, hierarchical — wired vs doc-only.",
            "DualIndexStrategy + HierarchicalRetriever wired in default bootstrap for book-scale (GAP-RAG-02/03).",
            "Short/medium docs: sync ingest OK with explicit profile.",
            "Multi-GB corpora: job orchestration / stream ingest — honest not-ready if missing.",
            "Retrieval poisoning defense on **all** surfaces including perform_rag_retrieve catalog path.",
            "MetadataFilter + tenant namespace enforcement with prod vector backends.",
            "Resilience: embedding retry, retriever retry, fallback chains, circuit breakers — per canon.",
            "RetrievalTrace + parser trace; OTel spans on retrieve/ingest hot paths.",
            "Citations: chunk metadata + composer; formal Citation on RetrievalResult if canon requires.",
            "Golden harness passes (retrieval, graph_rag, multi_hop, agentic scenarios).",
            "agentic_enabled defaults safe (false) unless Tier-3 opts in.",
            "Graph RAG (document graph) ≠ agent user memory (MEMORY boundary).",
            "Integration slugs: vector_store, document_parser, rerank_provider resolved via IntegrationProfile.",
            "Compare maturity table in architecture §Production readiness verdict — update if code changed.",
        ],
        "scale_probes": [
            "Single-page HTML, 100-page PDF, book-scale TOC/hierarchical path.",
            "High QPS retrieve with reranker latency budget.",
            "Poisoned chunk injection attempt on Nexus + catalog paths.",
            "Semantic chunking O(n) embedding cost on large doc.",
            "Multi-tenant corpus isolation scenario.",
        ],
        "overrides": "RagProfile + INTERGRAX_RAG_* env · IntegrationProfile vector_store/document_parser/rerank_provider · rag_runtime_bridge · ContextProfile.enable_rag · production_rag_profile()",
        "ci_scripts": [
            "uv run pytest tests/unit/rag/ -q",
            "uv run pytest tests/integration/ -q -k rag",
            "# .github/workflows/rag-guard.yml scenarios",
        ],
        "production_baseline": "LlamaIndex/Weaviate/Qdrant enterprise RAG · LangChain retrieval pipelines · multi-tenant vector stores · production ingest job queues",
        "anti_patterns": "Dense-only retrieval · RAG logic inside agent · multiple uncorrelated RAG paths · missing citations · retrieve without tenant filter",
        "appendix": "Appendix K §K.5",
    },
    {
        "id": "TOOLS",
        "title": "Tool Library and ToolRuntime",
        "layers": "11",
        "mission": (
            "Audit the **Tool Library** (190+ catalog tools) and **ToolRuntime** execution engine: "
            "selection/planning strategies, policy enforcement, idempotency, MCP export, catalog dispatch, "
            "and TOOL-ENG hardening queue — vs production tool-governance systems."
        ),
        "code": """intergrax/tools/core/contracts.py · intergrax/tools/registry/
intergrax/runtime/nexus/tools/tool_runtime.py · invoker.py · catalog_dispatch.py
intergrax/runtime/nexus/tools/tool_planning_service.py · catalog_tool_planner.py
intergrax/runtime/nexus/tools/tool_selection.py
intergrax/runtime/nexus/tools/tool_loop.py
intergrax/runtime/tools/idempotent_invoker.py · runtime_bound_catalog.py
applications/_shared/catalog_runtime_bridge.py · tool wiring
scripts/check_legacy_tool_plan_booleans.py · check_tool_mcp_schema_export.py
scripts/check_tool_injection_defense.py · check_agent_registry_bypass.py""",
        "key_symbols": "ToolContract · ToolRegistry · ToolProfile · ToolWiringContext · ToolRequest/ToolResponse · ToolAccessPolicy · ToolSelectionStrategy · ToolPlanDecision · ToolRiskLevel · tools_mode · tools_context_scope",
        "active_phases": "Phase O/T-EXPAND Done · **TOOL-ENG Closed** (2026-06-12, 36/36, S0–S8) · Phase V V-SEC/V-COST/V-EVAL",
        "known_gaps": "Deferred: hierarchical LLM category pass (ADR-TOOL-005 v1) · optional L1 critic per-tool output (CVL) · ACP invoke_tool/gateway consistency across 190 tools (cross-domain). TOOL-ENG register closed.",
        "dimensions": [
            "All invocations via ToolRuntime → policy → RuntimeToolInvoker — no bypass.",
            "Every tool: tool_id, input/output schema, risk level, description for LLM selection.",
            "ToolSelectionStrategy wired before LLM tool call (ENG-5) — not post-hoc only.",
            "Tool planning: catalog_tool_planner + tool_planning_service — allow-list respects AgentContract.",
            "ToolScopePolicy / StaticToolScopePolicy enforced (ENG-3).",
            "Catalog tool_id dispatch (ENG-1/2) — capability alias vs catalog id consistent.",
            "Idempotency keys on side-effect tools (idempotent_invoker).",
            "Concurrency, timeout, retry on invocation path.",
            "ops:tool_audit and TOOL_* trace events emitted.",
            "MCP export schema parity with OpenAI function schema — CI green.",
            "Legacy boolean flags (use_rag, tool_gateway) deprecated — check_legacy_tool_plan_booleans green.",
            "Injection defense middleware active.",
            "Agents cannot bypass registry — check_agent_registry_bypass green.",
            "Plugin model: ToolPlugin + entry points + bootstrap_catalogs.",
            "Skills merge into tool allow-list correctly at resolution time.",
            "HIGH-risk tools: post-tool verification (ENG-7 gap status).",
            "ReAct / iterative tool loop bounded (ENG-6 gap status).",
            "EnvironmentProfile tool_selection fields wired (recent catalog_runtime_bridge work).",
        ],
        "scale_probes": [
            "190 tools / 48 bundles registration at bootstrap.",
            "RunBudget.max_tool_calls (128 prod default) enforcement.",
            "Parallel read-only tool invocations (ENG-9 target).",
            "Large allow-list filtering performance.",
        ],
        "overrides": "ToolProfile.enabled/enabled_bundles · ReasoningProfile.tool_planner_prompt_id · tool_selection_mode on EnvironmentProfile · RuntimePolicyBundle.tool_access · tools_mode on engine plan · external ToolPlugin",
        "ci_scripts": [
            "python scripts/check_legacy_tool_plan_booleans.py",
            "python scripts/check_tool_mcp_schema_export.py",
            "python scripts/check_tool_injection_defense.py",
            "python scripts/check_agent_registry_bypass.py",
            "uv run pytest tests/unit/runtime/nexus/tools/ -q",
        ],
        "production_baseline": "OpenAI function calling / MCP · enterprise tool allow-lists and audit · Cursor-scale tool routing with policy",
        "anti_patterns": "Direct handler invoke bypassing ToolRuntime · boolean use_* flags parallel to tools · vendor SDK in tool handlers ·unbounded tool loops in agents",
        "appendix": "Appendix J",
    },
    {
        "id": "CODE_CRAFT",
        "title": "Ephemeral Code Craft",
        "layers": "11b",
        "mission": (
            "Audit **Ephemeral Code Craft runtime** at L3+: verify `codecraft.*` tools, "
            "`wire_application_codecraft()`, orchestrator loop, sandbox tiers, policy gates, "
            "CVL integration, ephemeral tool registry hygiene — confirm Done vs depth backlog."
        ),
        "code": """intergrax/codecraft/ · intergrax/runtime/codecraft/
intergrax/runtime/sandbox/
intergrax/tools/providers/sandbox/ · intergrax/tools/providers/codecraft/
intergrax/applications/_shared/codecraft_wiring.py
intergrax/runtime/critic/ (CVL hooks)
docs/architecture/CODE_CRAFT.md · docs/plan/CODE_CRAFT.md · ADR-CODECRAFT-001""",
        "key_symbols": "CodeCraftProfile · CodeCraftOrchestrator · CodeCraftSession · CraftResult · IterationRecord · StaticCodeGate · craft modes (disabled|dry_run|assist_only|supervised|autonomous) · EphemeralToolRegistry · wire_application_codecraft",
        "active_phases": "ECC-0…ECC-6 + S7–S11 **Done** (L3+, 2026-06-13) · ADR-CODECRAFT-001",
        "known_gaps": "GAP-ECC-20…23 **Closed** (ECC-MAINT-01..04) · local SandboxSession ≠ OS containment (accepted) · dedicated container runtime backend product opt-in beyond local fallback",
        "dimensions": [
            "CodeCraft uses existing sandbox ToolRuntime path — no parallel execution stack.",
            "L0 StaticCodeGate before any execute in autonomous/supervised paths.",
            "Codegen LLM separated from producer/judge LLM identity (template adapter shipped; profile ref → backlog).",
            "Ephemeral tools do not persist in global ToolRegistry after session.",
            "CraftResult promotion typed — not stdout-only.",
            "Fail-closed when codecraft_profile missing or mode=disabled.",
            "CODECRAFT_* events correlated with trace_id/run_id.",
            "Tier-2 invokes only codecraft.* / sandbox.* catalog tools.",
            "Network egress policy enforced per sandbox tier.",
            "CVL L0/L1 integrated — not parallel verification stack.",
            "Modes table §6.3 respected (supervised vs autonomous).",
            "Resource disposal releases craft_id / sandbox session.",
            "cloud substrate (e2b/modal/daytona) via IntegrationProfile — not agent SDK.",
            "max_total_exec_time_s enforced on session iteration paths.",
            "Document honest L3 maturity — depth backlog only for metrics/container/codegen LLM.",
        ],
        "scale_probes": [
            "generate→gate→exec→test→CVL iteration within max_iterations.",
            "max_code_bytes and max_total_exec_time_s enforcement.",
            "Concurrent codegen sessions without registry pollution.",
        ],
        "overrides": "ApplicationEnvironmentProfile.codecraft_profile · Task.metadata.codecraft_mode · sandbox_host_slug · codegen_llm_profile_ref · require_hitl_before_exec",
        "ci_scripts": [
            "python scripts/check_codecraft_layer.py",
            "uv run pytest tests/unit/codecraft/ tests/unit/tools/providers/codecraft/ tests/unit/runtime/codecraft/ -q",
            "uv run pytest tests/unit/runtime/sandbox/ -q",
            "python scripts/check_harness_no_getattr.py",
        ],
        "production_baseline": "Cursor ephemeral codegen · E2B/Modal sandboxes · CI codegen with semgrep/trivy gates",
        "anti_patterns": "Claiming ECC Planned when runtime shipped · arbitrary exec bypassing ToolRuntime · global registry pollution · local workspace labeled as OS sandbox",
        "appendix": "Appendix J (tool surfaces)",
    },
    {
        "id": "SKILLS",
        "title": "Skill Library",
        "layers": "12",
        "code": """intergrax/skills/registry/catalog.py · bootstrap.py · resolver.py
intergrax/skills/integration/contract_resolution.py
intergrax/skills/providers/*/ · importers/cursor_skill_md.py
applications/_shared/skill_wiring.py · skill_tool_profile.py · catalog_runtime_bridge.py
intergrax/runtime/registry/agent_registry.py (skill resolution at register)""",
        "key_symbols": "SkillManifest · SkillProfile · SkillRegistry · SkillResolver/SkillResolverProtocol · ResolvedSkillPack · SkillPlugin · SkillBundleEntry",
        "active_phases": "SK-EXP through SK-EXP5 Done · **SK-BRIDGE.1** prompt→ContextManager · **SK-BRIDGE.2** policy_fragment→bundle · SK-PRESET.1 · Phase TS-3",
        "known_gaps": "prompt_instruction_ids not auto-injected to ContextManager · policy_fragment_id not merged to RuntimePolicyBundle · knowledge bundle BETA",
        "mission": (
            "Audit **149 skills / 41 bundles** as composable capability packs above tools: "
            "resolution, policy fragments, registration, roster consistency, and honest SK-BRIDGE gap status."
        ),
        "dimensions": [
            "Skills are not LLM-callable directly — tools are the invocation surface.",
            "allowed_tools is output of registry resolution — not hand-maintained duplicate list.",
            "Unknown skill_id fails at register time — not runtime surprise.",
            "Resolved tool_ids exist in ToolRegistry.",
            "requires_skills topological expansion detects cycles.",
            "USAGE.md per skill/bundle where canon requires.",
            "External Cursor SKILL.md import traced (SKILL_IMPORT_FAILED/SKILL_RESOLVED events).",
            "Capability graph records skill edges.",
            "Environment roster ⊆ skill/tool profile intersection enforced.",
            "skill.resolve catalog tool works for diagnostics.",
            "Bundles STABLE except knowledge (BETA labeled).",
            "SK-BRIDGE.1/.2 gaps documented honestly — verify if closed since last audit.",
            "SkillProfile presets (legal_skill_profile, research_skill_profile) wired at Tier-3.",
            "extend_tool_profile_for_skills() pattern used — not duplicate tool lists.",
            "Clear separation: skill composition vs atomic tool operation.",
        ],
        "scale_probes": [
            "149 skills resolved for agent with deep requires_skills chain.",
            "Roster vs environment consistency check at host bootstrap.",
            "Import external SKILL.md at scale.",
        ],
        "overrides": "SkillProfile.enabled_bundles · presets · AgentContract.skills[] · extend_tool_profile_for_skills() · import_cursor_skill_file",
        "ci_scripts": [
            "uv run pytest tests/unit/skills/ -q",
            "uv run pytest tests/unit/ -q -k skill",
        ],
        "production_baseline": "Cursor SKILL.md packs · CrewAI role bundles · policy fragments per capability pack",
        "anti_patterns": "Skills as parallel tool runtimes · silent skill ignore on unknown id · policy fragments not merged · knowledge bundle treated as STABLE",
        "appendix": "Appendix J",
    },
    {
        "id": "LLM_ADAPTERS",
        "title": "LLM Adapters",
        "layers": "6",
        "mission": (
            "Audit **LLMAdapter** abstraction: typed response envelopes (M-LLM-R), 19 provider slugs, "
            "streaming, structured output, metering, tenant scope, guardrail middleware, and planner≠producer discipline."
        ),
        "code": """intergrax/llm_adapters/ (registry/, providers/*, call_lifecycle.py, tracking/)
intergrax/llm/messages.py (AttachmentRef)
intergrax/runtime/replay/trace_replay_bridge.py
intergrax/runtime/adaptive/llm_call_summary.py
scripts/check_llm_adapter_typed_returns.py · scripts/check_agents_llm_adapter_response.py""",
        "key_symbols": "LLMAdapter · LLMAdapterResponse · LLMFinishReason · LLMTokenUsage · LLMToolCall · LLMStructuredResult[T] · LLMProfile · LLMStreamEvent · LLMCallConfig",
        "active_phases": "M-LLM-R envelope Done · W-ML.1 capability flags · Phase V FAUDIT-LLM.1 residual · COG cross-ref planner≠producer",
        "known_gaps": "Planner LLM ≠ producer discipline incomplete at Nexus boundary · distributed rate limit needs Redis wiring · usage tracking layers not auto-merged",
        "dimensions": [
            "All completions return LLMAdapterResponse / LLMStructuredResult — not bare str.",
            "Agents do not annotate LLM returns as str — CI check_agents_llm_adapter_response.",
            "Vendor SDK only inside provider modules — check_agents_vendor_imports.",
            "refusal/content_filter surfaced on envelope.",
            "Streaming LLMStreamEvent parity with non-streaming paths.",
            "LLMProfile drives model selection — not hardcoded model per agent.",
            "Token/cost usage on LLMTokenUsage; aggregated per run/tenant.",
            "Retries, timeout, circuit breaker via LLMCallConfig.",
            "Structured output schema validation — Pydantic/generic T.",
            "Guardrail middleware AFTER_LLM_OUTPUT when profile configured.",
            "llm_tenant_scope and INTERGRAX_LLM_TENANT_MAX_TOKENS quota.",
            "Metrics plugin on TASK_COMPLETED; register_llm_metrics_routes.",
            "Attachments respect ModalityProfile.max_media_bytes.",
            "Capability flags default false until provider tested (W-ML.1).",
            "Secrets via SecretsStore llm/<provider>/api_key paths.",
            "Replay bridge maps historical trace to adapter calls.",
        ],
        "scale_probes": [
            "High token volume run with cost aggregation.",
            "Tool-call-heavy turns with streaming.",
            "Provider failover / rate-limit storm.",
            "19 provider slug registry bootstrap time.",
        ],
        "overrides": "LLMProfile per host/step · SecretsStore paths · options.use_distributed_rate_limit · guardrail middleware stack",
        "ci_scripts": [
            "python scripts/check_llm_adapter_typed_returns.py",
            "python scripts/check_agents_llm_adapter_response.py",
            "python scripts/check_agents_vendor_imports.py",
            "uv run pytest tests/unit/llm_adapters/ -q",
        ],
        "production_baseline": "OpenAI/Anthropic/Azure/Bedrock enterprise adapters · Helicone/LangSmith proxies · SaaS token metering gateways",
        "anti_patterns": "str returns from adapters · model hardcoded in agent · direct SDK in Tier-2 · manual JSON parse for structured output",
        "appendix": "N/A",
    },
    {
        "id": "MEMORY",
        "title": "Memory Platform",
        "layers": "15",
        "mission": (
            "Audit **memory stores**, scopes, lifecycle, consolidation, and Knowledge-vs-LTM boundary — "
            "explicit, governed, observable, retrieval-first. Context assembly is audited under CONTEXT_ENGINEERING."
        ),
        "code": """intergrax/memory/ (user_profile_memory.py, contracts/)
intergrax/runtime/nexus/session/ · intergrax/runtime/task_memory/
intergrax/runtime/organization/ · consolidation services
applications/_shared/memory_wiring.py · memory_runtime_bridge.py
EntityGraphMemoryStore · workspace_index_spike.py (RFC — CE owns production wiring)""",
        "key_symbols": "MemoryProfile · MemoryKind · MemoryWritePolicy · PolicyScopedMemoryView · MemoryConsolidationJob · MemoryView · SharedTaskContext",
        "active_phases": "MEM Done · MEM-DEPTH Done · MEM-OBS.1 · ADR-MEM-001",
        "known_gaps": "MEMORY-LC Done · MEM-DEPTH Done · §6.1av depth closed (procedural/org/temporal) · MEM-MAINT-03 LangMem/Zep entity graph parity **backlog** (not Mem0 SaaS; no Phase K)",
        "dimensions": [
            "Memory types separated: STM, task KV, session, user LTM, tenant, procedural, shared context.",
            "Agents do not write Redis/Postgres/vector DB directly.",
            "Session vs checkpoint vs task KV stores distinct.",
            "Every read/write scoped; subagent namespace isolation (task_id/delegation/{node_id}/).",
            "MemoryWritePolicy + BEFORE_MEMORY_WRITE hooks enforced.",
            "Retrieval-first for large history — consolidation not full dump.",
            "Knowledge (RAG) ≠ user LTM — graph RAG ≠ Zep-style entity memory.",
            "Retention_days, FIFO session limits, LTM top_k enforced.",
            "LTM logical delete tombstones vectors where applicable.",
            "Org profile vs user profile separation.",
            "Consolidation triggers configured in MemoryProfile.",
            "RAG knowledge does not silently mutate user memory profile.",
            "Layer C context compiler spec lives in CONTEXT_ENGINEERING canon — not duplicated here.",
        ],
        "scale_probes": [
            "Long session exceeding FIFO — summarization path.",
            "Delegation namespace isolation under parallel subagents.",
            "Large LTM corpus with vector search + dedup.",
            "Entity graph memory under concurrent writes.",
        ],
        "overrides": "MemoryProfile · memory_runtime_bridge · BEFORE_MEMORY_WRITE hooks · TaskMemoryViewBinding on ToolWiringContext",
        "ci_scripts": [
            "uv run pytest tests/unit/memory/ -q",
            "uv run pytest tests/unit/applications/test_memory_profile_runtime_bridge.py -m gate -q",
        ],
        "production_baseline": "Mem0/Zep/Letta taxonomies · LangMem consolidation",
        "anti_patterns": "Global memory store · graph RAG as user memory · unscoped writes · agents with DB drivers",
        "appendix": "Appendix G",
    },
    {
        "id": "CONTEXT_ENGINEERING",
        "title": "Context Engineering Engine",
        "layers": "16",
        "mission": (
            "Audit the **Tier-1 context compiler engine**: plugin providers, budget/degradation, step-aware assembly, "
            "provenance, quality scoring, observability, and Tier-3 ContextProfile control plane — integrated with Harness."
        ),
        "code": """intergrax/runtime/nexus/context/ (context_engine.py target, context_compiler.py, context_manager.py)
intergrax/context_engineering/ (ContextEngine · providers)
intergrax/runtime/architecture/context_engineering.py · context_regression_benchmark.py
intergrax/contracts/context_assembly.py
intergrax/context/ (target contracts + plugin registry)
applications/_shared/context_runtime_bridge.py · context_wiring.py
intergrax/runtime/events/context_skill_recording.py · payloads/canonical.py""",
        "key_symbols": "ContextEngine · ContextSourceProvider · ContextFragment · ContextAssemblyRequest · AssembledContext · ContextCompiler · ContextManager · AgentContextBundle · ContextBudgetPolicy · TaskContextAssemblyOptions · ContextDecisionProfile · ContextProfile · DegradationLadder",
        "active_phases": "CE-LC Done · CE-DEPTH Done · §6.1av CE-MAINT closed (OTel, cost, baselines)",
        "known_gaps": "GAP-CTX-12 adaptive ranking **Frozen** → AHI-MAINT-04 · CE-LC register closed",
        "dimensions": [
            "ContextEngine.assemble() is the single target entry (CE-3) — no agent prompt concatenation.",
            "ContextSourceProvider plugin catalog with register_context_plugin (CE-2).",
            "Global token budget + DegradationLadder never-overflow (ADR-MEM-001 / ContextCompiler).",
            "Provenance on every included/excluded fragment.",
            "CONTEXT_ASSEMBLED / CONTEXT_TRIMMED events on all paths.",
            "BEFORE_CONTEXT_BUILD / AFTER_CONTEXT_BUILD hooks wired.",
            "ContextProfile drives Tier-3 presets (default, codebase, regulated_minimal).",
            "Step-aware ContextAssemblyRequest (step_kind, objective) on UAEP path (CE-4).",
            "Quality scoring integrated in DefaultContextRanker (CE-10).",
            "OTel spans on assemble/collect/budget (CE-9).",
            "RAG/Memory/Tool outputs enter via providers — not CE owning retrieval.",
            "Codebase preset uses WorkspaceContextProvider — not full repo dump.",
            "Context regression benchmark gates preset drift.",
            "Forbidden: Tier-2 imports of Nexus context internals for assembly.",
        ],
        "scale_probes": [
            "128k budget with multi-source fragments — degradation ladder trace.",
            "Graph node SUMMARY_ONLY tier under tight budget.",
            "Codebase 1k+ files — retrieval-first workspace provider.",
            "Delegation child explore preset — parent synthesis only.",
        ],
        "overrides": "ContextProfile · context_runtime_bridge · context_wiring · context_plugins[] · engine_preset · BEFORE_CONTEXT_BUILD hooks",
        "ci_scripts": [
            "uv run pytest tests/unit/runtime/nexus/context/ -m gate -q",
            "uv run pytest tests/unit/applications/test_context_wiring.py -m gate -q",
            "uv run pytest tests/acceptance/test_acceptance_context_compiler_long_session.py -q",
        ],
        "production_baseline": "Cursor-class context engine · Anthropic-style budgeting · LangGraph-style state injection patterns",
        "anti_patterns": "Agent-built prompts · silent fragment drop · string-heuristic source detection as final state · CE logic in Tier-2",
        "appendix": "Appendix L",
    },
    {
        "id": "MODALITY",
        "title": "Modality (Vision, Audio, ML)",
        "layers": "29",
        "mission": (
            "Audit **three modality planes** (A: LLM multimodal, B: ingest, C: deterministic ML): "
            "ToolRuntime surfaces, ModalityProfile, Celery/worker execution, cost/observability — no agent SDK bypass."
        ),
        "code": """intergrax/llm_adapters/ (Plane A attachments)
intergrax/rag/document_loaders/ (Plane B ingest)
intergrax/multimedia/ · intergrax/model_inference/
intergrax/tools/providers/vision|speech|ml/
integrations/providers/speech_provider/
modality_celery_wiring.py · ThreadPoolModalityInferenceExecutor
intergrax/runtime/observability/modality_counters.py""",
        "key_symbols": "ModalityProfile · VisionInferenceAdapter · ModelInferenceAdapter · VisionModelProfile · ModalityExecutionMode (CELERY) · AttachmentRef · tool_ids vision.*, speech.*, ml.*",
        "active_phases": "W-ML harness Done · W-ML remote Triton/HF incremental · Phase W-ML registry extensions",
        "known_gaps": "model_inference/ partial · remote serving incremental · Plane A vs C boundary discipline · online training out of scope",
        "dimensions": [
            "Plane C operations via ToolRuntime tools — not agent importing torch/onnx directly.",
            "Plane A LLM vision attachments typed via AttachmentRef.",
            "require_deterministic_cv forces Plane C not LLM vision guess.",
            "Plane B ingest separate from Plane C inference (document_loaders vs model_inference).",
            "Speech via IntegrationSpeechAdapter slugs — not vendor SDK in agent.",
            "ModalityProfile caps: max_media_bytes, allowed_planes, vision_model_ids.",
            "Celery broker path (INTERGRAX_MODALITY_CELERY_BROKER_URL) with thread-pool fallback.",
            "Modality metrics on tool_invocation_end / TASK_COMPLETED.",
            "V-COST fields populated for modality tool calls.",
            "HF Hub not on hot path for production profile.",
            "tool_ids Done status matches actual handler implementation.",
            "Context budget policy caps media contribution to prompt.",
        ],
        "scale_probes": [
            "Large image batch via worker pool vs Celery.",
            "Long audio transcription path.",
            "YOLO/ONNX in-process vs remote Triton.",
        ],
        "overrides": "ModalityProfile · ContextBudgetPolicy caps · integration speech_provider slugs · tts_voice_id",
        "ci_scripts": [
            "uv run pytest tests/unit/model_inference/ -q",
            "uv run pytest tests/unit/ -q -k modality",
        ],
        "production_baseline": "Triton/TorchServe/YOLO CV pipelines · Deepgram/ElevenLabs speech · ONNX edge · HF Inference Endpoints",
        "anti_patterns": "Agent imports cv2/torch directly · LLM vision for regulated CV when deterministic required · binary blobs inline in agent without AttachmentRef",
        "appendix": "N/A",
    },
    {
        "id": "OBSERVABILITY",
        "title": "Observability Spine (HOS)",
        "layers": "21, 30",
        "mission": (
            "Audit the **Harness Observability Spine**: layered event catalog (spine + event_kind), "
            "typed DiagnosticPayload, unified journal, causal trees, and operator reconstructability."
        ),
        "code": """intergrax/runtime/events/runtime_event.py · event_catalog.py · signals.py · event_bus.py
intergrax/runtime/nexus/tracing/ · ObservabilityEmitter · TraceScope
intergrax/runtime/events/payload_registry.py · persistence_conformance.py
scripts/check_observability_gates.py · check_event_catalog.py""",
        "key_symbols": "RuntimeEvent · event_kind · EventCategory · EventCatalog · emit_domain_signal · DiagnosticPayload · TraceComponent · ops filter hints",
        "active_phases": "OBS-BUS 0–7 Done · OBS-EVOL-9 Planned · ADR-OBS-001 · ADR-OBS-003",
        "known_gaps": "OBS-LC Done · OBS-EVOL-9 M0–M3 Done · runtime_event.v2 preview registered · product dashboards §6.3a → Phase K",
        "dimensions": [
            "Single spine — no per-agent private trace SQLite DBs.",
            "Spine event_type frozen ~50 at publication; domain extends via event_kind.",
            "Every spine RuntimeEventType has EventCatalog entry — phase + ops hint + payload.",
            "Tier-2/3 use emit_domain_signal — not new RuntimeEventType.",
            "DiagnosticPayload guard rejects raw dicts where typed schema required.",
            "parent_event_id via TraceScope — causal tree reconstructable.",
            "AGENT_SELECTED, STEP_FAILED, TOOL_*, POLICY_* emitted on hot paths.",
            "Journal export includes parser/RAG summaries where applicable.",
            "redact() before persist in production_mode.",
            "Extension SDK registers schema_id for custom payloads.",
            "correlation_id defaults to task_id consistently.",
            "persistence_conformance assert passes.",
            "Multi-agent graph callbacks emit typed graph_node.v1 payloads.",
            "Metrics layer third after events — not replacing journal.",
            "Debug APIs documented; PII never in prod journal content fields.",
            "check_harness_observability_wiring.py green for reference hosts.",
            "External OTLP export optional — canonical journal always populated.",
        ],
        "scale_probes": [
            "Long run 10k+ events — journal merge performance.",
            "Nested subagents — trace tree depth.",
            "Export backpressure to OTLP sink.",
        ],
        "overrides": "ObservabilityProfile · wire_nexus_observability() · PersistingTaskTraceEmitter · custom RuntimeEventBus handlers (Tier-3 plugins)",
        "ci_scripts": [
            "uv run python scripts/check_observability_gates.py",
            "uv run python scripts/check_event_catalog.py",
            "uv run pytest tests/unit/runtime/observability/ -q",
            "uv run pytest tests/unit/runtime/events/ -q",
        ],
        "production_baseline": "OpenTelemetry + structured logging · Datadog/Honeycomb SLO workflows · Langfuse/LangSmith LLM trace UX",
        "anti_patterns": "Per-agent trace DB · raw prompt/completion in prod journal · Tier-2 adding RuntimeEventType · metrics-only observability",
        "appendix": "Appendix H (observability mandatory vs optional)",
    },
    {
        "id": "RELIABILITY_FAILURE_AND_HITL",
        "title": "Reliability, Failure Model, and HITL",
        "layers": "22",
        "mission": (
            "Audit **failure taxonomy**, three retry layers, circuit breakers, checkpoint recovery, "
            "HITL gates, autonomy levels, and ReliabilityProfile wiring — safe-failure across runtime."
        ),
        "code": """intergrax/runtime/nexus/retry/retry_engine.py
intergrax/runtime/resilience/ · intergrax/runtime/human/
applications/_shared/reliability_wiring.py
intergrax/runtime/sandbox/ · intergrax/runtime/shadow/
autonomy_middleware · CancellationCoordinator · ActiveTaskRegistry
tests/acceptance/agent_os/ (04, 05, 05b HITL/checkpoint)""",
        "key_symbols": "RetryPolicy · RetryRecord · RetryHint · ResiliencePolicy · AutonomyLevel (MANUAL|ASK|AUTONOMOUS) · PauseRecord · RuntimeCheckpoint · HumanRequest · failure taxonomy (UserError, PolicyError, DependencyError, RuntimeError, QualityError)",
        "active_phases": "REL Done · REL-ADV Done · H-APP-WIRING.1 HTTP surfaces",
        "known_gaps": "REL-LC Done · §6.1av REL-MAINT Done · durable async queue → ORCH-MAINT-04 · LLM failover → LLM-MAINT-03",
        "dimensions": [
            "Three retry layers A/B/C not conflated (ORCH §52.1 cross-check).",
            "Agents emit RETRY hints — not internal adapter while-loops.",
            "HITL via Nexus/policy — not Slack webhook in agent.",
            "Checkpoint includes plan+graph+UAEP cursor — recoverable.",
            "Cancel cooperative via CancellationCoordinator.",
            "Guardrail denial composes with HITL escalation path.",
            "idempotency_key on side-effect tool retries.",
            "Circuit breaker from IntegrationProfile/resilience registry.",
            "AutonomyLevel obeys policy ceiling (MANUAL|ASK|AUTONOMOUS).",
            "PARTIALLY_COMPLETED only when policy allows.",
            "Trace shows retry reason and attempt count.",
            "ReliabilityProfile wired via reliability_wiring at Tier-3.",
            "Incident-worthy failures emit ops:alert-class signals.",
            "Recovery reboot strategy documented for long-running runs.",
        ],
        "scale_probes": [
            "Flaky integration with circuit breaker open.",
            "HITL queue backlog scenario.",
            "Cascading failure across graph nodes.",
            "30-day long-running monitor (ORCH §26 cross-ref).",
        ],
        "overrides": "ReliabilityProfile · OrchestrationProfile.max_run_retries · apply_reliability_task_defaults · require_human_approval · mid-run autonomy API (lab hosts)",
        "ci_scripts": [
            "uv run pytest tests/acceptance/agent_os/ -q -k 'hitl or checkpoint'",
            "uv run pytest tests/unit/runtime/nexus/retry/ -q",
        ],
        "production_baseline": "PagerDuty/Opsgenie escalation · enterprise approval queues · AWS well-architected retry/backoff",
        "anti_patterns": "Unbounded agent retry loops · HITL bypass for HIGH risk · checkpoint without UAEP cursor · conflated retry layers",
        "appendix": "Appendix H (risk/HITL)",
    },
    {
        "id": "TIER3_APPLICATION_ENVIRONMENT",
        "title": "Tier-3 Application Environment",
        "layers": "3, 28",
        "mission": (
            "Audit **deployable application hosts** (architecture §24–§51): "
            "ApplicationEnvironmentProfile as composition root, host contracts §25–§32, "
            "environment state §42, production gates §40/§46, evolution §49, platform ops §50, "
            "and author DX — without Nexus business logic or duplicate registries."
        ),
        "code": """applications/*/host/factory.py
intergrax/applications/contracts/environment_profile/ · application_registry.py · environment_health_score.py
intergrax/applications/_shared/environment_wiring.py · harness_host_runtime.py · environment_snapshot_wiring.py
intergrax/applications/_shared/reference_capability_bundle.py · environment_conformance.py
intergrax/applications/_shared/*_wiring.py (snapshot, migration, package, health_score, recovery, certification, …)
scripts/check_application_production_gates.py · check_environment_profile_bundle_schema.py
intergrax/cli/apps.py · envs.py · doctor_health_app.py · doctor_diff_app.py
docs/guides/APPLICATION_CREATION_GUIDE.md""",
        "key_symbols": "ApplicationEnvironmentProfile · HostMeta · CapabilityBundle · CognitionBundle · GovernanceBundle · DomainPolicyFragments · ProfileInvariantValidator · ApplicationManifest · EnvironmentSnapshot · bundle_normalized_payload",
        "active_phases": "H-APP Done · APP-CON-1..8 Done · APP-PROD-1..9 Done · APP-EVOL-1..7 Done · APP-EVOL-8 M1 Done · APP-OPS-1..4 Done · APP-CON-DX Done",
        "known_gaps": "T3-LC Done · §6.1av T3-MAINT Done · CFG-14 LKW → ORCH-MAINT-02 · marketplace UI §6.3 defer",
        "dimensions": [
            "ApplicationManifest + full profile on product hosts (§45 checklist).",
            "wire_application_environment() without getattr; package closure when conformance_check.",
            "Business logic only in Tier-2 agents — not Tier-3 host factory (§28).",
            "Capability routing via capabilities[] not class names (§37.4).",
            "ApplicationHost hooks: timeout, BLOCK on error, audit events (APP-CON-5).",
            "ApplicationEnvironmentState lifecycle sync on Nexus hooks (APP-CON-3).",
            "RunArtifactBundle on ApplicationRunSummary (APP-CON-6).",
            "Tier-3 scenario matrix / UC-A* evidence per reference host (APP-CON-7).",
            "Workspace shadow/sandbox cleanup on lifespan (APP-CON-8).",
            "EnvironmentSnapshot on intake + profile_snapshot_id (APP-EVOL-1).",
            "ApplicationMigration CI + typed sub-migrations (APP-EVOL-2/2b).",
            "CapabilityAlias sunset routing in STRICT (APP-EVOL-3).",
            "AgentCertification on STRICT roster (APP-EVOL-4).",
            "ApplicationRecoveryContract + ARCHITECTURE recovery docs (APP-EVOL-5).",
            "ApplicationEnvironmentDiff + doctor diff-app (APP-EVOL-6).",
            "ApplicationPackage + package.json from scaffold (APP-EVOL-7).",
            "APP-EVOL-8 M1: nested profile bundles + flat property shims (ADR-APP-003).",
            "bundle_normalized_payload on EnvironmentSnapshot digests (APP-EVOL-8.3).",
            "ProfileInvariantValidator cross-bundle checks (APP-EVOL-8).",
            "check_environment_profile_bundle_schema.py (APP-EVOL-8.7).",
            "STRICT capability graph deploy gate + blast radius (APP-OPS-1).",
            "ApplicationOperationalOwnership on product manifests (APP-OPS-2).",
            "EnvironmentHealthScore + doctor health-app (APP-OPS-3).",
            "ApplicationRegistry + EnvironmentRegistry + apps/envs CLI (APP-OPS-4).",
            "check_application_production_gates.py aggregates APP-PROD + APP-CON + APP-EVOL + APP-OPS.",
            "Roster ⊆ skill/tool profiles (EnvironmentSkillToolConsistencyCheck).",
            "IdentityProfile + budget enforcement on STRICT product hosts.",
            "Deploy triad present on scaffolded standard hosts.",
            "APPLICATION_CREATION_GUIDE.md aligns with §31 · §45 · §47.",
        ],
        "scale_probes": [
            "Cold start bootstrap all catalogs across four product hosts.",
            "Registry sync + health score for full STRICT fleet.",
            "Pre-deploy diff between manifest versions.",
            "strict_multi_agent_defaults() on legal/finance hosts.",
        ],
        "overrides": "Full ApplicationEnvironmentProfile · ApplicationManifest · OrganizationalPolicyEnvelope per tenant · registry artifacts in build/",
        "ci_scripts": [
            "uv run pytest tests/unit/applications/ -q",
            "uv run python scripts/check_application_production_gates.py",
            "uv run python scripts/check_application_registry.py",
            "uv run python scripts/check_application_health_score.py",
            "uv run python scripts/check_environment_profile_bundle_schema.py",
            "python scripts/check_harness_no_getattr.py",
        ],
        "production_baseline": "Reference hosts (legal, research, dispute_sim, local_workspace) · enterprise FastAPI agent host · ops registry + health score on release",
        "anti_patterns": "Business pipeline in applications/host · README as ops registry · getattr wiring · Nexus fork per product · skipping production gates",
        "appendix": "APPLICATION_CREATION_GUIDE.md · Appendix F · Appendix H",
    },
    {
        "id": "EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
        "title": "Experimentation and Developer Experience",
        "layers": "25–27, 30",
        "mission": (
            "Audit **DX and experimentation**: scaffold, eval harness, CI architecture gates, "
            "doctor CLI, lab environment, W-OPS evidence, TTFRun <1h goal, and gate maintenance discipline."
        ),
        "code": """intergrax/scaffold/
intergrax/runtime/architecture/ (eval, maturity gates, online_evaluation_registry.py)
intergrax/experiments/ · nexus_eval_runner.py
scripts/check_*.py (harness gates) · scripts/test.bat
scripts/phase_v_closeout_gate.py · phase_w_ops_evidence.py
docs/guides/AGENT_CREATION_GUIDE.md · HARNESS_ENVIRONMENT.md""",
        "key_symbols": "EvaluationProfile · ExperimentSession · OnlineEvaluationRegistry · maturity gate evidence · TTFRun metric · shadow workspace bindings",
        "active_phases": "EVAL · CRIT-V cross-ref · MVP-EVOL · DX · AA · W-OPS · Phase V G5 Production PRR",
        "known_gaps": "DX-LC Done · §6.1av DX-MAINT Done · GOV-PROD.1 dashboard backlog · polished SaaS UI explicit non-goal",
        "dimensions": [
            "Scaffold new-agent runnable through Nexus — not standalone script only.",
            "new-application emits profile+wiring+docker+ADR per Phase N.",
            "intergrax doctor diagnoses lab stack accurately.",
            "Gate scripts pass after harness change (mandatory verification set).",
            "check_docs_domain_pairs enforces 22 pairs.",
            "Eval registry trends before promotion (require_baseline_for_release).",
            "Shadow workspace observe-only compare path works.",
            "Acceptance agent_os suite covers OS claims.",
            "Extension author guide aligned with plugin entry points.",
            "Phase V PRR evidence for production readiness claims.",
            "Structured output required on agent contracts per guide.",
            "Trace on every decision per DX checklist.",
            "Tier-0 reused in scaffold — not duplicated stubs.",
            "W-OPS release cycle docs match scripts/build artifacts.",
            "Single plan pair per domain — no orphan implementation docs.",
        ],
        "scale_probes": [
            "Full CI gate suite runtime on developer machine.",
            "Parallel eval workloads in lab.",
            "TTFRun: idea → first Nexus run timing evidence.",
        ],
        "overrides": "Scaffold templates · EvaluationProfile.shadow_eval_enabled · lab vs strict production defaults",
        "ci_scripts": [
            "uv run pytest -m gate -q",
            "python scripts/check_harness_no_getattr.py",
            "uv run python scripts/check_observability_gates.py",
            "uv run python scripts/check_docs_domain_pairs.py",
            "scripts/test.bat unit",
        ],
        "production_baseline": "Cursor agent iteration UX · Braintrust/prompt regression CI · platform engineering PRR culture",
        "anti_patterns": "Skipping gates after harness change · duplicate DX docs · eval only in notebooks not registry · false PRR without evidence",
        "appendix": "EXTENSION_AUTHOR_GUIDE",
    },
    {
        "id": "ADAPTIVE_HARNESS_INTELLIGENCE",
        "title": "Adaptive Harness Intelligence (L4)",
        "layers": "L4 AHI",
        "mission": (
            "Audit **L4 adaptive loops**: bounded self-tuning, utility function, shadow→canary→prod promotion, "
            "policy-bounded routing, signal emission — honest W-ADAPT Done vs product-gated L4 thresholds."
        ),
        "code": """intergrax/runtime/adaptive/ (signal_emission.py, SignalCollector, adaptive_governance.py, VerificationLoop)
intergrax/runtime/adaptive/cost_optimization.py
intergrax/runtime/architecture/ (ProcessPatternMiner W-ADAPT-6, ExecutionStrategyEngine)
runtime_governance_bridge.py
scripts/phase_w_adapt_report.py · scripts/phase_v_closeout_gate.py""",
        "key_symbols": "HarnessOutcomeSignal · AdaptiveLoopEnvelope · AdaptiveLoopKind · ProfileVersion · Utility U · AdaptationEngine · AdaptationExecutor · AdaptiveProfile · LLMCallSummary on signals",
        "active_phases": "W-ADAPT W0–W7 Done (70/70) · Phase V L4 evidence · L4 adaptive critic thresholds product-gated",
        "known_gaps": "AHI-LC Done · §6.1av AHI-MAINT Done · L4 auto-apply requires explicit product gate · live routing owner LLM-MAINT-02",
        "dimensions": [
            "Adaptations versioned with rollback pointer.",
            "PolicyEngine never bypassed by adaptive executor.",
            "Post-task outcome signals emitted (HarnessOutcomeSignal).",
            "Utility U computed from documented function.",
            "Proposals pass capability graph impact analysis.",
            "Shadow mode before canary — evidence in registry.",
            "Human gate on apply in production profiles.",
            "Tier-1 remains domain-agnostic — no Problem Radar business logic in core.",
            "Classical RL explicitly NOT the adaptation model.",
            "Evaluation registry consumes adaptive outcomes.",
            "Cost optimization under policy cap.",
            "Process miner emits proposals — not auto Tier-2 code generation.",
            "Kill switches and cooldowns on AdaptiveProfile.",
            "Observability: why route/model/tool changed.",
        ],
        "scale_probes": [
            "≥10% utility improvement target on golden scenarios.",
            "Rollback <5 min evidence.",
            "Feedback delay vs adaptation lag.",
        ],
        "overrides": "AdaptiveProfile on Tier-3 · shadow_eval_enabled · observe/recommend/apply/verify lifecycle modes",
        "ci_scripts": [
            "uv run python scripts/phase_w_adapt_report.py",
            "uv run pytest tests/unit/runtime/adaptive/ -q",
        ],
        "production_baseline": "Canary/feature-flag systems (LaunchDarkly/Unleash) · contextual bandits + regression gates — NOT OpenAI RLHF",
        "anti_patterns": "Unapproved policy mutation · unconstrained model switching · adaptive loop without shadow · RLHF-style training in Tier-1",
        "appendix": "N/A",
    },
    {
        "id": "CRITIC_VERIFICATION",
        "title": "Critic and Verification (CVL)",
        "layers": "25 (depth)",
        "mission": (
            "Audit **Critic Verification Layer**: L0/L1 gateways, evaluator loops, LLM-as-judge via ToolRuntime, "
            "trajectory eval, HITL on borderline — integrated in runtime, not bolt-on scripts."
        ),
        "code": """intergrax/runtime/critic/critic_orchestrator.py · contracts.py · policy_bridge.py
intergrax/runtime/critic/evaluator_loop_executor.py · CriticTraceEmitter
intergrax/runtime/nexus/validation_engine.py
intergrax/tools/providers/eval/judge.py
applications/_shared/critic_runtime_bridge.py · critic_assembly_resolver.py
eval/nexus_eval_runner.py""",
        "key_symbols": "CriticProfile · CriticRequest · CriticVerdict · L0Gateway · L1Gateway · EvaluatorLoopSpec · RubricSpec · ValidationResult · eval.judge · eval.trajectory · eval.record_observation",
        "active_phases": "CRIT-V 0–7 + FOLLOWUP Done · CVL-LC-1/2 layer completion (2026-06-13) · FAUDIT-EVAL.1 closed",
        "known_gaps": "CVL-LC Done · §6.1av CVL-MAINT Done · L4 thresholds Frozen → AHI · FLOW-8 host → §6.3",
        "dimensions": [
            "L0 static/rule gateway before L1 LLM judge always.",
            "Judge LLM ≠ producer LLM (separate profile/ref).",
            "eval.judge invoked via ToolRuntime — not direct adapter in agent.",
            "ValidatorAgents as graph nodes allowed — not parallel eval stack.",
            "No parallel SQLite eval store per agent.",
            "require_critic_on_completion fail-closed when profile set.",
            "Critic steps in trace (CriticTraceEmitter).",
            "Registry observations via eval.record_observation.",
            "Domain rubrics live in Tier-2 — not Nexus business rules.",
            "guardrail_scan merges into L0 where configured.",
            "node_partial vs graph_final verify scopes correct.",
            "EvaluatorLoopExecutor wired for CoordinationPattern.EVALUATOR_LOOP.",
            "Semantic NexusEvalRunner mode for harness eval.",
            "False positive/negative handling and retry semantics documented.",
        ],
        "scale_probes": [
            "Evaluator-loop until budget exhausted.",
            "CFG-16/CFG-20 strict multi-agent critic.",
            "High-volume eval latency impact on user path.",
        ],
        "overrides": "CriticProfile + EvaluationProfile · require_critic_on_completion · separate critic LLMProfile · CoordinationPattern.EVALUATOR_LOOP",
        "ci_scripts": [
            "uv run pytest tests/unit/runtime/critic/ -q",
            "uv run pytest tests/unit/tools/providers/eval/ -q",
        ],
        "production_baseline": "Guardrails AI Hub validators · Braintrust/Phoenix LLM-as-judge · legal/finance human sign-off workflows",
        "anti_patterns": "Judge same model as producer · critic as optional script · duplicate eval store · skipping L0 for speed",
        "appendix": "N/A",
    },
    {
        "id": "REASONING_AND_COGNITION",
        "title": "Reasoning and Cognition",
        "layers": "7",
        "mission": (
            "Audit **three cognition planes** (Nexus planning, agent on_next_step, tool planning): "
            "TaskClassifier, typed plans, DecisionRecord, planner strategies, reasoning failure taxonomy."
        ),
        "code": """intergrax/runtime/nexus/task_classifier.py
intergrax/runtime/nexus/planning/task_planner.py · EngineBackedNexusPlanner · nexus_llm_plan_builder.py
applications/_shared/graph_spec_to_plan.py
intergrax/runtime/nexus/tools/catalog_tool_planner.py · tool_planning_service.py · tool_selection.py
intergrax/agents/authoring/patterns/ (ReAct, plan_execute, …)
intergrax/contracts/decision_record.py
intergrax/prompts/registry/ (planner prompt ids)""",
        "key_symbols": "TaskClassification · NexusPlan/PlanStep · StepOutcome · ToolPlanDecision · DecisionRecord (decision_record.v1) · IntentRoute · ReasoningProfile · OrchestrationProfile.planner_kind/classifier_kind",
        "active_phases": "COG-DEPTH Done · COG-1..6 · COG-3.* classifier · ORCH-CONFIG.1 · COG-OBS residuals",
        "known_gaps": "ReasoningFailureKind enum on trace (COG-6 target) · allow_dynamic_replan partial · retired RuntimeEngine engine planner (ACP-CLOSE-LEG-5)",
        "dimensions": [
            "Classification precedes side-effectful execution.",
            "Plans are typed (NexusPlan) — not free-text-only.",
            "LLM planner falls back to TaskPlanner on parse failure.",
            "DecisionRecord on UAEP steps (decision_record.v1 schema).",
            "Nexus planning emits decision records (COG-4).",
            "Prompt registry ids for planners — not inline strings (COG-2).",
            "Tool planning plane separate from Nexus graph planning.",
            "MULTI_AGENT semantics ≠ cross-role pipeline conflation.",
            "Engine planner requires llm_adapter at bootstrap.",
            "Reasoning failures classified separately from tool/runtime failures.",
            "Planner LLM identity separable from producer LLM.",
            "graph_spec seeding respects trigger_capabilities.",
            "catalog_tool_planner single-pass — ReAct status cross-ref TOOL-ENG-6.",
            "IntentRoute table maps orchestration tokens correctly.",
            "DECISION_EMITTED gate regression (FLOW-12) green.",
        ],
        "scale_probes": [
            "research.pipeline 2-step planner.",
            "Engine planner with multiple routable agent_ids.",
            "Classifier fan-out on ambiguous intake.",
            "Replanning when allow_dynamic_replan enabled.",
        ],
        "overrides": "OrchestrationProfile · ReasoningProfile.tool_planner_prompt_id · IntentRoute · graph_spec · classifier_kind=rules|llm",
        "ci_scripts": [
            "uv run pytest tests/unit/runtime/nexus/planning/ -q",
            "uv run pytest tests/unit/runtime/nexus/tools/test_tool_planning_constraints.py -q",
            "uv run pytest tests/unit/runtime/nexus/tools/test_tool_selection_strategy.py -q",
        ],
        "production_baseline": "OpenAI/o1-style task decomposition · intent routers · Google ADK structured planners",
        "anti_patterns": "Free-text plan only · reasoning+tools in one agent method · no DecisionRecord · ad-hoc prompt strings for planners",
        "appendix": "Appendix I §I.4",
    },
    {
        "id": "ELASTIC_CAPACITY_AND_SCALING",
        "title": "Elastic Capacity and Platform Scaling (ECP)",
        "layers": "30",
        "mission": (
            "Audit **Elastic Capacity Plane**: signals, ScalingPolicy, backpressure vs autoscale distinction, "
            "queueing/workers, K8s integration path, ECP-DEPTH target modules — honest L0–L2 vs plan targets."
        ),
        "code": """intergrax/queueing/ · intergrax/distributed/
integrations/providers/cloud_platform/kubernetes/
integrations/providers/message_bus/celery/
intergrax/runtime/architecture/multi_agent_contention_simulation.py
intergrax/runtime/observability/harness_slos.py
target: intergrax/runtime/capacity/ (ECP-DEPTH ECP-1..8)
docs/adr/entries/2026-06-08/ADR-SCALE-001.md · ADR-SCALE-002.md""",
        "key_symbols": "ScalingProfile (target) · ScalingPolicy · ScalingAction · ScalingSignal · CapacitySignalCollector · ScalingProvisioner · SIG_QUEUE_DEPTH · GRAPH_BACKPRESSURE",
        "active_phases": "ECP-DOC · ECP-DEPTH (ECP-1..8, ECP-OBS) · ADR-SCALE-001/002 · cross-ref W-OPS.4 SLIs · ORCH GRAPH_BACKPRESSURE",
        "known_gaps": "ECP-LC Done · §6.1av ECP-MAINT Done · live K8s soak manual runbook · ingress slug → INT-MAINT-04",
        "dimensions": [
            "ECP control loop async outside Nexus hot path.",
            "Provisioning via integrations/tools — not Nexus importing K8s SDK.",
            "Backpressure (GRAPH_BACKPRESSURE) ≠ auto-scale — documented distinction.",
            "Hysteresis + cooldown on scaling rules (target ECP).",
            "Scale actions idempotent with NOTIFY_ONLY at max replicas.",
            "Tenant isolation on capacity signals.",
            "K8s HPA complementary — Intergrax rules orchestrate ceilings.",
            "Agent topology scaling (dimension B) separate from worker scaling.",
            "SCALE_* trace events when ECP implemented.",
            "Fail-safe on provisioner error — no runaway scale-up.",
            "Tier-3 owns deploy manifests (Helm/HPA in applications/*/docker/).",
            "PolicyEngine/HITL on scale-up when profile requires.",
            "Queue depth from intergrax/queueing/task_index.py as signal.",
            "Multi_agent_contention_simulation aligns with architecture claims.",
            "Honest: mark L0/L1 where ECP-DEPTH not yet implemented.",
        ],
        "scale_probes": [
            "GRAPH_BACKPRESSURE rate under max_inflight_nodes.",
            "Queue depth burst → worker autoscale (target ECP-5).",
            "Modality Celery W-OPS.12 cross-ref.",
            "Multi-replica Nexus vs in-process concurrency caps.",
        ],
        "overrides": "ScalingProfile on ApplicationEnvironmentProfile (target) · OrchestrationProfile.max_inflight_nodes ceiling · Helm/HPA per host",
        "ci_scripts": [
            "uv run pytest tests/unit/runtime/architecture/test_multi_agent_contention_simulation.py -q",
            "uv run pytest tests/unit/queueing/ -q",
        ],
        "production_baseline": "Kubernetes HPA/VPA · Celery autoscale · nginx upstream scaling · cloud autoscaler APIs · Prometheus SLI runbooks",
        "anti_patterns": "Nexus hot-path synchronous provisioning · conflating backpressure with autoscale · missing cooldown · scale without tenant bounds",
        "appendix": "N/A — cross-ref OBSERVABILITY (SLIs) and ORCHESTRATION (backpressure)",
    },
]


DEFAULT_PLAN_SCOPE = (
    "`## 6.` open queue rows only · gap/remediation registers tied to **Known open gaps** "
    "and **Active plan phases** · skip `(closed)`, `(complete)`, `Archived` unless re-validating a listed gap"
)


def _bullets(items: list[str], numbered: bool = False) -> str:
    if numbered:
        return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))
    return "\n".join(f"- {x}" for x in items)


def _context_budget_block(*, did: str, layers: str) -> str:
    return f"""## 0. Context budget (mandatory)

**Load first:** [`docs/guides/audit_slices/{did}.md`](../guides/audit_slices/{did}.md) — compact slice (layers **{layers}**); replaces bulk IDEAL + AUDIT_MAP + full plan/arch reads.

- One domain per chat · grep with path filters · respect `.cursorignore`
- Plan/arch: hub read-scope + **at most one** satellite (`plan/plan/` or `architecture/arch/`)
- Run **only** §10 scripts · no full-suite pytest unless listed · no `docs/audit_results/` unless RESUME

---
"""


def render(domain: dict) -> str:
    did = domain["id"]
    title = domain["title"]
    layers = domain["layers"]
    mission = domain["mission"]
    code = domain["code"]
    key_symbols = domain["key_symbols"]
    active_phases = domain["active_phases"]
    known_gaps = domain["known_gaps"]
    dims = domain["dimensions"]
    if len(dims) > 22:
        extra = len(dims) - 20
        dims = dims[:20] + [
            f"… plus {extra} more rows — grep `architecture/{did}.md` §21–§40 and plan hub §6.1 (do not load full arch)"
        ]
    scale = domain["scale_probes"]
    overrides = domain["overrides"]
    ci = domain["ci_scripts"]
    baseline = domain["production_baseline"]
    anti = domain["anti_patterns"]
    appendix = domain["appendix"]
    adr = domain.get("adr")
    context_budget = _context_budget_block(did=did, layers=layers)

    appendix_block = ""
    if appendix != "N/A":
        if "APPLICATION_CREATION_GUIDE" in appendix:
            appendix_block = "5. `@docs/guides/APPLICATION_CREATION_GUIDE.md` — on demand only (`.cursorignore`)\n"
        else:
            appendix_block = f"5. `@docs/guides/AGENT_CREATION_GUIDE.md` **{appendix}** — on demand\n"

    adr_block = ""
    if adr:
        adr_parts = [p.strip() for p in adr.split("·") if p.strip()]
        adr_links = " · ".join(
            f"[`{part}`](../../adr/{part}.md)" for part in adr_parts
        )
        adr_block = f"**ADR:** {adr_links}  \n"

    ci_block = "\n".join(ci)

    return f"""# {title} — Domain Layer Audit Instruction

**Status:** Audit control prompt (copy-paste for LLM agents)  
**Domain pair:** [`architecture/{did}.md`](../architecture/{did}.md) · [`plan/{did}.md`](../plan/{did}.md)  
**Audit map layers:** {layers} · compact slice: [`audit_slices/{did}.md`](../guides/audit_slices/{did}.md)  
{adr_block}**Shared checklist:** [audit/README.md](README.md#shared-production-harness-checklist)

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

domain: {did}
mode: audit-only
focus:

# mode: audit-only | audit-and-fix
# focus: optional narrow slice — e.g. "ingest only", "ToolRuntime policy path", "CFG-14 host wiring"

# ═══ END USER CONFIG ═══

# TASK: Deep production audit — {title} (`{did}`)

You are an **implementation audit agent** for the Intergrax Harness AI platform.

Perform a **rigorous, evidence-backed audit** of the **{title}** domain. You must inspect **architecture canon, implementation plan, source code, tests, and CI gates** and compare against **production-grade systems** in this problem space.

**Do not** produce a shallow documentation survey. **Do not** declare the whole platform complete.

## Mission

{mission}

## Key symbols and contracts

{key_symbols}

## Active plan phases (verify status vs code reality)

{active_phases}

## Known open gaps — re-validate every item (closed / still open / partial)

{known_gaps}

---

{context_budget}

## 1. Canonical reads (order)

1. **`docs/guides/audit_slices/{did}.md`** — mandatory; follow slice plan/arch/IDEAL scope lines
2. `docs/architecture/{did}.md` — hub read-scope + one `architecture/arch/` satellite max
3. `docs/plan/{did}.md` — hub + one `plan/plan/` satellite max
4. `docs/audit/README.md` — shared production Harness checklist
{appendix_block}**Do not** load full `IDEAL_HARNESS_AI_ARCHITECTURE.md` or `INTEGRAX_HARNESS_AUDIT_MAP.md` unless slice says so.
---

## 2. Code entry (grep first)

See **Code entry** in `docs/guides/audit_slices/{did}.md` — then inspect:

```text
{code}
```

Grep `tests/unit/`, `tests/integration/`, `tests/acceptance/` for this domain.

---

## 3. Domain-specific audit dimensions

For **each** item: **Yes / Partial / No / Unknown** + **evidence** (`path:symbol` or `test_name`).

{_bullets(dims, numbered=True)}

---

## 4. Workload and scale probes

For each probe describe **actual code path**, limits, and failure mode:

{_bullets(scale)}

---

## 5. Tier-3 / Tier-2 override surfaces

Confirm overrides are **wired in code**, not documentation-only:

{overrides}

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

Compare against: **{baseline}**

State explicitly:

| Category | Your finding |
|----------|--------------|
| Matches L3 Production Harness OS | … |
| L2 or below (name gaps with plan IDs) | … |
| Intentional design boundary | … |
| **incomplete_wiring** / missing wiring | … |

---

## 8. Anti-patterns (must not be present)

{_bullets([anti] if isinstance(anti, str) else anti)}

---

## 9. Maturity scoring

Per `INTEGRAX_HARNESS_AUDIT_MAP.md` §5 (L0–L4). Report **score before**, **target milestone**, **evidence**, **remaining risks**.

If architecture doc has a maturity table (e.g. RAG §Maturity score), reconcile with code findings.

---

## 10. Verification — run and cite

```bash
{ci_block}
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
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        path = OUT / f"{domain['id']}.md"
        path.write_text(render(domain), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
