# Symbol index (F5 — token-efficient code lookup)

Use this **before** repo-wide semantic search. Grep the path directly.

| Symbol | Primary definition |
|--------|-------------------|
| `HarnessKernel` | `intergrax/runtime/kernel/step_kernel.py` L113 |
| `NexusLoop` | `intergrax/runtime/nexus/nexus_loop.py` L89 |
| `StepOutcome` | `intergrax/agents/authoring/step_outcome.py` L17 |
| `AgentStepContext` | `intergrax/contracts/agent_step_context.py` L16 |
| `AgentRunRequest` | `intergrax/contracts/agent_run.py` L118 |
| `AgentRunTrace` | `intergrax/contracts/agent_run_trace.py` L110 |
| `ToolRuntime` | `intergrax/runtime/nexus/tools/tool_runtime.py` L129 |
| `PolicyEngine` | `intergrax/runtime/policy/policy_engine.py` L34 |
| `RuntimeEvent` | `intergrax/runtime/events/runtime_event.py` L79 |
| `UnifiedTaskRunner` | `intergrax/runtime/task/unified_task_runner.py` L21 |
| `HarnessApplication` | `intergrax/harness/app.py` L27 |
| `ApplicationHost` | `intergrax/harness/application_host.py` L14 |
| `AgentRegistry` | `intergrax/runtime/registry/agent_registry.py` L32 |
| `CapabilityGraph` | `intergrax/runtime/architecture/capability_graph.py` L65 |
| `CognitiveAgent` | `intergrax/agents/authoring/patterns/base.py` L36 |
| `DecisionRecord` | `intergrax/contracts/decision_record.py` L14 |
| `IntegrationProfile` | `intergrax/integrations/registry/profile.py` L41 |
| `ContextEngine` | `intergrax/context/protocols.py` L93 |
| `RetrievalEngine` | _grep repo_ |
| `StepLLMRouter` | `intergrax/agents/authoring/llm_router.py` L73 |
| `EffectiveAgentRunEnvironment` | `intergrax/agents/run_environment.py` L35 |

## Common paths

| Area | Path |
|------|------|
| Nexus core | `intergrax/runtime/nexus/` |
| Orchestration | `intergrax/runtime/nexus/orchestration/` |
| Tool runtime | `intergrax/tools/` |
| Agent contracts | `intergrax/runtime/nexus/agent/` |
| Tier-3 hosts | `applications/` |
| Tier-2 agents | `agents/` |

Regenerate: `uv run python scripts/generate_symbol_index.py`
