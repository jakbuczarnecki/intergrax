# © Artur Czarnecki. All rights reserved.

"""HARNESS-01 — canonical execution matrix and zero-bypass evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HarnessBypassStatus = Literal[
    "CANONICAL",
    "AUTHORIZED_INTERNAL",
    "BYPASS",
    "NOT_APPLICABLE",
]

FindingSeverity = Literal[
    "BLOCKER",
    "NON-BLOCKING DEBT",
    "INTENTIONAL DESIGN",
    "FALSE POSITIVE",
]


@dataclass(frozen=True, slots=True)
class Harness01EvidenceRef:
    pytest_node_id: str
    kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Harness01ExecutionRow:
    flow: str
    entry: str
    authority: str
    governance: str
    policy: str
    canonical_consumer: str
    evidence: str
    side_effect: str
    bypass_status: HarnessBypassStatus
    proof: tuple[Harness01EvidenceRef, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Harness01FindingRow:
    finding: str
    severity: FindingSeverity
    flow: str
    required_action: str


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


def _h01(test_name: str) -> str:
    return _nid("tests/qualification/harness_01/test_harness_01_gates.py", test_name)


def _u3(test_name: str) -> str:
    return _nid(
        "tests/unit/runtime/architecture/test_platform_execution_unification_u3_agent_plugin_execution_closure.py",
        test_name,
    )


def _plug02(test_name: str) -> str:
    return _nid(
        "tests/unit/runtime/nexus/tools/test_plug_02_r1_public_invocation_pattern_evidence.py",
        test_name,
    )


def _plug03(test_name: str) -> str:
    return _nid("tests/qualification/plug_03/test_plug_03_gates.py", test_name)


def _host01(test_name: str) -> str:
    return _nid("tests/qualification/host_01/test_host_01_gates.py", test_name)


def _bg01(test_name: str) -> str:
    return _nid("tests/qualification/bg_01/test_bg_01_gates.py", test_name)


def _gr10(test_name: str) -> str:
    return _nid("tests/qualification/governance/strategy/test_gr10_gates.py", test_name)


def _gr10_r8(test_name: str) -> str:
    return _nid(
        "tests/qualification/governance/strategy/test_gr10_r8_orchestration_inner_governance_qualification.py",
        test_name,
    )


def _ref(node_id: str, *kinds: str) -> Harness01EvidenceRef:
    return Harness01EvidenceRef(node_id, kinds)


HARNESS_01_EXECUTION_MATRIX: tuple[Harness01ExecutionRow, ...] = (
    Harness01ExecutionRow(
        "Standard single-agent run (NexusLoop)",
        "HostTaskExecutionPort / NexusWorkerRuntime → NexusLoop.run",
        "RuntimeState.run_id · ExecutionRequest.agent_id",
        "DefaultRootExecutionLauncher · agent_runtime_governance",
        "Declarative enforcer · scope policy · sandbox gate",
        "NexusLoop → tool_loop → RuntimeToolInvoker",
        "ToolCallTrace · state.tool_traces · run trace store",
        "ToolExecutor via registry",
        "CANONICAL",
        (_ref(_host01("test_host_q1_production_surfaces_use_host_task_execution_port")),),
    ),
    Harness01ExecutionRow(
        "Agentic / bounded tool-loop execution",
        "resolve_invocation_pattern → NexusToolInvocationPattern.execute",
        "RuntimeConfig.tool_invocation_pattern_id",
        "Inner execution guard · tool call budget",
        "ToolScopePolicy · meaningful side-effect authorization",
        "tool_loop.invoke_prepared_tool_execution_request",
        "PlannedToolCallOutcome.trace",
        "RegistryToolExecutor",
        "CANONICAL",
        (_ref(_plug03("test_tools_profile_selection_executes_custom_not_catalog_default"), "SELECTION"),),
    ),
    Harness01ExecutionRow(
        "Custom ToolInvocationPattern (public plugin)",
        "ToolInvocationPattern.execute via PublicToolInvocationPatternBridge",
        "state.request.agent_id (authoritative at invoker)",
        "Same as tool-loop",
        "Same as invoker stack",
        "bounded ToolInvocationInvokerPort → invoke_prepared_tool_execution_request",
        "bridge evidence recorder → ToolCallTrace",
        "RuntimeToolInvoker",
        "CANONICAL",
        (
            _ref(_plug02("test_public_pattern_single_invoke_via_bounded_tool_loop"), "CANONICAL_CONSUMPTION"),
            _ref(_h01("test_harness_01_public_pattern_bridge_does_not_trust_port_agent_id"), "GATE"),
        ),
    ),
    Harness01ExecutionRow(
        "UAEP / agent step tool capability",
        "exec_ctx.invoke_tool → BoundToolGateway",
        "ToolRequest.agent_id + RuntimeState",
        "MiddlewarePipeline tool hooks",
        "ToolAccessPolicy.is_tool_allowed",
        "RuntimeToolGateway → catalog_dispatch → RuntimeToolInvoker",
        "ToolResponse + gateway trace step",
        "Catalog / capability tools",
        "CANONICAL",
        (_ref(_u3("test_u3_uaep_tool_path_uses_runtime_tool_gateway_not_local_invoker"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "Application-owned / platform tools",
        "ToolProfile → build_registry_from_profile",
        "Tool registry composition root",
        "Production tool invoker composition",
        "Plugin admission + profile selection",
        "RuntimeToolInvoker (single composition root)",
        "PLUG-FINAL tool evidence domain",
        "Tool handlers",
        "CANONICAL",
        (_ref(_plug03("test_tools_discovered_but_unselected_not_in_execution_registry"), "FAIL_CLOSED"),),
    ),
    Harness01ExecutionRow(
        "Integration-backed tools",
        "IntegrationProfile → provider binding",
        "Integration registry / resolver",
        "Governance on tool path",
        "Integration + tool contracts",
        "Tool registry handler (vendor inside provider)",
        "Tool invocation trace",
        "External APIs via integration adapter",
        "CANONICAL",
        (_ref(_plug03("test_integration_explicit_slug_activates_fixture_provider"), "SELECTION"),),
    ),
    Harness01ExecutionRow(
        "Background / scheduled host task execution",
        "TaskQueue worker → HostTaskExecutionPort",
        "BackgroundExecutionIdentity",
        "Root launcher governance",
        "Tenant provenance gates",
        "Same Nexus host stack as foreground",
        "Task + run identity propagation",
        "Host-composed side effects only via runtime",
        "CANONICAL",
        (_ref(_bg01("test_bg_q1_production_background_execution_surfaces_use_host_port"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "Memory / session operations",
        "SessionManager / memory control plane contracts",
        "tenant_id · session_id from runtime context",
        "Memory profile + plugin admission",
        "Tenant isolation invariants",
        "Memory provider implementations (vendor-internal)",
        "Session/memory provenance evidence",
        "Storage backends",
        "CANONICAL",
        (_ref(_plug03("test_plug03_session_storage_canonical_session_manager_consumer"), "SELECTION"),),
    ),
    Harness01ExecutionRow(
        "RAG retrieval",
        "RetrievalService / nexus.rag capability",
        "RAG profile composition",
        "Tool gateway for retrieve tool",
        "Profile-selected retriever/reranker",
        "RAG managers — no agent→vectorstore direct",
        "Retrieve tool trace",
        "Vector store inside provider",
        "CANONICAL",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_retriever_entry_point_uses_retrieval_service",
                ),
                "SELECTION",
            ),
        ),
    ),
    Harness01ExecutionRow(
        "Policy-controlled / declarative tools",
        "RuntimeToolInvoker._prepare_invocation",
        "state + attempt authorization",
        "Declarative enforcer · HITL grants",
        "require_meaningful_side_effect_authorization",
        "Invoker only",
        "Policy trace diagnostics",
        "Guarded side effects",
        "CANONICAL",
        (_ref(_gr10_r8("test_gr10_r8_composition_requires_guard_on_production_mode_ast"), "FAIL_CLOSED"),),
    ),
    Harness01ExecutionRow(
        "Security middleware / tool hooks",
        "RuntimeToolGateway.invoke → run_tool_call_hooks",
        "MiddlewarePipeline on gateway",
        "Hook chain before/after invoke",
        "Denied tools fail at gateway",
        "Gateway inner invoke",
        "Hook context + ToolResponse status",
        "Middleware-controlled path",
        "CANONICAL",
        (_ref(_h01("test_harness_01_tool_gateway_wraps_invoke_with_hooks"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "RuntimePlugin hooks",
        "HOST_COMPOSED RuntimePlugin before/after/shutdown",
        "Host composition only",
        "Plugins cannot mint invoker",
        "No alternate tool port from plugin hook",
        "Hooks observe — execution stays in runtime",
        "Plugin evidence (PLUG-FINAL)",
        "Trust boundary: host-composed code",
        "AUTHORIZED_INTERNAL",
        (),
        "Sandboxing arbitrary plugin Python is out of HARNESS-01 scope; boundary audited only.",
    ),
    Harness01ExecutionRow(
        "Multi-agent / delegation",
        "Orchestration graph / delegation contracts",
        "Per-child run identity + narrowed grants",
        "Orchestration inner guard · delegation budget",
        "Child must not inherit parent tools implicitly",
        "Separate RuntimeState per delegated run",
        "Per-run tool_traces",
        "Delegated invoker stack",
        "CANONICAL",
        (_ref(_gr10_r8("test_gr10_r8_orchestration_inner_governance_qualified"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "LLM model invocation",
        "LLM adapter / provider registry",
        "Model profile + runtime config",
        "Token/context budget (CE-02 path)",
        "Policy on model usage where configured",
        "intergrax.llm adapters — no vendor SDK in runtime execution",
        "LLM trace / modality metrics",
        "Provider HTTP inside adapter",
        "CANONICAL",
        (_ref(_h01("test_harness_01_runtime_tier_no_direct_vendor_llm_imports"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "Context assembly",
        "ContextCompiler / ContextPlanner (CE-02 closed)",
        "Context engine profile",
        "Budget degradation policies",
        "Context plugin registry",
        "Nexus context assembly — not host string concat",
        "Context evidence / compaction trace",
        "Context engine outputs",
        "CANONICAL",
        (_ref(_nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q1_single_budget_resolution_entry"), "GATE"),),
    ),
)


HARNESS_01_ZERO_BYPASS_FINDINGS: tuple[Harness01FindingRow, ...] = (
    Harness01FindingRow(
        "ToolInvocationInvokerPort.agent_id is ignored at bounded bridge; "
        "authorization uses state.request.agent_id via invoke_prepared_tool_execution_request",
        "NON-BLOCKING DEBT",
        "Custom ToolInvocationPattern",
        "Future: remove redundant parameter or assert equality with authoritative agent_id",
    ),
    Harness01FindingRow(
        "platform_proofs scenarios may call RuntimeToolInvoker directly for scenario isolation",
        "INTENTIONAL DESIGN",
        "Proof / demo applications",
        "Keep proofs out of production application host trees; not a production bypass",
    ),
)

HARNESS_01_AUTHORIZED_RUNTIME_TOOL_INVOKER_CALLSITE_FILES: frozenset[str] = frozenset(
    {
        "intergrax/runtime/nexus/tools/catalog_dispatch.py",
        "intergrax/runtime/nexus/tools/catalog_context.py",
        "intergrax/runtime/nexus/tools/tool_loop.py",
        "intergrax/runtime/nexus/tools/patterns/deterministic_chain.py",
    }
)

HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py",
        "intergrax/runtime/nexus/engine/runtime_context.py",
    }
)

HARNESS_01_REQUIRED_FLOWS: frozenset[str] = frozenset(row.flow for row in HARNESS_01_EXECUTION_MATRIX)

HARNESS_01_MAPPED_NODE_IDS: frozenset[str] = frozenset(
    ref.pytest_node_id for row in HARNESS_01_EXECUTION_MATRIX for ref in row.proof
)
