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

Harness01ProofKind = Literal[
    "STATIC_BOUNDARY",
    "CANONICAL_CONSUMPTION",
    "FAIL_CLOSED",
    "BUDGET_PRE_EFFECT",
    "IDENTITY_AUTHORITY",
    "TRACE_CORRELATION",
    "GATE",
    "SELECTION",
    "EVIDENCE",
]

HARNESS_01_CANONICAL_MINIMUM_PROOF_KINDS: frozenset[str] = frozenset(
    {
        "STATIC_BOUNDARY",
        "CANONICAL_CONSUMPTION",
        "FAIL_CLOSED",
        "BUDGET_PRE_EFFECT",
        "IDENTITY_AUTHORITY",
        "TRACE_CORRELATION",
        "GATE",
        "EVIDENCE",
    }
)


@dataclass(frozen=True, slots=True)
class Harness01EvidenceRef:
    pytest_node_id: str
    kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Harness01ExecutionRow:
    flow_id: str
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


def _gr10_r8(test_name: str) -> str:
    return _nid(
        "tests/qualification/governance/strategy/test_gr10_r8_orchestration_inner_governance_qualification.py",
        test_name,
    )


def _ref(node_id: str, *kinds: str) -> Harness01EvidenceRef:
    return Harness01EvidenceRef(node_id, kinds)


# Independent architecture inventory — MUST NOT be derived from HARNESS_01_EXECUTION_MATRIX.
HARNESS_01_REQUIRED_FLOW_IDS: frozenset[str] = frozenset(
    {
        "execution.single_agent_nexus_loop",
        "execution.bounded_tool_loop",
        "execution.public_invocation_pattern",
        "execution.uaep_tool_capability",
        "execution.application_owned_tools",
        "execution.integration_backed_tools",
        "execution.background_host_task",
        "execution.session_memory",
        "execution.rag_retrieval",
        "execution.policy_controlled_tools",
        "execution.security_gateway_hooks",
        "execution.runtime_plugin_hooks",
        "execution.multi_agent_delegation",
        "execution.llm_invocation",
        "execution.context_assembly",
        "execution.declarative_compensation",
    }
)


HARNESS_01_EXECUTION_MATRIX: tuple[Harness01ExecutionRow, ...] = (
    Harness01ExecutionRow(
        "execution.single_agent_nexus_loop",
        "Standard single-agent run (NexusLoop)",
        "HostTaskExecutionPort / NexusWorkerRuntime → NexusLoop.run",
        "RuntimeState.run_id · ExecutionRequest.agent_id",
        "DefaultRootExecutionLauncher · agent_runtime_governance",
        "Declarative enforcer · scope policy · sandbox gate",
        "NexusLoop → tool_loop → RuntimeToolInvoker",
        "ToolCallTrace · state.tool_traces · run trace store",
        "ToolExecutor via registry",
        "CANONICAL",
        (_ref(_host01("test_host_q1_production_surfaces_use_host_task_execution_port"), "GATE"),),
    ),
    Harness01ExecutionRow(
        "execution.bounded_tool_loop",
        "Agentic / bounded tool-loop execution",
        "resolve_invocation_pattern → NexusToolInvocationPattern.execute",
        "RuntimeConfig.tool_invocation_pattern_id",
        "Inner execution guard · tool call budget",
        "ToolScopePolicy · meaningful side-effect authorization",
        "tool_loop.invoke_prepared_tool_execution_request",
        "PlannedToolCallOutcome.trace",
        "RegistryToolExecutor",
        "CANONICAL",
        (
            _ref(_plug03("test_tools_profile_selection_executes_custom_not_catalog_default"), "CANONICAL_CONSUMPTION"),
            _ref(_h01("test_harness_01_tool_budget_record_precedes_invoker_invoke"), "BUDGET_PRE_EFFECT"),
        ),
    ),
    Harness01ExecutionRow(
        "execution.public_invocation_pattern",
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
            _ref(_h01("test_harness_01_public_pattern_bridge_ignores_port_agent_id_statically"), "IDENTITY_AUTHORITY"),
        ),
    ),
    Harness01ExecutionRow(
        "execution.uaep_tool_capability",
        "UAEP / agent step tool capability",
        "exec_ctx.invoke_tool → BoundToolGateway",
        "ToolRequest.agent_id + RuntimeState",
        "MiddlewarePipeline tool hooks",
        "ToolAccessPolicy.is_tool_allowed",
        "RuntimeToolGateway → catalog_dispatch → RuntimeToolInvoker",
        "ToolResponse + gateway trace step",
        "Catalog / capability tools",
        "CANONICAL",
        (_ref(_u3("test_u3_uaep_tool_path_uses_runtime_tool_gateway_not_local_invoker"), "STATIC_BOUNDARY"),),
    ),
    Harness01ExecutionRow(
        "execution.application_owned_tools",
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
        "execution.integration_backed_tools",
        "Integration-backed tools",
        "IntegrationProfile → provider binding",
        "Integration registry / resolver",
        "Governance on tool path",
        "Integration + tool contracts",
        "Tool registry handler (vendor inside provider)",
        "Tool invocation trace",
        "External APIs via integration adapter",
        "CANONICAL",
        (_ref(_plug03("test_integration_explicit_slug_activates_fixture_provider"), "CANONICAL_CONSUMPTION"),),
    ),
    Harness01ExecutionRow(
        "execution.background_host_task",
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
        "execution.session_memory",
        "Memory / session operations",
        "SessionManager / memory control plane contracts",
        "tenant_id · session_id from runtime context",
        "Memory profile + plugin admission",
        "Tenant isolation invariants",
        "Memory provider implementations (vendor-internal)",
        "Session/memory provenance evidence",
        "Storage backends",
        "CANONICAL",
        (_ref(_plug03("test_plug03_session_storage_canonical_session_manager_consumer"), "CANONICAL_CONSUMPTION"),),
    ),
    Harness01ExecutionRow(
        "execution.rag_retrieval",
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
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    Harness01ExecutionRow(
        "execution.policy_controlled_tools",
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
        "execution.security_gateway_hooks",
        "Security middleware / tool hooks",
        "RuntimeToolGateway.invoke → run_tool_call_hooks",
        "MiddlewarePipeline on gateway",
        "Hook chain before/after invoke",
        "Denied tools fail at gateway",
        "Gateway inner invoke",
        "Hook context + ToolResponse status",
        "Middleware-controlled path",
        "CANONICAL",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/nexus/tools/test_tool_gateway.py",
                    "test_tool_gateway_denies_unknown_tool_when_not_allowed",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
    Harness01ExecutionRow(
        "execution.runtime_plugin_hooks",
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
        "execution.multi_agent_delegation",
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
        "execution.llm_invocation",
        "LLM model invocation",
        "LLM adapter / provider registry",
        "Model profile + runtime config",
        "Token/context budget (CE-02 path)",
        "Policy on model usage where configured",
        "intergrax.llm adapters — no vendor SDK in runtime execution",
        "LLM trace / modality metrics",
        "Provider HTTP inside adapter",
        "CANONICAL",
        (_ref(_h01("test_harness_01_runtime_tier_no_direct_vendor_llm_imports"), "STATIC_BOUNDARY"),),
    ),
    Harness01ExecutionRow(
        "execution.context_assembly",
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
    Harness01ExecutionRow(
        "execution.declarative_compensation",
        "Declarative compensation side effects",
        "BoundCompensationToolInvokeSession → ExecutionBoundDeclarativeToolInvoker",
        "Bound execution identity on declarative port",
        "ACP compensation admission",
        "Separate contract plane from RuntimeToolInvoker",
        "ExecutionBoundDeclarativeToolInvoker (not RuntimeToolInvoker bypass)",
        "CompensationSideEffectInvokeResult",
        "Declarative catalog tools via bound port",
        "AUTHORIZED_INTERNAL",
        (_ref(_h01("test_harness_01_declarative_compensation_is_separate_contract_plane"), "STATIC_BOUNDARY"),),
        "ExecutionBoundDeclarativeToolInvoker is owned by agents/persistence + contracts; "
        "must not be classified as RuntimeToolInvoker.invoke bypass.",
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
        "intergrax/runtime/nexus/tools/catalog_context.py",
        "intergrax/runtime/nexus/tools/catalog_dispatch.py",
        "intergrax/runtime/nexus/tools/tool_loop.py",
        "intergrax/runtime/nexus/tools/patterns/deterministic_chain.py",
    }
)

HARNESS_01_RUNTIME_TOOL_INVOKER_REFERENCE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "intergrax/runtime/attestation/boundary_emitter.py",
        "intergrax/runtime/agent_governance/ports.py",
        "intergrax/agents/persistence/catalog_declarative_invoker.py",
        "intergrax/runtime/nexus/config.py",
        "intergrax/runtime/nexus/config_sections.py",
        "intergrax/runtime/nexus/context/iterative_tool_context_assembly.py",
        "intergrax/runtime/nexus/tools/invoker.py",
        "intergrax/runtime/nexus/tools/patterns/bounded_react.py",
        "intergrax/runtime/nexus/tools/patterns/deterministic_chain.py",
        "intergrax/runtime/nexus/tools/patterns/parallel_batch.py",
        "intergrax/runtime/nexus/tools/patterns/parallel_semantic_batch.py",
        "intergrax/runtime/nexus/tools/patterns/single_pass.py",
        "intergrax/runtime/nexus/tools/planner_bootstrap.py",
        "intergrax/runtime/nexus/tools/public_tool_invocation_pattern_bridge.py",
        "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py",
        "intergrax/runtime/nexus/tools/tool_invocation_pattern.py",
        "intergrax/runtime/nexus/tools/tool_loop.py",
        "intergrax/runtime/nexus/tools/tool_planner_protocol.py",
        "intergrax/runtime/nexus/tools/tool_verify_hooks.py",
        "intergrax/runtime/nexus/tools/uaep_tool_gateway.py",
        "intergrax/runtime/sandbox/isolation_gate.py",
        "intergrax/runtime/tools/scope_policy.py",
    }
)

HARNESS_01_RUNTIME_TOOL_INVOKER_COMPOSITION_ROOTS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py",
    }
)

HARNESS_01_FORBIDDEN_DIRECT_VENDOR_SDK_PREFIXES: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "google.generativeai",
        "google.genai",
        "boto3",
    }
)

HARNESS_01_VENDOR_SDK_ALLOWED_REL_PREFIXES: frozenset[str] = frozenset(
    {
        "intergrax/llm_adapters/",
        "intergrax/rag/",
        "intergrax/integrations/",
        "intergrax/speech_adapters/",
        "intergrax/model_inference/",
        "intergrax/websearch/",
    }
)

HARNESS_01_MAPPED_NODE_IDS: frozenset[str] = frozenset(
    ref.pytest_node_id for row in HARNESS_01_EXECUTION_MATRIX for ref in row.proof
)

HARNESS_01_INDEPENDENT_ZERO_BYPASS_GATE_TEST_NAMES: frozenset[str] = frozenset(
    {
        "test_harness_01_matrix_covers_required_flow_id_inventory",
        "test_harness_01_matrix_flow_ids_are_unique",
        "test_harness_01_runtime_tool_invoker_callsites_are_authorized_internal",
        "test_harness_01_runtime_tool_invoker_reference_files_are_classified",
        "test_harness_01_runtime_tool_invoker_constructed_only_at_composition_roots",
        "test_harness_01_application_host_trees_do_not_construct_runtime_tool_invoker",
        "test_harness_01_cross_layer_private_nexus_member_access",
        "test_harness_01_runtime_agents_and_hosts_no_direct_forbidden_vendor_sdk",
        "test_harness_01_tool_budget_record_precedes_invoker_invoke",
        "test_harness_01_declarative_compensation_is_separate_contract_plane",
    }
)
