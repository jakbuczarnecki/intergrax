# © Artur Czarnecki. All rights reserved.

"""Proof stage runners: generation, planning, tool call, grounded Ask."""

from __future__ import annotations

import json
from typing import Any, Literal

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.registry.catalog_capabilities import (
    unwrap_catalog_capability_adapter,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import new_run_id

from local_workspace_application.conversation.interaction_models import (
    ConversationPlanningRequest,
    ConversationPlanningWorkspace,
    KnowledgeAddSourcesPlannedAction,
    WorkspaceReferenceKind,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    validate_plan_against_request,
)
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor
from local_workspace_application.model_runtime_proof.contracts import (
    ASK_QUESTION,
    BASIC_GENERATION_MARKER,
    FIXTURE_MARKER,
    MANAGED_WORKSPACE_USER_ID,
    PLANNING_MESSAGE,
    PLANNING_URL,
    WORKSPACE_SEARCH_TOOL,
    ProofFailureCode,
)
from local_workspace_application.model_runtime_proof.safety import (
    assert_no_secret_leak,
    normalize_provider_error,
)
from local_workspace_application.workspaces.ask_service import WorkspaceAskService
from local_workspace_application.workspaces.models import WorkspaceOperationStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.search_evidence import map_search_hits

WORKSPACE_SEARCH_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": WORKSPACE_SEARCH_TOOL,
            "description": (
                "Search indexed knowledge in one workspace and return grounded evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["workspace_id", "query"],
                "additionalProperties": False,
            },
        },
    }
]


def _workspace_ref_matches(
    action: KnowledgeAddSourcesPlannedAction, *, kind: str, value: str
) -> bool:
    ref = action.workspace
    return ref.kind.value == kind and ref.value == value


async def run_basic_generation(
    adapter: LLMAdapter,
    *,
    provider: str,
    configured_model: str,
) -> tuple[bool, str | None, ProofFailureCode | None, str | None]:
    prompt = f"Reply with one short sentence that includes the exact marker {BASIC_GENERATION_MARKER}."
    try:
        response = adapter.generate_messages([ChatMessage(role="user", content=prompt)])
    except Exception as exc:
        err_type, err_excerpt = normalize_provider_error(exc)
        return False, None, ProofFailureCode.BASIC_GENERATION_FAILED, err_excerpt

    content = str(getattr(response, "content", "") or "")
    metadata = getattr(response, "metadata", {}) or {}
    if metadata.get("provider") and str(metadata.get("provider")) != provider:
        return (
            False,
            content[:120],
            ProofFailureCode.PROVIDER_IDENTITY_MISMATCH,
            "metadata_provider",
        )
    if BASIC_GENERATION_MARKER not in content:
        return (
            False,
            content[:120],
            ProofFailureCode.BASIC_GENERATION_FAILED,
            "marker_missing",
        )
    if not content.strip():
        return False, None, ProofFailureCode.BASIC_GENERATION_FAILED, "empty_response"
    leak = assert_no_secret_leak(content)
    if leak:
        return False, content[:120], ProofFailureCode.PROOF_SECRET_LEAK_DETECTED, leak
    _ = configured_model
    return True, content[:120], None, None


async def run_structured_planning(
    adapter: LLMAdapter,
) -> tuple[bool, str | None, ProofFailureCode | None, str | None]:
    if not adapter.supports_structured_output():
        return (
            False,
            None,
            ProofFailureCode.STRUCTURED_PLANNING_UNSUPPORTED,
            "no_structured_output",
        )

    request = ConversationPlanningRequest(
        message_text=PLANNING_MESSAGE,
        available_workspaces=(
            ConversationPlanningWorkspace(
                workspace_id="ws-1", name="finanse", is_active=True
            ),
            ConversationPlanningWorkspace(
                workspace_id="ws-2", name="magazyn", is_active=False
            ),
        ),
        active_workspace_id="ws-1",
    )

    planner = ConversationInteractionPlanner(adapter)
    try:
        plan = await planner.plan(request, run_id="lkw-model-runtime-proof-plan")
    except Exception as exc:
        return (
            False,
            None,
            ProofFailureCode.STRUCTURED_PLANNING_FAILED,
            normalize_provider_error(exc)[1],
        )

    add_sources = [
        a for a in plan.actions if isinstance(a, KnowledgeAddSourcesPlannedAction)
    ]
    if len(plan.actions) != 1 or len(add_sources) != 1:
        return False, None, ProofFailureCode.STRUCTURED_PLANNING_FAILED, "action_count"
    if plan.clarifications:
        return (
            False,
            None,
            ProofFailureCode.STRUCTURED_PLANNING_FAILED,
            "clarification_present",
        )
    web_urls = [obj for obj in plan.objects if obj.object_type == "web_url"]
    if len(web_urls) != 1 or web_urls[0].value != PLANNING_URL:
        return False, None, ProofFailureCode.STRUCTURED_PLANNING_FAILED, "web_url"
    evidence = web_urls[0].evidence
    if request.message_text[evidence.start : evidence.end] != evidence.text:
        return False, None, ProofFailureCode.STRUCTURED_PLANNING_FAILED, "evidence_span"
    if not _workspace_ref_matches(
        add_sources[0], kind=WorkspaceReferenceKind["name"].value, value="magazyn"
    ):
        return False, None, ProofFailureCode.STRUCTURED_PLANNING_FAILED, "workspace_ref"
    if any(action.action_type == "workspace.activate" for action in plan.actions):
        return (
            False,
            None,
            ProofFailureCode.STRUCTURED_PLANNING_FAILED,
            "workspace_activate",
        )
    try:
        validate_plan_against_request(plan, request)
    except Exception as exc:
        return (
            False,
            None,
            ProofFailureCode.STRUCTURED_PLANNING_FAILED,
            normalize_provider_error(exc)[1],
        )
    return True, "validate_plan_against_request:pass", None, None


def validate_tool_call(
    tool_calls: list[Any],
    *,
    workspace_id: str,
) -> tuple[dict[str, Any] | None, ProofFailureCode | None, str | None]:
    if not tool_calls:
        return None, ProofFailureCode.TOOL_CALL_MISSING, "missing"
    if len(tool_calls) != 1:
        return None, ProofFailureCode.TOOL_CALL_MULTIPLE, f"count={len(tool_calls)}"
    call = tool_calls[0]
    name = str(getattr(call, "name", "") or getattr(call, "tool_name", ""))
    if name != WORKSPACE_SEARCH_TOOL:
        return None, ProofFailureCode.TOOL_CALL_UNEXPECTED_TOOL, name
    raw_args = (
        getattr(call, "arguments", None)
        or getattr(call, "args", None)
        or getattr(call, "arguments_json", None)
    )
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None, ProofFailureCode.TOOL_CALL_INVALID, "json"
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return None, ProofFailureCode.TOOL_CALL_INVALID, "args_type"
    if set(args.keys()) != {"workspace_id", "query"}:
        return None, ProofFailureCode.TOOL_CALL_INVALID, "extra_fields"
    if str(args.get("workspace_id", "")) != workspace_id:
        return None, ProofFailureCode.TOOL_CALL_INVALID, "workspace_id"
    query = str(args.get("query", "")).strip()
    if not query:
        return None, ProofFailureCode.TOOL_CALL_INVALID, "empty_query"
    return args, None, None


def _resolve_tool_choice(
    adapter: LLMAdapter,
    *,
    force_tool_choice: bool,
) -> tuple[str | dict[str, Any] | None, Literal["forced", "automatic"]]:
    if not force_tool_choice:
        return None, "automatic"
    inner = unwrap_catalog_capability_adapter(adapter)
    if (
        isinstance(inner, LangChainOllamaAdapter)
        or getattr(adapter, "provider", None) is LLMProvider.OLLAMA
    ):
        return "required", "forced"
    return {"type": "function", "function": {"name": WORKSPACE_SEARCH_TOOL}}, "forced"


async def run_tool_call_and_execution(
    adapter: LLMAdapter,
    *,
    tenant_id: str,
    workspace_id: str,
    task_executor: LocalWorkspaceTaskExecutor,
    repository: ManagedWorkspaceRepository,
    force_tool_choice: bool = True,
) -> tuple[
    bool, Literal["forced", "automatic"] | None, ProofFailureCode | None, str | None
]:
    if not adapter.supports_tools():
        return (
            False,
            None,
            ProofFailureCode.TOOL_CALL_UNSUPPORTED,
            "supports_tools_false",
        )

    prompt = (
        f"Call {WORKSPACE_SEARCH_TOOL} exactly once to find the portability verification code "
        f"{FIXTURE_MARKER} in workspace {workspace_id}."
    )
    tool_choice, mode = _resolve_tool_choice(
        adapter, force_tool_choice=force_tool_choice
    )
    try:
        response = adapter.generate_with_tools(
            [ChatMessage(role="user", content=prompt)],
            WORKSPACE_SEARCH_TOOL_SCHEMA,
            tool_choice=tool_choice,
        )
    except Exception as exc:
        return (
            False,
            mode,
            ProofFailureCode.TOOL_CALL_INVALID,
            normalize_provider_error(exc)[1],
        )

    tool_calls = list(getattr(response, "tool_calls", None) or [])
    args, failure, detail = validate_tool_call(tool_calls, workspace_id=workspace_id)
    if failure is not None or args is None:
        return False, mode, failure or ProofFailureCode.TOOL_CALL_INVALID, detail or "args_missing"

    run_id = new_run_id()
    task = Task(
        task_id=run_id,
        tenant_id=tenant_id,
        user_id=MANAGED_WORKSPACE_USER_ID,
        message=str(args["query"]),
        context=TaskContext(capability=WORKSPACE_SEARCH_TOOL),
        metadata={
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "collection_id": workspace_id,
            "query": str(args["query"]),
            "top_k": 5,
            "requested_by": "lkw.model_runtime_proof",
        },
    )
    try:
        result = await task_executor.execute(task)
    except Exception as exc:
        return (
            False,
            mode,
            ProofFailureCode.TOOL_EXECUTION_FAILED,
            normalize_provider_error(exc)[1],
        )

    hits = map_search_hits(
        repository=repository,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        task_result=result,
        limit=5,
    )
    if not hits:
        return False, mode, ProofFailureCode.TOOL_EXECUTION_FAILED, "no_hits"
    if not any(FIXTURE_MARKER in (hit.snippet or "") for hit in hits):
        return False, mode, ProofFailureCode.TOOL_EXECUTION_FAILED, "marker_missing"
    return True, mode, None, None


async def run_grounded_ask(
    ask_service: WorkspaceAskService,
    *,
    tenant_id: str,
    workspace_id: str,
    source_id: str,
    repository: ManagedWorkspaceRepository,
) -> tuple[bool, str | None, str | None, bool, ProofFailureCode | None, str | None]:
    try:
        run = await ask_service.ask(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            question=ASK_QUESTION,
        )
    except Exception as exc:
        return (
            False,
            None,
            None,
            False,
            ProofFailureCode.GROUNDED_ASK_FAILED,
            normalize_provider_error(exc)[1],
        )

    answer = str(run.answer or "")
    if FIXTURE_MARKER not in answer:
        return (
            False,
            answer[:160],
            None,
            False,
            ProofFailureCode.GROUNDED_ASK_FAILED,
            "answer_marker",
        )

    if not run.citations:
        return (
            False,
            answer[:160],
            None,
            False,
            ProofFailureCode.CITATION_MISSING,
            "no_citations",
        )

    citation = run.citations[0]
    excerpt = str(getattr(citation, "excerpt", "") or "")
    citation_source = str(getattr(citation, "source_id", "") or "")
    if citation_source != source_id:
        return (
            False,
            answer[:160],
            excerpt[:160],
            False,
            ProofFailureCode.CITATION_MISSING,
            "source_id",
        )
    if FIXTURE_MARKER not in excerpt:
        return (
            False,
            answer[:160],
            excerpt[:160],
            False,
            ProofFailureCode.CITATION_MARKER_MISSING,
            "excerpt",
        )

    payload = json.dumps(
        {
            "answer": answer[:160],
            "citation": excerpt[:160],
            "source_id": citation_source,
        }
    )
    leak = assert_no_secret_leak(payload)
    if leak:
        return (
            False,
            answer[:160],
            excerpt[:160],
            False,
            ProofFailureCode.PROOF_SECRET_LEAK_DETECTED,
            leak,
        )

    from local_workspace_application.workspaces.ask_repository import (
        WorkspaceAskRepository,
    )

    ask_repo = WorkspaceAskRepository(repository.document_store)
    ask_persisted = ask_repo.get_run(tenant_id=tenant_id, run_id=run.run_id) is not None
    return True, answer[:160], excerpt[:160], ask_persisted, None, None


def count_indexing_operations(
    repository: ManagedWorkspaceRepository,
    *,
    tenant_id: str,
    workspace_id: str,
) -> int:
    operations = repository.list_ingestion_operations(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        statuses={WorkspaceOperationStatus.COMPLETED},
    )
    return len(operations)
