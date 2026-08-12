# © Artur Czarnecki. All rights reserved.

"""Single frontend-neutral application boundary for LKW conversation interactions."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAttachmentReference,
)
from local_workspace_application.conversation.interaction_event_receipt import (
    ConversationEventReceipt,
    ConversationEventReceiptError,
    ConversationEventMemoryStatus,
    ConversationEventReceiptStatus,
    ConversationInteractionEventReceiptRepository,
)
from local_workspace_application.conversation.interaction_execution_models import (
    ConversationActionExecutionStatus,
    ConversationExecutionError,
    ConversationInteractionExecutionCommand,
    ConversationInteractionExecutionResult,
    ConversationInteractionOverallStatus,
)
from local_workspace_application.conversation.interaction_executor import (
    ConversationInteractionExecutor,
)
from local_workspace_application.conversation.interaction_models import (
    ConversationInteractionPlan,
    ConversationPlanningAttachment,
    ConversationPlanningRequest,
    ConversationPlanningSourceCandidate,
    ConversationPlanningTurn,
    ConversationPlanningWorkspace,
    KnowledgeAddAttachmentsPlannedAction,
    WorkspaceAskPlannedAction,
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
)
from local_workspace_application.conversation.conversation_ingress_bootstrap import (
    ConversationIngressBootstrapService,
    pre_workspace_placeholder_id,
)
from local_workspace_application.conversation.conversation_setup_onboarding import (
    ConversationSetupOnboardingPresenter,
)
from local_workspace_application.conversation.interaction_response_renderer import (
    ConversationInteractionResponseRenderer,
)
from local_workspace_application.conversation.conversation_thread_memory_service import (
    ConversationThreadMemoryService,
    ConversationThreadMemoryServiceError,
)
from local_workspace_application.workspaces.conversation_context_execution import (
    ConversationExecutionContextError,
    build_conversation_execution_context,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationExecutionContextV1,
    ConversationIngressContextV1,
    ConversationObservedAudience,
    ConversationProductCapability,
    ResolvedConversationWorkspaceContextV1,
)
from local_workspace_application.workspaces.workspace_setup_snapshot_service import (
    WorkspaceSetupSnapshotService,
    WorkspaceSetupSnapshotV1,
)
from local_workspace_application.workspaces.conversation_citation_context_service import (
    ConversationCitationContextError,
    ConversationCitationContextService,
)
from local_workspace_application.workspaces.conversation_connection_auth_context_service import (
    ConversationConnectionAuthContextService,
)
from local_workspace_application.workspaces.knowledge_plugin_configuration_service import (
    KnowledgePluginConfigurationService,
)
from local_workspace_application.workspaces.conversation_context_resolution import (
    ConversationContextResolutionError,
    ConversationContextResolver,
)

logger = logging.getLogger(__name__)

_WORKSPACE_INTENT_RE = re.compile(
    r"\b(workspaces?|create|switch|select|list)\b",
    re.IGNORECASE,
)
_SNAPSHOT_POLL_ATTEMPTS = 3
_SNAPSHOT_POLL_DELAY_SECONDS = 0.4


class TrustedAttachmentLoader(Protocol):
    async def fetch_attachment(
        self,
        attachment: ConversationAttachmentReference,
        *,
        max_bytes: int,
    ) -> object: ...


class ConversationInteractionApplicationCommand(BaseModel):
    """Verified application inputs only; no plan, principal choice or service object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    ingress: ConversationIngressContextV1
    message_text: str = Field(default="", max_length=16_000)
    attachments: tuple[ConversationAttachmentReference, ...] = ()

    @field_validator("tenant_id")
    @classmethod
    def _normalize_tenant_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("message_text")
    @classmethod
    def _reject_nul(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("message_text must not contain NUL")
        return value

    @model_validator(mode="after")
    def _validate_content(self) -> ConversationInteractionApplicationCommand:
        if not self.message_text.strip() and not self.attachments:
            raise ValueError("message_text or attachments required")
        return self


@dataclass(frozen=True, slots=True)
class ConversationInteractionApplicationResult:
    execution_result: ConversationInteractionExecutionResult
    response_text: str
    should_send: bool
    receipt: ConversationEventReceipt | None = None


class ConversationInteractionApplicationService:
    """Owns context resolution → snapshot → plan → execute → render exactly once."""

    def __init__(
        self,
        *,
        context_resolver: ConversationContextResolver,
        planner: ConversationInteractionPlanner,
        executor: ConversationInteractionExecutor,
        renderer: ConversationInteractionResponseRenderer,
        receipt_repository: ConversationInteractionEventReceiptRepository,
        workspace_service: Any,
        personal_allowed_capabilities: frozenset[ConversationProductCapability],
        source_candidate_service: Any | None = None,
        attachment_loader: TrustedAttachmentLoader | None = None,
        attachment_max_bytes: int = 25 * 1024 * 1024,
        thread_memory_service: ConversationThreadMemoryService | None = None,
        knowledge_plugin_configuration_service: KnowledgePluginConfigurationService | None = None,
        connection_auth_context_service: ConversationConnectionAuthContextService | None = None,
        ingress_bootstrap_service: ConversationIngressBootstrapService | None = None,
        setup_snapshot_service: WorkspaceSetupSnapshotService | None = None,
        setup_onboarding_presenter: ConversationSetupOnboardingPresenter | None = None,
        citation_context_service: ConversationCitationContextService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not personal_allowed_capabilities:
            raise ValueError("personal_allowed_capabilities must not be empty")
        self._context_resolver = context_resolver
        self._planner = planner
        self._executor = executor
        self._renderer = renderer
        self._receipts = receipt_repository
        self._workspace_service = workspace_service
        self._source_candidate_service = source_candidate_service
        self._attachment_loader = attachment_loader
        self._attachment_max_bytes = attachment_max_bytes
        self._thread_memory = thread_memory_service
        self._knowledge_plugin_configuration = knowledge_plugin_configuration_service
        self._connection_auth_context = connection_auth_context_service
        self._ingress_bootstrap = ingress_bootstrap_service
        self._setup_snapshot = setup_snapshot_service
        self._setup_onboarding = setup_onboarding_presenter or (
            ConversationSetupOnboardingPresenter()
            if setup_snapshot_service is not None
            else None
        )
        self._citation_context = citation_context_service
        self._clock = clock or (lambda: datetime.now(UTC))
        self._configured_personal_capabilities = personal_allowed_capabilities
        self._attachment_registry: contextvars.ContextVar[dict[str, object]] = (
            contextvars.ContextVar("lkw_conversation_attachment_registry", default={})
        )

    async def handle(
        self,
        command: ConversationInteractionApplicationCommand,
    ) -> ConversationInteractionApplicationResult:
        execution_id = _execution_id(command)
        try:
            claim = self._receipts.claim(
                tenant_id=command.tenant_id,
                conversation_connection_ref=command.ingress.conversation_connection_ref,
                provider_event_ref=command.ingress.provider_event_ref,
                execution_id=execution_id,
            )
        except ConversationEventReceiptError:
            return self._failure(
                command=command,
                execution_id=execution_id,
                code="conversation_receipt_unavailable",
            )

        if not claim.owned:
            return self._recover_duplicate(
                command=command,
                receipt=claim.receipt,
            )

        receipt = claim.receipt
        if self._ingress_bootstrap is not None:
            try:
                self._ingress_bootstrap.ensure_personal_binding(
                    tenant_id=command.tenant_id,
                    ingress=command.ingress,
                )
            except Exception:  # noqa: BLE001 - bootstrap must not crash ingress
                logger.warning("conversation ingress bootstrap failed")

        try:
            execution_context = self._resolve_context(command)
        except ConversationContextResolutionError as exc:
            if exc.error_code == "PERSONAL_WORKSPACE_SELECTION_MISSING":
                return await self._handle_missing_workspace_selection(
                    command=command,
                    receipt=receipt,
                    execution_id=execution_id,
                )
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code=_context_error_code(exc.error_code),
            )

        try:
            recent_turns = ()
            if self._thread_memory is not None:
                recent_turns = self._thread_memory.load_recent_turns(
                    context=execution_context,
                    now=self._clock(),
                )
            planning_request = await self._build_planning_request(
                command=command,
                execution_context=execution_context,
                recent_turns=recent_turns,
            )
        except ConversationContextResolutionError as exc:
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code=_context_error_code(exc.error_code),
            )
        except ConversationExecutionContextError:
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code="conversation_context_not_active",
            )
        except ConversationThreadMemoryServiceError as exc:
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code=exc.error_code,
            )
        except Exception:  # noqa: BLE001 - safe planning-context boundary
            logger.warning("conversation planning context construction failed")
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code="conversation_planning_failed",
            )

        try:
            plan = await _maybe_await(
                self._planner.plan(planning_request, run_id=execution_id)
            )
            if not isinstance(plan, ConversationInteractionPlan):
                raise TypeError("invalid conversation plan")
        except ConversationPlanningError as exc:
            code = (
                "conversation_plan_invalid"
                if "invalid_output" in exc.code.value
                else "conversation_planning_failed"
            )
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code=code,
            )
        except Exception:  # noqa: BLE001 - planner boundary
            logger.warning("conversation planner failed")
            return self._finish_failure(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                code="conversation_planning_failed",
            )

        setup_snapshot = (
            self._derive_setup_snapshot(
                tenant_id=command.tenant_id,
                workspace_id=execution_context.workspace_id,
            )
            if any(isinstance(action, WorkspaceAskPlannedAction) for action in plan.actions)
            else None
        )
        if (
            setup_snapshot is not None
            and self._should_gate_planned_ask(
                plan=plan,
                snapshot=setup_snapshot,
            )
        ):
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=self._setup_onboarding.render_ask_blocked(setup_snapshot),
            )

        registry = {}
        token = self._attachment_registry.set(registry)
        try:
            await self._prepare_attachments(command, plan, registry)
            try:
                raw_execution_result = await _maybe_await(
                    self._executor.execute(
                        ConversationInteractionExecutionCommand(
                            tenant_id=command.tenant_id,
                            planning_request=planning_request,
                            interaction_plan=plan,
                            execution_context=execution_context,
                            execution_id=execution_id,
                        )
                    )
                )
                if not isinstance(
                    raw_execution_result,
                    ConversationInteractionExecutionResult,
                ):
                    raise TypeError("invalid conversation execution result")
                execution_result = raw_execution_result
            except Exception:  # noqa: BLE001 - executor boundary
                logger.warning("conversation executor failed")
                return self._finish_failure(
                    command=command,
                    receipt=receipt,
                    execution_id=execution_id,
                    code="conversation_execution_failed",
                )
        finally:
            self._attachment_registry.reset(token)

        self._ensure_audience_policies_from_execution(
            tenant_id=command.tenant_id,
            execution_result=execution_result,
        )
        snapshot = await self._derive_setup_snapshot_after_intake(
            tenant_id=command.tenant_id,
            workspace_id=execution_context.workspace_id,
            execution_result=execution_result,
        )

        return self._finish_success(
            command=command,
            execution_context=execution_context,
            receipt=receipt,
            execution_result=execution_result,
            setup_snapshot=snapshot,
        )

    async def _handle_missing_workspace_selection(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        receipt: ConversationEventReceipt,
        execution_id: str,
    ) -> ConversationInteractionApplicationResult:
        workspaces = await _maybe_await(
            self._workspace_service.list_workspaces(tenant_id=command.tenant_id)
        )
        workspaces = cast(Sequence[Any], workspaces)

        if command.attachments:
            welcome = self._render_welcome(workspaces)
            response = (
                f"{welcome}\n\nSelect or create a workspace before sending file attachments."
            )
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=response,
            )

        message = command.message_text.strip()
        if not message or not _looks_like_workspace_intent(message):
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=self._render_welcome(workspaces),
            )

        if self._ingress_bootstrap is None:
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=self._render_welcome(workspaces),
            )

        binding = self._ingress_bootstrap.ensure_personal_binding(
            tenant_id=command.tenant_id,
            ingress=command.ingress,
        )
        execution_context = self._build_pre_workspace_execution_context(
            binding=binding,
            ingress=command.ingress,
        )
        try:
            planning_request = await self._build_planning_request(
                command=command,
                execution_context=execution_context,
                recent_turns=(),
            )
        except Exception:  # noqa: BLE001
            logger.warning("conversation pre-workspace planning request failed")
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=self._render_welcome(workspaces),
            )

        try:
            plan = await _maybe_await(
                self._planner.plan(planning_request, run_id=execution_id)
            )
            if not isinstance(plan, ConversationInteractionPlan):
                raise TypeError("invalid conversation plan")
        except Exception:  # noqa: BLE001
            logger.warning("conversation pre-workspace planner failed")
            return self._finish_text_response(
                command=command,
                receipt=receipt,
                execution_id=execution_id,
                response_text=self._render_welcome(workspaces),
            )

        registry = {}
        token = self._attachment_registry.set(registry)
        try:
            await self._prepare_attachments(command, plan, registry)
            try:
                raw_execution_result = await _maybe_await(
                    self._executor.execute(
                        ConversationInteractionExecutionCommand(
                            tenant_id=command.tenant_id,
                            planning_request=planning_request,
                            interaction_plan=plan,
                            execution_context=execution_context,
                            execution_id=execution_id,
                        )
                    )
                )
                if not isinstance(
                    raw_execution_result,
                    ConversationInteractionExecutionResult,
                ):
                    raise TypeError("invalid conversation execution result")
                execution_result = raw_execution_result
            except Exception:  # noqa: BLE001
                logger.warning("conversation pre-workspace executor failed")
                return self._finish_text_response(
                    command=command,
                    receipt=receipt,
                    execution_id=execution_id,
                    response_text=self._render_welcome(workspaces),
                )
        finally:
            self._attachment_registry.reset(token)

        self._ensure_audience_policies_from_execution(
            tenant_id=command.tenant_id,
            execution_result=execution_result,
        )

        try:
            resolved_context = self._resolve_context(command)
        except ConversationContextResolutionError:
            resolved_context = execution_context
            snapshot = None
        else:
            snapshot = self._derive_setup_snapshot(
                tenant_id=command.tenant_id,
                workspace_id=resolved_context.workspace_id,
            )

        return self._finish_success(
            command=command,
            execution_context=resolved_context,
            receipt=receipt,
            execution_result=execution_result,
            setup_snapshot=snapshot,
        )

    def _build_pre_workspace_execution_context(
        self,
        *,
        binding: object,
        ingress: ConversationIngressContextV1,
    ) -> ConversationExecutionContextV1:
        resolved = ResolvedConversationWorkspaceContextV1(
            tenant_id=str(getattr(binding, "tenant_id")),
            conversation_context_binding_id=str(
                getattr(binding, "conversation_context_binding_id")
            ),
            audience_mode=getattr(binding, "audience_mode"),
            workspace_id=pre_workspace_placeholder_id(),
            principal_ref=ingress.actor_principal_ref,
            canonical_thread_ref=ingress.opaque_thread_ref,
            activation_policy=getattr(binding, "activation_policy"),
            thread_context_policy=getattr(binding, "thread_context_policy"),
        )
        return build_conversation_execution_context(
            resolved=resolved,
            personal_allowed_capabilities=self._personal_allowed_capabilities,
        )

    def _render_welcome(self, workspaces: Sequence[Any]) -> str:
        if self._setup_onboarding is not None:
            return self._setup_onboarding.render_welcome(workspaces)
        return (
            "Welcome to LKW. Create or select a workspace to start adding knowledge."
        )

    def _should_gate_planned_ask(
        self,
        *,
        plan: ConversationInteractionPlan,
        snapshot: WorkspaceSetupSnapshotV1,
    ) -> bool:
        if self._setup_onboarding is None:
            return False
        return any(
            isinstance(action, WorkspaceAskPlannedAction) for action in plan.actions
        ) and self._setup_onboarding.should_gate_question(snapshot)

    def _derive_setup_snapshot(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceSetupSnapshotV1 | None:
        if self._setup_snapshot is None:
            return None
        workspace = (workspace_id or "").strip()
        if not workspace or workspace == pre_workspace_placeholder_id():
            return None
        try:
            return self._setup_snapshot.derive_snapshot(
                tenant_id=tenant_id,
                workspace_id=workspace,
            )
        except Exception:  # noqa: BLE001 - snapshot is optional UX boundary
            logger.warning("conversation setup snapshot unavailable")
            return None

    async def _derive_setup_snapshot_after_intake(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        execution_result: ConversationInteractionExecutionResult,
    ) -> WorkspaceSetupSnapshotV1 | None:
        if not _execution_accepted_attachments(execution_result):
            return self._derive_setup_snapshot(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        snapshot = self._derive_setup_snapshot(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if snapshot is None or not snapshot.sync_in_progress:
            return snapshot
        for _ in range(_SNAPSHOT_POLL_ATTEMPTS - 1):
            await asyncio.sleep(_SNAPSHOT_POLL_DELAY_SECONDS)
            snapshot = self._derive_setup_snapshot(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if snapshot is None or not snapshot.sync_in_progress:
                return snapshot
        return snapshot

    def _ensure_audience_policies_from_execution(
        self,
        *,
        tenant_id: str,
        execution_result: ConversationInteractionExecutionResult,
    ) -> None:
        if self._ingress_bootstrap is None:
            return
        workspace_ids: set[str] = set()
        for item in execution_result.action_results:
            if item.artifact is None:
                continue
            data = item.artifact.data
            if not isinstance(data, dict):
                continue
            workspace_id = str(data.get("workspace_id", "")).strip()
            if workspace_id:
                workspace_ids.add(workspace_id)
        for workspace_id in sorted(workspace_ids):
            self._ingress_bootstrap.ensure_workspace_audience_policy(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )

    def _record_citation_context_from_execution(
        self,
        *,
        execution_context: ConversationExecutionContextV1,
        execution_result: ConversationInteractionExecutionResult,
    ) -> None:
        if self._citation_context is None:
            return
        workspace_id = (
            execution_result.active_workspace_id or execution_context.workspace_id
        ).strip()
        if not workspace_id:
            return
        for item in execution_result.action_results:
            if item.action_type != "workspace.ask":
                continue
            if item.status is not ConversationActionExecutionStatus.COMPLETED:
                continue
            if item.artifact is None or not isinstance(item.artifact.data, dict):
                continue
            run_id = str(item.artifact.data.get("run_id", "")).strip()
            if not run_id:
                continue
            try:
                self._citation_context.record_ask_run(
                    context=execution_context,
                    run_id=run_id,
                    workspace_id=workspace_id,
                )
            except ConversationCitationContextError:
                logger.warning("conversation citation context update failed")

    def _append_setup_guidance(
        self,
        *,
        response_text: str,
        tenant_id: str,
        workspace_id: str,
        snapshot: WorkspaceSetupSnapshotV1 | None,
    ) -> str:
        if self._setup_onboarding is None:
            return response_text
        if snapshot is None:
            snapshot = self._derive_setup_snapshot(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        if snapshot is None:
            return response_text
        if not self._setup_onboarding.should_append_snapshot_guidance(snapshot):
            return response_text
        guidance = self._setup_onboarding.render_snapshot_guidance(snapshot)
        if not guidance:
            return response_text
        if guidance in response_text:
            return response_text
        if response_text.strip():
            return f"{response_text.strip()}\n\n{guidance}"
        return guidance

    def _finish_text_response(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        receipt: ConversationEventReceipt,
        execution_id: str,
        response_text: str,
    ) -> ConversationInteractionApplicationResult:
        result = _failure_result(
            execution_id=execution_id,
            tenant_id=command.tenant_id,
            code="conversation_context_not_found",
        )
        return self._finish_success(
            receipt=receipt,
            execution_result=result,
            response_text=response_text,
        )

    def resolve_attachment(self, attachment_id: str) -> object | None:
        """Trusted, event-scoped resolver supplied to the accepted executor."""
        return self._attachment_registry.get().get(attachment_id)

    def mark_response_sent(self, result: ConversationInteractionApplicationResult) -> None:
        if result.receipt is not None and result.should_send:
            try:
                self._receipts.mark_response_sent(receipt=result.receipt)
            except ConversationEventReceiptError:
                logger.warning("conversation response receipt update failed")

    def mark_response_failed(self, result: ConversationInteractionApplicationResult) -> None:
        if result.receipt is not None and result.should_send:
            try:
                self._receipts.mark_response_failed(receipt=result.receipt)
            except ConversationEventReceiptError:
                logger.warning("conversation response failure receipt update failed")

    def _resolve_context(
        self,
        command: ConversationInteractionApplicationCommand,
    ) -> ConversationExecutionContextV1:
        if command.ingress.observed_audience is not ConversationObservedAudience.PERSONAL:
            raise ConversationContextResolutionError(
                "AUDIENCE_NOT_SUPPORTED"
            )
        resolved = self._context_resolver.resolve(
            tenant_id=command.tenant_id,
            ingress=command.ingress,
        )
        if resolved.audience_mode.value != ConversationObservedAudience.PERSONAL.value:
            raise ConversationContextResolutionError(
                "AUDIENCE_NOT_SUPPORTED"
            )
        return build_conversation_execution_context(
            resolved=resolved,
            personal_allowed_capabilities=self._personal_allowed_capabilities,
        )

    @property
    def _personal_allowed_capabilities(
        self,
    ) -> frozenset[ConversationProductCapability]:
        # Kept as a property so composition can provide a configured policy object.
        return self._configured_personal_capabilities

    async def _build_planning_request(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        execution_context: ConversationExecutionContextV1,
        recent_turns: tuple[ConversationPlanningTurn, ...] = (),
    ) -> ConversationPlanningRequest:
        raw_workspaces = self._workspace_service.list_workspaces(
            tenant_id=command.tenant_id
        )
        workspaces = cast(Sequence[Any], await _maybe_await(raw_workspaces))
        planning_workspaces = tuple(
            ConversationPlanningWorkspace(
                workspace_id=str(item.workspace_id),
                name=str(item.name),
                is_active=str(item.workspace_id) == execution_context.workspace_id,
            )
            for item in workspaces
        )

        candidates: tuple[ConversationPlanningSourceCandidate, ...] = ()
        if self._source_candidate_service is not None:
            try:
                raw_candidates = self._source_candidate_service.list_candidates(
                    tenant_id=command.tenant_id,
                    workspace_id=execution_context.workspace_id,
                )
                raw_candidates = cast(
                    Sequence[Any],
                    await _maybe_await(raw_candidates),
                )
                candidates = tuple(
                    ConversationPlanningSourceCandidate(
                        candidate_id=str(item.candidate_id),
                        label=str(item.label),
                        source_type=str(item.source_type),
                        available=bool(item.available),
                    )
                    for item in raw_candidates
                )
            except Exception:  # noqa: BLE001 - optional snapshot boundary
                logger.warning("conversation source candidate snapshot unavailable")

        knowledge_plugin_configuration = None
        if self._knowledge_plugin_configuration is not None:
            knowledge_plugin_configuration = (
                await self._knowledge_plugin_configuration.get_configuration_snapshot(
                    tenant_id=command.tenant_id,
                    execution_context=execution_context,
                )
            )

        tenant_connection_inventory = None
        if self._connection_auth_context is not None:
            tenant_connection_inventory = self._connection_auth_context.build_planning_snapshot(
                tenant_id=command.tenant_id,
                context=execution_context,
            )

        return ConversationPlanningRequest(
            message_text=command.message_text.strip(),
            attachments=tuple(
                ConversationPlanningAttachment(
                    attachment_id=item.attachment_id,
                    file_name=item.file_name,
                    content_type=item.content_type,
                    size_bytes=item.size_bytes,
                )
                for item in command.attachments
            ),
            available_workspaces=planning_workspaces,
            active_workspace_id=execution_context.workspace_id,
            available_source_candidates=candidates,
            knowledge_plugin_configuration=knowledge_plugin_configuration,
            tenant_connection_inventory=tenant_connection_inventory,
            recent_turns=recent_turns,
        )

    async def _prepare_attachments(
        self,
        command: ConversationInteractionApplicationCommand,
        plan: ConversationInteractionPlan,
        registry: dict[str, object],
    ) -> None:
        if self._attachment_loader is None:
            return
        requested_ids = {
            attachment_id
            for action in plan.actions
            if isinstance(action, KnowledgeAddAttachmentsPlannedAction)
            for attachment_id in action.attachment_ids
        }
        references = {
            item.attachment_id: item
            for item in command.attachments
            if item.attachment_id in requested_ids
        }
        for attachment_id in sorted(requested_ids):
            reference = references.get(attachment_id)
            if reference is None:
                continue
            try:
                loaded = await self._attachment_loader.fetch_attachment(
                    reference,
                    max_bytes=self._attachment_max_bytes,
                )
            except Exception:  # noqa: BLE001 - trusted intake boundary
                logger.warning("conversation attachment preparation failed")
                continue
            if loaded is not None:
                registry[attachment_id] = loaded

    def _finish_success(
        self,
        *,
        command: ConversationInteractionApplicationCommand | None = None,
        execution_context: ConversationExecutionContextV1 | None = None,
        receipt: ConversationEventReceipt,
        execution_result: ConversationInteractionExecutionResult,
        response_text: str | None = None,
        setup_snapshot: WorkspaceSetupSnapshotV1 | None = None,
    ) -> ConversationInteractionApplicationResult:
        if response_text is None:
            try:
                response_text = self._renderer.render(execution_result)
            except Exception:  # noqa: BLE001 - safe response boundary
                logger.warning("conversation response renderer failed")
                response_text = "I could not prepare a safe response. Please try again."
        if (
            command is not None
            and execution_context is not None
            and self._setup_onboarding is not None
        ):
            response_text = self._append_setup_guidance(
                response_text=response_text,
                tenant_id=command.tenant_id,
                workspace_id=execution_context.workspace_id,
                snapshot=setup_snapshot,
            )
        if command is not None and execution_context is not None:
            self._record_citation_context_from_execution(
                execution_context=execution_context,
                execution_result=execution_result,
            )
        try:
            pending = self._receipts.mark_response_pending(
                receipt=receipt,
                response=response_text,
                memory_required=(
                    self._thread_memory is not None
                    and command is not None
                    and execution_context is not None
                ),
            )
        except ConversationEventReceiptError:
            pending = receipt
            logger.warning("conversation response receipt pending update failed")
        if (
            self._thread_memory is not None
            and command is not None
            and execution_context is not None
            and pending.status is ConversationEventReceiptStatus.RESPONSE_PENDING
            and pending.memory_status is ConversationEventMemoryStatus.PENDING
        ):
            try:
                user_created_at = self._clock()
                assistant_created_at = max(self._clock(), user_created_at)
                appended = self._thread_memory.append_exchange(
                    context=execution_context,
                    user_text=(
                        execution_result.thread_memory_user_text
                        if execution_result.thread_memory_user_text is not None
                        else command.message_text.strip()
                    ),
                    assistant_text=response_text,
                    user_created_at=user_created_at,
                    assistant_created_at=assistant_created_at,
                    exchange_id=receipt.execution_id,
                )
            except ConversationThreadMemoryServiceError as exc:
                logger.warning(
                    "conversation thread memory append failed code=%s",
                    exc.error_code,
                )
                try:
                    pending = self._receipts.mark_memory_failed(
                        receipt=pending,
                        error_code=exc.error_code,
                    )
                except ConversationEventReceiptError:
                    logger.warning("conversation memory failure marker update failed")
            else:
                try:
                    pending = self._receipts.mark_memory_completed(
                        receipt=pending,
                        revision_id=appended.snapshot.revision_id,
                    )
                except ConversationEventReceiptError:
                    logger.warning("conversation memory completion marker update failed")
        return ConversationInteractionApplicationResult(
            execution_result=execution_result,
            response_text=response_text,
            should_send=True,
            receipt=pending,
        )

    def _recover_duplicate(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        receipt: ConversationEventReceipt,
    ) -> ConversationInteractionApplicationResult:
        safe_response = receipt.safe_response
        if (
            receipt.status
            in {
                ConversationEventReceiptStatus.RESPONSE_PENDING,
                ConversationEventReceiptStatus.RESPONSE_FAILED,
            }
            and safe_response
        ):
            if (
                self._thread_memory is not None
                and receipt.memory_status
                in {
                    ConversationEventMemoryStatus.PENDING,
                    ConversationEventMemoryStatus.FAILED,
                }
            ):
                try:
                    execution_context = self._resolve_context(command)
                    user_created_at = self._clock()
                    assistant_created_at = max(self._clock(), user_created_at)
                    appended = self._thread_memory.append_exchange(
                        context=execution_context,
                        user_text=command.message_text.strip(),
                        assistant_text=safe_response,
                        user_created_at=user_created_at,
                        assistant_created_at=assistant_created_at,
                        exchange_id=receipt.execution_id,
                    )
                    receipt = self._receipts.mark_memory_completed(
                        receipt=receipt,
                        revision_id=appended.snapshot.revision_id,
                    )
                except (
                    ConversationContextResolutionError,
                    ConversationExecutionContextError,
                    ConversationThreadMemoryServiceError,
                    ConversationEventReceiptError,
                ) as exc:
                    logger.warning(
                        "conversation duplicate memory recovery failed kind=%s",
                        type(exc).__name__,
                    )
            return ConversationInteractionApplicationResult(
                execution_result=_failure_result(
                    execution_id=receipt.execution_id,
                    tenant_id=command.tenant_id,
                    code="conversation_duplicate_event",
                ),
                response_text=safe_response,
                should_send=True,
                receipt=receipt,
            )
        return ConversationInteractionApplicationResult(
            execution_result=_failure_result(
                execution_id=receipt.execution_id,
                tenant_id=command.tenant_id,
                code="conversation_duplicate_event",
            ),
            response_text="",
            should_send=False,
            receipt=receipt,
        )

    def _finish_failure(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        receipt: ConversationEventReceipt,
        execution_id: str,
        code: str,
    ) -> ConversationInteractionApplicationResult:
        result = _failure_result(
            execution_id=execution_id,
            tenant_id=command.tenant_id,
            code=code,
        )
        return self._finish_success(receipt=receipt, execution_result=result)

    def _failure(
        self,
        *,
        command: ConversationInteractionApplicationCommand,
        execution_id: str,
        code: str,
    ) -> ConversationInteractionApplicationResult:
        result = _failure_result(
            execution_id=execution_id,
            tenant_id=command.tenant_id,
            code=code,
        )
        try:
            response_text = self._renderer.render(result)
        except Exception:  # noqa: BLE001
            response_text = "I could not prepare a safe response. Please try again."
        return ConversationInteractionApplicationResult(
            execution_result=result,
            response_text=response_text,
            should_send=True,
        )


def _execution_id(command: ConversationInteractionApplicationCommand) -> str:
    canonical = "\x1f".join(
        (
            command.tenant_id,
            command.ingress.conversation_connection_ref,
            command.ingress.provider_event_ref,
        )
    )
    return f"lkw-conversation:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:48]}"


def _failure_result(
    *,
    execution_id: str,
    tenant_id: str,
    code: str,
) -> ConversationInteractionExecutionResult:
    now = datetime.now(UTC)
    return ConversationInteractionExecutionResult(
        execution_id=execution_id,
        tenant_id=tenant_id,
        plan_version="application",
        started_at=now,
        completed_at=now,
        status=ConversationInteractionOverallStatus.FAILED,
        error=ConversationExecutionError(code=code),
    )


def _context_error_code(error_code: str) -> str:
    if error_code in {
        "ACTIVATION_NOT_ALLOWED",
    }:
        return "conversation_activation_not_allowed"
    if error_code in {"AUDIENCE_MISMATCH", "AUDIENCE_NOT_SUPPORTED"}:
        return "conversation_audience_not_supported"
    if error_code in {
        "CONVERSATION_CONNECTION_UNAVAILABLE",
        "WORKSPACE_UNAVAILABLE",
        "WORKSPACE_NOT_AUTHORIZED",
        "NO_ACTIVE_BINDING",
        "AMBIGUOUS_ACTIVE_BINDING",
        "PERSONAL_PRINCIPAL_MISMATCH",
        "PERSONAL_WORKSPACE_SELECTION_MISSING",
    }:
        return "conversation_context_not_found"
    return "conversation_context_not_active"


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _looks_like_workspace_intent(message: str) -> bool:
    return bool(_WORKSPACE_INTENT_RE.search(message))


def _execution_accepted_attachments(
    execution_result: ConversationInteractionExecutionResult,
) -> bool:
    for item in execution_result.action_results:
        if item.action_type != "knowledge.add_attachments":
            continue
        if item.status is ConversationActionExecutionStatus.COMPLETED:
            return True
    return False


__all__ = [
    "ConversationInteractionApplicationCommand",
    "ConversationInteractionApplicationResult",
    "ConversationInteractionApplicationService",
]
