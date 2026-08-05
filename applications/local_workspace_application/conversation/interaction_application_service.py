# © Artur Czarnecki. All rights reserved.

"""Single frontend-neutral application boundary for LKW conversation interactions."""

from __future__ import annotations

import contextvars
import hashlib
import inspect
import logging
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
)
from local_workspace_application.conversation.interaction_planner import (
    ConversationInteractionPlanner,
    ConversationPlanningError,
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
)
from local_workspace_application.workspaces.knowledge_plugin_configuration_service import (
    KnowledgePluginConfigurationService,
)
from local_workspace_application.workspaces.conversation_context_resolution import (
    ConversationContextResolutionError,
    ConversationContextResolver,
)

logger = logging.getLogger(__name__)


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
        try:
            execution_context = self._resolve_context(command)
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

        return self._finish_success(
            command=command,
            execution_context=execution_context,
            receipt=receipt,
            execution_result=execution_result,
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
    ) -> ConversationInteractionApplicationResult:
        try:
            response_text = self._renderer.render(execution_result)
        except Exception:  # noqa: BLE001 - safe response boundary
            logger.warning("conversation response renderer failed")
            response_text = "I could not prepare a safe response. Please try again."
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
                    user_text=command.message_text.strip(),
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


__all__ = [
    "ConversationInteractionApplicationCommand",
    "ConversationInteractionApplicationResult",
    "ConversationInteractionApplicationService",
]
