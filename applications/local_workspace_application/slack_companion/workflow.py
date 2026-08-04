# © Artur Czarnecki. All rights reserved.

"""Inbound MESSAGE → configured/selected workspace → Ask HTTP → threaded answer."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAddress,
    ConversationAttachmentContent,
    ConversationAttachmentFetchError,
    ConversationAttachmentFetcher,
    ConversationChannelBackend,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.slack_companion.ask_client import WorkspaceAskHttpClient
from local_workspace_application.slack_companion.authorization import (
    SlackCompanionAuthConfig,
    authorize_inbound_message,
)
from local_workspace_application.slack_companion.commands import (
    SlackCommandContext,
    SlackCommandMatch,
    discover_slack_commands,
    render_command_help,
    slack_command,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.models import (
    AuthorizedSlackMessageContext,
    SlackAskClientError,
    SlackDedupeRecord,
    SlackSourceCandidateListItem,
    SlackSourceListItem,
    SlackWorkspaceListItem,
)
from local_workspace_application.slack_companion.pending_deletion_store import (
    InMemorySlackPendingDeletionStore,
)
from local_workspace_application.slack_companion.rendering import (
    MAX_SOURCE_CANDIDATE_ITEMS,
    MAX_WORKSPACE_NAME_CHARS,
    render_acknowledgement,
    render_ask_response,
    render_attachment_batch_response,
    render_attachment_fetch_failed,
    render_attachment_fetch_unavailable,
    render_attachment_intake_failed,
    render_attachment_receiving,
    render_attachment_too_many,
    render_error,
    render_no_workspace_available,
    render_selected_workspace_unavailable,
    render_source_candidate_accept_failed,
    render_source_candidate_accepted,
    render_source_candidate_already_attached,
    render_source_candidate_list,
    render_source_candidate_list_empty,
    render_source_candidate_list_load_failed,
    render_source_candidate_out_of_range,
    render_source_candidate_selection_conflict,
    render_source_candidate_service_unavailable,
    render_source_candidate_unavailable,
    render_source_candidate_usage,
    render_source_list,
    render_source_list_load_failed,
    render_source_workspace_unavailable,
    render_workspace_create_usage,
    render_workspace_created,
    render_workspace_delete_cancelled,
    render_workspace_delete_confirmation,
    render_workspace_delete_missing_pending,
    render_workspace_delete_usage,
    render_workspace_deleted,
    render_workspace_list,
    render_workspace_list_load_failed,
    render_workspace_out_of_range,
    render_workspace_selected,
    render_workspace_selection_usage,
)
from local_workspace_application.slack_companion.selection_store import (
    InMemorySlackWorkspaceSelectionStore,
    SlackWorkspaceSelection,
    slack_selection_actor_key,
)
from local_workspace_application.conversation.interaction_application_service import (
    ConversationInteractionApplicationCommand,
    ConversationInteractionApplicationService,
)
from local_workspace_application.workspaces.conversation_context_models import (
    ConversationActivationSignal,
    ConversationIngressContextV1,
    ConversationObservedAudience,
)

logger = logging.getLogger(__name__)

OutboundSender = Callable[[OutboundConversationMessage], Awaitable[object]]

_WORKSPACES_COMMAND = "workspaces"
_SOURCES_COMMAND = "sources"
_SOURCE_CANDIDATES_COMMAND = "source candidates"
_SAFE_SOURCE_LABEL_FALLBACK = "Source"
_WORKSPACE_SELECTION_RE = re.compile(
    r"^workspace\s+([1-9]\d*)$",
    re.IGNORECASE,
)
_SOURCE_CANDIDATE_ADD_RE = re.compile(
    r"^source\s+add\s+([1-9]\d*)$",
    re.IGNORECASE,
)
_WORKSPACE_CREATE_RE = re.compile(
    r"^workspace\s+create(?:\s+(.*))?$",
    re.IGNORECASE | re.DOTALL,
)
_WORKSPACE_DELETE_RE = re.compile(
    r"^workspace\s+delete\s+([1-9]\d*)$",
    re.IGNORECASE,
)
_WORKSPACE_DELETE_CONFIRM = "workspace delete confirm"
_WORKSPACE_DELETE_CANCEL = "workspace delete cancel"


def is_workspaces_command(text: str) -> bool:
    """Exact ``workspaces`` after trim; case-insensitive. No extra words."""
    return (text or "").strip().casefold() == _WORKSPACES_COMMAND


def is_sources_command(text: str) -> bool:
    """Exact ``sources`` after trim; case-insensitive. No extra words."""
    return (text or "").strip().casefold() == _SOURCES_COMMAND


def parse_workspace_selection(text: str) -> int | None:
    """Return 1-based index for ``workspace <positive integer>``, else ``None``."""
    stripped = (text or "").strip()
    match = _WORKSPACE_SELECTION_RE.fullmatch(stripped)
    if match is None:
        return None
    return int(match.group(1))


def is_workspace_selection_attempt(text: str) -> bool:
    """True when the first token is exactly ``workspace`` (not ``workspaces``)."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    first = stripped.split(None, 1)[0]
    return first.casefold() == "workspace"


def is_workspace_create_attempt(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    parts = stripped.split(None, 2)
    return (
        len(parts) >= 2
        and parts[0].casefold() == "workspace"
        and parts[1].casefold() == "create"
    )


def normalize_workspace_display_name(raw: str) -> str | None:
    """Collapse external whitespace; reject empty/control/overlong names."""
    if raw is None:
        return None
    if any(ord(ch) < 32 for ch in raw):
        return None
    collapsed = " ".join(raw.split())
    if not collapsed:
        return None
    if len(collapsed) > MAX_WORKSPACE_NAME_CHARS:
        return None
    return collapsed


def parse_workspace_create(text: str) -> str | None:
    """
    Return normalized name for ``workspace create <name>``.

    Returns ``None`` when the message is not a create command or is invalid.
    """
    stripped = (text or "").strip()
    if any(ord(ch) < 32 for ch in stripped):
        return None
    match = _WORKSPACE_CREATE_RE.fullmatch(stripped)
    if match is None:
        return None
    return normalize_workspace_display_name(match.group(1) or "")


def parse_workspace_delete(text: str) -> int | None:
    stripped = (text or "").strip()
    match = _WORKSPACE_DELETE_RE.fullmatch(stripped)
    if match is None:
        return None
    return int(match.group(1))


def is_workspace_delete_confirm(text: str) -> bool:
    return (text or "").strip().casefold() == _WORKSPACE_DELETE_CONFIRM


def is_workspace_delete_cancel(text: str) -> bool:
    return (text or "").strip().casefold() == _WORKSPACE_DELETE_CANCEL


def is_workspace_delete_attempt(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    parts = stripped.split(None, 2)
    return (
        len(parts) >= 2
        and parts[0].casefold() == "workspace"
        and parts[1].casefold() == "delete"
    )


def slack_attachment_intake_idempotency_key(
    *,
    team_id: str,
    event_id: str,
) -> str:
    """Deterministic intake idempotency key for one Slack attachment event."""
    canonical = json.dumps(
        {
            "version": 1,
            "team_id": team_id.strip(),
            "event_id": event_id.strip(),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"slack-attachment:v1:{digest}"


def slack_source_candidate_intake_idempotency_key(
    *,
    team_id: str,
    event_id: str,
) -> str:
    """Deterministic intake idempotency key for one Slack Source Candidate event."""
    canonical = json.dumps(
        {
            "event_id": event_id.strip(),
            "team_id": team_id.strip(),
            "version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"slack-source-candidate:v1:{digest}"


def parse_help_command(text: str) -> SlackCommandMatch | None:
    if (text or "").strip().casefold() == "help":
        return SlackCommandMatch()
    return None


def parse_workspaces_list_command(text: str) -> SlackCommandMatch | None:
    if is_workspaces_command(text):
        return SlackCommandMatch()
    return None


def parse_sources_list_command(text: str) -> SlackCommandMatch | None:
    if is_sources_command(text):
        return SlackCommandMatch()
    return None


def parse_source_candidates_list_command(text: str) -> SlackCommandMatch | None:
    if (text or "").strip().casefold() == _SOURCE_CANDIDATES_COMMAND:
        return SlackCommandMatch()
    return None


def parse_source_candidate_accept_command(text: str) -> SlackCommandMatch | None:
    stripped = (text or "").strip()
    match = _SOURCE_CANDIDATE_ADD_RE.fullmatch(stripped)
    if match is None:
        return None
    return SlackCommandMatch(payload=int(match.group(1)))


def is_source_candidate_accept_attempt(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    parts = stripped.split(None, 2)
    return (
        len(parts) >= 2
        and parts[0].casefold() == "source"
        and parts[1].casefold() == "add"
    )


def parse_source_candidate_accept_invalid_command(
    text: str,
) -> SlackCommandMatch | None:
    if is_source_candidate_accept_attempt(text) and parse_source_candidate_accept_command(
        text
    ) is None:
        return SlackCommandMatch()
    return None


def parse_workspace_delete_confirm_command(text: str) -> SlackCommandMatch | None:
    if is_workspace_delete_confirm(text):
        return SlackCommandMatch()
    return None


def parse_workspace_delete_cancel_command(text: str) -> SlackCommandMatch | None:
    if is_workspace_delete_cancel(text):
        return SlackCommandMatch()
    return None


def parse_workspace_delete_request_command(text: str) -> SlackCommandMatch | None:
    index = parse_workspace_delete(text)
    if index is None:
        return None
    return SlackCommandMatch(payload=index)


def parse_workspace_delete_invalid_command(text: str) -> SlackCommandMatch | None:
    if is_workspace_delete_attempt(text) and parse_workspace_delete(text) is None:
        return SlackCommandMatch()
    return None


def parse_workspace_create_command(text: str) -> SlackCommandMatch | None:
    name = parse_workspace_create(text)
    if name is None:
        return None
    return SlackCommandMatch(payload=name)


def parse_workspace_create_invalid_command(text: str) -> SlackCommandMatch | None:
    if is_workspace_create_attempt(text) and parse_workspace_create(text) is None:
        return SlackCommandMatch()
    return None


def parse_workspace_select_command(text: str) -> SlackCommandMatch | None:
    index = parse_workspace_selection(text)
    if index is None:
        return None
    return SlackCommandMatch(payload=index)


def parse_workspace_select_invalid_command(text: str) -> SlackCommandMatch | None:
    if is_workspace_selection_attempt(text) and parse_workspace_selection(text) is None:
        return SlackCommandMatch()
    return None


def order_workspaces_for_listing(
    workspaces: list[SlackWorkspaceListItem],
    *,
    active_workspace_id: str,
) -> list[SlackWorkspaceListItem]:
    """Effective active workspace first (if present); then name, then id."""
    active_id = (active_workspace_id or "").strip()
    active_items = [
        item for item in workspaces if (item.workspace_id or "").strip() == active_id
    ]
    others = [
        item for item in workspaces if (item.workspace_id or "").strip() != active_id
    ]
    others_sorted = sorted(
        others,
        key=lambda item: (
            (item.name or "").strip().casefold(),
            (item.workspace_id or "").strip(),
        ),
    )
    if active_items:
        return active_items + others_sorted
    return sorted(
        workspaces,
        key=lambda item: (
            (item.name or "").strip().casefold(),
            (item.workspace_id or "").strip(),
        ),
    )


def order_sources_for_listing(
    sources: list[SlackSourceListItem],
) -> list[SlackSourceListItem]:
    """Deterministic order: safe label → source type → source_id (case-insensitive)."""
    return sorted(
        sources,
        key=lambda item: (
            (item.label or "").strip().casefold(),
            (item.source_type or "").strip().casefold(),
            (item.source_id or "").strip(),
        ),
    )


def order_source_candidates_for_listing(
    candidates: list[SlackSourceCandidateListItem],
) -> list[SlackSourceCandidateListItem]:
    """Deterministic order: safe normalized label casefold → candidate_id."""
    return sorted(
        candidates,
        key=lambda item: (
            (item.label or "").strip().casefold(),
            (item.candidate_id or "").strip(),
        ),
    )


def available_source_candidates_for_listing(
    candidates: list[SlackSourceCandidateListItem],
) -> list[SlackSourceCandidateListItem]:
    """Visible/selectable candidates: available only, ordered, capped."""
    available = [item for item in candidates if item.available is True]
    ordered = order_source_candidates_for_listing(available)
    return ordered[:MAX_SOURCE_CANDIDATE_ITEMS]


def normalize_source_list_items(
    sources: list[SlackSourceListItem],
) -> list[SlackSourceListItem]:
    """Ensure each item has a non-blank safe label before rendering."""
    normalized: list[SlackSourceListItem] = []
    for item in sources:
        label = (item.label or "").strip()
        if not label:
            item = item.model_copy(update={"label": _SAFE_SOURCE_LABEL_FALLBACK})
        normalized.append(item)
    return normalized


def resolve_effective_workspace_id(
    selection: SlackWorkspaceSelection | None,
    *,
    configured_workspace_id: str,
    configured_suppressed: bool = False,
) -> str:
    """Selected workspace ID when present; otherwise configured fallback."""
    if selection is not None:
        selected_id = (selection.workspace_id or "").strip()
        if selected_id:
            return selected_id
    if configured_suppressed:
        return ""
    return (configured_workspace_id or "").strip()


class SlackAskWorkflow:
    """LKW product workflow over the conversation-channel contract."""

    def __init__(
        self,
        *,
        auth_config: SlackCompanionAuthConfig,
        dedupe: SlackEventDedupeRepository,
        ask_client: WorkspaceAskHttpClient,
        send: OutboundSender,
        selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
        pending_deletion_store: InMemorySlackPendingDeletionStore | None = None,
        attachment_fetcher: ConversationAttachmentFetcher | None = None,
        attachment_max_bytes: int = 25 * 1024 * 1024,
        attachment_max_batch_files: int = 20,
        interaction_application_service: ConversationInteractionApplicationService | None = None,
        conversation_connection_ref: str = "slack",
    ) -> None:
        if (
            not isinstance(attachment_max_bytes, int)
            or isinstance(attachment_max_bytes, bool)
            or attachment_max_bytes < 1
        ):
            raise ValueError("attachment_max_bytes must be >= 1")
        if (
            not isinstance(attachment_max_batch_files, int)
            or isinstance(attachment_max_batch_files, bool)
            or attachment_max_batch_files < 1
        ):
            raise ValueError("attachment_max_batch_files must be >= 1")
        self._auth = auth_config
        self._dedupe = dedupe
        self._ask = ask_client
        self._send = send
        self._selections = selection_store or InMemorySlackWorkspaceSelectionStore()
        self._pending_deletions = (
            pending_deletion_store or InMemorySlackPendingDeletionStore()
        )
        self._attachment_fetcher = attachment_fetcher
        self._attachment_max_bytes = attachment_max_bytes
        self._attachment_max_batch_files = attachment_max_batch_files
        self._interaction_application_service = interaction_application_service
        self._conversation_connection_ref = conversation_connection_ref.strip() or "slack"
        self._commands = discover_slack_commands(self)

    def _resolve_effective_workspace(
        self, actor_key: str, configured_workspace_id: str
    ) -> str:
        return resolve_effective_workspace_id(
            self._selections.get(actor_key),
            configured_workspace_id=configured_workspace_id,
            configured_suppressed=self._selections.is_configured_suppressed(actor_key),
        )

    @classmethod
    def from_backend(
        cls,
        backend: ConversationChannelBackend,
        *,
        auth_config: SlackCompanionAuthConfig,
        dedupe: SlackEventDedupeRepository,
        ask_client: WorkspaceAskHttpClient,
        selection_store: InMemorySlackWorkspaceSelectionStore | None = None,
        pending_deletion_store: InMemorySlackPendingDeletionStore | None = None,
        attachment_max_bytes: int = 25 * 1024 * 1024,
        attachment_max_batch_files: int = 20,
        interaction_application_service: ConversationInteractionApplicationService | None = None,
        conversation_connection_ref: str = "slack",
    ) -> SlackAskWorkflow:
        fetcher: ConversationAttachmentFetcher | None = (
            backend if isinstance(backend, ConversationAttachmentFetcher) else None
        )
        return cls(
            auth_config=auth_config,
            dedupe=dedupe,
            ask_client=ask_client,
            send=backend.send,
            selection_store=selection_store,
            pending_deletion_store=pending_deletion_store,
            attachment_fetcher=fetcher,
            attachment_max_bytes=attachment_max_bytes,
            attachment_max_batch_files=attachment_max_batch_files,
            interaction_application_service=interaction_application_service,
            conversation_connection_ref=conversation_connection_ref,
        )

    async def handle(self, event: InboundConversationEvent) -> None:
        authorized = authorize_inbound_message(event, config=self._auth)
        if authorized is None:
            return

        interaction_service = self._interaction_application_service
        if interaction_service is not None:
            await self._handle_interaction(event, authorized, interaction_service)
            return

        claim = self._dedupe.claim(
            team_id=authorized.team_id,
            event_id=authorized.event_id,
        )
        if claim is None:
            return

        address = event.address
        actor_key = slack_selection_actor_key(
            team_id=authorized.team_id,
            user_id=authorized.user_id,
        )
        context = SlackCommandContext(
            event=event,
            address=address,
            authorized=authorized,
            claim=claim,
            actor_key=actor_key,
        )

        if event.attachments:
            await self._handle_attachments(context)
            return

        resolved = self._commands.match(authorized.text)
        if resolved is not None:
            await resolved.handler(context, resolved.match)
            return

        await self._handle_regular_ask(context)

    async def _handle_interaction(
        self,
        event: InboundConversationEvent,
        authorized: AuthorizedSlackMessageContext,
        interaction_service: ConversationInteractionApplicationService,
    ) -> None:
        if event.metadata.get("slack_channel_type") != "im":
            return
        thread_ref = (event.address.thread_id or event.address.conversation_id).strip()
        ingress = ConversationIngressContextV1(
            conversation_connection_ref=self._conversation_connection_ref,
            opaque_conversation_ref=event.address.conversation_id,
            opaque_thread_ref=thread_ref,
            actor_principal_ref=authorized.user_id,
            observed_audience=ConversationObservedAudience.PERSONAL,
            activation_signal=ConversationActivationSignal.ORDINARY_MESSAGE,
            provider_event_ref=authorized.event_id,
        )
        command = ConversationInteractionApplicationCommand(
            tenant_id=authorized.tenant_id,
            ingress=ingress,
            message_text=authorized.text,
            attachments=event.attachments,
        )
        result = await interaction_service.handle(command)
        if not result.should_send or not result.response_text:
            return
        try:
            await self._send(
                OutboundConversationMessage(
                    address=event.address,
                    text=result.response_text,
                )
            )
        except Exception as exc:  # noqa: BLE001 - execution must not be retried
            logger.warning(
                "slack_companion interaction_response_send_failed kind=%s",
                type(exc).__name__,
            )
            interaction_service.mark_response_failed(result)
            return
        interaction_service.mark_response_sent(result)

    async def _handle_attachments(self, context: SlackCommandContext) -> None:
        address = context.address
        claim = context.claim
        actor_key = context.actor_key
        authorized = context.authorized
        attachments = context.event.attachments

        selection = self._selections.get(actor_key)
        workspace_id = self._resolve_effective_workspace(
            actor_key, authorized.workspace_id
        )
        used_in_memory_selection = (
            selection is not None and bool((selection.workspace_id or "").strip())
        )

        if not workspace_id:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_no_workspace_available(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        if len(attachments) > self._attachment_max_batch_files:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_too_many(self._attachment_max_batch_files),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        if self._attachment_fetcher is None:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_fetch_unavailable(),
                )
            )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return

        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_receiving(len(attachments)),
                )
            )
        except Exception as exc:  # noqa: BLE001 — ack failure must not block intake
            logger.warning(
                "slack_companion attachment_ack_failed kind=%s",
                type(exc).__name__,
            )

        downloaded: list[ConversationAttachmentContent] = []
        try:
            for reference in attachments:
                content = await self._attachment_fetcher.fetch_attachment(
                    reference,
                    max_bytes=self._attachment_max_bytes,
                )
                downloaded.append(content)
        except ConversationAttachmentFetchError as exc:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_fetch_failed(exc.kind),
                )
            )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion attachment_fetch_failed kind=%s",
                type(exc).__name__,
            )
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_fetch_failed("attachment_download_failed"),
                )
            )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return

        idempotency_key = slack_attachment_intake_idempotency_key(
            team_id=authorized.team_id,
            event_id=authorized.event_id,
        )
        try:
            batch = await self._ask.upload_managed_files(
                tenant_id=authorized.tenant_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
                attachments=downloaded,
            )
        except SlackAskClientError as exc:
            if used_in_memory_selection and exc.kind == "http_404":
                self._selections.clear(actor_key)
                try:
                    await self._send(
                        OutboundConversationMessage(
                            address=address,
                            text=render_selected_workspace_unavailable(),
                        )
                    )
                except Exception as send_exc:  # noqa: BLE001 — best-effort error delivery
                    logger.warning(
                        "slack_companion attachment_error_delivery_failed kind=%s",
                        type(send_exc).__name__,
                    )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_attachment_intake_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001 — best-effort error delivery
                logger.warning(
                    "slack_companion attachment_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion attachment_upload_failed kind=%s",
                type(exc).__name__,
            )
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_attachment_intake_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001 — best-effort error delivery
                logger.warning(
                    "slack_companion attachment_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return

        self._dedupe.mark_completed(
            dedupe_key=claim.dedupe_key,
            claim_token=claim.claim_token,
            ask_run_id=None,
        )
        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_attachment_batch_response(batch),
                )
            )
        except Exception as exc:  # noqa: BLE001 — intake already completed
            logger.warning(
                "slack_companion attachment_summary_delivery_failed kind=%s",
                type(exc).__name__,
            )

    @slack_command(
        command_id="help",
        syntax="help",
        description="Show available commands.",
        example="help",
        priority=10,
        parser=parse_help_command,
    )
    async def _command_help(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        text = render_command_help(self._commands.visible_commands())
        await self._send(
            OutboundConversationMessage(address=context.address, text=text)
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    @slack_command(
        command_id="workspaces.list",
        syntax="workspaces",
        description="List available workspaces and show the active one.",
        example="workspaces",
        priority=20,
        parser=parse_workspaces_list_command,
    )
    async def _command_workspaces_list(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._handle_workspaces_listing(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
        )

    @slack_command(
        command_id="sources.list",
        syntax="sources",
        description="List sources in the active workspace.",
        example="sources",
        priority=25,
        parser=parse_sources_list_command,
    )
    async def _command_sources_list(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._handle_sources_listing(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
        )

    @slack_command(
        command_id="source_candidates.list",
        syntax="source candidates",
        description="List sources that can be attached to the active workspace.",
        example="source candidates",
        priority=22,
        parser=parse_source_candidates_list_command,
    )
    async def _command_source_candidates_list(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._handle_source_candidates_listing(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
        )

    @slack_command(
        command_id="source_candidates.accept",
        syntax="source add <number>",
        description="Attach a source from the current candidate list.",
        example="source add 2",
        priority=23,
        parser=parse_source_candidate_accept_command,
    )
    async def _command_source_candidates_accept(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        index = match.payload
        if not isinstance(index, int):
            raise TypeError("source_candidates.accept payload must be int")
        await self._handle_source_candidate_accept(
            address=context.address,
            claim=context.claim,
            authorized=context.authorized,
            actor_key=context.actor_key,
            index=index,
        )

    @slack_command(
        command_id="source_candidates.accept.invalid",
        syntax="source add",
        description="Invalid source add usage.",
        example="",
        priority=24,
        parser=parse_source_candidate_accept_invalid_command,
        visible_in_help=False,
    )
    async def _command_source_candidates_accept_invalid(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._send(
            OutboundConversationMessage(
                address=context.address,
                text=render_source_candidate_usage(),
            )
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    @slack_command(
        command_id="workspace.delete.confirm",
        syntax="workspace delete confirm",
        description="Confirm the pending workspace deletion.",
        example="workspace delete confirm",
        priority=30,
        parser=parse_workspace_delete_confirm_command,
    )
    async def _command_workspace_delete_confirm(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._handle_workspace_delete_confirm(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
        )

    @slack_command(
        command_id="workspace.delete.cancel",
        syntax="workspace delete cancel",
        description="Cancel the pending workspace deletion.",
        example="workspace delete cancel",
        priority=40,
        parser=parse_workspace_delete_cancel_command,
    )
    async def _command_workspace_delete_cancel(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        self._pending_deletions.clear(context.actor_key)
        await self._send(
            OutboundConversationMessage(
                address=context.address,
                text=render_workspace_delete_cancelled(),
            )
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    @slack_command(
        command_id="workspace.delete.request",
        syntax="workspace delete <number>",
        description="Prepare a workspace for deletion.",
        example="workspace delete 2",
        priority=50,
        parser=parse_workspace_delete_request_command,
    )
    async def _command_workspace_delete_request(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        index = match.payload
        if not isinstance(index, int):
            raise TypeError("workspace.delete.request payload must be int")
        await self._handle_workspace_delete_request(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
            index=index,
        )

    @slack_command(
        command_id="workspace.delete.invalid",
        syntax="workspace delete",
        description="Invalid workspace delete usage.",
        example="",
        priority=55,
        parser=parse_workspace_delete_invalid_command,
        visible_in_help=False,
    )
    async def _command_workspace_delete_invalid(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._send(
            OutboundConversationMessage(
                address=context.address,
                text=render_workspace_delete_usage(),
            )
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    @slack_command(
        command_id="workspace.create",
        syntax="workspace create <name>",
        description="Create a workspace and select it.",
        example="workspace create Contracts",
        priority=60,
        parser=parse_workspace_create_command,
    )
    async def _command_workspace_create(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        name = match.payload
        if not isinstance(name, str):
            raise TypeError("workspace.create payload must be str")
        await self._handle_workspace_create(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            name=name,
        )

    @slack_command(
        command_id="workspace.create.invalid",
        syntax="workspace create",
        description="Invalid workspace create usage.",
        example="",
        priority=65,
        parser=parse_workspace_create_invalid_command,
        visible_in_help=False,
    )
    async def _command_workspace_create_invalid(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._send(
            OutboundConversationMessage(
                address=context.address,
                text=render_workspace_create_usage(),
            )
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    @slack_command(
        command_id="workspace.select",
        syntax="workspace <number>",
        description="Select a workspace from the current list.",
        example="workspace 2",
        priority=70,
        parser=parse_workspace_select_command,
    )
    async def _command_workspace_select(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        index = match.payload
        if not isinstance(index, int):
            raise TypeError("workspace.select payload must be int")
        await self._handle_workspace_selection(
            address=context.address,
            claim=context.claim,
            tenant_id=context.authorized.tenant_id,
            actor_key=context.actor_key,
            configured_workspace_id=context.authorized.workspace_id,
            index=index,
        )

    @slack_command(
        command_id="workspace.selection.invalid",
        syntax="workspace",
        description="Invalid workspace selection usage.",
        example="",
        priority=75,
        parser=parse_workspace_select_invalid_command,
        visible_in_help=False,
    )
    async def _command_workspace_selection_invalid(
        self,
        context: SlackCommandContext,
        match: SlackCommandMatch,
    ) -> None:
        del match
        await self._send(
            OutboundConversationMessage(
                address=context.address,
                text=render_workspace_selection_usage(),
            )
        )
        self._dedupe.mark_completed(
            dedupe_key=context.claim.dedupe_key,
            claim_token=context.claim.claim_token,
            ask_run_id=None,
        )

    async def _handle_regular_ask(self, context: SlackCommandContext) -> None:
        address = context.address
        claim = context.claim
        actor_key = context.actor_key
        authorized = context.authorized
        question = authorized.text

        selection = self._selections.get(actor_key)
        workspace_id = self._resolve_effective_workspace(
            actor_key, authorized.workspace_id
        )
        used_in_memory_selection = (
            selection is not None and bool((selection.workspace_id or "").strip())
        )

        if not workspace_id:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_no_workspace_available(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_acknowledgement(),
                )
            )
        except Exception as exc:  # noqa: BLE001 — continue to Ask; final reply still attempted
            logger.warning(
                "slack_companion acknowledgement_failed kind=%s",
                type(exc).__name__,
            )

        try:
            ask_response = await self._ask.ask(
                tenant_id=authorized.tenant_id,
                workspace_id=workspace_id,
                question=question,
            )
            final_text = render_ask_response(ask_response)
            await self._send(
                OutboundConversationMessage(address=address, text=final_text)
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=ask_response.run_id or None,
            )
        except SlackAskClientError as exc:
            if used_in_memory_selection and exc.kind == "http_404":
                self._selections.clear(actor_key)
                await self._send_selected_unavailable(address=address, claim=claim)
            else:
                await self._send_error(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001 — product-safe error path
            logger.warning(
                "slack_companion workflow_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_error(address=address, claim=claim)

    async def _handle_workspace_create(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        name: str,
    ) -> None:
        try:
            created = await self._ask.create_workspace(tenant_id=tenant_id, name=name)
            self._selections.set(
                actor_key,
                SlackWorkspaceSelection(
                    workspace_id=(created.workspace_id or "").strip(),
                    workspace_name=(created.name or "").strip(),
                ),
            )
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_created(created.name),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError:
            await self._send_error(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion workspace_create_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_error(address=address, claim=claim)

    async def _handle_workspace_delete_request(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
        index: int,
    ) -> None:
        try:
            effective_workspace_id = self._resolve_effective_workspace(
                actor_key, configured_workspace_id
            )
            items = await self._ask.list_workspaces(tenant_id=tenant_id)
            ordered = order_workspaces_for_listing(
                items,
                active_workspace_id=effective_workspace_id,
            )
            if not ordered or index < 1 or index > len(ordered):
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_workspace_out_of_range(),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return

            chosen = ordered[index - 1]
            self._pending_deletions.set(
                actor_key,
                workspace_id=(chosen.workspace_id or "").strip(),
                workspace_name=(chosen.name or "").strip(),
            )
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_delete_confirmation(chosen.name),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError:
            await self._send_list_load_failed(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion workspace_delete_request_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_list_load_failed(address=address, claim=claim)

    async def _handle_workspace_delete_confirm(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
    ) -> None:
        pending = self._pending_deletions.consume_valid(actor_key)
        if pending is None:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_delete_missing_pending(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        try:
            await self._ask.delete_workspace(
                tenant_id=tenant_id,
                workspace_id=pending.workspace_id,
            )
            selection = self._selections.get(actor_key)
            selected_id = (
                (selection.workspace_id or "").strip() if selection is not None else ""
            )
            if selected_id and selected_id == pending.workspace_id:
                self._selections.clear(actor_key)
            configured_id = (configured_workspace_id or "").strip()
            if configured_id and configured_id == pending.workspace_id:
                if not self._selections.get(actor_key):
                    self._selections.suppress_configured(actor_key)

            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_deleted(pending.workspace_name),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError:
            await self._send_error(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion workspace_delete_confirm_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_error(address=address, claim=claim)

    async def _handle_workspaces_listing(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
    ) -> None:
        try:
            effective_workspace_id = self._resolve_effective_workspace(
                actor_key, configured_workspace_id
            )
            items = await self._ask.list_workspaces(tenant_id=tenant_id)
            ordered = order_workspaces_for_listing(
                items,
                active_workspace_id=effective_workspace_id,
            )
            final_text = render_workspace_list(
                ordered,
                active_workspace_id=effective_workspace_id,
            )
            await self._send(
                OutboundConversationMessage(address=address, text=final_text)
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError:
            await self._send_error(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001 — product-safe error path
            logger.warning(
                "slack_companion workspaces_listing_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_error(address=address, claim=claim)

    async def _handle_sources_listing(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
    ) -> None:
        try:
            workspace_id = self._resolve_effective_workspace(
                actor_key, configured_workspace_id
            )
            if not workspace_id:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_no_workspace_available(),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return

            items = await self._ask.list_sources(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            ordered = order_sources_for_listing(normalize_source_list_items(items))
            final_text = render_source_list(ordered)
            await self._send(
                OutboundConversationMessage(address=address, text=final_text)
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError as exc:
            if exc.kind == "http_404":
                text = render_source_workspace_unavailable()
            else:
                text = render_source_list_load_failed()
            try:
                await self._send(
                    OutboundConversationMessage(address=address, text=text)
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion sources_listing_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except Exception as exc:  # noqa: BLE001 — product-safe error path
            logger.warning(
                "slack_companion sources_listing_failed kind=%s",
                type(exc).__name__,
            )
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_source_list_load_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion sources_listing_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )

    async def _handle_source_candidates_listing(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
    ) -> None:
        selection = self._selections.get(actor_key)
        used_in_memory_selection = (
            selection is not None and bool((selection.workspace_id or "").strip())
        )
        try:
            workspace_id = self._resolve_effective_workspace(
                actor_key, configured_workspace_id
            )
            if not workspace_id:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_no_workspace_available(),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return

            items = await self._ask.list_source_candidates(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            visible = available_source_candidates_for_listing(items)
            if not visible:
                final_text = render_source_candidate_list_empty()
            else:
                final_text = render_source_candidate_list(visible)
            await self._send(
                OutboundConversationMessage(address=address, text=final_text)
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError as exc:
            if used_in_memory_selection and exc.kind == "http_404":
                self._selections.clear(actor_key)
                text = render_selected_workspace_unavailable()
            elif exc.kind in {
                "timeout",
                "transport_error",
                "parse_error",
            } or exc.kind.startswith("http_5") or exc.kind == "http_503":
                text = render_source_candidate_list_load_failed()
            elif exc.kind == "http_404":
                text = render_source_candidate_list_load_failed()
            else:
                text = render_source_candidate_list_load_failed()
            try:
                await self._send(
                    OutboundConversationMessage(address=address, text=text)
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion source_candidates_listing_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion source_candidates_listing_failed kind=%s",
                type(exc).__name__,
            )
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_source_candidate_list_load_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion source_candidates_listing_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )

    async def _handle_source_candidate_accept(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        authorized: AuthorizedSlackMessageContext,
        actor_key: str,
        index: int,
    ) -> None:
        selection = self._selections.get(actor_key)
        used_in_memory_selection = (
            selection is not None and bool((selection.workspace_id or "").strip())
        )
        workspace_id = self._resolve_effective_workspace(
            actor_key, authorized.workspace_id
        )
        if not workspace_id:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_no_workspace_available(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        try:
            items = await self._ask.list_source_candidates(
                tenant_id=authorized.tenant_id,
                workspace_id=workspace_id,
            )
        except SlackAskClientError as exc:
            if used_in_memory_selection and exc.kind == "http_404":
                self._selections.clear(actor_key)
                try:
                    await self._send(
                        OutboundConversationMessage(
                            address=address,
                            text=render_selected_workspace_unavailable(),
                        )
                    )
                except Exception as send_exc:  # noqa: BLE001
                    logger.warning(
                        "slack_companion source_candidate_accept_error_delivery_failed kind=%s",
                        type(send_exc).__name__,
                    )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_source_candidate_list_load_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion source_candidate_accept_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        visible = available_source_candidates_for_listing(items)
        if not visible:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_source_candidate_list_empty(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        if index < 1 or index > len(visible):
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_source_candidate_out_of_range(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        chosen = visible[index - 1]
        candidate_id = (chosen.candidate_id or "").strip()
        if not candidate_id:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_source_candidate_unavailable(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        idempotency_key = slack_source_candidate_intake_idempotency_key(
            team_id=authorized.team_id,
            event_id=authorized.event_id,
        )
        try:
            accepted = await self._ask.accept_source_candidate(
                tenant_id=authorized.tenant_id,
                workspace_id=workspace_id,
                candidate_id=candidate_id,
                idempotency_key=idempotency_key,
            )
        except SlackAskClientError as exc:
            if (
                exc.kind == "http_404"
                or exc.kind == "source_candidate_unavailable"
            ):
                text = render_source_candidate_unavailable()
                complete = True
            elif exc.kind == "source_candidate_already_registered":
                text = render_source_candidate_already_attached()
                complete = True
            elif exc.kind == "source_candidate_idempotency_conflict":
                text = render_source_candidate_selection_conflict()
                complete = True
            elif exc.kind == "http_409":
                text = render_source_candidate_accept_failed()
                complete = True
            elif exc.kind == "http_503":
                text = render_source_candidate_service_unavailable()
                complete = True
            elif exc.kind in {"timeout", "transport_error", "parse_error"} or (
                exc.kind.startswith("http_5")
            ):
                text = render_source_candidate_accept_failed()
                complete = False
            else:
                text = render_source_candidate_accept_failed()
                complete = False
            try:
                await self._send(
                    OutboundConversationMessage(address=address, text=text)
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion source_candidate_accept_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            if complete:
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
            else:
                self._dedupe.mark_failed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion source_candidate_accept_failed kind=%s",
                type(exc).__name__,
            )
            try:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_source_candidate_accept_failed(),
                    )
                )
            except Exception as send_exc:  # noqa: BLE001
                logger.warning(
                    "slack_companion source_candidate_accept_error_delivery_failed kind=%s",
                    type(send_exc).__name__,
                )
            self._dedupe.mark_failed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
            )
            return

        self._dedupe.mark_completed(
            dedupe_key=claim.dedupe_key,
            claim_token=claim.claim_token,
            ask_run_id=None,
        )
        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_source_candidate_accepted(accepted.label),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion source_candidate_accept_summary_delivery_failed kind=%s",
                type(exc).__name__,
            )

    async def _handle_workspace_selection(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
        tenant_id: str,
        actor_key: str,
        configured_workspace_id: str,
        index: int,
    ) -> None:
        try:
            effective_workspace_id = self._resolve_effective_workspace(
                actor_key, configured_workspace_id
            )
            items = await self._ask.list_workspaces(tenant_id=tenant_id)
            ordered = order_workspaces_for_listing(
                items,
                active_workspace_id=effective_workspace_id,
            )
            if not ordered:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_workspace_list([], active_workspace_id=""),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return

            if index < 1 or index > len(ordered):
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_workspace_out_of_range(),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return

            chosen = ordered[index - 1]
            self._selections.set(
                actor_key,
                SlackWorkspaceSelection(
                    workspace_id=(chosen.workspace_id or "").strip(),
                    workspace_name=(chosen.name or "").strip(),
                ),
            )
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_selected(chosen.name),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
        except SlackAskClientError:
            await self._send_list_load_failed(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001 — product-safe error path
            logger.warning(
                "slack_companion workspace_selection_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_list_load_failed(address=address, claim=claim)

    async def _send_list_load_failed(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
    ) -> None:
        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_list_load_failed(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion list_load_failed_delivery_failed kind=%s",
                type(exc).__name__,
            )
        self._dedupe.mark_failed(
            dedupe_key=claim.dedupe_key,
            claim_token=claim.claim_token,
        )

    async def _send_selected_unavailable(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
    ) -> None:
        try:
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_selected_workspace_unavailable(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion selected_unavailable_delivery_failed kind=%s",
                type(exc).__name__,
            )
        self._dedupe.mark_failed(
            dedupe_key=claim.dedupe_key,
            claim_token=claim.claim_token,
        )

    async def _send_error(
        self,
        *,
        address: ConversationAddress,
        claim: SlackDedupeRecord,
    ) -> None:
        try:
            await self._send(
                OutboundConversationMessage(address=address, text=render_error())
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "slack_companion error_delivery_failed kind=%s",
                type(exc).__name__,
            )
        self._dedupe.mark_failed(
            dedupe_key=claim.dedupe_key,
            claim_token=claim.claim_token,
        )
