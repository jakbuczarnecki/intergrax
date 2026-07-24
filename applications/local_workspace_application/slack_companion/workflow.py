# © Artur Czarnecki. All rights reserved.

"""Inbound MESSAGE → configured/selected workspace → Ask HTTP → threaded answer."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from intergrax.integrations.contracts.conversation_channel import (
    ConversationAddress,
    ConversationChannelBackend,
    InboundConversationEvent,
    OutboundConversationMessage,
)
from local_workspace_application.slack_companion.ask_client import WorkspaceAskHttpClient
from local_workspace_application.slack_companion.authorization import (
    SlackCompanionAuthConfig,
    authorize_inbound_ask,
)
from local_workspace_application.slack_companion.dedupe_repository import (
    SlackEventDedupeRepository,
)
from local_workspace_application.slack_companion.models import (
    SlackAskClientError,
    SlackDedupeRecord,
    SlackWorkspaceListItem,
)
from local_workspace_application.slack_companion.pending_deletion_store import (
    InMemorySlackPendingDeletionStore,
)
from local_workspace_application.slack_companion.rendering import (
    MAX_WORKSPACE_NAME_CHARS,
    render_acknowledgement,
    render_ask_response,
    render_error,
    render_no_workspace_available,
    render_selected_workspace_unavailable,
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

logger = logging.getLogger(__name__)

OutboundSender = Callable[[OutboundConversationMessage], Awaitable[object]]

_WORKSPACES_COMMAND = "workspaces"
_WORKSPACE_SELECTION_RE = re.compile(
    r"^workspace\s+([1-9]\d*)$",
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
    ) -> None:
        self._auth = auth_config
        self._dedupe = dedupe
        self._ask = ask_client
        self._send = send
        self._selections = selection_store or InMemorySlackWorkspaceSelectionStore()
        self._pending_deletions = (
            pending_deletion_store or InMemorySlackPendingDeletionStore()
        )

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
    ) -> SlackAskWorkflow:
        return cls(
            auth_config=auth_config,
            dedupe=dedupe,
            ask_client=ask_client,
            send=backend.send,
            selection_store=selection_store,
            pending_deletion_store=pending_deletion_store,
        )

    async def handle(self, event: InboundConversationEvent) -> None:
        authorized = authorize_inbound_ask(event, config=self._auth)
        if authorized is None:
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
        question = authorized.question

        if is_workspaces_command(question):
            await self._handle_workspaces_listing(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                configured_workspace_id=authorized.workspace_id,
            )
            return

        if is_workspace_create_attempt(question):
            created_name = parse_workspace_create(question)
            if created_name is None:
                await self._send(
                    OutboundConversationMessage(
                        address=address,
                        text=render_workspace_create_usage(),
                    )
                )
                self._dedupe.mark_completed(
                    dedupe_key=claim.dedupe_key,
                    claim_token=claim.claim_token,
                    ask_run_id=None,
                )
                return
            await self._handle_workspace_create(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                name=created_name,
            )
            return

        if is_workspace_delete_confirm(question):
            await self._handle_workspace_delete_confirm(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                configured_workspace_id=authorized.workspace_id,
            )
            return

        if is_workspace_delete_cancel(question):
            self._pending_deletions.clear(actor_key)
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_delete_cancelled(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        delete_index = parse_workspace_delete(question)
        if delete_index is not None:
            await self._handle_workspace_delete_request(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                configured_workspace_id=authorized.workspace_id,
                index=delete_index,
            )
            return

        if is_workspace_delete_attempt(question):
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_delete_usage(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

        selection_index = parse_workspace_selection(question)
        if selection_index is not None:
            await self._handle_workspace_selection(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                configured_workspace_id=authorized.workspace_id,
                index=selection_index,
            )
            return

        if is_workspace_selection_attempt(question):
            await self._send(
                OutboundConversationMessage(
                    address=address,
                    text=render_workspace_selection_usage(),
                )
            )
            self._dedupe.mark_completed(
                dedupe_key=claim.dedupe_key,
                claim_token=claim.claim_token,
                ask_run_id=None,
            )
            return

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
