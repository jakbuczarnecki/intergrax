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
from local_workspace_application.slack_companion.rendering import (
    render_acknowledgement,
    render_ask_response,
    render_error,
    render_selected_workspace_unavailable,
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
) -> str:
    """Selected workspace ID when present; otherwise configured fallback."""
    if selection is not None:
        selected_id = (selection.workspace_id or "").strip()
        if selected_id:
            return selected_id
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
    ) -> None:
        self._auth = auth_config
        self._dedupe = dedupe
        self._ask = ask_client
        self._send = send
        self._selections = selection_store or InMemorySlackWorkspaceSelectionStore()

    def _resolve_effective_workspace(
        self, actor_key: str, configured_workspace_id: str
    ) -> str:
        return resolve_effective_workspace_id(
            self._selections.get(actor_key),
            configured_workspace_id=configured_workspace_id,
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
    ) -> SlackAskWorkflow:
        return cls(
            auth_config=auth_config,
            dedupe=dedupe,
            ask_client=ask_client,
            send=backend.send,
            selection_store=selection_store,
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

        if is_workspaces_command(authorized.question):
            await self._handle_workspaces_listing(
                address=address,
                claim=claim,
                tenant_id=authorized.tenant_id,
                actor_key=actor_key,
                configured_workspace_id=authorized.workspace_id,
            )
            return

        selection_index = parse_workspace_selection(authorized.question)
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

        if is_workspace_selection_attempt(authorized.question):
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
                question=authorized.question,
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
