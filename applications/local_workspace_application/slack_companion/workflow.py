# © Artur Czarnecki. All rights reserved.

"""Inbound MESSAGE → configured workspace → Ask HTTP → threaded answer."""

from __future__ import annotations

import logging
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
)
from local_workspace_application.slack_companion.rendering import (
    render_acknowledgement,
    render_ask_response,
    render_error,
)

logger = logging.getLogger(__name__)

OutboundSender = Callable[[OutboundConversationMessage], Awaitable[object]]


class SlackAskWorkflow:
    """LKW product workflow over the conversation-channel contract."""

    def __init__(
        self,
        *,
        auth_config: SlackCompanionAuthConfig,
        dedupe: SlackEventDedupeRepository,
        ask_client: WorkspaceAskHttpClient,
        send: OutboundSender,
    ) -> None:
        self._auth = auth_config
        self._dedupe = dedupe
        self._ask = ask_client
        self._send = send

    @classmethod
    def from_backend(
        cls,
        backend: ConversationChannelBackend,
        *,
        auth_config: SlackCompanionAuthConfig,
        dedupe: SlackEventDedupeRepository,
        ask_client: WorkspaceAskHttpClient,
    ) -> SlackAskWorkflow:
        return cls(
            auth_config=auth_config,
            dedupe=dedupe,
            ask_client=ask_client,
            send=backend.send,
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
                workspace_id=authorized.workspace_id,
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
        except SlackAskClientError:
            await self._send_error(address=address, claim=claim)
        except Exception as exc:  # noqa: BLE001 — product-safe error path
            logger.warning(
                "slack_companion workflow_failed kind=%s",
                type(exc).__name__,
            )
            await self._send_error(address=address, claim=claim)

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
