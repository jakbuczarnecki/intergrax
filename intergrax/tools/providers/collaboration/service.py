# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.tools.providers.collaboration.contracts import CollaborationSendMailInput, CollaborationSendMailOutput
from intergrax.tools.registry.wiring import ToolWiringContext

COLLABORATION_SEND_MAIL_TOOL_ID = "collaboration.send_mail"


def _require_suite(ctx: ToolWiringContext) -> CollaborationSuite:
    suite = ctx.collaboration_suite
    if suite is None:
        raise RuntimeError("collaboration_suite_not_configured")
    return suite


def collaboration_send_mail(ctx: ToolWiringContext, params: CollaborationSendMailInput) -> CollaborationSendMailOutput:
    recipients = [item.strip() for item in params.to if item.strip()]
    _require_suite(ctx).send_mail(
        params.user_id.strip(),
        subject=params.subject,
        body=params.body,
        to=recipients,
    )
    return CollaborationSendMailOutput(sent=True, recipient_count=len(recipients))
