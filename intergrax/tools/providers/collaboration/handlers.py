# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.collaboration.contracts import CollaborationSendMailInput, CollaborationSendMailOutput
from intergrax.tools.providers.collaboration.service import collaboration_send_mail


class CollaborationSendMailHandler(ServiceToolHandler[CollaborationSendMailInput, CollaborationSendMailOutput]):
    _service = collaboration_send_mail
