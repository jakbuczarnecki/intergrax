# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.notify.contracts import NotifySendInput, NotifySendOutput
from intergrax.tools.providers.notify.service import notify_send


class NotifySendHandler(ServiceToolHandler[NotifySendInput, NotifySendOutput]):
    _service = notify_send
