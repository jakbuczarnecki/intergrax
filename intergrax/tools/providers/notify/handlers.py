# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.notify.contracts import (
    NotifyScheduleInput,
    NotifyScheduleOutput,
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.providers.notify.service import notify_schedule, notify_send, notify_send_batch


class NotifySendHandler(ServiceToolHandler[NotifySendInput, NotifySendOutput]):
    _service = notify_send


class NotifySendBatchHandler(ServiceToolHandler[NotifySendBatchInput, NotifySendBatchOutput]):
    _service = notify_send_batch


class NotifyScheduleHandler(ServiceToolHandler[NotifyScheduleInput, NotifyScheduleOutput]):
    _service = notify_schedule
