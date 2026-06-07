# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.notify.contracts import (
    NotifyCancelScheduledInput,
    NotifyCancelScheduledOutput,
    NotifyListScheduledInput,
    NotifyListScheduledOutput,
    NotifyScheduleInput,
    NotifyScheduleOutput,
    NotifySendBatchInput,
    NotifySendBatchOutput,
    NotifySendInput,
    NotifySendOutput,
)
from intergrax.tools.providers.notify.service import (
    notify_cancel_scheduled,
    notify_list_scheduled,
    notify_schedule,
    notify_send,
    notify_send_batch,
)


class NotifySendHandler(ServiceToolHandler[NotifySendInput, NotifySendOutput]):
    _service = notify_send


class NotifySendBatchHandler(ServiceToolHandler[NotifySendBatchInput, NotifySendBatchOutput]):
    _service = notify_send_batch


class NotifyScheduleHandler(ServiceToolHandler[NotifyScheduleInput, NotifyScheduleOutput]):
    _service = notify_schedule


class NotifyListScheduledHandler(ServiceToolHandler[NotifyListScheduledInput, NotifyListScheduledOutput]):
    _service = notify_list_scheduled


class NotifyCancelScheduledHandler(ServiceToolHandler[NotifyCancelScheduledInput, NotifyCancelScheduledOutput]):
    _service = notify_cancel_scheduled
