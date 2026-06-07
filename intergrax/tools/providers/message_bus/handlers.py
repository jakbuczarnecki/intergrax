# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.message_bus.contracts import (
    MessageBusCancelInput,
    MessageBusCancelOutput,
    MessageBusEnqueueInput,
    MessageBusEnqueueOutput,
    MessageBusGetResultInput,
    MessageBusGetResultOutput,
    MessageBusGetStatusInput,
    MessageBusGetStatusOutput,
    MessageBusListTasksInput,
    MessageBusListTasksOutput,
)
from intergrax.tools.providers.message_bus.service import (
    message_bus_cancel,
    message_bus_enqueue,
    message_bus_get_result,
    message_bus_get_status,
    message_bus_list_tasks,
)


class MessageBusEnqueueHandler(ServiceToolHandler[MessageBusEnqueueInput, MessageBusEnqueueOutput]):
    _service = message_bus_enqueue


class MessageBusGetStatusHandler(ServiceToolHandler[MessageBusGetStatusInput, MessageBusGetStatusOutput]):
    _service = message_bus_get_status


class MessageBusGetResultHandler(ServiceToolHandler[MessageBusGetResultInput, MessageBusGetResultOutput]):
    _service = message_bus_get_result


class MessageBusListTasksHandler(ServiceToolHandler[MessageBusListTasksInput, MessageBusListTasksOutput]):
    _service = message_bus_list_tasks


class MessageBusCancelHandler(ServiceToolHandler[MessageBusCancelInput, MessageBusCancelOutput]):
    _service = message_bus_cancel
