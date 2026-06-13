# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.codecraft.contracts import (
    CodeCraftDisposeToolInput,
    CodeCraftDisposeToolOutput,
    CodeCraftGetStateToolInput,
    CodeCraftGetStateToolOutput,
    CodeCraftIterateToolInput,
    CodeCraftIterateToolOutput,
    CodeCraftListEphemeralToolsInput,
    CodeCraftListEphemeralToolsOutput,
    CodeCraftPromoteToolInput,
    CodeCraftPromoteToolOutput,
    CodeCraftRunToolInput,
    CodeCraftRunToolOutput,
    CodeCraftStartToolInput,
    CodeCraftStartToolOutput,
)
from intergrax.tools.providers.codecraft.service import (
    codecraft_dispose,
    codecraft_get_state,
    codecraft_iterate,
    codecraft_list_ephemeral_tools,
    codecraft_promote,
    codecraft_run,
    codecraft_start,
)


class CodeCraftRunHandler(ServiceToolHandler[CodeCraftRunToolInput, CodeCraftRunToolOutput]):
    _service = codecraft_run


class CodeCraftStartHandler(ServiceToolHandler[CodeCraftStartToolInput, CodeCraftStartToolOutput]):
    _service = codecraft_start


class CodeCraftIterateHandler(ServiceToolHandler[CodeCraftIterateToolInput, CodeCraftIterateToolOutput]):
    _service = codecraft_iterate


class CodeCraftGetStateHandler(ServiceToolHandler[CodeCraftGetStateToolInput, CodeCraftGetStateToolOutput]):
    _service = codecraft_get_state


class CodeCraftDisposeHandler(ServiceToolHandler[CodeCraftDisposeToolInput, CodeCraftDisposeToolOutput]):
    _service = codecraft_dispose


class CodeCraftPromoteHandler(ServiceToolHandler[CodeCraftPromoteToolInput, CodeCraftPromoteToolOutput]):
    _service = codecraft_promote


class CodeCraftListEphemeralToolsHandler(
    ServiceToolHandler[CodeCraftListEphemeralToolsInput, CodeCraftListEphemeralToolsOutput]
):
    _service = codecraft_list_ephemeral_tools
