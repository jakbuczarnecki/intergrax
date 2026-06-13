# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.codecraft.contracts import CodeCraftRunToolInput, CodeCraftRunToolOutput
from intergrax.tools.providers.codecraft.service import codecraft_run


class CodeCraftRunHandler(ServiceToolHandler[CodeCraftRunToolInput, CodeCraftRunToolOutput]):
    _service = codecraft_run
