# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.sandbox.contracts import (
    BrowserRunInput,
    BrowserRunOutput,
    CodeExecInput,
    SandboxExecOutput,
    SandboxListOperationsInput,
    SandboxListOperationsOutput,
    ScriptRunInput,
)
from intergrax.tools.providers.sandbox.extended_service import (
    browser_run,
    code_exec,
    sandbox_list_operations,
    script_run,
)


class CodeExecHandler(ServiceToolHandler[CodeExecInput, SandboxExecOutput]):
    _service = code_exec


class ScriptRunHandler(ServiceToolHandler[ScriptRunInput, SandboxExecOutput]):
    _service = script_run


class BrowserRunHandler(ServiceToolHandler[BrowserRunInput, BrowserRunOutput]):
    _service = browser_run


class SandboxListOperationsHandler(ServiceToolHandler[SandboxListOperationsInput, SandboxListOperationsOutput]):
    _service = sandbox_list_operations
