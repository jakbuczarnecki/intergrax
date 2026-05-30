# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput, SandboxExecOutput
from intergrax.tools.providers.sandbox.service import sandbox_exec


class SandboxExecHandler(ServiceToolHandler[SandboxExecInput, SandboxExecOutput]):
    _service = sandbox_exec
