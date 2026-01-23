# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.nexus.run_factory.contracts import RuntimeRunHandle


class RuntimeRunExecutor(Protocol):
    """
    Executes an API-originated runtime run.
    """

    async def execute(self, *, handle: RuntimeRunHandle) -> None: ...
