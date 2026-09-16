# © Artur Czarnecki. All rights reserved.

"""Subprocess TCP delegated execution provider (P2.1-S2D real external boundary)."""

from intergrax.integrations.providers.delegated_execution.subprocess.bundle import (
    SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID,
    create_subprocess_delegated_execution_provider,
)
from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.provider import (
    SubprocessDelegatedExecutionProvider,
)

__all__ = [
    "SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID",
    "SubprocessDelegatedExecutionProvider",
    "SubprocessDelegatedExecutionProviderConfig",
    "create_subprocess_delegated_execution_provider",
]
