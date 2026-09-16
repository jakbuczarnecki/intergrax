# © Artur Czarnecki. All rights reserved.

"""Composition factory for subprocess delegated execution provider."""

from __future__ import annotations

from intergrax.integrations.providers.delegated_execution.subprocess.config import (
    SubprocessDelegatedExecutionProviderConfig,
)
from intergrax.integrations.providers.delegated_execution.subprocess.provider import (
    SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID,
    SubprocessDelegatedExecutionProvider,
    SubprocessEchoPayload,
    SubprocessEchoResult,
)
from intergrax.integrations.providers.delegated_execution.subprocess.transport import (
    SubprocessDelegatedExecutionTransport,
)

__all__ = [
    "SUBPROCESS_DELEGATED_EXECUTION_PROVIDER_ID",
    "SubprocessEchoPayload",
    "SubprocessEchoResult",
    "create_subprocess_delegated_execution_provider",
]


def create_subprocess_delegated_execution_provider(
    *,
    config: SubprocessDelegatedExecutionProviderConfig | None = None,
    transport: SubprocessDelegatedExecutionTransport | None = None,
) -> SubprocessDelegatedExecutionProvider[SubprocessEchoPayload, SubprocessEchoResult]:
    resolved = config or SubprocessDelegatedExecutionProviderConfig(enabled=True)
    return SubprocessDelegatedExecutionProvider(resolved, transport=transport)
