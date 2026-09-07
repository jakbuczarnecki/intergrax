# © Artur Czarnecki. All rights reserved.

"""Delegated execution provider runtime adapters (P2.1-S1)."""

from intergrax.runtime.execution.delegated_execution.context_projection import (
    project_delegated_execution_context,
)
from intergrax.runtime.execution.delegated_execution.local_provider import (
    LocalDelegatedExecutionDelegate,
    LocalDelegatedExecutionProvider,
)

__all__ = [
    "LocalDelegatedExecutionDelegate",
    "LocalDelegatedExecutionProvider",
    "project_delegated_execution_context",
]
