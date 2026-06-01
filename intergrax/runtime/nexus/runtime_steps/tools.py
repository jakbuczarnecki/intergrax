# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Backward-compatible re-export (Phase Q-N.7). Prefer ``tool_context_helpers``."""

from intergrax.runtime.nexus.runtime_steps.tool_context_helpers import (
    format_rag_context,
    insert_context_before_last_user,
)

__all__ = ["format_rag_context", "insert_context_before_last_user"]
