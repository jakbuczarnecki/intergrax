# © Artur Czarnecki. All rights reserved.

"""Backward-compatible import path — canonical resolver lives in Execution Engine."""

from intergrax.runtime.execution.context_engine_resolver import (
    ContextEngineImportError,
    load_context_engine,
)

__all__ = ["ContextEngineImportError", "load_context_engine"]
