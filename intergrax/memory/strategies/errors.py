# © Artur Czarnecki. All rights reserved.

"""Strategy failure types (MEM-ENT-4)."""

from __future__ import annotations


class MemoryStrategyError(Exception):
    """Base error for memory strategy failures."""


class MemoryStrategyContractError(MemoryStrategyError):
    """Invalid strategy input or contract violation."""


class MemoryStrategyProviderError(MemoryStrategyError):
    """External provider (e.g. LLM) failure during strategy execution."""
