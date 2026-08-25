# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Type

from pydantic import BaseModel


class ToolRiskLevel(str, Enum):
    """Declared risk for governance, tracing, and planner filtering (§22)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SideEffectRetrySafety(str, Enum):
    """
    Positive retry authorization for side-effectful tools (TOOLS-05).

    ``NOT_RETRY_SAFE`` is the default — automatic runtime retry is forbidden
  until a tool contract explicitly proves retry safety.
    """

    NOT_RETRY_SAFE = "not_retry_safe"
    IDEMPOTENT = "idempotent"
    EXPLICITLY_RETRY_SAFE = "explicitly_retry_safe"


@dataclass(frozen=True, slots=True)
class ToolRetryPolicy:
    """
    Runtime-managed retry semantics for tool invocation (§22, §42.34).

    ``max_attempts=1`` means no retry (default). Agents MUST NOT implement
    their own retry loops for catalog tools.
    """

    max_attempts: int = 1
    backoff_ms: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("ToolRetryPolicy.max_attempts must be >= 1")
        if self.backoff_ms < 0:
            raise ValueError("ToolRetryPolicy.backoff_ms must be >= 0")


@dataclass(frozen=True, slots=True)
class ToolContract:
    """
    Formal runtime contract for an atomic **tool** (LLM/MCP invocable operation).

    Skills (composable capability packs) are defined separately — see architecture §7.1.8.

    Enforced by Nexus runtime (registry + validation + trace + error mapping).
    Optional metadata fields (§7.1.6, Phase O.1) default for backward compatibility.
    """

    tool_id: str
    name: str
    description: str

    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    error_mapping: Mapping[type[Exception], str]

    side_effects: bool

    version: str = "1.0.0"
    description_short: Optional[str] = None
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    timeout_ms: int = 30_000
    retry_policy: ToolRetryPolicy = ToolRetryPolicy()
    side_effect_retry_safety: SideEffectRetrySafety = SideEffectRetrySafety.NOT_RETRY_SAFE
    injects_context: bool = False
    category: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_ms < 1:
            raise ValueError("ToolContract.timeout_ms must be >= 1")

    def llm_description(self, *, compact: bool = False) -> str:
        """Description exposed to LLM tool-selection surfaces (OpenAI, MCP)."""
        if compact and self.description_short:
            return self.description_short
        return self.description
