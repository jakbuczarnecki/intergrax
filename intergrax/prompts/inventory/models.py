# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class PromptInstructionKind(str, Enum):
    """
    High level classification of LLM behavior instructions.

    These are not technical steps but semantic policies
    controlling model behavior.
    """

    # Planning & orchestration
    PLANNER = "planner"
    SUPERVISOR = "supervisor"

    # Knowledge usage
    RAG_POLICY = "rag_policy"
    CONTEXT_OVERFLOW = "context_overflow"
    HISTORY_SUMMARY = "history_summary"

    # Tools & skills
    TOOL_USAGE = "tool_usage"

    # System behavior
    SYSTEM_BEHAVIOR = "system_behavior"
    ERROR_POLICY = "error_policy"

    # Profiles & memory
    USER_PROFILE = "user_profile"
    ORG_PROFILE = "org_profile"


@dataclass(frozen=True)
class PromptInstructionLocation:
    """
    Static location of existing instruction in codebase.
    Used only for inventory / migration planning.
    """

    kind: PromptInstructionKind
    module: str
    symbol: str
    description: str


@dataclass(frozen=True)
class PromptInstructionInventory:
    """
    Source of truth about all LLM steering instructions
    currently present in the framework.

    This is a transitional structure used before
    full migration to Prompt Registry.
    """

    items: FrozenSet[PromptInstructionLocation]

    def by_kind(
        self, kind: PromptInstructionKind
    ) -> FrozenSet[PromptInstructionLocation]:
        return frozenset(i for i in self.items if i.kind == kind)
