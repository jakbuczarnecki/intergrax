# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from enum import Enum
from typing import Literal

ToolChoiceMode = Literal["off", "auto", "required"]


class ToolsContextScope(str, Enum):
    CURRENT_MESSAGE_ONLY = "current_message_only"
    CONVERSATION = "conversation"
    FULL = "full"


class ToolSelectionMode(str, Enum):
    """How ToolsStep narrows the planner tool schema before LLM selection (TOOL-ENG-5)."""

    STATIC = "static"
    SKILL_PACK = "skill_pack"
    RETRIEVAL_TOP_K = "retrieval_top_k"
    KEYWORD_TOP_K = "keyword_top_k"  # alias for retrieval_top_k (TOOL-ENG-15)
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    FULL_CATALOG = "full_catalog"


class ToolInvocationMode(str, Enum):
    """How ToolsStep orchestrates a tool call batch (TOOL-ENG-21)."""

    SINGLE_PASS = "single_pass"
    BOUNDED_REACT = "bounded_react"
    PARALLEL_BATCH = "parallel_batch"
    DETERMINISTIC_CHAIN = "deterministic_chain"
    PARALLEL_SEMANTIC_BATCH = "parallel_semantic_batch"
