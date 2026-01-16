# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.prompts.inventory.models import PromptInstructionInventory, PromptInstructionKind
from intergrax.prompts.inventory.registry import PromptInventoryBuilder


def build_default_inventory() -> PromptInstructionInventory:
    """
    Inventory of all current LLM steering instructions
    discovered in INTERGRAX bundle.

    This is a transitional map before migration to Prompt Registry.
    """

    b = PromptInventoryBuilder()

    # === NEXUS: CORE SYSTEM BEHAVIOR ===

    b.add(
        kind=PromptInstructionKind.SYSTEM_BEHAVIOR,
        module="intergrax.runtime.nexus.runtime_steps.instructions_step",
        symbol="InstructionsStep",
        description="Primary system behavior instructions defining how model should respond to user and context.",
    )

    # === PLANNING & SUPERVISOR ===

    b.add(
        kind=PromptInstructionKind.PLANNER,
        module="intergrax.runtime.nexus.planning.engine_planner",
        symbol="EnginePlanner",
        description="LLM instructions for planning next steps and selecting pipeline actions.",
    )

    b.add(
        kind=PromptInstructionKind.SUPERVISOR,
        module="intergrax.runtime.nexus.planning.step_planner",
        symbol="StepPlanner",
        description="Instructions controlling execution plan generation and validation.",
    )

    # === RAG & CONTEXT USAGE ===

    b.add(
        kind=PromptInstructionKind.RAG_POLICY,
        module="intergrax.runtime.nexus.prompts.rag_prompt_builder",
        symbol="RagPromptBuilder",
        description="Rules describing how retrieved knowledge must be used by the model.",
    )

    b.add(
        kind=PromptInstructionKind.CONTEXT_OVERFLOW,
        module="intergrax.runtime.nexus.context.engine_history_layer",
        symbol="EngineHistoryLayer",
        description="Policy describing how context should be reduced when exceeding limits.",
    )

    # === HISTORY & SUMMARIZATION ===

    b.add(
        kind=PromptInstructionKind.HISTORY_SUMMARY,
        module="intergrax.runtime.nexus.prompts.history_prompt_builder",
        symbol="HistoryPromptBuilder",
        description="Instructions for summarizing conversation history.",
    )

    b.add(
        kind=PromptInstructionKind.HISTORY_SUMMARY,
        module="intergrax.runtime.nexus.tracing.runtime.instructions_summary",
        symbol="SummaryInstructions",
        description="Additional summarization behavior for trace consolidation.",
    )

    # === TOOLS ===

    b.add(
        kind=PromptInstructionKind.TOOL_USAGE,
        module="intergrax.runtime.nexus.tools.tool_executor",
        symbol="ToolExecutor",
        description="Guidelines how model should call and interpret tools.",
    )

    # === PROFILES & MEMORY ===

    b.add(
        kind=PromptInstructionKind.USER_PROFILE,
        module="intergrax.runtime.nexus.profiles.user_profile_manager",
        symbol="UserProfileManager",
        description="Instructions for building and updating user profile memory.",
    )

    b.add(
        kind=PromptInstructionKind.ORG_PROFILE,
        module="intergrax.runtime.nexus.profiles.organization_profile_manager",
        symbol="OrganizationProfileManager",
        description="Instructions for organization memory consolidation.",
    )

    # === ERROR HANDLING ===

    b.add(
        kind=PromptInstructionKind.ERROR_POLICY,
        module="intergrax.runtime.nexus.policies.error_mapping",
        symbol="ErrorMappingPolicy",
        description="Implicit LLM behavior when errors and retries occur.",
    )

    return b.build()
