# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.policies.runtime_policies import ExecutionKind
from intergrax.runtime.nexus.tracing.memory.profile_base_memory_summary import ProfileBasedMemorySummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ProfileBasedMemoryStep(RuntimeStep):
    """
    Load profile-based instruction fragments for this request.

    Rules:
        - Use profile memory only if enabled in RuntimeConfig.
        - Do NOT rebuild or cache anything here yet (this is step 1 only).
        - Extract prebuilt 'system_prompt' strings from profile bundles.
        - Store the resulting fragments in RuntimeState so the engine
        can merge them into a system message later.
    """

    def execution_kind(self) -> ExecutionKind | None:
        return ExecutionKind.STORAGE

    async def run(self, state: RuntimeState) -> None:
        session = state.session
        assert session is not None, "Session must exist before memory layer."

        cfg = state.context.config

        user_instr: Optional[str] = None
        org_instr: Optional[str] = None

        # 1) User profile memory (optional)
        if cfg.enable_user_profile_memory:
            user_instr_candidate = (
                await state.context.session_manager.get_user_profile_instructions_for_session(
                    session=session
                )
            )
            if isinstance(user_instr_candidate, str):
                stripped = user_instr_candidate.strip()
                if stripped:
                    user_instr = stripped
                    state.used_user_profile = True

        # 2) Organization profile memory (optional)
        if cfg.enable_org_profile_memory:
            org_instr_candidate = (
                await state.context.session_manager.get_org_profile_instructions_for_session(
                    session=session
                )
            )
            if isinstance(org_instr_candidate, str):
                stripped = org_instr_candidate.strip()
                if stripped:
                    org_instr = stripped
                    # For now we reuse the same flag to indicate that some profile
                    # (user or organization) has been used.
                    state.used_user_profile = True

        # 3) Store extracted profile instruction fragments in state
        state.profile_user_instructions = user_instr
        state.profile_org_instructions = org_instr

        # 4) Trace memory layer step (typed payload).
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="memory_layer",
            message="Profile-based instructions loaded for session.",
            level=TraceLevel.INFO,
            payload=ProfileBasedMemorySummaryDiagV1(
                has_user_profile_instructions=bool(user_instr),
                has_org_profile_instructions=bool(org_instr),
                enable_user_profile_memory=cfg.enable_user_profile_memory,
                enable_org_profile_memory=cfg.enable_org_profile_memory,
            ),
        )