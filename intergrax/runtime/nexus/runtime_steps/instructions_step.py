# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.tracing.runtime.instructions_summary import InstructionsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel


class InstructionsStep(RuntimeStep):
    """
    Inject the final instructions as the first `system` message in the LLM prompt,
    if any instructions exist.

    Combines:
      1) per-request instructions (RuntimeRequest.instructions),
      2) user profile instructions (state.profile_user_instructions),
      3) organization profile instructions (state.profile_org_instructions).

    Must be called AFTER history step to ensure:
      - instructions are always the first system message,
      - instructions are never persisted in SessionStore,
      - history can be trimmed/summarized freely before injection.
    """

    async def run(self, state: RuntimeState) -> None:
        instructions_text = self._build_final_instructions(state)
        if not instructions_text:
            return

        system_message = ChatMessage(role="system", content=instructions_text)

        # `messages_for_llm` at this point should contain only history
        # (built by HistoryStep). We now prepend the system message.
        state.messages_for_llm = [system_message] + state.messages_for_llm

    def _build_final_instructions(self, state: RuntimeState) -> Optional[str]:
        parts: List[str] = []
        source_request = False
        source_user_profile = False
        source_org_profile = False

        # 1) User-provided instructions (per-request, ChatGPT/Gemini-style)
        if isinstance(state.request.instructions, str):
            user_instr = state.request.instructions.strip()
            if user_instr:
                parts.append(user_instr)
                source_request = True

        # 2) User profile instructions prepared by the memory layer
        if isinstance(state.profile_user_instructions, str):
            profile_user = state.profile_user_instructions.strip()
            if profile_user:
                parts.append(profile_user)
                source_user_profile = True

        # 3) Organization profile instructions prepared by the memory layer
        if isinstance(state.profile_org_instructions, str):
            profile_org = state.profile_org_instructions.strip()
            if profile_org:
                parts.append(profile_org)
                source_org_profile = True

        if not parts:
            state.trace_event(
                component="engine",
                step="instructions",
                message="Instructions resolved (none).",
                level=TraceLevel.INFO,
                payload=InstructionsSummaryDiagV1(
                    has_instructions=False,
                    source_request=source_request,
                    source_user_profile=source_user_profile,
                    source_org_profile=source_org_profile,
                ),
            )
            return None

        final_text = "\n\n".join(parts)

        state.trace_event(
            component="engine",
            step="instructions",
            message="Instructions resolved.",
            level=TraceLevel.INFO,
            payload=InstructionsSummaryDiagV1(
                has_instructions=True,
                source_request=source_request,
                source_user_profile=source_user_profile,
                source_org_profile=source_org_profile,
            ),
        )

        return final_text
