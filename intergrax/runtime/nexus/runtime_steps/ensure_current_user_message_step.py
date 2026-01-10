# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep


class EnsureCurrentUserMessageStep(RuntimeStep):
    """
    Ensure the current user message is present as the last prompt message.

    Rules:
      - If request.message is empty -> no-op.
      - If messages_for_llm is empty -> add user message.
      - If last message is the same user message -> no-op.
      - Otherwise append user message to enforce user-last semantics.
    """

    async def run(self, state: RuntimeState) -> None:
        msg = (state.request.message or "").strip()
        if not msg:
            return

        if not state.messages_for_llm:
            state.messages_for_llm.append(ChatMessage(role="user", content=msg))
            return

        last = state.messages_for_llm[-1]
        last_content = (last.content or "").strip()

        # If the last message already equals the current user prompt, do nothing.
        if last.role == "user" and last_content == msg:
            return

        # Otherwise append current user prompt to enforce user-last semantics.
        state.messages_for_llm.append(ChatMessage(role="user", content=msg))
