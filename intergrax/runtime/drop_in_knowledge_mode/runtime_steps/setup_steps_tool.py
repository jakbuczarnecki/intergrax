# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.build_base_history_step import BuildBaseHistoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.history_step import HistoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.instructions_step import InstructionsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.profile_based_memory_step import ProfileBasedMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.session_and_ingest_step import SessionAndIngestStep


SETUP_STEPS = [
            SessionAndIngestStep(),
            ProfileBasedMemoryStep(),
            BuildBaseHistoryStep(),
            HistoryStep(),
            InstructionsStep(),
            EnsureCurrentUserMessageStep(),
        ]