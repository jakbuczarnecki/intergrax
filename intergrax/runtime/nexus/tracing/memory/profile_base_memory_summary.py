# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class ProfileBasedMemorySummaryDiagV1(DiagnosticPayload):
    has_user_profile_instructions: bool
    has_org_profile_instructions: bool
    enable_user_profile_memory: bool
    enable_org_profile_memory: bool

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.memory_layer.profile_based.summary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_user_profile_instructions": self.has_user_profile_instructions,
            "has_org_profile_instructions": self.has_org_profile_instructions,
            "enable_user_profile_memory": self.enable_user_profile_memory,
            "enable_org_profile_memory": self.enable_org_profile_memory,
        }