# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.tracing.session.session_consolidation_diag import SessionConsolidationDiagV1

@dataclass(frozen=True)
class SessionMessageAppendResult:
    message: ChatMessage
    consolidation_diag: Optional[SessionConsolidationDiagV1]
