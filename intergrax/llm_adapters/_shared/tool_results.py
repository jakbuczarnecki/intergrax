# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List


def make_tool_result(
    *,
    content: str = "",
    tool_calls: List[Dict[str, Any]] | None = None,
    finish_reason: str = "completed",
) -> Dict[str, Any]:
    """Standard native-tools response shape consumed by ToolsAgent."""
    return {
        "content": content or "",
        "tool_calls": list(tool_calls or []),
        "finish_reason": finish_reason,
    }
