# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ExecutionRequest:
    run_id: str
    tenant_id: str
    user_id: Optional[str]
    input_payload: Dict[str, Any]

    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
