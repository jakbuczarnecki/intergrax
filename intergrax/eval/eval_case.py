# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass
from typing import Optional, Tuple

from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@dataclass(frozen=True)
class EvalCase:
    """
    Represents a single evaluation scenario.

    runtime_request:
        Fully constructed RuntimeRequest used to execute the agent.
        Eval layer does not build requests — it executes them.

    expected_output:
        Deterministic exact match expected result (P0).
    """

    case_id: str
    runtime_request: RuntimeRequest
    expected_output: str
    description: Optional[str] = None
    tags: Optional[Tuple[str, ...]] = None
    semantic_match_enabled: bool = False
    rubric_ref: Optional[str] = None
    semantic_threshold: Optional[float] = None