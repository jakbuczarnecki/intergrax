# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Type

from pydantic import BaseModel

from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode


@dataclass(frozen=True, slots=True)
class ToolContract:
    """
    Formal runtime contract for a tool/skill.

    This is enforced by Nexus runtime (registry + validation + trace + error mapping).
    """
    tool_id: str
    name: str
    description: str

    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]

    error_mapping: Mapping[type[Exception], RuntimeErrorCode]

    side_effects: bool
