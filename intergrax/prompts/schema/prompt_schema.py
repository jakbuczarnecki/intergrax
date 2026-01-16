# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional


@dataclass(frozen=True)
class PromptMeta:
    """
    Metadata required for safe and reproducible usage.
    """
    model_family: str
    output_schema_id: str
    tags: FrozenSet[str]
    description: Optional[str]



@dataclass(frozen=True)
class LocalizedContent:
    """
    Single language variant of a prompt.
    """
    system: str
    developer: Optional[str]
    user_template: Optional[str]


@dataclass(frozen=True)
class LocalizedPromptDocument:
    """
    Prompt document containing multiple locales.
    """
    id: str
    version: int
    locales: Dict[str, LocalizedContent]
    meta: PromptMeta
