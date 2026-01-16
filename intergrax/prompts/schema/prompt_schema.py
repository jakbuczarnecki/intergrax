# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class PromptContent:
    """
    Language model message templates.

    Separation reflects OpenAI/Anthropic message model.
    """
    system: str
    developer: Optional[str]
    user_template: Optional[str]


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
class PromptDocument:
    """
    Full prompt definition loaded from YAML.
    This is a product artifact, not code.
    """
    id: str
    version: int
    content: PromptContent
    meta: PromptMeta
