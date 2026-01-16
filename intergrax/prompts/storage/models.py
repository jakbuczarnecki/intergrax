# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from intergrax.prompts.schema.prompt_schema import PromptDocument


@dataclass(frozen=True)
class LoadedPrompt:
    """
    Prompt loaded from storage with calculated hash.
    """
    document: PromptDocument
    content_hash: str


class PromptStorageError(Exception):
    pass


class PromptParseError(PromptStorageError):
    pass


class PromptValidationError(PromptStorageError):
    pass


class PromptNotFound(PromptStorageError):
    pass
