# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"


