# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from enum import Enum


class RuntimeErrorCode(str, Enum):
    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
