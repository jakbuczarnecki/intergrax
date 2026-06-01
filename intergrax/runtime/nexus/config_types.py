# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from enum import Enum
from typing import Literal

ToolChoiceMode = Literal["off", "auto", "required"]


class ToolsContextScope(str, Enum):
    CURRENT_MESSAGE_ONLY = "current_message_only"
    CONVERSATION = "conversation"
    FULL = "full"
