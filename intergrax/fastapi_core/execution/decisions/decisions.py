# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from enum import Enum


class ExecutionDecision(str, Enum):
    RETRY = "retry"
    FAIL = "fail"
    IGNORE = "ignore"
