# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from enum import Enum


class FailureCategory(str, Enum):
    RETRYABLE = "retryable"
    TERMINAL = "terminal"
    TIMEOUT = "timeout"
    CANCELED = "canceled"
    UNKNOWN = "unknown"
