# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCapabilities:
    supports_retry: bool
    supports_timeout: bool
    supports_cancel: bool
