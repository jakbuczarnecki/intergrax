# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.security.contracts import SecurityScanInput, SecurityScanOutput
from intergrax.tools.providers.security.service import security_scan


class SecurityScanHandler(ServiceToolHandler[SecurityScanInput, SecurityScanOutput]):
    _service = security_scan
