# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner
from intergrax.integrations.providers.security_scanner.trivy.register import register_trivy_integration

__all__ = ["create_trivy_security_scanner", "register_trivy_integration"]
