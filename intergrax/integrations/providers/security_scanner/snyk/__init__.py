# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.security_scanner.snyk.bundle import create_snyk_security_scanner
from intergrax.integrations.providers.security_scanner.snyk.register import register_snyk_integration

__all__ = ["create_snyk_security_scanner", "register_snyk_integration"]
