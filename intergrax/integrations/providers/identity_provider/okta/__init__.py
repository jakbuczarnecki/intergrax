# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.identity_provider.okta.bundle import create_okta_identity_provider
from intergrax.integrations.providers.identity_provider.okta.register import register_okta_integration

__all__ = ["create_okta_identity_provider", "register_okta_integration"]
