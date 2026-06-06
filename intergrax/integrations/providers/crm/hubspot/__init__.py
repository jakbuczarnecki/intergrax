# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.crm.hubspot.bundle import create_hubspot_crm
from intergrax.integrations.providers.crm.hubspot.register import register_hubspot_integration

__all__ = ["create_hubspot_crm", "register_hubspot_integration"]
