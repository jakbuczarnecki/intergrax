# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.crm.salesforce.bundle import create_salesforce_crm
from intergrax.integrations.providers.crm.salesforce.register import register_salesforce_integration

__all__ = ["create_salesforce_crm", "register_salesforce_integration"]
