# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.observability_backend.splunk.bundle import create_splunk_observability_backend
from intergrax.integrations.providers.observability_backend.splunk.register import register_splunk_integration

__all__ = ["create_splunk_observability_backend", "register_splunk_integration"]
