# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.issue_tracker.zendesk.bundle import create_zendesk_issue_tracker
from intergrax.integrations.providers.issue_tracker.zendesk.register import register_zendesk_integration

__all__ = ["create_zendesk_issue_tracker", "register_zendesk_integration"]
