# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Log notification integration — wraps ``LoggingNotificationAdapter``."""

from __future__ import annotations

from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter


class _LogNotificationAdapter(LoggingNotificationAdapter):
    """Catalog facade for laboratory / dev outbound notifications (no network)."""
