# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Outbound notification contract — re-exports runtime adapter (§7.1.2, Phase M.2)."""

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter

NotificationChannel = NotificationAdapter

__all__ = ["NotificationAdapter", "NotificationChannel"]
