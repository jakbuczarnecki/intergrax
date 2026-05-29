# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inbound interaction contract — re-exports runtime adapter (§7.1.2, Phase M.2)."""

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter

InteractionSurface = InteractionAdapter

__all__ = ["InteractionAdapter", "InteractionSurface"]
