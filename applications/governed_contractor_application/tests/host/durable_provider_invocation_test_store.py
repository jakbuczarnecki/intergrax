# © Artur Czarnecki. All rights reserved.

"""Injectable durable ProviderInvocationStore double for production composition tests."""

from __future__ import annotations

from governed_contractor_application.host.stores import InMemoryProviderInvocationStore


class DurableTestProviderInvocationStore(InMemoryProviderInvocationStore):
    """Same behavior as in-memory fixture; declares durability for production gate tests."""

    @property
    def is_durable(self) -> bool:
        return True
