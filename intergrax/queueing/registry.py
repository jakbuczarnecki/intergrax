# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Dict, Type

from intergrax.queueing.contracts.task_queue import TaskQueue


class TaskQueueProviderRegistry:
    """
    Registry responsible for registering and resolving task queue backends.

    This class does not instantiate runtime components directly.
    It only maps backend identifiers to concrete TaskQueue implementations.

    The registry is open for extension: any backend that conforms to
    TaskQueue can be registered without modifying core code.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, Type[TaskQueue]] = {}

    def register(
        self,
        backend_id: str,
        provider_cls: Type[TaskQueue],
    ) -> None:
        if backend_id in self._providers:
            raise ValueError(
                f"Backend '{backend_id}' is already registered."
            )

        self._providers[backend_id] = provider_cls

    def get_provider(
        self,
        backend_id: str,
    ) -> Type[TaskQueue]:
        if backend_id not in self._providers:
            raise ValueError(
                f"Backend '{backend_id}' is not registered."
            )

        return self._providers[backend_id]