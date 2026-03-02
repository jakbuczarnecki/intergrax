# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class MessageConsumer(ABC):
    """
    Transport-level message consumer abstraction.

    This contract represents broker consumption capability only.
    It does NOT define:
    - execution model
    - retry semantics
    - status backend
    - acknowledgement strategy (backend dependent)

    Implementations may wrap:
    - Kafka consumer
    - RabbitMQ consumer
    - any other broker transport
    """

    @abstractmethod
    def poll(self, *, timeout_seconds: float) -> Optional[bytes]:
        """
        Poll broker for a single message.

        Args:
            timeout_seconds: Maximum time to wait for a message.

        Returns:
            Raw payload bytes if message is available,
            None if no message was received within timeout.
        """
        raise NotImplementedError