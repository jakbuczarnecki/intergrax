# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod


class MessageProducer(ABC):
    """
    Transport-level message producer abstraction.

    This contract represents a broker transport capability only.
    It does NOT define:
    - status backend
    - retry semantics
    - execution model

    Implementations may wrap:
    - Kafka producer
    - RabbitMQ publisher
    - any other broker transport
    """

    @abstractmethod
    def publish(
        self,
        topic: str,
        payload: bytes,
    ) -> None:
        """
        Publish raw bytes payload to a given topic/queue.

        Must be crash-safe according to backend guarantees.
        """
        ...