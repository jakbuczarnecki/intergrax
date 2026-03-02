# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


TransportBackend = Literal["kafka", "rabbitmq"]


@dataclass(frozen=True)
class KafkaTransportConfig:
    bootstrap_servers: str


@dataclass(frozen=True)
class RabbitMQTransportConfig:
    host: str
    port: int = 5672
    virtual_host: str = "/"
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass(frozen=True)
class TransportConfig:
    backend: TransportBackend
    kafka: Optional[KafkaTransportConfig] = None
    rabbitmq: Optional[RabbitMQTransportConfig] = None