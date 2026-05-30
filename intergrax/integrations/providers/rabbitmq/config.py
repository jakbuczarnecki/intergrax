# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RabbitMQ integration configuration (Phase M.4)."""

from __future__ import annotations

import os
from typing import Optional

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_RABBITMQ_HOST = "INTERGRAX_RABBITMQ_HOST"
ENV_RABBITMQ_PORT = "INTERGRAX_RABBITMQ_PORT"
ENV_RABBITMQ_VIRTUAL_HOST = "INTERGRAX_RABBITMQ_VIRTUAL_HOST"
ENV_RABBITMQ_USERNAME = "INTERGRAX_RABBITMQ_USERNAME"
ENV_RABBITMQ_PASSWORD = "INTERGRAX_RABBITMQ_PASSWORD"
ENV_RABBITMQ_QUEUE = "INTERGRAX_RABBITMQ_QUEUE"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5672
DEFAULT_VIRTUAL_HOST = "/"
DEFAULT_QUEUE = "intergrax-tasks"


class RabbitMQIntegrationConfig(BaseIntegrationConfig):
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    virtual_host: str = DEFAULT_VIRTUAL_HOST
    username: Optional[str] = None
    password: Optional[str] = None
    queue: str = DEFAULT_QUEUE

    @classmethod
    def from_env(cls, **overrides: object) -> RabbitMQIntegrationConfig:
        host = os.environ.get(ENV_RABBITMQ_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        port_raw = os.environ.get(ENV_RABBITMQ_PORT, str(DEFAULT_PORT)).strip()
        virtual_host = (
            os.environ.get(ENV_RABBITMQ_VIRTUAL_HOST, DEFAULT_VIRTUAL_HOST).strip()
            or DEFAULT_VIRTUAL_HOST
        )
        username = os.environ.get(ENV_RABBITMQ_USERNAME)
        password = os.environ.get(ENV_RABBITMQ_PASSWORD)
        queue = os.environ.get(ENV_RABBITMQ_QUEUE, DEFAULT_QUEUE).strip() or DEFAULT_QUEUE

        payload: dict[str, object] = {
            "host": host,
            "port": int(port_raw) if port_raw else DEFAULT_PORT,
            "virtual_host": virtual_host,
            "queue": queue,
        }
        if username is not None and username.strip():
            payload["username"] = username.strip()
        if password is not None and password.strip():
            payload["password"] = password.strip()
        payload.update(overrides)
        return cls.model_validate(payload)
