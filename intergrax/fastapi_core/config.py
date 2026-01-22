# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Optional

from intergrax.fastapi_core.auth.api_key import ApiKeyConfig


class ApiEnvironment(str, Enum):
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"


@dataclass(frozen=True)
class ApiConfig:
    """
    Typed configuration for FastAPI services built on Intergrax FastAPI Core.

    Design goals:
    - Fail-fast, explicit defaults
    - No implicit globals (config must be passed to create_app)
    - Production-oriented toggles (expanded incrementally in follow-up steps)
    """

    environment: ApiEnvironment = ApiEnvironment.DEV

    # Versioning / routing
    api_prefix: str = "/v1"

    # CORS (kept as an explicit allow-list; empty means "disabled")
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)

    # Host allow-list (optional; empty means "no restriction at this layer")
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)

    # Observability toggles (actual logging middleware comes in follow-up step)
    enable_structured_logging: bool = True

    api_key_config: Optional[ApiKeyConfig] = None


    def validate(self) -> None:
        """
        Validate config invariants. Must be called by create_app() (fail-fast).
        """
        if not self.api_prefix.startswith("/"):
            raise ValueError("ApiConfig.api_prefix must start with '/'.")
        if self.api_prefix != "/" and self.api_prefix.endswith("/"):
            raise ValueError("ApiConfig.api_prefix must not end with '/' (except '/').")

        # Keep a conservative invariant: "/v1" style prefixes only
        if self.api_prefix == "":
            raise ValueError("ApiConfig.api_prefix must not be empty.")

        # Security hardening: in prod we expect explicit host policy in real deployments,
        # but do not enforce it yet (no breaking constraints in MVP-1).
        _ = self.allowed_hosts
