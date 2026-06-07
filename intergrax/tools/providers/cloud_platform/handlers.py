# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.cloud_platform.contracts import (
    CloudPlatformHealthInput,
    CloudPlatformHealthOutput,
    CloudPlatformResolveInput,
    CloudPlatformResolveOutput,
)
from intergrax.tools.providers.cloud_platform.service import cloud_platform_health, cloud_platform_resolve


class CloudPlatformHealthHandler(ServiceToolHandler[CloudPlatformHealthInput, CloudPlatformHealthOutput]):
    _service = cloud_platform_health


class CloudPlatformResolveHandler(ServiceToolHandler[CloudPlatformResolveInput, CloudPlatformResolveOutput]):
    _service = cloud_platform_resolve
