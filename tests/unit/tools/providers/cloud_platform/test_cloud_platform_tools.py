# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional

import pytest

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.tools.providers.cloud_platform.contracts import CloudPlatformHealthInput, CloudPlatformResolveInput
from intergrax.tools.providers.cloud_platform.service import cloud_platform_health, cloud_platform_resolve
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeCloudPlatform:
    @property
    def slug(self) -> str:
        return "aws"

    @property
    def default_region(self) -> Optional[str]:
        return "eu-central-1"

    def resolve(self, category: str) -> Optional[str]:
        return "s3" if category == "object_storage" else None

    def health(self) -> HealthStatus:
        return HealthStatus(slug=self.slug, healthy=True, detail="ok")


def test_cloud_platform_health() -> None:
    ctx = ToolWiringContext(cloud_platform=FakeCloudPlatform())
    out = cloud_platform_health(ctx, CloudPlatformHealthInput())
    assert out.used is True
    assert out.slug == "aws"
    assert out.healthy is True
    assert out.default_region == "eu-central-1"


def test_cloud_platform_resolve() -> None:
    ctx = ToolWiringContext(cloud_platform=FakeCloudPlatform())
    out = cloud_platform_resolve(ctx, CloudPlatformResolveInput(category="object_storage"))
    assert out.used is True
    assert out.resolved_slug == "s3"
    assert out.reason == "ok"


def test_cloud_platform_not_configured() -> None:
    with pytest.raises(RuntimeError, match="cloud_platform_not_configured"):
        cloud_platform_health(ToolWiringContext(), CloudPlatformHealthInput())
