# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for P0C-8 durability conformance."""

from __future__ import annotations

import pytest

from tests.conformance.runtime.durability.provider_factories import (
    BACKGROUND_IDENTITY_PROVIDERS,
    DurableAdmissionBacking,
    DurableProviderKind,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-p0c8"


@pytest.fixture
def tenant_id() -> str:
    return _TENANT


@pytest.fixture(params=list(BACKGROUND_IDENTITY_PROVIDERS), ids=lambda kind: kind.value)
def admission_backing(request: pytest.FixtureRequest) -> DurableAdmissionBacking:
    kind: DurableProviderKind = request.param
    if kind is DurableProviderKind.KV:
        return DurableAdmissionBacking.fresh_kv()
    return DurableAdmissionBacking.fresh_document_store()
