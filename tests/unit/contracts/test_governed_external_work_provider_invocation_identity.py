# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-IMPL-01 — stable CREATE invocation identity contract."""

from __future__ import annotations

import pytest

from intergrax.contracts.governed_external_work_provider_invocation_identity import (
    derive_governed_external_work_create_provider_invocation_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

# NOT PROVEN — pytest=0 until operator authorization:
# uv run pytest -p no:xdist tests/unit/contracts/test_governed_external_work_provider_invocation_identity.py -q


def test_derive_create_invocation_id_is_deterministic() -> None:
    kwargs = {
        "tenant_id": "tenant-a",
        "principal_id": "principal-a",
        "execution_id": "exec_" + "a" * 32,
        "provider_id": "gec3_deterministic_fake",
        "idempotency_key": "idem-stable-01",
    }
    first = derive_governed_external_work_create_provider_invocation_id(**kwargs)
    second = derive_governed_external_work_create_provider_invocation_id(**kwargs)
    assert first == second
    assert first.startswith("gca-pinv:")


def test_derive_create_invocation_id_tenant_and_principal_isolation() -> None:
    base = {
        "execution_id": "exec_" + "b" * 32,
        "provider_id": "prov",
        "idempotency_key": "idem-1",
    }
    a = derive_governed_external_work_create_provider_invocation_id(
        tenant_id="t1",
        principal_id="p1",
        **base,
    )
    b = derive_governed_external_work_create_provider_invocation_id(
        tenant_id="t2",
        principal_id="p1",
        **base,
    )
    c = derive_governed_external_work_create_provider_invocation_id(
        tenant_id="t1",
        principal_id="p2",
        **base,
    )
    assert a != b
    assert a != c


def test_derive_create_invocation_id_requires_idempotency_key() -> None:
    with pytest.raises(ValueError, match="idempotency_key_required"):
        derive_governed_external_work_create_provider_invocation_id(
            tenant_id="t",
            principal_id="p",
            execution_id="exec_" + "c" * 32,
            provider_id="prov",
            idempotency_key="   ",
        )
