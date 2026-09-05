# © Artur Czarnecki. All rights reserved.

"""P0C-1 — identity continuity across process restart."""

from __future__ import annotations

import pytest

from tests.conformance.runtime.durability._helpers import admit, transport_ref
from tests.conformance.runtime.durability.provider_factories import (
    DurableAdmissionBacking,
    DurableProviderKind,
    create_admission_dependencies,
)
from tests.conformance.runtime.durability.restart import fresh_admission_composition

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_redelivery_preserves_identity_after_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    transport = transport_ref(tenant_id=tenant_id, task_id="identity-continuity")
    first = admit(transport=transport, deps=process_a)

    process_b = fresh_admission_composition(admission_backing)
    second = admit(transport=transport, deps=process_b)

    assert second.identity.task_id == first.identity.task_id
    assert second.identity.run_id == first.identity.run_id
    assert second.identity.attempt_id == first.identity.attempt_id


def test_provider_namespace_isolation_after_restart(
    admission_backing: DurableAdmissionBacking,
    tenant_id: str,
) -> None:
    process_a = create_admission_dependencies(admission_backing)
    kafka = transport_ref(tenant_id=tenant_id, task_id="shared-transport", provider="kafka")
    rabbit = transport_ref(tenant_id=tenant_id, task_id="shared-transport", provider="rabbitmq")
    kafka_identity = admit(transport=kafka, deps=process_a).identity

    process_b = fresh_admission_composition(admission_backing)
    rabbit_identity = admit(transport=rabbit, deps=process_b).identity
    kafka_redelivery = admit(transport=kafka, deps=process_b).identity

    assert rabbit_identity.task_id != kafka_identity.task_id
    assert rabbit_identity.run_id != kafka_identity.run_id
    assert kafka_redelivery.task_id == kafka_identity.task_id
    assert kafka_redelivery.run_id == kafka_identity.run_id


@pytest.mark.parametrize(
    "kind",
    list(DurableProviderKind),
    ids=[kind.value for kind in DurableProviderKind],
)
def test_identity_store_is_durable(kind: DurableProviderKind) -> None:
    backing = (
        DurableAdmissionBacking.fresh_kv()
        if kind is DurableProviderKind.KV
        else DurableAdmissionBacking.fresh_document_store()
    )
    deps = create_admission_dependencies(backing)
    assert deps.identity_persistence is not None
