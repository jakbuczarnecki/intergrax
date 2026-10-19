# © Artur Czarnecki. All rights reserved.

"""Sanctioned reflex metadata — canonical MoneyAmount budget coercion."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from external_contractor_adapter.external_work_adapter import (
    META_BUDGET_LIMIT,
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DIGEST,
    _coerce_meta_budget_limit,
)
from external_contractor_adapter.sanctioned_operation_from_metadata import (
    resolve_sanctioned_operation_from_metadata,
)
from external_contractor_adapter.tests.fakes.adapter_test_wiring import (
    EXTERNAL_WORK_TEST_RUN_ID as _RUN,
    EXTERNAL_WORK_TEST_TASK_ID as _TASK,
    allow_adapter,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.money import MoneyAmount

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CANONICAL_JSON = '{"amount":"25.00","currency":"USD"}'


def _base_meta(**overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        "external_work.scope_description": "scope",
        META_SCOPE_DIGEST: "sha256:" + ("aa" * 32),
        META_IDEMPOTENCY_KEY: "idem-budget",
        "external_work.principal_id": "principal-a",
        "external_work.tenant_id": "tenant-a",
    }
    meta.update(overrides)
    return meta


@pytest.mark.parametrize(
    "budget_raw",
    [
        MoneyAmount(amount=Decimal("25.00"), currency="USD"),
        {"amount": "25.00", "currency": "USD"},
        _CANONICAL_JSON,
    ],
)
def test_sanctioned_create_budget_round_trips_to_adapter_create_request(
    budget_raw: object,
) -> None:
    meta = _base_meta(**{META_BUDGET_LIMIT: budget_raw})
    operation = resolve_sanctioned_operation_from_metadata(
        meta,
        task_id=str(_TASK),
        run_id=str(_RUN),
        message="budget qualification",
    )
    create_payload = operation.create
    assert create_payload is not None
    adapter, _ = allow_adapter(DeterministicExternalWorkFake())
    adapter_metadata = create_payload.adapter_metadata.to_adapter_metadata()
    request = adapter.build_create_request(
        task_id=str(_TASK),
        run_id=str(_RUN),
        metadata=adapter_metadata,
    )
    assert request.budget_limit == MoneyAmount(amount=Decimal("25.00"), currency="USD")


def test_absent_budget_remains_optional() -> None:
    meta = _base_meta()
    assert META_BUDGET_LIMIT not in meta
    operation = resolve_sanctioned_operation_from_metadata(
        meta,
        task_id=str(_TASK),
        run_id=str(_RUN),
        message="budget optional",
    )
    create_payload = operation.create
    assert create_payload is not None
    assert create_payload.adapter_metadata.budget_limit is None
    fake = DeterministicExternalWorkFake()
    adapter, _ = allow_adapter(fake)
    request = adapter.build_create_request(
        task_id=str(_TASK),
        run_id=str(_RUN),
        metadata=create_payload.adapter_metadata.to_adapter_metadata(),
    )
    assert request.budget_limit is None
    assert fake.create_calls == 0


def test_coerce_meta_budget_limit_none_is_absent() -> None:
    assert _coerce_meta_budget_limit(None) is None


@pytest.mark.parametrize(
    "budget_raw",
    [
        "",
        "   ",
        "{not-json",
        {"amount": "-1.00", "currency": "USD"},
        {"amount": "25.00", "currency": "US"},
        42,
    ],
)
def test_invalid_budget_fail_closed_before_provider(budget_raw: object) -> None:
    meta = _base_meta(**{META_BUDGET_LIMIT: budget_raw})
    fake = DeterministicExternalWorkFake()
    with pytest.raises((ValueError, ValidationError)):
        resolve_sanctioned_operation_from_metadata(
            meta,
            task_id=str(_TASK),
            run_id=str(_RUN),
            message="budget reject",
        )
    assert fake.create_calls == 0


def test_invalid_budget_rejected_at_adapter_build_create_request() -> None:
    meta = _base_meta()
    operation = resolve_sanctioned_operation_from_metadata(
        meta,
        task_id=str(_TASK),
        run_id=str(_RUN),
        message="budget adapter gate",
    )
    create_payload = operation.create
    assert create_payload is not None
    fake = DeterministicExternalWorkFake()
    adapter, _ = allow_adapter(fake)
    adapter_metadata = create_payload.adapter_metadata.to_adapter_metadata()
    adapter_metadata[META_BUDGET_LIMIT] = ""
    with pytest.raises(ValueError, match="empty or whitespace"):
        adapter.build_create_request(
            task_id=str(_TASK),
            run_id=str(_RUN),
            metadata=adapter_metadata,
        )
    assert fake.create_calls == 0
