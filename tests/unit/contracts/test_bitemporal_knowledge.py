# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.bitemporal_knowledge import (
    BitemporalKnowledgeBasis,
    CrossScopeKnowledgeOrderError,
    KnowledgeOrderingScope,
    KnowledgeRevisionPosition,
    KnowledgeRevisionWatermark,
    KnowledgeRevisionWatermarkSet,
    RevisionAcceptanceKey,
    SystemTimeBasis,
    ValidTimeBasis,
    mint_revision_acceptance_key,
    validate_revision_acceptance_key,
)
from intergrax.runtime.events.execution_position import ExecutionEventPosition

_UTC = timezone.utc
_T0 = datetime(2026, 1, 15, 12, 0, tzinfo=_UTC)
_T1 = datetime(2026, 2, 1, 0, 0, tzinfo=_UTC)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_time_instant_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ValidTimeBasis.instant(datetime(2026, 1, 15, 12, 0))


@pytest.mark.unit
@pytest.mark.gate
def test_valid_time_interval_open_ended_and_impossible_range() -> None:
    open_ended = ValidTimeBasis.interval(_T0, None)
    assert open_ended.is_open_ended is True
    bounded = ValidTimeBasis.interval(_T0, _T1)
    assert bounded.is_open_ended is False
    with pytest.raises(ValueError, match="strictly after start"):
        ValidTimeBasis.interval(_T1, _T0)
    with pytest.raises(ValueError, match="strictly after start"):
        ValidTimeBasis.interval(_T0, _T0)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_time_instant_rejects_end_and_allows_future_effective() -> None:
    future = datetime(2027, 1, 1, tzinfo=_UTC)
    point = ValidTimeBasis.instant(future)
    assert point.end is None
    with pytest.raises(ValueError, match="must not set end"):
        ValidTimeBasis(kind=point.kind, start=future, end=future + timedelta(days=1))


@pytest.mark.unit
@pytest.mark.gate
def test_system_time_rejects_naive_and_is_not_ordering_type() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        SystemTimeBasis.instant(datetime(2026, 8, 17, 7, 0))
    recorded = SystemTimeBasis.instant(_T0)
    belief = SystemTimeBasis.interval(_T0, None)
    assert recorded.start == _T0
    assert belief.is_open_ended is True
    assert type(recorded) is not ValidTimeBasis


@pytest.mark.unit
@pytest.mark.gate
def test_bitemporal_basis_contains_only_temporal_axes() -> None:
    basis = BitemporalKnowledgeBasis(
        valid_time=ValidTimeBasis.interval(_T0, None),
        system_time=SystemTimeBasis.instant(_T1),
    )
    assert set(basis.__dataclass_fields__) == {"valid_time", "system_time"}
    assert type(basis.valid_time) is ValidTimeBasis
    assert type(basis.system_time) is SystemTimeBasis


@pytest.mark.unit
@pytest.mark.gate
def test_revision_acceptance_key_is_nominal_and_not_event_id() -> None:
    key = mint_revision_acceptance_key()
    assert key.value.startswith("rack_")
    assert validate_revision_acceptance_key(key) == key.value
    with pytest.raises(ValueError, match="must start with"):
        RevisionAcceptanceKey("evt_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(TypeError):
        validate_revision_acceptance_key(123)


@pytest.mark.unit
@pytest.mark.gate
def test_scope_and_position_reject_cross_tenant_ordering() -> None:
    tenant_a = KnowledgeOrderingScope(tenant_id="tenant-a")
    tenant_b = KnowledgeOrderingScope(tenant_id="tenant-b")
    k12_a = KnowledgeRevisionPosition(scope=tenant_a, value=12)
    k20_b = KnowledgeRevisionPosition(scope=tenant_b, value=20)
    assert k12_a != k20_b
    with pytest.raises(CrossScopeKnowledgeOrderError):
        k12_a.precedes(k20_b)
    with pytest.raises(ValueError, match="positive int"):
        KnowledgeRevisionPosition(scope=tenant_a, value=0)
    assert isinstance(ExecutionEventPosition(12), ExecutionEventPosition)
    assert type(k12_a) is not type(ExecutionEventPosition(12))


@pytest.mark.unit
@pytest.mark.gate
def test_watermark_set_rejects_duplicate_scope_and_cross_scope_includes() -> None:
    tenant_a = KnowledgeOrderingScope(tenant_id="tenant-a")
    tenant_b = KnowledgeOrderingScope(tenant_id="tenant-b")
    wm_a = KnowledgeRevisionWatermark(scope=tenant_a, finalized_through_value=12)
    wm_b = KnowledgeRevisionWatermark(scope=tenant_b, finalized_through_value=20)
    grouped = KnowledgeRevisionWatermarkSet(watermarks=(wm_a, wm_b))
    assert len(grouped.watermarks) == 2
    with pytest.raises(ValueError, match="duplicate scopes"):
        KnowledgeRevisionWatermarkSet(watermarks=(wm_a, wm_a))
    other = KnowledgeRevisionPosition(scope=tenant_b, value=12)
    with pytest.raises(CrossScopeKnowledgeOrderError):
        wm_a.includes(other)
    assert wm_a.includes(KnowledgeRevisionPosition(scope=tenant_a, value=12)) is True
    assert wm_a.includes(KnowledgeRevisionPosition(scope=tenant_a, value=13)) is False
