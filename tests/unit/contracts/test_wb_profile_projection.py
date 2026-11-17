# © Artur Czarnecki. All rights reserved.

"""W-B Part B — declared profile projection refuses unmapped, stale and lenient shapes (NT-WB-08..10)."""

from __future__ import annotations

import pytest
from intergrax.contracts.profile_projection import (
    ProfileProjection,
    ProjectionTargetIgnoresExtrasError,
    StaleProjectionEntryError,
    UnmappedProfileFieldError,
)
from pydantic import BaseModel, ConfigDict

pytestmark = [pytest.mark.unit]


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: int = 1
    beta: str = "b"


class _Target(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: int = 1
    beta: str = "b"


class _LenientTarget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alpha: int = 1
    beta: str = "b"


def test_nt_wb_08_unmapped_source_field_refused() -> None:
    class _WiderSource(_Source):
        gamma: bool = False

    projection = ProfileProjection(name="synthetic", target=_Target, projected=("alpha", "beta"))
    with pytest.raises(UnmappedProfileFieldError) as excinfo:
        projection.verify(_WiderSource)
    assert "gamma" in str(excinfo.value)


def test_nt_wb_09_ignore_target_refused() -> None:
    projection = ProfileProjection(
        name="synthetic",
        target=_LenientTarget,
        projected=("alpha", "beta"),
    )
    with pytest.raises(ProjectionTargetIgnoresExtrasError):
        projection.verify(_Source)


def test_nt_wb_10_stale_map_entry_refused() -> None:
    projection = ProfileProjection(
        name="synthetic",
        target=_Target,
        projected=("alpha", "beta"),
        not_projected={"removed_field": "was consumed elsewhere"},
    )
    with pytest.raises(StaleProjectionEntryError) as excinfo:
        projection.verify(_Source)
    assert "removed_field" in str(excinfo.value)


def test_verified_projection_carries_only_declared_fields() -> None:
    class _WiderSource(_Source):
        gamma: bool = True

    projection = ProfileProjection(
        name="synthetic",
        target=_Target,
        projected=("alpha", "beta"),
        not_projected={"gamma": "read from the full model elsewhere"},
    )
    projection.verify(_WiderSource)
    projected = projection.project(_WiderSource(alpha=3, beta="x", gamma=False))
    assert projected == _Target(alpha=3, beta="x")
