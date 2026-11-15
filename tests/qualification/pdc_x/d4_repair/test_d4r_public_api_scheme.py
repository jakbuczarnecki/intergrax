# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-07: the public API fingerprint sees model config, constraints and validators."""

from __future__ import annotations

import pytest
from intergrax.hosting import HostedApplicationProfile
from intergrax.hosting.public_api import (
    HOSTING_PUBLIC_API_FINGERPRINTS,
    HOSTING_PUBLIC_API_VERSION,
    HOSTING_PUBLIC_API_VERSION_HISTORY,
    compute_public_api_fingerprints,
    fingerprint_public_symbol,
    verify_public_api,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

pytestmark = [pytest.mark.unit]

# Recorded for version 1.0.0 before the fingerprint scheme was extended; history is never edited.
_RECORDED_1_0_0 = "sha256:88814bef45fe1494f1a2a5d22165c932063eba8a77df96db83bf59efb9134907"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limit: int = Field(default=1, ge=1, le=5)


class _LooseExtra(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    limit: int = Field(default=1, ge=1, le=5)


class _WideBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limit: int = Field(default=1, ge=-100, le=500)


class _Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    limit: int = Field(default=1, ge=1, le=5)


class _Validated(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limit: int = Field(default=1, ge=1, le=5)

    @field_validator("limit")
    @classmethod
    def _limit_is_even(cls, value: int) -> int:
        return value


class _StrictProfile(HostedApplicationProfile):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _LooseProfile(HostedApplicationProfile):
    model_config = ConfigDict(extra="allow", frozen=True)


@pytest.mark.parametrize("loosened", [_LooseExtra, _WideBounds, _Mutable, _Validated])
def test_a_contract_change_with_the_same_annotations_changes_the_fingerprint(loosened: type[BaseModel]) -> None:
    assert fingerprint_public_symbol(loosened) != fingerprint_public_symbol(_Strict)


def test_loosening_the_extra_policy_of_a_real_hosting_model_changes_its_fingerprint() -> None:
    assert fingerprint_public_symbol(_LooseProfile) != fingerprint_public_symbol(_StrictProfile)


def test_the_fingerprint_of_an_unchanged_symbol_is_deterministic() -> None:
    assert fingerprint_public_symbol(_Strict) == fingerprint_public_symbol(_Strict)
    assert compute_public_api_fingerprints() == compute_public_api_fingerprints()


def test_history_keeps_the_recorded_1_0_0_entry_and_adds_1_0_1() -> None:
    assert HOSTING_PUBLIC_API_VERSION_HISTORY["1.0.0"] == _RECORDED_1_0_0
    assert HOSTING_PUBLIC_API_VERSION == "1.0.1"
    assert list(HOSTING_PUBLIC_API_VERSION_HISTORY) == ["1.0.0", "1.0.1"]
    assert (
        verify_public_api(
            compute_public_api_fingerprints(),
            HOSTING_PUBLIC_API_FINGERPRINTS,
            HOSTING_PUBLIC_API_VERSION_HISTORY,
            HOSTING_PUBLIC_API_VERSION,
        )
        == []
    )
