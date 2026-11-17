# © Artur Czarnecki. All rights reserved.

"""W-B Part B — the twelve mirrored profile pairs are declared and verified (NT-WB-11..13, 29)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from intergrax.applications._shared.runtime_boundary_adapters import (
    PROFILE_PROJECTIONS,
    application_profile_to_runtime_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.host_profile_slices import (
    ContextProfile,
    CostProfile,
    MemoryProfile,
    ReliabilityProfile,
)
from pydantic import ValidationError

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_UNPROJECTED = {
    "MemoryProfile": 17,
    "CostProfile": 9,
    "ReliabilityProfile": 9,
    "PolicyRulesProfile": 2,
    "AdaptiveProfile": 1,
}
_MODULE_PATH = re.compile(r"[A-Za-z0-9_/]+\.py")


def test_nt_wb_11_twelve_pairs_verified_with_exact_unprojected_counts() -> None:
    assert len(PROFILE_PROJECTIONS) == 12
    sizes = {
        projection.name: len(projection.not_projected)
        for projection, _source in PROFILE_PROJECTIONS
        if projection.not_projected
    }
    assert sizes == _EXPECTED_UNPROJECTED
    for projection, source in PROFILE_PROJECTIONS:
        projection.verify(source)
        assert all(reason.strip() for reason in projection.not_projected.values())


@pytest.mark.parametrize("slice_cls", [MemoryProfile, CostProfile, ReliabilityProfile, ContextProfile])
def test_nt_wb_12_slices_forbid_unknown_keys(slice_cls: type) -> None:
    with pytest.raises(ValidationError):
        slice_cls.model_validate({"retention_days": 7})


def test_nt_wb_13_projection_output_unchanged() -> None:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="wb.nt13")
    profile = profile.model_copy(
        update={
            "memory_profile": profile.memory_profile.model_copy(update={"retention_days": 7}),
        },
    )
    runtime_profile = application_profile_to_runtime_profile(profile)
    assert runtime_profile is not None

    for name, target in (
        ("memory_profile", MemoryProfile),
        ("context_profile", ContextProfile),
        ("cost_profile", CostProfile),
        ("reliability_profile", ReliabilityProfile),
    ):
        dumped = getattr(profile, name).model_dump(mode="json")
        expected = target.model_validate({k: v for k, v in dumped.items() if k in target.model_fields})
        assert getattr(runtime_profile, name) == expected, name


def test_nt_wb_29_every_unprojected_field_has_a_verified_consumer() -> None:
    checked = 0
    for projection, _source in PROFILE_PROJECTIONS:
        for field_name, reason in projection.not_projected.items():
            match = _MODULE_PATH.search(reason)
            assert match is not None, f"{projection.name}.{field_name}: no module named in reason"
            module_file = _REPO_ROOT / match.group(0)
            assert module_file.is_file(), f"{projection.name}.{field_name}: {match.group(0)} missing"
            source_text = module_file.read_text(encoding="utf-8")
            assert re.search(rf"\b{re.escape(field_name)}\b", source_text), (
                f"{projection.name}.{field_name}: not found in {match.group(0)}"
            )
            checked += 1
    assert checked == 38
