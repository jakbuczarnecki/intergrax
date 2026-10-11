# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R2-R1-R1 anti-bypass qualification probes."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_persisted_r2_r2_gates import (
    compute_persisted_r2_r2_metrics,
    persisted_r2_r2_gate_passes,
)
from tests.qualification.compat_x._compat_x_r2_r2_r1_implicit_compatibility import (
    ImplicitCompatibilityDisposition,
    build_implicit_compatibility_report_cached,
    classify_implicit_compatibility_in_source,
    r2_r2_r1_r1_implicit_gate_passes,
    synthetic_module_fails_implicit_gate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_OWNER_USER_PROFILE = "intergrax/memory/user_profile_serialization.py"
_OWNER_AD_LEGACY = "intergrax/agent_distribution/sqlite_payload_legacy.py"
_PROBE_A = "synthetic/qualification/compat_x_r2_r2_r1_r1_probe_a.py"
_PROBE_A2 = "synthetic/qualification/compat_x_r2_r2_r1_r1_probe_a2.py"
_PROBE_B = "synthetic/qualification/compat_x_r2_r2_r1_r1_probe_b.py"
_PROBE_C = "synthetic/qualification/compat_x_r2_r2_r1_r1_probe_c.py"
_PROBE_D = "synthetic/qualification/compat_x_r2_r2_r1_r1_probe_d.py"


def test_cx_r2_r2_r1_r1_gate_passes() -> None:
    assert r2_r2_r1_r1_implicit_gate_passes()
    assert persisted_r2_r2_gate_passes()
    metrics = compute_persisted_r2_r2_metrics()
    assert metrics.implicit_compatibility_branches == 0
    assert metrics.orphan_migration_readers == 0
    report = build_implicit_compatibility_report_cached()
    assert report.module_wide_sanctioned_exemptions == 0


def test_cx_r2_r2_r1_r1_probe_a_before_validator_blocker_and_gate_fails() -> None:
    source = '''
from pydantic import BaseModel, model_validator

class TaskMemoryRecord(BaseModel):
    schema_version: str = "task_memory_record.v1"

    @model_validator(mode="before")
    @classmethod
    def normalize_old_format(cls, data):
        return data
'''
    classified = classify_implicit_compatibility_in_source(_PROBE_A, source)
    assert any(
        item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        for item in classified
    )
    assert not any(
        item.disposition == ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION
        for item in classified
    )
    assert synthetic_module_fails_implicit_gate(_PROBE_A, source)


def test_cx_r2_r2_r1_r1_probe_a2_runtime_event_normalization_passes() -> None:
    runtime_event_module = "intergrax/contracts/runtime_event.py"
    source = '''
from pydantic import BaseModel, field_validator

class RuntimeEvent(BaseModel):
    event_id: str = ""

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id_field(cls, value):
        return value
'''
    classified = classify_implicit_compatibility_in_source(runtime_event_module, source)
    assert any(
        item.disposition == ImplicitCompatibilityDisposition.CURRENT_ONLY_NORMALIZATION
        for item in classified
    )
    assert not any(
        item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        for item in classified
    )


def test_cx_r2_r2_r1_r1_probe_b_schema_injection_inside_owner_blocked() -> None:
    source = '''
def unsanctioned_read(raw):
    if "schema_version" not in raw:
        raw["schema_version"] = "user_profile.v1"
    return raw
'''
    classified = classify_implicit_compatibility_in_source(_OWNER_USER_PROFILE, source)
    blockers = [
        item
        for item in classified
        if item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
    ]
    assert blockers
    assert synthetic_module_fails_implicit_gate(_OWNER_USER_PROFILE, source)


def test_cx_r2_r2_r1_r1_probe_c_bare_validate_blocker_and_gate_fails() -> None:
    source = '''
from pydantic import BaseModel

class Payload(BaseModel):
    schema_version: str = "payload.v1"

def read_persisted(raw):
    return Payload.model_validate(raw)
'''
    classified = classify_implicit_compatibility_in_source(_PROBE_C, source)
    assert any(
        item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        for item in classified
    )
    assert synthetic_module_fails_implicit_gate(_PROBE_C, source)


def test_cx_r2_r2_r1_r1_probe_d_symbol_level_sanctioning() -> None:
    source = '''
def decode_agent_artifact_metadata_payload(payload):
    return payload

def rogue_reader(raw):
    if "schema_version" not in raw:
        raw["schema_version"] = "x"
    return raw
'''
    classified = classify_implicit_compatibility_in_source(_OWNER_AD_LEGACY, source)
    assert any(
        item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        and item.candidate.signal == "missing_schema_version_injection"
        for item in classified
    )
    assert synthetic_module_fails_implicit_gate(_OWNER_AD_LEGACY, source)


def test_cx_r2_r2_r1_r1_probe_d_sanctioned_real_module_passes() -> None:
    report = build_implicit_compatibility_report_cached()
    assert not any(
        b.candidate.module_path == _OWNER_AD_LEGACY for b in report.blockers
    )
