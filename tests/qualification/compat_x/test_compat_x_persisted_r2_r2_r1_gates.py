# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R2-R2-R1 typed boundaries and implicit-compatibility gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.memory.user_profile_legacy_wire import (
    LegacyUserIdentityWire,
    LegacyUserPreferencesWire,
    LegacyUserProfileMemoryEntryWire,
    LegacyUserProfileWire,
)
from tests.qualification.compat_x._compat_x_persisted_r2_r2_gates import (
    compute_persisted_r2_r2_metrics,
    persisted_r2_r2_gate_passes,
)
from tests.qualification.compat_x._compat_x_r2_r2_r1_implicit_compatibility import (
    ImplicitCompatibilityDisposition,
    build_implicit_compatibility_report,
    classify_implicit_compatibility_in_source,
    r2_r2_r1_implicit_gate_passes,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "compat_x"
    / "historical_user_profile_wire_e5ae6a3.json"
)

_PROBE_A = "synthetic/qualification/compat_x_r2_r2_r1_probe_a.py"
_PROBE_B = "synthetic/qualification/compat_x_r2_r2_r1_probe_b.py"
_PROBE_C = "synthetic/qualification/compat_x_r2_r2_r1_probe_c.py"
_PROBE_D = "synthetic/qualification/compat_x_r2_r2_r1_probe_d.py"


def test_cx_r2_r2_r1_gate_passes() -> None:
    assert r2_r2_r1_implicit_gate_passes()
    assert persisted_r2_r2_gate_passes()
    metrics = compute_persisted_r2_r2_metrics()
    assert metrics.orphan_migration_readers == 0


def test_cx_r2_r2_r1_historical_user_profile_wire_fixture_fidelity() -> None:
    evidence = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert LegacyUserProfileWire.model_fields.keys() == set(evidence["legacy_user_profile_wire_fields"])
    assert LegacyUserIdentityWire.model_fields.keys() == set(evidence["legacy_user_identity_wire_fields"])
    assert LegacyUserPreferencesWire.model_fields.keys() == set(
        evidence["legacy_user_preferences_wire_fields"]
    )
    assert LegacyUserProfileMemoryEntryWire.model_fields.keys() == set(
        evidence["legacy_user_profile_memory_entry_wire_fields"]
    )


def test_cx_r2_r2_r1_probe_a_before_validator_flagged() -> None:
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
    assert any(item.candidate.signal == "pydantic_before_validator" for item in classified)
    assert any(
        item.disposition
        in {
            ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY,
            ImplicitCompatibilityDisposition.NON_COMPAT_VALIDATION,
        }
        for item in classified
    )


def test_cx_r2_r2_r1_probe_b_schema_version_injection_blocked() -> None:
    source = '''
def read_persisted(raw):
    if "schema_version" not in raw:
        raw["schema_version"] = "current.v1"
    return raw
'''
    classified = classify_implicit_compatibility_in_source(_PROBE_B, source)
    assert any(
        item.disposition == ImplicitCompatibilityDisposition.BLOCKER_IMPLICIT_COMPATIBILITY
        for item in classified
    )


def test_cx_r2_r2_r1_probe_c_validate_before_gate_flagged() -> None:
    source = '''
from pydantic import BaseModel

class Payload(BaseModel):
    schema_version: str = "payload.v1"

def read_persisted(raw):
    return Payload.model_validate(raw)
'''
    classified = classify_implicit_compatibility_in_source(_PROBE_C, source)
    assert any(
        item.candidate.signal == "bare_model_validate_without_schema_gate"
        for item in classified
    )


def test_cx_r2_r2_r1_probe_d_sanctioned_migration_accepted() -> None:
    report = build_implicit_compatibility_report()
    sanctioned_modules = {item.candidate.module_path for item in report.sanctioned}
    assert "intergrax/agent_distribution/sqlite_payload_legacy.py" in report.scanned_modules
    assert "intergrax/agent_distribution/sqlite_payload_legacy.py" in sanctioned_modules or not any(
        b.candidate.module_path == "intergrax/agent_distribution/sqlite_payload_legacy.py"
        for b in report.blockers
    )


def test_cx_r2_r2_r1_strong_typing_report_clean() -> None:
    report = build_implicit_compatibility_report()
    assert report.strong_typing_violations == ()
