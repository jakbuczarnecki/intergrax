# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 silent-substitution matrix (critical authority/admission paths)."""

from __future__ import annotations

import re
from typing import Final

from tests.qualification.tenant_x._r1_structural_io import read_production
from tests.qualification.tenant_x._r1_types import (
    R1SilentSubstitutionClass,
    R1SilentSubstitutionRow,
    SANCTIONED_ROOT_TENANT_CONTRACT,
)

_DISCOVERY_SCAN_FILES: Final[tuple[str, ...]] = (
    "intergrax/runtime/execution/agent_runtime_io.py",
    "intergrax/contracts/runtime_execution_admission.py",
    "intergrax/contracts/functional_evidence/tenant.py",
    "intergrax/applications/_shared/environment_wiring.py",
    "intergrax/fastapi_core/auth/providers/allow_all.py",
    "intergrax/runtime/capacity/scheduler.py",
    "intergrax/integrations/_shared/p3/configs.py",
)

_DISCOVERY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("or_default_tenant_literal", re.compile(r'or\s+["\']default["\']', re.IGNORECASE)),
    ("or_global_literal", re.compile(r'or\s+["\']global["\']', re.IGNORECASE)),
    ("or_system_literal", re.compile(r'or\s+["\']system["\']', re.IGNORECASE)),
    ("default_tenant_symbol", re.compile(r"DEFAULT_TENANT")),
    ("fallback_tenant_phrase", re.compile(r"fallback\s+tenant", re.IGNORECASE)),
)

R1_SILENT_SUBSTITUTION_MATRIX: Final[tuple[R1SilentSubstitutionRow, ...]] = (
    R1SilentSubstitutionRow(
        pattern_id="runtime_request_metadata_override",
        surface_hint="intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest",
        classification=R1SilentSubstitutionClass.LEGAL_NON_TENANT,
        critical_path=True,
        evidence=("metadata tenant_id cannot override canonical RuntimeRequest.tenant_id",),
    ),
    R1SilentSubstitutionRow(
        pattern_id="root_admission_missing_tenant_fail_closed",
        surface_hint=SANCTIONED_ROOT_TENANT_CONTRACT,
        classification=R1SilentSubstitutionClass.LEGAL_NON_TENANT,
        critical_path=True,
        evidence=("empty tenant_id raises ValueError — no silent default/global substitution",),
    ),
    R1SilentSubstitutionRow(
        pattern_id="functional_evidence_default",
        surface_hint="intergrax/contracts/functional_evidence/tenant.py",
        classification=R1SilentSubstitutionClass.LEGAL_NON_TENANT,
        critical_path=True,
        evidence=("require_tenant_id_from_exec_ctx — never guess or default",),
    ),
    R1SilentSubstitutionRow(
        pattern_id="environment_harness_literal",
        surface_hint="intergrax/applications/_shared/environment_wiring.py",
        classification=R1SilentSubstitutionClass.COMPATIBILITY_SEAM,
        critical_path=False,
        evidence=("tenant_id=harness bounded application wiring — not platform root",),
    ),
    R1SilentSubstitutionRow(
        pattern_id="observability_bridge_default",
        surface_hint="**/tracking/observability_bridge.py",
        classification=R1SilentSubstitutionClass.LEGAL_NON_TENANT,
        critical_path=False,
        evidence=("observability tracking label default — non-tenant admission surface",),
    ),
    R1SilentSubstitutionRow(
        pattern_id="allow_all_dev_auth",
        surface_hint="intergrax/fastapi_core/auth/providers/allow_all.py",
        classification=R1SilentSubstitutionClass.COMPATIBILITY_SEAM,
        critical_path=False,
        evidence=("development auth provider — cannot widen production execution tenant",),
    ),
)


def discover_silent_substitution_hits() -> tuple[tuple[str, str, str], ...]:
    hits: list[tuple[str, str, str]] = []
    for rel in _DISCOVERY_SCAN_FILES:
        text = read_production(rel)
        for pattern_id, pattern in _DISCOVERY_PATTERNS:
            if pattern.search(text):
                hits.append((pattern_id, rel, pattern.pattern))
    return tuple(hits)


def _matrix_pattern_ids() -> frozenset[str]:
    return frozenset(row.pattern_id for row in R1_SILENT_SUBSTITUTION_MATRIX)


def r1_undiscovered_critical_substitution_patterns() -> tuple[str, ...]:
    """Mechanical hits with no review row (critical paths only)."""
    known = _matrix_pattern_ids()
    undiscovered: list[str] = []
    for pattern_id, rel, _ in discover_silent_substitution_hits():
        synthetic_id = f"discovered:{pattern_id}:{rel}"
        if pattern_id not in known and "agent_runtime_io" in rel:
            undiscovered.append(synthetic_id)
    return tuple(undiscovered)


def r1_critical_unknown_substitutions() -> tuple[str, ...]:
    unknown_rows = tuple(
        row.pattern_id
        for row in R1_SILENT_SUBSTITUTION_MATRIX
        if row.critical_path and row.classification == R1SilentSubstitutionClass.UNKNOWN
    )
    return (*unknown_rows, *r1_undiscovered_critical_substitution_patterns())


def r1_illegal_critical_substitutions() -> tuple[str, ...]:
    return tuple(
        row.pattern_id
        for row in R1_SILENT_SUBSTITUTION_MATRIX
        if row.critical_path
        and row.classification == R1SilentSubstitutionClass.TENANT_REQUIRED_ILLEGAL_SUBSTITUTION
    )
