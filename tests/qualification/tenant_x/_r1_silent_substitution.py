# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 silent-substitution matrix (critical authority/admission paths)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r1_types import (
    R1SilentSubstitutionClass,
    R1SilentSubstitutionRow,
    SANCTIONED_ROOT_TENANT_CONTRACT,
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


def r1_critical_unknown_substitutions() -> tuple[str, ...]:
    return tuple(
        row.pattern_id
        for row in R1_SILENT_SUBSTITUTION_MATRIX
        if row.critical_path and row.classification == R1SilentSubstitutionClass.UNKNOWN
    )


def r1_illegal_critical_substitutions() -> tuple[str, ...]:
    return tuple(
        row.pattern_id
        for row in R1_SILENT_SUBSTITUTION_MATRIX
        if row.critical_path
        and row.classification == R1SilentSubstitutionClass.TENANT_REQUIRED_ILLEGAL_SUBSTITUTION
    )
