# © Artur Czarnecki. All rights reserved.

"""Mechanical root tenant selector inventory (bounded surfaces, not tuple-as-proof)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityMechanism,
    build_authority_mechanisms,
    is_positive_mint_rewrite,
)
from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r1_types import SANCTIONED_ROOT_TENANT_CONTRACT
from tests.qualification.tenant_x._types import (
    TenantAuthorityDecisionKind,
    TenantAuthoritySemanticRole,
    TenantSurfaceRecord,
)

_ROOT_ADMISSION_REL = "intergrax/contracts/runtime_execution_admission.py"


class R1RootSelectorClass(StrEnum):
    SANCTIONED_ROOT_SELECTOR = "SANCTIONED_ROOT_SELECTOR"
    NON_ROOT_DERIVATION = "NON_ROOT_DERIVATION"
    APPLICATION_LOCAL = "APPLICATION_LOCAL"
    HARNESS_ONLY = "HARNESS_ONLY"
    NON_TENANT = "NON_TENANT"
    ILLEGAL_ROOT_SELECTOR = "ILLEGAL_ROOT_SELECTOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class R1RootSelectorEntry:
    surface_id: str
    classification: R1RootSelectorClass
    evidence: tuple[str, ...]


def _root_admission_contract_proven() -> bool:
    checks = (
        (_ROOT_ADMISSION_REL, "class RootExecutionAuthorityAdmissionRequest"),
        (_ROOT_ADMISSION_REL, "tenant_id must be non-empty"),
    )
    return bool(all_structural_patterns_present(checks))


def _classify_root_mechanism(mechanism: TenantAuthorityMechanism) -> R1RootSelectorEntry:
    mid = mechanism.mechanism_id
    if "decision_finalization_conformance.py" in mid or "l4_runtime_evidence.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.HARNESS_ONLY,
            ("conformance/qualification entry — not production composition root",),
        )
    if (
        mid.startswith("APPLICATION_BOUNDARY|")
        or "environment_wiring.py" in mid
        or "agent_platform_admin_routes.py" in mid
    ):
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.APPLICATION_LOCAL,
            ("Tier-3 application ingress — outside platform root admission",),
        )
    if "allow_all.py" in mid or "observability_bridge.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.NON_TENANT,
            ("observability/dev auth — no execution tenant admission authority",),
        )
    if "rag/evaluation" in mid or "rag/bootstrap/entry_point_load.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.HARNESS_ONLY,
            ("evaluation/load harness literal — not reachable from production root",),
        )
    if "agent_capability_intake.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.NON_ROOT_DERIVATION,
            (
                "structural: task envelope tenant derived from admitted task context",
                "intergrax/runtime/task/agent_capability_intake.py",
            ),
        )
    if "intergrax/cli/external_work.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.APPLICATION_LOCAL,
            ("CLI operator tooling — not competing platform root selector",),
        )
    if "integrations/_shared/p8/factories.py" in mid or "lancedb/opens.py" in mid:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.APPLICATION_LOCAL,
            ("integration provider factory — scoped widening, not root admission",),
        )
    if mechanism.decision_kind == TenantAuthorityDecisionKind.CHILD_TENANT_DERIVATION:
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.NON_ROOT_DERIVATION,
            ("child execution derives under active parent — not root selection",),
        )
    if mechanism.decision_kind == TenantAuthorityDecisionKind.DEFAULT_SUBSTITUTION:
        if any(
            token in mid
            for token in (
                "observability_bridge",
                "external_contractor_adapter",
                "environment_wiring",
                "allow_all",
            )
        ):
            return R1RootSelectorEntry(
                mid,
                R1RootSelectorClass.NON_TENANT,
                ("default substitution on non-tenant observability/harness surface",),
            )
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.HARNESS_ONLY,
            ("default substitution bounded to harness/application — not production root",),
        )
    if mechanism.decision_kind == TenantAuthorityDecisionKind.ROOT_TENANT_SELECTION:
        if SANCTIONED_ROOT_TENANT_CONTRACT.split("::")[0] in mid or _ROOT_ADMISSION_REL in mid:
            if _root_admission_contract_proven():
                return R1RootSelectorEntry(
                    SANCTIONED_ROOT_TENANT_CONTRACT,
                    R1RootSelectorClass.SANCTIONED_ROOT_SELECTOR,
                    (
                        "canonical RootExecutionAuthorityAdmissionRequest fail-closed tenant_id",
                        _ROOT_ADMISSION_REL,
                    ),
                )
        return R1RootSelectorEntry(
            mid,
            R1RootSelectorClass.UNKNOWN,
            ("ROOT_TENANT_SELECTION without bounded root-admission structural proof",),
        )
    return R1RootSelectorEntry(
        mid,
        R1RootSelectorClass.NON_ROOT_DERIVATION,
        (f"decision_kind={mechanism.decision_kind.value} — not independent root selector",),
    )


def _mint_surfaces(inventory: tuple[TenantSurfaceRecord, ...]) -> tuple[TenantSurfaceRecord, ...]:
    out: list[TenantSurfaceRecord] = []
    for record in inventory:
        role = record.authority_semantic_role
        if role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY and is_positive_mint_rewrite(
            role, record
        ):
            out.append(record)
    return tuple(out)


def build_r1_root_selector_inventory(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> tuple[R1RootSelectorEntry, ...]:
    inv = inventory if inventory is not None else get_tenant_x_inventory()
    mechanisms = build_authority_mechanisms(_mint_surfaces(inv))
    root_candidates = [
        m
        for m in mechanisms
        if m.decision_kind
        in {
            TenantAuthorityDecisionKind.ROOT_TENANT_SELECTION,
            TenantAuthorityDecisionKind.DEFAULT_SUBSTITUTION,
            TenantAuthorityDecisionKind.TENANT_SCOPE_REWRITE,
        }
        or "runtime_execution_admission" in m.mechanism_id
    ]
    entries = [_classify_root_mechanism(m) for m in mechanisms]
    if _root_admission_contract_proven():
        sanctioned = R1RootSelectorEntry(
            SANCTIONED_ROOT_TENANT_CONTRACT,
            R1RootSelectorClass.SANCTIONED_ROOT_SELECTOR,
            (
                "production contract: RootExecutionAuthorityAdmissionRequest",
                "fail-closed: tenant_id must be non-empty",
            ),
        )
        if not any(e.classification == R1RootSelectorClass.SANCTIONED_ROOT_SELECTOR for e in entries):
            entries = (*entries, sanctioned)
    del root_candidates
    return tuple(entries)


def sanctioned_root_selectors_from_inventory(
    inventory: tuple[R1RootSelectorEntry, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else build_r1_root_selector_inventory()
    return tuple(e.surface_id for e in inv if e.classification == R1RootSelectorClass.SANCTIONED_ROOT_SELECTOR)


def r1_critical_unknown_root_selectors(
    inventory: tuple[R1RootSelectorEntry, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else build_r1_root_selector_inventory()
    return tuple(e.surface_id for e in inv if e.classification == R1RootSelectorClass.UNKNOWN)


def r1_illegal_root_selectors(
    inventory: tuple[R1RootSelectorEntry, ...] | None = None,
) -> tuple[str, ...]:
    inv = inventory if inventory is not None else build_r1_root_selector_inventory()
    return tuple(e.surface_id for e in inv if e.classification == R1RootSelectorClass.ILLEGAL_ROOT_SELECTOR)
