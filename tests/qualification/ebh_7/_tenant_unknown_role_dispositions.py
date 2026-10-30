# © Artur Czarnecki. All rights reserved.

"""GAP-TEN-UNK-05 — five unknown_role tenant surfaces @ HEAD."""

from __future__ import annotations

from typing import Final

TenantUnknownRoleDisposition = str

GAP_TEN_UNK_05_SURFACE_IDS: Final[frozenset[str]] = frozenset(
    {
        "intergrax/integrations/contracts/credential.py::CredentialRef.tenant_id::class.field.tenant",
        "intergrax/integrations/contracts/credential.py::CredentialResolutionContext.tenant_id::class.field.tenant",
        "intergrax/integrations/contracts/credential.py::CredentialUseEvidence.tenant_id::class.field.tenant",
        "intergrax/integrations/contracts/credential.py::CredentialUseGrant.tenant_id::class.field.tenant",
        "intergrax/integrations/contracts/credential.py::CredentialUseScope.tenant_id::class.field.tenant",
    }
)

EBH7_UNKNOWN_ROLE_DISPOSITIONS: Final[dict[str, TenantUnknownRoleDisposition]] = {
    surface_id: "N/A_WITH_EVIDENCE" for surface_id in sorted(GAP_TEN_UNK_05_SURFACE_IDS)
}


def assert_ebh7_gap_ten_unk_05_closed() -> None:
    from tests.qualification.tenant_x._inventory import get_tenant_x_inventory
    from tests.qualification.tenant_x._types import TenantPropagationRole

    unknown = frozenset(
        r.surface_id for r in get_tenant_x_inventory() if r.propagation_role == TenantPropagationRole.UNKNOWN
    )
    assert unknown == GAP_TEN_UNK_05_SURFACE_IDS
    assert set(EBH7_UNKNOWN_ROLE_DISPOSITIONS) == GAP_TEN_UNK_05_SURFACE_IDS
    assert all(v == "N/A_WITH_EVIDENCE" for v in EBH7_UNKNOWN_ROLE_DISPOSITIONS.values())
