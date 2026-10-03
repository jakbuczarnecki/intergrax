# © Artur Czarnecki. All rights reserved.

"""STATE-X-P0 mechanical baseline gates (SX-P0-Q01..Q12)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.qualification.state_x.inventory import (
    AuthorityRole,
    IdentityRole,
    MANDATORY_FAMILY_IDS,
    ProjectionOrTruth,
    STATE_X_FAMILY_INVENTORY,
    STATE_X_KNOWN_BLOCKERS,
    STATE_X_P0_ALLOWLIST_PATHS,
    STATE_X_P0_AUDITED_HEAD,
    StateFamilyInventoryEntry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_CHANGE_PREFIXES = (
    "intergrax/",
    "applications/",
    "scripts/",
)

_PROJECTION_TRUTH_CLASSES = frozenset(
    {
        ProjectionOrTruth.DERIVED_PROJECTION,
        ProjectionOrTruth.READ_MODEL,
    },
)


def _inventory_by_id() -> dict[str, StateFamilyInventoryEntry]:
    return {entry.family_id: entry for entry in STATE_X_FAMILY_INVENTORY}


def test_sx_p0_q01_mandatory_family_ids_exactly_once() -> None:
    ids = [entry.family_id for entry in STATE_X_FAMILY_INVENTORY]
    assert ids == list(MANDATORY_FAMILY_IDS)
    assert len(ids) == len(set(ids))


def test_sx_p0_q02_single_semantic_owner_contract_composition() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.semantic_owner.strip()
        assert entry.canonical_contract.strip()
        assert entry.composition_owner.strip()


def test_sx_p0_q03_no_unknown_semantic_owner_tokens() -> None:
    forbidden = {"UNKNOWN", "TBD", "unknown", "tbd"}
    for entry in STATE_X_FAMILY_INVENTORY:
        owner = entry.semantic_owner.upper()
        for token in forbidden:
            assert token not in owner


def test_sx_p0_q04_contract_paths_exist() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        path = _REPO_ROOT / entry.contract_path
        assert path.is_file(), f"missing contract: {entry.contract_path}"


def test_sx_p0_q05_projection_or_truth_classified() -> None:
    allowed = frozenset(ProjectionOrTruth)
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.projection_or_truth in allowed


def test_sx_p0_q06_projections_not_mint_authority() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        if entry.projection_or_truth in _PROJECTION_TRUTH_CLASSES:
            assert entry.authority_role != AuthorityRole.MINT_AUTHORITY
        if entry.family_id == "SX-F07":
            assert entry.authority_role != AuthorityRole.MINT_AUTHORITY


def test_sx_p0_q07_checkpoint_not_mint_authority_or_identity() -> None:
    checkpoint = _inventory_by_id()["SX-F01"]
    assert checkpoint.authority_role != AuthorityRole.MINT_AUTHORITY
    assert checkpoint.identity_role != IdentityRole.MINT_IDENTITY


def test_sx_p0_q08_known_blockers_exact_set() -> None:
    ids = {blocker.blocker_id for blocker in STATE_X_KNOWN_BLOCKERS}
    assert ids == {"SX-B01", "SX-B02", "SX-B03"}


def test_sx_p0_q09_applicable_frz_mapping_present() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.applicable_frz
        for code in entry.applicable_frz:
            assert code.startswith("FRZ-")


def test_sx_p0_q10_backup_restore_responsibility_classified() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.backup_restore_responsibility.value
        assert "TBD" not in entry.backup_restore_responsibility.value.upper()
        assert "UNKNOWN" not in entry.backup_restore_responsibility.value.upper()


def test_sx_p0_q11_tenant_audit_disposition_present() -> None:
    for entry in STATE_X_FAMILY_INVENTORY:
        assert entry.tenant_audit_disposition.value


def test_sx_p0_q12_no_production_file_changed_since_audited_head() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", STATE_X_P0_AUDITED_HEAD],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not changed:
        return
    for path in changed:
        normalized = path.replace("\\", "/")
        assert normalized in STATE_X_P0_ALLOWLIST_PATHS, (
            f"P0 scope violation: {normalized}"
        )
        for prefix in _FORBIDDEN_CHANGE_PREFIXES:
            assert not normalized.startswith(prefix), normalized


def test_sx_p0_no_family_mints_authority() -> None:
    mint_count = sum(
        1
        for entry in STATE_X_FAMILY_INVENTORY
        if entry.authority_role == AuthorityRole.MINT_AUTHORITY
    )
    assert mint_count == 0


@pytest.mark.parametrize("blocker_id", ["SX-B01", "SX-B02", "SX-B03"])
def test_sx_p0_blockers_have_owner(blocker_id: str) -> None:
    blocker = next(b for b in STATE_X_KNOWN_BLOCKERS if b.blocker_id == blocker_id)
    assert blocker.owner_child.startswith("STATE-X")
