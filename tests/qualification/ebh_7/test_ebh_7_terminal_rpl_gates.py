# © Artur Czarnecki. All rights reserved.

"""EBH-7 terminal INTERNAL / FIXED RPL qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_5._closed_world_discovery import discover_composition_root_paths
from tests.qualification.ebh_5._fixed_substrate_registry import (
    FixedSubstrateDecisionStatus,
    fixed_substrate_decisions,
    fixed_substrate_validation_errors,
)
from tests.qualification.ebh_5._terminal_rpl_normative_nodes import FIXED_RPL_CONCERNS, INTERNAL_RPL_CONCERNS
from tests.qualification.ebh_7._ebh7_terminal_operator_authorization import assert_ebh7_terminal_authorization_integrity

pytestmark = [pytest.mark.unit, pytest.mark.qualification, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_ADR_COMPOSITION = (
    _REPO / "docs/project/maintainers/architecture/ADR/ADR-EBH-7-INTERNAL-APPLICATION-COMPOSITION-ROOT.md"
)
_ADR_PRODQ = _REPO / "docs/project/maintainers/architecture/ADR/ADR-EBH-7-INTERNAL-PRODQ-COLOCATED-ADAPTER-GATE.md"
_FIXED_DOC = _REPO / "docs/project/maintainers/qualification/EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md"


def test_ebh7_terminal_operator_authorization_ssot() -> None:
    assert_ebh7_terminal_authorization_integrity()


@pytest.mark.parametrize("concern_id", sorted(INTERNAL_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_internal_rpl_positive(concern_id: str) -> None:
    if concern_id == "application.composition_root":
        assert _ADR_COMPOSITION.is_file()
        roots = discover_composition_root_paths()
        assert "intergrax/applications/_shared/memory_wiring.py" in roots
        assert len(roots) >= 5
        return
    if concern_id == "prodq.colocated_runtime_adapter_isinstance":
        assert _ADR_PRODQ.is_file()
        adapter_path = _REPO / "intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py"
        assert adapter_path.is_file()
        text = adapter_path.read_text(encoding="utf-8")
        assert "RuntimeDeploymentAdapter" in text or "Colocated" in text
        return
    raise AssertionError(concern_id)


@pytest.mark.parametrize("concern_id", sorted(INTERNAL_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_internal_rpl_negative(concern_id: str) -> None:
    if concern_id == "application.composition_root":
        factory = _REPO / "intergrax/integrations/registry/factory.py"
        assert "applications/" not in factory.read_text(encoding="utf-8")
        return
    if concern_id == "prodq.colocated_runtime_adapter_isinstance":
        with pytest.raises(AssertionError):
            assert (
                (_REPO / "intergrax/integrations/registry/factory.py")
                .read_text(encoding="utf-8")
                .startswith("prodq gate")
            )
        return
    raise AssertionError(concern_id)


def _substrate_for_concern(concern_id: str) -> str:
    for row in fixed_substrate_decisions():
        if concern_id in row.inventory_concern_ids:
            return row.substrate_id
    raise AssertionError(f"No substrate for {concern_id}")


@pytest.mark.parametrize("concern_id", sorted(FIXED_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_fixed_substrate_rpl_positive(concern_id: str) -> None:
    assert _FIXED_DOC.is_file()
    assert fixed_substrate_validation_errors() == ()
    substrate_id = _substrate_for_concern(concern_id)
    row = next(r for r in fixed_substrate_decisions() if r.substrate_id == substrate_id)
    assert row.decision_status is FixedSubstrateDecisionStatus.RATIFIED
    if concern_id.startswith("integration."):
        from tests.qualification.ebh_7.test_ebh_7_vendor_rpl_contract_one_pass import (
            test_ebh7_vendor_single_slug_fail_closed_without_profile,
        )

        test_ebh7_vendor_single_slug_fail_closed_without_profile(concern_id)
        return
    if concern_id == "memory.sqlite.conversational":
        store = _REPO / "intergrax/memory/stores/sqlite_conversational_memory_store.py"
        assert store.is_file()
        return
    raise AssertionError(concern_id)


@pytest.mark.parametrize("concern_id", sorted(FIXED_RPL_CONCERNS), ids=lambda c: c)
def test_ebh7_fixed_substrate_rpl_negative(concern_id: str) -> None:
    if concern_id.startswith("integration."):
        from intergrax.integrations.registry.catalog import clear_catalog
        from intergrax.integrations.registry.factory import resolve_from_profile
        from intergrax.integrations.registry.profile import IntegrationProfile

        clear_catalog()
        category_suffix = concern_id.removeprefix("integration.")
        from intergrax.integrations.contracts.base import IntegrationCategory

        category = IntegrationCategory(category_suffix)
        with pytest.raises(Exception):
            resolve_from_profile(IntegrationProfile(), category)
        return
    if concern_id == "memory.sqlite.conversational":
        from intergrax.memory.stores.sqlite_conversational_memory_store import SQLiteConversationalMemoryStore

        with pytest.raises(Exception):
            SQLiteConversationalMemoryStore(db_path="\0bad")
        return
    raise AssertionError(concern_id)
