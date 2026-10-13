# © Artur Czarnecki. All rights reserved.

"""Negative matrix for sandbox factory AST ownership (PROD-Q-P0-R1-R1-R1-R1-R1-R1-R1)."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tests.qualification.prod_q._p0_audit_gate import (
    prod_q_p0_core_violations,
    prod_q_p0_r1_r1_r1_r1_r1_r1_r1_gate_violations,
)
from tests.qualification.prod_q._p0_classification import classified_inventory
from tests.qualification.prod_q._p0_discovery import repo_root
from tests.qualification.prod_q._p0_factory_ast_evidence import (
    analyze_sandbox_factory_ast_evidence,
    sandbox_factory_provider_spec,
)
from tests.qualification.prod_q._p0_findings_register import (
    build_reconciled_prod_q_findings,
    build_semantic_concern_records,
    concern_finding_reconciliation_violations,
    factory_anchor_ownership_current_head_evidence,
)
from tests.qualification.prod_q._p0_types import (
    OwnershipResolutionState,
    SemanticConcernResolutionState,
)

pytestmark = [pytest.mark.unit, pytest.mark.qualification]


def _provider_root(repo: Path, provider: str) -> Path:
    return repo / "intergrax/integrations/providers/sandbox_host" / provider


def _copy_provider(repo: Path, dest: Path, provider: str) -> None:
    shutil.copytree(_provider_root(repo, provider), _provider_root(dest, provider))


def _factory_member(provider: str):
    symbol = f"create_{provider}_sandbox_host_integration"
    return next(
        record
        for record in classified_inventory()
        if record.symbol == symbol
    )


@pytest.mark.parametrize("provider", ("e2b", "modal", "daytona"))
def test_valid_provider_factory_graph_proven(provider: str) -> None:
    spec = sandbox_factory_provider_spec(provider)
    assert spec is not None
    result = analyze_sandbox_factory_ast_evidence(
        repo_root(),
        provider,
        spec.typed_integration_factory,
        spec.bundle_relpath,
    )
    assert result.proven
    member = _factory_member(provider)
    evidence = factory_anchor_ownership_current_head_evidence((member,))
    assert evidence is not None
    assert "ownership_assertion:OWNERSHIP_RESOLVED" in evidence
    assert member.composition_owner == spec.register_relpath


def test_remove_contract_factory_binding_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "e2b")
    path = _provider_root(tmp_path, "e2b") / "contract_spec.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "    contract_factory=create_e2b_sandbox_host_integration,\n",
        "",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "e2b",
        "create_e2b_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    )
    assert not result.proven


def test_wrong_contract_factory_function_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "modal")
    bundle = _provider_root(tmp_path, "modal") / "bundle.py"
    bundle_text = bundle.read_text(encoding="utf-8")
    bundle_text += (
        "\n\n"
        "def create_modal_sandbox_host_integration_decoy() -> ModalSandboxHostIntegration:\n"
        "    return ModalSandboxHostIntegration.for_provider(provider_id='modal', display_name='x')\n"
    )
    bundle.write_text(bundle_text, encoding="utf-8")
    spec_path = _provider_root(tmp_path, "modal") / "contract_spec.py"
    spec_text = spec_path.read_text(encoding="utf-8")
    spec_text = spec_text.replace(
        "contract_factory=create_modal_sandbox_host_integration",
        "contract_factory=create_modal_sandbox_host_integration_decoy",
    )
    spec_path.write_text(spec_text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "modal",
        "create_modal_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    )
    assert not result.proven


def test_contract_factory_keyword_only_in_comment_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "daytona")
    path = _provider_root(tmp_path, "daytona") / "contract_spec.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "    contract_factory=create_daytona_sandbox_host_integration,\n",
        "    # contract_factory=create_daytona_sandbox_host_integration\n",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "daytona",
        "create_daytona_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
    )
    assert not result.proven


def test_integration_class_mismatch_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "e2b")
    path = _provider_root(tmp_path, "e2b") / "contract_spec.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "integration_class=E2bSandboxHostIntegration",
        "integration_class=ModalSandboxHostIntegration",
    )
    text = text.replace(
        "from intergrax.integrations.providers.sandbox_host.e2b.integration import",
        "from intergrax.integrations.providers.sandbox_host.modal.integration import ModalSandboxHostIntegration\n"
        "from intergrax.integrations.providers.sandbox_host.e2b.integration import",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "e2b",
        "create_e2b_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    )
    assert not result.proven


def test_integration_no_canonical_base_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "modal")
    path = _provider_root(tmp_path, "modal") / "integration.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "class ModalSandboxHostIntegration(SandboxHostIntegrationContract):",
        "class ModalSandboxHostIntegration:",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "modal",
        "create_modal_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    )
    assert not result.proven


def test_backend_register_wrong_class_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "daytona")
    path = _provider_root(tmp_path, "daytona") / "integration.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "SandboxHostBackend.register(DaytonaSandboxHostIntegration)",
        "SandboxHostBackend.register(ModalSandboxHostIntegration)",
    )
    text = text.replace(
        "from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract",
        "from intergrax.integrations.providers.sandbox_host.modal.integration import ModalSandboxHostIntegration\n"
        "from intergrax.runtime.integrations.categories.devops import SandboxHostIntegrationContract",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "daytona",
        "create_daytona_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
    )
    assert not result.proven


def test_contract_specs_missing_spec_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "e2b")
    path = _provider_root(tmp_path, "e2b") / "contract_spec.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("CONTRACT_SPECS = (CONTRACT_SPEC,)", "CONTRACT_SPECS = ()")
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "e2b",
        "create_e2b_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/e2b/bundle.py",
    )
    assert not result.proven


def test_register_from_manifest_only_in_unused_function_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "modal")
    path = _provider_root(tmp_path, "modal") / "register.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "def register_modal_integration(*, override: bool = False) -> None:\n"
        "    register_from_manifest(\n"
        "        MANIFEST,\n"
        "        create_modal_sandbox_host,\n"
        "        override=override,\n"
        "        contract_specs=CONTRACT_SPECS,\n"
        "    )\n",
        "def register_modal_integration(*, override: bool = False) -> None:\n"
        "    pass\n\n\n"
        "def _unused_register() -> None:\n"
        "    register_from_manifest(\n"
        "        MANIFEST,\n"
        "        create_modal_sandbox_host,\n"
        "        override=override,\n"
        "        contract_specs=CONTRACT_SPECS,\n"
        "    )\n",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "modal",
        "create_modal_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/modal/bundle.py",
    )
    assert not result.proven


def test_register_contract_specs_wrong_provider_not_proven(tmp_path: Path) -> None:
    repo = repo_root()
    _copy_provider(repo, tmp_path, "daytona")
    path = _provider_root(tmp_path, "daytona") / "register.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from intergrax.integrations.providers.sandbox_host.daytona.contract_spec import CONTRACT_SPECS",
        "from intergrax.integrations.providers.sandbox_host.e2b.contract_spec import CONTRACT_SPECS",
    )
    path.write_text(text, encoding="utf-8")
    result = analyze_sandbox_factory_ast_evidence(
        tmp_path,
        "daytona",
        "create_daytona_sandbox_host_integration",
        "intergrax/integrations/providers/sandbox_host/daytona/bundle.py",
    )
    assert not result.proven


def test_factory_record_wrong_composition_owner_not_proven() -> None:
    factory = _factory_member("e2b")
    wrong = replace(
        factory,
        composition_owner="intergrax/integrations/providers/sandbox_host/e2b/wrong_owner.py",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    assert factory_anchor_ownership_current_head_evidence((wrong,)) is None


def test_factory_composition_unresolved_does_not_assert_ownership_resolved() -> None:
    factory = _factory_member("e2b")
    unresolved = replace(factory, composition_owner="COMPOSITION_UNRESOLVED")
    evidence = factory_anchor_ownership_current_head_evidence((unresolved,))
    assert evidence is not None
    assert "ownership_assertion:FACTORY_ANCHOR_IDENTIFIED" in evidence
    assert "ownership_assertion:OWNERSHIP_RESOLVED" not in evidence
    assert "composition_ownership:NOT_PROVEN" in evidence


def test_factory_empty_composition_owner_does_not_assert_ownership_resolved() -> None:
    factory = _factory_member("modal")
    unresolved = replace(factory, composition_owner="")
    evidence = factory_anchor_ownership_current_head_evidence((unresolved,))
    assert evidence is not None
    assert "ownership_assertion:OWNERSHIP_RESOLVED" not in evidence


def test_factory_sanctioned_owner_without_ast_support_not_resolved() -> None:
    factory = _factory_member("daytona")
    wrong_role = replace(
        factory,
        composition_owner="intergrax/integrations/providers/sandbox_host/daytona/contract_spec.py",
    )
    evidence = factory_anchor_ownership_current_head_evidence((wrong_role,))
    assert evidence is not None
    assert "ownership_assertion:OWNERSHIP_RESOLVED" not in evidence


def test_core_rejects_false_factory_resolution_on_unresolved_composition() -> None:
    factory = _factory_member("e2b")
    key = "concern:operation:PQ-F09:sandbox_provider:e2b:factory_anchor"
    falsely_resolved = replace(
        factory,
        composition_owner="COMPOSITION_UNRESOLVED",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    mutated = tuple(
        falsely_resolved if record.candidate_id == factory.candidate_id else record
        for record in classified_inventory()
    )
    concerns = build_semantic_concern_records(mutated)
    anchor = next(c for c in concerns if c.concern_key == key)
    assert anchor.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    assert any(
        "not evidenced" in item for item in prod_q_p0_core_violations(mutated)
    )


def test_core_detects_incorrect_factory_composition_owner() -> None:
    factory = _factory_member("modal")
    wrong = replace(
        factory,
        composition_owner="intergrax/integrations/providers/sandbox_host/modal/wrong_owner.py",
        ownership_resolution=OwnershipResolutionState.RESOLVED_WITH_CURRENT_HEAD_EVIDENCE,
    )
    mutated = tuple(
        wrong if record.candidate_id == factory.candidate_id else record
        for record in classified_inventory()
    )
    assert any(
        "ownership" in item.lower() or "composition" in item.lower()
        for item in prod_q_p0_core_violations(mutated)
    )


def test_broken_ast_evaluated_through_shared_core_not_cached_gate() -> None:
    factory = _factory_member("daytona")
    broken = replace(
        factory,
        module="intergrax/integrations/providers/sandbox_host/daytona/missing_bundle.py",
        composition_owner="COMPOSITION_UNRESOLVED",
        canonical_contract="CONTRACT_UNRESOLVED",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
    )
    mutated = tuple(
        broken if record.candidate_id == factory.candidate_id else record
        for record in classified_inventory()
    )
    concerns = build_semantic_concern_records(mutated)
    key = "concern:operation:PQ-F09:sandbox_provider:daytona:factory_anchor"
    anchor = next(c for c in concerns if c.concern_key == key)
    assert anchor.resolution_state == SemanticConcernResolutionState.UNRESOLVED
    assert prod_q_p0_core_violations(mutated) == ()
    findings = build_reconciled_prod_q_findings(mutated)
    assert concern_finding_reconciliation_violations(mutated, findings) == ()
    assert prod_q_p0_r1_r1_r1_r1_r1_r1_r1_gate_violations() == ()


def test_restored_factory_composition_clears_core_violations() -> None:
    factory = _factory_member("daytona")
    broken = replace(
        factory,
        module="intergrax/integrations/providers/sandbox_host/daytona/missing_bundle.py",
        composition_owner="COMPOSITION_UNRESOLVED",
        canonical_contract="CONTRACT_UNRESOLVED",
        ownership_resolution=OwnershipResolutionState.NOT_YET_RESOLVED,
    )
    mutated = tuple(
        broken if record.candidate_id == factory.candidate_id else record
        for record in classified_inventory()
    )
    assert prod_q_p0_core_violations(mutated) == ()
    restored = tuple(
        factory if record.candidate_id == factory.candidate_id else record
        for record in classified_inventory()
    )
    assert prod_q_p0_core_violations(restored) == ()
