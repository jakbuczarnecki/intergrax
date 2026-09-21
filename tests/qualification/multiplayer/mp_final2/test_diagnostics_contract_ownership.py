# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-2-C1 — operator-facing diagnostics contract ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.diagnostics.functional_diagnostic_check_status import (
    FunctionalDiagnosticCheckStatus as ContractCheckStatus,
)
from intergrax.contracts.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId as ContractCheckId,
)
from intergrax.contracts.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticSpecificationId as ContractSpecId,
)
from intergrax.contracts.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection as ContractProjection,
)
from intergrax.contracts.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus as ContractOutcomeStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import (
    FunctionalDiagnosticCheckStatus as RuntimeCheckStatus,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId as RuntimeCheckId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticSpecificationId as RuntimeSpecId,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection as RuntimeProjection,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus as RuntimeOutcomeStatus,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorProjector,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_DIAG = _REPO_ROOT / "intergrax" / "contracts" / "diagnostics"
_OPERATOR_CONTRACT = _CONTRACTS_DIAG / "functional_operator_projection.py"
_IDENTITY_CONTRACT = _CONTRACTS_DIAG / "functional_diagnostic_identity.py"
_STATUS_CONTRACT = _CONTRACTS_DIAG / "functional_diagnostic_check_status.py"
_E2E = Path(__file__).resolve().parent / "test_diagnostics_operability_e2e.py"
_PLUGIN = Path(__file__).resolve().parent / "test_diagnostics_pluginability.py"
_HOST = Path(__file__).resolve().parent / "host_operability.py"

_OPERATOR_DTO_NAMES = (
    "FunctionalDiagnosticOperatorProjection",
    "FunctionalDiagnosticOperatorFinding",
    "FunctionalDiagnosticOperatorLimitation",
    "FunctionalDiagnosticSummary",
    "FunctionalCheckPassResult",
    "FunctionalOperatorOutcomeStatus",
)


def _import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
    return modules


def _class_defs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _iter_production_python(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_operator_projection_canonical_module_is_contracts() -> None:
    assert (
        ContractProjection.__module__
        == "intergrax.contracts.diagnostics.functional_operator_projection"
    )
    assert (
        ContractOutcomeStatus.__module__
        == "intergrax.contracts.diagnostics.functional_operator_projection"
    )


def test_identity_and_status_canonical_modules_are_contracts() -> None:
    assert (
        ContractCheckId.__module__
        == "intergrax.contracts.diagnostics.functional_diagnostic_identity"
    )
    assert (
        ContractSpecId.__module__
        == "intergrax.contracts.diagnostics.functional_diagnostic_identity"
    )
    assert (
        ContractCheckStatus.__module__
        == "intergrax.contracts.diagnostics.functional_diagnostic_check_status"
    )


def test_runtime_reexports_are_same_objects_not_duplicates() -> None:
    assert RuntimeProjection is ContractProjection
    assert RuntimeOutcomeStatus is ContractOutcomeStatus
    assert RuntimeCheckId is ContractCheckId
    assert RuntimeSpecId is ContractSpecId
    assert RuntimeCheckStatus is ContractCheckStatus


def test_exactly_one_canonical_operator_projection_class_definition() -> None:
    definitions: list[str] = []
    for path in _iter_production_python(_REPO_ROOT / "intergrax"):
        text = path.read_text(encoding="utf-8")
        if "class FunctionalDiagnosticOperatorProjection" in text:
            definitions.append(path.relative_to(_REPO_ROOT).as_posix())
    assert definitions == [
        "intergrax/contracts/diagnostics/functional_operator_projection.py",
    ]


def test_exactly_one_canonical_outcome_status_enum_definition() -> None:
    definitions: list[str] = []
    for path in _iter_production_python(_REPO_ROOT / "intergrax"):
        text = path.read_text(encoding="utf-8")
        if "class FunctionalOperatorOutcomeStatus" in text:
            definitions.append(path.relative_to(_REPO_ROOT).as_posix())
    assert definitions == [
        "intergrax/contracts/diagnostics/functional_operator_projection.py",
    ]


def test_operator_contract_modules_do_not_import_runtime_or_applications() -> None:
    forbidden_prefixes = (
        "intergrax.runtime",
        "intergrax.collaborative_work",
        "applications",
        "agents",
    )
    for path in (_OPERATOR_CONTRACT, _IDENTITY_CONTRACT, _STATUS_CONTRACT):
        for module in _import_modules(path):
            for prefix in forbidden_prefixes:
                assert not (module == prefix or module.startswith(f"{prefix}.")), (
                    f"{path.relative_to(_REPO_ROOT)} imports {module}"
                )


def test_runtime_projector_imports_operator_dtos_from_contracts() -> None:
    projector_path = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "diagnostics"
        / "functional_operator_projection.py"
    )
    modules = _import_modules(projector_path)
    assert "intergrax.contracts.diagnostics.functional_operator_projection" in modules
    assert "FunctionalOperatorProjector" in _class_defs(projector_path)
    for dto_name in _OPERATOR_DTO_NAMES:
        assert dto_name not in _class_defs(projector_path)


def test_projector_remains_runtime_implementation() -> None:
    assert FunctionalOperatorProjector.__module__ == (
        "intergrax.runtime.diagnostics.functional_operator_projection"
    )


def test_e2e_consumer_imports_outcome_from_contracts_not_runtime_dto_ownership() -> None:
    for path in (_E2E, _PLUGIN):
        modules = _import_modules(path)
        assert "intergrax.contracts.diagnostics.functional_operator_projection" in modules
        assert "intergrax.runtime.diagnostics.functional_operator_projection" not in modules


def test_host_composition_may_import_projector_implementation() -> None:
    modules = _import_modules(_HOST)
    assert "intergrax.runtime.diagnostics.functional_operator_projection" in modules
    assert "intergrax.contracts.diagnostics.functional_operator_projection" in modules
