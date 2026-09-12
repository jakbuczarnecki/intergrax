"""Full Data Pack validation framework tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DataPackValidationVerdict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.service import (
    DataPackValidationPreconditionError,
    validate_full_data_pack,
)
from platform_proofs.scenarios.verified_product_identification.scripts.dataset.dataset_validation.run_data_pack_validation import (
    main as run_validation_cli,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_full_validation_test_support import (
    CorruptionKind,
    apply_corruption,
    build_valid_validation_fixture,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VALIDATION_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/validation"
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_valid_fixture_passes_full_validation(tmp_path: Path, monkeypatch) -> None:
    fixture = build_valid_validation_fixture(tmp_path, monkeypatch)
    scratch_root = tmp_path / "scratch"
    report = validate_full_data_pack(
        fixture.artifact_root,
        expectations=fixture.expectations,
        scratch_root=scratch_root,
    )
    assert report.verdict is DataPackValidationVerdict.PASS
    assert report.summary.finalized_artifact_valid is True
    assert report.summary.observed_relational_count == fixture.expectations.record_count
    assert report.summary.observed_embedding_count == fixture.expectations.record_count


def test_precondition_fail_for_missing_artifact_root(tmp_path: Path) -> None:
    with pytest.raises(DataPackValidationPreconditionError):
        validate_full_data_pack(tmp_path / "missing")


@pytest.mark.parametrize("kind", list(CorruptionKind))
def test_corruption_matrix_fails_deterministically(
    tmp_path: Path,
    monkeypatch,
    kind: CorruptionKind,
) -> None:
    fixture = build_valid_validation_fixture(tmp_path / kind.value, monkeypatch)
    apply_corruption(fixture, kind)
    report = validate_full_data_pack(
        fixture.artifact_root,
        expectations=fixture.expectations,
        scratch_root=tmp_path / kind.value / "scratch",
    )
    assert report.verdict is DataPackValidationVerdict.FAIL, kind.value


def test_cli_precondition_exit_code(tmp_path: Path) -> None:
    assert run_validation_cli(["--artifact-root", str(tmp_path / "missing")]) == 2


def test_cli_validation_fail_exit_code(tmp_path: Path, monkeypatch) -> None:
    fixture = build_valid_validation_fixture(tmp_path, monkeypatch)
    apply_corruption(fixture, CorruptionKind.MANIFEST_RECORD_COUNT_MISMATCH)
    assert (
        run_validation_cli(
            [
                "--artifact-root",
                str(fixture.artifact_root),
                "--scratch-root",
                str(tmp_path / "scratch-fail"),
            ]
        )
        == 1
    )


def test_validation_core_has_no_database_or_model_imports() -> None:
    forbidden_fragments = (
        "qdrant",
        "postgresql",
        "mysql",
        "pgvector",
        "torch",
        "sentence_transformers",
        "transformers",
    )
    violations: list[str] = []
    for module_path in sorted(_VALIDATION_ROOT.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.name} -> {imported}")
        source = module_path.read_text(encoding="utf-8")
        for fragment in ("dict[str, Any]", "getattr", "setattr", "hasattr", "inspect"):
            if fragment in source:
                violations.append(f"{module_path.name} contains {fragment}")
    assert violations == []


def test_canonical_expectations_shard_math() -> None:
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
        canonical_v1_validation_expectations,
    )

    expectations = canonical_v1_validation_expectations()
    assert expectations.record_count == 3_770_377
    assert expectations.shard_size == 1_000
    assert expectations.shard_count == 3_771
    assert expectations.expected_shard_record_count(3770) == 1_000
    assert expectations.expected_shard_record_count(3771) == 377
