# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GEN = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "generate-lkw-platform-certification-matrix.py"
)
_WINDOWS_SRC = (
    _REPO_ROOT / "docs/project/maintainers/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json"
)
_LINUX_SRC = (
    _REPO_ROOT / "docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json"
)
_MATRIX_JSON = (
    _REPO_ROOT / "docs/project/maintainers/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json"
)
_MATRIX_MD = _REPO_ROOT / "docs/project/maintainers/public-adoption/LKW_PLATFORM_CERTIFICATION_MATRIX.md"

_SECRET_NEEDLES = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "token=",
    "mongodb://",
    "mongodb+srv://",
    "connection_string",
    "private_key",
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load(_GEN, "lkw_platform_certification_matrix_gen")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _build(
    gen: ModuleType,
    *,
    windows: dict[str, Any] | None = None,
    linux: dict[str, Any] | None = None,
    windows_sha: str | None = None,
    linux_sha: str | None = None,
    commit: str = "4847e9578251382c6a02eb8f00137dbf16b03553",
) -> dict[str, Any]:
    win = windows if windows is not None else _load_json(_WINDOWS_SRC)
    lin = linux if linux is not None else _load_json(_LINUX_SRC)
    return gen.build_matrix(
        repo_root=_REPO_ROOT,
        windows_evidence=win,
        linux_evidence=lin,
        windows_sha256=windows_sha or _sha256(_WINDOWS_SRC),
        linux_sha256=linux_sha or _sha256(_LINUX_SRC),
        generated_from_commit=commit,
    )


def test_valid_evidence_produces_valid_matrix(gen: ModuleType) -> None:
    matrix = _build(gen)
    assert matrix["matrix_status"] == "VALID"
    assert matrix["schema_version"] == "lkw.platform_certification_matrix.v1"


def test_matrix_contains_exactly_four_profiles(gen: ModuleType) -> None:
    matrix = _build(gen)
    assert set(matrix["profiles"]) == {
        "windows_native_runtime",
        "linux_docker_runtime",
        "linux_native_runtime",
        "macos_native_runtime",
    }


def test_windows_native_status_correct(gen: ModuleType) -> None:
    profile = _build(gen)["profiles"]["windows_native_runtime"]
    assert profile["operating_system"] == "windows"
    assert profile["execution_environment"] == "native_host"
    assert profile["implementation_status"] == "implemented"
    assert profile["application_hosting_status"] == "live_certified"
    assert profile["os_interaction_status"] == "live_certified"
    assert profile["full_core_platform_proof_certified_by_profile"] is False
    assert profile["native_host_certified"] is True
    assert profile["evidence_available"] is True


def test_linux_docker_status_correct(gen: ModuleType) -> None:
    profile = _build(gen)["profiles"]["linux_docker_runtime"]
    assert profile["operating_system"] == "linux"
    assert profile["execution_environment"] == "container"
    assert profile["implementation_status"] == "implemented"
    assert profile["application_hosting_status"] == "live_certified"
    assert profile["os_interaction_status"] == "live_certified"
    assert profile["full_core_platform_proof_certified_by_profile"] is False
    assert profile["native_host_certified"] is False
    assert profile["evidence_available"] is True


def test_linux_native_remains_uncertified(gen: ModuleType) -> None:
    profile = _build(gen)["profiles"]["linux_native_runtime"]
    assert profile["application_hosting_status"] == "not_live_certified"
    assert profile["os_interaction_status"] == "not_live_certified"
    assert profile["native_host_certified"] is False
    assert profile["evidence_available"] is False
    assert profile["evidence_source"] is None
    assert profile["certification_result"] == "NOT_CERTIFIED"
    assert profile["proofs"] == {}
    limitations = "\n".join(profile["limitations"])
    assert "Linux entrypoints are implemented." in limitations
    assert "No separate native Linux host live certification artifact exists." in (
        limitations
    )
    assert "Linux Docker runtime evidence does not certify native Linux" in limitations


def test_macos_remains_uncertified(gen: ModuleType) -> None:
    profile = _build(gen)["profiles"]["macos_native_runtime"]
    assert profile["application_hosting_status"] == "not_live_certified"
    assert profile["os_interaction_status"] == "not_live_certified"
    assert profile["native_host_certified"] is False
    assert profile["evidence_available"] is False
    assert profile["evidence_source"] is None
    assert profile["certification_result"] == "NOT_CERTIFIED"
    assert profile["proofs"] == {}
    limitations = "\n".join(profile["limitations"])
    assert "macOS entrypoints are implemented." in limitations
    assert "No macOS live certification artifact exists." in limitations
    assert "No macOS ProofReceipt has been recorded for this matrix." in limitations


def test_windows_hosting_proof_copied_from_source(gen: ModuleType) -> None:
    source = _load_json(_WINDOWS_SRC)["application_hosting_proof"]
    copied = _build(gen)["profiles"]["windows_native_runtime"]["proofs"][
        "application_hosting"
    ]
    assert copied["proof_id"] == source["proof_id"]
    assert copied["run_id"] == source["run_id"]
    assert copied["correlation_id"] == source["correlation_id"]
    assert copied["proof_kind"] == "platform_application_hosting"


def test_windows_interaction_proof_copied_from_source(gen: ModuleType) -> None:
    source = _load_json(_WINDOWS_SRC)["interaction_proof"]
    copied = _build(gen)["profiles"]["windows_native_runtime"]["proofs"][
        "os_interaction"
    ]
    assert copied["proof_id"] == source["proof_id"]
    assert copied["run_id"] == source["run_id"]
    assert copied["adapter_id"] == "lkw.windows_powershell"
    assert copied["source"] == "windows_powershell"
    assert copied["wrapper_runtime"] == "windows_powershell"
    assert copied["powershell_runtime"] == "Windows PowerShell"


def test_linux_hosting_proof_copied_from_source(gen: ModuleType) -> None:
    source = _load_json(_LINUX_SRC)["application_hosting_proof"]
    copied = _build(gen)["profiles"]["linux_docker_runtime"]["proofs"][
        "application_hosting"
    ]
    assert copied["proof_id"] == source["proof_id"]
    assert copied["run_id"] == source["run_id"]
    assert copied["correlation_id"] == source["correlation_id"]


def test_linux_interaction_proof_copied_from_source(gen: ModuleType) -> None:
    source = _load_json(_LINUX_SRC)["interaction_proof"]
    copied = _build(gen)["profiles"]["linux_docker_runtime"]["proofs"]["os_interaction"]
    assert copied["proof_id"] == source["proof_id"]
    assert copied["adapter_id"] == "lkw.linux_shell"
    assert copied["source"] == "linux_shell"
    assert copied["wrapper_runtime"] == "posix_sh"
    assert copied["powershell_runtime"] is None


def test_source_file_sha256_computed_correctly(gen: ModuleType) -> None:
    matrix = _build(gen)
    assert matrix["source_artifacts"]["windows_native"]["sha256"] == _sha256(
        _WINDOWS_SRC
    )
    assert matrix["source_artifacts"]["linux_docker"]["sha256"] == _sha256(_LINUX_SRC)
    assert matrix["source_artifacts"]["windows_native"]["sha256"] != _load_json(
        _WINDOWS_SRC
    ).get("source_tree_diff_sha256")


def test_matrix_id_is_deterministic(gen: ModuleType) -> None:
    first = _build(gen)["matrix_id"]
    second = _build(gen)["matrix_id"]
    assert first == second
    assert first.startswith("lkw-platform-matrix-")
    assert len(first) == len("lkw-platform-matrix-") + 12


def test_generation_byte_identical_for_unchanged_inputs(gen: ModuleType) -> None:
    first = gen.serialize_matrix_json(_build(gen))
    second = gen.serialize_matrix_json(_build(gen))
    assert first == second
    assert first.endswith("\n")


def test_check_passes_for_fresh_files(gen: ModuleType, tmp_path: Path) -> None:
    _matrix, json_text, md_text = gen.generate_artifacts(repo_root=_REPO_ROOT)
    staging = tmp_path / "repo"
    (staging / "docs/project/maintainers/public-adoption/evidence").mkdir(parents=True)
    (staging / gen.MATRIX_JSON_REL).write_text(json_text, encoding="utf-8", newline="\n")
    (staging / gen.MATRIX_MD_REL).write_text(md_text, encoding="utf-8", newline="\n")
    gen.check_artifacts(staging, json_text, md_text)


def test_check_mode_reuses_committed_generated_from_commit(
    gen: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate post-commit HEAD drift while matrix retains pre-commit stamp.
    monkeypatch.setattr(
        gen,
        "git_rev_parse_head",
        lambda _root: "ffffffffffffffffffffffffffffffffffffffff",
    )
    assert gen.main(["--check"]) == 0


def test_check_fails_for_stale_json(gen: ModuleType, tmp_path: Path) -> None:
    _matrix, json_text, md_text = gen.generate_artifacts(repo_root=_REPO_ROOT)
    staging = tmp_path / "repo"
    (staging / "docs/project/maintainers/public-adoption/evidence").mkdir(parents=True)
    (staging / gen.MATRIX_JSON_REL).write_text(
        json_text.replace("VALID", "STALE", 1), encoding="utf-8", newline="\n"
    )
    (staging / gen.MATRIX_MD_REL).write_text(md_text, encoding="utf-8", newline="\n")
    with pytest.raises(gen.MatrixGenerationError, match="stale_matrix_json"):
        gen.check_artifacts(staging, json_text, md_text)


def test_check_fails_for_stale_markdown(gen: ModuleType, tmp_path: Path) -> None:
    _matrix, json_text, md_text = gen.generate_artifacts(repo_root=_REPO_ROOT)
    staging = tmp_path / "repo"
    (staging / "docs/project/maintainers/public-adoption/evidence").mkdir(parents=True)
    (staging / gen.MATRIX_JSON_REL).write_text(json_text, encoding="utf-8", newline="\n")
    (staging / gen.MATRIX_MD_REL).write_text(
        md_text + "\nstale\n", encoding="utf-8", newline="\n"
    )
    with pytest.raises(gen.MatrixGenerationError, match="stale_matrix_markdown"):
        gen.check_artifacts(staging, json_text, md_text)


def test_missing_windows_evidence_fails(gen: ModuleType, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs/project/maintainers/public-adoption/evidence").mkdir(parents=True)
    (root / gen.LINUX_SOURCE_REL).write_text(_LINUX_SRC.read_text(encoding="utf-8"))
    with pytest.raises(gen.MatrixGenerationError, match="missing_source"):
        gen.generate_artifacts(repo_root=root)


def test_missing_linux_evidence_fails(gen: ModuleType, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "docs/project/maintainers/public-adoption/evidence").mkdir(parents=True)
    (root / gen.WINDOWS_SOURCE_REL).write_text(_WINDOWS_SRC.read_text(encoding="utf-8"))
    with pytest.raises(gen.MatrixGenerationError, match="missing_source"):
        gen.generate_artifacts(repo_root=root)


def test_malformed_source_json_fails(gen: ModuleType, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(gen.MatrixGenerationError, match="malformed_json"):
        gen.load_json(bad)


def test_failed_certification_result_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["certification_result"] = "FAIL"
    with pytest.raises(gen.MatrixGenerationError, match="certification_result_not_pass"):
        _build(gen, windows=windows)


def test_wrong_certification_profile_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["certification_profile"] = "windows_wrong"
    with pytest.raises(
        gen.MatrixGenerationError, match="certification_profile_mismatch"
    ):
        _build(gen, windows=windows)


def test_wrong_execution_environment_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["execution_environment"] = "native_host"
    with pytest.raises(
        gen.MatrixGenerationError, match="execution_environment_mismatch"
    ):
        _build(gen, linux=linux)


def test_wrong_os_family_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["execution_os_family"] = "linux"
    with pytest.raises(gen.MatrixGenerationError, match="execution_os_family_mismatch"):
        _build(gen, windows=windows)


def test_wrong_adapter_identity_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["interaction_proof"]["adapter_id"] = "wrong.adapter"
    with pytest.raises(gen.MatrixGenerationError, match="adapter_id_mismatch"):
        _build(gen, windows=windows)


def test_wrong_source_identity_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["interaction_proof"]["source"] = "wrong_source"
    with pytest.raises(gen.MatrixGenerationError, match="source_mismatch"):
        _build(gen, linux=linux)


def test_wrong_wrapper_runtime_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["interaction_proof"]["wrapper_runtime"] = "bash"
    with pytest.raises(gen.MatrixGenerationError, match="wrapper_runtime_mismatch"):
        _build(gen, linux=linux)


def test_false_receipt_flag_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["application_hosting_proof"]["receipt_verified"] = False
    with pytest.raises(gen.MatrixGenerationError, match="receipt_verified"):
        _build(gen, windows=windows)


def test_blank_proof_id_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["application_hosting_proof"]["proof_id"] = "   "
    with pytest.raises(gen.MatrixGenerationError, match="proof_id"):
        _build(gen, windows=windows)


def test_blank_run_id_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["interaction_proof"]["run_id"] = ""
    with pytest.raises(gen.MatrixGenerationError, match="run_id"):
        _build(gen, linux=linux)


def test_blank_correlation_id_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["interaction_proof"]["correlation_id"] = ""
    with pytest.raises(gen.MatrixGenerationError, match="correlation_id"):
        _build(gen, linux=linux)


def test_full_core_claim_in_windows_evidence_fails(gen: ModuleType) -> None:
    windows = _load_json(_WINDOWS_SRC)
    windows["full_core_platform_proof_certified_by_this_run"] = True
    with pytest.raises(gen.MatrixGenerationError, match="full_core_claim_rejected"):
        _build(gen, windows=windows)


def test_full_core_claim_in_linux_evidence_fails(gen: ModuleType) -> None:
    linux = _load_json(_LINUX_SRC)
    linux["full_core_platform_proof_certified"] = True
    with pytest.raises(gen.MatrixGenerationError, match="full_core_claim_rejected"):
        _build(gen, linux=linux)


def test_linux_native_cannot_become_certified_without_evidence(gen: ModuleType) -> None:
    matrix = _build(gen)
    profile = matrix["profiles"]["linux_native_runtime"]
    assert profile["evidence_available"] is False
    assert profile["application_hosting_status"] != "live_certified"
    assert profile["certification_result"] == "NOT_CERTIFIED"
    # Mutating after build must not be how certification is claimed.
    mutated = copy.deepcopy(profile)
    mutated["application_hosting_status"] = "live_certified"
    assert matrix["profiles"]["linux_native_runtime"]["evidence_available"] is False
    assert matrix["claims"]["native_linux_host_live_certified"] is False


def test_macos_cannot_become_certified_without_evidence(gen: ModuleType) -> None:
    matrix = _build(gen)
    profile = matrix["profiles"]["macos_native_runtime"]
    assert profile["evidence_available"] is False
    assert profile["application_hosting_status"] != "live_certified"
    assert matrix["claims"]["macos_live_certified"] is False


def test_markdown_contains_required_public_summary(gen: ModuleType) -> None:
    md = gen.render_markdown(_build(gen))
    assert (
        "The current shared LKW proof architecture is receipt-backed and "
        "live-certified"
    ) in md
    assert "on native Windows and in a Linux Docker runtime" in md
    assert "Native Linux host and macOS" in md
    assert "runtime certification remain pending." in md


def test_markdown_states_hosting_is_not_full_core(gen: ModuleType) -> None:
    md = gen.render_markdown(_build(gen))
    assert (
        "Application Hosting certification is not the same as complete multi-phase"
    ) in md
    assert "Core Platform Proof certification." in md


def test_markdown_references_both_source_artifacts(gen: ModuleType) -> None:
    md = gen.render_markdown(_build(gen))
    assert "docs/project/maintainers/public-adoption/evidence/LKW_WINDOWS_NATIVE_CERTIFICATION.json" in md
    assert "docs/project/maintainers/public-adoption/evidence/LKW_LINUX_DOCKER_CERTIFICATION.json" in md
    assert "docs/project/maintainers/public-adoption/evidence/LKW_PLATFORM_CERTIFICATION_MATRIX.json" in md


def test_no_secrets_in_json_or_markdown(gen: ModuleType) -> None:
    matrix = _build(gen)
    json_text = gen.serialize_matrix_json(matrix)
    md_text = gen.render_markdown(matrix)
    for text in (json_text.lower(), md_text.lower()):
        for needle in _SECRET_NEEDLES:
            assert needle not in text


def test_committed_matrix_matches_generator(gen: ModuleType) -> None:
    committed = _load_json(_MATRIX_JSON)
    _matrix, json_text, md_text = gen.generate_artifacts(
        repo_root=_REPO_ROOT,
        generated_from_commit=committed["generated_from_commit"],
        generated_at_utc=committed["generated_at_utc"],
    )
    assert _MATRIX_JSON.read_text(encoding="utf-8") == json_text
    assert _MATRIX_MD.read_text(encoding="utf-8") == md_text
    assert gen.main(["--check"]) == 0


def test_generator_script_does_not_hardcode_proof_ids() -> None:
    source = _GEN.read_text(encoding="utf-8")
    windows = _load_json(_WINDOWS_SRC)
    linux = _load_json(_LINUX_SRC)
    for proof_id in (
        windows["application_hosting_proof"]["proof_id"],
        windows["interaction_proof"]["proof_id"],
        linux["application_hosting_proof"]["proof_id"],
        linux["interaction_proof"]["proof_id"],
    ):
        assert proof_id not in source
