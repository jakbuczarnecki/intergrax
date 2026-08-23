# Â© Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
    PlatformProofDescriptor,
    ProofLibraryClass,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import (
    DescriptorLoadError,
    descriptor_to_manifest_entry,
    load_descriptor,
    normalize_to_manifest_entry,
)
from scripts.proof.intergrax_proof_contracts import (
    EnvRequirementKind,
    ProofProfile,
    ProofSafetyClass,
)
_MINIMAL_MECHANISMS = ("tools.sample_mechanism",)
_TEST_DOMAIN = "test_domain"
_SAMPLE_PROOF_ID = "TEST-DOMAIN-SAMPLE"


def _write_descriptor_package(
    tmp_path: Path,
    *,
    slug: str = "sample",
    payload: dict[str, object] | None = None,
    write_run_proof: bool = True,
) -> Path:
    package = tmp_path / "platform_proofs" / _TEST_DOMAIN / slug
    package.mkdir(parents=True, exist_ok=True)
    if write_run_proof:
        (package / "run_proof.py").write_text("print('ok')\n", encoding="utf-8")
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(payload or _minimal_conformance_descriptor()),
        encoding="utf-8",
    )
    return descriptor_path


def _full_conformance_descriptor_payload() -> dict[str, object]:
    return _minimal_conformance_descriptor(
        proof_id="TEST-DOMAIN-FULL",
        domains_exercised=(_TEST_DOMAIN,),
        profiles=["full", "live"],
        timeout_seconds=3600,
        safety_class="LOCAL_MUTATING",
        evidence_schema="intergrax.platform_proof_evidence.v2",
        report_required=True,
        environment_requirements=[
            {"kind": "COMMAND_AVAILABLE", "name": "uv"},
            {"kind": "DOCKER_AVAILABLE", "name": "docker"},
            {"kind": "ENV_PRESENT", "name": "INTERGRAX_LLM_PROVIDER"},
        ],
        evidence_required=True,
        expected_artifacts=[
            {"kind": "EVIDENCE_JSON", "relative_path": "evidence.json", "required": True},
            {"kind": "DOMAIN_RESULT_JSON", "relative_path": "proof-result.json", "required": True},
            {"kind": "REPORT_HTML", "relative_path": "report.html", "required": True},
        ],
        command={
            "executable": "uv",
            "argv": [
                "run",
                "python",
                f"platform_proofs/{_TEST_DOMAIN}/full_proof/run_proof.py",
            ],
        },
    )


def _minimal_conformance_descriptor(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": ProofLibraryClass.CONFORMANCE.value,
        "proof_id": _SAMPLE_PROOF_ID,
        "title": "sample",
        "domains_exercised": [_TEST_DOMAIN],
        "proof_kind": "sample",
        "mechanisms_exercised": list(_MINIMAL_MECHANISMS),
        "package_version": "1.0.0",
        "profiles": ["full"],
        "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
    }
    base.update(overrides)
    return base


def _minimal_scenario_descriptor(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": ProofLibraryClass.SCENARIO.value,
        "proof_id": "SCENARIO-SAMPLE",
        "title": "scenario sample",
        "domains_exercised": ["tools"],
        "proof_kind": "scenario_sample",
        "mechanisms_exercised": ["tools.sample_mechanism"],
        "package_version": "1.0.0",
        "profiles": ["full"],
        "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
        "problem_category": "data_integrity",
        "problem_summary": "Investigation must not conclude without evidence.",
        "failure_mode_summary": "Premature conclusion from a single SQL observation.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_full_descriptor_parses(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        slug="full_proof",
        payload=_full_conformance_descriptor_payload(),
    )
    descriptor = load_descriptor(descriptor_path, repo_root=tmp_path)
    assert descriptor.schema_version == PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.library_class is ProofLibraryClass.CONFORMANCE
    assert descriptor.proof_id == "TEST-DOMAIN-FULL"
    assert descriptor.domains_exercised == (_TEST_DOMAIN,)
    assert descriptor.package_version == "1.0.0"
    assert descriptor.evidence_schema == "intergrax.platform_proof_evidence.v2"
    assert descriptor.report_required is True


def test_v1_descriptor_rejected(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        payload=_minimal_conformance_descriptor(
            schema_version="intergrax.platform_proof_descriptor.v1",
            public_evidence_eligible=False,
        ),
    )
    with pytest.raises((DescriptorLoadError, ValidationError)):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_v2_descriptor_rejected(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        payload=_minimal_conformance_descriptor(
            schema_version="intergrax.platform_proof_descriptor.v2",
        ),
    )
    with pytest.raises((DescriptorLoadError, ValidationError)):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_legacy_domain_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PlatformProofDescriptor.model_validate(
            {**_minimal_conformance_descriptor(), "domain": _TEST_DOMAIN}
        )


def test_primary_domain_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PlatformProofDescriptor.model_validate(
            {**_minimal_conformance_descriptor(), "primary_domain": _TEST_DOMAIN}
        )


def test_empty_domains_exercised_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(domains_exercised=[])
        )


def test_empty_domain_element_rejected() -> None:
    with pytest.raises(ValidationError, match="empty values"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(domains_exercised=[" "])
        )


def test_duplicate_domains_exercised_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(
                domains_exercised=[_TEST_DOMAIN, _TEST_DOMAIN]
            )
        )


def test_unsafe_domain_identifier_rejected() -> None:
    with pytest.raises(ValidationError, match="unsafe"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(domains_exercised=["../escape"])
        )


def test_multiple_domains_exercised_accepted() -> None:
    descriptor = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            domains_exercised=["tools", "runtime"]
        )
    )
    assert descriptor.domains_exercised == ("runtime", "tools")


def test_single_domain_exercised_accepted() -> None:
    descriptor = PlatformProofDescriptor.model_validate(_minimal_conformance_descriptor())
    assert descriptor.domains_exercised == (_TEST_DOMAIN,)


def test_domains_exercised_canonical_lexicographic_order() -> None:
    descriptor_a = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            domains_exercised=["TOOLS", "EXECUTION", "OBSERVABILITY"]
        )
    )
    descriptor_b = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            domains_exercised=["OBSERVABILITY", "TOOLS", "EXECUTION"]
        )
    )
    expected = ("EXECUTION", "OBSERVABILITY", "TOOLS")
    assert descriptor_a.domains_exercised == expected
    assert descriptor_b.domains_exercised == expected
    assert "primary_domain" not in PlatformProofDescriptor.model_fields


def test_domains_exercised_trimming_before_canonical_order() -> None:
    descriptor = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            domains_exercised=["  TOOLS  ", "EXECUTION", " OBSERVABILITY "]
        )
    )
    assert descriptor.domains_exercised == ("EXECUTION", "OBSERVABILITY", "TOOLS")


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        payload=_minimal_conformance_descriptor(
            schema_version="intergrax.platform_proof_descriptor.v99",
        ),
    )
    with pytest.raises((DescriptorLoadError, ValidationError)):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PlatformProofDescriptor.model_validate(
            {**_minimal_conformance_descriptor(), "unexpected": True}
        )


def test_valid_conformance_passes() -> None:
    descriptor = PlatformProofDescriptor.model_validate(_minimal_conformance_descriptor())
    assert descriptor.library_class is ProofLibraryClass.CONFORMANCE


def test_valid_scenario_passes() -> None:
    descriptor = PlatformProofDescriptor.model_validate(_minimal_scenario_descriptor())
    assert descriptor.library_class is ProofLibraryClass.SCENARIO


def test_unknown_library_class_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(library_class="EXPERIMENTAL")
        )


def test_empty_mechanisms_exercised_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(mechanisms_exercised=[])
        )


def test_empty_mechanism_element_rejected() -> None:
    with pytest.raises(ValidationError, match="empty values"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(mechanisms_exercised=[" "])
        )


def test_duplicate_mechanisms_exercised_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(
                mechanisms_exercised=["tools.sample_mechanism", "tools.sample_mechanism"]
            )
        )


def test_valid_unique_mechanisms_exercised_passes() -> None:
    descriptor = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            mechanisms_exercised=["tools.alpha", "tools.beta"]
        )
    )
    assert descriptor.mechanisms_exercised == ("tools.alpha", "tools.beta")


def test_mechanisms_exercised_whitespace_trimmed() -> None:
    descriptor = PlatformProofDescriptor.model_validate(
        _minimal_conformance_descriptor(
            mechanisms_exercised=["  tools.alpha  ", "tools.beta"]
        )
    )
    assert descriptor.mechanisms_exercised == ("tools.alpha", "tools.beta")


@pytest.mark.parametrize(
    ("invalid_value",),
    [
        ({"foo": "bar"},),
        (123,),
        (True,),
        ("tools.sample_mechanism",),
    ],
)
def test_invalid_mechanisms_exercised_container_rejected(
    invalid_value: object,
) -> None:
    with pytest.raises(ValidationError, match="mechanisms_exercised"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(mechanisms_exercised=invalid_value)
        )


@pytest.mark.parametrize(
    ("invalid_element",),
    [
        (123,),
        (True,),
    ],
)
def test_invalid_mechanisms_exercised_element_rejected(
    invalid_element: object,
) -> None:
    with pytest.raises(ValidationError, match="only strings"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(
                mechanisms_exercised=["tools.sample_mechanism", invalid_element]
            )
        )


@pytest.mark.parametrize(
    ("missing_field",),
    [
        ("problem_category",),
        ("problem_summary",),
        ("failure_mode_summary",),
    ],
)
def test_scenario_missing_required_field_rejected(missing_field: str) -> None:
    payload = _minimal_scenario_descriptor()
    payload.pop(missing_field)
    with pytest.raises(ValidationError, match=missing_field):
        PlatformProofDescriptor.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("problem_category", "   "),
        ("problem_summary", ""),
        ("failure_mode_summary", " "),
    ],
)
def test_scenario_blank_required_field_rejected(field_name: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        PlatformProofDescriptor.model_validate(
            _minimal_scenario_descriptor(**{field_name: value})
        )


@pytest.mark.parametrize(
    "field_name",
    ["problem_category", "problem_summary", "failure_mode_summary"],
)
@pytest.mark.parametrize(
    "invalid_value",
    [123, True, ["list"], {"key": "value"}],
)
def test_scenario_problem_field_non_string_rejected(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValidationError, match="must be a string or null"):
        PlatformProofDescriptor.model_validate(
            _minimal_scenario_descriptor(**{field_name: invalid_value})
        )


@pytest.mark.parametrize(
    "field_name",
    ["problem_category", "problem_summary", "failure_mode_summary"],
)
def test_conformance_rejects_scenario_fields(field_name: str) -> None:
    with pytest.raises(ValidationError, match="forbidden"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(**{field_name: "must not appear"})
        )


def test_duplicate_profiles_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(profiles=["full", "full"])
        )


def test_malformed_env_requirement_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(
                environment_requirements=[{"kind": "ENV_PRESENT", "name": ""}],
            )
        )


def test_invalid_safety_class_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(safety_class="REMOTE_MUTATING")
        )


def test_unsafe_proof_id_rejected() -> None:
    with pytest.raises(ValidationError, match="proof_id"):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(proof_id="../escape")
        )


def test_command_string_instead_of_argv_object_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            _minimal_conformance_descriptor(
                command="uv run python foo.py && rm -rf /",
            )
        )


def test_path_traversal_rejected(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        payload=_minimal_conformance_descriptor(
            command={
                "executable": "python",
                "argv": ["../../outside.py"],
            },
        ),
    )
    with pytest.raises(DescriptorLoadError, match="traverse"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_missing_entrypoint_rejected(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        write_run_proof=False,
        payload=_minimal_conformance_descriptor(
            command={
                "executable": "python",
                "argv": [f"platform_proofs/{_TEST_DOMAIN}/sample/run_proof.py"],
            },
        ),
    )
    with pytest.raises(DescriptorLoadError, match="missing declared entrypoint"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_deterministic_normalization(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        slug="full_proof",
        payload=_full_conformance_descriptor_payload(),
    )
    first = descriptor_to_manifest_entry(descriptor_path, repo_root=tmp_path)
    second = descriptor_to_manifest_entry(descriptor_path, repo_root=tmp_path)
    assert first == second
    assert first.profiles == frozenset({ProofProfile.FULL, ProofProfile.LIVE})
    assert first.safety_class is ProofSafetyClass.LOCAL_MUTATING
    assert first.timeout_seconds == 3600


def test_descriptor_contains_no_secret_values(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(tmp_path)
    descriptor_path.write_text(
        json.dumps({**_minimal_conformance_descriptor(), "password": "must-not-appear"}),
        encoding="utf-8",
    )
    with pytest.raises(DescriptorLoadError, match="secret field"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_env_requirements_declare_names_only(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        slug="full_proof",
        payload=_full_conformance_descriptor_payload(),
    )
    descriptor = load_descriptor(descriptor_path, repo_root=tmp_path)
    for requirement in descriptor.environment_requirements:
        assert requirement.kind is EnvRequirementKind.ENV_PRESENT or requirement.kind in {
            EnvRequirementKind.COMMAND_AVAILABLE,
            EnvRequirementKind.DOCKER_AVAILABLE,
        }
        assert requirement.name
        assert "=" not in requirement.name


def test_normalize_ignores_environment_state(tmp_path: Path) -> None:
    descriptor_path = _write_descriptor_package(
        tmp_path,
        slug="full_proof",
        payload=_full_conformance_descriptor_payload(),
    )
    descriptor = load_descriptor(descriptor_path, repo_root=tmp_path)
    entry = normalize_to_manifest_entry(
        descriptor,
        package_root=descriptor_path.parent,
        repo_root=tmp_path,
    )
    assert entry.proof_id == "TEST-DOMAIN-FULL"


def test_descriptor_models_remain_frozen_and_forbid_extra() -> None:
    descriptor = PlatformProofDescriptor.model_validate(_minimal_conformance_descriptor())
    with pytest.raises(ValidationError, match="frozen"):
        descriptor.title = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        PlatformProofDescriptor.model_validate(
            {**_minimal_conformance_descriptor(), "hub_tags": ["forbidden"]}
        )


def test_committed_proof_descriptors_use_v3_contract(repo_root: Path) -> None:
    proofs_root = repo_root / "platform_proofs"
    if not proofs_root.is_dir():
        return
    for descriptor_path in sorted(proofs_root.rglob(PROOF_DESCRIPTOR_FILENAME)):
        descriptor = load_descriptor(descriptor_path, repo_root=repo_root)
        assert descriptor.schema_version == PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION
        assert descriptor.domains_exercised
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        assert "domain" not in payload
