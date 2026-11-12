# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R027: the demo attestation signing seed is reachable only from lab and proof paths."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.applications._shared.security_assembly_resolver import (
    HOST_ATTESTATION_SIGNING_KEY_ENV,
    validate_strict_security_configuration,
)
from intergrax.applications.contracts.environment_profile import ExecutionBoundaryExportProfile
from intergrax.core.security.security_configuration import (
    DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
    SecurityConfigurationError,
)
from intergrax.runtime.execution_evidence import attestor as attestor_module
from intergrax.runtime.execution_evidence.attestor import (
    build_deterministic_test_attestor,
    resolve_demo_attestation_seed,
)

from ._support import REPO_ROOT, generated_secret, lab_env, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SEED_CONSTANT = "POC_ATTESTATION_DEMO_SIGNING_SEED"
_BUILDER = "build_deterministic_test_attestor"
_SIGNING_ENV = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"

# Explicit lab / proof / demo paths that may reference the seed or the lab attestor builder.
_LAB_PROOF_PATHS = frozenset(
    {
        "intergrax/runtime/attestation/host_attestation.py",
        "intergrax/runtime/execution_evidence/attestor.py",
        "intergrax/runtime/execution_evidence/__init__.py",
        "intergrax/cli/external_work.py",
        "applications/governed_contractor_application/host/cli_commands.py",
        "applications/governed_contractor_application/host/offline_demo.py",
    }
)
_LAB_PROOF_PREFIXES = ("applications/attestation_demo/",)

# OPEN ITEM (outside the B-D6 write set): this product wiring module calls
# ``resolve_host_attestation_sealer_from_env``, which falls back to the demo seed when the host
# signing key variable is unset. It is pinned here so that any NEW product-wiring reference is
# caught; the strict assembly check below refuses that configuration until the fallback in
# ``host_attestation.py`` is removed by the integration owner.
_KNOWN_OPEN_PRODUCT_WIRING = frozenset({"intergrax/applications/_shared/attestation_runtime_bridge.py"})


def _product_files() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for root in ("intergrax", "agents", "applications"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            found.append((relative, path.read_text(encoding="utf-8")))
    return found


def _imports_host_attestation(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("host_attestation"):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name.endswith("host_attestation") for alias in node.names):
                return True
    return False


def _is_lab_proof_path(relative: str) -> bool:
    return relative in _LAB_PROOF_PATHS or relative.startswith(_LAB_PROOF_PREFIXES)


def test_d6_r027_seed_and_lab_builder_are_referenced_only_from_lab_proof_paths() -> None:
    offenders = [
        relative
        for relative, text in _product_files()
        if (_SEED_CONSTANT in text or _BUILDER in text) and not _is_lab_proof_path(relative)
    ]
    assert offenders == []


def test_d6_r027_product_wiring_has_no_new_reference_to_the_seed() -> None:
    offenders = {
        relative
        for relative, text in _product_files()
        if relative.startswith("intergrax/applications/_shared/")
        and (_SEED_CONSTANT in text or _BUILDER in text or _imports_host_attestation(text))
    }
    assert offenders == set(_KNOWN_OPEN_PRODUCT_WIRING)


def test_d6_r027_attestor_module_does_not_import_the_seed_at_module_level() -> None:
    tree = ast.parse(Path(attestor_module.__file__).read_text(encoding="utf-8"))
    for node in tree.body:  # module level only
        if isinstance(node, ast.ImportFrom):
            assert node.module != "intergrax.runtime.attestation.host_attestation"
            assert all(alias.name != _SEED_CONSTANT for alias in node.names)
    assert not hasattr(attestor_module, _SEED_CONSTANT)


def test_d6_r027_strict_profile_cannot_resolve_the_demo_seed() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        resolve_demo_attestation_seed(strict_profile=True)
    assert excinfo.value.reason_code == DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE
    with pytest.raises(SecurityConfigurationError):
        build_deterministic_test_attestor(strict_profile=True)


def test_d6_r027_strict_profile_with_explicit_key_material_still_works() -> None:
    attestor = build_deterministic_test_attestor(seed=bytes(range(32)), strict_profile=True)
    assert attestor.public_key_bytes


def test_d6_r027_lab_and_proof_paths_keep_working_through_the_named_builder() -> None:
    attestor = build_deterministic_test_attestor(key_id="lab-proof-1")
    attestation = attestor.attest(b"payload", schema="proof.v1")
    assert attestation.key_id == "lab-proof-1"
    assert len(resolve_demo_attestation_seed(strict_profile=False)) == 32


def _with_host_signing(env):
    env.execution_boundary_export_profile = ExecutionBoundaryExportProfile(enabled=True, host_signing_enabled=True)
    return env


def test_d6_r027_strict_assembly_refuses_host_signing_that_would_fall_back_to_the_seed() -> None:
    env = _with_host_signing(product_env())
    issues = validate_strict_security_configuration(
        env,
        environ={_SIGNING_ENV: generated_secret()},
        audit_evidence_persistence=object(),  # type: ignore[arg-type]
    )
    assert [issue.reason_code for issue in issues] == [DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE]
    assert issues[0].item == "host_attestation"


def test_d6_r027_strict_assembly_accepts_host_signing_with_an_explicit_key() -> None:
    env = _with_host_signing(product_env())
    issues = validate_strict_security_configuration(
        env,
        environ={_SIGNING_ENV: generated_secret(), HOST_ATTESTATION_SIGNING_KEY_ENV: generated_secret()},
        audit_evidence_persistence=object(),  # type: ignore[arg-type]
    )
    assert issues == ()


def test_d6_r027_lab_profile_is_not_constrained() -> None:
    env = _with_host_signing(lab_env())
    assert validate_strict_security_configuration(env, environ={}) == ()
