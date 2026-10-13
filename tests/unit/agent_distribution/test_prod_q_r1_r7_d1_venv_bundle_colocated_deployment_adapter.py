# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-D1 — VenvBundleColocatedRuntimeDeploymentAdapter qualification."""

from __future__ import annotations

import ast
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.deployment import (
    DrainPolicy,
    FakeInMemoryRuntimeDeploymentAdapter,
    RuntimeDeploymentAdapter,
    ServingUnitAttestationError,
    ServingUnitAttestationRequest,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_resource import (
    ColocatedVenvRuntimeResource,
)
from intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (
    VenvBundleColocatedDeploymentError,
    VenvBundleColocatedRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.venv_bundle_deployment_artifact import (
    VenvBundleDeploymentArtifactError,
    resolve_venv_bundle_artifact_directory,
)
from testing_support.agent_distribution.venv_bundle_deployment_fixtures import (
    file_artifact_locator,
    write_minimal_venv_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app-a"
_ENV = "env-prod"
def _revision(
    revision_id: str,
    *,
    artifact_digest: str,
    lock_id: str,
    lock_digest: str,
    application_id: str = _APP,
    environment: str = _ENV,
    topology: MaterializationTopology = MaterializationTopology.VENV_BUNDLE,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=application_id,
        application_environment_id=environment,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id=lock_id,
        materialized_runtime_lock_digest=lock_digest,
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest=artifact_digest,
        materialization_topology=topology,
        revision_state=RuntimeRevisionState.VALIDATED,
    )


def _bundle(tmp_path: Path) -> tuple[Path, str, str, MaterializedRuntimeLock]:
    root = tmp_path / "bundle"
    root.mkdir()
    digest, lock = write_minimal_venv_bundle(root)
    return root, digest, file_artifact_locator(root), lock


def _revision_for(
    revision_id: str,
    *,
    digest: str,
    lock: MaterializedRuntimeLock,
    application_id: str = _APP,
    environment: str = _ENV,
    topology: MaterializationTopology = MaterializationTopology.VENV_BUNDLE,
) -> RuntimeRevision:
    return _revision(
        revision_id,
        artifact_digest=digest,
        lock_id=lock.lock_id,
        lock_digest=lock.lock_digest,
        application_id=application_id,
        environment=environment,
        topology=topology,
    )


def _prepare_ready(
    adapter: VenvBundleColocatedRuntimeDeploymentAdapter,
    revision: RuntimeRevision,
    locator: str,
    *,
    authorize_admission: bool = False,
) -> str:
    prepared = adapter.prepare(revision, artifact_locator=locator)
    adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    if authorize_admission:
        adapter.authorize_serving_execution_admission(prepared.serving_unit_ref)
    return prepared.serving_unit_ref


def test_prepare_immutable_venv_bundle(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-1", digest=digest, lock=lock)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    assert prepared.materialization_artifact_digest == digest
    evidence = adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    assert evidence.startswith("venv-colocated-readiness:")


def test_unsupported_topology_rejected(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for(
        "rev-oci",
        digest=digest,
        lock=lock,
        topology=MaterializationTopology.OCI_IMAGE,
    )
    with pytest.raises(VenvBundleColocatedDeploymentError, match="VENV_BUNDLE"):
        adapter.prepare(revision, artifact_locator=locator)


def test_missing_artifact_rejected(tmp_path: Path) -> None:
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    missing = tmp_path / "missing"
    locator = file_artifact_locator(missing)
    _, digest, _, lock = _bundle(tmp_path)
    revision = _revision_for("rev-m", digest=digest, lock=lock)
    with pytest.raises(VenvBundleColocatedDeploymentError):
        adapter.prepare(revision, artifact_locator=locator)


def test_wrong_digest_rejected(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-wd", digest="sha256:" + ("b" * 64), lock=lock)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="digest mismatch"):
        adapter.prepare(revision, artifact_locator=locator)


def test_wrong_locator_scheme_production_mode(tmp_path: Path) -> None:
    root, digest, _, lock = _bundle(tmp_path)
    with pytest.raises(VenvBundleDeploymentArtifactError, match="test://"):
        resolve_venv_bundle_artifact_directory(
            f"test://{root.as_posix()}",
            allow_qualification_locators=False,
        )


def test_untrusted_artifact_path_rejected(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path / "allowed"],
    )
    revision = _revision_for("rev-trust", digest=digest, lock=lock)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="trusted"):
        adapter.prepare(revision, artifact_locator=locator)


def test_unit_without_traffic_admission_before_readiness(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-adm", digest=digest, lock=lock)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="resources are invalid"):
        adapter.admit_unit_execution(prepared.serving_unit_ref)
    adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.admit_unit_execution(prepared.serving_unit_ref)
    adapter.authorize_serving_execution_admission(prepared.serving_unit_ref)
    adapter.admit_unit_execution(prepared.serving_unit_ref)
    adapter.release_unit_execution(prepared.serving_unit_ref)


def test_readiness_fails_before_check_readiness(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-rdy", digest=digest, lock=lock)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    request = ServingUnitAttestationRequest(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-rdy",
        serving_unit_ref=prepared.serving_unit_ref,
        materialization_artifact_digest=digest,
        artifact_locator=locator,
    )
    with pytest.raises(ServingUnitAttestationError, match="not live"):
        adapter.attest_serving_unit(revision, request=request)


def test_valid_unit_attests(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-att", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator)
    attestation = adapter.attest_serving_unit(
        revision,
        request=ServingUnitAttestationRequest(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-att",
            serving_unit_ref=unit_ref,
            materialization_artifact_digest=digest,
            artifact_locator=locator,
        ),
    )
    assert attestation.physical_unit_live is True


def test_unknown_unit_attestation_fails(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-unk", digest=digest, lock=lock)
    with pytest.raises(ServingUnitAttestationError, match="unknown"):
        adapter.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id=_APP,
                application_environment_id=_ENV,
                runtime_revision_id="rev-unk",
                serving_unit_ref="venv-colocated:deadbeef",
                materialization_artifact_digest=digest,
                artifact_locator=locator,
            ),
        )


def test_attestation_identity_mismatches(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-id", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator)
    with pytest.raises(ServingUnitAttestationError, match="application identity"):
        adapter.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id="other-app",
                application_environment_id=_ENV,
                runtime_revision_id="rev-id",
                serving_unit_ref=unit_ref,
                materialization_artifact_digest=digest,
                artifact_locator=locator,
            ),
        )


def test_restart_empty_adapter_rejects_stale_handle(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter1 = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-rs", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter1, revision, locator)
    adapter2 = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    with pytest.raises(ServingUnitAttestationError, match="unknown"):
        adapter2.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id=_APP,
                application_environment_id=_ENV,
                runtime_revision_id="rev-rs",
                serving_unit_ref=unit_ref,
                materialization_artifact_digest=digest,
                artifact_locator=locator,
            ),
        )


def test_repeated_prepare_single_physical_unit(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-dup", digest=digest, lock=lock)
    first = adapter.prepare(revision, artifact_locator=locator)
    second = adapter.prepare(revision, artifact_locator=locator)
    assert first.serving_unit_ref == second.serving_unit_ref


def test_concurrent_prepare_stop_attest(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-conc", digest=digest, lock=lock)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            unit = _prepare_ready(adapter, revision, locator)
            adapter.attest_serving_unit(
                revision,
                request=ServingUnitAttestationRequest(
                    application_id=_APP,
                    application_environment_id=_ENV,
                    runtime_revision_id="rev-conc",
                    serving_unit_ref=unit,
                    materialization_artifact_digest=digest,
                    artifact_locator=locator,
                ),
            )
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker) for _ in range(4)]
        for future in futures:
            future.result()
    assert not errors


def test_begin_drain_blocks_admission(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-drain", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator, authorize_admission=True)
    adapter.admit_unit_execution(unit_ref)
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="not admitting"):
        adapter.admit_unit_execution(unit_ref)
    adapter.release_unit_execution(unit_ref)


def test_drain_completion_reflects_in_flight(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-inflight", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator, authorize_admission=True)
    adapter.admit_unit_execution(unit_ref)
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    policy = DrainPolicy(timeout_seconds=30.0)
    assert adapter.check_drain(
        revision, serving_unit_ref=unit_ref, policy=policy
    ).completed is False
    adapter.release_unit_execution(unit_ref)
    assert adapter.check_drain(
        revision, serving_unit_ref=unit_ref, policy=policy
    ).completed is True


def test_drain_timeout_evidence(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-to", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator, authorize_admission=True)
    adapter.admit_unit_execution(unit_ref)
    adapter.begin_drain(revision, serving_unit_ref=unit_ref)
    policy = DrainPolicy(timeout_seconds=0.01)
    time.sleep(0.02)
    status = adapter.check_drain(revision, serving_unit_ref=unit_ref, policy=policy)
    assert status.completed is False
    assert status.evidence_ref is not None and "timeout" in status.evidence_ref


def test_stop_invalidates_unit(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-stop", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator)
    adapter.stop(revision, serving_unit_ref=unit_ref)
    with pytest.raises(ServingUnitAttestationError):
        adapter.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id=_APP,
                application_environment_id=_ENV,
                runtime_revision_id="rev-stop",
                serving_unit_ref=unit_ref,
                materialization_artifact_digest=digest,
                artifact_locator=locator,
            ),
        )


def test_resume_does_not_resurrect_stopped_unit(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-res", digest=digest, lock=lock)
    unit_ref = _prepare_ready(adapter, revision, locator)
    adapter.stop(revision, serving_unit_ref=unit_ref)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="stopped"):
        adapter.resume_serving(revision, serving_unit_ref=unit_ref)


def test_alternate_adapter_satisfies_protocol() -> None:
    alt: RuntimeDeploymentAdapter = FakeInMemoryRuntimeDeploymentAdapter()
    assert hasattr(alt, "attest_serving_unit")


def test_runtime_initialization_failure_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-init-fail", digest=digest, lock=lock)

    def _fail_initialize(self: ColocatedVenvRuntimeResource) -> str:
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(ColocatedVenvRuntimeResource, "initialize", _fail_initialize)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    with pytest.raises(VenvBundleColocatedDeploymentError, match="warmup failed"):
        adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    with pytest.raises(ServingUnitAttestationError, match="unknown"):
        adapter.attest_serving_unit(
            revision,
            request=ServingUnitAttestationRequest(
                application_id=_APP,
                application_environment_id=_ENV,
                runtime_revision_id="rev-init-fail",
                serving_unit_ref=prepared.serving_unit_ref,
                materialization_artifact_digest=digest,
                artifact_locator=locator,
            ),
        )


def test_runtime_initialization_success_evidence(tmp_path: Path) -> None:
    root, digest, locator, lock = _bundle(tmp_path)
    adapter = VenvBundleColocatedRuntimeDeploymentAdapter(
        trusted_artifact_parent_roots=[tmp_path],
    )
    revision = _revision_for("rev-init-ok", digest=digest, lock=lock)
    prepared = adapter.prepare(revision, artifact_locator=locator)
    evidence = adapter.check_readiness(revision, serving_unit_ref=prepared.serving_unit_ref)
    assert "venv-colocated-runtime:" in evidence


def test_deployment_port_has_no_applications_import() -> None:
    source = Path("intergrax/agent_distribution/deployment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax.applications")
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.applications")
