# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Topology-specific runtime materialization adapters (AP-8 §19)."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from intergrax.agent_distribution.errors import MaterializationError, MaterializationUnsupportedTopology
from intergrax.agent_distribution.materialization import MaterializationInput, MaterializationOutput
from intergrax.agent_distribution.package_artifact_provider import PackageArtifactProvider
from intergrax.agent_distribution.runtime_context_staging import (
    ARTIFACTS_STAGING_DIR,
    RUNTIME_GRAPH_MANIFEST_FILENAME,
    RUNTIME_INSTALL_MANIFEST_FILENAME,
    directory_content_digest,
    resolve_safe_path,
    stage_graph_authorized_context,
    validate_materialization_output_root,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology

DockerBuildRunner = Callable[[Sequence[str], Path, str], str]


class RuntimeMaterializationAdapter(Protocol):
    """Explicit topology adapter port — no dynamic discovery (§19)."""

    @property
    def topology(self) -> MaterializationTopology:
        """Supported materialization topology."""

    @property
    def materializer_id(self) -> str:
        """Stable adapter identity for audit."""

    @property
    def materializer_version(self) -> str:
        """Stable adapter version for audit."""

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        """Produce one immutable candidate artifact from logical inputs."""


def _candidate_output_dir(build_context_output_root: str, runtime_revision_id: str) -> Path:
    output_root = Path(build_context_output_root).resolve()
    candidate_name = f"candidate-{runtime_revision_id}"
    if ".." in Path(candidate_name).parts:
        raise MaterializationError("candidate directory traversal rejected")
    return output_root / candidate_name


def _render_minimal_oci_dockerfile(*, application_package: str, entrypoint_module: str | None) -> str:
    target = entrypoint_module or f"{application_package}.host.main:app"
    return (
        "# syntax=docker/dockerfile:1\n"
        "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder\n"
        "WORKDIR /app\n"
        f"COPY {RUNTIME_INSTALL_MANIFEST_FILENAME} ./\n"
        f"COPY {RUNTIME_GRAPH_MANIFEST_FILENAME} ./\n"
        f"COPY {'.intergrax-runtime-lock.json'} ./\n"
        f"COPY {ARTIFACTS_STAGING_DIR}/ ./{ARTIFACTS_STAGING_DIR}/\n"
        "COPY intergrax/ ./intergrax/\n"
        "COPY applications/ ./applications/\n"
        "RUN uv venv /app/.venv && "
        "UV_PROJECT_ENVIRONMENT=/app/.venv "
        "uv pip install --python /app/.venv/bin/python --no-deps "
        f"-r {RUNTIME_INSTALL_MANIFEST_FILENAME}\n"
        "FROM python:3.12-slim-bookworm AS runtime\n"
        "WORKDIR /app\n"
        "COPY --from=builder /app /app\n"
        f'CMD ["/app/.venv/bin/uvicorn", "{target}", "--host", "0.0.0.0", "--port", "8000"]\n'
    )


def _default_docker_build_runner(
    args: Sequence[str],
    cwd: Path,
    image_ref: str,
) -> str:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise MaterializationError(
            "docker build failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        )
    inspect = subprocess.run(
        ["docker", "inspect", "--format={{index .RepoDigests 0}}", image_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    digest = inspect.stdout.strip()
    if inspect.returncode == 0 and digest:
        return _normalize_image_digest(digest)
    image_id = subprocess.run(
        ["docker", "inspect", "--format={{.Id}}", image_ref],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    image_id_text = image_id.stdout.strip()
    if image_id.returncode != 0 or not image_id_text:
        raise MaterializationError("docker build succeeded but image digest unavailable")
    return _normalize_image_digest(image_id_text)


def _normalize_image_digest(image_ref: str) -> str:
    ref = image_ref.strip()
    if "@sha256:" in ref:
        digest = ref.rsplit("@sha256:", 1)[1]
        return f"sha256:{digest.lower()}"
    if ref.startswith("sha256:"):
        return ref.lower()
    raise MaterializationError("artifact digest unavailable from docker output")


@dataclass(frozen=True)
class FakeRuntimeMaterializationAdapter:
    """Deterministic in-memory/fake adapter for coordinator tests."""

    package_artifact_provider: PackageArtifactProvider
    materializer_id: str = "intergrax.test-materializer"
    materializer_version: str = "1.0.0"

    @property
    def topology(self) -> MaterializationTopology:
        return MaterializationTopology.OCI_IMAGE

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        build_context = materialization_input.application_build_context
        source_root = Path(build_context.source_context_root).resolve()
        candidate_dir = _candidate_output_dir(
            build_context.output_root,
            materialization_input.runtime_revision.runtime_revision_id,
        )
        validate_materialization_output_root(
            output_root=candidate_dir.parent,
            source_context_root=source_root,
        )
        stage_graph_authorized_context(
            build_context=build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=build_context.application_release_id,
            candidate_dir=candidate_dir,
            package_artifact_provider=self.package_artifact_provider,
        )
        digest = directory_content_digest(candidate_dir)
        manifest_path = RUNTIME_GRAPH_MANIFEST_FILENAME
        return MaterializationOutput(
            materialization_artifact_digest=digest,
            artifact_locator=f"test://{candidate_dir.as_posix()}",
            health_check_evidence_ref=f"test://build-evidence/{digest}",
            runtime_graph_manifest_path=manifest_path,
            topology=self.topology,
        )


@dataclass(frozen=True)
class OciImageMaterializationAdapter:
    """Production OCI adapter reusing graph-authoritative staging (§19.2)."""

    package_artifact_provider: PackageArtifactProvider
    materializer_id: str = "intergrax.oci-image-materializer"
    materializer_version: str = "1.0.0"
    docker_build_runner: DockerBuildRunner | None = None
    docker_timeout_seconds: int = 600

    @property
    def topology(self) -> MaterializationTopology:
        return MaterializationTopology.OCI_IMAGE

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        build_context = materialization_input.application_build_context
        source_root = Path(build_context.source_context_root).resolve()
        candidate_dir = _candidate_output_dir(
            build_context.output_root,
            materialization_input.runtime_revision.runtime_revision_id,
        )
        validate_materialization_output_root(
            output_root=candidate_dir.parent,
            source_context_root=source_root,
        )
        stage_graph_authorized_context(
            build_context=build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=build_context.application_release_id,
            candidate_dir=candidate_dir,
            package_artifact_provider=self.package_artifact_provider,
        )

        app_rel = resolve_safe_path(source_root, build_context.application_source_root)
        app_name = app_rel.name
        dockerfile = _render_minimal_oci_dockerfile(
            application_package=app_name,
            entrypoint_module=build_context.entrypoint_module,
        )
        (candidate_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")

        image_tag = build_context.docker_image_tag or (
            f"intergrax/{app_name}:candidate-{materialization_input.runtime_revision.runtime_revision_id}"
        )
        if any(char in image_tag for char in (";", "`", "$", "\n")):
            raise MaterializationError("malformed docker image tag rejected")

        runner = self.docker_build_runner or _default_docker_build_runner
        artifact_digest = runner(
            [
                "docker",
                "build",
                "-f",
                "Dockerfile",
                "-t",
                image_tag,
                ".",
            ],
            candidate_dir,
            image_tag,
        )
        return MaterializationOutput(
            materialization_artifact_digest=artifact_digest,
            artifact_locator=image_tag,
            health_check_evidence_ref=f"oci://build-context/{directory_content_digest(candidate_dir)}",
            runtime_graph_manifest_path=RUNTIME_GRAPH_MANIFEST_FILENAME,
            topology=self.topology,
        )


@dataclass(frozen=True)
class UnsupportedVenvBundleMaterializationAdapter:
    """Explicit unsupported-capability port for venv bundle topology."""

    materializer_id: str = "intergrax.venv-bundle-materializer"
    materializer_version: str = "0.0.0"

    @property
    def topology(self) -> MaterializationTopology:
        return MaterializationTopology.VENV_BUNDLE

    def materialize(self, materialization_input: MaterializationInput) -> MaterializationOutput:
        raise MaterializationUnsupportedTopology(
            "VENV_BUNDLE materialization is not supported until deterministic lock-driven "
            "installation is available"
        )


def default_materialization_adapters(
    *,
    package_artifact_provider: PackageArtifactProvider,
    docker_build_runner: DockerBuildRunner | None = None,
) -> Mapping[MaterializationTopology, RuntimeMaterializationAdapter]:
    """Register explicit topology adapters for coordinator wiring."""
    oci = OciImageMaterializationAdapter(
        package_artifact_provider=package_artifact_provider,
        docker_build_runner=docker_build_runner,
    )
    venv = UnsupportedVenvBundleMaterializationAdapter()
    return {
        MaterializationTopology.OCI_IMAGE: oci,
        MaterializationTopology.VENV_BUNDLE: venv,
    }


def new_candidate_workspace(output_root: Path) -> Path:
    """Create one bounded candidate directory under ``output_root``."""
    candidate = output_root / f"candidate-{uuid.uuid4().hex}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
