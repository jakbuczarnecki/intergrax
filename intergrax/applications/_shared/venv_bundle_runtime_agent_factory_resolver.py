# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production ``RuntimeAgentFactoryResolver`` for ``MaterializationTopology.VENV_BUNDLE``."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
    resolve_safe_path,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
    RuntimeAgentFactoryResolver,
)
from intergrax.applications.contracts.factory import AgentFactory

PRODUCTION_RUNTIME_FACTORY_TOPOLOGY_STATUS: Final = {
    MaterializationTopology.VENV_BUNDLE: "implemented",
    MaterializationTopology.OCI_IMAGE: "deferred",
    MaterializationTopology.SANDBOX_SIDECAR: "deferred",
}

_FlatSitePackagesName = "site-packages"
_ImportPathLock = threading.RLock()
_FactoryCacheLock = threading.RLock()
_FactoryCache: dict[tuple[str, str, str | None, str | None], AgentFactory] = {}


def production_runtime_factory_topology_status(
    topology: MaterializationTopology | None,
) -> str:
    """Return ``implemented`` or ``deferred`` for one materialization topology."""
    if topology is None:
        return "deferred"
    return PRODUCTION_RUNTIME_FACTORY_TOPOLOGY_STATUS.get(topology, "deferred")


def is_production_runtime_factory_adapter_deferred(
    topology: MaterializationTopology | None = None,
) -> bool:
    """True when the given topology lacks a production factory resolver."""
    return production_runtime_factory_topology_status(topology) != "implemented"


def build_production_runtime_agent_factory_resolver(
    *,
    runtime_revision: RuntimeRevision,
    artifact_root: Path,
    expected_artifact_digest: str | None = None,
) -> RuntimeAgentFactoryResolver:
    """Construct the production resolver for one revision-bound artifact."""
    topology = runtime_revision.materialization_topology
    if topology == MaterializationTopology.VENV_BUNDLE:
        digest = expected_artifact_digest or runtime_revision.materialization_artifact_digest
        return VenvBundleRuntimeAgentFactoryResolver(
            artifact_root=artifact_root,
            expected_artifact_digest=digest,
        )
    if topology in {
        MaterializationTopology.OCI_IMAGE,
        MaterializationTopology.SANDBOX_SIDECAR,
    }:
        raise RuntimeAgentFactoryResolutionError(
            f"{topology.value} production factory resolver is deferred"
        )
    raise RuntimeAgentFactoryResolutionError(
        f"unsupported materialization topology for production factory resolver: {topology!r}"
    )


def _parse_factory_path(factory_path: str) -> tuple[str, str]:
    normalized = factory_path.strip()
    if not normalized:
        raise RuntimeAgentFactoryResolutionError("factory_path must be non-empty")
    if ":" in normalized:
        module_path, attr_name = normalized.split(":", 1)
    else:
        module_path, _, attr_name = normalized.rpartition(".")
    module_path = module_path.strip()
    attr_name = attr_name.strip()
    if not module_path or not attr_name:
        raise RuntimeAgentFactoryResolutionError(
            f"ambiguous factory_path reference: {factory_path!r}"
        )
    return module_path, attr_name


def _discover_site_packages(artifact_root: Path) -> Path:
    venv_root = artifact_root / ".venv"
    if venv_root.is_dir():
        matches = sorted(venv_root.glob("lib/python*/site-packages"))
        for candidate in matches:
            if candidate.is_dir():
                return candidate
    flat = artifact_root / _FlatSitePackagesName
    if flat.is_dir():
        return flat
    raise RuntimeAgentFactoryResolutionError(
        f"venv bundle artifact at {artifact_root!s} lacks importable site-packages layout"
    )


def _load_lock_manifest(artifact_root: Path) -> MaterializedRuntimeLock:
    manifest_path = artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RuntimeAgentFactoryResolutionError(
            f"missing runtime lock manifest in artifact: {RUNTIME_LOCK_MANIFEST_FILENAME}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeAgentFactoryResolutionError(
            f"invalid runtime lock manifest JSON in {RUNTIME_LOCK_MANIFEST_FILENAME}"
        ) from exc
    try:
        return MaterializedRuntimeLock.model_validate(payload)
    except ValueError as exc:
        raise RuntimeAgentFactoryResolutionError(
            "runtime lock manifest failed validation"
        ) from exc


def _verify_package_digest_in_lock(
    lock: MaterializedRuntimeLock,
    package_digest: str,
) -> None:
    trusted = {entry.package_digest for entry in lock.agent_closure}
    trusted.update(
        pkg.package_digest
        for pkg in lock.packages
        if pkg.package_digest is not None
    )
    if package_digest not in trusted:
        raise RuntimeAgentFactoryResolutionError(
            f"package_digest {package_digest!r} is not authorized by artifact lock"
        )


def _resolve_module_file(site_packages: Path, module_path: str) -> Path:
    relative = Path(*module_path.split("."))
    module_file = site_packages / relative.with_suffix(".py")
    package_init = site_packages / relative / "__init__.py"
    if module_file.is_file():
        target = module_file
    elif package_init.is_file():
        raise RuntimeAgentFactoryResolutionError(
            f"factory_path {module_path!r} must name a module attribute, not a package"
        )
    else:
        raise RuntimeAgentFactoryResolutionError(
            f"factory module {module_path!r} not found in artifact site-packages"
        )
    resolved = target.resolve()
    try:
        resolved.relative_to(site_packages.resolve())
    except ValueError as exc:
        raise RuntimeAgentFactoryResolutionError(
            f"factory module path escapes artifact site-packages: {module_path!r}"
        ) from exc
    return resolved


def _artifact_scope_key(artifact_digest: str) -> str:
    return artifact_digest.removeprefix("sha256:").replace("-", "")[:24]


def _load_callable_from_site_packages(
    *,
    site_packages: Path,
    module_path: str,
    attr_name: str,
    artifact_digest: str,
) -> Callable[..., Any]:
    module_file = _resolve_module_file(site_packages, module_path)
    module_key = f"_intergrax_artifact_{_artifact_scope_key(artifact_digest)}_{module_path}"
    site_str = str(site_packages.resolve())
    with _ImportPathLock:
        inserted = False
        if site_str not in sys.path:
            sys.path.insert(0, site_str)
            inserted = True
        try:
            module = sys.modules.get(module_key)
            if module is None:
                spec = importlib.util.spec_from_file_location(
                    module_key,
                    module_file,
                    submodule_search_locations=[site_str],
                )
                if spec is None or spec.loader is None:
                    raise RuntimeAgentFactoryResolutionError(
                        f"cannot create import spec for {module_path!r}"
                    )
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_key] = module
                spec.loader.exec_module(module)
            target = getattr(module, attr_name, None)
        finally:
            if inserted:
                try:
                    sys.path.remove(site_str)
                except ValueError:
                    pass
    if target is None:
        raise RuntimeAgentFactoryResolutionError(
            f"factory attribute {attr_name!r} missing from module {module_path!r}"
        )
    if not callable(target):
        raise RuntimeAgentFactoryResolutionError(
            f"factory reference {module_path}.{attr_name} is not callable"
        )
    return target


class VenvBundleRuntimeAgentFactoryResolver:
    """Load agent factories from one immutable VENV_BUNDLE artifact directory."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        expected_artifact_digest: str | None = None,
    ) -> None:
        self._artifact_root = artifact_root.resolve()
        self._expected_artifact_digest = (
            expected_artifact_digest.strip().lower()
            if expected_artifact_digest is not None
            else None
        )
        self._site_packages = _discover_site_packages(self._artifact_root)
        self._lock = _load_lock_manifest(self._artifact_root)
        self._verified_digest: str | None = None
        self._digest_lock = threading.Lock()

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def materialized_runtime_lock(self) -> MaterializedRuntimeLock:
        return self._lock

    def _ensure_artifact_digest(self) -> str:
        if self._expected_artifact_digest is None:
            raise RuntimeAgentFactoryResolutionError(
                "artifact digest evidence is required before loading executable factories"
            )
        with self._digest_lock:
            if self._verified_digest == self._expected_artifact_digest:
                return self._verified_digest
            actual = directory_content_digest(self._artifact_root)
            if actual != self._expected_artifact_digest:
                raise RuntimeAgentFactoryResolutionError(
                    "materialization artifact digest mismatch before factory load"
                )
            self._verified_digest = actual
            return actual

    def _validate_revision(self, runtime_revision: RuntimeRevision) -> None:
        if runtime_revision.materialization_topology != MaterializationTopology.VENV_BUNDLE:
            raise RuntimeAgentFactoryResolutionError(
                "VenvBundleRuntimeAgentFactoryResolver requires VENV_BUNDLE topology"
            )
        if runtime_revision.materialized_runtime_lock_digest is not None:
            lock_digest = self._lock.lock_digest
            if lock_digest is None:
                lock_digest = self._lock.with_content_identity().lock_digest
            if lock_digest != runtime_revision.materialized_runtime_lock_digest:
                raise RuntimeAgentFactoryResolutionError(
                    "artifact lock digest does not match runtime revision"
                )
        if runtime_revision.materialized_runtime_lock_id is not None:
            lock_id = self._lock.lock_id
            if lock_id is None:
                lock_id = self._lock.with_content_identity().lock_id
            if lock_id != runtime_revision.materialized_runtime_lock_id:
                raise RuntimeAgentFactoryResolutionError(
                    "artifact lock id does not match runtime revision"
                )
        try:
            resolve_safe_path(self._artifact_root, ".")
        except Exception as exc:
            raise RuntimeAgentFactoryResolutionError(
                "artifact root path is not a safe bounded directory"
            ) from exc

    def resolve_factory(
        self,
        *,
        runtime_revision: RuntimeRevision,
        package_digest: str,
        factory_reference: AgentBindingFactoryReference,
    ) -> AgentFactory:
        self._validate_revision(runtime_revision)
        digest = normalize_package_digest(package_digest)
        trusted = frozenset(runtime_revision.installed_agent_package_digests)
        if digest not in trusted:
            raise RuntimeAgentFactoryResolutionError(
                f"package_digest {digest!r} is not part of runtime revision "
                f"{runtime_revision.runtime_revision_id!r}"
            )
        _verify_package_digest_in_lock(self._lock, digest)
        if factory_reference.builder_key is not None:
            raise RuntimeAgentFactoryResolutionError(
                "builder_key factory references require a host-side builder map and are "
                "forbidden for production VENV_BUNDLE artifact authority"
            )
        factory_path = factory_reference.factory_path
        if factory_path is None:
            raise RuntimeAgentFactoryResolutionError(
                "production VENV_BUNDLE resolution requires factory_path"
            )
        artifact_digest = self._ensure_artifact_digest()
        cache_key = (
            artifact_digest,
            digest,
            factory_reference.builder_key,
            factory_reference.factory_path,
        )
        with _FactoryCacheLock:
            cached = _FactoryCache.get(cache_key)
        if cached is not None:
            return cached
        module_path, attr_name = _parse_factory_path(factory_path)
        loaded = _load_callable_from_site_packages(
            site_packages=self._site_packages,
            module_path=module_path,
            attr_name=attr_name,
            artifact_digest=artifact_digest,
        )
        with _FactoryCacheLock:
            _FactoryCache[cache_key] = loaded
        return loaded


__all__ = [
    "PRODUCTION_RUNTIME_FACTORY_TOPOLOGY_STATUS",
    "VenvBundleRuntimeAgentFactoryResolver",
    "build_production_runtime_agent_factory_resolver",
    "is_production_runtime_factory_adapter_deferred",
    "production_runtime_factory_topology_status",
]
