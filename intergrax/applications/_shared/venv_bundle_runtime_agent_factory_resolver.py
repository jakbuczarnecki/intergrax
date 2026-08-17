# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Production ``RuntimeAgentFactoryResolver`` for ``MaterializationTopology.VENV_BUNDLE``."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import threading
import types
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
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
_FactoryCacheLock = threading.RLock()
_FactoryCache: dict[tuple[str, str, str | None, str | None], AgentFactory] = {}
_ArtifactScopeRegistryLock = threading.RLock()
_ArtifactScopeRegistry: dict[str, _ArtifactImportScope] = {}
_TopLevelPackageDispatchLock = threading.RLock()
_TopLevelPackageDispatchModules: dict[str, types.ModuleType] = {}
_ScopeLoadLock = threading.RLock()
_ActiveArtifactImportScope: ContextVar[_ArtifactImportScope | None] = ContextVar(
    "_active_artifact_import_scope",
    default=None,
)


@dataclass(frozen=True, slots=True)
class _ArtifactImportScope:
    artifact_digest: str
    scope_root: str
    site_packages: Path


def _artifact_scope_root(artifact_digest: str) -> str:
    normalized = artifact_digest.removeprefix("sha256:").replace("-", "").lower()
    if not normalized:
        raise RuntimeAgentFactoryResolutionError(
            "artifact digest is required for import isolation scope"
        )
    return f"_intergrax_artifact_{normalized}"


def _scoped_module_name(scope_root: str, original_module_path: str) -> str:
    return f"{scope_root}.{original_module_path}"


def _original_module_name(scope_root: str, scoped_name: str) -> str | None:
    prefix = f"{scope_root}."
    if scoped_name == scope_root:
        return ""
    if scoped_name.startswith(prefix):
        return scoped_name[len(prefix) :]
    return None


def _resolve_artifact_module(
    site_packages: Path,
    module_path: str,
) -> tuple[Path, bool]:
    relative = Path(*module_path.split("."))
    module_file = site_packages / relative.with_suffix(".py")
    package_init = site_packages / relative / "__init__.py"
    if module_file.is_file():
        target = module_file
        is_package = False
    elif package_init.is_file():
        target = package_init
        is_package = True
    else:
        raise RuntimeAgentFactoryResolutionError(
            f"module {module_path!r} not found in artifact site-packages"
        )
    resolved = target.resolve()
    try:
        resolved.relative_to(site_packages.resolve())
    except ValueError as exc:
        raise RuntimeAgentFactoryResolutionError(
            f"module path escapes artifact site-packages: {module_path!r}"
        ) from exc
    return resolved, is_package


def _module_exists_in_artifact(site_packages: Path, module_path: str) -> bool:
    try:
        _resolve_artifact_module(site_packages, module_path)
    except RuntimeAgentFactoryResolutionError:
        return False
    return True


def _ensure_top_level_package_dispatch(package_name: str) -> types.ModuleType:
    with _TopLevelPackageDispatchLock:
        existing = _TopLevelPackageDispatchModules.get(package_name)
        if existing is not None:
            return existing
        dispatch = types.ModuleType(package_name)
        dispatch.__doc__ = "Intergrax artifact-scoped top-level package dispatch stub."
        dispatch.__path__ = []
        dispatch.__package__ = package_name
        _TopLevelPackageDispatchModules[package_name] = dispatch
        sys.modules.setdefault(package_name, dispatch)
        return dispatch


def _register_top_level_package_dispatchers(site_packages: Path) -> None:
    for entry in site_packages.iterdir():
        if not entry.is_dir() or entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if not (entry / "__init__.py").is_file():
            continue
        has_submodules = any(
            child.name != "__init__.py" and ((child.is_file() and child.suffix == ".py") or child.is_dir())
            for child in entry.iterdir()
        )
        if has_submodules:
            _ensure_top_level_package_dispatch(entry.name)


def _register_artifact_scope(*, artifact_digest: str, site_packages: Path) -> _ArtifactImportScope:
    resolved_site_packages = site_packages.resolve()
    _register_top_level_package_dispatchers(resolved_site_packages)
    scope_root = _artifact_scope_root(artifact_digest)
    with _ArtifactScopeRegistryLock:
        existing = _ArtifactScopeRegistry.get(scope_root)
        if existing is not None:
            if existing.artifact_digest != artifact_digest:
                raise RuntimeAgentFactoryResolutionError(
                    f"artifact import scope collision for {scope_root!r}"
                )
            return existing
        scope = _ArtifactImportScope(
            artifact_digest=artifact_digest,
            scope_root=scope_root,
            site_packages=resolved_site_packages,
        )
        _ArtifactScopeRegistry[scope_root] = scope
        return scope


def _scope_for_scoped_name(fullname: str) -> _ArtifactImportScope | None:
    if not fullname.startswith("_intergrax_artifact_"):
        return None
    head = fullname.split(".", 1)[0]
    with _ArtifactScopeRegistryLock:
        return _ArtifactScopeRegistry.get(head)


def _scope_for_import_target(target: object | None) -> _ArtifactImportScope | None:
    if target is None:
        return None
    target_name = getattr(target, "__name__", None)
    if not isinstance(target_name, str):
        return None
    return _scope_for_scoped_name(target_name)


def _package_search_locations(
    site_packages: Path,
    original_module_path: str,
    *,
    is_package: bool,
) -> list[str] | None:
    if not is_package:
        return None
    relative = Path(*original_module_path.split("."))
    package_dir = (site_packages / relative).resolve()
    return [str(package_dir)]


def _cleanup_scoped_modules(module_names: Sequence[str], *, scope_root: str | None = None) -> None:
    for name in reversed(module_names):
        sys.modules.pop(name, None)
    if scope_root is not None:
        prefix = f"{scope_root}."
        for name in list(sys.modules):
            if name == scope_root or name.startswith(prefix):
                sys.modules.pop(name, None)


class _ArtifactScopedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        scope = _scope_for_scoped_name(fullname)
        if scope is not None:
            original = _original_module_name(scope.scope_root, fullname)
            if original is None:
                return None
            if original == "":
                return self._spec_for_scope_root(scope)
            return self._spec_for_original(scope, original)

        redirect_scope = _scope_for_import_target(target)
        if redirect_scope is None:
            redirect_scope = _ActiveArtifactImportScope.get()
        if redirect_scope is None:
            return None
        if not _module_exists_in_artifact(redirect_scope.site_packages, fullname):
            return None
        scoped_name = _scoped_module_name(redirect_scope.scope_root, fullname)
        if scoped_name in sys.modules:
            return importlib.util.spec_from_loader(
                scoped_name,
                _LoadedModuleShim(sys.modules[scoped_name]),
            )
        return self._spec_for_original(redirect_scope, fullname)

    def _spec_for_scope_root(
        self,
        scope: _ArtifactImportScope,
    ) -> importlib.machinery.ModuleSpec:
        if scope.scope_root in sys.modules:
            return importlib.util.spec_from_loader(
                scope.scope_root,
                _LoadedModuleShim(sys.modules[scope.scope_root]),
            )
        loader = _ArtifactScopeRootLoader(scope)
        spec = importlib.util.spec_from_loader(scope.scope_root, loader)
        if spec is not None:
            spec.submodule_search_locations = [str(scope.site_packages.resolve())]
        return spec

    def _spec_for_original(
        self,
        scope: _ArtifactImportScope,
        original_module_path: str,
    ) -> importlib.machinery.ModuleSpec:
        scoped_name = _scoped_module_name(scope.scope_root, original_module_path)
        if scoped_name in sys.modules:
            return importlib.util.spec_from_loader(
                scoped_name,
                _LoadedModuleShim(sys.modules[scoped_name]),
            )
        module_file, is_package = _resolve_artifact_module(
            scope.site_packages,
            original_module_path,
        )
        loader = _ArtifactModuleLoader(
            scoped_name,
            str(module_file),
            scope=scope,
            original_module_path=original_module_path,
            is_package=is_package,
        )
        spec = importlib.util.spec_from_file_location(
            scoped_name,
            module_file,
            loader=loader,
        )
        if spec is None:
            raise RuntimeAgentFactoryResolutionError(
                f"cannot create import spec for {original_module_path!r}"
            )
        if is_package:
            locations = _package_search_locations(
                scope.site_packages,
                original_module_path,
                is_package=True,
            )
            spec.submodule_search_locations = locations
        return spec


class _LoadedModuleShim(importlib.abc.Loader):
    def __init__(self, module: object) -> None:
        self._module = module

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object:
        return self._module

    def exec_module(self, module: object) -> None:
        return None


class _ArtifactScopeRootLoader(importlib.abc.Loader):
    def __init__(self, scope: _ArtifactImportScope) -> None:
        self._scope = scope

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> object | None:
        return None

    def exec_module(self, module: object) -> None:
        module.__path__ = [str(self._scope.site_packages.resolve())]


class _ArtifactModuleLoader(importlib.machinery.SourceFileLoader):
    def __init__(
        self,
        fullname: str,
        path: str,
        *,
        scope: _ArtifactImportScope,
        original_module_path: str,
        is_package: bool,
    ) -> None:
        super().__init__(fullname, path)
        self._scope = scope
        self._original_module_path = original_module_path
        self._is_package = is_package

    def exec_module(self, module: object) -> None:
        if self._is_package:
            locations = _package_search_locations(
                self._scope.site_packages,
                self._original_module_path,
                is_package=True,
            )
            if locations is not None:
                module.__path__ = locations
        if "." in self._original_module_path:
            parent_original = self._original_module_path.rpartition(".")[0]
            module.__package__ = _scoped_module_name(
                self._scope.scope_root,
                parent_original,
            )
        else:
            module.__package__ = ""
        super().exec_module(module)


_ARTIFACT_IMPORT_FINDER = _ArtifactScopedImportFinder()
if not any(isinstance(finder, _ArtifactScopedImportFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ARTIFACT_IMPORT_FINDER)


def _ensure_scope_root(scope: _ArtifactImportScope) -> None:
    if scope.scope_root in sys.modules:
        return
    spec = _ARTIFACT_IMPORT_FINDER._spec_for_scope_root(scope)
    if spec is None or spec.loader is None:
        raise RuntimeAgentFactoryResolutionError(
            f"cannot create import scope root for artifact {scope.artifact_digest!r}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[scope.scope_root] = module
    spec.loader.exec_module(module)


def _ensure_package_hierarchy(
    scope: _ArtifactImportScope,
    original_module_path: str,
) -> list[str]:
    touched: list[str] = []
    parts = original_module_path.split(".")
    if len(parts) <= 1:
        return touched
    _ensure_scope_root(scope)
    for index in range(1, len(parts)):
        package_original = ".".join(parts[:index])
        scoped_name = _scoped_module_name(scope.scope_root, package_original)
        if scoped_name in sys.modules:
            continue
        package_dir = scope.site_packages.joinpath(*parts[:index])
        init_file = package_dir / "__init__.py"
        if not init_file.is_file():
            raise RuntimeAgentFactoryResolutionError(
                f"artifact package {package_original!r} requires __init__.py; "
                "PEP 420 namespace packages are not supported for VENV_BUNDLE factories"
            )
        spec = _ARTIFACT_IMPORT_FINDER._spec_for_original(scope, package_original)
        if spec is None or spec.loader is None:
            raise RuntimeAgentFactoryResolutionError(
                f"cannot create import spec for package {package_original!r}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[scoped_name] = module
        touched.append(scoped_name)
        try:
            spec.loader.exec_module(module)
        except Exception:
            _cleanup_scoped_modules(touched)
            raise
    return touched


def _import_scoped_module(
    scope: _ArtifactImportScope,
    original_module_path: str,
) -> object:
    scoped_name = _scoped_module_name(scope.scope_root, original_module_path)
    existing = sys.modules.get(scoped_name)
    if existing is not None:
        return existing
    with _ScopeLoadLock:
        existing = sys.modules.get(scoped_name)
        if existing is not None:
            return existing
        touched: list[str] = []
        scope_token = _ActiveArtifactImportScope.set(scope)
        try:
            _ensure_scope_root(scope)
            touched.extend(_ensure_package_hierarchy(scope, original_module_path))
            if scoped_name in sys.modules:
                return sys.modules[scoped_name]
            spec = _ARTIFACT_IMPORT_FINDER._spec_for_original(scope, original_module_path)
            if spec is None or spec.loader is None:
                raise RuntimeAgentFactoryResolutionError(
                    f"cannot create import spec for {original_module_path!r}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[scoped_name] = module
            touched.append(scoped_name)
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            _cleanup_scoped_modules(touched, scope_root=scope.scope_root)
            if isinstance(exc, RuntimeAgentFactoryResolutionError):
                raise
            raise RuntimeAgentFactoryResolutionError(
                f"failed to import artifact module {original_module_path!r}"
            ) from exc
        finally:
            _ActiveArtifactImportScope.reset(scope_token)


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
    module_file, is_package = _resolve_artifact_module(site_packages, module_path)
    if is_package:
        raise RuntimeAgentFactoryResolutionError(
            f"factory_path {module_path!r} must name a module attribute, not a package"
        )
    return module_file


def _load_callable_from_site_packages(
    *,
    site_packages: Path,
    module_path: str,
    attr_name: str,
    artifact_digest: str,
) -> Callable[..., Any]:
    _resolve_module_file(site_packages, module_path)
    scope = _register_artifact_scope(
        artifact_digest=artifact_digest,
        site_packages=site_packages,
    )
    module = _import_scoped_module(scope, module_path)
    target = getattr(module, attr_name, None)
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
