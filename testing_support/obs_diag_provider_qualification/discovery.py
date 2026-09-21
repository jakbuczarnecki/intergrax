# © Artur Czarnecki. All rights reserved.

"""Manifest-driven discovery of OBS/DIAG-relevant integration provider surfaces."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.contracts.manifest import IntegrationManifest

from testing_support.obs_diag_provider_qualification.descriptor import ObsDiagProviderDomain

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROVIDERS_ROOT = _REPO_ROOT / "intergrax" / "integrations" / "providers"

_OBS_DIAG_CATEGORY_DIRS: tuple[tuple[str, ObsDiagProviderDomain], ...] = (
    ("document_store", ObsDiagProviderDomain.PERSISTENCE),
    ("message_bus", ObsDiagProviderDomain.TRANSPORT),
    ("observability_backend", ObsDiagProviderDomain.TELEMETRY),
)

_CATEGORY_TO_DOMAIN: dict[IntegrationCategory, ObsDiagProviderDomain] = {
    IntegrationCategory.DOCUMENT_STORE: ObsDiagProviderDomain.PERSISTENCE,
    IntegrationCategory.MESSAGE_BUS: ObsDiagProviderDomain.TRANSPORT,
    IntegrationCategory.OBSERVABILITY_BACKEND: ObsDiagProviderDomain.TELEMETRY,
}


@dataclass(frozen=True, slots=True)
class DiscoveredObsDiagProvider:
    provider_id: str
    domain: ObsDiagProviderDomain
    manifest_path: str
    integration_status: IntegrationStatus
    adapter_package: str


def _load_manifest_from_path(manifest_file: Path) -> IntegrationManifest:
    module_name = f"_obs_diag_manifest_{manifest_file.parent.name}_{manifest_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, manifest_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load manifest module from {manifest_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.MANIFEST
    if not isinstance(manifest, IntegrationManifest):
        raise TypeError(f"{manifest_file} MANIFEST is not IntegrationManifest")
    return manifest


def _domain_for_manifest(
    manifest: IntegrationManifest,
    fallback: ObsDiagProviderDomain,
) -> ObsDiagProviderDomain:
    for category in manifest.categories:
        mapped = _CATEGORY_TO_DOMAIN.get(category)
        if mapped is not None:
            return mapped
    return fallback


def discover_obs_diag_provider_surfaces(
    *,
    providers_root: Path | None = None,
    supplemental_manifest_dirs: tuple[Path, ...] = (),
) -> tuple[DiscoveredObsDiagProvider, ...]:
    """
    Discover external catalog providers under OBS/DIAG domains.

    ``supplemental_manifest_dirs`` supports anti-drift tests (unclassified manifest injection).
    """
    root = providers_root or _DEFAULT_PROVIDERS_ROOT
    discovered: list[DiscoveredObsDiagProvider] = []

    for category_dir, default_domain in _OBS_DIAG_CATEGORY_DIRS:
        category_root = root / category_dir
        if not category_root.is_dir():
            continue
        for manifest_file in sorted(category_root.glob("*/manifest.py")):
            manifest = _load_manifest_from_path(manifest_file)
            rel_manifest = manifest_file.relative_to(_REPO_ROOT).as_posix()
            adapter_package = (
                f"intergrax.integrations.providers.{category_dir}.{manifest_file.parent.name}"
            )
            discovered.append(
                DiscoveredObsDiagProvider(
                    provider_id=manifest.slug,
                    domain=_domain_for_manifest(manifest, default_domain),
                    manifest_path=rel_manifest,
                    integration_status=manifest.status,
                    adapter_package=adapter_package,
                ),
            )

    for extra_dir in supplemental_manifest_dirs:
        manifest_file = extra_dir / "manifest.py"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"supplemental manifest missing: {manifest_file}")
        manifest = _load_manifest_from_path(manifest_file)
        rel_manifest = manifest_file.resolve().as_posix()
        discovered.append(
            DiscoveredObsDiagProvider(
                provider_id=manifest.slug,
                domain=ObsDiagProviderDomain.PERSISTENCE,
                manifest_path=rel_manifest,
                integration_status=manifest.status,
                adapter_package="supplemental",
            ),
        )

    discovered.sort(key=lambda row: row.provider_id)
    return tuple(discovered)
