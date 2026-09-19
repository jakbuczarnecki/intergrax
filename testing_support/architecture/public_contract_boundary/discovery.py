# © Artur Czarnecki. All rights reserved.

"""Discover public contract Python modules under Tier-0 intergrax/."""

from __future__ import annotations

from pathlib import Path

from testing_support.architecture.public_contract_boundary.supplemental_surfaces import (
    SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES,
    SupplementalPublicContractSurface,
)


def path_to_module_name(path: Path, *, intergrax_root: Path) -> str:
    rel = path.relative_to(intergrax_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return "intergrax." + ".".join(parts)


def validate_supplemental_public_contract_surfaces(
    repo_root: Path,
    *,
    supplemental_surfaces: tuple[SupplementalPublicContractSurface, ...],
) -> tuple[str, ...]:
    errors: list[str] = []
    for surface in supplemental_surfaces:
        candidate = repo_root / surface.repo_relative_path
        if not candidate.is_file():
            errors.append(
                "Missing supplemental public contract surface: "
                f"{surface.repo_relative_path!r} "
                f"(owner={surface.owner_domain}, "
                f"stage={surface.remediation_stage.value})",
            )
    return tuple(errors)


def discover_public_contract_source_files(
    repo_root: Path,
    *,
    supplemental_surfaces: tuple[SupplementalPublicContractSurface, ...] | None = None,
) -> tuple[Path, ...]:
    intergrax_root = repo_root / "intergrax"
    discovered: list[Path] = []
    if intergrax_root.is_dir():
        for path in sorted(intergrax_root.rglob("*.py")):
            rel_parts = path.relative_to(intergrax_root).parts
            if not rel_parts:
                continue
            if rel_parts[0] == "runtime":
                continue
            if "contracts" not in rel_parts:
                continue
            discovered.append(path)
    surfaces = (
        SUPPLEMENTAL_PUBLIC_CONTRACT_SURFACES
        if supplemental_surfaces is None
        else supplemental_surfaces
    )
    for surface in surfaces:
        candidate = repo_root / surface.repo_relative_path
        discovered.append(candidate)
    return tuple(sorted(set(discovered)))


def discover_public_contract_modules(
    repo_root: Path,
    *,
    supplemental_surfaces: tuple[SupplementalPublicContractSurface, ...] | None = None,
) -> tuple[str, ...]:
    intergrax_root = repo_root / "intergrax"
    modules = [
        path_to_module_name(path, intergrax_root=intergrax_root)
        for path in discover_public_contract_source_files(
            repo_root,
            supplemental_surfaces=supplemental_surfaces,
        )
    ]
    return tuple(sorted(set(modules)))
