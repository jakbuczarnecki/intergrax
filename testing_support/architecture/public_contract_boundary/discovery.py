# © Artur Czarnecki. All rights reserved.

"""Discover public contract Python modules under Tier-0 intergrax/."""

from __future__ import annotations

from pathlib import Path

_SUPPLEMENTAL_CONTRACT_FILES: tuple[str, ...] = (
    "intergrax/agents/agent_contract.py",
)


def path_to_module_name(path: Path, *, intergrax_root: Path) -> str:
    rel = path.relative_to(intergrax_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return "intergrax." + ".".join(parts)


def discover_public_contract_source_files(
    repo_root: Path,
) -> tuple[Path, ...]:
    intergrax_root = repo_root / "intergrax"
    discovered: list[Path] = []
    for path in sorted(intergrax_root.rglob("*.py")):
        rel_parts = path.relative_to(intergrax_root).parts
        if not rel_parts:
            continue
        if rel_parts[0] == "runtime":
            continue
        if "contracts" not in rel_parts:
            continue
        discovered.append(path)
    for rel in _SUPPLEMENTAL_CONTRACT_FILES:
        candidate = repo_root / rel
        if candidate.is_file():
            discovered.append(candidate)
    return tuple(sorted(set(discovered)))


def discover_public_contract_modules(repo_root: Path) -> tuple[str, ...]:
    intergrax_root = repo_root / "intergrax"
    modules = [
        path_to_module_name(path, intergrax_root=intergrax_root)
        for path in discover_public_contract_source_files(repo_root)
    ]
    return tuple(sorted(set(modules)))
