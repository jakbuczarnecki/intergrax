# © Artur Czarnecki. All rights reserved.

"""Load scaffolded Tier-3 packages from a temporary ``applications/`` tree for gate E2E."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from intergrax.scaffold.application_names import pascal_case
from intergrax.scaffold.new_application import create_application


def purge_scaffold_package(pkg: str) -> None:
    for key in list(sys.modules):
        if key == pkg or key.startswith(f"{pkg}."):
            del sys.modules[key]


def prepare_scaffold_package(
    tmp_path: Path,
    *,
    name: str,
    profile: str,
    port: int,
    route_prefix: str,
    agents: list[str] | None = None,
) -> tuple[Path, str, str]:
    """Create application under ``tmp_path/repo/applications/`` and expose it on ``sys.path``."""
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "applications").mkdir(parents=True, exist_ok=True)
    target = create_application(
        name=name,
        agents=agents or ["echo"],
        profile=profile,
        root=root,
        port=port,
        route_prefix=route_prefix,
    )
    pkg = target.name
    short = pkg.removesuffix("_application")
    apps_dir = str(target.parent)
    if apps_dir not in sys.path:
        sys.path.insert(0, apps_dir)
    purge_scaffold_package(pkg)
    return target, pkg, short


def import_scaffold_modules(pkg: str) -> tuple[ModuleType, ModuleType]:
    purge_scaffold_package(pkg)
    factory_mod = importlib.import_module(f"{pkg}.host.factory")
    settings_mod = importlib.import_module(f"{pkg}.host.settings")
    return factory_mod, settings_mod


def lab_settings_class(settings_mod: ModuleType, short: str) -> type:
    return getattr(settings_mod, f"{pascal_case(short)}ApplicationSettings")


def product_settings_class(settings_mod: ModuleType, short: str) -> type:
    return getattr(settings_mod, f"{pascal_case(short)}BackendSettings")
