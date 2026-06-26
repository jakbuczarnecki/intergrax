# © Artur Czarnecki. All rights reserved.

"""Promote minimal lab application to standard scaffold (Phase DX-3.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.applications._shared.build_deploy_doc import render_build_deploy_doc
from intergrax.applications._shared.docker_templates import write_application_docker
from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames, app_slug
from intergrax.scaffold.new_application import (
    _agent_dirs,
    _factory_py,
    _mcp_server_py,
    _serving_router_py,
    _write,
)


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "expand",
        help="Promote a minimal lab application to standard (Docker, MCP, full factory)",
    )
    parser.add_argument("name", help="Application name (e.g. my_lab → my_lab_application)")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true", help="Overwrite generated files")


def expand_application(*, name: str, root: Path, force: bool) -> Path:
    pkg = app_slug(name)
    names = ScaffoldApplicationNames.resolve(name)
    target = root / "applications" / pkg
    if not target.is_dir():
        raise ValueError(f"Application directory not found: {target}")

    manifest_path = target / "manifest.py"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest.py in {target}")

    agent_slugs: list[str] = []
    manifest_text = manifest_path.read_text(encoding="utf-8")
    for line in manifest_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") and ". import " in stripped:
            module = stripped.split("from ", 1)[1].split(" import ", 1)[0].strip()
            if module and not module.startswith("intergrax"):
                agent_slugs.append(module.split(".", 1)[0])
    if not agent_slugs:
        agent_slugs = [names.short]
    specs = resolve_agent_specs(agent_slugs)
    agent_dirs = _agent_dirs(specs)
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else f"{names.short}.basic"
    health_path = f"{names.route_prefix}/agents"

    _write(target / "host" / "factory.py", _factory_py(names), force=force)
    _write(target / "serving" / "__init__.py", "", force=force)
    _write(target / "serving" / "fastapi_router.py", _serving_router_py(names), force=force)
    _write(target / "mcp" / "__init__.py", "", force=force)
    _write(target / "mcp" / "server.py", _mcp_server_py(names, specs), force=force)

    write_application_docker(
        target,
        pkg=names.pkg,
        short=names.short,
        port=names.port,
        env_prefix=names.env_prefix,
        agent_dirs=agent_dirs,
        health_path=health_path,
        factory_import=f"from {names.pkg}.host.factory import create_{names.short}_application",
        factory_call=f"create_{names.short}_application()",
        route_prefix=names.route_prefix,
        force=force,
    )
    docs_dir = target / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write(
        docs_dir / "BUILD_AND_DEPLOY.md",
        render_build_deploy_doc(
            pkg=names.pkg,
            short=names.short,
            port=names.port,
            env_prefix=names.env_prefix,
            route_prefix=names.route_prefix,
            profile="lab",
            agent_dirs=agent_dirs,
            example_capability=cap,
            health_path=health_path,
            tests_pkg=names.tests_pkg,
            display=names.display,
        ),
        force=force,
    )
    return target


def run_expand(args: argparse.Namespace) -> int:
    try:
        path = expand_application(name=args.name, root=args.root.resolve(), force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Expanded application at {path}")
    print("  Added: docker/, docs/BUILD_AND_DEPLOY.md, serving/, mcp/, full host/factory.py")
    return 0
