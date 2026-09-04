# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Normalize Tier-3 application names for scaffold templates."""

from __future__ import annotations

import re
from dataclasses import dataclass


def app_slug(name: str) -> str:
    """Python package under ``applications/`` (always ends with ``_application``)."""
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or slug[0].isdigit():
        raise ValueError(f"Invalid application name: {name!r}")
    if not slug.endswith("_application"):
        slug = f"{slug}_application"
    return slug


def short_id(app_slug_value: str) -> str:
    """Stable id without ``_application`` suffix (``app_id``, routes, functions)."""
    if app_slug_value.endswith("_application"):
        return app_slug_value[: -len("_application")]
    return app_slug_value


def env_prefix(short: str) -> str:
    """Environment variable prefix, e.g. ``MY_LAB_``."""
    return re.sub(r"[^A-Z0-9]", "_", short.upper()).strip("_") + "_"


def pascal_case(short: str) -> str:
    """``my_lab`` → ``MyLab`` (class names)."""
    return "".join(part.capitalize() for part in short.split("_"))


def display_name(short: str) -> str:
    """``my_lab`` → ``My Lab`` (titles in docs)."""
    return " ".join(part.capitalize() for part in short.split("_"))


def agent_builders_const(short: str) -> str:
    """Module-level builders dict, e.g. ``CONCEPT_LAB_AGENT_BUILDERS``."""
    return f"{env_prefix(short).rstrip('_')}_AGENT_BUILDERS"


def docker_image_tag(short: str) -> str:
    """Default Docker image name: ``{short}-application``."""
    return f"{short}-application"


@dataclass(frozen=True)
class ScaffoldApplicationNames:
    """All derived identifiers for one scaffolded application."""

    input_name: str
    pkg: str
    short: str
    pascal: str
    display: str
    env_prefix: str
    route_prefix: str
    port: int
    tests_pkg: str
    builders_const: str
    docker_image: str
    factory_fn: str
    manifest_fn: str
    registry_fn: str
    settings_class: str

    @classmethod
    def resolve(
        cls,
        name: str,
        *,
        route_prefix: str | None = None,
        port: int = 8091,
    ) -> ScaffoldApplicationNames:
        pkg = app_slug(name)
        short = short_id(pkg)
        prefix = env_prefix(short)
        route = route_prefix or f"/v1/{short}"
        pascal = pascal_case(short)
        return cls(
            input_name=name,
            pkg=pkg,
            short=short,
            pascal=pascal,
            display=display_name(short),
            env_prefix=prefix,
            route_prefix=route,
            port=port,
            tests_pkg="tests",
            builders_const=agent_builders_const(short),
            docker_image=docker_image_tag(short),
            factory_fn=f"create_{short}_application",
            manifest_fn=f"build_{short}_manifest",
            registry_fn=f"build_{short}_development_registry",
            settings_class=f"{pascal}ApplicationSettings",
        )
