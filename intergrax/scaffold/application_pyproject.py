# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generate Tier-3 application ``pyproject.toml`` (workspace dependency project)."""

from __future__ import annotations

from textwrap import dedent


def distribution_name_for_pkg(pkg: str) -> str:
    """Stable PyPI-style distribution name for an application folder."""
    if pkg == "attestation_demo":
        return "intergrax-attestation-demo"
    if pkg == "intergrax_assistant_application":
        return "intergrax-assistant-application"
    slug = pkg
    if slug.endswith("_application"):
        slug = slug[: -len("_application")]
    return f"intergrax-{slug.replace('_', '-')}-application"


def render_application_pyproject(
    *,
    pkg: str,
    display: str,
    platform_extras: list[str] | None = None,
    application_dependencies: list[str] | None = None,
) -> str:
    """Render application workspace member ``pyproject.toml`` content."""
    extras = list(platform_extras or [])
    app_deps = list(application_dependencies or [])
    dist = distribution_name_for_pkg(pkg)
    if extras:
        intergrax_dep = f'  "Intergrax-ai[{",".join(extras)}]",'
    else:
        intergrax_dep = '  "Intergrax-ai",'
    dep_lines = [intergrax_dep, *[f'  "{dep}",' for dep in app_deps]]
    deps_block = "\n".join(dep_lines)
    return dedent(
        f"""\
        # © Artur Czarnecki. All rights reserved.
        # Tier-3 application dependency project (workspace member).
        # Application source remains importable via PYTHONPATH=applications/.
        # Canonical platform doc: docs/architecture/APPLICATION_DEPENDENCY_MODEL.md

        [project]
        name = "{dist}"
        version = "0.1.0"
        description = "{display} Tier-3 application"
        requires-python = ">=3.12,<3.13"
        dependencies = [
        {deps_block}
        ]

        [tool.uv]
        package = false

        [tool.uv.sources]
        Intergrax-ai = {{ workspace = true }}
        """
    )


def platform_extras_for_profile(profile: str, *, minimal: bool = False) -> list[str]:
    """Smallest correct Intergrax extras for a scaffold profile."""
    del minimal  # reserved for future slim profiles
    if profile == "product":
        # Product hosts may use RAG/LLM; those remain in the platform base for now.
        return []
    # lab / default
    return []
