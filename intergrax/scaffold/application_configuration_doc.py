# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CONFIGURATION.md and .env.example renderers for scaffolded applications."""

from __future__ import annotations

from collections import defaultdict
from textwrap import dedent

from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.application_setting_specs import (
    ENV_CATEGORY_ORDER,
    ENV_CATEGORY_TITLES,
    ApplicationSettingSpec,
    application_setting_specs,
)

PLATFORM_CONFIGURATION_RELATIVE = (
    "../../../docs/project/technical/guides/PLATFORM_CONFIGURATION.md"
)

_LAB_PLATFORM_GUARDRAILS = """\
# -----------------------------------------------------------------------------
# LLM guardrails (optional)
# -----------------------------------------------------------------------------
# Enable application-level LLM guardrails (M.12)
# {env_prefix}ENABLE_LLM_GUARDRAILS=false
# Primary guardrail integration slug
# {env_prefix}LLM_GUARDRAIL_PRIMARY=llm_guard
# INTERGRAX_LAKERA_API_KEY=
# INTERGRAX_OPENGUARDRAILS_BASE_URL=
"""


def _format_spec_value(value: str, *, port: int, route_prefix: str) -> str:
    return value.replace("{port}", str(port)).replace("{route_prefix}", route_prefix)


def _settings_class_name(names: ScaffoldApplicationNames, profile: str) -> str:
    if profile == "product":
        return f"{names.pascal}BackendSettings"
    return names.settings_class


def _render_setting_entry(
    spec: ApplicationSettingSpec,
    *,
    env_prefix: str,
    port: int,
    route_prefix: str,
) -> str:
    name = f"{env_prefix}{spec.env_suffix}"
    default = _format_spec_value(spec.default, port=port, route_prefix=route_prefix)
    example_value = _format_spec_value(spec.example, port=port, route_prefix=route_prefix)
    required = spec.required
    if spec.required_note:
        required = f"{spec.required} — {spec.required_note}"
    blocks = [
        f"### {name}",
        "",
        "Purpose:",
        spec.purpose,
        "",
        "Default:",
        default,
        "",
        "Required:",
        required,
        "",
        "Example:",
        f"{name}={example_value}",
    ]
    if spec.related_suffixes:
        related = ", ".join(f"{env_prefix}{suffix}" for suffix in spec.related_suffixes)
        blocks.extend(["", "Related:", related])
    if spec.platform_relation:
        blocks.extend(["", "Platform relation:", spec.platform_relation])
    return "\n".join(blocks)


def _env_comment(spec: ApplicationSettingSpec) -> str:
    if spec.env_comment:
        return spec.env_comment
    return spec.purpose.split(".", maxsplit=1)[0]


def _append_env_section(lines: list[str], title: str) -> None:
    lines.extend(
        [
            "",
            "# -----------------------------------------------------------------------------",
            f"# {title}",
            "# -----------------------------------------------------------------------------",
        ]
    )


def _append_env_assignment(
    lines: list[str],
    *,
    spec: ApplicationSettingSpec,
    env_prefix: str,
    port: int,
    route_prefix: str,
) -> None:
    example_value = _format_spec_value(spec.example, port=port, route_prefix=route_prefix)
    assignment = f"{env_prefix}{spec.env_suffix}={example_value}"
    lines.append(f"# {_env_comment(spec)}")
    lines.append(f"# {assignment}" if spec.comment_in_env_example else assignment)


def render_application_configuration_doc(
    *,
    names: ScaffoldApplicationNames,
    profile: str,
) -> str:
    """Render ``docs/CONFIGURATION.md`` for a scaffolded application."""
    specs = application_setting_specs(profile)
    settings_class = _settings_class_name(names, profile)
    entries = "\n\n".join(
        _render_setting_entry(
            spec,
            env_prefix=names.env_prefix,
            port=names.port,
            route_prefix=names.route_prefix,
        )
        for spec in specs
    )
    header = dedent(
        f"""\
        # {names.display} Configuration

        Settings that belong to **{names.display}** (`{names.pkg}`).
        Application variables use the prefix `{names.env_prefix}`.

        ## Platform configuration

        Shared Intergrax platform settings such as `INTERGRAX_LLM_PROVIDER`,
        `INTERGRAX_LLM_MODEL`, `INTERGRAX_EMBEDDING_PROVIDER`,
        `INTERGRAX_EMBEDDING_MODEL`, and other `INTERGRAX_*` keys are **not**
        specific to this application. They are documented in:

        [Platform configuration]({PLATFORM_CONFIGURATION_RELATIVE})

        Do not copy the platform catalog into this file.

        ## Application configuration

        These settings are generated with the application scaffold. They configure
        this application host, not the shared platform.

        """
    )
    footer = dedent(
        f"""\

        ## Adding application-specific settings

        Scaffold-owned settings in this file and `.env.example` are generated
        from `ApplicationSettingSpec` metadata
        (`intergrax/scaffold/application_setting_specs.py`). Typed fields and
        runtime loaders are emitted separately: keep `{settings_class}` in
        `host/settings.py` (and `IntergraxApplicationSettingsBase` for shared
        host keys) aligned with the spec. Do not treat the spec as a generator
        for arbitrary later app-local code.

        For a scaffold-owned setting:

        1. Add or update the `ApplicationSettingSpec` metadata.
        2. Ensure the typed settings class and runtime loader consume it
           (`{settings_class}` / `_load_app_env`, or
           `IntergraxApplicationSettingsBase` for shared host keys).
        3. Re-run the application scaffold. `CONFIGURATION.md` and
           `.env.example` are generated from the shared spec — do not
           hand-copy the same setting into those generated files.
        4. Add or update tests for the setting.

        Custom settings added in this application after scaffolding are not generated from `ApplicationSettingSpec`.
        Add a typed field and load it in `_load_app_env` on `{settings_class}`
        (prefix `{names.env_prefix}`). Those app-local keys are outside the
        scaffold catalog.
        """
    )
    return header + entries + footer


def render_application_env_example(
    *,
    env_prefix: str,
    route_prefix: str,
    port: int,
    profile: str,
    example_capability: str,
) -> str:
    """Render application `.env.example` from the same setting specs as CONFIGURATION.md."""
    lines = [
        f"# {env_prefix}* - copy to .env (gitignored) in this application directory.",
        f"# Application settings use the {env_prefix}* prefix.",
        f"# Platform INTERGRAX_* catalog: docs/project/technical/guides/PLATFORM_CONFIGURATION.md",
        "",
        "# =============================================================================",
        "# Application configuration",
        "# =============================================================================",
    ]

    by_category: dict[str, list[ApplicationSettingSpec]] = defaultdict(list)
    for spec in application_setting_specs(profile):
        by_category[spec.category].append(spec)

    for category in ENV_CATEGORY_ORDER:
        specs = by_category.get(category)
        if not specs:
            continue
        _append_env_section(lines, ENV_CATEGORY_TITLES[category])
        for spec in specs:
            _append_env_assignment(
                lines,
                spec=spec,
                env_prefix=env_prefix,
                port=port,
                route_prefix=route_prefix,
            )

    lines.extend(
        [
            "",
            "# Example run capability for POST {route_prefix}/run".format(
                route_prefix=route_prefix
            ),
            f"# DEFAULT_CAPABILITY={example_capability}",
            "",
            "# =============================================================================",
            "# Platform configuration",
            "# =============================================================================",
        ]
    )
    _append_env_section(lines, "Runtime")
    lines.extend(
        [
            "# Shared platform runtime environment (dev, test, stage, prod)",
            "INTERGRAX_ENV=dev",
        ]
    )

    if profile == "lab":
        lines.append(_LAB_PLATFORM_GUARDRAILS.format(env_prefix=env_prefix).rstrip())

    return "\n".join(lines) + "\n"
