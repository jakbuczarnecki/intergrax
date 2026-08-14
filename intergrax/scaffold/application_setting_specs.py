# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit scaffold metadata for generated application settings and docs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RequiredFlag = Literal["Yes", "No", "Conditional"]


@dataclass(frozen=True)
class ApplicationSettingSpec:
    """One scaffold-owned application environment variable (suffix without prefix)."""

    env_suffix: str
    purpose: str
    default: str
    required: RequiredFlag
    example: str
    related_suffixes: tuple[str, ...] = ()
    platform_relation: str = ""
    required_note: str = ""
    comment_in_env_example: bool = False


def _host_setting_specs(
    *,
    include_scheduler_default: str,
    include_queue_worker_default: str,
) -> tuple[ApplicationSettingSpec, ...]:
    return (
        ApplicationSettingSpec(
            env_suffix="BACKEND_ENV",
            purpose="Runtime environment for this application host (dev, test, stage, prod).",
            default="dev (uses INTERGRAX_ENV when this variable is unset)",
            required="No",
            example="dev",
            platform_relation="Falls back to the shared INTERGRAX_ENV platform setting.",
        ),
        ApplicationSettingSpec(
            env_suffix="BACKEND_HOST",
            purpose="Bind address for the application HTTP server.",
            default="127.0.0.1",
            required="No",
            example="127.0.0.1",
            related_suffixes=("BACKEND_PORT",),
            platform_relation="This host process runs the shared Intergrax runtime.",
        ),
        ApplicationSettingSpec(
            env_suffix="BACKEND_PORT",
            purpose="TCP port for the application HTTP server.",
            default="{port}",
            required="No",
            example="{port}",
            related_suffixes=("BACKEND_HOST",),
        ),
        ApplicationSettingSpec(
            env_suffix="ROUTE_PREFIX",
            purpose="URL prefix for this application's HTTP routes.",
            default="{route_prefix}",
            required="No",
            example="{route_prefix}",
        ),
        ApplicationSettingSpec(
            env_suffix="INCLUDE_INTERACTIONS",
            purpose="Expose interaction HTTP routes on this application.",
            default="true",
            required="No",
            example="true",
            related_suffixes=("INTERACTION_ROUTE_PREFIX", "INTERACTION_SURFACE"),
        ),
        ApplicationSettingSpec(
            env_suffix="INTERACTION_ROUTE_PREFIX",
            purpose="URL prefix for interaction routes.",
            default="/v1/interactions",
            required="No",
            example="/v1/interactions",
            related_suffixes=("INCLUDE_INTERACTIONS",),
        ),
        ApplicationSettingSpec(
            env_suffix="INTERACTION_SURFACE",
            purpose="Which interaction UI/API surface to mount (auto selects the default).",
            default="auto",
            required="No",
            example="auto",
            related_suffixes=("INCLUDE_INTERACTIONS",),
        ),
        ApplicationSettingSpec(
            env_suffix="INCLUDE_SCHEDULER",
            purpose="Run the in-process scheduler with this application.",
            default=include_scheduler_default,
            required="No",
            example=include_scheduler_default,
            platform_relation="Poll interval is the platform setting INTERGRAX_SCHEDULER_POLL_SECONDS.",
        ),
        ApplicationSettingSpec(
            env_suffix="INCLUDE_MCP",
            purpose="Mount the MCP server on the same process as this application.",
            default="false",
            required="No",
            example="false",
            related_suffixes=("MCP_MOUNT_PATH",),
        ),
        ApplicationSettingSpec(
            env_suffix="MCP_MOUNT_PATH",
            purpose="HTTP path where the MCP server is mounted when enabled.",
            default="/mcp",
            required="No",
            example="/mcp",
            related_suffixes=("INCLUDE_MCP",),
        ),
        ApplicationSettingSpec(
            env_suffix="INCLUDE_TASK_CONTROL",
            purpose="Expose task-control HTTP routes on this application.",
            default="true",
            required="No",
            example="true",
            related_suffixes=("TASK_CONTROL_ROUTE_PREFIX",),
        ),
        ApplicationSettingSpec(
            env_suffix="TASK_CONTROL_ROUTE_PREFIX",
            purpose="URL prefix for task-control routes.",
            default="/v1/tasks",
            required="No",
            example="/v1/tasks",
            related_suffixes=("INCLUDE_TASK_CONTROL",),
        ),
        ApplicationSettingSpec(
            env_suffix="INCLUDE_QUEUE_WORKER",
            purpose="Run the in-process queue worker with this application.",
            default=include_queue_worker_default,
            required="No",
            example=include_queue_worker_default,
        ),
    )


_PRODUCT_SETTING_SPECS: tuple[ApplicationSettingSpec, ...] = (
    ApplicationSettingSpec(
        env_suffix="DEFAULT_AGENT_ID",
        purpose="Default agent id used when a request does not name one.",
        default="echo",
        required="No",
        example="echo",
    ),
    ApplicationSettingSpec(
        env_suffix="IDENTITY_SOURCE",
        purpose="Where the API reads caller identity from (body_or_context or context_only).",
        default="body_or_context (context_only when BACKEND_ENV=prod)",
        required="Conditional",
        example="body_or_context",
        related_suffixes=("BACKEND_ENV",),
        required_note="Production requires context_only (or omit the variable to use that default).",
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_CORS_ORIGINS",
        purpose="Comma-separated CORS origins allowed by this application.",
        default="(empty)",
        required="No",
        example="https://app.example.com",
        related_suffixes=("BACKEND_ALLOWED_HOSTS",),
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_ALLOWED_HOSTS",
        purpose="Comma-separated Host headers this application accepts.",
        default="(empty)",
        required="No",
        example="api.example.com",
        related_suffixes=("BACKEND_CORS_ORIGINS",),
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_OPENAPI",
        purpose="Override whether OpenAPI docs are enabled for this application.",
        default="unset (framework default)",
        required="No",
        example="true",
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_BOOTSTRAP_API_KEY",
        purpose="Single development API key for this application.",
        default="(empty)",
        required="Conditional",
        example="dev-key",
        related_suffixes=(
            "BACKEND_BOOTSTRAP_TENANT_ID",
            "BACKEND_BOOTSTRAP_USER_ID",
            "BACKEND_API_KEYS_JSON",
            "BACKEND_ALLOW_UNAUTHENTICATED",
        ),
        required_note="Required in production unless BACKEND_API_KEYS_JSON or BACKEND_ALLOW_UNAUTHENTICATED is set.",
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_BOOTSTRAP_TENANT_ID",
        purpose="Tenant id bound to BACKEND_BOOTSTRAP_API_KEY.",
        default="(empty)",
        required="Conditional",
        example="dev-tenant",
        related_suffixes=("BACKEND_BOOTSTRAP_API_KEY", "BACKEND_BOOTSTRAP_USER_ID"),
        required_note="Required when BACKEND_BOOTSTRAP_API_KEY is set.",
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_BOOTSTRAP_USER_ID",
        purpose="User id bound to BACKEND_BOOTSTRAP_API_KEY.",
        default="(empty)",
        required="Conditional",
        example="dev-user",
        related_suffixes=("BACKEND_BOOTSTRAP_API_KEY", "BACKEND_BOOTSTRAP_TENANT_ID"),
        required_note="Required when BACKEND_BOOTSTRAP_API_KEY is set.",
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_API_KEYS_JSON",
        purpose="JSON object of API keys to identities. Do not set together with BACKEND_BOOTSTRAP_API_KEY.",
        default="(empty)",
        required="Conditional",
        example='{"dev-key":{"tenant_id":"dev-tenant","user_id":"dev-user"}}',
        related_suffixes=("BACKEND_BOOTSTRAP_API_KEY", "BACKEND_ALLOW_UNAUTHENTICATED"),
        required_note="Required in production unless BACKEND_BOOTSTRAP_API_KEY or BACKEND_ALLOW_UNAUTHENTICATED is set.",
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="BACKEND_ALLOW_UNAUTHENTICATED",
        purpose="Allow unauthenticated access. Local disaster debugging only.",
        default="false",
        required="No",
        example="true",
        related_suffixes=("BACKEND_BOOTSTRAP_API_KEY", "BACKEND_API_KEYS_JSON"),
        comment_in_env_example=True,
    ),
    ApplicationSettingSpec(
        env_suffix="INTERACTION_EXECUTE_DEFAULT",
        purpose="Whether interaction execute is enabled by default on this application.",
        default="true",
        required="No",
        example="true",
        related_suffixes=("INCLUDE_INTERACTIONS",),
    ),
)


def application_setting_specs(profile: str) -> tuple[ApplicationSettingSpec, ...]:
    """Scaffold-owned application settings for *profile* (``lab`` or ``product``)."""
    if profile == "product":
        host = _host_setting_specs(
            include_scheduler_default="false",
            include_queue_worker_default="false",
        )
        return host + _PRODUCT_SETTING_SPECS
    if profile != "lab":
        raise ValueError(f"Unsupported profile {profile!r}; choose: lab, product")
    return _host_setting_specs(
        include_scheduler_default="true",
        include_queue_worker_default="true",
    )


def application_env_names(profile: str, env_prefix: str) -> tuple[str, ...]:
    """Full environment variable names documented for *profile*."""
    return tuple(f"{env_prefix}{spec.env_suffix}" for spec in application_setting_specs(profile))
