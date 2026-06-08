# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tracing scaffold templates for agent and application extension SDK (OBS-BUS-4)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def write_agent_tracing_scaffold(*, target: Path, slug: str, force: bool) -> None:
    _write(target / "tracing" / "__init__.py", _agent_tracing_init(slug), force=force)
    _write(target / "tracing" / "example_diag.py", _agent_example_diag(slug), force=force)
    _write(target / "tracing" / "registry.py", _agent_registry_py(slug), force=force)
    _write(target / "tests" / f"test_{slug}_tracing.py", _agent_tracing_test(slug), force=force)


def write_application_tracing_scaffold(
    *,
    target: Path,
    pkg: str,
    short: str,
    force: bool,
) -> None:
    _write(target / "tracing" / "__init__.py", _application_tracing_init(pkg), force=force)
    _write(target / "tracing" / "example_diag.py", _application_example_diag(short), force=force)
    _write(target / "tracing" / "registry.py", _application_registry_py(pkg, short), force=force)


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _agent_tracing_init(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Agent observability extensions for {slug}."""

        from {slug}.tracing.registry import register_tracing_schemas

        __all__ = ["register_tracing_schemas"]
        '''
    )


def _agent_example_diag(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any, Dict

        from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
        from intergrax.runtime.observability.extension_sdk import agent_diagnostic_schema_id


        @dataclass(frozen=True)
        class CustomCheckDiagV1(DiagnosticPayload):
            """Example agent diagnostic payload — replace with domain semantics."""

            check_name: str
            passed: bool
            detail: str = ""

            @classmethod
            def schema_id(cls) -> str:
                return agent_diagnostic_schema_id("{slug}", "custom_check")

            def to_dict(self) -> Dict[str, Any]:
                return {{
                    "check_name": self.check_name,
                    "passed": self.passed,
                    "detail": self.detail,
                }}

            def redact(self) -> CustomCheckDiagV1:
                return self
        '''
    )


def _agent_registry_py(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.observability.extension_sdk import PayloadSchemaRegistry
        from {slug}.tracing.example_diag import CustomCheckDiagV1


        def register_tracing_schemas() -> None:
            """Register agent diagnostic schemas with the Harness observability spine."""
            PayloadSchemaRegistry.register_agent_diagnostic(
                CustomCheckDiagV1,
                agent_slug="{slug}",
            )
        '''
    )


def _agent_tracing_test(slug: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        import pytest

        from {slug}.tracing.example_diag import CustomCheckDiagV1
        from {slug}.tracing.registry import register_tracing_schemas
        from intergrax.runtime.observability.extension_sdk import (
            get_registered_diagnostic_payload,
            list_registered_diagnostic_schema_ids,
        )

        pytestmark = pytest.mark.gate


        def test_agent_tracing_schema_registers() -> None:
            register_tracing_schemas()
            schema_id = CustomCheckDiagV1.schema_id()
            assert schema_id in list_registered_diagnostic_schema_ids()
            assert get_registered_diagnostic_payload(schema_id) is CustomCheckDiagV1
        '''
    )


def _application_tracing_init(pkg: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        """Application observability extensions for {pkg}."""

        from applications.{pkg}.tracing.registry import register_tracing_schemas

        __all__ = ["register_tracing_schemas"]
        '''
    )


def _application_example_diag(short: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any, Dict

        from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
        from intergrax.runtime.observability.extension_sdk import application_diagnostic_schema_id


        @dataclass(frozen=True)
        class HostLifecycleDiagV1(DiagnosticPayload):
            """Example application diagnostic payload — replace with product semantics."""

            phase: str
            status: str

            @classmethod
            def schema_id(cls) -> str:
                return application_diagnostic_schema_id("{short}", "host_lifecycle")

            def to_dict(self) -> Dict[str, Any]:
                return {{"phase": self.phase, "status": self.status}}

            def redact(self) -> HostLifecycleDiagV1:
                return self
        '''
    )


def _application_registry_py(pkg: str, short: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.runtime.observability.extension_sdk import PayloadSchemaRegistry
        from applications.{pkg}.tracing.example_diag import HostLifecycleDiagV1


        def register_tracing_schemas() -> None:
            """Register application diagnostic schemas with the Harness observability spine."""
            PayloadSchemaRegistry.register_application_diagnostic(
                HostLifecycleDiagV1,
                app_slug="{short}",
            )
        '''
    )
