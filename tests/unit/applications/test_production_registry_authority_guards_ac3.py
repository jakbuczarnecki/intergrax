# © Artur Czarnecki. All rights reserved.

"""AC-3-FINAL production authority source guards."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[3]

_PRODUCTION_AUTHORITY_SOURCES = (
    REPO / "intergrax" / "applications" / "_shared" / "production_registry_projection_input_bundle.py",
    REPO / "intergrax" / "applications" / "_shared" / "production_host_composition.py",
    REPO / "intergrax" / "applications" / "_shared" / "active_registry_projection.py",
    REPO / "intergrax" / "applications" / "_shared" / "venv_bundle_runtime_agent_factory_resolver.py",
)

_FORBIDDEN_IMPORT_TOKENS = (
    "InMemoryRuntimeAgentFactoryResolver",
    "FakeInMemoryRuntimeDeploymentAdapter",
    "testing_support",
    "build_reference_registry_projection_input_bundle",
    "BuilderMap",
)


def _import_block(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("from ", "import ")):
            lines.append(stripped)
    return "\n".join(lines)


@pytest.mark.parametrize("source_path", _PRODUCTION_AUTHORITY_SOURCES, ids=lambda p: p.name)
def test_production_authority_sources_forbid_synthetic_runtime_tokens(
    source_path: Path,
) -> None:
    text = source_path.read_text(encoding="utf-8")
    imports = _import_block(text)
    for token in _FORBIDDEN_IMPORT_TOKENS:
        assert token not in imports, f"{source_path.name} imports forbidden symbol {token!r}"


def test_production_registry_input_module_has_no_dynamic_attribute_access() -> None:
    source = (
        REPO
        / "intergrax"
        / "applications"
        / "_shared"
        / "production_registry_projection_input_bundle.py"
    )
    text = source.read_text(encoding="utf-8")
    for token in ("getattr(", "setattr(", "hasattr("):
        assert token not in text
