# © Artur Czarnecki. All rights reserved.

"""Architecture gates for Agent Manager — no direct lifecycle service imports."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_MANAGER_DIR = REPO_ROOT / "intergrax" / "agent_distribution"

_FORBIDDEN_LIFECYCLE_FRAGMENTS = (
    "InstallationService",
    "BindingService",
    "RuntimeRevisionService",
    "ActivationService",
)

_ALLOWED_MANAGER_MODULES = (
    "agent_manager_models.py",
    "agent_manager_query_service.py",
    "agent_manager_command_facade.py",
    "federated_catalog.py",
)


def test_agent_manager_modules_have_no_direct_lifecycle_service_imports() -> None:
    violations: list[str] = []
    for module_name in _ALLOWED_MANAGER_MODULES:
        path = AGENT_MANAGER_DIR / module_name
        text = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_LIFECYCLE_FRAGMENTS:
            if fragment in text:
                violations.append(f"{module_name} contains forbidden fragment {fragment!r}")
    assert not violations, "\n".join(violations)


def test_agent_manager_command_facade_delegates_only_to_admin_service() -> None:
    path = AGENT_MANAGER_DIR / "agent_manager_command_facade.py"
    text = path.read_text(encoding="utf-8")
    assert "AgentPlatformAdminService" in text
    for fragment in _FORBIDDEN_LIFECYCLE_FRAGMENTS:
        assert fragment not in text
