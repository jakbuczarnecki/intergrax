# © Artur Czarnecki. All rights reserved.

from pathlib import Path

import pytest

pytestmark = pytest.mark.gate

BANNED_PREFIXES = (
    "intergrax.integrations.providers",
    "boto3",
    "httpx",
    "openai",
    "slack_sdk",
    "google.cloud",
    "azure.",
)


def _iter_agent_python_files(repo_root: Path):
    agents_root = repo_root / "agents"
    if not agents_root.is_dir():
        return
    for path in agents_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        yield path


def test_agents_do_not_import_vendor_sdks_directly():
    repo_root = Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for path in _iter_agent_python_files(repo_root):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("import ", "from ")):
                for banned in BANNED_PREFIXES:
                    if banned in stripped:
                        rel = path.relative_to(repo_root)
                        violations.append(f"{rel}:{line_no}: {stripped}")
    assert violations == [], "Tier-2 agents must not import vendor SDKs:\n" + "\n".join(violations)
