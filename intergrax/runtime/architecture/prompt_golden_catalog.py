# © Artur Czarnecki. All rights reserved.

"""Golden prompt catalog content checks (FAUDIT-PE.1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field


class PromptGoldenExpectation(BaseModel):
    prompt_id: str
    version: int
    content_sha256: str


class PromptGoldenCatalogReport(BaseModel):
    schema_version: str = "1.0.0"
    checked: int = 0
    passed: bool = True
    failures: list[str] = Field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_version_path(catalog_dir: Path, *, prompt_id: str, version: int) -> Path:
    """Resolve a versioned YAML path using the production catalog layout."""
    return catalog_dir / prompt_id / f"{version}.yaml"


def load_golden_expectations(path: Path) -> tuple[PromptGoldenExpectation, ...]:
    """Load golden prompt expectations from a JSON fixture."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("expectations", [])
    return tuple(PromptGoldenExpectation.model_validate(item) for item in items)


def verify_prompt_golden_catalog(
    *,
    catalog_dir: Path,
    expectations: tuple[PromptGoldenExpectation, ...],
) -> PromptGoldenCatalogReport:
    failures: list[str] = []
    for item in expectations:
        prompt_path = prompt_version_path(
            catalog_dir,
            prompt_id=item.prompt_id,
            version=item.version,
        )
        if not prompt_path.is_file():
            failures.append(f"missing prompt file: {prompt_path.as_posix()}")
            continue
        digest = _sha256(prompt_path.read_text(encoding="utf-8"))
        if digest != item.content_sha256:
            failures.append(
                f"content hash mismatch for {item.prompt_id} v{item.version}"
            )
    return PromptGoldenCatalogReport(
        checked=len(expectations),
        passed=not failures,
        failures=failures,
    )
